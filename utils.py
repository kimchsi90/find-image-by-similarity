import os
import math
import numpy as np
import torch
import chromadb
from collections.abc import Sequence
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
from tqdm import tqdm
from chromadb.api.types import QueryResult
from typing import Union


torch.backends.cudnn.deterministic = True
torch.manual_seed(0)
N_QUERY_RESULTS = 3
SIMILARITY_THRESHOLD = 0.9
DB_DIR = os.getcwd() + "/chromadb"

client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(
    name="imagenet-1k", metadata={"hnsw:space": "cosine"}
)
device = "cuda" if torch.cuda.is_available() else "cpu"
# imagenet-1k 데이터 셋으로 사전 훈련된 ViT 모델을 로드
model_ckpt = "facebook/dino-vits16"
feature_extractor = ViTImageProcessor.from_pretrained(model_ckpt)
model = ViTModel.from_pretrained(model_ckpt).to(device)
model.eval()


# 경로로부터 이미지를 읽고 전처리하는 함수
def load_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path)
    image = image.resize((224, 224))

    # 이미지가 그레이스케일인지 확인
    if image.mode == "L":
        # NumPy 배열로 변환
        image_array = np.array(image)
        # 3차원 이미지로 변환
        image_array = np.stack((image_array,) * 3, axis=-1)
    else:
        # 이미지가 그레이스케일이 아니면, 단순히 NumPy 배열로 변환
        image_array = np.array(image)

    return image_array


# ViT 모델에서 이미지의 임베딩 벡터를 추출하는 함수
def extract_features(images: list[np.ndarray]) -> list[Sequence[float]]:
    img_tensor = feature_extractor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**img_tensor)
        embedding = outputs.pooler_output.detach().cpu().numpy().squeeze().tolist()
    # pooler_output: 최종 embedding vector

    return embedding


# 모든 이미지 벡터화
def get_embedding_vectors(img_list: list[str], BATCH_SIZE: int) -> list[Sequence[float]]:
    images = []
    metadatas = []
    ids = []
    embeddings = []
    for i, img_path in enumerate(tqdm(img_list), start=1):
        images.append(load_image(img_path))
        metadatas.append(
            {
                "file_name": os.path.basename(img_path),
            }
        )  # 파일명을 메타 데이터로 사용
        ids.append(str(i - 1))  # index 값을 id로 사용

        # 이미지 500장을 한 번에 임베딩
        if i % (BATCH_SIZE) == 0:
            embeddings.extend(extract_features(images))
            # 임베딩 벡터를 DB에 저장
            collection.add(
                embeddings=embeddings[
                    i - BATCH_SIZE : i
                ],  # 벡터 형태를 list로 저장한 형태
                metadatas=metadatas,
                ids=ids,
            )

            # 변수 초기화
            images = []
            metadatas = []
            ids = []

    return embeddings


# ViT에서 추출한 임베딩 벡터를 DB에 쿼리를 통해 유사도 0.9 이상인 이미지 찾기
def get_similarity(n_images: int, BATCH_SIZE: int, embeddings: list[Sequence[float]]) -> list[list[Union[str, int, float]]]:
    similarity_data = list()
    for i in range(math.ceil(n_images / BATCH_SIZE)):
        query_result = collection.query(
            query_embeddings=embeddings[i * BATCH_SIZE : (i + 1) * BATCH_SIZE],
            n_results=N_QUERY_RESULTS,
            include=["metadatas", "distances"],
        )

        similarity_data.extend(filter_by_similarity(query_result))

    return similarity_data


# query 결과 데이터에서 유사도가 threshold 이상인 값만 필터링하는 함수
def filter_by_similarity(query_result: QueryResult) -> list[list[Union[str, int, float]]]:
    new_data = []
    # 각 'distances'와 'metadatas'의 리스트를 순회
    for dist_list, meta_list in zip(
        query_result["distances"], query_result["metadatas"]
    ):
        # 0번째 index는 자기 자신이므로 1번째 index부터 순회
        for dist, meta in zip(dist_list[1:], meta_list[1:]):
            similarity = 1 - dist
            first_image_name = meta_list[0]["file_name"]
            second_image_name = meta["file_name"]
            # similarity가 THRESHOLD 이상인 경우
            if similarity >= SIMILARITY_THRESHOLD:
                # 유사도가 1일 때와 0.99일 때를 구분하기 위해
                # if math.floor(similarity * 100) / 100 == 0.99:
                #     similarity = 0.99
                # else:
                #     similarity = round(similarity, 2)

                # 해당 file_name과 similarity를 추가
                new_data.append(
                    [first_image_name, second_image_name, round(similarity, 2)]
                )

    return new_data
