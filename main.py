import pandas as pd
from glob import glob
from datetime import datetime
from utils import get_embedding_vectors, get_similarity


program_start_time = datetime.now()
print("프로그램 시작 시간:", program_start_time.strftime("%Y-%m-%d %H:%M:%S"))
DATASET_DIR = "c:/test_images"
BATCH_SIZE = 500
img_list = sorted(glob(DATASET_DIR + "/*.jpeg"))
n_images = len(img_list)

embeddings = get_embedding_vectors(img_list, BATCH_SIZE)
similarity_data = get_similarity(n_images, BATCH_SIZE, embeddings)

# 유사도를 기준으로 내림차순으로 데이터 정렬
sorted_data = sorted(similarity_data, key=lambda x: x[2], reverse=True)

# 데이터프레임 생성
df = pd.DataFrame(sorted_data, columns=["File1", "File2", "Similarity"])

# filename1과 filename2의 순서에 상관없이 중복 제거
df["sorted_filenames"] = df.apply(lambda x: sorted([x["File1"], x["File2"]]), axis=1)
df = df.drop_duplicates(subset="sorted_filenames").drop("sorted_filenames", axis=1)

# CSV 파일로 저장
df.to_csv("image_similarity.csv", index=False)

program_end_time = datetime.now()
print("프로그램 종료 시간:", program_end_time.strftime("%Y-%m-%d %H:%M:%S"))
print("프로그램 실행 시간:", program_end_time - program_start_time)
print(f"유사한 이미지 쌍의 개수: {len(sorted_data)}")
