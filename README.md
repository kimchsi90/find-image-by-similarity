# 주어진 이미지 중에서 서로 유사한 이미지 찾기

### e-mail: <kimchsi9004@naver.com>, mobile: 010-4058-8210

## 기술 요소

Python

- ViT(facebook/dino-vits16)
- chromadb(vector DB)
- Pillow, Pandas, Numpy

## 실행 전 설치해야 하는 라이브러리 및 참고사항

- requirements.txt 참고
- chromadb, transformers, pillow, pandas, numpy, torch2.2.0+cu118
- GPU 활용을 위해 CUDA 11.8, cuDNN 8.6.0 설치
- torch는 pip3 install torch --index-url https://download.pytorch.org/whl/cu118 명령어를 통해 설치
- 데이터셋은 imagenet-1k의 validation set(50,000장)을 "c:/val_images" 경로에 미리 저장
- 데이터셋을 다른 경로에 저장할 경우 main.py의 DATASET_DIR 변수에 데이터셋이 저장된 경로를 할당

## 프로그램 실행 흐름

1. 이미지를 임베딩하기 위한 ViT 모델 로드
2. 임베딩 벡터를 저장하기 위한 벡터 DB(chromadb) 구성
3. 미리 저장해 둔 데이터셋의 경로로부터 이미지를 읽고 500장씩 임베딩 및 DB에 저장
4. 500개씩 벡터를 DB에 쿼리하여 유사도를 측정하고 유사도가 0.9 이상인 이미지 쌍 찾기
5. 유사도를 기준으로 내림차순으로 정렬
6. 중복된 이미지 쌍 제거
7. csv 파일로 결과 저장
