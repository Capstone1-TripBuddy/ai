import os
import sqlite3
import numpy as np
import cv2
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.cluster import DBSCAN
import random
from pydantic import BaseModel

class FaceData(BaseModel):
    i: int
    x: int
    y: int
    w: int
    h: int
    label: str
    embeddingId: int
    def __init__(self, i, x, y, w, h, l, e):
        super().__init__(i=i, x=x, y=y, w=w, h=h, label=l, embeddingId=e)
    def __init__(self, l: list):
        super().__init__(i=l[0], x=l[1], y=l[2], w=l[3], h=l[4], label=l[5], embeddingId=l[6])

class PhotoFaceData(BaseModel):
    x: int
    y: int
    w: int
    h: int
    label: str
    embeddingId: int
    def __init__(self, x, y, w, h, l, e):
        super().__init__(x=x, y=y, w=w, h=h, label=l, embeddingId=e)
    def __init__(self, l: list):
        super().__init__(x=l[1], y=l[2], w=l[3], h=l[4], label=l[5], embeddingId=l[6])

class PhotoData(BaseModel):
    category: str
    faces: list[PhotoFaceData]
    def __init__(self, category, faces):
        super().__init__(category=category, faces=faces)

def max_without_str(lst):
    non_str_elements = []

    for elem in lst:
        if isinstance(elem, str):
            try:
                int_value = int(elem)
                non_str_elements.append(int_value)
            except ValueError:
                continue
        else:
            non_str_elements.append(elem)

    return max(non_str_elements, default=0)

# MTCNN 얼굴 탐지기 및 FaceNet 임베딩 생성기 초기화
detector = MTCNN()
embedder = FaceNet()
dbscan = DBSCAN(metric='cosine', eps=0.2, min_samples=1)

def get_new_faces(embeddings_bytes_old: list[bytes], labels_old: list[str], photos_path_new: list[str]) -> list[list]:

    embeddings_old = []
    embeddings_new = []
    face_locations_new = []

    for embedding_bytes_old in embeddings_bytes_old:
        embeddings_old.append(np.frombuffer(embedding_bytes_old, np.float32))

    # 각 이미지에서 얼굴 탐지 및 임베딩 생성
    for idx, photo_path in enumerate(photos_path_new):
        image = cv2.imread(photo_path)
        
        if image is None:
            raise ValueError(f"Failed to read image from path: {photo_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detections = detector.detect_faces(image_rgb)

        for detection in detections:
            x, y, width, height = detection['box']
            face = image_rgb[y:y+height, x:x+width]
            face_resized = cv2.resize(face, (160, 160))
            face_embedding = embedder.embeddings([face_resized])[0]
            embeddings_new.append(face_embedding)
            face_locations_new.append([idx, x, y, width, height])

    # DBSCAN으로 얼굴 임베딩 군집화
    embeddings = embeddings_old + embeddings_new
    labels = dbscan.fit_predict(embeddings)

    labels_new_len = len(embeddings_new) # 추가된 얼굴 라벨 수
    labels_new = labels_old.copy()

    for i in range(len(labels_old), len(labels)):
        new = True
        for j in range(len(labels_new)):
            if labels[i] == labels[j]:
                labels_new.append(labels_new[j])
                new = False
                break
        if new:
            labels_new.append(max_without_str(labels_new) + 1)

    labels_new = labels_new[-labels_new_len:] # 추가된 얼굴 라벨만 저장
    for i in range(labels_new_len):
        if isinstance(labels_new[i], int):
            labels_new[i] = str(labels_new[i])

    data = []
    for i in range(labels_new_len):
        data.append(face_locations_new[i] + [labels_new[i], embeddings_new[i].tobytes()])
    return data

# embedding들을 저장하기 위해 데이터베이스를 초기화
def init_db():
    conn = sqlite3.connect("face_embeddings.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            embedding BLOB NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# 데이터베이스에 embedding을 저장하고 embedding ID를 반환
def save_embedding_to_db(embedding: bytes) -> int:
    conn = sqlite3.connect("face_embeddings.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO embeddings (embedding) VALUES (?)", (embedding,))
    conn.commit()
    embedding_id = cursor.lastrowid
    conn.close()
    return embedding_id

# 데이터베이스에서 embedding들과 label들을 받기
def get_embeddings_from_db():
    conn = sqlite3.connect("face_embeddings.db")
    cursor = conn.cursor()
    cursor.execute("SELECT id, embedding FROM embeddings")
    rows = cursor.fetchall()
    conn.close()
    embedding_ids = []
    embeddings = []
    for row in rows:
        embedding_ids.append(row[0])
        embeddings.append(np.frombuffer(row[1], np.float32))
    return embedding_ids, embeddings

import openai
import io
from PIL import Image
import base64
import requests
import mimetypes

def compress_image_base64(image_path: str, quality=50) -> str:
    try:
        # 이미지 열기
        img = Image.open(image_path)
        
        # 압축된 이미지 저장
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)  # JPEG 압축 사용
        buffer.seek(0)
        
        # Base64로 인코딩
        base64_image = base64.b64encode(buffer.read()).decode("utf-8")

        return base64_image
    except Exception as e:
        raise ValueError(f"Error compressing image: {e}")

# OpenAI API 키 설정
client = openai.OpenAI(api_key="",)

def get_categories(photos_path_new: list[str]) -> list[str]:
    answers = []
    for photo_path in photos_path_new:
        # 이미지 읽기
        base64_image = compress_image_base64(photo_path, quality=20)
        img_type = 'image/jpeg'

        # ChatGPT에 이미지 분석 요청
        # Prompt 준비
        prompt = (
            "이미지를 분석하고, 이미지의 주제를 Person, Nature, City, Food, Object, Animal, 또는 Others 중 하나로 한 영단어로 분류해 주세요. "
            "인물이면 Person, 자연경관이면 Nature, 도시이면 City, 음식이면 Food, 물건이면 Object, 동물이면 Animal, 그 외의 주제들은 Others를 고르면 됩니다. "
            "이미지는 Base64로 인코딩된 데이터로 제공됩니다: "
            f"{base64_image}... (생략)."
        )
        content = [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{img_type};base64,{base64_image}"},
            },
        ]
        
        try:
            # ChatGPT 모델에 요청
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an assistant specializing in image classification."},
                    {"role": "user", "content": content},
                ],
            )
            # 결과 가져오기
            answer = response.choices[0].message.content
            answers.append(answer.strip('\'"'))
            print(answer)

        except Exception as e:
            return ValueError(f"Error during API call: {e}")
    return answers

# 새로운 사진들을 get_new_faces와 get_categories로 처리하고 결과를 반환
def process_photos_logic(photos_path_new: list[str], embedding_ids_old: list[int], labels_old: list[str]) -> list[PhotoData]:
    embeddings_bytes_old = []
    for embedding_id in embedding_ids_old:
        conn = sqlite3.connect("face_embeddings.db")
        cursor = conn.cursor()
        cursor.execute("SELECT embedding FROM embeddings WHERE id = ?", (embedding_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            embeddings_bytes_old.append(row[0])

    faces_data = get_new_faces(embeddings_bytes_old, labels_old, photos_path_new)
    category_data = get_categories(photos_path_new)
    print(category_data)
    
    # 새 embedding들을 저장하고 embedding ID들로 업데이트함
    for item in faces_data:
        embedding = item[-1]
        label = item[-2]
        embedding_id = save_embedding_to_db(embedding)
        item[-1] = embedding_id  # Replace the embedding blob with the embedding ID

    form = []
    for i in range(len(photos_path_new)):
        photo_path = photos_path_new[i]
        face_list = []
        for item in faces_data:
            if i == item[0]:
                face_list.append(PhotoFaceData(item))
        form.append(PhotoData('Person' if face_list else category_data[i], face_list))
    return form

from fastapi import FastAPI, UploadFile, Form
from fastapi.encoders import jsonable_encoder
from typing import List
import uvicorn

init_db()
app = FastAPI()

'''
photo_paths: 새로 추가된 사진 경로들 
embedding_ids: 기존 사진들의 얼굴들의 임베딩id들
labels: 기존 사진들의 얼굴들의 이름
'''
# API Endpoints
@app.post("/process_photos/")
async def process_photos(
    photo_paths: str = Form(),
    embedding_ids: str = Form(),
    labels: str = Form()
):
    result = process_photos_logic(eval(photo_paths), eval(embedding_ids), eval(labels))

    return result

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
