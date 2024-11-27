import os
import sqlite3
import numpy as np
import cv2
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.cluster import DBSCAN
import random
from pydantic import BaseModel
import time

class PhotoFaceData(BaseModel):
    x: int
    y: int
    w: int
    h: int
    label: str | None
    def __init__(self, x, y, w, h, l, e):
        super().__init__(x=x, y=y, w=w, h=h, label=l)
    def __init__(self, l: list):
        super().__init__(x=l[1], y=l[2], w=l[3], h=l[4], label=l[5])

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

def get_new_faces(profile_image_paths: list[str], labels_old: list[str], photos_path_new: list[str], create_labels: bool = True) -> list[list]:

    embeddings_old = []
    embeddings_new = []
    face_locations_new = []

    # 각 이미지에서 얼굴 탐지 및 임베딩 생성
    for idx, photo_path in enumerate(profile_image_paths):
        image = cv2.imread(photo_path)

        if image is None:
            raise ValueError(f"Failed to read image from path: {photo_path}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detections = detector.detect_faces(image_rgb)
        x, y, width, height = detections[0]['box']
        face = image_rgb[y:y+height, x:x+width]
        face_resized = cv2.resize(face, (160, 160))
        face_embedding = embedder.embeddings([face_resized])[0]
        embeddings_old.append(face_embedding)

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
    if len(embeddings) == 0:
        return []
    labels = dbscan.fit_predict(embeddings)

    labels_new_len = len(embeddings_new) # 추가된 얼굴 라벨 수
    labels_new = labels_old.copy()

    for i in range(len(labels_old), len(labels)):
        new = True
        for j in range(len(labels_new) if create_labels else len(labels_old)):
            if labels[i] == labels[j]:
                labels_new.append(labels_new[j])
                new = False
                break
        if new:
            if create_labels:
                labels_new.append(max_without_str(labels_new) + 1)
            else:
                labels_new.append(None)

    labels_new = labels_new[-labels_new_len:] # 추가된 얼굴 라벨만 저장
    if create_labels:
        for i in range(labels_new_len):
            if isinstance(labels_new[i], int):
                labels_new[i] = str(labels_new[i])

    data = []
    for i in range(labels_new_len):
        data.append(face_locations_new[i] + [labels_new[i]])
    return data

import openai
import io
from PIL import Image
import base64
import requests
import mimetypes

def compress_image_base64(image_path: str, max_size_kb: int = 80, initial_quality: int = 100) -> str:
    try:
        # 이미지 열기
        img = Image.open(image_path)

        # 압축 품질 조정
        quality = initial_quality
        buffer = io.BytesIO()

        # 반복적으로 압축하여 파일 크기를 확인
        while True:
            buffer.seek(0)
            buffer.truncate()
            img.save(buffer, format="JPEG", quality=quality)
            size_kb = buffer.tell() / 1024  # 현재 크기 (KB)

            if size_kb <= max_size_kb or quality <= 2:  # 크기 만족 또는 최소 품질 도달
                break

            quality //= 2  # 품질 점진적으로 감소
        print(f'Q: {quality}, S: {size_kb}')

        # Base64로 인코딩
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.read()).decode("utf-8")

        return base64_image
    except Exception as e:
        raise ValueError(f"Error compressing image: {e}")

# OpenAI API 키 설정
client = openai.OpenAI(api_key="",)

def get_category(photo_path: str) -> str:
    #t = time.time()
    base64_image = compress_image_base64(photo_path)
    #print('COMPRESSING TIME:', time.time() - t)
    img_type = 'image/jpeg'

    # ChatGPT에 이미지 분석 요청
    # Prompt 준비
    prompt = (
        "이미지를 분석하고, 이미지의 주제를 PERSON, NATURE, CITY, FOOD, OBJECT, ANIMAL, 또는 OTHERS 중 하나로 한 대문자 영단어로 분류해 주세요. "
        "인물이면 PERSON, 자연경관이면 NATURE, 도시이면 CITY, 음식이면 FOOD, 물건이면 OBJECT, 동물이면 ANIMAL, 그 외의 주제들은 OTHERS를 고르면 됩니다. "
        "너무 모호하면 (예를 들어 50%가 인물이 주제인 것 같고 50%는 자연경관이 주제인 것 같으면) 여러 개 선택하면 됩니다."
        "여러 개 선택할 때는 각각을 쉼표로 구분하고, 공백은 없어야 합니다."
        "이미지는 Base64로 인코딩된 데이터로 제공됩니다."
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
        answer = answer.strip('\'"')
        print(answer)
        return answer

    except Exception as e:
        return ValueError(f"Error during API call: {e}")

def get_categories(photos_path: list[str]) -> list[str]:
    answers = []
    for photo_path in photos_path:
        # 이미지 읽기
        #t = time.time()
        base64_image = compress_image_base64(photo_path)
        #print('COMPRESSING TIME:', time.time() - t)
        img_type = 'image/jpeg'

        # ChatGPT에 이미지 분석 요청
        # Prompt 준비
        prompt = (
            "이미지를 분석하고, 이미지의 주제를 PERSON, NATURE, CITY, FOOD, OBJECT, ANIMAL, 또는 OTHERS 중 하나로 한 대문자 영단어로 분류해 주세요. "
            "인물이면 PERSON, 자연경관이면 NATURE, 도시이면 CITY, 음식이면 FOOD, 물건이면 OBJECT, 동물이면 ANIMAL, 그 외의 주제들은 OTHERS를 고르면 됩니다. "
            "너무 모호하면 (예를 들어 50%가 인물이 주제인 것 같고 50%는 자연경관이 주제인 것 같으면) 여러 개 선택하면 됩니다."
            "여러 개 선택할 때는 각각을 쉼표로 구분하고, 공백은 없어야 합니다."
            "이미지는 Base64로 인코딩된 데이터로 제공됩니다."
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

import concurrent.futures

def get_categories_parallel(photo_paths: list[str]):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(get_category, photo_paths))
    return results

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.encoders import jsonable_encoder
from typing import List
import uvicorn
import requests
import tempfile
from urllib.parse import urlparse

def convert_to_jpg(image_path):
    """이미지를 JPG로 변환하고 변환된 파일 경로 반환"""
    with Image.open(image_path) as img:
        if img.format != "JPEG":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                rgb_img = img.convert("RGB")  # JPG는 RGB 포맷을 사용해야 함
                rgb_img.save(temp_file.name, format="JPEG")
                return temp_file.name
        return image_path

def process_path(path):
    """URL은 임시 파일로 변환하고 로컬 경로는 그대로 반환."""
    # URL인지 로컬 경로인지 확인
    if urlparse(path).scheme in ("http", "https"):
        try:
            # URL에서 이미지 다운로드
            response = requests.get(path, stream=True)
            if response.status_code == 200:
                # 임시 파일 생성
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                    temp_file.write(response.content)
                    temp_file_path = temp_file.name
                # JPG로 변환 및 기존 임시 파일 삭제
                converted_path = convert_to_jpg(temp_file_path)
                if converted_path != temp_file_path:
                    os.remove(temp_file_path)
                return converted_path
            else:
                raise ValueError(f"Failed to fetch image from URL: {path}. Status code: {response.status_code}")
        except Exception as e:
            raise ValueError(f"Error occurred while processing URL {path}: {str(e)}")
    else:
        # 로컬 경로일 경우
        if os.path.exists(path):
            return convert_to_jpg(path)
        else:
            raise ValueError(f"Invalid local path: {path}")

def process_paths(paths):
    try:
        processed_paths = []
        temp_files = []

        for path in paths:
            processed_path = process_path(path)
            processed_paths.append(processed_path)
            if path != processed_path:
                temp_files.append(processed_path)

        return processed_paths, temp_files
    except ValueError as e:
        # 임시 파일 삭제
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        raise e

app = FastAPI()

'''
image_path: 사진 한 장의 경로
'''
@app.post("/test/faces")
async def get_faces(image: UploadFile = File(...)):
    t = time.time()
    temp_file_path = None  # 임시 파일 경로 저장
    try:
        # 업로드된 이미지를 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(await image.read())

        # get_new_faces 호출
        faces_data = get_new_faces([], [], [temp_file_path], False)

        # 결과를 포맷팅하여 반환
        form = []
        for item in faces_data:
            form.append(PhotoFaceData(item))
        return form
    except Exception as e:
        print(e)
        return {"error": f'{e}'}
    finally:
        # 임시 파일 삭제
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        t = time.time() - t
        print(f'PROCESS TIME: {t}')

'''
image_path: 사진 한 장의 경로
'''
@app.get("/test/faces")
async def get_faces(image_path: str):
    t = time.time()
    try:
        processed_image_path = process_path(image_path)
        faces_data = get_new_faces([], [], [processed_image_path], False)

        form = []
        for item in faces_data:
            form.append(PhotoFaceData(item))
        return form
    except Exception as e:
        print(e)
        return {"error": f'{e}'}
    finally:
        # 임시 파일 삭제
        if not processed_image_path and processed_image_path != image_path:
            if os.path.exists(processed_image_path):
                os.remove(processed_image_path)
        t = time.time() - t
        print(f'PROCESS TIME: {t}')

'''
profile_image_paths: 프로필 사진 경로들
profile_names: 프로필 이름들
photo_paths: 새로 추가된 사진 경로들 
'''
# API Endpoints
@app.post("/process_photos/faces")
async def process_photos_faces(
        profile_image_paths: str = Form(),
        profile_names: str = Form(),
        photo_paths: str = Form()
):
    t = time.time()
    try:
        # 문자열을 리스트로 변환
        profile_image_paths = eval(profile_image_paths)
        profile_names = eval(profile_names)
        photo_paths = eval(photo_paths)

        # profile_image_paths와 photo_paths 처리
        processed_profile_image_paths, temp_files_profile = process_paths(profile_image_paths)
        processed_photo_paths, temp_files_photos = process_paths(photo_paths)

        # get_new_faces 함수 호출
        faces_data = get_new_faces(processed_profile_image_paths, profile_names, processed_photo_paths, False)

        # 결과 포맷팅
        form = []
        for i in range(len(photo_paths)):
            photo_path = photo_paths[i]
            face_list = []
            for item in faces_data:
                if i == item[0]:
                    face_list.append(PhotoFaceData(item))
            form.append(face_list)
        return form
    
    except Exception as e:
        print(e)
        return {"error": f'{e}'}
    
    finally:
        # 임시 파일 삭제
        temp_files = temp_files_profile + temp_files_photos
        for temp_file_path in temp_files:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        t = time.time() - t
        print(f'PROCESS TIME: {t}')

'''
photo_paths: 새로 추가된 사진 경로들
'''
# API Endpoints
@app.post("/process_photos/category")
async def process_photos_category(
        photo_paths: str = Form(),
):
    t = time.time()
    try:
        # 문자열로 받은 photo_paths를 리스트로 변환
        photo_paths = eval(photo_paths)
        
        # 변환된 경로를 저장할 리스트
        processed_paths, temp_files = process_paths(photo_paths)

        # get_categories 함수에 변환된 경로 전달
        result = get_categories_parallel(processed_paths)

        return result
    
    except Exception as e:
        print(e)
        return {"error": f'{e}'}
    finally:
        # 처리 중 생성된 임시 파일 삭제
        for temp_file_path in temp_files:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        t = time.time() - t
        print(f'PROCESS TIME: {t}')

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
