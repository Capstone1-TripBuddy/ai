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
import traceback

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
        if not detections:
            raise ValueError(f"Failed to detect a face in profile image: index({idx})")
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
        print(f'Quality(Percentage): {quality}, Size(KB): {size_kb}')

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
        "이미지를 분석하고, 이미지의 주제를 PERSON, NATURE, CITY, FOOD, ANIMAL, 또는 OTHERS 중 하나로 한 대문자 영단어로 분류해 주세요. "
        "인물이면 PERSON, 자연 경관이면 NATURE, 도시 풍경이면 CITY, 음식이면 FOOD, 동물이면 ANIMAL, 그 외의 주제들은 OTHERS를 고르면 됩니다. "
        "너무 모호하면 (예를 들어 50%가 인물이 주제인 것 같고 50%는 자연이 주제인 것 같으면) 여러 개 선택하면 됩니다."
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
            "이미지를 분석하고, 이미지의 주제를 PERSON, NATURE, CITY, FOOD, ANIMAL, 또는 OTHERS 중 하나로 한 대문자 영단어로 분류해 주세요. "
            "인물이면 PERSON, 자연 경관이면 NATURE, 도시 풍경이면 CITY, 음식이면 FOOD, 동물이면 ANIMAL, 그 외의 주제들은 OTHERS를 고르면 됩니다. "
            "너무 모호하면 (예를 들어 50%가 인물이 주제인 것 같고 50%는 자연이 주제인 것 같으면) 여러 개 선택하면 됩니다."
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
            t = time.time()
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
            print('PROCESSING TIME:', time.time() - t)

        except Exception as e:
            return ValueError(f"Error during API call: {e}")
    return answers

import concurrent.futures

def get_categories_parallel(photo_paths: list[str]):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(get_category, photo_paths))
    return results

def get_questions(photo_path: str) -> str:
    #t = time.time()
    base64_image = compress_image_base64(photo_path)
    #print('COMPRESSING TIME:', time.time() - t)
    img_type = 'image/jpeg'

    # ChatGPT에 이미지 분석 요청
    # Prompt 준비
    prompt = (
        "주어진 이미지와 관련하여 내가 추억을 떠올릴 수 있을만한 질문을 3개 생성해주세요."
        "다른 문장도 필요 없고 오로지 질문 리스트만 나열해주세요. 항목 번호도 필요 없습니다."
        "각 질문은 개행으로만 구분해주세요."
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
        print(answer)
        answer = [x.strip() for x in answer.split('\n') if x.strip()]
        return answer

    except Exception as e:
        return ValueError(f"Error during API call: {e}")

