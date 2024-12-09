import photo_processing

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.encoders import jsonable_encoder
from typing import List
import uvicorn
import requests
import tempfile
from urllib.parse import urlparse

photo_processing.get_new_faces([], [], [], False)

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
        print(traceback.format_exc())
        return {"error": e}
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
        print(traceback.format_exc())
        return {"error": e}
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
        print(profile_image_paths)
        print(profile_names)
        print(photo_paths)

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
        print(traceback.format_exc())
        return {"error": e}

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
        result = get_categories(processed_paths)
        print(result)

        return result

    except Exception as e:
        print(traceback.format_exc())
        return {"error": e}
    finally:
        # 처리 중 생성된 임시 파일 삭제
        for temp_file_path in temp_files:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        t = time.time() - t
        print(f'PROCESS TIME: {t}')

@app.post("/process_photos/questions")
async def process_photos_questions(image: UploadFile = File(...)):
    t = time.time()
    temp_file_path = None  # 임시 파일 경로 저장
    try:
        # 업로드된 이미지를 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(await image.read())

        # get_questions 호출
        faces_data = get_questions(temp_file_path)

        # 결과를 포맷팅하여 반환
        form = faces_data
        return form
    except Exception as e:
        print(traceback.format_exc())
        return {"error": e}
    finally:
        # 임시 파일 삭제
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        t = time.time() - t
        print(f'PROCESS TIME: {t}')

# Run the server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
