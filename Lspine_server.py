## 필요한 모듈 import

import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import logging

import torch
from typing import Tuple, List, Sequence, Callable, Dict
from pathlib import Path
import sqlite3
import bcrypt
from pydantic import BaseModel

from detectron2.structures import BoxMode
from detectron2.engine import HookBase
from detectron2.utils.events import get_event_storage
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog

from fastapi import FastAPI, Request, HTTPException, File, UploadFile, Depends, Form, status
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.responses import RedirectResponse

import subprocess

app = FastAPI()
security = HTTPBasic()


# 디렉터리 및 파일 경로
UPLOAD_DIR = "./test_imgs"
OUTPUT_DIR = "./out"

logging.basicConfig(level=logging.DEBUG)

# 디렉터리 존재 여부 확인
for directory in [UPLOAD_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# FastAPI용 정적 파일 경로 마운트
app.mount("/static", StaticFiles(directory="/app/static"), name="static")
app.mount("/images", StaticFiles(directory="/app/images"), name="images")
app.mount("/test_imgs", StaticFiles(directory=UPLOAD_DIR), name="test_imgs")
app.mount("/out", StaticFiles(directory=OUTPUT_DIR), name="out")


# Jinja2 템플릿 디렉토리
templates = Jinja2Templates(directory="templates")


@app.post("/upload_and_analyze/")
async def upload_and_analyze(request: Request, image: UploadFile = File(...)):
    # 업로드한 이미지를 특정 위치에 저장
    uploaded_image_path = os.path.join(UPLOAD_DIR, image.filename)
    with open(uploaded_image_path, "wb") as buffer:
        buffer.write(await image.read())  # 비동기적으로 파일 읽기

    # 분석 스크립트를 사용하여 이미지 분석
    result = subprocess.run(["python", "Lspine_inference.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error in inference script:", result.stderr)
        return {"error": "Analysis failed"}

    # 결과 파일명 설정 (예: "out/pred_<원본 파일명>.png")
    analyzed_image_filename = f"pred_{image.filename}"
    analyzed_image_url = f"out/{analyzed_image_filename}"

    # 분석된 이미지의 URL 및 변환된 파일명을 포함한 JSON 응답 반환
    return {"imageUrl": analyzed_image_url, "transformedFileName": analyzed_image_filename}


@app.get("/delete_folders_content/{selected_image_name}")
async def delete_folders_content(selected_image_name: str):
    folders = [UPLOAD_DIR, OUTPUT_DIR]
    for folder in folders:
        folder_path = Path(folder)
        if folder_path.exists() and folder_path.is_dir():
            for file in folder_path.iterdir():
                if file.name != selected_image_name:  # 선택한 이미지를 제외하고 삭제
                    try:
                        file.unlink()
                    except Exception as e:
                        logging.error(f"Error deleting {file.name}: {str(e)}")
                        raise HTTPException(status_code=500, detail=f"Error deleting {file.name}: {str(e)}")
    return {"message": "Folders content deleted successfully"}

@app.get("/")
async def serve_index():
    # 현재 스크립트의 위치를 기준으로 상대 경로 사용
    current_directory = os.path.join(os.path.dirname(__file__))
    index_path = os.path.join(current_directory, "templates", "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)

@app.get("/result")
async def serve_index():
    # 현재 스크립트의 위치를 기준으로 상대 경로 사용
    current_directory = os.path.join(os.path.dirname(__file__))
    index_path = os.path.join(current_directory, "templates", "result.html")
    with open(index_path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

