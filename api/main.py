from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
import re
import os
import boto3
from functools import lru_cache
from botocore.exceptions import ClientError, BotoCoreError

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from backend import HybridRecommender

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = BASE_DIR / "pictures"


R2_ENDPOINT = os.getenv("R2_ENDPOINT")
R2_BUCKET = os.getenv("R2_BUCKET", "muzzal-images")

r2_client = (
    boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
    )
    if R2_ENDPOINT
    else None
)


@lru_cache(maxsize=40000)
def find_r2_key(doc_id: str):
    if r2_client is None:
        return None

    for ext in (".jpg", ".png", ".jpeg", ".webp"):
        key = f"{doc_id}{ext}"
        try:
            r2_client.head_object(Bucket=R2_BUCKET, Key=key)
            return key
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code not in ("404", "NoSuchKey", "NotFound"):
                raise

    return None


class RecommendRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    user_id: str = "U101"
    top_k: int = Field(default=8, ge=1, le=20)

class RewardRequest(BaseModel):
    alpha: float = Field(ge=0, le=1)
    reward: float = Field(ge=0, le=1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = HybridRecommender(
        artifacts_dir=BASE_DIR,
        images_dir=IMAGES_DIR,
    )
    app.state.engine_lock = Lock()
    print("MUZZAL API: 추천 엔진 준비 완료", flush=True)
    yield
    app.state.engine = None

app = FastAPI(title="MUZZAL API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health(request: Request):
    engine = request.app.state.engine
    return {"status": "ok", "model_loaded": engine is not None,
            "faiss_vectors": int(engine.faiss_index.ntotal)}

@app.get("/users/{user_id}/known")
def known_user(user_id: str, request: Request):
    return {"user_id": user_id, "known": user_id in request.app.state.engine.user2idx}

@app.post("/recommend")
async def recommend(data: RecommendRequest, request: Request):
    query = data.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="검색어를 입력해주세요")

    def calculate():
        with request.app.state.engine_lock:
            return request.app.state.engine.recommend(
                query=query, user_id=data.user_id, top_k=data.top_k
            )
    try:
        results = await run_in_threadpool(calculate)
    except Exception as exc:
        print(f"[RECOMMEND ERROR] {exc!r}", flush=True)
        raise HTTPException(status_code=500, detail="추천 처리 오류") from exc

    items = []
    for rank, res in enumerate(results, 1):
        doc_id = str(res.doc_id)
        image_url = None
        if r2_client is not None or (res.image_path and Path(res.image_path).is_file()):
            image_url = str(request.url_for("get_image", doc_id=doc_id))
        items.append({
            "rank": rank,
            "doc_id": doc_id,
            "image_url": image_url,
            "final_score": float(res.final_score),
            "retrieval_score": float(res.retrieval_score),
            "personalization_score": float(res.personalization_score),
            "alpha": float(res.alpha),
        })
    return {"query": query, "user_id": data.user_id,
            "count": len(items), "recommendations": items}

@app.post("/reward")
async def reward(data: RewardRequest, request: Request):
    def update():
        with request.app.state.engine_lock:
            request.app.state.engine.reward(data.alpha, data.reward)
    await run_in_threadpool(update)
    return {"status": "ok"}


@app.get("/images/{doc_id}", name="get_image")
def get_image(doc_id: str, request: Request):
    if not re.fullmatch(r"P?[0-9]+", doc_id):
        raise HTTPException(status_code=400, detail="잘못된 이미지 ID")

    if r2_client is not None:
        try:
            key = find_r2_key(doc_id)
            if key is None:
                raise HTTPException(status_code=404, detail="R2 이미지가 없습니다")

            obj = r2_client.get_object(Bucket=R2_BUCKET, Key=key)
            try:
                content = obj["Body"].read()
            finally:
                obj["Body"].close()

            return Response(
                content=content,
                media_type=obj.get("ContentType") or "image/jpeg",
                headers={"Cache-Control": "public, max-age=3600"},
            )

        except HTTPException:
            raise
        except (ClientError, BotoCoreError) as exc:
            print(f"[R2 ERROR] {exc!r}", flush=True)
            raise HTTPException(status_code=502, detail="R2 이미지 조회 실패") from exc

    path = request.app.state.engine._find_image_path(doc_id)
    if not path or not Path(path).is_file():
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다")

    return FileResponse(path)
