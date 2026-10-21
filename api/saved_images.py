import re
import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .auth import authenticated_user, supabase_config

router = APIRouter(prefix="/saved-images", tags=["Saved Images"])

class SaveImageRequest(BaseModel):
    doc_id: str

def validate_doc_id(doc_id: str):
    if not re.fullmatch(r"P?[0-9]+", doc_id):
        raise HTTPException(422, "잘못된 이미지 ID")
    return doc_id

async def db_request(method, path, user, **kwargs):
    url, key = supabase_config()
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {user['token']}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.request(
                method,
                f"{url}/rest/v1/{path}",
                headers=headers,
                **kwargs,
            )
    except httpx.RequestError as exc:
        raise HTTPException(503, "DB 연결 실패") from exc

    if response.status_code >= 400:
        if response.status_code in (401, 403):
            raise HTTPException(403, "DB 접근 권한이 없습니다")
        if response.status_code == 409:
            raise HTTPException(409, "이미 저장된 이미지입니다")
        raise HTTPException(502, "DB 요청 처리 실패")

    return response.json() if response.content else []

@router.get("")
async def list_saved_images(user=Depends(authenticated_user)):
    return await db_request(
        "GET",
        "saved_images",
        user,
        params={
            "select": "doc_id,saved_at",
            "user_id": f"eq.{user['user_id']}",
            "order": "saved_at.desc",
        },
    )

@router.post("", status_code=201)
async def save_image(
    data: SaveImageRequest,
    user=Depends(authenticated_user),
):
    doc_id = validate_doc_id(data.doc_id)
    return await db_request(
        "POST",
        "saved_images",
        user,
        json={
            "user_id": user["user_id"],
            "doc_id": doc_id,
        },
    )

@router.delete("/{doc_id}")
async def delete_saved_image(
    doc_id: str,
    user=Depends(authenticated_user),
):
    doc_id = validate_doc_id(doc_id)
    deleted = await db_request(
        "DELETE",
        "saved_images",
        user,
        params={
            "user_id": f"eq.{user['user_id']}",
            "doc_id": f"eq.{doc_id}",
            "select": "id",
        },
    )
    return {"deleted": bool(deleted)}
