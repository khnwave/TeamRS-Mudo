import os
import httpx
from fastapi import Header, HTTPException

def supabase_config():
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_PUBLISHABLE_KEY", "")
    if not url or not key:
        raise HTTPException(503, "Supabase 설정이 없습니다")
    return url, key

async def authenticated_user(
    authorization: str | None = Header(default=None),
):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "로그인이 필요합니다")

    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(401, "유효한 토큰이 없습니다")

    url, key = supabase_config()
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{url}/auth/v1/user",
                headers={
                    "apikey": key,
                    "Authorization": f"Bearer {token}",
                },
            )
        if response.status_code != 200:
            raise HTTPException(401, "인증되지 않은 사용자입니다")

        user_id = response.json().get("id")
        if not user_id:
            raise HTTPException(401, "사용자 정보가 없습니다")
        return {"user_id": user_id, "token": token}

    except httpx.RequestError as exc:
        raise HTTPException(503, "인증 서버 연결 실패") from exc
