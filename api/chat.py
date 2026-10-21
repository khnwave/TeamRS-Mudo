"""MUZZAL chat endpoints, backed by Supabase PostgREST and membership RLS.

Requires existing api.auth.authenticated_user and api.auth.supabase_config.
"""
from __future__ import annotations

import re
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field, model_validator

from .auth import authenticated_user, supabase_config

router = APIRouter(prefix="/chat-rooms", tags=["Chat"])


class CreateRoom(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class SendMessage(BaseModel):
    content: str | None = Field(default=None, max_length=4000)
    image_doc_id: str | None = None

    @model_validator(mode="after")
    def check_message(self):
        if self.image_doc_id is not None and not re.fullmatch(r"P?[0-9]+", self.image_doc_id):
            raise ValueError("image_doc_id must be numeric, optionally prefixed with P")
        if not (self.content and self.content.strip()) and not self.image_doc_id:
            raise ValueError("content or image_doc_id is required")
        return self


def room_uuid(room_id: str) -> str:
    try:
        return str(UUID(room_id))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid room UUID") from exc


async def supabase_request(method: str, endpoint: str, user: dict, *, params=None, json=None):
    base_url, anon_key = supabase_config()
    headers = {
        "apikey": anon_key,
        "Authorization": f"Bearer {user['token']}",
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=False) as client:
            response = await client.request(
                method, f"{base_url}/rest/v1/{endpoint}",
                headers=headers, params=params, json=json,
            )
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail="Supabase connection failed") from exc

    if response.status_code >= 400:
        # No database details or tokens are exposed to browser clients.
        if response.status_code in (401, 403):
            raise HTTPException(status_code=403, detail="Not authorized for this action")
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail="Resource not found")
        if response.status_code == 409:
            raise HTTPException(status_code=409, detail="Conflicting database entry")
        raise HTTPException(status_code=502, detail=f"Database request failed ({response.status_code})")
    return response.json() if response.content else None


@router.get("")
async def my_rooms(user: dict = Depends(authenticated_user)):
    """List rooms where this user is an active member (membership RLS enforced)."""
    members = await supabase_request(
        "GET", "chat_room_members", user,
        params={"select": "room_id,joined_at", "user_id": f"eq.{user['user_id']}", "left_at": "is.null"},
    )
    if not members:
        return []
    ids = ",".join(row["room_id"] for row in members)
    rooms = await supabase_request(
        "GET", "chat_rooms", user,
        params={"select": "id,name,created_by,created_at", "id": f"in.({ids})", "order": "created_at.desc"},
    )
    join_times = {row["room_id"]: row["joined_at"] for row in members}
    return [{**room, "joined_at": join_times.get(room["id"])} for room in rooms]


@router.get("/discover")
async def discover_rooms(user: dict = Depends(authenticated_user)):
    """Discover public rooms; existing chat_rooms SELECT policy permits this."""
    return await supabase_request(
        "GET", "chat_rooms", user,
        params={"select": "id,name,created_by,created_at", "order": "created_at.desc", "limit": "100"},
    )


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_room(body: CreateRoom, user: dict = Depends(authenticated_user)):
    name = body.name.strip()
    if not name:
        raise HTTPException(422, "Room name cannot be blank")
    result = await supabase_request(
        "POST", "chat_rooms", user,
        json={"name": name, "created_by": user["user_id"]},
    )
    # Database trigger adds creator membership. If the trigger fails, INSERT rolls back.
    return result[0]


@router.post("/{room_id}/join")
async def join_room(room_id: str, user: dict = Depends(authenticated_user)):
    await supabase_request(
        "POST", "rpc/join_chat_room", user, json={"p_room_id": room_uuid(room_id)},
    )
    return {"joined": True}


@router.post("/{room_id}/leave")
async def leave_room(room_id: str, user: dict = Depends(authenticated_user)):
    await supabase_request(
        "POST", "rpc/leave_chat_room", user, json={"p_room_id": room_uuid(room_id)},
    )
    return {"left": True}


@router.get("/{room_id}/messages")
async def list_messages(room_id: str, user: dict = Depends(authenticated_user)):
    # Explicit active membership check, so a nonmember gets 403 rather than silently []
    active = await supabase_request(
        "GET", "chat_room_members", user,
        params={"select": "room_id", "room_id": f"eq.{room_uuid(room_id)}", "user_id": f"eq.{user['user_id']}", "left_at": "is.null", "limit": "1"},
    )
    if not active:
        raise HTTPException(403, "Join this room to view messages")
    return await supabase_request(
        "GET", "messages", user,
        params={"select": "id,room_id,user_id,content,image_doc_id,created_at", "room_id": f"eq.{room_uuid(room_id)}", "order": "created_at.asc", "limit": "200"},
    )


@router.post("/{room_id}/messages", status_code=status.HTTP_201_CREATED)
async def send_message(room_id: str, body: SendMessage, user: dict = Depends(authenticated_user)):
    result = await supabase_request(
        "POST", "messages", user,
        json={
            "room_id": room_uuid(room_id),
            "user_id": user["user_id"],
            "content": body.content.strip() if body.content else None,
            "image_doc_id": body.image_doc_id.removeprefix("P") if body.image_doc_id else None,
        },
    )
    return result[0]
