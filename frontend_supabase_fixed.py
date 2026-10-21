import re
import html
from functools import partial
# MUZZAL — 초저지연 입력반응 + 이미지 LRU 캐시 + 안전 취소 + GPU 가속 (1글자부터 동작)
# -----------------------------------------------------------------------------

import os
# GPU 비활성화 옵션 제거 (사용 금지)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["TF_DISABLE_GPU"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gradio as gr
from packaging.version import Version

GRADIO6 = Version(gr.__version__) >= Version("6.0.0")
import requests
import socket
import base64
import json
from pathlib import Path
from io import BytesIO
from math import log2
from datetime import datetime
from collections import OrderedDict, defaultdict

from PIL import Image as PILImage


from zoneinfo import ZoneInfo 
APP_TZ = os.environ.get("APP_TZ", "Asia/Seoul")  # 기본은 한국 시간

# ===== 경로 및 엔진 초기화 =====
BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "pictures"

API_BASE_URL = os.environ.get("MUZZAL_API_URL", "http://127.0.0.1:8000").rstrip("/")


def api_post(path, payload, timeout=60):
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def api_get(path, timeout=10):
    response = requests.get(f"{API_BASE_URL}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()

# 로그인/보관함은 FastAPI + Supabase에 저장합니다.
# 기존 all_data.jsonl은 읽거나 덮어쓰지 않습니다.
MBTI_TO_USER = {
    "ENFP": "U101", "ESFP": "U101", "ENFJ": "U101", "ESFJ": "U101",
    "ENTP": "U102", "ESTP": "U102", "ENTJ": "U102", "ESTJ": "U102",
    "INFP": "U103", "ISFP": "U103", "INFJ": "U103", "ISFJ": "U103",
    "INTP": "U104", "ISTP": "U104", "INTJ": "U104", "ISTJ": "U104",
}

def _normalize_item_id(doc_id):
    doc_id = str(doc_id)
    return doc_id if doc_id.startswith("P") else f"P{doc_id}"

def _canonical_doc_id(doc_id):
    return str(doc_id).removeprefix("P")

# ===== 초경량 LRU(이미지 b64 캐시) =====
class LRU:
    def __init__(self, cap=1500):
        self.cap = cap
        self.d = OrderedDict()
    def get(self, k):
        if k in self.d:
            self.d.move_to_end(k)
            return self.d[k]
        return None
    def put(self, k, v):
        self.d[k] = v
        self.d.move_to_end(k)
        if len(self.d) > self.cap:
            self.d.popitem(last=False)

b64_cache = LRU(cap=1500)

# gr.State는 브라우저 세션마다 독립적으로 유지됩니다.
def initial_state():
    return {
        "token": None, "email": "", "user_id": None, "model_user": "U101",
        "saved": [], "recommendations": [], "alpha": None, "selected": None,
        "rooms": {
            "room-1": {"name": "2025 D&X:W Conference 방명록", "history": []},
            "room-2": {"name": "LG vs. 한화 5차전 같이 볼 사람 ~", "history": []},
        },
        "room_counter": 3, "room_id": "room-1",
    }

def require_login(state):
    if not state or not state.get("token"):
        raise gr.Error("먼저 Supabase 계정으로 로그인해 주세요.")
    return state

def api_auth(state, method, path, **kwargs):
    require_login(state)
    try:
        resp = requests.request(
            method, f"{API_BASE_URL}{path}",
            headers={"Authorization": f"Bearer {state['token']}"},
            timeout=20, **kwargs,
        )
        resp.raise_for_status()
        return resp.json() if resp.content else None
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 401:
            raise gr.Error("로그인 세션이 만료되었습니다. 다시 로그인해 주세요.") from exc
        raise gr.Error(f"보관함 API 오류: HTTP {exc.response.status_code if exc.response is not None else '?'}") from exc
    except requests.RequestException as exc:
        raise gr.Error("FastAPI 서버에 연결할 수 없습니다.") from exc

def fetch_saved(state):
    rows = api_auth(state, "GET", "/saved-images")
    return sorted({_normalize_item_id(r["doc_id"]) for r in rows})

def saved_picker_update(state):
    return gr.update(choices=state.get("saved", []), value=None)

def login_user(email, password, model_user, state):
    if not email or not password:
        raise gr.Error("이메일과 비밀번호를 입력해 주세요.")
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_PUBLISHABLE_KEY", "")
    if not url or not key:
        raise gr.Error("SUPABASE_URL 및 SUPABASE_PUBLISHABLE_KEY 환경변수가 필요합니다.")
    from supabase import create_client
    try:
        res = create_client(url, key).auth.sign_in_with_password({"email": email.strip(), "password": password})
    except Exception as exc:
        raise gr.Error("로그인 실패: 이메일·비밀번호 또는 이메일 인증 상태를 확인해 주세요.") from exc
    if not res.session or not res.user:
        raise gr.Error("로그인 세션을 생성하지 못했습니다.")
    new_state = initial_state()
    candidate = (model_user or "U101").strip().upper()
    new_state.update({
        "token": res.session.access_token,
        "user_id": str(res.user.id),
        "email": email.strip(),
        "model_user": MBTI_TO_USER.get(candidate, candidate) if candidate in MBTI_TO_USER or candidate in ("U101", "U102", "U103", "U104") else "U101",
    })
    new_state["saved"] = fetch_saved(new_state)
    return new_state, f"✅ 로그인 완료: {new_state['email']}", saved_picker_update(new_state), gr.update(value=""), gr.update(value=""), gr.update(visible=False), gr.update(visible=False), gr.update(value=""), gr.update(samples=build_room_samples(new_state))

def logout_user(state):
    fresh = initial_state()
    return fresh, "로그인 필요", saved_picker_update(fresh), gr.update(value=""), gr.update(value=""), gr.update(visible=False), gr.update(visible=False), gr.update(value=""), gr.update(samples=build_room_samples(fresh))

# ===== 유틸 =====
# MUZZAL_R2_FRONTEND_PATCH_V1
# Images are always requested through FastAPI; FastAPI can fetch from private R2.
def get_image_path(doc_id):
    doc_id = str(doc_id)
    if not re.fullmatch(r"P?[0-9]+", doc_id):
        return None
    return f"{API_BASE_URL}/images/{doc_id}"


def image_to_base64_cached(path, size=(256,256)):
    if not path:
        return None
    key = (path, size[0], size[1])
    cached = b64_cache.get(key)
    if cached is not None:
        return cached
    try:
        if str(path).startswith(("http://", "https://")):
            # Only fetch from our configured API, never from arbitrary input URLs.
            if not str(path).startswith(API_BASE_URL + "/images/"):
                return None
            response = requests.get(path, timeout=20)
            response.raise_for_status()
            img = PILImage.open(BytesIO(response.content))
        else:
            img = PILImage.open(path)
        with img:
            img.thumbnail(size, PILImage.LANCZOS)
            buf = BytesIO()
            img.convert("RGB").save(buf, format="PNG")
        value = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        b64_cache.put(key, value)
        return value
    except Exception as exc:
        print(f"[IMAGE API] Image load failed: {exc}", flush=True)
        return None


def k_time():
    # 한국 시간(Asia/Seoul) 기준 현재 시각
    now = datetime.now(ZoneInfo(APP_TZ))
    hh = now.hour
    ampm = "오전" if hh < 12 else "오후"
    h12 = 12 if hh % 12 == 0 else hh % 12
    return f"{ampm} {h12}:{now.minute:02d}"



# ===== 추천 + 선택 + 보관함 =====
def search_and_recommend(query, state):
    require_login(state)
    state["recommendations"] = []
    state["selected"] = None
    state["alpha"] = None
    blank = ["<div class='thumb'></div>"] * 8
    hidden = [gr.update(visible=False) for _ in range(8)]
    if not query or not query.strip():
        return state, "", *blank, *hidden, gr.update(value=""), gr.update(visible=False), gr.update(visible=False)
    try:
        data = api_post("/recommend", {
            "query": query.strip(), "user_id": state["model_user"], "top_k": 8,
        })
        results = data["recommendations"]
    except (requests.RequestException, KeyError, ValueError) as exc:
        raise gr.Error(f"추천 서버 연결 오류: {exc}") from exc
    images, buttons = [], []
    state["alpha"] = results[0].get("alpha") if results else None
    for rank, res in enumerate(results[:8], 1):
        doc_id = str(res["doc_id"])
        img_path = res.get("image_url") or get_image_path(doc_id)
        state["recommendations"].append({
            "doc_id": doc_id, "item_id": _normalize_item_id(doc_id),
            "rank": rank, "img_path": img_path,
        })
        b64 = image_to_base64_cached(img_path, size=(256,256))
        images.append(f"<div class='thumb'><img src='{b64}' /></div>" if b64 else "<div class='thumb'></div>")
        buttons.append(gr.update(visible=bool(b64)))
    while len(images) < 8:
        images.append("<div class='thumb'></div>")
        buttons.append(gr.update(visible=False))
    alpha = state["alpha"]
    info = f"'{query.strip()}' | {state['model_user']}" + (f" | α={alpha:.2f}" if alpha is not None else "")
    return state, info, *images, *buttons, gr.update(value=""), gr.update(visible=False), gr.update(visible=False)

def image_preview(selected, state):
    if not selected:
        return state, gr.update(value=""), gr.update(visible=False), gr.update(visible=False)
    path = selected.get("img_path") or get_image_path(selected["doc_id"])
    b64 = image_to_base64_cached(path, size=(400,400))
    if not b64:
        raise gr.Error("이미지 로드 실패")
    state["selected"] = dict(selected)
    label = "🗑️ 저장 취소" if selected["item_id"] in state["saved"] else "💾 저장"
    return (state, gr.update(value=f'<div class="selected-preview"><img src="{b64}" /></div>'),
            gr.update(value=label, visible=True), gr.update(visible=True))

def select_image_for_preview(idx, state):
    require_login(state)
    recs = state.get("recommendations", [])
    if not (0 <= idx < len(recs)):
        raise gr.Error("추천 결과가 갱신되었습니다. 다시 검색해 주세요.")
    return image_preview(recs[idx], state)

def select_recommendation(state, idx):
    """그리디오 콜백 인자(state)와 고정된 이미지 번호(idx)를 분리."""
    return select_image_for_preview(idx, state)

def on_saved_select(item_id, state):
    require_login(state)
    if not item_id:
        return image_preview(None, state)
    if item_id not in state["saved"]:
        raise gr.Error("이 보관함에 없는 짤입니다.")
    doc_id = _canonical_doc_id(item_id)
    return image_preview({"doc_id": doc_id, "item_id": _normalize_item_id(doc_id), "rank": None, "img_path": get_image_path(doc_id)}, state)

def toggle_save(state):
    require_login(state)
    selected = state.get("selected")
    if not selected:
        raise gr.Error("먼저 짤을 선택해 주세요.")
    doc_id = _canonical_doc_id(selected["doc_id"])
    item_id = _normalize_item_id(doc_id)
    if item_id in state["saved"]:
        api_auth(state, "DELETE", f"/saved-images/{doc_id}")
        label = "💾 저장"
    else:
        try:
            api_auth(state, "POST", "/saved-images", json={"doc_id": doc_id})
        except gr.Error as exc:
            if "HTTP 409" not in str(exc):
                raise
        label = "🗑️ 저장 취소"
    state["saved"] = fetch_saved(state)
    return state, gr.update(), gr.update(visible=True, value=label), saved_picker_update(state)

# ===== 채팅 (현재는 각 브라우저 세션의 임시 채팅) =====
def build_room_samples(state):
    chosen = state["room_id"]
    samples = []
    for rid, room in state["rooms"].items():
        selected = " selected" if rid == chosen else ""
        name = html.escape(room["name"])
        count = len(room["history"])
        samples.append([f"<div class='conv-item{selected}'><div class='avatar'>{name[:1]}</div><div class='meta'><div class='title'>{name}</div><div class='sub'>대화 {count}개</div></div></div>"])
    return samples

def send_selected_image(chat_history, state):
    require_login(state)
    selected = state.get("selected")
    if not selected:
        raise gr.Error("먼저 짤을 선택해 주세요.")
    b64 = image_to_base64_cached(selected.get("img_path") or get_image_path(selected["doc_id"]), size=(400,400))
    if not b64:
        raise gr.Error("이미지 로드 실패")
    # 화면에 렌더링되는 HTML에 이메일을 직접 넣지 않습니다.
    html = f"""<div class="chat-bubble"><div class="media"><img src="{b64}" alt="selected" /></div><div class="meta"><div class="uname">MUZZAL</div><div class="ts">{k_time()}</div></div></div>"""
    history = list(chat_history or [])
    history.append({"role": "assistant", "content": html} if GRADIO6 else (None, html))
    state["rooms"][state["room_id"]]["history"] = history
    if selected.get("rank") and state.get("alpha") is not None:
        r = selected["rank"]
        try:
            api_post("/reward", {"alpha": float(state["alpha"]), "reward": 1.0 / log2(r+1)}, timeout=10)
        except requests.RequestException as exc:
            print(f"[API] Bandit 보상 저장 실패: {exc}", flush=True)
    state["selected"] = None
    return (state, history, gr.update(samples=build_room_samples(state)), gr.update(value=""),
            gr.update(visible=False), gr.update(visible=False), saved_picker_update(state))

def on_room_select(state, evt: gr.SelectData):
    require_login(state)
    ids = list(state["rooms"])
    idx = evt.index[0] if isinstance(evt.index, (tuple,list)) else evt.index
    if not isinstance(idx,int) or idx < 0 or idx >= len(ids):
        return state, [], gr.update()
    state["room_id"] = ids[idx]
    return state, state["rooms"][state["room_id"]]["history"], gr.update(samples=build_room_samples(state))

def add_room(name, state):
    require_login(state)
    rid = f"room-{state['room_counter']}"
    state["room_counter"] += 1
    state["room_id"] = rid
    state["rooms"][rid] = {"name": (name or "").strip() or "Conference", "history": []}
    return state, [], gr.update(samples=build_room_samples(state)), ""

def rename_room(name, state):
    require_login(state)
    if name and name.strip():
        state["rooms"][state["room_id"]]["name"] = name.strip()
    return state, gr.update(samples=build_room_samples(state))

def delete_room(state):
    require_login(state)
    if len(state["rooms"]) <= 1:
        gr.Info("마지막 채팅방은 삭제할 수 없습니다. 새 채팅방을 만든 뒤 삭제해 주세요.")
    else:
        old_room = state["room_id"]
        ids = list(state["rooms"])
        # 삭제 이후 이전 방이 있으면 이전 방, 없으면 다음 방으로 이동
        index = ids.index(old_room)
        state["rooms"].pop(old_room)
        remaining = list(state["rooms"])
        state["room_id"] = remaining[min(index, len(remaining) - 1)]
    # 선택 이미지 자체는 유지해 저장/전송이 계속 가능하도록 한다.
    return state, state["rooms"][state["room_id"]]["history"], gr.update(samples=build_room_samples(state)), ""

def delete_saved_direct(item_id, state):
    """보관함에서 별도로 삭제. 클릭/미리보기 없이도 동작한다."""
    require_login(state)
    if not item_id:
        raise gr.Error("보관함에서 삭제할 짤을 먼저 선택해 주세요.")
    if item_id not in state.get("saved", []):
        raise gr.Error("현재 계정의 보관함에 없는 짤입니다.")
    doc_id = _canonical_doc_id(item_id)
    api_auth(state, "DELETE", f"/saved-images/{doc_id}")
    state["saved"] = fetch_saved(state)
    selected = state.get("selected")
    if selected and _canonical_doc_id(selected["doc_id"]) == doc_id:
        # 미리보기는 남겨두되 버튼은 저장으로 전환
        return state, saved_picker_update(state), gr.update(visible=True, value="💾 저장"), "✅ 보관함에서 삭제했습니다."
    return state, saved_picker_update(state), gr.update(), "✅ 보관함에서 삭제했습니다."

# ===== CSS =====
custom_css = r"""
.app-root { min-height: 92vh; display: flex; flex-direction: column; }
.section-row { flex: 1; }

/* 타이틀 */
.title-large h1 {
  font-size: 3.2em !important; font-weight: 900 !important; text-align: center; margin: 12px 0 6px 0;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

/* 스크롤바 숨김 */
::-webkit-scrollbar { display: none }
* { scrollbar-width: none; -ms-overflow-style: none }

/* 레이아웃 */
#chat-col { display: flex; flex-direction: column; height: calc(100vh - 180px); }
#chat-col .chatbox { flex: 1; min-height: 360px; }

/* Chatbot */
.gr-chatbot { background: #B2C7D9 !important; border-radius: 14px !important; }
.gr-chatbot .message { 
  display: flex !important; justify-content: flex-end !important;
  background: transparent !important; width: auto !important; padding: 0 !important; margin: 0 !important;
  max-width: none !important; overflow: visible !important;
}
.gr-chatbot .message .content {
  background: transparent !important; padding: 0 !important; margin: 0 !important;
  box-shadow: none !important; width: auto !important; max-width: none !important; overflow: visible !important;
}
.gr-chatbot .message .bubble { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }

/* 라벨/도움말 숨김 */
.no-label .label, .no-label label, .gr-chatbot .tag, .gr-chatbot [data-testid="block-info"] { display: none !important; }

/* ===== 우리 커스텀 말풍선 ===== */
.chat-bubble {
  max-width: 320px;
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  padding: 10px;
  box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
}
.chat-bubble .media {
  border-radius: 12px;
  background: #f8fafc;
  overflow: hidden;
  display: flex; align-items: center; justify-content: center;
}
.chat-bubble .media img {
  width: 100%; height: auto;
  max-height: 420px;
  object-fit: contain;
  display: block;
}
.chat-bubble .meta {
  display: flex; justify-content: space-between; align-items: center;
  font-size: 14px; color: #374151; margin-top: 6px;
}
.chat-bubble .meta .uname { font-weight: 600; }
.chat-bubble .meta .ts { font-size: 13px; color: #6b7280; }

/* ===== 추천 썸네일: 자르지 않음 ===== */
.img-container { position: relative !important; width: 220px !important; height: 220px !important; margin: 6px auto !important; padding: 0 !important; box-sizing: border-box; }
.img-container .thumb {
  width: 100%; height: 100%;
  border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden;
  box-sizing: border-box; background: #fff;
  display: flex; align-items: center; justify-content: center;
}
.img-container .thumb img {
  max-width: 100%; max-height: 100%;
  width: auto; height: auto; display: block;
  object-fit: contain;
}
.img-container button {
  position: absolute !important; inset: 0 !important; width: 100% !important; height: 100% !important;
  background: transparent !important; border: none !important; border-radius: 12px !important; cursor: pointer !important; z-index: 10 !important;
}
.img-container button:hover { box-shadow: inset 0 0 0 3px rgba(76, 175, 80, .35) !important; }
.img-container button:focus { outline: none !important; box-shadow: inset 0 0 0 3px rgba(99, 102, 241, .55) !important; }

/* 사이드 패널: 채팅 목록 */
.scroll-panel { height: 420px; overflow: auto; border: 1px solid #e5e7eb; border-radius: 10px; padding: 8px; background: #f9fafb; }
.conv-list .gr-dataset { background: transparent !important; border: none !important; }
.conv-item { display: flex; gap: 10px; align-items: center; padding: 10px 12px; border: 1px solid #e5e7eb; border-radius: 10px; background: #fff; }
.conv-item:hover { background: #f8fafc; }
.conv-item.selected { background: #eef2ff; border-color: #c7d2fe; }
.conv-item .avatar { width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; background: #e5e7eb; color: #374151; font-weight: 700; font-size: 13px; flex: none; }
.conv-item .meta { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
.conv-item .title { font-weight: 700; color: #111827; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.conv-item .sub { font-size: 12px; color: #6b7280; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

/* 선택 미리보기 영역 */
.selected-preview-area { min-height: 100px; padding: 12px; background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 10px; margin: 12px 0; display: flex; align-items: center; justify-content: flex-start; gap: 16px; }
.selected-preview { flex: 1; display: flex; align-items: center; justify-content: center; }
.selected-preview img { max-width: 200px; width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
.selected-preview-buttons { display: flex; flex-direction: column; gap: 8px; min-width: 120px; }
.selected-preview-buttons button { width: 100%; }

/* 타이포 & 입력들 */
body, .app-root, .gradio-container, input, textarea, button, label, .gr-markdown, .gr-button { font-size: 18.5px !important; line-height: 1.6 !important; }
button, .gr-button { font-weight: 600 !important; padding: 10px 18px !important; }
input[type="text"], textarea, .gr-textbox input, .gr-textbox textarea { font-size: 18px !important; padding: 12px 16px !important; }
.gr-markdown h3 { font-size: 20px !important; font-weight: 700 !important; margin-bottom: 10px !important; }
::placeholder { font-size: 17px !important; color: #9ca3af !important; }

/* 데이터셋(채팅 목록) 레이아웃 */
.conv-list { font-size: 15px !important; }
.conv-list .gr-dataset, .conv-list .gr-dataset .container, .conv-list .gr-dataset .grid, .conv-list .gr-dataset .gallery { display: block !important; }
.conv-list .gr-dataset .grid > div, .conv-list .gr-dataset .container > div, .conv-list .gr-dataset .gallery > div { width: 100% !important; margin: 0 0 8px 0 !important; }
.conv-list .gr-dataset { max-height: 270px !important; overflow: auto !important; }
"""

# ===== UI =====
with gr.Blocks(title="MUZZAL", theme=gr.themes.Soft(), css=custom_css, elem_classes="app-root") as demo:
    gr.Markdown("# MUZZAL", elem_classes="title-large")
    session_state = gr.State(value=initial_state)

    with gr.Row(elem_classes="section-row"):
        with gr.Column(scale=2, min_width=640):
            gr.Markdown("### 👤 사용자")
            email_input = gr.Textbox(placeholder="Supabase 계정 이메일", label="이메일")
            password_input = gr.Textbox(placeholder="비밀번호", type="password", label="비밀번호")
            model_user_input = gr.Dropdown(choices=["U101", "U102", "U103", "U104"], value="U101", label="추천 모델 프로필 (기존 학습 ID)")
            with gr.Row():
                login_btn = gr.Button("🔐 로그인", variant="primary", size="sm")
                logout_btn = gr.Button("🚪 로그아웃", size="sm")
            login_status = gr.Textbox(value="로그인 필요", interactive=False, show_label=False, max_lines=1)

            gr.Markdown("### 🖼️ 선택된 짤")
            with gr.Row(elem_classes="selected-preview-area"):
                selected_preview = gr.HTML(value="")
                with gr.Column(elem_classes="selected-preview-buttons"):
                    save_btn = gr.Button("💾 저장", variant="secondary", size="sm", visible=False)
                    send_btn = gr.Button("📤 전송", variant="primary", size="sm", visible=False)

            gr.Markdown("### 📦 추천 짤")
            info_text = gr.Textbox(interactive=False, show_label=False, visible=False)

            image_htmls, image_buttons = [], []
            for r in range(2):
                with gr.Row():
                    for c in range(4):
                        with gr.Column(scale=1, min_width=140, elem_classes="img-container"):
                            ih = gr.HTML("<div class='thumb'></div>")
                            image_htmls.append(ih)
                            btn = gr.Button("", visible=False)
                            image_buttons.append(btn)

            query_input = gr.Textbox(
                placeholder="키워드 입력하면 바로바로 바뀝니다",
                show_label=False,
                elem_id="query_input",
            )

        with gr.Column(scale=1, min_width=320):
            gr.Markdown("### 💬 채팅 목록")
            with gr.Group(elem_classes="conv-list scroll-panel"):
                room_list = gr.Dataset(
                    components=[gr.HTML()],
                    samples=build_room_samples(initial_state()),
                    headers=None,
                    label="채팅 목록",
                    samples_per_page=100,
                )
            new_room_name = gr.Textbox(placeholder="채팅방 이름 입력", show_label=False)
            with gr.Row():
                btn_add = gr.Button("➕ 생성", size="sm")
                btn_rename = gr.Button("✏️ 이름변경", size="sm")
                btn_delete = gr.Button("🗑️ 삭제", size="sm")

            gr.Markdown("### 📁 저장한 짤")
            saved_picker = gr.Radio(
                choices=[],
                value=None,
                label="",
                show_label=False,
                interactive=True,
            )
            delete_saved_btn = gr.Button("🗑️ 보관함에서 삭제", variant="secondary", size="sm")
            saved_delete_status = gr.Markdown("")

        with gr.Column(scale=1, min_width=360, elem_id="chat-col"):
            gr.Markdown("### 채팅")
            if GRADIO6:
                chatbot = gr.Chatbot(height=680, buttons=[], sanitize_html=False, label=None, show_label=False, elem_classes="no-label chatbox")
            else:
                chatbot = gr.Chatbot(height=680, show_copy_button=False, sanitize_html=False, label=None, show_label=False, elem_classes="no-label chatbox")

    # 사용자별 상태를 모든 이벤트의 입력/출력으로 전달
    login_btn.click(
        fn=login_user, inputs=[email_input, password_input, model_user_input, session_state],
        outputs=[session_state, login_status, saved_picker, chatbot, selected_preview, save_btn, send_btn, info_text, room_list],
    ).then(fn=lambda: "", inputs=None, outputs=password_input)
    logout_btn.click(
        fn=logout_user, inputs=[session_state],
        outputs=[session_state, login_status, saved_picker, chatbot, selected_preview, save_btn, send_btn, info_text, room_list],
    ).then(fn=lambda: "", inputs=None, outputs=password_input)

    room_list.select(fn=on_room_select, inputs=[session_state], outputs=[session_state, chatbot, room_list])
    btn_add.click(fn=add_room, inputs=[new_room_name, session_state], outputs=[session_state, chatbot, room_list, new_room_name])
    btn_rename.click(fn=rename_room, inputs=[new_room_name, session_state], outputs=[session_state, room_list])
    btn_delete.click(fn=delete_room, inputs=[session_state], outputs=[session_state, chatbot, room_list, new_room_name])

    query_input.submit(
        fn=search_and_recommend, inputs=[query_input, session_state],
        outputs=[session_state, info_text] + image_htmls + image_buttons + [selected_preview, save_btn, send_btn],
    )
    for i, btn in enumerate(image_buttons):
        btn.click(
            fn=partial(select_recommendation, idx=i),
            inputs=[session_state], outputs=[session_state, selected_preview, save_btn, send_btn],
        )
    # input: 사용자가 직접 라디오를 클릭했을 때만 실행.
    # change는 저장 후 choices 갱신에도 실행되어 미리보기/버튼을 지워 버릴 수 있음.
    saved_picker.input(
        fn=on_saved_select, inputs=[saved_picker, session_state],
        outputs=[session_state, selected_preview, save_btn, send_btn],
    )
    delete_saved_btn.click(
        fn=delete_saved_direct, inputs=[saved_picker, session_state],
        outputs=[session_state, saved_picker, save_btn, saved_delete_status],
    )
    save_btn.click(
        fn=toggle_save, inputs=[session_state],
        outputs=[session_state, selected_preview, save_btn, saved_picker],
    )
    send_btn.click(
        fn=send_selected_image, inputs=[chatbot, session_state],
        outputs=[session_state, chatbot, room_list, selected_preview, save_btn, send_btn, saved_picker],
    )

# ===== 실행 =====
def find_free_port(start_port=7860, max_attempts=10):
    for port in range(start_port, start_port+max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port)); return port
        except OSError:
            continue
    return None

if __name__ == "__main__":
    p = find_free_port()
    if p:
        print(f"🚀 포트 {p}에서 실행합니다...")
        demo.launch(server_port=p, server_name="127.0.0.1", share=False)
    else:
        print("❌ 사용 가능한 포트를 찾을 수 없습니다.")
