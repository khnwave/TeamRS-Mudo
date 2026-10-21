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

LOG_FILE = BASE_DIR / "all_data.jsonl"
LOG_FILE.touch(exist_ok=True)

save_balance = defaultdict(int)
saved_items = defaultdict(set)

MBTI_TO_USER = {
    "ENFP": "U101", "ESFP": "U101", "ENFJ": "U101", "ESFJ": "U101",
    "ENTP": "U102", "ESTP": "U102", "ENTJ": "U102", "ESTJ": "U102",
    "INFP": "U103", "ISFP": "U103", "INFJ": "U103", "ISFJ": "U103",
    "INTP": "U104", "ISTP": "U104", "INTJ": "U104", "ISTJ": "U104",
}

def _normalize_item_id(doc_id: str) -> str:
    doc_id = str(doc_id)
    return doc_id if doc_id.startswith("P") else f"P{doc_id}"

def load_save_state():
    if not LOG_FILE.exists():
        return
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("action") != "save":
                    continue
                user = entry.get("user_id")
                item = entry.get("item_id")
                score = entry.get("score", 0)
                if not user or not item:
                    continue
                key = (user, item)
                save_balance[key] += score
                if save_balance[key] > 0:
                    saved_items[user].add(item)
                elif key in save_balance and save_balance[key] <= 0 and item in saved_items.get(user, set()):
                    saved_items[user].discard(item)
    except FileNotFoundError:
        pass

def append_log(action: str, item_id: str, score: int):
    entry = {
        "user_id": current_user_id,
        "item_id": item_id,
        "action": action,
        "score": score,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

load_save_state()
print("✅ 추천 엔진 로드 완료")

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

# ===== 상태 =====
current_recommendations = []
current_alpha = None
current_user_id = "U101"
current_user_input = "U101"
selected_image_data = {"b64": None, "item_id": None, "doc_id": None, "rank": None, "path": None}

# “가짜-취소”용 최신 쿼리&UI 페이로드
latest_query = ""
last_payload = ([""]*8, [gr.update(visible=False)]*8)  # (htmls, btns)

rooms = {}
room_counter = 1
current_room_id = "room-1"

def init_rooms():
    global rooms, room_counter, current_room_id
    if not rooms:
        rooms["room-1"] = {"name": "2025 D&X:W Conference 방명록", "history": []}
        rooms["room-2"] = {"name": "LG vs. 한화 5차전 같이 볼 사람 ~", "history": []}
        room_counter = 3
        current_room_id = "room-1"
def order_rooms(): return list(rooms.keys())
init_rooms()

# ===== 유틸 =====
def get_image_path(doc_id):
    candidates = [
        IMAGES_DIR / f"{doc_id}.png",
        IMAGES_DIR / f"{doc_id}.jpg",
        IMAGES_DIR / f"{doc_id}.jpeg",
        IMAGES_DIR / f"P{doc_id}.png",
        IMAGES_DIR / f"P{doc_id}.jpg",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None

def image_to_base64_cached(path, size=(256,256)):
    if not path: return None
    key = (path, size[0])
    v = b64_cache.get(key)
    if v is not None: return v
    try:
        img = PILImage.open(path)
        img.thumbnail(size, PILImage.LANCZOS)
        buf = BytesIO(); img.save(buf, format="PNG")
        v = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
        b64_cache.put(key, v)
        return v
    except Exception:
        return None

def k_time():
    # 한국 시간(Asia/Seoul) 기준 현재 시각
    now = datetime.now(ZoneInfo(APP_TZ))
    hh = now.hour
    ampm = "오전" if hh < 12 else "오후"
    h12 = 12 if hh % 12 == 0 else hh % 12
    return f"{ampm} {h12}:{now.minute:02d}"

# ===== 저장 리스트(Radio) 유틸 =====
def saved_choices(user_id: str):
    return sorted(saved_items.get(user_id, []))

def saved_picker_update(user_id: str):
    choices = saved_choices(user_id)
    return gr.update(choices=choices, value=None)

# ===== 로그인 =====
def login_user(uid):
    global current_user_id, current_user_input
    if uid and uid.strip():
        raw = uid.strip()
        canonical = MBTI_TO_USER.get(raw.upper(), raw)
        current_user_input = raw
        current_user_id = canonical
        try:
            known = api_get(f"/users/{current_user_id}/known")["known"]
        except requests.RequestException as exc:
            print(f"[API] 사용자 확인 실패: {exc}", flush=True)
            known = False
        status = "✅ 기존 사용자" if known else "🆕 신규 사용자"
        if canonical != raw:
            message = f"로그인 완료: {raw} → {canonical} ({status})"
        else:
            message = f"로그인 완료: {current_user_id} ({status})"
        return message, saved_picker_update(current_user_id)
    return "사용자 ID를 입력하세요", saved_picker_update(current_user_id)

# ===== 추천(초저지연, 1글자부터 동작) =====
def search_and_recommend(query):
    """텍스트 입력마다 추천 리스트 업데이트"""
    global current_recommendations, current_alpha, latest_query, last_payload, selected_image_data

    latest_query = query

    if not query or not query.strip():
        current_recommendations = []
        current_alpha = None
        selected_image_data = {"b64": None, "item_id": None, "doc_id": None, "rank": None, "path": None}
        htmls = ["" for _ in range(8)]
        btns = [gr.update(visible=False) for _ in range(8)]
        last_payload = (htmls, btns)
        return "", *htmls, *btns

    my_query = query.strip()
    try:
        data = api_post("/recommend", {
            "query": my_query,
            "user_id": current_user_id,
            "top_k": 8,
        })
        results = data["recommendations"]
    except (requests.RequestException, KeyError, ValueError) as exc:
        print(f"[API] 추천 요청 실패: {exc}", flush=True)
        htmls = ["<div class='thumb'></div>" for _ in range(8)]
        btns = [gr.update(visible=False) for _ in range(8)]
        last_payload = (htmls, btns)
        return f"추천 서버 연결 오류: {exc}", *htmls, *btns

    if latest_query != query:
        htmls, btns = last_payload
        return "", *htmls, *btns

    current_recommendations = []
    htmls, btns = [], []

    if results:
        current_alpha = results[0]["alpha"]
    else:
        current_alpha = None

    for rank, res in enumerate(results, start=1):
        doc_id = str(res["doc_id"])
        item_id = _normalize_item_id(doc_id)
        path = get_image_path(doc_id)
        current_recommendations.append({
            "doc_id": doc_id,
            "item_id": item_id,
            "rank": rank,
            "img_path": path,
            "alpha": res["alpha"],
        })
        b64 = image_to_base64_cached(path, size=(256, 256)) if path else None
        if b64:
            htmls.append(f"<div class='thumb'><img src='{b64}' /></div>")
            btns.append(gr.update(visible=True))
        else:
            htmls.append("<div class='thumb'></div>")
            btns.append(gr.update(visible=False))

    while len(htmls) < 8:
        htmls.append("<div class='thumb'></div>")
        btns.append(gr.update(visible=False))

    last_payload = (htmls, btns)
    selected_image_data = {"b64": None, "item_id": None, "doc_id": None, "rank": None, "path": None}

    if current_alpha is not None:
        info = f"'{my_query}' | {current_user_id} | α={current_alpha:.2f}"
    else:
        info = f"'{my_query}' | {current_user_id}"

    return info, *htmls, *btns

# ===== 추천 썸네일 클릭 → 미리보기 =====
def select_image_for_preview(idx):
    """짤 클릭 시 미리보기 영역에 표시"""
    global selected_image_data, current_recommendations
    if idx >= len(current_recommendations):
        selected_image_data = {"b64": None, "item_id": None, "doc_id": None, "rank": None, "path": None}
        return (
            gr.update(value=""),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    
    rec = current_recommendations[idx]
    doc_id = rec["doc_id"]
    item_id = rec["item_id"]
    p = rec['img_path'] or get_image_path(doc_id)
    if p and os.path.exists(p):
        b64 = image_to_base64_cached(p, size=(400, 400))
        if b64:
            append_log("click", item_id, 1)
            selected_image_data = {
                "b64": b64,
                "item_id": item_id,
                "doc_id": doc_id,
                "rank": rec['rank'],
                "path": p,
            }
            balance = save_balance.get((current_user_id, item_id), 0)
            label = "🗑️ 저장 취소" if balance > 0 else "💾 저장"
            preview_html = f'''
            <div class="selected-preview">
                <img src="{b64}" style="max-width: 200px; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" />
            </div>
            '''
            return (
                gr.update(value=preview_html),
                gr.update(visible=True, value=label),
                gr.update(visible=True),
            )
    
    selected_image_data = {"b64": None, "item_id": None, "doc_id": None, "rank": None, "path": None}
    return (
        gr.update(value=""),
        gr.update(visible=False),
        gr.update(visible=False),
    )

# ===== 저장 리스트 선택 → 미리보기 =====
def on_saved_select(item_id):
    """저장 리스트에서 항목 클릭 시, 선택된 짤 미리보기로 올리고 버튼 활성화"""
    global selected_image_data

    if not item_id:
        selected_image_data = {"b64": None, "item_id": None, "doc_id": None, "rank": None, "path": None}
        return (
            gr.update(value=""),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    # 'P12345' -> '12345'
    doc_id = item_id[1:] if str(item_id).startswith("P") else str(item_id)
    p = get_image_path(doc_id)
    if not p or not os.path.exists(p):
        selected_image_data = {"b64": None, "item_id": None, "doc_id": None, "rank": None, "path": None}
        return (
            gr.update(value=f"<div style='color:#9ca3af'>이미지 경로 없음: {item_id}</div>"),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    b64 = image_to_base64_cached(p, size=(400, 400))
    if not b64:
        return (
            gr.update(value=f"<div style='color:#9ca3af'>이미지 로드 실패: {item_id}</div>"),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    selected_image_data = {
        "b64": b64,
        "item_id": item_id,
        "doc_id": doc_id,
        "rank": None,     # 추천 순위 없음
        "path": p,
    }
    balance = save_balance.get((current_user_id, item_id), 0)
    label = "🗑️ 저장 취소" if balance > 0 else "💾 저장"

    preview_html = f"""
    <div class="selected-preview">
        <img src="{b64}" style="max-width: 200px; height: auto; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);" />
    </div>
    """

    return (
        gr.update(value=preview_html),
        gr.update(visible=True, value=label),
        gr.update(visible=True),
    )

# ===== 전송 =====
def send_selected_image(chat_history):
    """전송 버튼 클릭 시 채팅창으로 전송"""
    global rooms, current_room_id, current_user_id, selected_image_data, current_alpha
    
    if not selected_image_data.get("b64"):
        return (
            chat_history,
            gr.update(samples=build_room_samples(current_room_id)),
            gr.update(value=""),
            gr.update(visible=False),
            gr.update(visible=False),
            saved_picker_update(current_user_id),
        )
    
    item_id = selected_image_data.get("item_id")
    if item_id:
        append_log("send", item_id, 3)

    b64 = selected_image_data["b64"]
    uname = current_user_input
    html = f"""
    <div class="chat-bubble" style="max-width:320px; background:#fff; border:1px solid #e5e7eb; border-radius:16px; padding:10px; box-shadow:0 4px 12px rgba(15,23,42,.08);">
      <div class="media" style="width:100%; max-height:420px; background:#f8fafc; border-radius:12px; overflow:hidden; display:flex; align-items:center; justify-content:center;">
        <img src="{b64}" alt="selected" style="max-width:100%; max-height:100%; width:auto; height:auto; object-fit:contain; display:block;" />
      </div>
      <div class="meta" style="display:flex; justify-content:space-between; align-items:center; font-size:14px; color:#374151; margin-top:6px;">
        <div class="uname" style="font-weight:600;">{uname}</div>
        <div class="ts" style="font-size:13px; color:#6b7280;">{k_time()}</div>
      </div>
    </div>
    """.strip()

    chat_history = chat_history or []
    chat_history.append((None, html))
    rooms[current_room_id]['history'].append((None, html))

    # Bandit 업데이트 (추천 순위가 있을 때만)
    if selected_image_data.get("rank") and current_alpha is not None:
        r = selected_image_data["rank"]
        rew = 1.0 / log2(r + 1)
        try:
            api_post("/reward", {
                "alpha": float(current_alpha),
                "reward": float(rew / (1.0 / log2(2))),
            }, timeout=10)
        except requests.RequestException as exc:
            print(f"[API] Bandit 보상 저장 실패: {exc}", flush=True)
    
    # 선택 초기화
    selected_image_data = {"b64": None, "item_id": None, "doc_id": None, "rank": None, "path": None}
    
    return (
        chat_history,
        gr.update(samples=build_room_samples(current_room_id)),
        gr.update(value=""),
        gr.update(visible=False),
        gr.update(visible=False),
        saved_picker_update(current_user_id),
    )

# ===== 저장 토글 =====
def toggle_save():
    """저장 버튼 토글"""
    global selected_image_data
    item_id = selected_image_data.get("item_id")
    if not item_id:
        return (
            gr.update(),
            gr.update(visible=False),
            saved_picker_update(current_user_id),
        )

    key = (current_user_id, item_id)
    current_score = save_balance.get(key, 0)
    if current_score > 0:
        delta = -2
        label = "💾 저장"
    else:
        delta = 2
        label = "🗑️ 저장 취소"

    save_balance[key] = current_score + delta
    if save_balance[key] > 0:
        saved_items[current_user_id].add(item_id)
    else:
        saved_items[current_user_id].discard(item_id)

    append_log("save", item_id, delta)

    return (
        gr.update(),
        gr.update(visible=True, value=label),
        saved_picker_update(current_user_id),  # ✅ 저장 리스트 갱신
    )

# ===== 채팅 목록 =====
rooms = rooms or {}
def build_room_samples(selected_id=None):
    if selected_id is None: selected_id = current_room_id
    samples = []
    for rid in order_rooms():
        name = rooms[rid]['name']; count = len(rooms[rid]['history'])
        selected = " selected" if rid == selected_id else ""
        html = f"""
        <div class='conv-item{selected}'>
          <div class='avatar'>{name[:1]}</div>
          <div class='meta'>
            <div class='title'>{name}</div>
            <div class='sub'>대화 {count}개</div>
          </div>
        </div>
        """
        samples.append([html])
    return samples

def on_room_select(evt: gr.SelectData):
    global current_room_id
    idx = evt.index; rid = order_rooms()[idx]
    current_room_id = rid
    return rooms[rid]['history'], gr.update(samples=build_room_samples(rid))

def add_room(name):
    global room_counter, current_room_id
    nm = name.strip() if name else "Conference"
    rid = f"room-{room_counter}"; room_counter += 1
    rooms[rid] = {"name": nm, "history": []}; current_room_id = rid
    return rooms[rid]['history'], gr.update(samples=build_room_samples(rid)), ""

def rename_room(name):
    if name and name.strip(): rooms[current_room_id]['name'] = name.strip()
    return gr.update(samples=build_room_samples(current_room_id))

def delete_room():
    global current_room_id
    if len(rooms) > 1 and current_room_id in rooms:
        del rooms[current_room_id]; current_room_id = order_rooms()[0]
    return rooms[current_room_id]['history'], gr.update(samples=build_room_samples(current_room_id)), ""

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

    with gr.Row(elem_classes="section-row"):
        with gr.Column(scale=2, min_width=640):
            gr.Markdown("### 👤 사용자")
            user_id_input = gr.Textbox(placeholder="사용자 ID (예: U101)", value="U101", show_label=False)
            login_btn = gr.Button("🔐 로그인", variant="primary", size="sm")
            login_status = gr.Textbox(value="로그인 완료: U101 (🆕 신규 사용자)", interactive=False, show_label=False, max_lines=1)

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
                    samples=build_room_samples(),
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
                choices=saved_choices(current_user_id),
                value=None,
                label="",
                show_label=False,   # ← 라벨 숨김
                interactive=True
            )

        with gr.Column(scale=1, min_width=360, elem_id="chat-col"):
            gr.Markdown("### 채팅")
            chatbot = gr.Chatbot(height=680, show_copy_button=False, sanitize_html=False, label=None, show_label=False, elem_classes="no-label chatbox")

    # 이벤트 바인딩
    login_btn.click(fn=login_user, inputs=[user_id_input], outputs=[login_status, saved_picker])

    room_list.select(fn=on_room_select, inputs=None, outputs=[chatbot, room_list])
    btn_add.click(fn=add_room, inputs=[new_room_name], outputs=[chatbot, room_list, new_room_name])
    btn_rename.click(fn=rename_room, inputs=[new_room_name], outputs=[room_list])
    btn_delete.click(fn=delete_room, inputs=None, outputs=[chatbot, room_list, new_room_name])

    # 추천(라이브)
    query_input.submit(
        fn=search_and_recommend,
        inputs=[query_input],
        outputs=[info_text] + image_htmls + image_buttons,
    )

    for i, btn in enumerate(image_buttons):
        btn.click(
            fn=lambda idx=i: select_image_for_preview(idx),
            inputs=None,
            outputs=[selected_preview, save_btn, send_btn]
        )

    # 저장 리스트 선택 시 → 미리보기로
    saved_picker.change(
        fn=on_saved_select,
        inputs=[saved_picker],
        outputs=[selected_preview, save_btn, send_btn]
    )
    
    save_btn.click(
        fn=toggle_save,
        inputs=None,
        outputs=[selected_preview, save_btn, saved_picker]  # ✅ 저장 리스트 갱신
    )
    
    # 전송 버튼 클릭 시 채팅창으로 전송
    send_btn.click(
        fn=send_selected_image,
        inputs=[chatbot],
        outputs=[chatbot, room_list, selected_preview, save_btn, send_btn, saved_picker]  # ✅ 저장 리스트 갱신
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

p = find_free_port()
if p:
    print(f"🚀 포트 {p}에서 실행합니다...")
    demo.launch(server_port=p, server_name="0.0.0.0", share=False)
else:
    print("❌ 사용 가능한 포트를 찾을 수 없습니다.")
