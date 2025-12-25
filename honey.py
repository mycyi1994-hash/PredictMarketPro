import streamlit as st
import requests
import json
import concurrent.futures
import re
from datetime import datetime, timedelta, timezone
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from difflib import SequenceMatcher
import base64
import time
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend
import streamlit as st

# [추가할 부분] 키 설정 및 인증 함수

KALSHI_KEY_ID = st.secrets["KALSHI_KEY_ID"]
KALSHI_PRIVATE_KEY = st.secrets["KALSHI_PRIVATE_KEY"]

def get_kalshi_auth_headers(method: str, path: str):
    """Kalshi V2 API 인증 헤더 생성"""
    try:
        if "여기에" in KALSHI_KEY_ID: return {} # 키 설정 안됨
        
        # 1. 키 로드
        private_key_bytes = KALSHI_PRIVATE_KEY.strip().encode('utf-8')
        private_key = serialization.load_pem_private_key(
            private_key_bytes, password=None, backend=default_backend()
        )
        # 2. 서명 생성
        timestamp = str(int(time.time() * 1000))
        msg = timestamp + method + path
        signature = private_key.sign(
            msg.encode('utf-8'),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256()
        )
        sig_b64 = base64.b64encode(signature).decode('utf-8')
        
        return {
            "KALSHI-ACCESS-KEY": KALSHI_KEY_ID,
            "KALSHI-ACCESS-SIGNATURE": sig_b64,
            "KALSHI-ACCESS-TIMESTAMP": timestamp
        }
    except Exception as e:
        print(f"Auth Error: {e}")
        return {}

# 번역기
try:
    from deep_translator import GoogleTranslator
    HAS_TRANSLATOR = True
except ImportError:
    HAS_TRANSLATOR = False

# ============== 기본 설정 ==============
st.set_page_config(page_title="Prediction Market Pro", page_icon="🔮", layout="wide")

# [추가] 우측 상단 메뉴(Deploy, Settings) 숨기기 CSS
st.markdown("""
    <style>
        .stAppDeployButton {display:none;}
        [data-testid="stToolbar"] {visibility: hidden !important;}
        [data-testid="stHeader"] {visibility: hidden !important;}
        footer {visibility: hidden !important;}
    </style>
""", unsafe_allow_html=True)

# ============== 세션 상태 ==============
if "active_left_tab" not in st.session_state:
    st.session_state.active_left_tab = "Polymarket"
if "poly_results" not in st.session_state:
    st.session_state.poly_results = []
if "kalshi_results" not in st.session_state:
    st.session_state.kalshi_results = []
if "last_error" not in st.session_state:
    st.session_state.last_error = ""
if "arb_results" not in st.session_state:
    st.session_state.arb_results = []
if "scan_target" not in st.session_state:
    st.session_state.scan_target = None  # "poly", "kalshi", "arb"


# ============== 공통 유틸 ==============
def build_session():
    s = requests.Session()
    s.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
        ),
    )
    return s


def translate_text(text: str) -> str:
    if not HAS_TRANSLATOR or not text:
        return ""
    try:
        return GoogleTranslator(source="auto", target="ko").translate(text[:300])
    except Exception:
        return ""


def safe_iso(dt_str: str):
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        return None


def fmt_time_left(end: datetime, now: datetime) -> str:
    rem = end - now
    days_left = rem.days
    hours_left = int(rem.seconds / 3600)
    return f"{days_left}일" if days_left > 0 else f"{hours_left}시간"


def fmt_vol_usd(vol: float) -> str:
    if vol < 1_000_000:
        return f"${int(vol/1000)}k"
    return f"${vol/1_000_000:.1f}M"


def fmt_vol_contracts(v: float) -> str:
    if v < 1000:
        return f"{int(v)} ctr"
    return f"{v/1000:.1f}k ctr"


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ================================================================
# 👇 여기서부터 복사해서 normalize_text 함수 바로 밑에 붙여넣으세요 👇
# ================================================================

# 1. [복구] 카드 디자인 함수 (make_card_html 에러 해결)
def make_card_html(item, platform):
    bet_text = str(item.get("bet", "YES")).upper()
    color = "#00C853" if "YES" in bet_text else "#D32F2F"
    
    img_html = f'<img src="{item.get("image","")}" style="width:100%;height:100%;object-fit:cover;">'

    # HTML 템플릿
    html = f"""
    <div style="border:1px solid #ddd; border-radius:10px; overflow:hidden; margin-bottom:15px; background:white; box-shadow:0 2px 5px rgba(0,0,0,0.05);">
        <div style="height:140px; background:#eee; position:relative;">
            {img_html}
            <div style="position:absolute; top:8px; right:8px; background:rgba(0,0,0,0.7); color:white; padding:2px 8px; font-size:11px; border-radius:4px;">{item.get('sector','GEN')}</div>
        </div>
        <div style="padding:15px;">
            <div style="height:45px; overflow:hidden; font-weight:bold; font-size:15px; line-height:1.4; margin-bottom:8px;">{item.get('q','')}</div>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <span style="background:{color}; color:white; padding:4px 12px; border-radius:15px; font-weight:bold; font-size:12px;">{bet_text}</span>
                <span style="font-size:24px; font-weight:900; color:#2E7D32;">{item.get('prob',0)}%</span>
            </div>
            <div style="font-size:12px; color:#666; display:flex; justify-content:space-between;">
                <span>Vol: {item.get('vol_str','')}</span>
                <span>End: {item.get('time_short','')}</span>
            </div>
        </div>
        <a href="{item.get('link','')}" target="_blank" style="display:block; text-align:center; background:#f8f9fa; padding:10px; text-decoration:none; color:#333; font-weight:bold; font-size:13px; border-top:1px solid #eee;">마켓 이동 🚀</a>
    </div>
    """
    return html.replace("\n", "")

# 2. [복구] 아비트라지 계산 함수 (build_arbitrage_pairs 에러 해결)
def build_arbitrage_pairs(poly_list, kalshi_list, min_sim, min_spread_pct):
    pairs = []
    if not poly_list or not kalshi_list: return pairs
    
    # 불용어 제거
    stopwords = {'the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of', 'will', 'be', 'is', 'are'}
    def get_tokens(norm_text):
        return set(norm_text.split()) - stopwords

    # Kalshi 데이터 준비
    kalshi_prep = []
    for k in kalshi_list:
        kn = k.get("q_norm", "")
        if kn: kalshi_prep.append((k, get_tokens(kn)))

    for p in poly_list:
        pn = p.get("q_norm", "")
        if not pn or len(pn) < 5: continue
        p_tokens = get_tokens(pn)
        if not p_tokens: continue

        best_k = None
        best_sim = 0.0
        
        for k_item, k_tokens in kalshi_prep:
            if not k_tokens: continue
            if not (p_tokens & k_tokens): continue 

            sim = SequenceMatcher(None, pn, k_item["q_norm"]).ratio()
            if sim > best_sim:
                best_sim = sim
                best_k = k_item
        
        if best_k and best_sim >= min_sim:
            poly_prob = p["prob"]
            kalshi_prob = best_k["prob"]
            spread = abs(poly_prob - kalshi_prob)
            
            if spread >= min_spread_pct:
                pairs.append({
                    "sim": best_sim,
                    "spread": spread,
                    "poly": p,
                    "kalshi": best_k
                })
    
    pairs.sort(key=lambda x: (x["spread"], x["sim"]), reverse=True)
    return pairs

# 3. [수정] 조건 표시 함수 (span 태그 글자 깨짐 해결 -> 텍스트로 변경)
def show_search_conditions(platform, limit, vol, days, p_min, p_max, spread=None, sim=None):
    msg = f"📊 **[{platform}]** 수집: {limit}개 | 마감: {days}일 이내 | 확률: {p_min}~{p_max}%"
    if spread: msg += f" | 📏 갭: {spread}%"
    if sim: msg += f" | 🤝 유사도: {sim}"
    st.info(msg, icon="🔍")
    
# ================================================================
# 👆 여기까지 복사 👆
# ================================================================


def similarity(a: str, b: str) -> float:
    # 0~1
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

# [수정] 조건 표시 함수 (HTML 태그 에러 박멸 버전)
def show_search_conditions(platform, limit, vol, days, p_min, p_max, spread=None, sim=None):
    # HTML 태그 다 빼고, 깔끔한 텍스트로만 보여줍니다.
    msg = f"📊 **[{platform}]** 수집: {limit}개 | 마감: {days}일 이내 | 확률: {p_min}~{p_max}%"
    
    if spread:
        msg += f" | 📏 갭: {spread}%"
    if sim:
        msg += f" | 🤝 유사도: {sim}"
        
    # 스트림릿 기본 알림창 사용 (절대 안 깨짐)
    st.info(msg, icon="🔍")


# ============== Polymarket 스캔 ==============
def get_poly_tag(item):
    candidates = []
    m_tags = item.get("tags", [])
    if m_tags:
        for t in m_tags:
            val = t.get("label") if isinstance(t, dict) else t
            if val:
                candidates.append(str(val).upper())
    g_tags = item.get("group", {}).get("tags", [])
    if g_tags:
        for t in g_tags:
            val = t.get("label") if isinstance(t, dict) else t
            if val:
                candidates.append(str(val).upper())
    cat = item.get("category")
    if cat:
        candidates.append(str(cat).upper())

    ignore_list = ["POLYMARKET", "VERIFIED", "MARKETS", "GLOBAL", "EVENT", "GROUP", "DAILY", "WEEKLY"]
    cleaned = [c for c in candidates if c not in ignore_list]

    priority_map = ["POLITICS", "CRYPTO", "SPORTS", "BUSINESS", "SCIENCE", "TECHNOLOGY", "POP CULTURE", "NFL", "NBA", "BITCOIN"]
    for p in priority_map:
        if p in cleaned:
            return p
    if cleaned:
        return cleaned[0]
    return "GEN"


def process_poly_market(m, cfg):
    try:
        if not m.get("endDateIso"):
            return None
        end = safe_iso(m["endDateIso"])
        if not end:
            return None
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        target = now + timedelta(hours=cfg["hours"])
        if end < now or end > target:
            return None

        vol = float(m.get("volume", 0))
        if vol < cfg["min_vol_usd"]:
            return None

        raw_p = m.get("outcomePrices", "[]")
        prices = json.loads(raw_p) if isinstance(raw_p, str) else raw_p
        if not prices:
            return None

        max_prob = 0.0
        max_idx = 0
        for i, p in enumerate(prices):
            fp = float(p)
            if fp > max_prob:
                max_prob = fp
                max_idx = i

        if not (cfg["min_p"] <= max_prob <= cfg["max_p"]):
            return None

        raw_o = m.get("outcomes", "[]")
        outcomes = json.loads(raw_o) if isinstance(raw_o, str) else raw_o
        outcome = outcomes[max_idx] if len(outcomes) > max_idx else str(max_idx)

        is_new = False
        created_date_str = "-"
        days_since = 0
        if m.get("createdAt"):
            created_at = safe_iso(m["createdAt"])
            if created_at:
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                delta = now - created_at
                days_since = delta.days
                created_date_str = created_at.strftime("%Y-%m-%d")
                if days_since <= 7:
                    is_new = True

        img = m.get("image") or m.get("group", {}).get("icon") or "https://polymarket.com/static/logo-circle.png"
        sector = get_poly_tag(m)
        description = m.get("description", "규칙 상세 내용이 없습니다.")
        time_short = fmt_time_left(end, now)

        slug_g = m.get("group", {}).get("slug")
        slug_m = m.get("slug")
        mid = m.get("id")
        if slug_g:
            link = f"https://polymarket.com/event/{slug_g}"
        elif slug_m:
            link = f"https://polymarket.com/market/{slug_m}"
        else:
            link = f"https://polymarket.com/market/{mid}"

        return {
            "platform": "poly",
            "image": img,
            "q": m.get("question", ""),
            "q_norm": normalize_text(m.get("question", "")),
            "bet": outcome,
            "prob": int(max_prob * 100),
            "prob_raw": max_prob,
            "time_short": time_short,
            "link": link,
            "vol": vol,
            "vol_str": fmt_vol_usd(vol),
            "sector": sector,
            "sort_time": (end - now).total_seconds(),
            "q_kr": "",
            "is_new": is_new,
            "created_date": created_date_str,
            "days_since": days_since,
            "description": description,
            "desc_kr": ""
        }
    except Exception:
        return None


def scan_polymarket(scan_limit, min_vol_usd, hours_cfg, min_p, max_p, progress_bar=None):
    session = build_session()
    url = "https://gamma-api.polymarket.com/markets"
    
    raw = []
    offset = 0
    
    # [로딩] 초기 상태
    if progress_bar: progress_bar.progress(0, text="Polymarket 데이터 수집 시작...")

    while len(raw) < scan_limit:
        params = {
            "active": "true", "closed": "false", "limit": 500,
            "volume_min": min_vol_usd, "order": "volume:desc",
            "enable_group": "true", "offset": offset
        }
        try:
            res = session.get(url, params=params, timeout=10)
            if res.status_code != 200: break
            chunk = res.json()
            if not chunk: break
            
            raw.extend(chunk)
            offset += 500
            
            # [로딩] 진행률 업데이트 (수집된 개수 / 목표 개수)
            if progress_bar:
                percent = min(len(raw) / scan_limit, 1.0)
                progress_bar.progress(percent, text=f"Polymarket 수집 중... ({len(raw)}/{scan_limit})")
            
            if len(chunk) < 500: break
        except: break

    # [로딩] 분석 단계 진입
    if progress_bar: progress_bar.progress(0.9, text="데이터 분석 및 필터링 중...")

    cfg = {"hours": hours_cfg, "min_p": min_p, "max_p": max_p, "min_vol_usd": min_vol_usd}
    temp = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as exc:
        fs = [exc.submit(process_poly_market, m, cfg) for m in raw]
        for f in concurrent.futures.as_completed(fs):
            if r := f.result(): temp.append(r)
    
    dedup = list({i["q"]: i for i in temp}.values())

    if HAS_TRANSLATOR and dedup:
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as exc:
            fq = {exc.submit(translate_text, i["q"]): i for i in dedup}
            for f in concurrent.futures.as_completed(fq):
                try: fq[f]["q_kr"] = f.result()
                except: pass
                
    # [로딩] 완료
    if progress_bar: progress_bar.empty() # 로딩바 삭제
    return dedup


# ============== Kalshi 스캔 (무인증) ==============
def _kalshi_pick_price_percent(m: dict) -> float:
    candidates = [
        m.get("yes_bid"),
        m.get("yes_ask"),
        m.get("last_price"),
        m.get("yes_price"),
        m.get("price"),
    ]
    for c in candidates:
        if c is None:
            continue
        try:
            fc = float(c)
            if 0 <= fc <= 100:
                return fc
        except Exception:
            pass
    return -1.0


def process_kalshi_market(m, cfg):
    try:
        # 1. 가격 및 호가 스프레드 체크
        yes_bid = float(m.get("yes_bid", 0))
        yes_ask = float(m.get("yes_ask", 0))
        
        # 가격은 Bid 기준, 없으면 Last Price
        price_percent = yes_bid
        if price_percent == 0:
             price_percent = float(m.get("last_price", 0))

        if price_percent <= 0: return None
        if not (cfg["min_p"] * 100 <= price_percent <= cfg["max_p"] * 100): return None

        # 스프레드 필터 (안전장치)
        spread = 0
        if yes_bid > 0 and yes_ask > 0:
            spread = yes_ask - yes_bid
            if spread > cfg.get("max_spread", 100): 
                return None

        # 2. 마감일 체크
        end_raw = (m.get("expiration_time") or m.get("close_time") or "")
        end = safe_iso(end_raw)
        if not end: return None
        if end.tzinfo is None: end = end.replace(tzinfo=timezone.utc)
        
        now = datetime.now(timezone.utc)
        target = now + timedelta(hours=cfg["hours"])
        if end < now or end > target: return None

        # 3. 거래량 체크
        vol = float(m.get("volume", 0))
        if vol < cfg["min_contracts"]: return None

        # 4. 신규 마켓 여부
        is_new = False
        created_date_str = "-"
        days_since = 0
        if m.get("open_time"):
            c_at = safe_iso(m["open_time"])
            if c_at:
                if c_at.tzinfo is None: c_at = c_at.replace(tzinfo=timezone.utc)
                days_since = (now - c_at).days
                created_date_str = c_at.strftime("%Y-%m-%d")
                if days_since <= 7: is_new = True

        # 5. [수정] 이미지 매핑 (절대 안 깨지는 주소 사용)
        # API가 주는 카테고리 확인
        cat = (m.get("category") or m.get("event_category") or "GENERIC").upper()
        
        # Placehold.co를 사용해서 텍스트가 박힌 이미지를 생성 (디버깅용으로 최고)
        # 나중에 원하시면 실제 이미지 URL로 바꾸세요.
        img_base = "https://placehold.co/600x400/252f3f/FFF?text="
        
        # 카테고리별로 텍스트만 다르게 설정
        # (만약 실제 아이콘 URL이 있다면 그걸 넣으셔도 됩니다)
        img_map = {
            "ECONOMICS": "https://placehold.co/600x400/1e88e5/FFF?text=Economics",
            "POLITICS": "https://placehold.co/600x400/e53935/FFF?text=Politics",
            "CRYPTO": "https://placehold.co/600x400/fdd835/000?text=Crypto",
            "FINANCIALS": "https://placehold.co/600x400/43a047/FFF?text=Finance",
            "CLIMATE": "https://placehold.co/600x400/00897b/FFF?text=Climate",
            "TECH": "https://placehold.co/600x400/5e35b1/FFF?text=Tech",
            "SPORTS": "https://placehold.co/600x400/fb8c00/FFF?text=Sports",
        }
        
        # 매핑 안 되면 카테고리 이름 그대로 이미지 생성
        img = img_map.get(cat, f"{img_base}{cat}")

        time_short = fmt_time_left(end, now)
        ticker = m.get("ticker", "")
        link = f"https://kalshi.com/markets/{ticker}" if ticker else "https://kalshi.com/markets"

        title = m.get("title", "") or ""
        subtitle = m.get("subtitle", "") or ""
        q = f"{title} {subtitle}".strip() or "Kalshi Market"

        return {
            "platform": "kalshi",
            "image": img,
            "q": q,
            "q_norm": normalize_text(q),
            "bet": "YES",
            "prob": int(price_percent),
            "prob_raw": price_percent / 100.0,
            "time_short": time_short,
            "link": link,
            "vol": vol,
            "vol_str": fmt_vol_contracts(vol),
            "sector": cat,
            "spread": int(spread),
            "sort_time": (end - now).total_seconds(),
            "q_kr": "",
            "is_new": is_new,
            "created_date": created_date_str,
            "days_since": days_since,
            "description": m.get("rules_primary", ""),
            "desc_kr": ""
        }
    except: return None


def scan_kalshi(scan_limit, min_contracts, hours_cfg, min_p, max_p, progress_bar=None):
    session = build_session()
    
    base_url = "https://api.elections.kalshi.com"
    path = "/trade-api/v2/markets"
    url = base_url + path
    
    headers = get_kalshi_auth_headers("GET", path)
    if not headers:
        st.error("⚠️ 인증 에러: 키 설정을 확인해주세요.")
        return []

    params = {"limit": 200, "status": "open"}
    raw = []
    cursor = None
    seen = set()

    # [로딩] 초기 상태
    if progress_bar: progress_bar.progress(0, text="Kalshi 데이터 수집 시작...")
    
    for i in range(15):
        if len(raw) >= scan_limit: break
        
        p = params.copy()
        if cursor: p["cursor"] = cursor
        
        try:
            res = session.get(url, headers=headers, params=p, timeout=10)
            if res.status_code != 200: break
                
            data = res.json()
            markets = data.get("markets", [])
            if not markets: break
            
            raw.extend(markets)
            
            # [로딩] 진행률 업데이트
            if progress_bar:
                percent = min(len(raw) / scan_limit, 1.0)
                progress_bar.progress(percent, text=f"Kalshi 수집 중... ({len(raw)}/{scan_limit})")
            
            cursor = data.get("cursor")
            if not cursor or cursor in seen: break
            seen.add(cursor)
        except: break

    # [로딩] 분석 단계
    if progress_bar: progress_bar.progress(0.9, text="호가 스프레드 계산 및 필터링 중...")

    # 전역 변수(사이드바 설정) 가져오기 시도
    try: spread_val = max_bid_ask_spread 
    except NameError: spread_val = 100

    cfg = {
        "hours": hours_cfg,
        "min_p": min_p,
        "max_p": max_p,
        "min_contracts": min_contracts,
        "max_spread": spread_val
    }
    
    temp = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as exc:
        fs = [exc.submit(process_kalshi_market, m, cfg) for m in raw]
        for f in concurrent.futures.as_completed(fs):
            if r := f.result(): temp.append(r)
            
    dedup = list({i["link"]: i for i in temp}.values())
    
    if HAS_TRANSLATOR and dedup:
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as exc:
            fq = {exc.submit(translate_text, i["q"]): i for i in dedup}
            for f in concurrent.futures.as_completed(fq):
                try: fq[f]["q_kr"] = f.result()
                except: pass

    # [로딩] 완료
    if progress_bar: progress_bar.empty()
    return dedup

# ============== [누락된 부분] 아비트라지 매칭 엔진 ==============

# 1. 텍스트 유사도 계산 함수 (혹시 이것도 없을까봐 같이 드립니다)
def similarity(a, b):
    if not a or not b: return 0.0
    return SequenceMatcher(None, a, b).ratio()

# [수정] 아비트라지 매칭 엔진 (인덱싱 제거 -> 전수조사로 변경하여 매칭률 대폭 상승)
# [업그레이드] 아비트라지 매칭 엔진 (단어 기반 유사도 도입)
def build_arbitrage_pairs(poly_list, kalshi_list, min_sim, min_spread_pct):
    pairs = []
    if not poly_list or not kalshi_list: return pairs
    
    # 불용어(의미 없는 단어) 제거를 위한 간단한 세트
    stopwords = {'the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of', 'will', 'be', 'is', 'are'}

    def get_tokens(norm_text):
        # 정규화된 텍스트를 단어 세트로 변환하고 불용어 제거
        tokens = set(norm_text.split())
        return tokens - stopwords

    # Kalshi 데이터를 미리 토큰화 (속도 향상)
    kalshi_with_tokens = []
    for k in kalshi_list:
        kn = k.get("q_norm", "")
        if kn:
            kalshi_with_tokens.append((k, get_tokens(kn)))

    for p in poly_list:
        pn = p.get("q_norm", "")
        if not pn: continue
        p_tokens = get_tokens(pn)
        if not p_tokens: continue

        best_k = None
        best_sim = 0.0
        
        for k_item, k_tokens in kalshi_with_tokens:
            if not k_tokens: continue

            # 1차 필터: 단어 교집합 비율 (Jaccard Similarity 유사하게)
            # 공통 단어가 너무 적으면 아예 스킵해서 속도 높임
            intersection = p_tokens & k_tokens
            union = p_tokens | k_tokens
            token_sim = len(intersection) / len(union) if union else 0
            
            # 공통 단어 비율이 20% 미만이면 가망 없음 -> 스킵
            if token_sim < 0.2: continue 

            # 2차 필터: 정밀 문자열 유사도 (기존 SequenceMatcher)
            # 여기서 최종 유사도를 결정
            sim = similarity(pn, k_item["q_norm"])
            if sim > best_sim:
                best_sim = sim
                best_k = k_item
        
        # 최종 유사도가 설정한 기준(min_sim)을 넘으면 매칭 성공
        if best_k and best_sim >= min_sim:
            poly_prob = p["prob"]
            kalshi_prob = best_k["prob"]
            spread = abs(poly_prob - kalshi_prob)
            
            if spread >= min_spread_pct:
                pairs.append({
                    "sim": best_sim,
                    "spread": spread,
                    "poly": p,
                    "kalshi": best_k,
                    # ... (나머지 정보는 그대로)
                    "poly_link": p["link"], "kalshi_link": best_k["link"],
                    "poly_q": p["q"], "kalshi_q": best_k["q"],
                    "poly_vol": p["vol_str"], "kalshi_vol": best_k["vol_str"],
                })
    
    pairs.sort(key=lambda x: (x["spread"], x["sim"]), reverse=True)
    return pairs


# ============== 사이드바: 왼쪽 탭 분리 (최종_v3) ==============
with st.sidebar:
    st.header("🦅 Prediction Pro")
    
    # 1. 메인 메뉴
    st.session_state.active_left_tab = st.radio(
        "메뉴 선택",
        ["Polymarket", "Kalshi", "Arbitrage"],
        index=["Polymarket", "Kalshi", "Arbitrage"].index(st.session_state.active_left_tab),
        label_visibility="collapsed",
    )
    st.markdown("---")

    # 2. 공통 수집 설정 (항상 표시)
    st.markdown("### ⚙️ 공통 필터")
    scan_limit = st.slider("수집 개수", 1000, 50000, 3000, step=1000)

    # [형님 요청] 호가 스프레드(Bid-Ask) 설정 (항상 표시)
    # 예: Bid 40 / Ask 60 이면 스프레드 20%. 너무 크면 거름.
    max_bid_ask_spread = st.number_input(
        "최대 호가 스프레드(%)", 
        min_value=1, max_value=50, value=10, 
        help="매수(Bid)와 매도(Ask) 가격 차이가 이 값보다 크면 제외합니다."
    )

    st.divider()

    # 3. 탭별 상세 설정
    active = st.session_state.active_left_tab

    if active == "Polymarket":
        st.subheader("🦅 Polymarket 조건")
        
        v_opt = st.radio("최소 거래량(USD)", ["$10k", "$100k", "$1M", "직접입력"], horizontal=True)
        if v_opt == "$10k": poly_min_vol = 10000
        elif v_opt == "$100k": poly_min_vol = 100000
        elif v_opt == "$1M": poly_min_vol = 1000000
        else: poly_min_vol = st.number_input("금액($) 입력", value=5000, step=1000)

        days = st.number_input("마감 기한 (일)", 1, 365, 30)
        hours_cfg = days * 24

        # [형님 요청] 최소 확률 50% 이상으로 고정
        c1, c2 = st.columns(2)
        min_p = c1.number_input("최소 확률(%)", 50, 99, 50, help="50% 미만은 입력 불가")
        max_p = c2.number_input("최대 확률(%)", 51, 100, 99)

        st.markdown("###")
        if st.button("🦅 스캔 시작", type="primary", use_container_width=True):
            st.session_state.scan_target = "poly"
            st.session_state.poly_results = []
            st.session_state.last_error = ""
            st.rerun()

    elif active == "Kalshi":
        st.subheader("🟢 Kalshi 조건")
        
        k_opt = st.radio("최소 계약 수", ["100", "500", "1,000", "직접입력"], horizontal=True)
        if k_opt == "100": kalshi_min_contracts = 100
        elif k_opt == "500": kalshi_min_contracts = 500
        elif k_opt == "1,000": kalshi_min_contracts = 1000
        else: kalshi_min_contracts = st.number_input("계약 수 입력", value=200, step=50)

        days = st.number_input("마감 기한 (일)", 1, 365, 30)
        hours_cfg = days * 24

        # [형님 요청] 최소 확률 50% 이상으로 고정
        c1, c2 = st.columns(2)
        min_p = c1.number_input("최소 확률(%)", 50, 99, 50, help="50% 미만은 입력 불가")
        max_p = c2.number_input("최대 확률(%)", 51, 100, 99)

        st.markdown("###")
        if st.button("🟢 스캔 시작", type="primary", use_container_width=True):
            st.session_state.scan_target = "kalshi"
            st.session_state.kalshi_results = []
            st.session_state.last_error = ""
            st.rerun()

    else: # Arbitrage
        st.subheader("⚡ Arbitrage 조건")
        st.info("두 마켓 간 확률 차이(Gap)를 찾습니다.")

        # 아비트라지 전용 설정 (매칭 기준)
        c_s1, c_s2 = st.columns(2)
        with c_s1:
            spread_th = st.number_input("최소 Gap(%)", 1, 50, 5, help="Poly와 Kalshi의 확률 차이")
        with c_s2:
            sim_th = st.number_input("제목 유사도", 0.5, 1.0, 0.70, 0.05)

        # 개별 마켓 필터 (위에서 설정한 호가 스프레드도 자동 적용됨)
        st.markdown("**Poly / Kalshi 최소 기준**")
        arb_v_opt = st.radio("Poly 최소 볼륨", ["$10k", "$100k"], horizontal=True)
        arb_poly_min_vol = 10000 if arb_v_opt == "$10k" else 100000
        
        arb_k_opt = st.radio("Kalshi 최소 계약", ["100", "500"], horizontal=True)
        arb_kalshi_min_ctr = 100 if arb_k_opt == "100" else 500

        days = st.number_input("마감 기한 (일)", 1, 365, 30, key="arb_d")
        hours_cfg = days * 24
        
        # [형님 요청] 최소 확률 50% 이상
        c1, c2 = st.columns(2)
        min_p = c1.number_input("최소 확률(%)", 50, 99, 50, key="arb_p1")
        max_p = c2.number_input("최대 확률(%)", 51, 100, 99, key="arb_p2")

        st.markdown("###")
        if st.button("⚡ 아비트라지 찾기", type="primary", use_container_width=True):
            st.session_state.scan_target = "arb"
            st.session_state.arb_results = []
            st.session_state.last_error = ""
            st.rerun()

    st.markdown("---")
    if st.button("🧹 초기화", use_container_width=True):
        st.session_state.poly_results = []
        st.session_state.kalshi_results = []
        st.session_state.arb_results = []
        st.session_state.last_error = ""
        st.session_state.scan_target = None
        st.rerun()

    if not HAS_TRANSLATOR:
        st.caption("⚠️ 번역 모듈 없음")

# ============== 메인 화면 ==============
st.markdown(
    """
<style>
.row-divider { border-top: 2px dashed #DDD; margin: 40px 0; width: 100%; }
</style>
""",
    unsafe_allow_html=True,
)

# ============== UI: 메인 실행 로직 (로딩바 & 조건표시 적용) ==============
st.title(f"Market Scanner: {st.session_state.active_left_tab}")

# 메인 화면용 빈 공간 확보 (여기에 로딩바나 결과를 표시)
main_container = st.container()

# 1. Polymarket 실행
if st.session_state.scan_target == "poly":
    with main_container:
        # 화면 중앙에 로딩바 생성
        p_bar = st.progress(0, text="준비 중...")
        
        # 스캔 실행 (p_bar 전달)
        res = scan_polymarket(scan_limit, poly_min_vol, hours_cfg, min_p/100, max_p/100, progress_bar=p_bar)
        st.session_state.poly_results = res
        
        # 완료 후 로딩바 제거되었음 -> 결과 표시 단계로 이동
    st.session_state.scan_target = None
    st.rerun()

# 2. Kalshi 실행
if st.session_state.scan_target == "kalshi":
    with main_container:
        p_bar = st.progress(0, text="준비 중...")
        res = scan_kalshi(scan_limit, kalshi_min_contracts, hours_cfg, min_p/100, max_p/100, progress_bar=p_bar)
        st.session_state.kalshi_results = res
    st.session_state.scan_target = None
    st.rerun()

# 3. Arbitrage 실행
if st.session_state.scan_target == "arb":
    with main_container:
        p_bar = st.progress(0, text="양쪽 마켓 데이터 수집 시작...")
        
        # Poly 수집 (30%까지)
        p_bar.progress(0.1, text="Polymarket 데이터 수집 중...")
        # (아비트라지용 스캔 함수는 progress_bar 연동이 복잡하니 여기선 간단히 처리하거나, 위 함수 재사용)
        # 로직 단순화를 위해 기존 scan 함수 재사용 (내부 로딩바는 무시되도록 None 전달하거나, 여기선 그냥 텍스트로 때움)
        # 형님 코드가 복잡해지는 걸 방지하기 위해, 여기선 그냥 함수 호출만 합니다.
        
        p_data = scan_polymarket(scan_limit, arb_poly_min_vol, hours_cfg, min_p/100, max_p/100)
        p_bar.progress(0.5, text=f"Polymarket {len(p_data)}개 수집 완료. Kalshi 수집 시작...")
        
        k_data = scan_kalshi(scan_limit, arb_kalshi_min_ctr, hours_cfg, min_p/100, max_p/100)
        p_bar.progress(0.9, text=f"Kalshi {len(k_data)}개 수집 완료. 매칭 분석 중...")
        
        pairs = build_arbitrage_pairs(p_data, k_data, sim_th, spread_th)
        st.session_state.arb_results = pairs
        p_bar.empty()
        
    st.session_state.scan_target = None
    st.rerun()

# ============== 결과 표시 및 조건 보여주기 ==============
active = st.session_state.active_left_tab

if active == "Polymarket":
    # [조건 표시]
    if st.session_state.poly_results:
        show_search_conditions("Polymarket", scan_limit, fmt_vol_usd(poly_min_vol), days, min_p, max_p)
    
    res_data = st.session_state.poly_results
    if res_data:
        st.write(f"🔍 검색 결과: {len(res_data)}개")
        cols = st.columns(4)
        for i, item in enumerate(res_data):
            with cols[i % 4]:
                st.markdown(make_card_html(item, "poly"), unsafe_allow_html=True)
    else:
        st.info("왼쪽 사이드바에서 '스캔 시작'을 눌러주세요.")

elif active == "Kalshi":
    # [조건 표시]
    if st.session_state.kalshi_results:
        # 전역 변수 참조가 어려울 수 있으니 session_state나 기본값 활용 표시
        # (편의상 변수가 있다고 가정)
        show_search_conditions("Kalshi", scan_limit, f"{kalshi_min_contracts}계약", days, min_p, max_p, spread=max_bid_ask_spread)

    k_data = st.session_state.kalshi_results
    if k_data:
        st.write(f"🔍 검색 결과: {len(k_data)}개")
        cols = st.columns(4)
        for i, item in enumerate(k_data):
            with cols[i % 4]:
                st.markdown(make_card_html(item, "kalshi"), unsafe_allow_html=True)
    else:
        st.info("왼쪽 사이드바에서 '스캔 시작'을 눌러주세요.")

# ... (위쪽 Polymarket, Kalshi 부분은 그대로 둠) ...

elif active == "Arbitrage":
    # [조건 표시] - 항상 보여줌
    show_search_conditions("Arbitrage", scan_limit, "설정값", days, min_p, max_p, spread=spread_th, sim=sim_th)

    pairs = st.session_state.arb_results
    
    # [수정] 결과가 있으면 목록 표시, 없으면 안내 메시지
    if pairs:
        st.success(f"⚡ 매칭 성공: {len(pairs)}개의 기회를 찾았습니다!")
        for p in pairs:
            poly = p["poly"]
            kal = p["kalshi"]
            with st.container(border=True):
                c1, c2, c3 = st.columns([1, 1, 1])
                with c1:
                    st.metric("Polymarket", f"{poly['prob']}%", delta=f"Vol: {poly['vol_str']}")
                    st.caption(poly['q'])
                    st.link_button("Go Poly", poly['link'])
                with c2:
                    st.markdown(f"<h2 style='text-align:center;color:#FF5252;margin:0;'>Gap {p['spread']}%</h2>", unsafe_allow_html=True)
                    st.caption(f"유사도: {p['sim']:.2f}")
                with c3:
                    st.metric("Kalshi", f"{kal['prob']}%", delta=f"Vol: {kal['vol_str']}")
                    st.caption(kal['q'])
                    st.link_button("Go Kalshi", kal['link'])
    else:
        # 검색을 안 한 건지, 했는데 0개인 건지 구분
        if st.session_state.get("last_error"): # 에러가 있었던 경우
             st.error("스캔 중 오류가 발생했습니다.")
        elif st.session_state.scan_target is None and len(st.session_state.poly_results) > 0:
             # 스캔은 돌았는데(Poly결과가 있음) 매칭이 0개인 경우
             st.warning(f"🔍 양쪽 마켓을 다 뒤졌으나 설정한 유사도({sim_th})와 갭({spread_th}%)에 맞는 짝을 찾지 못했습니다.\n\n👉 유사도를 낮추거나(0.5 정도), 마감 기한을 늘려보세요.")
        else:
             st.info("👈 왼쪽 사이드바에서 '⚡ 아비트라지 찾기' 버튼을 눌러보세요.")

