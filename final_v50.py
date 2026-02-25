import streamlit as st
import pandas as pd
import json
import os
import base64
import re
import time
import html
import io
import csv
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from datetime import datetime, timedelta, timezone

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

_I18N_PATCHED = False
_ORIG_ST_FUNCS = {}
try:
    from streamlit_gsheets import GSheetsConnection
except Exception:
    GSheetsConnection = None  # streamlit-gsheets not installed or misconfigured

try:
    import bcrypt
    BCRYPT_OK = True
except Exception:
    bcrypt = None
    BCRYPT_OK = False


def _debug_enabled() -> bool:
    try:
        env_v = str(os.getenv("DMD_DEBUG", "")).strip().lower()
        if env_v in {"1", "true", "yes", "on"}:
            return True
        return bool(st.session_state.get("debug_mode", False))
    except Exception:
        return False


def _debug_log(message: str, exc: Exception | None = None) -> None:
    if not _debug_enabled():
        return
    try:
        if exc is None:
            print(f"[DMD DEBUG] {message}")
        else:
            print(f"[DMD DEBUG] {message}: {type(exc).__name__}: {exc}")
    except Exception:
        return

# --- 1. SAYFA YAPILANDIRMASI & KURUMSAL KİMLİK ---
st.set_page_config(
    page_title="NIZEN | Neurodegenerative Clinical Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://nizen.ai',
        'Report a bug': 'mailto:support@nizen.ai',
        'About': "# NIZEN\nNeurodegenerative Clinical Platform"
    }
)

# --- GLOBAL AÇIK TEMA ZORLAMA (SİYAH ARKAPLAN ENGELLEME) ---
st.markdown("""
<style>

/* ANA ARKA PLAN */
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
main {
    background-color: #ffffff !important;
    color: #1f2937 !important;
}

/* Tarayici koyu mod iptal */
:root {
    color-scheme: light !important;
}

/* SIDEBAR AÇIK */
section[data-testid="stSidebar"] {
    background-color: #f8fafc !important;
}

/* CARD / CONTAINER */
[data-testid="stMetric"],
.stExpander,
div.block-container {
    background-color: transparent !important;
}

/* TABLO */
.dataframe {
    background-color: white !important;
}

/* BUTON */
.stButton>button {
    background-color: #1c83e1 !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
}

/* INPUT */
input, textarea, select {
    background-color: white !important;
}

/* HER TÜRLÜ DARK OVERRIDE ENGEL */
[class*="dark"] {
    background-color: white !important;
    color: #1f2937 !important;
}

/* Sistem dark modda bile acik tema zorla */
@media (prefers-color-scheme: dark) {
    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    section[data-testid="stSidebar"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    .stMarkdown, .stText, .stAlert,
    .stExpander, [data-testid="stExpander"] {
        background-color: #ffffff !important;
        color: #1f2937 !important;
    }
}

</style>
""", unsafe_allow_html=True)

# --- GLOBAL STİL DÜZENLEME (UI/UX) ---
st.markdown("""
<style>
/* Fontlar */
/* --- NEW/UPDATED --- Google Fonts import fix */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #ffffff !important;
    color: #1f2937 !important;
}

h1, h2, h3 {
    font-family: 'Poppins', sans-serif !important;
    font-weight: 700 !important;
    color: #0f172a !important;
}

/* Sidebar (AÇIK TEMA) */
section[data-testid="stSidebar"] {
    background-color: #f8fafc !important;
    border-right: 1px solid rgba(148, 163, 184, 0.24) !important;
}

section[data-testid="stSidebar"] * {
    color: #0f172a !important;
}

/* Buton */
.stButton>button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    background-color: #1c83e1 !important;
    color: #ffffff !important;
    border: none !important;
}

/* Metric kart (Streamlit yeni DOM: data-testid kullanımı daha stabil) */
[data-testid="stMetric"] {
    background: #ffffff !important;
    padding: 15px !important;
    border-radius: 15px !important;
    box-shadow: 0 4px 12px rgba(148,163,184,0.20) !important;
    border: 1px solid rgba(148, 163, 184, 0.22) !important;
}

/* Expander / container'larda koyu zemin oluşmasın */
[data-testid="stExpander"], .stExpander {
    background: #ffffff !important;
    border-radius: 14px !important;
    border: 1px solid rgba(148, 163, 184, 0.22) !important;
}

/* Başlık altı çizgi */
.header-line {
    height: 5px;
    width: 80px;
    background: linear-gradient(to right, #1c83e1, #ff4b4b);
    border-radius: 10px;
    margin-bottom: 25px;
}

/* Dataframe/tablo beyaz kalsın */
[data-testid="stDataFrame"] {
    background: #ffffff !important;
}

</style>
""", unsafe_allow_html=True)

# --- 2. GOOGLE SHEETS BAGLANTISI & VERI YONETIMI ---
# --- NEW/UPDATED --- Google Sheets URL format fix
_DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/1HYeM5o-LrgDK1Fyu3yd382VQYQFcpwDa7rs7UeQRByM/edit?gid=274038883"
try:
    _secrets_sheet_url = str(st.secrets.get("SHEET_URL", "")).strip()
except Exception:
    _secrets_sheet_url = ""


def _normalize_sheet_url(raw_url: str) -> str:
    v = str(raw_url or "").strip()
    if not v:
        return ""
    # "https://docs.google.com/spreadsheets/d/<ID>/edit?gid=..." -> canonical
    m = re.search(r"/spreadsheets/d/([A-Za-z0-9-_]+)", v)
    if m:
        return f"https://docs.google.com/spreadsheets/d/{m.group(1)}"
    # plain sheet id verilirse URL'ye cevir
    if re.fullmatch(r"[A-Za-z0-9-_]{20,}", v):
        return f"https://docs.google.com/spreadsheets/d/{v}"
    return v


def _sheet_url_looks_valid(url: str) -> bool:
    u = str(url or "").strip()
    if not u:
        return False
    if not (u.startswith("https://") or u.startswith("http://")):
        return False
    return bool(re.search(r"docs\.google\.com/spreadsheets/d/[A-Za-z0-9-_]+", u))


sheet_url = _normalize_sheet_url(_secrets_sheet_url or str(os.getenv("SHEET_URL", "")).strip() or _DEFAULT_SHEET_URL)
USERS_WORKSHEET = "users"
PROFILES_WORKSHEET = "profiles"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_DB = DATA_DIR / "dmd_local.db"
LEGACY_USERS_STORE = DATA_DIR / "dmd_users.json"
LEGACY_PROFILES_STORE = DATA_DIR / "dmd_profiles.json"
SYNC_QUEUE_STORE = DATA_DIR / "sync_queue.json"
UPLOADS_DIR = DATA_DIR / "uploads"
AUTH_SECRET_STORE = DATA_DIR / "auth_secret.txt"
LOCAL_USER_ROLES_STORE = DATA_DIR / "dmd_user_roles.json"
ADMIN_OWNER_STORE = DATA_DIR / "dmd_admin_owner.json"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
VALID_USER_ROLES = {"family", "doctor", "researcher", "admin"}


def _canonical_username(username: str) -> str:
    return (username or "").strip().lower()


def _safe_link(url: str) -> str:
    url = (url or "").strip()
    if url.startswith("https://") or url.startswith("http://"):
        return url
    return "#"


def _db_connect() -> sqlite3.Connection | None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(LOCAL_DB), timeout=8)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    except Exception:
        return None


def _init_local_db() -> None:
    conn = _db_connect()
    if conn is None:
        return
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                username TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS system_kv (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
    except Exception as e:
        _debug_log("Local DB init failed", e)
    finally:
        conn.close()


def _db_get_kv(key: str) -> str:
    kk = str(key or "").strip()
    if not kk:
        return ""
    conn = _db_connect()
    if conn is None:
        return ""
    try:
        row = conn.execute("SELECT value FROM system_kv WHERE key = ?", (kk,)).fetchone()
        if not row:
            return ""
        return str(row[0] or "")
    except Exception:
        return ""
    finally:
        conn.close()


def _db_set_kv(key: str, value: str) -> bool:
    kk = str(key or "").strip()
    vv = str(value or "")
    if not kk:
        return False
    conn = _db_connect()
    if conn is None:
        return False
    try:
        conn.execute(
            """
            INSERT INTO system_kv(key, value, updated_at)
            VALUES(?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value=excluded.value,
                updated_at=CURRENT_TIMESTAMP
            """,
            (kk, vv),
        )
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def _read_json_file(path: Path, default):
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        _debug_log(f"JSON read failed: {path}", e)
    return default


def _write_json_file(path: Path, data) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        _debug_log(f"JSON write failed: {path}", e)
        return False


def _load_local_user_roles() -> dict[str, str]:
    raw = _read_json_file(LOCAL_USER_ROLES_STORE, {})
    raw_db = {}
    try:
        db_blob = _db_get_kv("user_roles_json")
        if db_blob:
            parsed = json.loads(db_blob)
            if isinstance(parsed, dict):
                raw_db = parsed
    except Exception:
        raw_db = {}
    if isinstance(raw_db, dict) and raw_db:
        raw = {**(raw if isinstance(raw, dict) else {}), **raw_db}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for u, r in raw.items():
        uu = _canonical_username(str(u))
        rr = str(r or "").strip().lower()
        if uu and rr in VALID_USER_ROLES:
            out[uu] = rr
    return out


def _save_local_user_roles(role_map: dict[str, str]) -> None:
    safe: dict[str, str] = {}
    for u, r in (role_map or {}).items():
        uu = _canonical_username(str(u))
        rr = str(r or "").strip().lower()
        if uu and rr in VALID_USER_ROLES:
            safe[uu] = rr
    _write_json_file(LOCAL_USER_ROLES_STORE, safe)
    try:
        _db_set_kv("user_roles_json", json.dumps(safe, ensure_ascii=False))
    except Exception:
        pass


def _save_local_user_role(username: str, role: str) -> None:
    uu = _canonical_username(username)
    rr = str(role or "").strip().lower()
    if not uu or rr not in VALID_USER_ROLES:
        return
    role_map = _load_local_user_roles()
    role_map[uu] = rr
    _save_local_user_roles(role_map)


def _load_admin_owner_config() -> dict:
    raw = _read_json_file(ADMIN_OWNER_STORE, {})
    try:
        db_blob = _db_get_kv("admin_owner_json")
        if db_blob:
            parsed = json.loads(db_blob)
            if isinstance(parsed, dict):
                raw = parsed
    except Exception:
        pass
    if not isinstance(raw, dict):
        return {}
    username = _canonical_username(str(raw.get("username", "")))
    password_hash = str(raw.get("password_hash", "")).strip()
    if not username or not password_hash:
        return {}
    return {"username": username, "password_hash": password_hash}


def _save_admin_owner_config(username: str, password_hash: str) -> bool:
    uu = _canonical_username(username)
    ph = str(password_hash or "").strip()
    if not uu or not ph:
        return False
    payload = {
        "username": uu,
        "password_hash": ph,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    ok_json = _write_json_file(ADMIN_OWNER_STORE, payload)
    ok_db = False
    try:
        ok_db = _db_set_kv("admin_owner_json", json.dumps(payload, ensure_ascii=False))
    except Exception:
        ok_db = False
    return bool(ok_json or ok_db)


def _admin_owner_username() -> str:
    cfg = _load_admin_owner_config()
    return _canonical_username(str(cfg.get("username", "")))


def _bootstrap_admin_owner_from_config() -> None:
    # Optional bootstrap via secrets/env; local file takes precedence once created.
    if _admin_owner_username():
        return
    try:
        sec_user = str(st.secrets.get("admin_owner_username", "")).strip()
    except Exception:
        sec_user = ""
    try:
        sec_hash = str(st.secrets.get("admin_owner_password_hash", "")).strip()
    except Exception:
        sec_hash = ""
    try:
        sec_plain = str(st.secrets.get("admin_owner_password", "")).strip()
    except Exception:
        sec_plain = ""
    env_user = str(os.getenv("DMD_ADMIN_OWNER_USERNAME", "")).strip()
    env_hash = str(os.getenv("DMD_ADMIN_OWNER_PASSWORD_HASH", "")).strip()
    env_plain = str(os.getenv("DMD_ADMIN_OWNER_PASSWORD", "")).strip()
    user = sec_user or env_user
    hash_v = sec_hash or env_hash
    plain_v = sec_plain or env_plain
    if not user:
        return
    if not hash_v and plain_v:
        hash_v = _hash_password(plain_v)
    if hash_v:
        _save_admin_owner_config(user, hash_v)


def _role_map_effective() -> dict[str, str]:
    # Local role mapping (registration-time) + config override (secrets/env).
    out = _load_local_user_roles()
    try:
        sec_roles = st.secrets.get("user_roles", {})
    except Exception:
        sec_roles = {}
    if isinstance(sec_roles, dict):
        for u, r in sec_roles.items():
            uu = _canonical_username(str(u))
            rr = str(r).strip().lower()
            if uu and rr in VALID_USER_ROLES:
                out[uu] = rr
    try:
        env_roles_raw = str(os.getenv("DMD_USER_ROLES_JSON", "")).strip()
        if env_roles_raw:
            env_roles = json.loads(env_roles_raw)
            if isinstance(env_roles, dict):
                for u, r in env_roles.items():
                    uu = _canonical_username(str(u))
                    rr = str(r).strip().lower()
                    if uu and rr in VALID_USER_ROLES:
                        out[uu] = rr
    except Exception:
        pass
    owner = _admin_owner_username()
    if owner:
        out[owner] = "admin"
        for u in list(out.keys()):
            if u != owner and out.get(u) == "admin":
                out[u] = "family"
    return out


def _allowed_roles_for_username(username: str) -> list[str]:
    uu = _canonical_username(username)
    if not uu:
        return ["family"]
    mapped = _role_map_effective().get(uu)
    if mapped in VALID_USER_ROLES:
        return [mapped]
    return ["family"]


def _load_sync_queue() -> list[dict]:
    q = _read_json_file(SYNC_QUEUE_STORE, [])
    return q if isinstance(q, list) else []


def _sync_action_identity(action: dict) -> tuple[str, str, str]:
    if not isinstance(action, dict):
        return ("", "", "")
    return (
        str(action.get("kind", "")).strip(),
        str(action.get("url", "")).strip(),
        _canonical_username(str(action.get("username", ""))),
    )


def _save_sync_queue(queue: list[dict]) -> None:
    safe = queue if isinstance(queue, list) else []
    _write_json_file(SYNC_QUEUE_STORE, safe[-500:])


def _enqueue_sync_action(action: dict) -> None:
    if not isinstance(action, dict) or not action:
        return
    q = _load_sync_queue()
    ident = _sync_action_identity(action)
    # Keep only the newest instance of the same action target.
    if any(ident):
        q = [it for it in q if _sync_action_identity(it) != ident]
    action["queued_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    q.append(action)
    _save_sync_queue(q)


def _safe_filename(name: str) -> str:
    raw = str(name or "").strip()
    raw = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    return raw[:120] or f"file_{int(time.time())}.bin"


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _parse_updated_at_safe(value) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        return datetime.min
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        return datetime.min


def _migrate_legacy_json_to_db_once() -> None:
    conn = _db_connect()
    if conn is None:
        return
    try:
        has_users = conn.execute("SELECT COUNT(1) FROM users").fetchone()
        has_profiles = conn.execute("SELECT COUNT(1) FROM profiles").fetchone()
        users_count = int(has_users[0]) if has_users else 0
        profiles_count = int(has_profiles[0]) if has_profiles else 0

        if users_count == 0 and LEGACY_USERS_STORE.exists():
            legacy_users = _read_json_file(LEGACY_USERS_STORE, {})
            if isinstance(legacy_users, dict):
                for u, p in legacy_users.items():
                    user = _canonical_username(str(u))
                    if not user:
                        continue
                    conn.execute(
                        """
                        INSERT INTO users(username, password, updated_at)
                        VALUES(?, ?, CURRENT_TIMESTAMP)
                        ON CONFLICT(username) DO UPDATE SET
                            password=excluded.password,
                            updated_at=CURRENT_TIMESTAMP
                        """,
                        (user, str(p or "")),
                    )

        if profiles_count == 0 and LEGACY_PROFILES_STORE.exists():
            legacy_profiles = _read_json_file(LEGACY_PROFILES_STORE, {})
            if isinstance(legacy_profiles, dict):
                for u, payload in legacy_profiles.items():
                    user = _canonical_username(str(u))
                    if not user:
                        continue
                    safe_payload = payload if isinstance(payload, dict) else {}
                    conn.execute(
                        """
                        INSERT INTO profiles(username, payload_json, updated_at)
                        VALUES(?, ?, CURRENT_TIMESTAMP)
                        ON CONFLICT(username) DO UPDATE SET
                            payload_json=excluded.payload_json,
                            updated_at=CURRENT_TIMESTAMP
                        """,
                        (user, json.dumps(safe_payload, ensure_ascii=False)),
                    )
        conn.commit()
    except Exception as e:
        _debug_log("Legacy JSON->DB migration failed", e)
    finally:
        conn.close()


_init_local_db()
_migrate_legacy_json_to_db_once()


def _set_cloud_status(available: bool, message: str = "", detail: str = "") -> None:
    st.session_state["cloud_sync_available"] = bool(available)
    st.session_state["_cloud_unavailable_read"] = not bool(available)
    if message:
        st.session_state["_cloud_error_msg"] = str(message)
    if detail:
        st.session_state["_cloud_error_detail"] = str(detail)
        _debug_log(detail)


def get_gsheets_conn():
    # Birden fazla yol dene: bu sayede farkli Streamlit/surum konfiglerinde cloud daha kolay aktif olur.
    if not _sheet_url_looks_valid(sheet_url):
        _set_cloud_status(
            False,
            "Cloud sync unavailable: SHEET_URL is invalid. Use a Google Sheets URL like https://docs.google.com/spreadsheets/d/<ID>.",
            f"Invalid SHEET_URL: {sheet_url}",
        )
        return None
    cached = st.session_state.get("_gsheets_conn_obj")
    if cached is not None:
        return cached
    try:
        if hasattr(st, "connection"):
            if GSheetsConnection is not None:
                try:
                    conn = st.connection("gsheets", type=GSheetsConnection)
                    st.session_state["_gsheets_conn_obj"] = conn
                    return conn
                except Exception:
                    pass
            try:
                # secrets.toml icinde connection tip tanimliysa typesiz de calisabilir.
                conn = st.connection("gsheets")
                st.session_state["_gsheets_conn_obj"] = conn
                return conn
            except Exception:
                pass
        if hasattr(st, "experimental_connection"):
            if GSheetsConnection is not None:
                try:
                    conn = st.experimental_connection("gsheets", type=GSheetsConnection)
                    st.session_state["_gsheets_conn_obj"] = conn
                    return conn
                except Exception:
                    pass
            try:
                conn = st.experimental_connection("gsheets")
                st.session_state["_gsheets_conn_obj"] = conn
                return conn
            except Exception:
                pass
    except Exception as e:
        _set_cloud_status(
            False,
            "Cloud sync unavailable: could not initialize GSheets connection. Check .streamlit/secrets.toml and service account access.",
            f"GSheets connection init failed: {type(e).__name__}: {e}",
        )
        return None
    _set_cloud_status(
        False,
        "Cloud sync unavailable: no Streamlit gsheets connection found. Install/configure streamlit-gsheets and secrets.",
        "No supported Streamlit connection API returned a gsheets connection.",
    )
    return None


def _gsheets_read_df(conn, worksheet: str | None = None, url: str | None = None):
    if conn is None:
        return None
    try:
        kwargs = {"ttl": "2m"}
        if url:
            kwargs["spreadsheet"] = url
        if worksheet:
            kwargs["worksheet"] = worksheet
        return conn.read(**kwargs)
    except TypeError:
        try:
            kwargs = {"ttl": "2m"}
            if worksheet:
                kwargs["worksheet"] = worksheet
            return conn.read(**kwargs)
        except Exception:
            _debug_log(f"GSheets read failed (worksheet={worksheet})")
            return None
    except Exception as e:
        _debug_log(f"GSheets read failed (worksheet={worksheet})", e)
        return None


def _gsheets_update_df(conn, df: pd.DataFrame, worksheet: str, url: str | None = None) -> bool:
    if conn is None:
        return False
    for kwargs in (
        {"data": df, "worksheet": worksheet, "spreadsheet": url},
        {"worksheet": worksheet, "data": df, "spreadsheet": url},
        {"data": df, "worksheet": worksheet},
        {"worksheet": worksheet, "data": df},
    ):
        clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        try:
            conn.update(**clean_kwargs)
            return True
        except Exception as e:
            _debug_log(f"GSheets update failed (worksheet={worksheet})", e)
            continue
    return False


def _check_cloud_sync_health(force: bool = False) -> bool:
    now_ts = time.time()
    if (not force) and (now_ts - float(st.session_state.get("_cloud_health_ts", 0.0)) < 60):
        return bool(st.session_state.get("cloud_sync_available", False))

    if not _sheet_url_looks_valid(sheet_url):
        _set_cloud_status(
            False,
            "Cloud sync unavailable: SHEET_URL is invalid. Expected docs.google.com/spreadsheets/d/<ID>.",
            f"Health check failed: invalid SHEET_URL {sheet_url}",
        )
        st.session_state["_cloud_health_ts"] = now_ts
        return False

    conn = get_gsheets_conn()
    if conn is None:
        st.session_state["_cloud_health_ts"] = now_ts
        return False

    users_df = _gsheets_read_df(conn, worksheet=USERS_WORKSHEET, url=sheet_url)
    profiles_df = _gsheets_read_df(conn, worksheet=PROFILES_WORKSHEET, url=sheet_url)

    if users_df is None:
        _set_cloud_status(
            False,
            "Cloud sync unavailable: worksheet 'users' is not readable. Verify tab name and sharing permissions.",
            "Health check failed: users worksheet read returned None.",
        )
        st.session_state["_cloud_health_ts"] = now_ts
        return False
    if profiles_df is None:
        _set_cloud_status(
            False,
            "Cloud sync unavailable: worksheet 'profiles' is not readable. Verify tab name and sharing permissions.",
            "Health check failed: profiles worksheet read returned None.",
        )
        st.session_state["_cloud_health_ts"] = now_ts
        return False

    if not users_df.empty:
        cols = {str(c).strip().lower() for c in users_df.columns}
        if not {"username", "password"}.issubset(cols):
            _set_cloud_status(
                False,
                "Cloud sync unavailable: worksheet 'users' must contain columns username,password.",
                f"Health check failed: users columns={sorted(cols)}",
            )
            st.session_state["_cloud_health_ts"] = now_ts
            return False

    if not profiles_df.empty:
        cols = {str(c).strip().lower() for c in profiles_df.columns}
        required = {"username", "kilo", "yas", "nsaa_total", "nsaa_prev_total", "nsaa_history"}
        if not required.issubset(cols):
            _set_cloud_status(
                False,
                "Cloud sync unavailable: worksheet 'profiles' has missing required columns.",
                f"Health check failed: profiles columns={sorted(cols)}",
            )
            st.session_state["_cloud_health_ts"] = now_ts
            return False

    _set_cloud_status(True)
    st.session_state["_cloud_health_ts"] = now_ts
    return True


@st.cache_data(ttl=600)
def fetch_dmd_news(lang: str = "TR", limit: int = 20) -> list[dict]:
    query = "duchenne muscular dystrophy" if lang == "EN" else "duchenne muskuler distrofi"
    rss_url = (
        "https://news.google.com/rss/search?q="
        + quote_plus(query)
        + ("&hl=en-US&gl=US&ceid=US:en" if lang == "EN" else "&hl=tr&gl=TR&ceid=TR:tr")
    )
    req = Request(rss_url, headers={"User-Agent": "Mozilla/5.0"})
    items: list[dict] = []
    try:
        with urlopen(req, timeout=12) as resp:
            xml_data = resp.read()
        root = ET.fromstring(xml_data)
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            source_el = item.find("source")
            source = (source_el.text or "").strip() if source_el is not None else "Google News"
            pub_raw = (item.findtext("pubDate") or "").strip()
            published = ""
            if pub_raw:
                try:
                    dt = parsedate_to_datetime(pub_raw)
                    published = dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    published = pub_raw
            if title and link:
                items.append(
                    {
                        "title": title,
                        "link": link,
                        "source": source,
                        "published": published,
                    }
                )
            if len(items) >= limit:
                break
    except Exception:
        return []

    # basit title+link dedupe
    seen = set()
    deduped = []
    for it in items:
        key = (it["title"], it["link"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(it)
    return deduped


def _get_openai_api_key() -> str:
    runtime_key = str(st.session_state.get("openai_api_key", "")).strip()
    if runtime_key:
        return runtime_key
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        key = ""
    if not key:
        key = os.getenv("OPENAI_API_KEY", "")
    return str(key).strip()


def ask_openai_medical_assistant(question: str, context_text: str = "") -> str:
    api_key = _get_openai_api_key()
    if not api_key:
        return "AI hizmeti için OPENAI_API_KEY tanımlı değil."

    q = (question or "").strip()
    if not q:
        return "Lütfen bir soru yazın."

    system_prompt = (
        "Sen DMD konusunda yardımcı bir dijital asistansın. "
        "Tıbbi tanı koymazsın, kesin tedavi önermezsin. "
        "Yanıtların kısa, anlaşılır ve güvenli olsun. "
        "Acil riskte kullanıcıyı sağlık profesyoneline yönlendir."
    )
    if context_text:
        user_text = f"Soru: {q}\n\nKullanıcı bağlamı:\n{context_text}"
    else:
        user_text = f"Soru: {q}"

    payload = {
        "model": "gpt-4.1-mini",
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_text}]},
        ],
        "temperature": 0.2,
        "max_output_tokens": 500,
    }
    req = Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = str(data.get("output_text", "")).strip()
        if text:
            return text
        # Fallback parser
        out = data.get("output", [])
        if isinstance(out, list):
            for block in out:
                for c in block.get("content", []):
                    t = c.get("text", "")
                    if t:
                        return str(t).strip()
        return "AI yanıtı alınamadı."
    except HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            err_body = ""
        return f"AI servis hatası ({e.code}). {err_body[:220]}".strip()
    except URLError:
        return "AI bağlantı hatası: İnternet veya DNS erişimi kontrol edilmeli."
    except Exception as e:
        return f"AI bağlantı hatası: {e}"


def load_local_users() -> dict:
    conn = _db_connect()
    if conn is None:
        return {}
    try:
        rows = conn.execute("SELECT username, password FROM users").fetchall()
        out = {}
        for u, p in rows:
            user = _canonical_username(str(u))
            if user:
                out[user] = str(p or "")
        return out
    except Exception:
        return {}
    finally:
        conn.close()


def save_local_users(users: dict) -> None:
    conn = _db_connect()
    if conn is None:
        return
    try:
        for k, v in (users or {}).items():
            user = _canonical_username(str(k))
            if not user:
                continue
            conn.execute(
                """
                INSERT INTO users(username, password, updated_at)
                VALUES(?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(username) DO UPDATE SET
                    password=excluded.password,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (user, str(v or "")),
            )
        conn.commit()
    except Exception as e:
        _debug_log("save_local_users failed", e)
    finally:
        conn.close()


def save_local_user(username: str, password: str) -> None:
    username = _canonical_username(username)
    if not username:
        return
    conn = _db_connect()
    if conn is None:
        return
    try:
        conn.execute(
            """
            INSERT INTO users(username, password, updated_at)
            VALUES(?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(username) DO UPDATE SET
                password=excluded.password,
                updated_at=CURRENT_TIMESTAMP
            """,
            (username, str(password or "")),
        )
        conn.commit()
    except Exception as e:
        _debug_log("save_local_user failed", e)
    finally:
        conn.close()


def load_gsheets_users(url: str) -> dict:
    users = {}
    if not _check_cloud_sync_health(force=False):
        return users
    conn = get_gsheets_conn()
    if conn is None:
        _set_cloud_status(
            False,
            "Cloud sync unavailable: could not obtain gsheets connection while reading users.",
            "load_gsheets_users: connection is None",
        )
        return users
    df = _gsheets_read_df(conn, worksheet=USERS_WORKSHEET, url=url)
    if df is None or df.empty:
        if df is None:
            _set_cloud_status(
                False,
                "Cloud sync unavailable: worksheet 'users' read failed. Check worksheet name and permissions.",
                "load_gsheets_users: users worksheet returned None",
            )
            return users
        # empty sheet is acceptable
        _set_cloud_status(True)
        return users
    try:
        df.columns = [str(c).strip().lower() for c in df.columns]
        if not {"username", "password"}.issubset(df.columns):
            _set_cloud_status(
                False,
                "Cloud sync unavailable: worksheet 'users' must include username,password columns.",
                f"load_gsheets_users: invalid columns={list(df.columns)}",
            )
            return users
        df = df[["username", "password"]].dropna(how="all")
        df["username"] = df["username"].astype(str).map(_canonical_username)
        df["password"] = df["password"].astype(str).str.strip()
        df = df[df["username"] != ""]
        users = dict(zip(df["username"], df["password"]))
        _set_cloud_status(True)
    except Exception:
        _set_cloud_status(
            False,
            "Cloud sync unavailable: failed to parse 'users' worksheet data.",
            "load_gsheets_users: dataframe parse error",
        )
        return {}
    return users


def save_gsheets_users(url: str, users: dict) -> bool:
    if not _check_cloud_sync_health(force=False):
        return False
    conn = get_gsheets_conn()
    if conn is None:
        return False
    rows = [{"username": u, "password": p} for u, p in users.items() if str(u).strip()]
    df = pd.DataFrame(rows, columns=["username", "password"])
    return _gsheets_update_df(conn, df, worksheet=USERS_WORKSHEET, url=url)


def _merge_user_maps(base: dict, incoming: dict) -> dict:
    out = dict(base or {})
    for u, p in (incoming or {}).items():
        uu = _canonical_username(str(u))
        if not uu:
            continue
        pv = str(p or "")
        if pv or uu not in out:
            out[uu] = pv
    return out


def _merge_profile_payload(existing: dict, incoming: dict) -> dict:
    ex = existing if isinstance(existing, dict) else {}
    inc = incoming if isinstance(incoming, dict) else {}
    ex_dt = _parse_updated_at_safe(ex.get("updated_at"))
    inc_dt = _parse_updated_at_safe(inc.get("updated_at"))
    if inc_dt > ex_dt:
        return dict(inc)
    if ex_dt > inc_dt:
        return dict(ex)
    return {**ex, **inc}


def _merge_profile_maps(base: dict, incoming: dict) -> dict:
    out: dict = {}
    for src in (base or {}, incoming or {}):
        for u, p in src.items():
            uu = _canonical_username(str(u))
            if not uu:
                continue
            if uu not in out:
                out[uu] = dict(p) if isinstance(p, dict) else {}
            else:
                out[uu] = _merge_profile_payload(out.get(uu, {}), p if isinstance(p, dict) else {})
    return out


def save_gsheets_user(url: str, username: str, password: str, queue_on_fail: bool = True) -> bool:
    username = _canonical_username(username)
    if not username:
        return False
    if not _check_cloud_sync_health(force=False) or get_gsheets_conn() is None:
        if queue_on_fail:
            _enqueue_sync_action(
                {
                    "kind": "user_upsert",
                    "url": url,
                    "username": username,
                    "password": str(password or ""),
                }
            )
        return False
    target_password = str(password or "")
    rescue = _merge_user_maps({}, load_gsheets_users(url))
    for _ in range(5):
        current = load_gsheets_users(url)
        merged = _merge_user_maps(rescue, current)
        merged[username] = target_password
        if save_gsheets_users(url, merged):
            verify = load_gsheets_users(url)
            missing_users = [u for u in merged.keys() if u not in verify]
            if verify.get(username) == target_password and not missing_users:
                return True
            rescue = _merge_user_maps(merged, verify)
        time.sleep(0.35)
    if queue_on_fail:
        _enqueue_sync_action(
            {
                "kind": "user_upsert",
                "url": url,
                "username": username,
                "password": str(password or ""),
            }
        )
    return False


def load_all_profiles() -> dict:
    conn = _db_connect()
    if conn is None:
        return {}
    try:
        rows = conn.execute("SELECT username, payload_json FROM profiles").fetchall()
        out = {}
        for u, payload_raw in rows:
            user = _canonical_username(str(u))
            if not user:
                continue
            try:
                payload = json.loads(str(payload_raw or "{}"))
            except Exception:
                payload = {}
            out[user] = payload if isinstance(payload, dict) else {}
        return out
    except Exception:
        return {}
    finally:
        conn.close()


def load_local_profile(username: str) -> dict:
    username = _canonical_username(username)
    if not username:
        return {}
    conn = _db_connect()
    if conn is None:
        return {}
    try:
        row = conn.execute(
            "SELECT payload_json FROM profiles WHERE username = ?",
            (username,),
        ).fetchone()
        if not row:
            return {}
        payload_raw = str(row[0] or "{}")
        payload = json.loads(payload_raw)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}
    finally:
        conn.close()


def load_user_profile(username: str) -> dict:
    username = _canonical_username(username)
    if not username:
        return {}
    # local/remote profil seciminde en guncel kaydi kullan
    remote = load_gsheets_profiles(sheet_url).get(username, {})
    local = load_local_profile(username)
    has_local = isinstance(local, dict) and bool(local)
    has_remote = isinstance(remote, dict) and bool(remote)
    if has_local and not has_remote:
        return local
    if has_remote and not has_local:
        return remote
    if not has_local and not has_remote:
        return {}
    local_dt = _parse_updated_at_safe(local.get("updated_at"))
    remote_dt = _parse_updated_at_safe(remote.get("updated_at"))
    if remote_dt > local_dt:
        return remote
    # esitse/belirsizse local tercih edilir (rollback hissini azaltir)
    return local


def save_user_profile(username: str, payload: dict) -> None:
    username = _canonical_username(username)
    if not username:
        return
    safe_payload = payload if isinstance(payload, dict) else {}
    safe_payload = dict(safe_payload)
    safe_payload["updated_at"] = _now_iso()
    conn = _db_connect()
    if conn is not None:
        try:
            conn.execute(
                """
                INSERT INTO profiles(username, payload_json, updated_at)
                VALUES(?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(username) DO UPDATE SET
                    payload_json=excluded.payload_json,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (username, json.dumps(safe_payload, ensure_ascii=False)),
            )
            conn.commit()
        except Exception as e:
            _debug_log("Local profile save failed", e)
        finally:
            conn.close()
    cloud_ok = save_gsheets_profile(sheet_url, username, safe_payload)
    if not cloud_ok and not st.session_state.get("_warned_cloud_save_profile", False):
        st.session_state["_warned_cloud_save_profile"] = True
        msg = "Cloud sync failed; saved to local backup."
        try:
            st.toast(msg)
        except Exception as e:
            _debug_log("toast failed for cloud save warning", e)
        st.warning(msg)


def load_gsheets_profiles(url: str) -> dict:
    profiles = {}
    if not _check_cloud_sync_health(force=False):
        return profiles
    conn = get_gsheets_conn()
    df = _gsheets_read_df(conn, worksheet=PROFILES_WORKSHEET, url=url)
    if df is None or df.empty:
        if df is None:
            _set_cloud_status(
                False,
                "Cloud sync unavailable: worksheet 'profiles' is not readable.",
                "load_gsheets_profiles: profiles worksheet returned None",
            )
        else:
            _set_cloud_status(True)
        return profiles
    try:
        df.columns = [str(c).strip().lower() for c in df.columns]
        required = {"username", "kilo", "yas", "nsaa_total", "nsaa_prev_total", "nsaa_history"}
        if not required.issubset(df.columns):
            _set_cloud_status(
                False,
                "Cloud sync unavailable: worksheet 'profiles' missing required columns.",
                f"load_gsheets_profiles: invalid columns={list(df.columns)}",
            )
            return profiles
        for _, row in df.iterrows():
            try:
                user = _canonical_username(str(row.get("username", "")))
                if not user:
                    continue
                raw_history = row.get("nsaa_history", "[]")
                try:
                    hist = json.loads(raw_history) if isinstance(raw_history, str) else (raw_history or [])
                except Exception:
                    hist = []
                profiles[user] = {
                    "kilo": float(row.get("kilo", 30.0)),
                    "yas": int(float(row.get("yas", 6))),
                    "nsaa_total": int(float(row.get("nsaa_total", 0))),
                    "nsaa_prev_total": None if str(row.get("nsaa_prev_total", "")).strip() == "" else int(float(row.get("nsaa_prev_total", 0))),
                    "nsaa_history": hist if isinstance(hist, list) else [],
                    "updated_at": str(row.get("updated_at", "")).strip(),
                }
            except Exception:
                continue
    except Exception:
        _set_cloud_status(
            False,
            "Cloud sync unavailable: failed to parse 'profiles' worksheet data.",
            "load_gsheets_profiles: dataframe parse error",
        )
        return {}
    _set_cloud_status(True)
    return profiles


def save_gsheets_profile(url: str, username: str, payload: dict, queue_on_fail: bool = True) -> bool:
    username = _canonical_username(username)
    if not username:
        return False
    if not _check_cloud_sync_health(force=False):
        if queue_on_fail:
            _enqueue_sync_action(
                {
                    "kind": "profile_upsert",
                    "url": url,
                    "username": username,
                    "payload": payload if isinstance(payload, dict) else {},
                }
            )
        return False
    conn = get_gsheets_conn()
    if conn is None:
        if queue_on_fail:
            _enqueue_sync_action(
                {
                    "kind": "profile_upsert",
                    "url": url,
                    "username": username,
                    "payload": payload if isinstance(payload, dict) else {},
                }
            )
        return False
    p_new = payload if isinstance(payload, dict) else {}
    p_new = dict(p_new)
    if not str(p_new.get("updated_at", "")).strip():
        p_new["updated_at"] = _now_iso()
    rescue = _merge_profile_maps({}, load_gsheets_profiles(url))
    for _ in range(5):
        current = load_gsheets_profiles(url)
        all_profiles = _merge_profile_maps(rescue, current)
        all_profiles[username] = _merge_profile_payload(all_profiles.get(username, {}), p_new)
        rescue = _merge_profile_maps(rescue, all_profiles)

        p_new_write = dict(all_profiles.get(username, {}))
        if not str(p_new_write.get("updated_at", "")).strip():
            p_new_write["updated_at"] = _now_iso()

        rows_with_updated = []
        rows_legacy = []
        for user, p in all_profiles.items():
            user = _canonical_username(user)
            if not user:
                continue
            row_base = {
                "username": user,
                "kilo": p.get("kilo", 30.0),
                "yas": p.get("yas", 6),
                "nsaa_total": p.get("nsaa_total", 0),
                "nsaa_prev_total": p.get("nsaa_prev_total", ""),
                "nsaa_history": json.dumps(p.get("nsaa_history", []), ensure_ascii=False),
            }
            rows_legacy.append(dict(row_base))
            row_base["updated_at"] = str(p.get("updated_at", "") or "")
            rows_with_updated.append(row_base)

        df_with_updated = pd.DataFrame(
            rows_with_updated,
            columns=["username", "kilo", "yas", "nsaa_total", "nsaa_prev_total", "nsaa_history", "updated_at"],
        )
        wrote = _gsheets_update_df(conn, df_with_updated, worksheet=PROFILES_WORKSHEET, url=url)
        if not wrote:
            # worksheet semasinda updated_at yoksa geriye donuk yazi dene
            df_legacy = pd.DataFrame(
                rows_legacy,
                columns=["username", "kilo", "yas", "nsaa_total", "nsaa_prev_total", "nsaa_history"],
            )
            wrote = _gsheets_update_df(conn, df_legacy, worksheet=PROFILES_WORKSHEET, url=url)
        if wrote:
            verify = load_gsheets_profiles(url)
            missing_users = [u for u in all_profiles.keys() if u not in verify]
            target_ok = username in verify and (
                _parse_updated_at_safe(verify.get(username, {}).get("updated_at"))
                >= _parse_updated_at_safe(p_new_write.get("updated_at"))
            )
            if target_ok and not missing_users:
                return True
            rescue = _merge_profile_maps(all_profiles, verify)
        time.sleep(0.35)
    if queue_on_fail:
        _enqueue_sync_action(
            {
                "kind": "profile_upsert",
                "url": url,
                "username": username,
                "payload": payload if isinstance(payload, dict) else {},
            }
        )
    return False


def delete_gsheets_profile(url: str, username: str, queue_on_fail: bool = True) -> bool:
    username = _canonical_username(username)
    if not username:
        return False
    if not _check_cloud_sync_health(force=False):
        if queue_on_fail:
            _enqueue_sync_action(
                {
                    "kind": "profile_delete",
                    "url": url,
                    "username": username,
                }
            )
        return False
    conn = get_gsheets_conn()
    if conn is None:
        if queue_on_fail:
            _enqueue_sync_action(
                {
                    "kind": "profile_delete",
                    "url": url,
                    "username": username,
                }
            )
        return False
    rescue = _merge_profile_maps({}, load_gsheets_profiles(url))
    rescue.pop(username, None)
    for _ in range(5):
        current = load_gsheets_profiles(url)
        current.pop(username, None)
        all_profiles = _merge_profile_maps(rescue, current)
        all_profiles.pop(username, None)
        rows_with_updated = []
        rows_legacy = []
        for user, p in all_profiles.items():
            user = _canonical_username(user)
            if not user:
                continue
            row_base = {
                "username": user,
                "kilo": p.get("kilo", 30.0),
                "yas": p.get("yas", 6),
                "nsaa_total": p.get("nsaa_total", 0),
                "nsaa_prev_total": p.get("nsaa_prev_total", ""),
                "nsaa_history": json.dumps(p.get("nsaa_history", []), ensure_ascii=False),
            }
            rows_legacy.append(dict(row_base))
            row_base["updated_at"] = str(p.get("updated_at", "") or "")
            rows_with_updated.append(row_base)

        df_with_updated = pd.DataFrame(
            rows_with_updated,
            columns=["username", "kilo", "yas", "nsaa_total", "nsaa_prev_total", "nsaa_history", "updated_at"],
        )
        wrote = _gsheets_update_df(conn, df_with_updated, worksheet=PROFILES_WORKSHEET, url=url)
        if not wrote:
            df_legacy = pd.DataFrame(
                rows_legacy,
                columns=["username", "kilo", "yas", "nsaa_total", "nsaa_prev_total", "nsaa_history"],
            )
            wrote = _gsheets_update_df(conn, df_legacy, worksheet=PROFILES_WORKSHEET, url=url)
        if wrote:
            verify = load_gsheets_profiles(url)
            missing_users = [u for u in all_profiles.keys() if u not in verify]
            if username not in verify and not missing_users:
                return True
            rescue = _merge_profile_maps(all_profiles, verify)
            rescue.pop(username, None)
        time.sleep(0.35)
    if queue_on_fail:
        _enqueue_sync_action(
            {
                "kind": "profile_delete",
                "url": url,
                "username": username,
            }
        )
    return False


def _drain_sync_queue(max_items: int = 20) -> tuple[int, int]:
    if not _check_cloud_sync_health(force=False) or get_gsheets_conn() is None:
        return (0, len(_load_sync_queue()))
    q = _load_sync_queue()
    if not q:
        return (0, 0)
    processed = 0
    remaining: list[dict] = []
    for rec in q:
        if processed >= max_items:
            remaining.append(rec)
            continue
        kind = str(rec.get("kind", ""))
        ok = False
        if kind == "user_upsert":
            ok = save_gsheets_user(
                str(rec.get("url", sheet_url)),
                str(rec.get("username", "")),
                str(rec.get("password", "")),
                queue_on_fail=False,
            )
        elif kind == "profile_upsert":
            payload = rec.get("payload", {})
            ok = save_gsheets_profile(
                str(rec.get("url", sheet_url)),
                str(rec.get("username", "")),
                payload if isinstance(payload, dict) else {},
                queue_on_fail=False,
            )
        elif kind == "profile_delete":
            ok = delete_gsheets_profile(
                str(rec.get("url", sheet_url)),
                str(rec.get("username", "")),
                queue_on_fail=False,
            )
        if ok:
            processed += 1
        else:
            remaining.append(rec)
    _save_sync_queue(remaining)
    return (processed, len(remaining))

def load_gsheets_data(url: str) -> pd.DataFrame:
    """
    Google Sheets'ten kullanıcı verisini güvenli şekilde okur.
    - Hata olursa UI'yi kirletmez (gereksiz tablo/exception metni basmaz)
    - Şema: username, password
    """
    empty_schema = pd.DataFrame(columns=["username", "password"])

    users = load_gsheets_users(url)
    if not users:
        return empty_schema
    return pd.DataFrame(
        [{"username": u, "password": p} for u, p in users.items()],
        columns=["username", "password"]
    )

# --- VERİYİ ÇEK VE GÜVENLİ HALE GETİR ---
_check_cloud_sync_health(force=True)
existing_data = load_gsheets_data(sheet_url)

# Ekstra güvenlik katmanı (defansif programlama)
if not isinstance(existing_data, pd.DataFrame):
    existing_data = pd.DataFrame(columns=["username", "password"])

# Boşsa standart şema garanti edilir
if existing_data.empty:
    existing_data = pd.DataFrame(columns=["username", "password"])

# --- 3. OTURUM HAFIZASI & USER DATABASE DİNAMİĞİ ---

# Giriş durumu kontrolü
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "profile_loaded_for" not in st.session_state:
    st.session_state.profile_loaded_for = None
if "login_attempts" not in st.session_state or not isinstance(st.session_state.get("login_attempts"), dict):
    st.session_state["login_attempts"] = {}
if "last_activity_ts" not in st.session_state:
    st.session_state["last_activity_ts"] = time.time()
if "session_timeout_sec" not in st.session_state:
    st.session_state["session_timeout_sec"] = 0
if "lock_window_sec" not in st.session_state:
    st.session_state["lock_window_sec"] = 300
if "max_login_attempts" not in st.session_state:
    st.session_state["max_login_attempts"] = 5
if "lang" not in st.session_state or st.session_state.get("lang") not in ("TR", "EN", "DE"):
    st.session_state["lang"] = "TR"

# Kullanıcı veritabanını session_state üzerinde initialize et
if "users_db" not in st.session_state:
    temp_db = {}

    # 1) Öncelik: Google Sheets'ten gelen veriler (güvenli okuma)
    if isinstance(existing_data, pd.DataFrame) and not existing_data.empty:
        cols = [c.lower().strip() for c in existing_data.columns]
        if "username" in cols and "password" in cols:
            # Kolon adlarını normalize edip kesin erişim sağla
            dfu = existing_data.copy()
            dfu.columns = cols

            dfu["username"] = dfu["username"].astype(str).map(_canonical_username)
            dfu["password"] = dfu["password"].astype(str).str.strip()

            # boş username olanları alma
            dfu = dfu[dfu["username"] != ""]

            temp_db.update(dict(zip(dfu["username"], dfu["password"])))

    # 2) Öncelik: Secrets Entegrasyonu (Sistem Adminleri)
    try:
        secrets_usernames = st.secrets.get("usernames", {})
        if isinstance(secrets_usernames, dict):
            for user, pwd in secrets_usernames.items():
                u = _canonical_username(str(user))
                p = str(pwd).strip()
                if u:
                    temp_db[u] = p
    except Exception:
        _debug_log("secrets usernames load failed")

    # 3) Kalıcı yerel kullanıcı kayıtları (register sonrası kalıcı hafıza)
    local_users = load_local_users()
    if isinstance(local_users, dict):
        for user, pwd in local_users.items():
            u = _canonical_username(str(user))
            p = str(pwd).strip()
            if u:
                temp_db[u] = p

    # 4) Tek admin sahibi hesabı (varsa) zorunlu olarak eklenir
    admin_cfg = _load_admin_owner_config()
    admin_user = _canonical_username(str(admin_cfg.get("username", "")))
    admin_hash = str(admin_cfg.get("password_hash", "")).strip()
    if admin_user and admin_hash:
        temp_db[admin_user] = admin_hash

    # Session'a yaz
    st.session_state.users_db = temp_db

# Opsiyonel: None / bozuk tipleri düzelt
if st.session_state.get("kilo") is None:
    st.session_state["kilo"] = 30.0
if st.session_state.get("yas") is None:
    st.session_state["yas"] = 6
if st.session_state.get("nsaa_total") is None:
    st.session_state["nsaa_total"] = 0

# --- 4. GİRİŞ VE KAYIT EKRANI (SECURE AUTHENTICATION) ---
import hashlib
import hmac

def _hash_password(pw: str) -> str:
    pw_bytes = (pw or "").encode("utf-8")
    if BCRYPT_OK:
        try:
            return "bcrypt$" + bcrypt.hashpw(pw_bytes, bcrypt.gensalt()).decode("utf-8")
        except Exception as e:
            _debug_log("bcrypt hash failed, falling back to pbkdf2", e)
    iterations = 200_000
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw_bytes, salt, iterations)
    return "pbkdf2${}${}${}".format(
        iterations,
        base64.urlsafe_b64encode(salt).decode("utf-8"),
        base64.urlsafe_b64encode(dk).decode("utf-8"),
    )

def _verify_password(stored: str, provided: str) -> bool:
    stored = str(stored or "")
    provided = str(provided or "")

    if stored.startswith("bcrypt$") and BCRYPT_OK:
        try:
            return bool(bcrypt.checkpw(provided.encode("utf-8"), stored.split("$", 1)[1].encode("utf-8")))
        except Exception:
            return False
    if stored.startswith("$2") and BCRYPT_OK:
        try:
            return bool(bcrypt.checkpw(provided.encode("utf-8"), stored.encode("utf-8")))
        except Exception:
            return False

    # Yeni format: pbkdf2$iterations$salt$hash
    if stored.startswith("pbkdf2$"):
        try:
            _, iter_s, salt_b64, hash_b64 = stored.split("$", 3)
            iterations = int(iter_s)
            salt = base64.urlsafe_b64decode(salt_b64.encode("utf-8"))
            expected = base64.urlsafe_b64decode(hash_b64.encode("utf-8"))
            actual = hashlib.pbkdf2_hmac("sha256", provided.encode("utf-8"), salt, iterations)
            return hmac.compare_digest(actual, expected)
        except Exception:
            return False

    # Eski format: sha256$...
    if stored.startswith("sha256$"):
        legacy = "sha256$" + hashlib.sha256(provided.encode("utf-8")).hexdigest()
        return hmac.compare_digest(stored, legacy)

    # Eski format (plain) ile geriye dönük uyum:
    return hmac.compare_digest(stored, provided)


# --- NEW/UPDATED --- Legacy password -> bcrypt/pbkdf2 migration
def _migrate_legacy_password_if_needed(username: str, provided_password: str) -> None:
    user = _canonical_username(username)
    if not user:
        return
    stored = str(st.session_state.get("users_db", {}).get(user, ""))
    if not stored or stored.startswith("pbkdf2$") or stored.startswith("bcrypt$") or stored.startswith("$2"):
        return
    new_hash = _hash_password(provided_password)
    st.session_state["users_db"][user] = new_hash
    save_local_user(user, new_hash)
    save_gsheets_user(sheet_url, user, new_hash)


_bootstrap_admin_owner_from_config()
_admin_cfg_after_boot = _load_admin_owner_config()
_admin_u_after_boot = _canonical_username(str(_admin_cfg_after_boot.get("username", "")))
_admin_h_after_boot = str(_admin_cfg_after_boot.get("password_hash", "")).strip()
if _admin_u_after_boot and _admin_h_after_boot:
    st.session_state.setdefault("users_db", {})
    if isinstance(st.session_state.get("users_db"), dict):
        st.session_state["users_db"][_admin_u_after_boot] = _admin_h_after_boot


def _auth_secret() -> str:
    def _is_strong_secret(sec: str) -> bool:
        s = str(sec or "").strip()
        if not s:
            return False
        # Accept hex secrets (>=32 bytes => 64 hex chars)
        if re.fullmatch(r"[0-9a-fA-F]{64,}", s):
            return (len(s) // 2) >= 32
        # Accept URL-safe base64 secrets (>=32 bytes decoded)
        if len(s) < 43:
            return False
        try:
            raw = base64.urlsafe_b64decode((s + "==").encode("utf-8"))
            return len(raw) >= 32
        except Exception:
            return False

    def _harden_secret_file(path: Path) -> None:
        try:
            os.chmod(path, 0o600)
        except Exception:
            pass

    # 1) Streamlit secrets
    try:
        sec = str(st.secrets.get("auth_secret", "")).strip()
        if sec and _is_strong_secret(sec):
            return sec
        if sec:
            _debug_log("auth_secret in secrets is weak; expecting >=32 bytes entropy")
    except Exception as e:
        _debug_log("auth_secret load failed", e)
    # 2) Environment variable fallback
    try:
        env_sec = str(os.getenv("AUTH_SECRET", "")).strip()
        if env_sec and _is_strong_secret(env_sec):
            return env_sec
        if env_sec:
            _debug_log("AUTH_SECRET env is weak; expecting >=32 bytes entropy")
    except Exception:
        pass
    # 3) Local persistent secret fallback (keeps login across refresh/restart)
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if AUTH_SECRET_STORE.exists():
            local_sec = AUTH_SECRET_STORE.read_text(encoding="utf-8").strip()
            if local_sec and _is_strong_secret(local_sec):
                return local_sec
            _debug_log("auth_secret.txt is weak/invalid; rotating")
        generated = base64.urlsafe_b64encode(os.urandom(48)).decode("utf-8").rstrip("=")
        AUTH_SECRET_STORE.write_text(generated, encoding="utf-8")
        _harden_secret_file(AUTH_SECRET_STORE)
        return generated
    except Exception as e:
        _debug_log("local auth_secret fallback failed", e)
    return ""


# --- NEW/UPDATED --- persistent login security gate
def _persistent_login_query_enabled() -> bool:
    db_pref = _db_get_kv("persistent_login_enabled")
    if db_pref in {"1", "true", "yes", "on"}:
        return True
    if db_pref in {"0", "false", "no", "off"}:
        return False
    try:
        sec_v = str(st.secrets.get("persistent_login_via_query", "")).strip().lower()
    except Exception:
        sec_v = ""
    if sec_v in {"1", "true", "yes", "on"}:
        return True
    if sec_v in {"0", "false", "no", "off"}:
        return False
    env_v = str(os.getenv("DMD_PERSISTENT_LOGIN_QUERY", "1")).strip().lower()
    return env_v in {"1", "true", "yes", "on"}


def _persistent_login_enabled() -> bool:
    if not _persistent_login_query_enabled():
        return False
    sec = _auth_secret()
    if not sec:
        return False
    low = sec.lower()
    if len(sec) < 24:
        return False
    if "change_this" in low or "default" in low or "example" in low:
        return False
    return True


def _persistent_login_ttl_sec() -> int:
    raw = _db_get_kv("persistent_login_ttl_sec")
    if not raw:
        raw = str(os.getenv("DMD_PERSISTENT_LOGIN_TTL_SEC", "")).strip()
    if not raw:
        raw = "2592000"  # 30 days
    try:
        ttl = int(raw)
    except Exception:
        ttl = 2592000
    return max(3600, min(ttl, 7776000))  # 1 hour .. 90 days


def _build_persistent_token(username: str, ts: int | None = None) -> str:
    if not _persistent_login_enabled():
        return ""
    ts_i = int(ts if ts is not None else time.time())
    exp_i = ts_i + int(_persistent_login_ttl_sec())
    payload = f"{_canonical_username(username)}|{ts_i}|{exp_i}"
    sig = hmac.new(_auth_secret().encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{ts_i}.{exp_i}.{sig}"


def _is_valid_persistent_token(username: str, token: str, max_age_sec: int = 60 * 60 * 24 * 30) -> bool:
    if not _persistent_login_enabled():
        return False
    legacy_mode = False
    try:
        ts_s, exp_s, sig = str(token).split(".", 2)
        ts_i = int(ts_s)
        exp_i = int(exp_s)
    except Exception:
        # Backward compatibility: legacy token format ts.sig
        try:
            ts_s, sig = str(token).split(".", 1)
            ts_i = int(ts_s)
            exp_i = ts_i + int(max_age_sec)
            legacy_mode = True
        except Exception:
            return False
    now_i = int(time.time())
    if ts_i <= 0 or exp_i <= ts_i:
        return False
    if now_i > exp_i or ts_i > (now_i + 300):
        return False
    if legacy_mode:
        payload = f"{_canonical_username(username)}|{ts_i}"
    else:
        payload = f"{_canonical_username(username)}|{ts_i}|{exp_i}"
    expected_sig = hmac.new(_auth_secret().encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    return hmac.compare_digest(str(sig), expected_sig)


def _set_persistent_login(username: str) -> None:
    if not _persistent_login_enabled():
        _clear_persistent_login()
        return
    try:
        token = _build_persistent_token(username)
        if token:
            st.query_params["u"] = username
            st.query_params["auth"] = token
    except Exception as e:
        _debug_log("set persistent login failed", e)


def _clear_persistent_login() -> None:
    try:
        if "u" in st.query_params:
            del st.query_params["u"]
        if "auth" in st.query_params:
            del st.query_params["auth"]
    except Exception as e:
        _debug_log("clear persistent login failed", e)


def _try_restore_login_from_query() -> None:
    if st.session_state.get("logged_in"):
        return
    if not _persistent_login_enabled():
        _clear_persistent_login()
        return
    try:
        raw_user = st.query_params.get("u", "")
        raw_token = st.query_params.get("auth", "")
    except Exception:
        return
    if isinstance(raw_user, list):
        raw_user = raw_user[0] if raw_user else ""
    if isinstance(raw_token, list):
        raw_token = raw_token[0] if raw_token else ""
    user = _canonical_username(str(raw_user))
    token = str(raw_token or "")
    if not user or not token:
        return
    if user not in st.session_state.get("users_db", {}):
        _clear_persistent_login()
        return
    if not _is_valid_persistent_token(user, token):
        _clear_persistent_login()
        return
    st.session_state.logged_in = True
    st.session_state.current_user = user
    st.session_state.profile_loaded_for = None
    st.session_state["last_activity_ts"] = time.time()
    # Sayfa yenilemelerinde oturumu korumak için token'ı URL'de tut.
    # Güvenli çıkışta ve geçersiz durumda zaten temizlenir.


def _normalize_username(username: str) -> str:
    return _canonical_username(username)


def _password_policy_ok(pw: str) -> bool:
    # Minimum 8 karakter ve en az bir harf + bir rakam
    pw = str(pw or "")
    return len(pw) >= 8 and bool(re.search(r"[A-Za-z]", pw)) and bool(re.search(r"\d", pw))


def _ensure_core_state() -> None:
    if "audits" not in st.session_state or not isinstance(st.session_state.get("audits"), list):
        st.session_state["audits"] = []
    if "reminders" not in st.session_state or not isinstance(st.session_state.get("reminders"), list):
        st.session_state["reminders"] = []
    if "medications" not in st.session_state or not isinstance(st.session_state.get("medications"), list):
        st.session_state["medications"] = []
    if "side_effects" not in st.session_state or not isinstance(st.session_state.get("side_effects"), list):
        st.session_state["side_effects"] = []
    if "pul_score" not in st.session_state:
        st.session_state["pul_score"] = 0
    if "vignos_score" not in st.session_state:
        st.session_state["vignos_score"] = 1
    if "pul_history" not in st.session_state or not isinstance(st.session_state.get("pul_history"), list):
        st.session_state["pul_history"] = []
    if "vignos_history" not in st.session_state or not isinstance(st.session_state.get("vignos_history"), list):
        st.session_state["vignos_history"] = []
    # --- NEW/UPDATED --- Rol tabanli durum yonetimi
    if "user_role" not in st.session_state:
        st.session_state["user_role"] = "family"
    if st.session_state.get("user_role") not in {"family", "doctor", "researcher", "admin"}:
        st.session_state["user_role"] = "family"
    if "user_mode" not in st.session_state:
        st.session_state["user_mode"] = "Doktor" if st.session_state.get("user_role") in {"doctor", "admin"} else "Aile"
    if "doctor_notes" not in st.session_state or not isinstance(st.session_state.get("doctor_notes"), list):
        st.session_state["doctor_notes"] = []
    if "emergency_card" not in st.session_state or not isinstance(st.session_state.get("emergency_card"), dict):
        st.session_state["emergency_card"] = {}
    if "molecular_profile" not in st.session_state or not isinstance(st.session_state.get("molecular_profile"), dict):
        st.session_state["molecular_profile"] = {}
    if "care_plan" not in st.session_state or not isinstance(st.session_state.get("care_plan"), dict):
        st.session_state["care_plan"] = {}
    if "visits" not in st.session_state or not isinstance(st.session_state.get("visits"), list):
        st.session_state["visits"] = []
    if "documents" not in st.session_state or not isinstance(st.session_state.get("documents"), list):
        st.session_state["documents"] = []


# --- NEW/UPDATED --- Role-based default session timeout policy
def _default_session_timeout_sec(role: str) -> int:
    return 3600 if role in {"doctor", "admin"} else 1800


def _apply_session_timeout_policy(force: bool = False) -> None:
    role = st.session_state.get("user_role", "family")
    target = _default_session_timeout_sec(role)
    cur = st.session_state.get("session_timeout_sec")
    if force or not isinstance(cur, (int, float)) or float(cur) <= 0:
        st.session_state["session_timeout_sec"] = int(target)


def _add_audit(event: str, detail: str = "") -> None:
    _ensure_core_state()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state["audits"].append(
        {
            "time": ts,
            "user": st.session_state.get("current_user") or "-",
            "event": str(event),
            "detail": str(detail),
        }
    )
    st.session_state["audits"] = st.session_state["audits"][-300:]


def _safe_int(value, default: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_list(value) -> list:
    return value if isinstance(value, list) else []


def _safe_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


@dataclass
class PatientRecord:
    name: str = "Hasta"
    kilo: float = 30.0
    yas: int = 6
    nsaa_total: int = 0
    nsaa_prev_total: int | None = None
    nsaa_history: list = field(default_factory=list)
    pul_score: int = 0
    vignos_score: int = 1
    pul_history: list = field(default_factory=list)
    vignos_history: list = field(default_factory=list)
    medications: list = field(default_factory=list)
    side_effects: list = field(default_factory=list)
    reminders: list = field(default_factory=list)
    doctor_notes: list = field(default_factory=list)
    emergency_card: dict = field(default_factory=dict)
    molecular_profile: dict = field(default_factory=dict)
    care_plan: dict = field(default_factory=dict)
    visits: list = field(default_factory=list)
    documents: list = field(default_factory=list)

    @staticmethod
    def from_obj(obj: dict | None, fallback_name: str = "Hasta") -> "PatientRecord":
        obj = obj if isinstance(obj, dict) else {}
        return PatientRecord(
            name=str(obj.get("name", fallback_name) or fallback_name),
            kilo=_safe_float(obj.get("kilo", 30.0), 30.0),
            yas=_safe_int(obj.get("yas", 6), 6),
            nsaa_total=_safe_int(obj.get("nsaa_total", 0), 0),
            nsaa_prev_total=(
                None
                if str(obj.get("nsaa_prev_total", "")).strip() == ""
                else _safe_int(obj.get("nsaa_prev_total", 0), 0)
            ),
            nsaa_history=_safe_list(obj.get("nsaa_history", [])),
            pul_score=_safe_int(obj.get("pul_score", 0), 0),
            vignos_score=_safe_int(obj.get("vignos_score", 1), 1),
            pul_history=_safe_list(obj.get("pul_history", [])),
            vignos_history=_safe_list(obj.get("vignos_history", [])),
            medications=_safe_list(obj.get("medications", [])),
            side_effects=_safe_list(obj.get("side_effects", [])),
            reminders=_safe_list(obj.get("reminders", [])),
            doctor_notes=_safe_list(obj.get("doctor_notes", [])),
            emergency_card=_safe_dict(obj.get("emergency_card", {})),
            molecular_profile=_safe_dict(obj.get("molecular_profile", {})),
            care_plan=_safe_dict(obj.get("care_plan", {})),
            visits=_safe_list(obj.get("visits", [])),
            documents=_safe_list(obj.get("documents", [])),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kilo": float(self.kilo),
            "yas": int(self.yas),
            "nsaa_total": int(self.nsaa_total),
            "nsaa_prev_total": self.nsaa_prev_total,
            "nsaa_history": list(self.nsaa_history),
            "pul_score": int(self.pul_score),
            "vignos_score": int(self.vignos_score),
            "pul_history": list(self.pul_history),
            "vignos_history": list(self.vignos_history),
            "medications": list(self.medications),
            "side_effects": list(self.side_effects),
            "reminders": list(self.reminders),
            "doctor_notes": list(self.doctor_notes),
            "emergency_card": dict(self.emergency_card),
            "molecular_profile": dict(self.molecular_profile),
            "care_plan": dict(self.care_plan),
            "visits": list(self.visits),
            "documents": list(self.documents),
        }


def _normalize_patient_record(payload: dict, fallback_name: str) -> dict:
    return PatientRecord.from_obj(payload, fallback_name=fallback_name).to_dict()


def _ensure_patient_state() -> None:
    _ensure_core_state()
    if "patients" not in st.session_state or not isinstance(st.session_state.get("patients"), dict):
        st.session_state["patients"] = {}
    normalized_patients = {}
    for pid, payload in st.session_state["patients"].items():
        pid_s = str(pid or "").strip()
        if not pid_s:
            continue
        normalized_patients[pid_s] = _normalize_patient_record(payload if isinstance(payload, dict) else {}, fallback_name=pid_s)
    st.session_state["patients"] = normalized_patients
    if not st.session_state["patients"]:
        pid = "hasta_1"
        st.session_state["patients"][pid] = _normalize_patient_record(
            {
                "name": "Hasta 1",
                "kilo": float(st.session_state.get("kilo", 30.0)),
                "yas": int(st.session_state.get("yas", 6)),
                "nsaa_total": int(st.session_state.get("nsaa_total", 0)),
                "nsaa_prev_total": st.session_state.get("nsaa_prev_total"),
                "nsaa_history": st.session_state.get("nsaa_history", []),
                "pul_score": int(st.session_state.get("pul_score", 0)),
                "vignos_score": int(st.session_state.get("vignos_score", 1)),
                "pul_history": st.session_state.get("pul_history", []),
                "vignos_history": st.session_state.get("vignos_history", []),
                "medications": st.session_state.get("medications", []),
                "side_effects": st.session_state.get("side_effects", []),
                "reminders": st.session_state.get("reminders", []),
                "doctor_notes": st.session_state.get("doctor_notes", []),
                "emergency_card": st.session_state.get("emergency_card", {}),
                "molecular_profile": st.session_state.get("molecular_profile", {}),
                "care_plan": st.session_state.get("care_plan", {}),
                "visits": st.session_state.get("visits", []),
                "documents": st.session_state.get("documents", []),
            },
            fallback_name="Hasta 1",
        )
        st.session_state["active_patient_id"] = pid
    if "active_patient_id" not in st.session_state or st.session_state["active_patient_id"] not in st.session_state["patients"]:
        st.session_state["active_patient_id"] = next(iter(st.session_state["patients"]))


def _sync_active_patient_to_session() -> None:
    _ensure_patient_state()
    pid = st.session_state["active_patient_id"]
    p = _normalize_patient_record(st.session_state["patients"].get(pid, {}), fallback_name=pid)
    st.session_state["patients"][pid] = p
    st.session_state["kilo"] = float(p.get("kilo", 30.0))
    st.session_state["yas"] = int(p.get("yas", 6))
    st.session_state["nsaa_total"] = int(p.get("nsaa_total", 0))
    st.session_state["nsaa_prev_total"] = p.get("nsaa_prev_total")
    st.session_state["nsaa_history"] = p.get("nsaa_history", [])
    st.session_state["pul_score"] = int(p.get("pul_score", 0))
    st.session_state["vignos_score"] = int(p.get("vignos_score", 1))
    st.session_state["pul_history"] = p.get("pul_history", [])
    st.session_state["vignos_history"] = p.get("vignos_history", [])
    st.session_state["medications"] = p.get("medications", [])
    st.session_state["side_effects"] = p.get("side_effects", [])
    st.session_state["reminders"] = p.get("reminders", [])
    st.session_state["doctor_notes"] = p.get("doctor_notes", [])
    st.session_state["emergency_card"] = p.get("emergency_card", {})
    st.session_state["molecular_profile"] = p.get("molecular_profile", {})
    st.session_state["care_plan"] = p.get("care_plan", {})
    st.session_state["visits"] = p.get("visits", [])
    st.session_state["documents"] = p.get("documents", [])


def _sync_session_to_active_patient() -> None:
    _ensure_patient_state()
    pid = st.session_state["active_patient_id"]
    p = st.session_state["patients"].setdefault(pid, {"name": pid})
    merged = {
        **(p if isinstance(p, dict) else {}),
        "kilo": float(st.session_state.get("kilo", 30.0)),
        "yas": int(st.session_state.get("yas", 6)),
        "nsaa_total": int(st.session_state.get("nsaa_total", 0)),
        "nsaa_prev_total": st.session_state.get("nsaa_prev_total"),
        "nsaa_history": st.session_state.get("nsaa_history", []),
        "pul_score": int(st.session_state.get("pul_score", 0)),
        "vignos_score": int(st.session_state.get("vignos_score", 1)),
        "pul_history": st.session_state.get("pul_history", []),
        "vignos_history": st.session_state.get("vignos_history", []),
        "medications": st.session_state.get("medications", []),
        "side_effects": st.session_state.get("side_effects", []),
        "reminders": st.session_state.get("reminders", []),
        "doctor_notes": st.session_state.get("doctor_notes", []),
        "emergency_card": st.session_state.get("emergency_card", {}),
        "molecular_profile": st.session_state.get("molecular_profile", {}),
        "care_plan": st.session_state.get("care_plan", {}),
        "visits": st.session_state.get("visits", []),
        "documents": st.session_state.get("documents", []),
    }
    st.session_state["patients"][pid] = _normalize_patient_record(merged, fallback_name=pid)


def _save_pipeline(audit_event: str | None = None, detail: str = "") -> None:
    _sync_session_to_active_patient()
    if audit_event:
        _add_audit(audit_event, detail)
    save_current_session_profile()


def _quality_checks() -> list[str]:
    issues = []
    kilo = float(st.session_state.get("kilo", 0))
    yas = int(st.session_state.get("yas", 0))
    nsaa = int(st.session_state.get("nsaa_total", 0))
    if kilo <= 0 or kilo > 200:
        issues.append("Kilo değeri olağan dışı görünüyor.")
    if yas < 0 or yas > 80:
        issues.append("Yaş değeri olağan dışı görünüyor.")
    if nsaa < 0 or nsaa > 34:
        issues.append("NSAA skoru 0-34 aralığında olmalıdır.")
    if len(st.session_state.get("nsaa_history", [])) == 0:
        issues.append("NSAA geçmiş kaydı henüz yok.")
    return issues


def _visit_snapshot(source: str, note: str = "") -> dict:
    # --- NEW/UPDATED --- Ziyaret modeli (yeni sema + geriye donuk alanlar)
    today_s = datetime.now().strftime("%Y-%m-%d")
    now_s = datetime.now().strftime("%Y-%m-%d %H:%M")
    mp = st.session_state.get("molecular_profile", {}) or {}
    return {
        "date": today_s,
        "age": int(st.session_state.get("yas", 6)),
        "weight": float(st.session_state.get("kilo", 30.0)),
        "nsaa": int(st.session_state.get("nsaa_total", 0)),
        "pul": int(st.session_state.get("pul_score", 0)),
        "vignos": int(st.session_state.get("vignos_score", 1)),
        "ef": int(mp.get("ef")) if str(mp.get("ef", "")).strip() != "" else None,
        "fvc": int(mp.get("fvc")) if str(mp.get("fvc", "")).strip() != "" else None,
        "notes": str(note or ""),
        # geriye donuk
        "time": now_s,
        "source": str(source or "manual"),
        "note": str(note or ""),
        "kilo": float(st.session_state.get("kilo", 30.0)),
        "yas": int(st.session_state.get("yas", 6)),
        "nsaa_total": int(st.session_state.get("nsaa_total", 0)),
        "pul_score": int(st.session_state.get("pul_score", 0)),
        "vignos_score": int(st.session_state.get("vignos_score", 1)),
    }


def _append_visit(source: str, note: str = "") -> None:
    _ensure_core_state()
    visits = st.session_state.get("visits", [])
    if not isinstance(visits, list):
        visits = []
    visits.append(_visit_snapshot(source=source, note=note))
    st.session_state["visits"] = visits[-300:]
    _sync_session_to_active_patient()


# --- NEW/UPDATED --- Visit semasi/yardimcilari
def _normalize_visit_record(v: dict) -> dict:
    v = v if isinstance(v, dict) else {}
    date_s = str(v.get("date") or "").strip()
    if not date_s:
        raw_time = str(v.get("time") or "").strip()
        date_s = raw_time[:10] if len(raw_time) >= 10 else datetime.now().strftime("%Y-%m-%d")

    def _i(val):
        try:
            if val is None or str(val).strip() == "":
                return None
            return int(float(val))
        except Exception:
            return None

    def _f(val):
        try:
            if val is None or str(val).strip() == "":
                return None
            return float(val)
        except Exception:
            return None

    age = _i(v.get("age", v.get("yas")))
    weight = _f(v.get("weight", v.get("kilo")))
    nsaa = _i(v.get("nsaa", v.get("nsaa_total")))
    pul = _i(v.get("pul", v.get("pul_score")))
    vignos = _i(v.get("vignos", v.get("vignos_score")))
    ef = _i(v.get("ef"))
    fvc = _i(v.get("fvc"))
    notes = str(v.get("notes", v.get("note", "")) or "").strip()

    return {
        "date": date_s,
        "age": age,
        "weight": weight,
        "nsaa": nsaa,
        "pul": pul,
        "vignos": vignos,
        "ef": ef,
        "fvc": fvc,
        "notes": notes,
        # compatibility
        "time": str(v.get("time") or f"{date_s} 00:00"),
        "source": str(v.get("source", "manual")),
        "note": notes,
        "kilo": weight,
        "yas": age,
        "nsaa_total": nsaa,
        "pul_score": pul,
        "vignos_score": vignos,
    }


def _ensure_patient_visits_schema() -> None:
    _ensure_patient_state()
    patients = st.session_state.get("patients", {})
    for pid, p in list(patients.items()):
        if not isinstance(p, dict):
            continue
        raw_visits = p.get("visits", [])
        if not isinstance(raw_visits, list):
            raw_visits = []
        # --- NEW/UPDATED --- one-time migration: nsaa_history -> visits
        if not raw_visits:
            ns_hist = p.get("nsaa_history", [])
            if isinstance(ns_hist, list) and ns_hist:
                migrated = []
                for rec in ns_hist:
                    if not isinstance(rec, dict):
                        continue
                    t = str(rec.get("time", "")).strip()
                    d = t[:10] if len(t) >= 10 else datetime.now().strftime("%Y-%m-%d")
                    migrated.append(
                        {
                            "date": d,
                            "age": int(p.get("yas", st.session_state.get("yas", 6))),
                            "weight": float(p.get("kilo", st.session_state.get("kilo", 30.0))),
                            "nsaa": int(rec.get("total", p.get("nsaa_total", 0))),
                            "pul": int(p.get("pul_score", st.session_state.get("pul_score", 0))),
                            "vignos": int(p.get("vignos_score", st.session_state.get("vignos_score", 1))),
                            "ef": None,
                            "fvc": None,
                            "notes": "NSAA geçmişinden migrasyon",
                        }
                    )
                raw_visits = migrated
        normalized = [_normalize_visit_record(v) for v in raw_visits]
        normalized = sorted(normalized, key=lambda x: str(x.get("date", "")))
        p["visits"] = normalized[-300:]
        patients[pid] = p
    st.session_state["patients"] = patients
    apid = st.session_state.get("active_patient_id")
    if apid in patients:
        st.session_state["visits"] = patients[apid].get("visits", [])


def _upsert_active_patient_visit(note: str = "", source: str = "manual") -> dict:
    _ensure_patient_visits_schema()
    _sync_session_to_active_patient()
    today_s = datetime.now().strftime("%Y-%m-%d")
    new_visit = _visit_snapshot(source=source, note=note)
    new_visit["date"] = today_s
    pid = st.session_state.get("active_patient_id")
    p = st.session_state.get("patients", {}).get(pid, {})
    visits = p.get("visits", []) if isinstance(p, dict) else []
    if not isinstance(visits, list):
        visits = []
    replaced = False
    out = []
    for rec in visits:
        nrec = _normalize_visit_record(rec)
        if nrec.get("date") == today_s:
            keep_notes = str(new_visit.get("notes") or "").strip() or str(nrec.get("notes") or "").strip()
            merged = _normalize_visit_record({**nrec, **new_visit, "notes": keep_notes, "note": keep_notes})
            out.append(merged)
            replaced = True
        else:
            out.append(nrec)
    if not replaced:
        out.append(_normalize_visit_record(new_visit))
    out = sorted(out, key=lambda x: str(x.get("date", "")))[-300:]
    st.session_state["visits"] = out
    _sync_session_to_active_patient()
    return next((x for x in out if x.get("date") == today_s), out[-1] if out else _normalize_visit_record(new_visit))


def _infer_targeted_options(mut_type: str, exon_text: str, nonsense_flag: bool) -> list[str]:
    """
    Basit karar desteği: kesin tedavi önerisi değildir.
    """
    options: list[str] = []
    ex = (exon_text or "").lower()
    mt = (mut_type or "").lower()
    if nonsense_flag or "nonsense" in mt:
        options.append("Stop-codon readthrough yaklaşımı (ülke/endikasyon uygunluğu ile) değerlendirilebilir.")
    if "51" in ex:
        options.append("Exon 51 hedefli exon-skipping seçenekleri uygunluk açısından incelenebilir.")
    if "53" in ex:
        options.append("Exon 53 hedefli exon-skipping seçenekleri uygunluk açısından incelenebilir.")
    if "45" in ex:
        options.append("Exon 45 hedefli exon-skipping seçenekleri uygunluk açısından incelenebilir.")
    if not options:
        options.append("Hedefe yönelik tedavi uygunluğu için mutasyon raporu uzman merkezde ayrıntılı değerlendirilmelidir.")
    return options


def _cleanup_login_attempts() -> None:
    attempts = st.session_state.get("login_attempts", {})
    if not isinstance(attempts, dict):
        st.session_state["login_attempts"] = {}
        return
    now_ts = time.time()
    lock_window = float(st.session_state.get("lock_window_sec", 300))
    cleaned = {}
    for user, state in attempts.items():
        try:
            lock_until = float((state or {}).get("lock_until", 0.0))
            count = int((state or {}).get("count", 0))
            # Süresi geçen kilitleri temizle; aktif sayaçlar kalsın.
            if lock_until > 0 and (now_ts - lock_until) > lock_window:
                lock_until = 0.0
                count = 0
            cleaned[user] = {"count": count, "lock_until": lock_until}
        except Exception:
            continue
    # Boyutu sınırlı tut
    if len(cleaned) > 1000:
        for k in list(cleaned.keys())[: len(cleaned) - 1000]:
            cleaned.pop(k, None)
    st.session_state["login_attempts"] = cleaned


_cleanup_login_attempts()


def _build_text_report() -> str:
    _sync_session_to_active_patient()
    username = st.session_state.get("current_user", "-")
    pid = st.session_state.get("active_patient_id", "-")
    pname = st.session_state.get("patients", {}).get(pid, {}).get("name", pid)
    lines = [
        "Neurodegenerative Clinical Platform - Klinik Rapor",
        f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Kullanıcı: {username}",
        f"Hasta: {pname} ({pid})",
        "",
        f"Kilo: {st.session_state.get('kilo', '')}",
        f"Yaş: {st.session_state.get('yas', '')}",
        f"NSAA Toplam: {st.session_state.get('nsaa_total', '')}/34",
        f"PUL: {st.session_state.get('pul_score', '')}",
        f"Vignos: {st.session_state.get('vignos_score', '')}",
        "",
        "Kalite Kontrolleri:",
    ]
    issues = _quality_checks()
    if issues:
        lines.extend([f"- {i}" for i in issues])
    else:
        lines.append("- Kritik veri hatası bulunmadı.")
    return "\n".join(lines)


def _build_emergency_card_text() -> str:
    _sync_session_to_active_patient()
    card = st.session_state.get("emergency_card", {}) or {}
    pid = st.session_state.get("active_patient_id", "-")
    pname = st.session_state.get("patients", {}).get(pid, {}).get("name", pid)
    lines = [
        "DMD ACIL DURUM KARTI",
        f"Olusturma Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Hasta: {card.get('patient_name') or pname}",
        f"Yas: {card.get('age') or st.session_state.get('yas', '-')}",
        f"Kilo: {card.get('weight') or st.session_state.get('kilo', '-')}",
        f"Yakini Tel: {card.get('contact_phone', '-')}",
        f"Sorumlu Hekim: {card.get('doctor_name', '-')}",
        f"Hastane: {card.get('hospital', '-')}",
        f"Kullandigi Steroid: {card.get('steroid', '-')}",
        f"Alerji Notu: {card.get('allergy', '-')}",
        "",
        "KRITIK UYARI:",
        "- Suksinilkolin ve inhalasyon anestezikleri DMD'de risklidir.",
        "- Solunum ve kardiyak izlem geciktirilmemelidir.",
        "- Duzenli steroid kullaniyorsa stres dozu gereksinimi degerlendirilmelidir.",
    ]
    return "\n".join(lines)


def _patients_csv_bytes() -> bytes:
    _sync_session_to_active_patient()
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["patient_id", "name", "kilo", "yas", "nsaa_total", "pul_score", "vignos_score"])
    for pid, p in st.session_state.get("patients", {}).items():
        writer.writerow([
            pid,
            p.get("name", pid),
            p.get("kilo", ""),
            p.get("yas", ""),
            p.get("nsaa_total", ""),
            p.get("pul_score", ""),
            p.get("vignos_score", ""),
        ])
    return out.getvalue().encode("utf-8")


def _visits_csv_bytes() -> bytes:
    _sync_session_to_active_patient()
    _ensure_patient_visits_schema()
    out = io.StringIO()
    writer = csv.writer(out)
    # --- NEW/UPDATED --- Yeni ziyaret modeli export
    writer.writerow(["date", "age", "weight", "nsaa", "pul", "vignos", "ef", "fvc", "notes"])
    for v in st.session_state.get("visits", []):
        nv = _normalize_visit_record(v)
        writer.writerow(
            [
                nv.get("date", ""),
                nv.get("age", ""),
                nv.get("weight", ""),
                nv.get("nsaa", ""),
                nv.get("pul", ""),
                nv.get("vignos", ""),
                nv.get("ef", ""),
                nv.get("fvc", ""),
                nv.get("notes", ""),
            ]
        )
    return out.getvalue().encode("utf-8")


def _build_pdf_report_bytes() -> bytes | None:
    if not REPORTLAB_OK:
        return None
    text = _build_text_report().splitlines()
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 40
    c.setFont("Helvetica", 10)
    for line in text:
        c.drawString(40, y, line[:110])
        y -= 14
        if y < 40:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = height - 40
    c.save()
    return buf.getvalue()


def save_current_session_profile() -> None:
    username = st.session_state.get("current_user")
    if not username:
        return

    _sync_session_to_active_patient()
    audits_compact = st.session_state.get("audits", [])[-200:]
    payload = {
        # Geriye dönük uyum için temel alanlar tutuluyor.
        "kilo": float(st.session_state.get("kilo", 30.0)),
        "yas": int(st.session_state.get("yas", 6)),
        "nsaa_total": int(st.session_state.get("nsaa_total", 0)),
        "nsaa_prev_total": st.session_state.get("nsaa_prev_total"),
        "nsaa_history": st.session_state.get("nsaa_history", []),
        "doctor_notes": st.session_state.get("doctor_notes", []),
        "emergency_card": st.session_state.get("emergency_card", {}),
        "molecular_profile": st.session_state.get("molecular_profile", {}),
        "care_plan": st.session_state.get("care_plan", {}),
        "visits": st.session_state.get("visits", []),
        "documents": st.session_state.get("documents", []),
        # Yeni model
        "patients": st.session_state.get("patients", {}),
        "active_patient_id": st.session_state.get("active_patient_id"),
        "audits": audits_compact,
        "user_mode": st.session_state.get("user_mode", "Aile"),
        # --- NEW/UPDATED --- rol bilgisini kalici tut
        "user_role": st.session_state.get("user_role", "family"),
        "privacy_settings": st.session_state.get("privacy_settings", {}),
        "notification_ack": st.session_state.get("notification_ack", {}),
        "updated_at": _now_iso(),
    }
    # Gereksiz sık kaydı engelle: değişmediyse veya çok kısa aralıksa yazma.
    payload_sig = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    now = time.time()
    last_sig = st.session_state.get("_last_profile_sig")
    last_ts = float(st.session_state.get("_last_profile_save_ts", 0.0))
    if payload_sig == last_sig and (now - last_ts) < 10:
        return
    save_user_profile(username, payload)
    st.session_state["_last_profile_sig"] = payload_sig
    st.session_state["_last_profile_save_ts"] = now


# Yenilemede oturumu koru (imzalı query param ile)
_try_restore_login_from_query()
_apply_session_timeout_policy(force=False)
if ("_last_sync_drain_ts" not in st.session_state) or (time.time() - float(st.session_state.get("_last_sync_drain_ts", 0)) > 45):
    done, pending = _drain_sync_queue(max_items=25)
    st.session_state["_last_sync_drain_ts"] = time.time()
    st.session_state["_sync_queue_pending"] = pending
    st.session_state["_sync_queue_done_last"] = done

# Oturum zaman aşımı
now_activity = time.time()
if st.session_state.get("logged_in"):
    last_activity = float(st.session_state.get("last_activity_ts", now_activity))
    timeout_sec = float(st.session_state.get("session_timeout_sec", 0))
    if timeout_sec > 0 and (now_activity - last_activity > timeout_sec):
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.session_state.profile_loaded_for = None
        _clear_persistent_login()
        st.warning("Oturum süresi doldu. Lütfen tekrar giriş yapın.")
st.session_state["last_activity_ts"] = now_activity


if not st.session_state.logged_in:
    lang_labels = {
        "TR": "Turkce",
        "EN": "English",
        "DE": "Deutsch",
    }
    lang_options = ["TR", "EN", "DE"]
    login_lang = st.selectbox(
        "Dil / Language / Sprache",
        options=lang_options,
        index=lang_options.index(st.session_state.get("lang", "TR")),
        format_func=lambda x: f"{x} - {lang_labels.get(x, x)}",
        key="login_lang_select",
    )
    if login_lang != st.session_state.get("lang"):
        st.session_state["lang"] = login_lang
        st.rerun()

    st.markdown("""
        <div style="text-align: center; padding-bottom: 20px;">
            <h2 style="color: #1c83e1; font-family: 'Poppins'; margin-bottom: 6px;">Guardian Access</h2>
            <p style="color: #666; font-size: 0.9rem; margin-top: 0;">
                Neurodegenerative Clinical Platform sistemine giriş yapın
            </p>
        </div>
    """, unsafe_allow_html=True)

    auth_col1, auth_col2, auth_col3 = st.columns([1, 2, 1])

    with auth_col2:
        tab1, tab2 = st.tabs(["Mevcut Kullanıcı", "Yeni Kayıt"])

        # -------- LOGIN --------
        with tab1:
            with st.form("login_form"):
                role_labels = {
                    "family": "Aile",
                    "doctor": "Doktor",
                    "researcher": "Araştırmacı",
                    "admin": "Admin",
                }
                login_role = st.selectbox(
                    "Rol",
                    options=["family", "doctor", "researcher", "admin"],
                    format_func=lambda r: role_labels.get(r, r),
                )
                u_name = st.text_input("Kullanıcı Adı", placeholder="Kullanıcı adınızı giriniz...")
                u_pass = st.text_input("Şifre", type="password", placeholder="••••••••")
                submit_login = st.form_submit_button("SİSTEME GİRİŞ YAP", use_container_width=True)

                if submit_login:
                    user_clean = _normalize_username(u_name)
                    pass_clean = (u_pass or "").strip()
                    picked_role = str(login_role).strip().lower()
                    now_login = time.time()
                    lock_state = st.session_state["login_attempts"].get(user_clean, {"count": 0, "lock_until": 0.0})
                    is_locked = now_login < float(lock_state.get("lock_until", 0.0))
                    if is_locked:
                        remain = int(lock_state["lock_until"] - now_login)
                        st.error(f"Bu hesap geçici olarak kilitlendi. {remain} sn sonra tekrar deneyin.")

                    # 1) Boş alan kontrolü HER ZAMAN önce
                    if not user_clean or not pass_clean:
                        st.warning("Lütfen tüm alanları doldurunuz.")
                    elif picked_role not in VALID_USER_ROLES:
                        st.warning("Geçerli bir rol seçiniz.")
                    # 2) Kimlik doğrulama
                    elif is_locked:
                        pass
                    elif user_clean in st.session_state.users_db and _verify_password(
                        st.session_state.users_db[user_clean],
                        pass_clean
                    ):
                        allowed_roles = _allowed_roles_for_username(user_clean)
                        if picked_role not in allowed_roles:
                            allowed_lbl = ", ".join([role_labels.get(r, r) for r in allowed_roles])
                            st.error(f"Bu kullanıcı için izinli rol: {allowed_lbl}")
                            cur = st.session_state["login_attempts"].get(user_clean, {"count": 0, "lock_until": 0.0})
                            cur["count"] = int(cur.get("count", 0)) + 1
                            if cur["count"] >= int(st.session_state.get("max_login_attempts", 5)):
                                cur["lock_until"] = now_login + float(st.session_state.get("lock_window_sec", 300))
                                cur["count"] = 0
                            st.session_state["login_attempts"][user_clean] = cur
                            _add_audit("login_role_mismatch", f"{user_clean}:{picked_role}")
                        else:
                            _migrate_legacy_password_if_needed(user_clean, pass_clean)
                            st.session_state.logged_in = True
                            st.session_state.current_user = user_clean
                            st.session_state.profile_loaded_for = None
                            st.session_state["user_role"] = picked_role
                            _apply_session_timeout_policy(force=True)
                            _set_persistent_login(user_clean)
                            st.session_state["login_attempts"][user_clean] = {"count": 0, "lock_until": 0.0}
                            _add_audit("login_success", f"{user_clean}:{picked_role}")
                            st.toast(f"Hoş geldiniz, {user_clean}!")
                            st.rerun()
                    else:
                        cur = st.session_state["login_attempts"].get(user_clean, {"count": 0, "lock_until": 0.0})
                        cur["count"] = int(cur.get("count", 0)) + 1
                        if cur["count"] >= int(st.session_state.get("max_login_attempts", 5)):
                            cur["lock_until"] = now_login + float(st.session_state.get("lock_window_sec", 300))
                            cur["count"] = 0
                        st.session_state["login_attempts"][user_clean] = cur
                        _add_audit("login_failed", user_clean)
                        st.error("Kimlik doğrulama başarısız. Bilgilerinizi kontrol edin.")

        # -------- REGISTER --------
        with tab2:
            with st.form("register_form", clear_on_submit=True):
                role_labels = {
                    "family": "Aile",
                    "doctor": "Doktor",
                    "researcher": "Araştırmacı",
                    "admin": "Admin",
                }
                new_user = st.text_input("Yeni Kullanıcı Adı", placeholder="Örn: berfin")
                reg_role = st.selectbox(
                    "Rol",
                    options=["family", "doctor", "researcher"],
                    index=0,
                    format_func=lambda r: role_labels.get(r, r),
                )
                new_pass = st.text_input("Yeni Şifre", type="password", placeholder="••••••••")
                confirm_pass = st.text_input("Şifre Tekrar", type="password", placeholder="••••••••")
                submit_reg = st.form_submit_button("HESAP OLUŞTUR", use_container_width=True)

                if submit_reg:
                    reg_user_clean = _normalize_username(new_user)
                    reg_role_clean = str(reg_role or "").strip().lower()
                    new_pass_clean = (new_pass or "").strip()
                    confirm_clean = (confirm_pass or "").strip()
                    allowed_roles = _allowed_roles_for_username(reg_user_clean)

                    if not reg_user_clean or not new_pass_clean or not confirm_clean:
                        st.error("Alanlar boş bırakılamaz.")
                    elif reg_role_clean not in VALID_USER_ROLES:
                        st.error("Geçerli bir rol seçiniz.")
                    elif reg_role_clean == "admin":
                        st.error("Admin hesabı kayıt ekranından oluşturulamaz.")
                    elif reg_role_clean not in allowed_roles:
                        allowed_lbl = ", ".join([role_labels.get(r, r) for r in allowed_roles])
                        st.error(f"Bu kullanıcı adı için izinli kayıt rolü: {allowed_lbl}")
                    elif not re.fullmatch(r"[a-z0-9_.-]{3,32}", reg_user_clean):
                        st.error("Kullanıcı adı 3-32 karakter olmalı; yalnızca harf, rakam, _, -, . içerebilir.")
                    elif new_pass_clean != confirm_clean:
                        st.warning("Şifreler eşleşmiyor.")
                    elif not _password_policy_ok(new_pass_clean):
                        st.warning("Şifre en az 8 karakter olmalı ve harf ile rakam içermelidir.")
                    elif reg_user_clean in st.session_state.users_db:
                        st.error("Bu kullanıcı adı zaten mevcut.")
                    else:
                        # Şifreyi hashleyerek sakla (artık güvenli format)
                        hashed = _hash_password(new_pass_clean)
                        st.session_state.users_db[reg_user_clean] = hashed
                        _save_local_user_role(reg_user_clean, reg_role_clean)
                        save_local_user(reg_user_clean, hashed)
                        if not save_gsheets_user(sheet_url, reg_user_clean, hashed):
                            st.warning("Bulut kaydı başarısız; kullanıcı yalnızca yerel yedekte saklandı.")
                        _add_audit("register_success", f"{reg_user_clean}:{reg_role_clean}")
                        st.success(f"Hesap oluşturuldu: {reg_user_clean} ({role_labels.get(reg_role_clean, reg_role_clean)})")
                        st.toast("Kayıt tamamlandı.")

        st.markdown("---")
        admin_cfg_ui = _load_admin_owner_config()
        admin_owner_ui = _canonical_username(str(admin_cfg_ui.get("username", "")))
        if admin_owner_ui:
            st.info(f"Admin hesabı tanımlı: {admin_owner_ui}")
        else:
            with st.expander("Admin Kurulumu (Tek Seferlik)"):
                with st.form("admin_bootstrap_form", clear_on_submit=True):
                    a_user = st.text_input("Admin kullanıcı adı", placeholder="örn: berfin_admin")
                    a_pass = st.text_input("Admin şifre", type="password", placeholder="••••••••")
                    a_pass2 = st.text_input("Admin şifre tekrar", type="password", placeholder="••••••••")
                    a_ok = st.form_submit_button("Admin Hesabını Oluştur", use_container_width=True)
                    if a_ok:
                        au = _normalize_username(a_user)
                        p1 = (a_pass or "").strip()
                        p2 = (a_pass2 or "").strip()
                        if _admin_owner_username():
                            st.error("Admin hesabı zaten tanımlı.")
                        elif not au or not p1 or not p2:
                            st.error("Alanlar boş bırakılamaz.")
                        elif not re.fullmatch(r"[a-z0-9_.-]{3,32}", au):
                            st.error("Kullanıcı adı 3-32 karakter olmalı; yalnızca harf, rakam, _, -, . içerebilir.")
                        elif p1 != p2:
                            st.error("Şifreler eşleşmiyor.")
                        elif not _password_policy_ok(p1):
                            st.error("Şifre en az 8 karakter olmalı ve harf ile rakam içermelidir.")
                        else:
                            h = _hash_password(p1)
                            if _save_admin_owner_config(au, h):
                                save_local_user(au, h)
                                st.session_state.setdefault("users_db", {})
                                if isinstance(st.session_state.get("users_db"), dict):
                                    st.session_state["users_db"][au] = h
                                _save_local_user_role(au, "admin")
                                _add_audit("admin_owner_created", au)
                                st.success("Admin hesabı oluşturuldu. Artık admin kaydı yapılamaz.")
                                st.rerun()
                            else:
                                st.error("Admin hesabı kaydedilemedi.")

    st.markdown("""
        <br>
        <p style="text-align: center; color: #aaa; font-size: 0.7rem; margin-top: 10px;">
            DMD Guardian Pro v1.0 | Berfin Nida Öztürk tarafından
        </p>
    """, unsafe_allow_html=True)

    st.stop()

# Logged-in kullanıcılar için güvenli otomatik kalıcılık.
# save_current_session_profile içinde imza+zaman eşiği olduğu için gereksiz yazım baskılanır.
if st.session_state.get("logged_in") and st.session_state.get("current_user"):
    try:
        _sync_session_to_active_patient()
        save_current_session_profile()
    except Exception as e:
        _debug_log("auto persist failed", e)

# --- 5. ANA UYGULAMA (CORE ARCHITECTURE) ---

# Giriş kutlaması (sade geçiş)
if "first_load" not in st.session_state:
    st.toast("Sisteme giriş yapıldı.")
    st.session_state.first_load = True

# Dil ve Hafıza Ayarları (güvenli varsayılan)
if "lang" not in st.session_state or st.session_state.lang not in ("TR", "EN", "DE"):
    st.session_state.lang = "TR"

# Kullanıcı profili (kalıcı hafıza) oturuma yüklenir.
active_user = st.session_state.get("current_user")
if active_user and st.session_state.get("profile_loaded_for") != active_user:
    profile = load_user_profile(active_user)
    if isinstance(profile, dict) and profile:
        st.session_state["kilo"] = float(profile.get("kilo", st.session_state.get("kilo", 30.0)))
        st.session_state["yas"] = int(profile.get("yas", st.session_state.get("yas", 6)))
        st.session_state["nsaa_total"] = int(profile.get("nsaa_total", st.session_state.get("nsaa_total", 0)))
        st.session_state["nsaa_prev_total"] = profile.get("nsaa_prev_total")
        history = profile.get("nsaa_history", [])
        st.session_state["nsaa_history"] = history if isinstance(history, list) else []
        raw_patients = profile.get("patients", st.session_state.get("patients", {}))
        if isinstance(raw_patients, dict):
            st.session_state["patients"] = {
                str(pid): _normalize_patient_record(p if isinstance(p, dict) else {}, fallback_name=str(pid))
                for pid, p in raw_patients.items()
                if str(pid).strip()
            }
        else:
            st.session_state["patients"] = st.session_state.get("patients", {})
        st.session_state["active_patient_id"] = profile.get("active_patient_id", st.session_state.get("active_patient_id"))
        st.session_state["audits"] = profile.get("audits", st.session_state.get("audits", []))
        st.session_state["user_mode"] = profile.get("user_mode", st.session_state.get("user_mode", "Aile"))
        # --- NEW/UPDATED --- rol bilgisi yukle
        st.session_state["user_role"] = profile.get("user_role", st.session_state.get("user_role", "family"))
        st.session_state["pul_score"] = int(profile.get("pul_score", st.session_state.get("pul_score", 0)))
        st.session_state["vignos_score"] = int(profile.get("vignos_score", st.session_state.get("vignos_score", 1)))
        st.session_state["pul_history"] = profile.get("pul_history", st.session_state.get("pul_history", []))
        st.session_state["vignos_history"] = profile.get("vignos_history", st.session_state.get("vignos_history", []))
        st.session_state["medications"] = profile.get("medications", st.session_state.get("medications", []))
        st.session_state["side_effects"] = profile.get("side_effects", st.session_state.get("side_effects", []))
        st.session_state["reminders"] = profile.get("reminders", st.session_state.get("reminders", []))
        st.session_state["doctor_notes"] = profile.get("doctor_notes", st.session_state.get("doctor_notes", []))
        st.session_state["emergency_card"] = profile.get("emergency_card", st.session_state.get("emergency_card", {}))
        st.session_state["molecular_profile"] = profile.get("molecular_profile", st.session_state.get("molecular_profile", {}))
        st.session_state["care_plan"] = profile.get("care_plan", st.session_state.get("care_plan", {}))
        st.session_state["visits"] = profile.get("visits", st.session_state.get("visits", []))
        st.session_state["documents"] = profile.get("documents", st.session_state.get("documents", []))
        st.session_state["privacy_settings"] = profile.get("privacy_settings", st.session_state.get("privacy_settings", {}))
        st.session_state["notification_ack"] = profile.get("notification_ack", st.session_state.get("notification_ack", {}))
    # Profil yuklendikten sonra role gore timeout politikasini yeniden uygula.
    _apply_session_timeout_policy(force=True)
    st.session_state.profile_loaded_for = active_user

_ensure_patient_state()
_ensure_patient_visits_schema()
_sync_active_patient_to_session()
if "privacy_settings" not in st.session_state or not isinstance(st.session_state.get("privacy_settings"), dict):
    st.session_state["privacy_settings"] = {
        "consent_accepted": False,
        "consent_ts": "",
        "retention_days": 365,
        "notify_in_app": True,
        "notify_email": False,
        "notify_email_addr": "",
    }
if "notification_ack" not in st.session_state or not isinstance(st.session_state.get("notification_ack"), dict):
    st.session_state["notification_ack"] = {}

# Otomatik kalici kayit: girisli kullanicinin son durumunu her rerun'da persist et.
# save_current_session_profile zaten degisim imzasi + zaman esigi ile gereksiz yazimi engeller.
if st.session_state.get("logged_in") and st.session_state.get("current_user"):
    save_current_session_profile()

# --- DINAMIK DIL SOZLUGU (EXPANDED + SAFE) ---
LANG = {
    "TR": {
        "welcome": "Neurodegenerative Clinical Platform",
        "nav": [
            "Ana Panel",
            "Klinik Hesaplayıcı",
            "Tam Ölçekli NSAA Testi",
            "Klinik Operasyon Merkezi",
            "Klinik Takvim & Haklar",
            "Acil Durum & Kritik Bakım",
            "Sıkça Sorulan Sorular",
            "Güncel DMD Haberleri",
            "AI'ya Sor",
            "Vizyon & İmza",
        ],
        "anes_warn": "KRİTİK: Anestezi Uyarısı!",
        "ster_warn": "Steroidler asla aniden kesilmemelidir!",
        "calc_h": "Klinik Hesaplayıcı & Veri Girişi",
        "weight": "Vücut Ağırlığı (kg)",
        "age": "Mevcut Yaş",
        "mut": "Mutasyon Tipi",
        "ster_res": "**Günlük Steroid Dozaj Tahmini (Deflazacort):**",
        "nsaa_h": "Klinik Kuzey Yıldızı (NSAA) Takibi",
        "score_h": "Toplam NSAA Skoru",
        "faq_h": "Sık Sorulan Sorular & Akademik Rehber",
        "cal_h": "Klinik Takvim & Yasal Haklar",
        "emer_h": "Acil Durum & Kritik Bakım",
        "leader": "PROJE LİDERİ",
        "news_h": "Güncel DMD Haberleri",
        "news_all": "Tüm Haberleri Google News'te Gör",
        "advanced_h": "Klinik Operasyon Merkezi",
        "ai_h": "AI Destekli Soru-Cevap",
    },
    "EN": {
        "welcome": "Neurodegenerative Clinical Platform",
        "nav": [
            "Dashboard",
            "Clinical Calculator",
            "Full Scale NSAA Test",
            "Clinical Operations Hub",
            "Clinical Calendar & Rights",
            "Emergency & Critical Care",
            "FAQ & Clinical Guide",
            "Latest DMD News",
            "Ask AI",
            "Vision & Strategic Leadership",
        ],
        "anes_warn": "CRITICAL: Anesthesia Warning!",
        "ster_warn": "Steroids must never be stopped abruptly!",
        "calc_h": "Clinical Calculator & Data Entry",
        "weight": "Body Weight (kg)",
        "age": "Current Age",
        "mut": "Mutation Type",
        "ster_res": "**Daily Steroid Dosage Estimate (Deflazacort):**",
        "nsaa_h": "North Star Ambulatory Assessment (NSAA)",
        "score_h": "Total NSAA Score",
        "faq_h": "Frequently Asked Questions & Guide",
        "cal_h": "Clinical Calendar & Legal Rights",
        "emer_h": "Emergency & Critical Care",
        "leader": "PROJECT LEADER",
        "news_h": "Latest DMD News",
        "news_all": "View All on Google News",
        "advanced_h": "Clinical Operations Hub",
        "ai_h": "AI Assisted Q&A",
    },
    "DE": {
        "welcome": "Neurodegenerative Clinical Platform",
        "nav": [
            "Dashboard",
            "Klinik-Rechner",
            "Vollständiger NSAA-Test",
            "Klinisches Operationszentrum",
            "Klinikplan & Rechte",
            "Notfall & Kritische Versorgung",
            "FAQ & Leitfaden",
            "Aktuelle DMD-Nachrichten",
            "KI Fragen",
            "Vision & Strategische Führung",
        ],
        "anes_warn": "KRITISCH: Anästhesie-Warnung!",
        "ster_warn": "Steroide dürfen nicht abrupt abgesetzt werden!",
        "calc_h": "Klinik-Rechner & Dateneingabe",
        "weight": "Körpergewicht (kg)",
        "age": "Aktuelles Alter",
        "mut": "Mutationstyp",
        "ster_res": "**Tägliche Steroid-Dosis (Deflazacort):**",
        "nsaa_h": "NSAA Verlauf",
        "score_h": "NSAA Gesamtscore",
        "faq_h": "Häufige Fragen & Leitfaden",
        "cal_h": "Klinikplan & Rechtliche Ansprüche",
        "emer_h": "Notfall & Kritische Versorgung",
        "leader": "PROJEKTLEITUNG",
        "news_h": "Aktuelle DMD-Nachrichten",
        "news_all": "Alle Nachrichten bei Google News",
        "advanced_h": "Klinisches Operationszentrum",
        "ai_h": "KI-gestützte Fragen",
    },
}

# --- GLOBAL I18N PATCH (hardcoded metinleri de çevir) ---
I18N_REPLACE = {
    "EN": {
        "Mevcut Kullanıcı": "Existing User",
        "Yeni Kayıt": "New Registration",
        "Kullanıcı Adı": "Username",
        "Şifre": "Password",
        "Şifre Tekrar": "Confirm Password",
        "SİSTEME GİRİŞ YAP": "LOG IN",
        "HESAP OLUŞTUR": "CREATE ACCOUNT",
        "Lütfen tüm alanları doldurunuz.": "Please fill in all fields.",
        "Kimlik doğrulama başarısız. Bilgilerinizi kontrol edin.": "Authentication failed. Please check your credentials.",
        "Alanlar boş bırakılamaz.": "Fields cannot be empty.",
        "Şifreler eşleşmiyor.": "Passwords do not match.",
        "Bu kullanıcı adı zaten mevcut.": "This username already exists.",
        "Hesap oluşturuldu": "Account created",
        "Güvenli Çıkış": "Secure Logout",
        "Kullanım Modu": "Usage Mode",
        "Aile": "Family",
        "Doktor": "Doctor",
        "Sistem Durumu": "System Status",
        "Veri Gizliliği": "Data Privacy",
        "Klinik Rehber": "Clinical Guide",
        "Mevcut Klinik Profil Özeti": "Current Clinical Profile Summary",
        "Vücut Ağırlığı": "Body Weight",
        "Mevcut Yaş": "Current Age",
        "Son NSAA Skoru": "Latest NSAA Score",
        "Fonksiyonel Seyir İzleme": "Functional Trend Monitoring",
        "Akıllı Notlar": "Smart Notes",
        "Randevu Hatırlatıcı": "Appointment Reminder",
        "Toplam NSAA Skoru": "Total NSAA Score",
        "Puanlama Kriterleri Rehberi": "Scoring Criteria Guide",
        "Tümünü 2 Yap": "Set All to 2",
        "Tümünü 1 Yap": "Set All to 1",
        "Sıfırla": "Reset",
        "Alt Kategori Skorları": "Subcategory Scores",
        "Öncelikli rehabilitasyon odağı": "Priority rehabilitation focus",
        "NSAA Raporunu Kaydet": "Save NSAA Report",
        "Son NSAA Kayıtları": "Recent NSAA Records",
        "Gelişmiş Modüller": "Advanced Modules",
        "Hasta Yönetimi": "Patient Management",
        "Trend & Rapor": "Trends & Report",
        "Hatırlatıcı": "Reminder",
        "İlaç & Yan Etki": "Medication & Side Effects",
        "Kalite & Audit": "Quality & Audit",
        "Yedekleme": "Backup",
        "Aktif Hasta": "Active Patient",
        "Yeni Hasta Adı": "New Patient Name",
        "Hasta Ekle": "Add Patient",
        "Aktif Hastayı Sil": "Delete Active Patient",
        "Trendler ve Klinik Rapor": "Trends and Clinical Report",
        "Ek Skorlar": "Additional Scores",
        "Ek Skorları Kaydet": "Save Additional Scores",
        "Klinik Raporu İndir": "Download Clinical Report",
        "Randevu/Hatırlatıcı": "Appointments/Reminders",
        "Tarih": "Date",
        "Başlık": "Title",
        "Not": "Note",
        "Hatırlatıcı Ekle": "Add Reminder",
        "Takvim Dosyası İndir": "Download Calendar File",
        "İlaç ve Yan Etki Günlüğü": "Medication and Side Effect Log",
        "İlaç Adı": "Medication Name",
        "Doz": "Dose",
        "İlaç Kaydı Ekle": "Add Medication Record",
        "Yan Etki": "Side Effect",
        "Şiddet": "Severity",
        "Detay": "Detail",
        "Yan Etki Ekle": "Add Side Effect",
        "Veri Kalite Kontrolü ve Audit": "Data Quality Control and Audit",
        "Yedekleme / Geri Yükleme": "Backup / Restore",
        "Tam Yedek İndir": "Download Full Backup",
        "JSON Yedek Yükle": "Upload JSON Backup",
        "Yedeği Geri Yükle": "Restore Backup",
        "Güncel DMD Haberleri": "Latest DMD News",
        "Haberleri Yenile": "Refresh News",
        "Haber içinde ara": "Search in news",
        "Kaynak filtresi": "Source filter",
        "Tümü": "All",
        "AI Destekli Soru-Cevap": "AI Assisted Q&A",
        "Hasta bağlamını soruya ekle": "Include patient context in question",
        "Sorunuz": "Your question",
        "Sohbet Geçmişini Temizle": "Clear Chat History",
        "Son Sorular": "Recent Questions",
        "Kayıt yolu": "Storage path",
        "Bulut senkron": "Cloud sync",
        "Kaynak": "Source",
        "Tarih:": "Date:",
        "Klinik Operasyon Merkezi": "Clinical Operations Hub",
        "Operasyon Özeti": "Operations Overview",
        "Operasyon Özeti ve Hızlı Aksiyonlar": "Operations Overview and Quick Actions",
        "Toplam Hasta": "Total Patients",
        "7 Gün İçinde Randevu": "Appointments in 7 Days",
        "Son Ziyaret": "Last Visit",
        "Anlık Ziyaret Kaydı Oluştur": "Create Quick Visit Snapshot",
        "Profili Hemen Kaydet": "Save Profile Now",
        "Ziyaret CSV İndir": "Download Visits CSV",
        "Kritik kalite alarmı bulunmuyor.": "No critical quality alerts.",
    },
    "DE": {
        "Mevcut Kullanıcı": "Bestehender Benutzer",
        "Yeni Kayıt": "Neue Registrierung",
        "Kullanıcı Adı": "Benutzername",
        "Şifre": "Passwort",
        "Şifre Tekrar": "Passwort bestätigen",
        "SİSTEME GİRİŞ YAP": "ANMELDEN",
        "HESAP OLUŞTUR": "KONTO ERSTELLEN",
        "Lütfen tüm alanları doldurunuz.": "Bitte füllen Sie alle Felder aus.",
        "Kimlik doğrulama başarısız. Bilgilerinizi kontrol edin.": "Authentifizierung fehlgeschlagen. Bitte Daten prüfen.",
        "Alanlar boş bırakılamaz.": "Felder dürfen nicht leer sein.",
        "Şifreler eşleşmiyor.": "Passwörter stimmen nicht überein.",
        "Bu kullanıcı adı zaten mevcut.": "Dieser Benutzername existiert bereits.",
        "Hesap oluşturuldu": "Konto erstellt",
        "Güvenli Çıkış": "Sicher abmelden",
        "Kullanım Modu": "Nutzungsmodus",
        "Aile": "Familie",
        "Doktor": "Arzt",
        "Gelişmiş Modüller": "Erweiterte Module",
        "Güncel DMD Haberleri": "Aktuelle DMD-Nachrichten",
        "AI Destekli Soru-Cevap": "KI-gestützte Fragen",
        "Hasta bağlamını soruya ekle": "Patientenkontext hinzufügen",
        "Sorunuz": "Ihre Frage",
        "Klinik Operasyon Merkezi": "Klinisches Operationszentrum",
        "Operasyon Özeti": "Operationsübersicht",
        "Operasyon Özeti ve Hızlı Aksiyonlar": "Operationsübersicht und Schnellaktionen",
        "Toplam Hasta": "Patienten gesamt",
        "7 Gün İçinde Randevu": "Termine in 7 Tagen",
        "Son Ziyaret": "Letzter Besuch",
        "Anlık Ziyaret Kaydı Oluştur": "Sofortigen Besuchseintrag erstellen",
        "Profili Hemen Kaydet": "Profil sofort speichern",
        "Ziyaret CSV İndir": "Besuche als CSV herunterladen",
        "Kritik kalite alarmı bulunmuyor.": "Keine kritischen Qualitätswarnungen.",
    },
}


EXTRA_I18N_REPLACE = {
    "EN": {
        "Randevu Takvimi": "Appointment Calendar",
        "Devlet Hakları": "Government Rights",
        "Başvuru Rehberi": "Application Guide",
        "Gizlilik / KVKK": "Privacy / Compliance",
        "Gizlilik ayarlarini kaydet": "Save privacy settings",
        "Bildirimler": "Notifications",
        "Dokuman Metadata Yukleme": "Document Metadata Upload",
        "Dokumani kaydet": "Save Document",
        "Rapor/PDF/Gorsel yukle (metadata kaydi)": "Upload report/PDF/image (metadata only)",
        "Verimi disa aktar (JSON)": "Export my data (JSON)",
        "Profil verimi sil (geri alinamaz)": "Delete my profile data (irreversible)",
        "Araştırma Özeti (Anonim)": "Research Summary (Anonymous)",
        "Solunum Desteği": "Respiratory Support",
        "Kırık ve Travma": "Fracture and Trauma",
        "Acil Durum Kartı": "Emergency Card",
        "Acil Durum Kartını İndir (TXT)": "Download Emergency Card (TXT)",
        "Periyodik Kontrol Listesi": "Periodic Follow-up Checklist",
        "Yasal Haklar ve Sosyal Destekler": "Legal Rights and Social Support",
        "Rapor ve Başvuru Adımları": "Report and Application Steps",
        "Bu ölçümü ziyarete kaydet": "Save this measurement to visit",
    },
    "DE": {
        "Randevu Takvimi": "Terminkalender",
        "Devlet Hakları": "Staatliche Rechte",
        "Başvuru Rehberi": "Antragsleitfaden",
        "Gizlilik / KVKK": "Datenschutz / Compliance",
        "Gizlilik ayarlarini kaydet": "Datenschutzeinstellungen speichern",
        "Bildirimler": "Benachrichtigungen",
        "Dokuman Metadata Yukleme": "Dokument-Metadaten-Upload",
        "Dokumani kaydet": "Dokument speichern",
        "Rapor/PDF/Gorsel yukle (metadata kaydi)": "Bericht/PDF/Bild hochladen (nur Metadaten)",
        "Verimi disa aktar (JSON)": "Meine Daten exportieren (JSON)",
        "Profil verimi sil (geri alinamaz)": "Meine Profildaten löschen (nicht rückgängig)",
        "Araştırma Özeti (Anonim)": "Forschungsübersicht (Anonym)",
        "Solunum Desteği": "Atemunterstützung",
        "Kırık ve Travma": "Fraktur und Trauma",
        "Acil Durum Kartı": "Notfallkarte",
        "Acil Durum Kartını İndir (TXT)": "Notfallkarte herunterladen (TXT)",
        "Periyodik Kontrol Listesi": "Periodische Kontrollliste",
        "Yasal Haklar ve Sosyal Destekler": "Rechtliche Ansprüche und soziale Unterstützung",
        "Rapor ve Başvuru Adımları": "Berichts- und Antragsschritte",
        "Bu ölçümü ziyarete kaydet": "Diese Messung zum Besuch speichern",
    },
}

_I18N_NORMALIZED_MAPS: dict[str, dict[str, str]] = {}


def _normalize_mojibake_text(text: str) -> str:
    if not isinstance(text, str) or not text:
        return text
    out = text
    replacements = (
        ("ı", "ı"),
        ("İ", "İ"),
        ("ş", "ş"),
        ("ş", "ş"),
        ("Ş", "Ş"),
        ("ğ", "ğ"),
        ("Ğ", "Ğ"),
        ("ü", "ü"),
        ("Ü", "Ü"),
        ("ö", "ö"),
        ("Ö", "Ö"),
        ("ç", "ç"),
        ("Ç", "Ç"),
        ("'", "'"),
        ("-", "-"),
        ("-", "-"),
        ("©", "©"),
        ("", ""),
        ("🧠", "🧠"),
        ("✅", "✅"),
        ("⚠️", "⚠️"),
        ("❤️", "❤️"),
        ("⬇️", "⬇️"),
        ("⬆️", "⬆️"),
        ("➡️", "➡️"),
        ("🫁", "🫁"),
        ("🙌", "🙌"),
        ("🌍", "🌍"),
        ("⚡", "⚡"),
        ("🧬", "🧬"),
        ("🚀", "🚀"),
        ("ℹ", "ℹ"),
    )
    for src, tgt in replacements:
        out = out.replace(src, tgt)
    return out


def _get_i18n_map(lang: str) -> dict[str, str]:
    lang = str(lang or "TR")
    if lang in _I18N_NORMALIZED_MAPS:
        return _I18N_NORMALIZED_MAPS[lang]
    raw = {}
    raw.update(I18N_REPLACE.get(lang, {}))
    raw.update(EXTRA_I18N_REPLACE.get(lang, {}))
    norm = {}
    for k, v in raw.items():
        kk = _normalize_mojibake_text(str(k))
        vv = _normalize_mojibake_text(str(v))
        if kk:
            norm[kk] = vv
    _I18N_NORMALIZED_MAPS[lang] = norm
    return norm


def _i18n_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    lang = st.session_state.get("lang", "TR")
    out = _normalize_mojibake_text(text)
    if lang == "TR":
        return out
    mapping = _get_i18n_map(lang)
    # Uzun ifadeleri önce değiştir
    for src in sorted(mapping.keys(), key=len, reverse=True):
        out = out.replace(src, mapping[src])
    return out


def _i18n_seq(seq):
    if isinstance(seq, (list, tuple)):
        return type(seq)(_i18n_text(x) if isinstance(x, str) else x for x in seq)
    return seq


def t(text: str) -> str:
    return _i18n_text(text)


def _i18n_patch_enabled() -> bool:
    try:
        sec_v = str(st.secrets.get("i18n_patch_enabled", "")).strip().lower()
    except Exception:
        sec_v = ""
    if sec_v in {"1", "true", "yes", "on"}:
        return True
    if sec_v in {"0", "false", "no", "off"}:
        return False
    env_v = str(os.getenv("DMD_ENABLE_ST_I18N_PATCH", "1")).strip().lower()
    return env_v in {"1", "true", "yes", "on"}


def _patch_streamlit_i18n() -> None:
    global _I18N_PATCHED
    if _I18N_PATCHED:
        return

    def _save(name):
        if name not in _ORIG_ST_FUNCS:
            _ORIG_ST_FUNCS[name] = getattr(st, name)

    # text-first wrappers
    for fn_name in ["title", "header", "subheader", "caption", "text", "info", "warning", "error", "success", "write", "markdown"]:
        _save(fn_name)
        orig = _ORIG_ST_FUNCS[fn_name]

        def make_wrap(o):
            def w(body=None, *args, **kwargs):
                if isinstance(body, str):
                    body = _i18n_text(body)
                return o(body, *args, **kwargs)
            return w

        setattr(st, fn_name, make_wrap(orig))

    # label wrappers
    for fn_name in [
        "button",
        "checkbox",
        "radio",
        "selectbox",
        "text_input",
        "text_area",
        "form_submit_button",
        "download_button",
        "link_button",
        "date_input",
        "metric",
        "expander",
        "slider",
        "multiselect",
        "number_input",
        "file_uploader",
        "select_slider",
        "segmented_control",
        "toggle",
    ]:
        if hasattr(st, fn_name):
            _save(fn_name)
            orig = _ORIG_ST_FUNCS[fn_name]

            def make_wrap_label(o, name):
                def w(*args, **kwargs):
                    args = list(args)
                    if name == "metric":
                        if "label" in kwargs and isinstance(kwargs["label"], str):
                            kwargs["label"] = _i18n_text(kwargs["label"])
                        elif args and isinstance(args[0], str):
                            args[0] = _i18n_text(args[0])
                    else:
                        if "label" in kwargs and isinstance(kwargs["label"], str):
                            kwargs["label"] = _i18n_text(kwargs["label"])
                        elif args and isinstance(args[0], str):
                            args[0] = _i18n_text(args[0])
                    if "options" in kwargs:
                        kwargs["options"] = _i18n_seq(kwargs["options"])
                    return o(*args, **kwargs)
                return w

            setattr(st, fn_name, make_wrap_label(orig, fn_name))

    # tabs wrapper
    if hasattr(st, "tabs"):
        _save("tabs")
        orig_tabs = _ORIG_ST_FUNCS["tabs"]

        def tabs_wrap(tabs, *args, **kwargs):
            return orig_tabs(_i18n_seq(tabs), *args, **kwargs)

        st.tabs = tabs_wrap

    _I18N_PATCHED = True


# patch once; language her çağrıda dinamik çevrilir
if _i18n_patch_enabled():
    try:
        _patch_streamlit_i18n()
    except Exception as e:
        _debug_log("streamlit i18n patch failed", e)

# Aktif dil sözlüğü
D = LANG.get(st.session_state.lang, LANG["TR"])

# Sayfa indeks sabitleri (LANG["nav"] ile birebir hizali)
PAGE_DASHBOARD = 0
PAGE_CALCULATOR = 1
PAGE_NSAA = 2
PAGE_OPS = 3
PAGE_CALENDAR = 4
PAGE_EMERGENCY = 5
PAGE_FAQ = 6
PAGE_NEWS = 7
PAGE_AI = 8
PAGE_VISION = 9

# --- NEW/UPDATED --- Rol bazli sayfa erisim kurallari
ROLE_PAGE_INDEXES = {
    "family": [PAGE_DASHBOARD, PAGE_EMERGENCY, PAGE_CALENDAR, PAGE_NEWS, PAGE_AI],
    "doctor": [PAGE_DASHBOARD, PAGE_CALCULATOR, PAGE_NSAA, PAGE_EMERGENCY, PAGE_OPS, PAGE_AI],
    "researcher": [PAGE_DASHBOARD, PAGE_OPS],
    "admin": [
        PAGE_DASHBOARD, PAGE_CALCULATOR, PAGE_NSAA, PAGE_OPS, PAGE_CALENDAR,
        PAGE_EMERGENCY, PAGE_FAQ, PAGE_NEWS, PAGE_AI, PAGE_VISION
    ],
}


def _role_map_from_config() -> dict[str, str]:
    out = _role_map_effective()
    return {u: r for u, r in out.items() if r in ROLE_PAGE_INDEXES}


def _allowed_roles_for_current_user() -> list[str]:
    username = _canonical_username(st.session_state.get("current_user", ""))
    if not username:
        return ["family"]
    out = [r for r in _allowed_roles_for_username(username) if r in ROLE_PAGE_INDEXES]
    return out or ["family"]


def _enforce_effective_role() -> str:
    allowed = _allowed_roles_for_current_user()
    cur = str(st.session_state.get("user_role", "family")).strip().lower()
    if cur not in allowed:
        cur = allowed[0]
        st.session_state["user_role"] = cur
    st.session_state["user_mode"] = _role_mode_text(cur)
    return cur


def _role_page_labels(d: dict, role: str) -> list[str]:
    idxs = ROLE_PAGE_INDEXES.get(role, ROLE_PAGE_INDEXES["family"])
    nav = d.get("nav", [])
    return [nav[i] for i in idxs if 0 <= i < len(nav)]


def _role_mode_text(role: str) -> str:
    return "Doktor" if role in {"doctor", "admin"} else "Aile"


def _can_manage_all_patients(role: str) -> bool:
    return role in {"doctor", "admin"}


def _can_export_personal_data(role: str) -> bool:
    return role in {"family", "doctor", "admin"}


def _can_export_system_backup(role: str) -> bool:
    return role in {"doctor", "admin"}


def _can_export_research_data(role: str) -> bool:
    return role in {"researcher", "doctor", "admin"}


def _research_salt() -> str:
    try:
        salt = str(st.secrets.get("research_salt", "")).strip()
    except Exception:
        salt = ""
    return salt


def _build_research_export_data() -> list[dict]:
    rows = []
    salt = _research_salt()
    if not salt:
        return rows
    patients = st.session_state.get("patients", {})
    for pid, p in patients.items():
        p = p if isinstance(p, dict) else {}
        anon_id = hashlib.sha256(f"{pid}{salt}".encode("utf-8")).hexdigest()[:16]
        visits = p.get("visits", [])
        visits = visits if isinstance(visits, list) else []
        nv = [_normalize_visit_record(v) for v in visits]
        meds = p.get("medications", [])
        reminders = p.get("reminders", [])
        meds_count = len(meds) if isinstance(meds, list) else 0
        reminders_count = len(reminders) if isinstance(reminders, list) else 0
        if not nv:
            rows.append(
                {
                    "anon_id": anon_id,
                    "date": "",
                    "age": None,
                    "weight": None,
                    "nsaa": None,
                    "pul": None,
                    "vignos": None,
                    "ef": None,
                    "fvc": None,
                    "notes": "",
                    "has_notes": False,
                    "meds_count": meds_count,
                    "reminders_count": reminders_count,
                }
            )
        for v in nv:
            rows.append(
                {
                    "anon_id": anon_id,
                    "date": v.get("date"),
                    "age": v.get("age"),
                    "weight": v.get("weight"),
                    "nsaa": v.get("nsaa"),
                    "pul": v.get("pul"),
                    "vignos": v.get("vignos"),
                    "ef": v.get("ef"),
                    "fvc": v.get("fvc"),
                    "notes": "",
                    "has_notes": bool(str(v.get("notes", "")).strip()),
                    "meds_count": meds_count,
                    "reminders_count": reminders_count,
                }
            )
    return rows


def _clinical_alerts() -> list[dict]:
    alerts: list[dict] = []
    visits = sorted([_normalize_visit_record(v) for v in st.session_state.get("visits", [])], key=lambda x: x.get("date", ""))
    current_nsaa = int(st.session_state.get("nsaa_total", 0) or 0)
    mp = st.session_state.get("molecular_profile", {}) or {}
    ef = mp.get("ef")
    fvc = mp.get("fvc")
    falls = int((st.session_state.get("care_plan", {}) or {}).get("falls_month", 0) or 0)

    if isinstance(ef, (int, float)) and ef < 45:
        alerts.append({"level": "critical", "text": f"Kardiyak EF düşük ({ef}%). Kardiyoloji değerlendirmesi hızlandırılmalı."})
    elif isinstance(ef, (int, float)) and ef < 55:
        alerts.append({"level": "warning", "text": f"Kardiyak EF sınırda ({ef}%). Yakın izlem önerilir."})

    if isinstance(fvc, (int, float)) and fvc < 50:
        alerts.append({"level": "critical", "text": f"FVC düşük ({fvc}%). Solunum desteği/NIV değerlendirmesi gerekli olabilir."})
    elif isinstance(fvc, (int, float)) and fvc < 70:
        alerts.append({"level": "warning", "text": f"FVC orta düzeyde azalmış ({fvc}%). Solunum izlemi sıklaştırılmalı."})

    if falls >= 3:
        alerts.append({"level": "warning", "text": f"Son 1 ay düşme sayısı yüksek ({falls}). Ev içi güvenlik planı gözden geçirilmeli."})

    if len(visits) >= 2:
        prev = visits[-2].get("nsaa")
        last = visits[-1].get("nsaa")
        if isinstance(prev, int) and isinstance(last, int):
            delta = last - prev
            if delta <= -3:
                alerts.append({"level": "critical", "text": f"Son ziyarette NSAA düşüşü belirgin ({delta})."})
            elif delta <= -1:
                alerts.append({"level": "warning", "text": f"NSAA düşüş eğilimi var ({delta})."})
    elif current_nsaa <= 10:
        alerts.append({"level": "warning", "text": f"NSAA düşük seviyede ({current_nsaa}/34). Multidisipliner takip önerilir."})
    return alerts


def _visit_delta(v_old: dict, v_new: dict) -> list[dict]:
    out = []
    fields = [("age", "Yaş"), ("weight", "Kilo"), ("nsaa", "NSAA"), ("pul", "PUL"), ("vignos", "Vignos"), ("ef", "EF"), ("fvc", "FVC")]
    for key, label in fields:
        a = v_old.get(key)
        b = v_new.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            out.append({"metric": label, "old": a, "new": b, "delta": round(float(b) - float(a), 2)})
        else:
            out.append({"metric": label, "old": a, "new": b, "delta": None})
    return out


def _store_uploaded_document(uploaded_file, patient_id: str) -> dict | None:
    if uploaded_file is None:
        return None
    try:
        file_size = int(getattr(uploaded_file, "size", 0) or 0)
        if file_size > MAX_UPLOAD_BYTES:
            return None
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = _safe_filename(getattr(uploaded_file, "name", "document.bin"))
        safe_patient_id = _safe_filename(str(patient_id or "hasta"))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_name = f"{safe_patient_id}_{ts}_{safe_name}"
        path = UPLOADS_DIR / full_name
        with path.open("wb") as f:
            f.write(uploaded_file.getbuffer())
        return {
            "name": safe_name,
            "stored_name": full_name,
            "path": str(path),
            "size_bytes": int(path.stat().st_size),
            "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception:
        return None


def _ensure_privacy_state() -> None:
    if "privacy_settings" not in st.session_state or not isinstance(st.session_state.get("privacy_settings"), dict):
        st.session_state["privacy_settings"] = {
            "consent_accepted": False,
            "consent_ts": "",
            "retention_days": 365,
            "notify_in_app": True,
            "notify_email": False,
            "notify_email_addr": "",
        }
    if "notification_ack" not in st.session_state or not isinstance(st.session_state.get("notification_ack"), dict):
        st.session_state["notification_ack"] = {}


def _build_notifications(window_days: int = 3) -> list[dict]:
    _ensure_privacy_state()
    out = []
    today = datetime.now().date()
    for r in st.session_state.get("reminders", []):
        try:
            d = datetime.strptime(str(r.get("date", "")), "%Y-%m-%d").date()
            diff = (d - today).days
            if 0 <= diff <= window_days:
                key = f"rem::{r.get('date','')}::{r.get('title','')}"
                out.append({"id": key, "kind": "Randevu", "title": str(r.get("title", "Randevu")), "date": str(r.get("date", "")), "days": diff})
        except Exception:
            continue
    for m in st.session_state.get("medications", []):
        try:
            d = datetime.strptime(str(m.get("date", "")), "%Y-%m-%d").date()
            diff = (d - today).days
            if 0 <= diff <= window_days:
                key = f"med::{m.get('date','')}::{m.get('name','')}"
                out.append({"id": key, "kind": "Ilac", "title": str(m.get("name", "Ilac")), "date": str(m.get("date", "")), "days": diff})
        except Exception:
            continue
    return sorted(out, key=lambda x: (x.get("days", 999), x.get("date", "")))


def _apply_retention_policy() -> int:
    _ensure_privacy_state()
    days = int(st.session_state.get("privacy_settings", {}).get("retention_days", 365) or 365)
    cutoff = datetime.now() - timedelta(days=max(30, days))
    removed = 0

    audits = st.session_state.get("audits", [])
    kept_audits = []
    for a in audits if isinstance(audits, list) else []:
        ts = str((a or {}).get("time", ""))
        try:
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            if dt >= cutoff:
                kept_audits.append(a)
            else:
                removed += 1
        except Exception:
            kept_audits.append(a)
    st.session_state["audits"] = kept_audits
    return removed


def _export_current_user_profile_bytes() -> bytes:
    username = _canonical_username(st.session_state.get("current_user", ""))
    payload = {
        "username": username,
        "exported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "profile": load_user_profile(username) if username else {},
        "sync_queue_pending": int(st.session_state.get("_sync_queue_pending", 0)),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _delete_current_user_profile_data() -> bool:
    username = _canonical_username(st.session_state.get("current_user", ""))
    if not username:
        return False
    ok_local = False
    conn = _db_connect()
    if conn is not None:
        try:
            conn.execute("DELETE FROM profiles WHERE username = ?", (username,))
            conn.commit()
            ok_local = True
        except Exception:
            ok_local = False
        finally:
            conn.close()
    delete_gsheets_profile(sheet_url, username, queue_on_fail=True)
    st.session_state["patients"] = {}
    st.session_state["active_patient_id"] = None
    st.session_state["audits"] = []
    st.session_state["reminders"] = []
    st.session_state["medications"] = []
    st.session_state["side_effects"] = []
    st.session_state["doctor_notes"] = []
    st.session_state["visits"] = []
    st.session_state["documents"] = []
    return ok_local

# --- SIDEBAR TASARIMI (UI/UX) ---   TEK SIDEBAR BLOĞU + AÇIK TEMA
with st.sidebar:
    # Dil Değiştirme
    lang_labels = {
        "TR": "Turkce",
        "EN": "English",
        "DE": "Deutsch",
    }
    lang_options = ["TR", "EN", "DE"]
    cur_lang = st.session_state.lang if st.session_state.lang in lang_options else "TR"
    picked_lang = st.selectbox(
        "Dil / Language / Sprache",
        options=lang_options,
        index=lang_options.index(cur_lang),
        format_func=lambda x: f"{x} - {lang_labels.get(x, x)}",
    )
    if picked_lang != st.session_state.lang:
        st.session_state.lang = picked_lang
        st.rerun()

    st.markdown("---")
    st.title("NIZEN")

    # NIZEN Brand Card (light-only)
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
            padding: 18px;
            border-radius: 15px;
            border: 1px solid #bae6fd;
            margin-bottom: 18px;">
            <p style="
                margin:0;
                font-size: 0.72rem;
                color: #0369a1;
                font-weight: 800;
                letter-spacing: 1.4px;
                text-transform: uppercase;">
                NIZEN
            </p>
            <b style="
                color: #0f172a;
                font-size: 1.05rem;
                font-family: 'Poppins', sans-serif;">
                Neurodegenerative Clinical Platform
            </b>
            <p style="margin:7px 0 0 0; font-size:0.8rem; color:#334155;">
                support@nizen.ai
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- NEW/UPDATED --- Rol secimi + aktif hasta secimi + role gore menu
    role_options = ["family", "doctor", "researcher", "admin"]
    role_labels = {
        "family": "Family",
        "doctor": "Doctor",
        "researcher": "Researcher",
        "admin": "Admin",
    }
    _enforce_effective_role()
    allowed_roles = _allowed_roles_for_current_user()
    role_options = [r for r in role_options if r in allowed_roles]
    cur_role = st.session_state.get("user_role", "family")
    if cur_role not in role_options:
        cur_role = role_options[0] if role_options else "family"
    picked_role = st.selectbox(
        "User Role",
        options=role_options,
        index=role_options.index(cur_role),
        format_func=lambda r: role_labels.get(r, r),
        disabled=len(role_options) <= 1,
    )
    if picked_role != st.session_state.get("user_role"):
        _sync_session_to_active_patient()
        save_current_session_profile()
        st.session_state["user_role"] = picked_role
        st.session_state["user_mode"] = _role_mode_text(picked_role)
        _apply_session_timeout_policy(force=True)
        _add_audit("role_change", picked_role)
        st.rerun()
    st.session_state["user_mode"] = _role_mode_text(st.session_state.get("user_role", "family"))

    _ensure_patient_state()
    _ensure_patient_visits_schema()
    patients_sidebar = st.session_state.get("patients", {})
    active_pid = st.session_state.get("active_patient_id")
    if not _can_manage_all_patients(st.session_state.get("user_role", "family")):
        patient_ids = [active_pid] if active_pid in patients_sidebar else list(patients_sidebar.keys())[:1]
    else:
        patient_ids = list(patients_sidebar.keys())
    if patient_ids:
        if st.session_state.get("active_patient_id") not in patient_ids:
            st.session_state["active_patient_id"] = patient_ids[0]
        selected_pid = st.selectbox(
            "Aktif Hasta",
            options=patient_ids,
            index=patient_ids.index(st.session_state.get("active_patient_id")),
            format_func=lambda pid: f"{patients_sidebar.get(pid, {}).get('name', pid)} ({pid})",
        )
        if selected_pid != st.session_state.get("active_patient_id"):
            _sync_session_to_active_patient()
            save_current_session_profile()
            st.session_state["active_patient_id"] = selected_pid
            _sync_active_patient_to_session()
            _add_audit("patient_switch_sidebar", selected_pid)
            st.rerun()

    visible_pages = _role_page_labels(D, st.session_state.get("user_role", "family"))
    default_page = visible_pages[0] if visible_pages else D["nav"][0]
    page_options = visible_pages if visible_pages else [default_page]
    prev_page = st.session_state.get("selected_page", default_page)
    if prev_page not in page_options:
        prev_page = default_page
    page = st.radio(
        "MENÜ",
        page_options,
        index=page_options.index(prev_page),
        key="sidebar_menu_radio",
    )
    st.session_state["selected_page"] = page

    st.markdown("---")
    if st.button("Güvenli Çıkış", use_container_width=True):
        _add_audit("logout", st.session_state.get("current_user", ""))
        save_current_session_profile()
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.session_state.profile_loaded_for = None
        _clear_persistent_login()
        st.rerun()

    st.markdown("---")
    st.error(f"**{D['anes_warn']}**")
    st.warning(f"{D['ster_warn']}")

    st.markdown(
        """
        <div style="text-align:center; opacity:0.65; font-size:0.74rem; margin-top: 18px; color:#64748b;">
            © 2026 NIZEN. All rights reserved.
        </div>
        """,
        unsafe_allow_html=True
    )
    now_ts = time.time()
    if ("_cloud_status" not in st.session_state) or (now_ts - float(st.session_state.get("_cloud_status_ts", 0)) > 60):
        is_cloud_ok = _check_cloud_sync_health(force=False)
        st.session_state["_cloud_status"] = "Active" if is_cloud_ok else "Cloud sync disabled; using local backup"
        st.session_state["_cloud_status_ts"] = now_ts
    cloud_status = st.session_state["_cloud_status"]
    if "disabled" in str(cloud_status).lower():
        st.info("Cloud sync disabled; using local backup.")
    if st.session_state.get("_cloud_unavailable_read", False) and not st.session_state.get("_warned_cloud", False):
        st.session_state["_warned_cloud"] = True
        st.warning(str(st.session_state.get("_cloud_error_msg", "Cloud sync unavailable; using local backup. Check gsheets connection / worksheet names / permissions.")))
    if (not _persistent_login_enabled()) and (not st.session_state.get("_warned_persistent_login_disabled", False)):
        st.session_state["_warned_persistent_login_disabled"] = True
        st.info("Persistent login is disabled. Enable it below (Admin) and ensure auth_secret is configured with strong entropy.")
    st.caption(f"Kayıt yolu: {LOCAL_DB}")
    st.caption(f"Bulut senkron: {cloud_status} | Sheet sekmeleri: {USERS_WORKSHEET}, {PROFILES_WORKSHEET}")
    st.caption(f"Sync kuyruk bekleyen: {int(st.session_state.get('_sync_queue_pending', 0))}")

    notices = _build_notifications(window_days=3)
    unread = [n for n in notices if not st.session_state.get("notification_ack", {}).get(n["id"], False)]
    with st.expander(f"Bildirimler ({len(unread)})"):
        if not notices:
            st.caption("Yakın tarihli bildirim yok.")
        else:
            for n in notices[:8]:
                mark = "okundu" if st.session_state.get("notification_ack", {}).get(n["id"], False) else "yeni"
                st.write(f"- [{mark}] {n['kind']} | {n['title']} | {n['date']} ({n['days']} gün)")
            if st.button("Tumunu okundu isle", key="mark_all_notices"):
                ack = st.session_state.get("notification_ack", {})
                for n in notices:
                    ack[n["id"]] = True
                st.session_state["notification_ack"] = ack
                save_current_session_profile()
                st.rerun()
            ps_local = st.session_state.get("privacy_settings", {})
            if bool(ps_local.get("notify_email", False)) and str(ps_local.get("notify_email_addr", "")).strip():
                email_text = "\n".join([f"{n['kind']} | {n['title']} | {n['date']} | {n['days']} gun" for n in notices])
                st.download_button(
                    "E-posta ozeti indir (beta)",
                    data=email_text.encode("utf-8"),
                    file_name="dmd_notification_email.txt",
                    mime="text/plain",
                )

    with st.expander("Gizlilik / KVKK"):
        role_now = st.session_state.get("user_role", "family")
        ps = st.session_state.get("privacy_settings", {})
        consent_now = st.checkbox(
            "Aydinlatma metnini okudum ve veri isleme onayi veriyorum",
            value=bool(ps.get("consent_accepted", False)),
            key="privacy_consent_toggle",
        )
        retention = st.slider(
            "Veri saklama suresi (gun)",
            min_value=90,
            max_value=1825,
            value=int(ps.get("retention_days", 365) or 365),
            step=30,
            key="retention_days_slider",
        )
        notify_in_app = st.checkbox("Uygulama ici bildirim", value=bool(ps.get("notify_in_app", True)), key="notify_in_app_toggle")
        notify_email = st.checkbox("E-posta bildirimi (beta)", value=bool(ps.get("notify_email", False)), key="notify_email_toggle")
        notify_email_addr = st.text_input("Bildirim e-postasi", value=str(ps.get("notify_email_addr", "")), key="notify_email_addr_input")

        if st.button("Gizlilik ayarlarini kaydet", key="save_privacy_btn", use_container_width=True):
            st.session_state["privacy_settings"] = {
                "consent_accepted": bool(consent_now),
                "consent_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S") if consent_now else "",
                "retention_days": int(retention),
                "notify_in_app": bool(notify_in_app),
                "notify_email": bool(notify_email),
                "notify_email_addr": str(notify_email_addr).strip(),
            }
            removed = _apply_retention_policy()
            _add_audit("privacy_settings_saved", f"retention={retention}, removed={removed}")
            save_current_session_profile()
            st.success(f"Gizlilik ayarlari kaydedildi. Temizlenen kayit: {removed}")

        st.markdown("---")
        st.caption("Kalici giris (query token): HMAC-imzali token URL'de tasinir; sure dolunca otomatik iptal olur.")
        db_persist_enabled = _persistent_login_query_enabled()
        persist_toggle = st.checkbox(
            "Enable persistent login",
            value=bool(db_persist_enabled),
            disabled=(role_now != "admin"),
            key="persistent_login_toggle_ui",
        )
        if role_now != "admin":
            st.caption("Bu ayari sadece admin degistirebilir.")
        else:
            ttl_days = st.slider(
                "Persistent login suresi (gun)",
                min_value=1,
                max_value=90,
                value=max(1, int(_persistent_login_ttl_sec() // 86400)),
                key="persistent_login_ttl_days_ui",
            )
            if st.button("Kalici giris ayarini kaydet", use_container_width=True, key="save_persistent_login_btn"):
                _db_set_kv("persistent_login_enabled", "1" if persist_toggle else "0")
                _db_set_kv("persistent_login_ttl_sec", str(int(ttl_days) * 86400))
                _add_audit("persistent_login_setting_changed", f"enabled={persist_toggle}, ttl_days={ttl_days}")
                st.success("Kalici giris ayari kaydedildi.")
                st.rerun()

        if _can_export_personal_data(role_now):
            st.download_button(
                "Verimi disa aktar (JSON)",
                data=_export_current_user_profile_bytes(),
                file_name="dmd_my_data_export.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.info("Bu rolde profil export yetkisi yok.")
        if st.button("Profil verimi sil (geri alinamaz)", key="delete_my_profile_data", use_container_width=True):
            if _delete_current_user_profile_data():
                st.warning("Profil verisi silindi. Sayfa yenilenecek.")
                st.rerun()

# --- ANA ICERIK BASLIGI (Acik Tema) ---
st.markdown(
    """
    <div style="padding: 6px 0 10px 0;">
        <h1 style="margin-bottom: 4px; color:#0f172a; font-family:'Poppins',sans-serif; letter-spacing:0.5px;">NIZEN</h1>
        <p style="margin:0; color:#475569; font-size:1.02rem;">Neurodegenerative Clinical Platform</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown('<div class="header-line"></div>', unsafe_allow_html=True)

# --- SAYFA 0: ANA PANEL (DASHBOARD) ---
if page == D['nav'][PAGE_DASHBOARD]:
    # --- NEW/UPDATED --- Researcher rolunde sadece anonim ozet
    if st.session_state.get("user_role") == "researcher":
        all_patients = st.session_state.get("patients", {})
        visit_total = 0
        nsaa_vals = []
        for _, p in all_patients.items():
            p = p if isinstance(p, dict) else {}
            visit_total += len(p.get("visits", []) if isinstance(p.get("visits", []), list) else [])
            try:
                nsaa_vals.append(int(p.get("nsaa_total", 0)))
            except Exception:
                continue
        avg_nsaa = round(sum(nsaa_vals) / len(nsaa_vals), 1) if nsaa_vals else 0.0
        st.subheader("Araştırma Özeti (Anonim)")
        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Toplam Hasta", len(all_patients))
        with r2:
            st.metric("Toplam Ziyaret", visit_total)
        with r3:
            st.metric("Ortalama NSAA", avg_nsaa)
        st.info("Bu görünüm anonim araştırma özeti ile sınırlıdır.")
        st.stop()

    # Dashboard hesaplamaları için session değerlerini güvenli tipe çek.
    try:
        kilo_val = float(st.session_state.get("kilo", 30.0))
    except (TypeError, ValueError):
        kilo_val = 30.0
    try:
        yas_val = int(st.session_state.get("yas", 6))
    except (TypeError, ValueError):
        yas_val = 6
    try:
        nsaa_val = int(st.session_state.get("nsaa_total", 0))
    except (TypeError, ValueError):
        nsaa_val = 0
    nsaa_val = max(0, min(34, nsaa_val))

    st.markdown(f"""
        <div style=\"background: linear-gradient(to right, #1c83e1, #00d4ff); padding: 20px; border-radius: 15px; margin-bottom: 25px;\">
            <h1 style=\"color: white; margin: 0;\"> {D['nav'][0]}</h1>
            <p style=\"color: rgba(255,255,255,0.85); margin: 0;\">Hasta Veri Takibi ve Sistem Analitiği</p>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(label="Sistem Durumu", value="Aktif", delta="v1.2 Global")
    with c2:
        st.metric(label="Veri Gizliliği", value="Yerel (Local)", delta="KVKK Uyumlu")
    with c3:
        st.metric(label="Klinik Rehber", value="2024/26", delta="Güncel")
    with c4:
        today = datetime.now().date()
        next7 = 0
        for r in st.session_state.get("reminders", []):
            try:
                d = datetime.strptime(str(r.get("date", "")), "%Y-%m-%d").date()
                if 0 <= (d - today).days <= 7:
                    next7 += 1
            except Exception:
                continue
        st.metric(label="7 Gün Randevu", value=str(next7), delta="Yaklaşan")

    st.markdown("<br>", unsafe_allow_html=True)

        # --- BASIT TREND & RISK (UPGRADED) ---
    visits = st.session_state.get("visits", [])
    mp = st.session_state.get("molecular_profile", {}) or {}

    delta = 0
    direction = "stable"
    risk = "LOW"
    per_year = 0.0

    try:
        if isinstance(visits, list) and len(visits) >= 2:

            # tarih sıralama güvenli olsun
            visits_sorted = sorted(
                visits,
                key=lambda x: str(x.get("date") or x.get("time") or "")
            )

            first = visits_sorted[0]
            last = visits_sorted[-1]

            nsaa_first = int(float(first.get("nsaa", first.get("nsaa_total", 0))))
            nsaa_last = int(float(last.get("nsaa", last.get("nsaa_total", 0))))

            delta = nsaa_last - nsaa_first

            # yearly hız (tahmini)
            d0 = str(first.get("date") or first.get("time") or "")[:10]
            d1 = str(last.get("date") or last.get("time") or "")[:10]

            try:
                dt0 = datetime.strptime(d0, "%Y-%m-%d")
                dt1 = datetime.strptime(d1, "%Y-%m-%d")
                days = max((dt1 - dt0).days, 1)
                per_year = delta * (365.0 / days)
            except Exception:
                per_year = 0.0

            if delta < 0:
                direction = "⬇️ Declining"
            elif delta > 0:
                direction = "⬆️ Improving"
            else:
                direction = "➡️ Stable"

            if per_year <= -4:
                risk = "HIGH"
            elif per_year <= -2:
                risk = "MEDIUM"

    except Exception:
        pass

    t1, t2, t3 = st.columns(3)

    with t1:
        st.metric("NSAA Trend", f"{delta:+d}", help=f"Estimated yearly change: {per_year:+.1f}/year")

    with t2:
        st.metric("Risk Level", risk)

    with t3:
        st.metric("Direction", direction)

    # --- ALERTS ---
    alerts = []

    try:
        if delta <= -3:
            alerts.append("⚠️ NSAA düşüşü tespit edildi.")
    except Exception:
        pass

    try:
        ef = mp.get("ef")
        if str(ef).strip() != "" and int(float(ef)) < 55:
            alerts.append("❤️ EF düşük olabilir (kardiyak kontrol önerilir).")
    except Exception:
        pass

    try:
        fvc = mp.get("fvc")
        if str(fvc).strip() != "" and int(float(fvc)) < 80:
            alerts.append("🫁 FVC düşük olabilir (solunum değerlendirmesi önerilir).")
    except Exception:
        pass

    if alerts:
        for a in alerts:
            st.warning(a)
    else:
        st.success("✅ Kritik uyarı yok")

    # --- HASTA ÖZET KARTI (HTML yerine native Streamlit; kod metni görünmesini engeller) ---
    st.subheader("Mevcut Klinik Profil Özeti")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.metric("Vücut Ağırlığı", f"{kilo_val} kg")
    with p2:
        st.metric("Mevcut Yaş", f"{yas_val} Yıl")
    with p3:
        st.metric("Son NSAA Skoru", f"{nsaa_val} / 34")

    st.caption("Veriler son muayene kayıtlarına göre senkronize edilmiştir.")

    st.markdown("<br>", unsafe_allow_html=True)
    patients_all = st.session_state.get("patients", {})
    active_pid = st.session_state.get("active_patient_id", "")
    active_name = patients_all.get(active_pid, {}).get("name", active_pid or "-")
    reminders = st.session_state.get("reminders", [])
    visits = st.session_state.get("visits", [])
    nsaa_hist = st.session_state.get("nsaa_history", [])
    issues = _quality_checks()

    nsaa_delta = "Baz"
    if isinstance(nsaa_hist, list) and len(nsaa_hist) >= 2:
        try:
            nsaa_delta = f"{int(nsaa_hist[-1].get('total', 0)) - int(nsaa_hist[-2].get('total', 0)):+d}"
        except Exception:
            nsaa_delta = "Baz"

    kx1, kx2, kx3, kx4 = st.columns(4)
    with kx1:
        st.metric("Aktif Hasta", active_name)
    with kx2:
        st.metric("NSAA Değişim", nsaa_delta)
    with kx3:
        st.metric("Toplam Hatırlatıcı", len(reminders))
    with kx4:
        st.metric("Toplam Ziyaret", len(visits))

    t_dash1, t_dash2, t_dash3 = st.tabs(["Performans", "Ziyaret Geçmişi", "Öncelikler"])

    with t_dash1:
        st.markdown("#### Fonksiyonel Seyir İzleme")
        st.markdown(
            f"""
            <div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:14px; padding:16px;">
                <p style="margin:0 0 10px 0; color:#475569; font-weight:600;">Son 6 Ay NSAA Özeti</p>
                <div style="height:10px; border-radius:999px; background:#e2e8f0; overflow:hidden;">
                    <div style="width:{(nsaa_val / 34) * 100:.1f}%; height:100%; background:linear-gradient(to right,#1c83e1,#00d4ff);"></div>
                </div>
                <p style="margin:10px 0 0 0; color:#334155; font-size:0.9rem;">
                    Güncel skor: <b>{nsaa_val}/34</b> ({((nsaa_val / 34) * 100):.1f}%)
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if nsaa_hist:
            df_nsaa = pd.DataFrame(nsaa_hist)
            if {"time", "total"}.issubset(df_nsaa.columns):
                st.line_chart(df_nsaa.set_index("time")[["total"]], use_container_width=True)

    with t_dash2:
        st.markdown("#### Son Klinik Ziyaretler")
        if visits:
            vdf = pd.DataFrame(visits)
            if {"time", "nsaa_total", "pul_score", "vignos_score"}.issubset(vdf.columns):
                st.line_chart(
                    vdf.set_index("time")[["nsaa_total", "pul_score", "vignos_score"]],
                    use_container_width=True,
                )
            show_cols = [c for c in ["time", "source", "note", "nsaa_total", "pul_score", "vignos_score"] if c in vdf.columns]
            if show_cols:
                st.dataframe(vdf[show_cols].tail(8), use_container_width=True, hide_index=True)
        else:
            st.info("Henüz ziyaret kaydı yok.")

    with t_dash3:
        st.markdown("#### Klinik Öncelikler")
        if issues:
            for issue in issues:
                st.warning(issue)
        else:
            st.success("Kritik kalite sorunu görünmüyor.")
        alerts = _clinical_alerts()
        for a in alerts:
            if a.get("level") == "critical":
                st.error(f"Klinik Alarm: {a.get('text')}")
            else:
                st.warning(f"Klinik Uyari: {a.get('text')}")

        today = datetime.now().date()
        upcoming = []
        for r in reminders:
            try:
                d = datetime.strptime(str(r.get("date", "")), "%Y-%m-%d").date()
                diff = (d - today).days
                if diff >= 0:
                    upcoming.append((diff, r))
            except Exception:
                continue
        if upcoming:
            upcoming.sort(key=lambda x: x[0])
            st.markdown("**Yaklaşan Randevular (İlk 5)**")
            for diff, rec in upcoming[:5]:
                st.write(f"- {rec.get('title', 'Randevu')} | {rec.get('date')} | {diff} gün kaldı")
        else:
            st.info("Yaklaşan randevu bulunmuyor.")

        with st.expander("Hızlı Randevu Ekle"):
            with st.form("quick_reminder_dashboard", clear_on_submit=True):
                q_date = st.date_input("Randevu Tarihi", key="qrem_date")
                q_title = st.text_input("Başlık", key="qrem_title")
                q_note = st.text_input("Not", key="qrem_note")
                q_ok = st.form_submit_button("Ekle")
                if q_ok and q_title.strip():
                    st.session_state["reminders"].append(
                        {"date": str(q_date), "title": q_title.strip(), "note": q_note.strip()}
                    )
                    _save_pipeline("reminder_add_dashboard", q_title.strip())
                    st.success("Randevu eklendi.")
                    st.rerun()

    st.divider()

    # Güvenlik Alt Bilgisi
    st.markdown("""
        <div style="text-align: center; padding: 20px; opacity: 0.6;">
            <p style="margin: 0; font-size: 0.85rem; color: #666;">
                 <b>Güvenlik Notu:</b> Verileriniz bu cihazdaki yerel veritabanında ve oturum durumunda saklanır.
            </p>
        </div>
    """, unsafe_allow_html=True)

# --- SAYFA 1: KLİNİK HESAPLAYICI (PREMIUM KARAR DESTEK SİSTEMİ) ---
elif page == D['nav'][PAGE_CALCULATOR]:
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1c83e1 0%, #155ea1 100%); padding: 30px; border-radius: 20px; text-align: center; color: white; margin-bottom: 30px;">
            <h1 style="margin: 0; font-size: 2.5rem;">{D['calc_h']}</h1>
            <p style="opacity: 0.9;">Kişiselleştirilmiş Steroid ve Evreleme Analizi</p>
        </div>
    """, unsafe_allow_html=True)

    # Veri Giriş Kartı
    with st.container():
        st.markdown("<div style='background: #f8fbff; padding: 25px; border-radius: 20px; border: 1px solid #e1e9f5;'>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            kilo = st.number_input(D['weight'], 10.0, 150.0, float(st.session_state.kilo), step=0.5)
            st.session_state.kilo = kilo
        with c2:
            yas = st.number_input(D['age'], 0, 50, int(st.session_state.yas))
            st.session_state.yas = yas
        with c3:
            mut_tipi = st.selectbox(D['mut'], ["Delesyon", "Duplikasyon", "Nonsense (Nokta)", "Diğer"])
        st.markdown("</div>", unsafe_allow_html=True)
    save_current_session_profile()

    st.markdown("<br>", unsafe_allow_html=True)

    # Hesaplama ve Analiz Bölümü
    col_res, col_stage = st.columns([1.5, 1])

    with col_res:
        st.markdown("###  Dozaj ve Tedavi Öngörüsü")
        
        # Steroid Hesaplama (Deflazacort 0.9mg/kg standardı)
        ster_dose = round(kilo * 0.9, 1)
        
        st.markdown(f"""
            <div style="background: white; padding: 30px; border-radius: 20px; box-shadow: 0 4px 20px rgba(148,163,184,0.22); border-top: 8px solid #28a745;">
                <p style="color: #666; font-size: 0.9rem; margin-bottom: 5px;">{D['ster_res']}</p>
                <h2 style="color: #28a745; font-size: 3rem; margin: 0;">{ster_dose} <span style="font-size: 1.2rem;">mg/gün</span></h2>
                <hr style="opacity: 0.2; margin: 20px 0;">
                <p style="font-size: 0.85rem; color: #555;"><b>Mutasyon Bazlı Not:</b> {mut_tipi} tespiti sonrası uygun <b>Ekzon Atlama</b> veya <b>Stop-Codon</b> tedavileri için genetik raporunuzu doktorunuza onaylatın.</p>
            </div>
        """, unsafe_allow_html=True)

    with col_stage:
        st.markdown("###  Klinik Evre Analizi")
        
        if yas <= 5:
            color, stage_name, focus = "#28a745", "Erken Çocukluk", "Tanı ve Steroid Hazırlığı"
            icon = ""
        elif yas <= 12:
            color, stage_name, focus = "#ffc107", "Geçiş / Ambulatuar", "Yürüme Kapasitesinin Korunması"
            icon = ""
        else:
            color, stage_name, focus = "#dc3545", "Erişkin / Non-Ambulatuar", "Solunum ve Kardiyak Destek"
            icon = ""

        st.markdown(f"""
            <div style="background: white; padding: 25px; border-radius: 20px; box-shadow: 0 4px 20px rgba(148,163,184,0.22); border-right: 8px solid {color}; height: 100%;">
                <h4 style="color: {color}; margin-top: 0;">{icon} {stage_name}</h4>
                <p style="font-size: 0.9rem; color: #333;"><b>Yaş:</b> {yas}</p>
                <p style="font-size: 0.9rem; color: #333;"><b>Kritik Odak:</b><br>{focus}</p>
                <p style="font-size: 0.75rem; color: #888; margin-top: 15px;">*Bu evreleme genel literatür bilgisidir.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- BERFİN NİDA ÖZTÜRK VİZYONU: ÖZEL TAKİP PANELİ ---
    st.markdown("""
        <div style="background: #f0f2f6; padding: 25px; border-radius: 20px; border-left: 10px solid #1c83e1;">
            <h4 style="margin-top: 0; color: #1c83e1;"> Berfin Nida Öztürk Vizyonu: Takip Önerileri</h4>
    """, unsafe_allow_html=True)

    # --- NEW/UPDATED --- rol tabanli klinik mod
    doctor_mode = st.session_state.get("user_role") in {"doctor", "admin"}
    if yas <= 5:
        msg = (
            "Erken dönem: NSAA baz hattı ve eklem açıklığı takibi önerilir."
            if doctor_mode
            else "Erken Çocukluk: Fizyoterapi değerlendirmeleri (NSAA baz hattı) için uygun dönemdir."
        )
        st.info(msg)
    elif 6 <= yas <= 12:
        msg = (
            "Geçiş dönemi: skolyoz takibi, kontraktür önleme ve kardiyoloji izlemi sıklaştırılmalıdır."
            if doctor_mode
            else "Geçiş Dönemi: Skolyoz takibi ve gece splintleri bu evrede önem taşır."
        )
        st.warning(msg)
    else:
        msg = (
            "İleri dönem: NIV/BiPAP gereksinimi, SFT ve kardiyak izlemle birlikte değerlendirilmelidir."
            if doctor_mode
            else "Erişkin Dönem: Non-invaziv ventilasyon (BiPAP) ihtiyacı SFT sonuçlarına göre değerlendirilmelidir."
        )
        st.error(msg)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # --- NEW/UPDATED --- Klinik Hesaplayici olcumunu ziyarete kaydet
    if st.button("Bu ölçümü ziyarete kaydet", use_container_width=True):
        rec = _upsert_active_patient_visit(note="Klinik Hesaplayıcı ölçümü", source="clinical_calculator")
        save_current_session_profile()
        _add_audit("visit_upsert_calculator", rec.get("date", ""))
        st.success("Ölçüm ziyaret kaydına işlendi.")

    st.divider()
    t_calc1, t_calc2, t_calc3 = st.tabs(
        ["Moleküler/Genetik Profil", "Birey ve Aile Planı", "Klinik Özet ve Kaydet"]
    )

    with t_calc1:
        st.subheader("Moleküler ve Genetik Değerlendirme")
        mp = st.session_state.get("molecular_profile", {})
        mg1, mg2, mg3 = st.columns(3)
        with mg1:
            test_yontemi = st.selectbox(
                "Genetik Test Yöntemi",
                ["MLPA", "NGS Panel", "WES/WGS", "Bilinmiyor"],
                index=(["MLPA", "NGS Panel", "WES/WGS", "Bilinmiyor"].index(mp.get("test_method")) if mp.get("test_method") in ["MLPA", "NGS Panel", "WES/WGS", "Bilinmiyor"] else 3),
            )
            varyant_notasyonu = st.text_input("Varyant Notasyonu (HGVS)", value=str(mp.get("variant_notation", "")))
            exon_bolge = st.text_input("Etkilenen Ekzon(lar)", value=str(mp.get("exon_region", "")), placeholder="Örn: 45-50 delesyon")
        with mg2:
            ck_degeri = st.number_input("CK (U/L)", min_value=0, max_value=200000, value=int(mp.get("ck", 0)), step=100)
            ambulasyon = st.selectbox(
                "Ambulasyon Durumu",
                ["Ambulatuar", "Kısmi Ambulatuar", "Non-ambulatuar"],
                index=(["Ambulatuar", "Kısmi Ambulatuar", "Non-ambulatuar"].index(mp.get("ambulation")) if mp.get("ambulation") in ["Ambulatuar", "Kısmi Ambulatuar", "Non-ambulatuar"] else 0),
            )
            nonsense_olasi = st.checkbox("Nonsense/stop-codon olasılığı", value=bool(mp.get("nonsense_flag", False)))
        with mg3:
            ef = st.number_input("Kardiyak EF (%)", min_value=0, max_value=100, value=int(mp.get("ef", 60)))
            fvc = st.number_input("FVC (% beklenen)", min_value=0, max_value=150, value=int(mp.get("fvc", 100)))
            steroid_rejimi = st.selectbox(
                "Steroid Rejimi",
                ["Deflazacort", "Prednizolon", "Yok", "Diğer"],
                index=(["Deflazacort", "Prednizolon", "Yok", "Diğer"].index(mp.get("steroid_regimen")) if mp.get("steroid_regimen") in ["Deflazacort", "Prednizolon", "Yok", "Diğer"] else 0),
            )

        options = _infer_targeted_options(mut_tipi, exon_bolge, nonsense_olasi)
        st.markdown("**Hedefe Yönelik Tedavi Uygunluk Notu (karar desteği):**")
        for opt in options:
            st.write(f"- {opt}")
        st.caption("Not: Bu bölüm bilgilendirme amaçlıdır; kesin tedavi kararı uzman hekim tarafından verilmelidir.")

        if st.button("Moleküler Profili Kaydet", use_container_width=True):
            st.session_state["molecular_profile"] = {
                "test_method": test_yontemi,
                "variant_notation": varyant_notasyonu.strip(),
                "exon_region": exon_bolge.strip(),
                "ck": int(ck_degeri),
                "ambulation": ambulasyon,
                "nonsense_flag": bool(nonsense_olasi),
                "ef": int(ef),
                "fvc": int(fvc),
                "steroid_regimen": steroid_rejimi,
                "mutation_type": mut_tipi,
            }
            _append_visit("molecular_profile", exon_bolge.strip())
            _sync_session_to_active_patient()
            save_current_session_profile()
            _add_audit("molecular_profile_saved", exon_bolge.strip())
            st.success("Moleküler profil kaydedildi.")

    with t_calc2:
        st.subheader("Birey ve Aile Odaklı Günlük Yaşam Planı")
        cp = st.session_state.get("care_plan", {})
        cpa1, cpa2 = st.columns(2)
        with cpa1:
            yorgunluk = st.slider("Günlük yorgunluk düzeyi", 0, 10, int(cp.get("fatigue", 3)))
            agri = st.slider("Ağrı düzeyi", 0, 10, int(cp.get("pain", 2)))
            uyku = st.slider("Uyku kalitesi", 0, 10, int(cp.get("sleep_quality", 6)))
        with cpa2:
            dusme = st.number_input("Son 1 ay düşme sayısı", min_value=0, max_value=100, value=int(cp.get("falls_month", 0)))
            okul_is = st.selectbox(
                "Okul/iş katılım durumu",
                ["Aktif", "Kısmi", "Destek gerekli"],
                index=(["Aktif", "Kısmi", "Destek gerekli"].index(cp.get("participation")) if cp.get("participation") in ["Aktif", "Kısmi", "Destek gerekli"] else 0),
            )
            hedefler = st.multiselect(
                "Öncelikli hedefler",
                ["Yürüme dayanıklılığı", "Üst ekstremite fonksiyonu", "Solunum egzersizi", "Postür/skolyoz takibi", "Okul/iş uyumu", "Psikososyal destek"],
                default=cp.get("goals", []),
            )

        st.markdown("**Önerilen haftalık odak planı:**")
        if yorgunluk >= 7:
            st.warning("- Yorgunluk yüksek: yoğun aktiviteler gün içine yayılmalı, enerji koruma planı uygulanmalı.")
        else:
            st.info("- Enerji yönetimi stabil: planlı aktivite + dinlenme döngüsü korunabilir.")
        if dusme >= 2:
            st.warning("- Düşme riski artmış: ev içi güvenlik düzenlemeleri ve denge egzersizi önceliklendirilmeli.")
        if uyku <= 4:
            st.warning("- Uyku kalitesi düşük: gece solunum semptomları ve uyku hijyeni gözden geçirilmeli.")
        if "Solunum egzersizi" in hedefler or yas >= 12:
            st.info("- Solunum takibi: düzenli SFT/NIV değerlendirme randevusu planlanmalı.")

        if st.button("Birey/Aile Planını Kaydet", use_container_width=True):
            st.session_state["care_plan"] = {
                "fatigue": int(yorgunluk),
                "pain": int(agri),
                "sleep_quality": int(uyku),
                "falls_month": int(dusme),
                "participation": okul_is,
                "goals": hedefler,
            }
            _append_visit("care_plan", okul_is)
            _sync_session_to_active_patient()
            save_current_session_profile()
            _add_audit("care_plan_saved", okul_is)
            st.success("Birey/aile planı kaydedildi.")

    with t_calc3:
        st.subheader("Klinik Özet")
        mp = st.session_state.get("molecular_profile", {})
        cp = st.session_state.get("care_plan", {})
        csum1, csum2, csum3 = st.columns(3)
        with csum1:
            st.metric("Mutasyon Tipi", mut_tipi)
            st.metric("CK (U/L)", mp.get("ck", "-"))
        with csum2:
            st.metric("Kardiyak EF", f"{mp.get('ef', '-')}%")
            st.metric("FVC", f"{mp.get('fvc', '-')}%")
        with csum3:
            st.metric("Yorgunluk", cp.get("fatigue", "-"))
            st.metric("Düşme (ay)", cp.get("falls_month", "-"))

        st.caption("Bu özet klinik değerlendirmeyi desteklemek içindir; tanı/tedavi kararı yerine geçmez.")
        if st.button("Sayfadaki Tüm Verileri Profile Kaydet", use_container_width=True):
            _sync_session_to_active_patient()
            save_current_session_profile()
            _add_audit("clinical_calculator_full_save", st.session_state.get("active_patient_id", ""))
            st.success("Klinik Hesaplayıcı verileri kaydedildi.")

# --- SAYFA 2: TAM ÖLÇEKLİ NSAA (CLINICAL ASSESSMENT MODE) ---
elif page == D['nav'][PAGE_NSAA]:
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4433ff 0%, #00d4ff 100%); padding: 30px; border-radius: 20px; text-align: center; color: white; margin-bottom: 30px;">
            <h1 style="margin: 0;">{D['nsaa_h']}</h1>
            <p style="opacity: 0.9;">Fonksiyonel Mobilite Ölçümü (Standart 17 Madde)</p>
        </div>
    """, unsafe_allow_html=True)

    # Rehber Bilgi Paneli
    with st.expander("ℹ Puanlama Kriterleri Rehberi"):
        st.markdown("""
        - **2 Puan:** Normal (Hareketi herhangi bir yardım almadan, modifiye etmeden tamamlar).
        - **1 Puan:** Modifiye (Hareketi tamamlar ancak telafi edici mekanizmalar/destek kullanır).
        - **0 Puan:** Yapamıyor (Hareketi hiçbir şekilde başlatamıyor veya tamamlayamıyor).
        """)

    # Hızlı aksiyonlar
    a1, a2, a3 = st.columns(3)
    if a1.button(" Tümünü 2 Yap", use_container_width=True):
        for i in range(17):
            st.session_state[f"n_{i}"] = 2
        st.rerun()
    if a2.button(" Tümünü 1 Yap", use_container_width=True):
        for i in range(17):
            st.session_state[f"n_{i}"] = 1
        st.rerun()
    if a3.button(" Sıfırla", use_container_width=True):
        for i in range(17):
            st.session_state[f"n_{i}"] = 0
        st.rerun()

    # NSAA Maddeleri + Kategori
    maddeler = [
        ("1. Ayakta Durma", "Dik pozisyonda stabil duruş", "Temel Duruş ve Geçiş"),
        ("2. Sandalyeden Kalkma", "Kollar göğüste çapraz kalkış", "Temel Duruş ve Geçiş"),
        ("3. Tek Ayak Üstünde Durma (Sağ)", "Minimum 3 saniye", "Denge ve Stabilite"),
        ("4. Tek Ayak Üstünde Durma (Sol)", "Minimum 3 saniye", "Denge ve Stabilite"),
        ("5. Sırt Üstü Yatıştan Kalkma", "Supine position to standing", "Temel Duruş ve Geçiş"),
        ("6. Sandalyeye Oturma", "Kontrollü iniş", "Temel Duruş ve Geçiş"),
        ("7. Topuk Üstünde Durma", "Dorsifleksiyon kapasitesi", "Denge ve Stabilite"),
        ("8. Parmak Ucunda Durma", "Plantarfleksiyon gücü", "Denge ve Stabilite"),
        ("9. Zıplama", "İki ayağın yerden kesilmesi", "Lokomasyon"),
        ("10. Sağ Merdiven Çıkma", "Desteksiz yükselme", "Lokomasyon"),
        ("11. Sol Merdiven Çıkma", "Desteksiz yükselme", "Lokomasyon"),
        ("12. Sağ Merdiven İnme", "Kontrollü iniş", "Lokomasyon"),
        ("13. Sol Merdiven İnme", "Kontrollü iniş", "Lokomasyon"),
        ("14. Koşma", "10 metre hızlı tempo", "Lokomasyon"),
        ("15. Yerden Kalkma", "Gowers belirtisi kontrolü", "Lokomasyon"),
        ("16. Zıplayarak İlerleme", "Sıçrama koordinasyonu", "Lokomasyon"),
        ("17. Başını Kaldırma", "Sırt üstü yatarken (boyun fleksörleri)", "Temel Duruş ve Geçiş"),
    ]

    cat_max = {
        "Temel Duruş ve Geçiş": 10,  # 5 madde * 2
        "Denge ve Stabilite": 8,     # 4 madde * 2
        "Lokomasyon": 16,            # 8 madde * 2
    }
    cat_score = {k: 0 for k in cat_max}
    score = 0

    c_n1, c_n2 = st.columns(2)
    for i, (m, desc, cat) in enumerate(maddeler):
        with (c_n1 if i < 9 else c_n2):
            st.markdown(f"""
                <div style="background: #f8f9fa; padding: 15px; border-radius: 12px; border-left: 5px solid #4433ff; margin-bottom: 8px; box-shadow: 2px 2px 5px rgba(148,163,184,0.18);">
                    <p style="margin: 0; font-weight: 700; color: #333;">{m}</p>
                    <p style="margin: 0; font-size: 0.8rem; color: #666;">{desc}</p>
                    <p style="margin: 4px 0 0 0; font-size: 0.72rem; color: #475569;"><b>Kategori:</b> {cat}</p>
                </div>
            """, unsafe_allow_html=True)

            if hasattr(st, "segmented_control"):
                res = st.segmented_control(
                    label=f"Puan_{i}",
                    options=[0, 1, 2],
                    key=f"n_{i}",
                    default=2,
                    label_visibility="collapsed"
                )
            else:
                res = st.radio(
                    label=f"Puan_{i}",
                    options=[0, 1, 2],
                    index=2,
                    key=f"n_{i}",
                    horizontal=True,
                    label_visibility="collapsed"
                )
            val = int(res) if res is not None else 0
            score += val
            cat_score[cat] += val
            st.markdown("<br>", unsafe_allow_html=True)

    st.session_state.nsaa_total = score
    save_current_session_profile()

    # --- SONUÇ ANALİZ PANELİ ---
    st.divider()
    pct = (score / 34) * 100
    prev_score = st.session_state.get("nsaa_prev_total")
    delta_txt = f"{score - prev_score:+d} puan (önceki teste göre)" if isinstance(prev_score, int) else f"{pct:.1f}%"

    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.metric(label=D['score_h'], value=f"{score} / 34", delta=delta_txt)
        st.progress(score / 34)

    with res_col2:
        if score >= 25:
            st.success(" **Yüksek Fonksiyonel Kapasite:** Bağımsız mobilite büyük ölçüde korunuyor. Mevcut fizyoterapi ve izlem planı sürdürülmeli.")
        elif score >= 15:
            st.warning(" **Orta Seviye Etkilenim:** Telafi mekanizmaları artmış olabilir. Kontraktür önleme ve denge egzersizleri yoğunlaştırılmalıdır.")
        else:
            st.error(" **Belirgin Fonksiyon Kaybı:** Günlük aktivitede destek ihtiyacı artar. Solunum ve kardiyak değerlendirme yakın aralıklı planlanmalıdır.")

    # --- NEW/UPDATED --- NSAA olcumunu ziyarete kaydet
    if st.button("Bu ölçümü ziyarete kaydet", key="save_nsaa_visit", use_container_width=True):
        rec = _upsert_active_patient_visit(note=f"NSAA ölçümü: {score}/34", source="nsaa")
        save_current_session_profile()
        _add_audit("visit_upsert_nsaa", rec.get("date", ""))
        st.success("NSAA ölçümü ziyaret kaydına işlendi.")

    # Alt kategori skorları
    st.markdown("#### Alt Kategori Skorları")
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Temel Duruş ve Geçiş", f"{cat_score['Temel Duruş ve Geçiş']} / {cat_max['Temel Duruş ve Geçiş']}")
    with k2:
        st.metric("Denge ve Stabilite", f"{cat_score['Denge ve Stabilite']} / {cat_max['Denge ve Stabilite']}")
    with k3:
        st.metric("Lokomasyon", f"{cat_score['Lokomasyon']} / {cat_max['Lokomasyon']}")

    weakest_cat = min(cat_score.keys(), key=lambda k: (cat_score[k] / cat_max[k]) if cat_max[k] else 1)
    st.info(f" Öncelikli rehabilitasyon odağı: **{weakest_cat}**")

    # Rapor kaydı (session içinde)
    if "nsaa_history" not in st.session_state:
        st.session_state.nsaa_history = []

    if st.button(" NSAA Raporunu Kaydet", use_container_width=True):
        ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        st.session_state.nsaa_history.append({
            "time": ts,
            "total": score,
            "pct": round(pct, 1),
            "temel": cat_score["Temel Duruş ve Geçiş"],
            "denge": cat_score["Denge ve Stabilite"],
            "lokomasyon": cat_score["Lokomasyon"],
        })
        st.session_state.nsaa_prev_total = score
        _append_visit("nsaa", f"Toplam {score}/34")
        save_current_session_profile()
        st.toast("NSAA raporu kaydedildi.", icon="")

    if st.session_state.nsaa_history:
        with st.expander(" Son NSAA Kayıtları (Son 5)"):
            for rec in reversed(st.session_state.nsaa_history[-5:]):
                st.markdown(
                    f"- **{rec['time']}** | Toplam: **{rec['total']}/34** ({rec['pct']}%) | "
                    f"Temel: {rec['temel']} | Denge: {rec['denge']} | Lokomasyon: {rec['lokomasyon']}"
                )

# --- SAYFA 3: SSS (KAPSAMLI AKADEMİK REHBER) ---
elif page == D['nav'][PAGE_FAQ]:
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1c83e1 0%, #00d4ff 100%); padding: 40px; border-radius: 25px; text-align: center; color: white; margin-bottom: 30px; box-shadow: 0 10px 30px rgba(28,131,225,0.3);">
            <h1 style="margin: 0; font-size: 2.8rem;">{D['faq_h']}</h1>
            <p style="font-size: 1.2rem; opacity: 0.9;">WDO ve TREAT-NMD Standartlarına Göre Hazırlanmış Bilgi Bankası</p>
        </div>
    """, unsafe_allow_html=True)

    # Arama ve Filtreleme Alanı
    faq_search = st.text_input(" Bilgi Bankasında Arayın (Örn: Ekzon, Steroid, Fizik Tedavi...)", "").lower()

    # --- GENİŞLETİLMİŞ VERİ SETİ ---
    faq_data = [
        {
            "cat": " GENETİK VE TEŞHİS",
            "q": "DMD ve BMD Arasındaki Fark Nedir",
            "a": "DMD (Duchenne), distrofin proteininin tamamen eksik olduğu daha ağır seyreden tiptir. BMD (Becker) ise proteinin az veya kusurlu olduğu, semptomların daha geç ve hafif başladığı formdur.",
            "l": "https://mda.org",
            "tag": "Genetik"
        },
        {
            "cat": " FARMAKOLOJİ VE İLAÇ",
            "q": "Steroid (Deflazacort vs Prednisolone) Farkı Nedir",
            "a": "Her iki ilaç da benzer etkinliktedir; ancak Deflazacort'un kilo alımı ve davranışsal yan etkileri daha az, kemik yoğunluğu üzerindeki etkisi ise biraz daha fazla olabilir. Seçim doktora özeldir.",
            "l": "https://dmd.org.tr",
            "tag": "Tedavi"
        },
        {
            "cat": " KLİNİK BULGULAR",
            "q": "Pseudohypertrophy (Yalancı Kas Büyümesi) Nedir",
            "a": "Özellikle baldır kaslarında (Gastrocnemius) görülen büyümedir. Bu büyüme gerçek kas değil, kas dokusunun yerini yağ ve bağ dokusunun almasıdır.",
            "l": "https://nadirx.com",
            "tag": "Klinik"
        },
        {
            "cat": " SOLUNUM VE KALP",
            "q": "Kardiyak Takip Neden Erken Başlamalıdır",
            "a": "DMD hastalarında kalp kası (miyokard) distrofin eksikliğinden etkilenir. Belirti olmasa dahi 10 yaşından önce ACE inhibitörleri gibi koruyucu tedavilere başlamak hayati olabilir.",
            "l": "https://worldduchenne.org",
            "tag": "Kritik"
        },
        {
            "cat": " YENİ NESİL TEKNOLOJİLER",
            "q": "Ekzon Atlama (Exon Skipping) Herkese Uygulanır mı",
            "a": "Hayır. Bu tedavi mutasyona özeldir. Örneğin Eteplirsen sadece 51. ekzonu atlanabilen hastalar içindir. Genetik raporunuzdaki silinme bölgeleri bu tedaviyi belirler.",
            "l": "https://treat-nmd.org",
            "tag": "Ar-Ge"
        },
        {
            "cat": " SOSYAL HAKLAR",
            "q": "ÇÖZGER Raporu Alırken Nelere Dikkat Edilmeli",
            "a": "Raporun 'Özel Koşul Gereksinimi Vardır' (ÖKGV) ibaresini içermesi, evde bakım ve ÖTV muafiyeti gibi haklar için kritiktir. Multidisipliner bir hastaneden alınmalıdır.",
            "l": "https://engelsiz.gov.tr",
            "tag": "Yasal"
        },
        {
            "cat": " FİZİKSEL AKTİVİTE",
            "q": "Yüzme DMD İçin Uygun mudur",
            "a": "Evet, suyun kaldırma kuvveti eklemlere binen yükü azaltır. Ancak suyun çok soğuk olmaması ve çocuğun aşırı yorulmaması (fatigue) şarttır. Hidroterapi en iyi egzersizdir.",
            "l": "https://parentprojectmd.org",
            "tag": "Egzersiz"
        },
        {
            "cat": " GENETİK VE TEŞHİS",
            "q": "DMD Tanısı İçin Hangi Testler Yapılır",
            "a": "Tanıda CK düzeyi, genetik analiz (MLPA/NGS) ve gerekli durumlarda kas biyopsisi kullanılır. Kesin tanı genetik doğrulama ile konur.",
            "l": "https://treat-nmd.org",
            "tag": "Tanı"
        },
        {
            "cat": " FARMAKOLOJİ VE İLAÇ",
            "q": "Steroid Tedavisine Ne Zaman Başlanır",
            "a": "Genellikle motor fonksiyonlar halen korunurken başlanması önerilir. Başlama zamanı klinik tablo, yaş ve hekim değerlendirmesine göre belirlenir.",
            "l": "https://worldduchenne.org",
            "tag": "Tedavi"
        },
        {
            "cat": " FARMAKOLOJİ VE İLAÇ",
            "q": "Steroidlerin Sık Yan Etkileri Nelerdir",
            "a": "Kilo artışı, davranış değişiklikleri, kemik mineral yoğunluğunda azalma, katarakt ve büyüme hızında yavaşlama görülebilir. Düzenli izlem şarttır.",
            "l": "https://mda.org",
            "tag": "Yan Etki"
        },
        {
            "cat": " SOLUNUM VE KALP",
            "q": "Solunum Takibi Ne Sıklıkla Yapılmalıdır",
            "a": "Yaşa ve klinik evreye göre değişmekle birlikte düzenli SFT, gece hipoventilasyon değerlendirmesi ve gerektiğinde NIV/BiPAP planlaması yapılmalıdır.",
            "l": "https://treat-nmd.org",
            "tag": "Takip"
        },
        {
            "cat": " SOLUNUM VE KALP",
            "q": "Kardiyak İzlemde Hangi Tetkikler Önemlidir",
            "a": "Ekokardiyografi, EKG ve uygun hastalarda kardiyak MR önemlidir. Kardiyomiyopati bulguları erken dönemde sessiz olabilir.",
            "l": "https://worldduchenne.org",
            "tag": "Kardiyak"
        },
        {
            "cat": " FİZİKSEL AKTİVİTE",
            "q": "Hangi Egzersizlerden Kaçınılmalıdır",
            "a": "Aşırı zorlayıcı, eksantrik ağırlıklı ve kas yıkımını artırabilecek yüksek yoğunluklu egzersizlerden kaçınılmalıdır. Amaç fonksiyon korumaktır.",
            "l": "https://parentprojectmd.org",
            "tag": "Egzersiz"
        },
        {
            "cat": " KLİNİK BULGULAR",
            "q": "Skolyoz Riski Ne Zaman Artar",
            "a": "Yürüme kaybı sonrası skolyoz riski artabilir. Düzenli ortopedik değerlendirme ve oturma postürü izlemi önemlidir.",
            "l": "https://treat-nmd.org",
            "tag": "Ortopedi"
        },
        {
            "cat": " SOSYAL HAKLAR",
            "q": "Evde Bakım Aylığı İçin Temel Koşullar Nelerdir",
            "a": "Fonksiyonel bağımlılık düzeyi ve hane gelir kriterleri birlikte değerlendirilir. Güncel mevzuat il/ilçe sosyal hizmet birimlerinden teyit edilmelidir.",
            "l": "https://engelsiz.gov.tr",
            "tag": "Yasal"
        },
        {
            "cat": " SOSYAL HAKLAR",
            "q": "Okul Döneminde Hangi Destekler Alınabilir",
            "a": "BEP planı, erişilebilir sınıf düzeni, RAM yönlendirmesi ve fiziksel destek hizmetleri talep edilebilir.",
            "l": "https://engelsiz.gov.tr",
            "tag": "Eğitim"
        },
        {
            "cat": " YENİ NESİL TEKNOLOJİLER",
            "q": "Gen Tedavileri Her DMD Hastasına Uygun mu",
            "a": "Hayır. Uygunluk; mutasyon tipi, yaş, klinik evre ve tedavinin endikasyon kriterlerine göre belirlenir. Karar mutlaka uzman merkezde verilmelidir.",
            "l": "https://worldduchenne.org",
            "tag": "Ar-Ge"
        },
        {
            "cat": " ACİL DURUM",
            "q": "Acil Serviste İlk Hangi Bilgiler Verilmelidir",
            "a": "DMD tanısı, kullanılan ilaçlar (özellikle steroid), solunum destek ihtiyacı ve anesteziye ilişkin kritik uyarılar ilk anda sağlık ekibine iletilmelidir.",
            "l": "https://dmd.org.tr",
            "tag": "Kritik"
        },
        {
            "cat": " BESLENME VE METABOLİZMA",
            "q": "Beslenmede Nelere Dikkat Edilmelidir",
            "a": "Steroid tedavisi alan hastalarda kilo yönetimi, yeterli protein alımı, D vitamini ve kalsiyum dengesi önemlidir. Kişiye özel plan için diyetisyen desteği önerilir.",
            "l": "https://mda.org",
            "tag": "Beslenme"
        }
    ]

    # --- FİLTRELEME VE GÖSTERİM MANTIĞI ---
    filtered_faq = [item for item in faq_data if faq_search in item['q'].lower() or faq_search in item['a'].lower()]

    if not filtered_faq:
        st.warning("Aradığınız kriterlere uygun bilgi bulunamadı. Lütfen farklı kelimeler deneyin.")
    
    for item in filtered_faq:
        with st.expander(f"{item['cat']} | {item['q']}"):
            st.markdown(f"""
                <div style="padding: 10px; border-left: 3px solid #1c83e1; background: #f9f9f9; border-radius: 0 10px 10px 0;">
                    <p style="font-size: 1.05rem; line-height: 1.6; color: #333;">{item['a']}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Alt Bilgi Satırı
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(f"[Akademik Kaynağa Git]({item['l']})")
            with c2:
                st.markdown(f"<p style='text-align:right;'><small style='background:#1c83e1; color:white; padding:3px 8px; border-radius:5px;'>{item['tag']}</small></p>", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # İletişim / Soru Sor Paneli
    st.markdown("""
        <div style="background: #f0f2f6; padding: 25px; border-radius: 15px; border: 1px dashed #1c83e1; text-align: center;">
            <h4>Aradığınız cevabı bulamadınız mı</h4>
            <p>DMD Guardian topluluğuna katılın veya uzman ekibimize danışın.</p>
            <a href="mailto:info@dmdguardian.com" style="text-decoration:none;">
                <button style="background:#1c83e1; color:white; border:none; padding:10px 20px; border-radius:8px; cursor:pointer;">Bize Soru Gönderin</button>
            </a>
        </div>
    """, unsafe_allow_html=True)

# --- SAYFA 4: ACİL DURUM (KRİTİK MÜDAHALE PANELİ) ---
elif page == D['nav'][PAGE_EMERGENCY]:
    st.title(D['emer_h'])
    
    # Doktorlar için hızlı uyarı kartı
    st.warning("**Tıbbi Personel İçin Özet:** Bu hasta Distrofin eksikliği (DMD) tanılıdır. Standart anestezi protokolleri ölümcül olabilir.")

    # Acil Servis Butonu ve Genişletilmiş Görünüm
    if st.button("ACİL SERVİS: DOKTORA GÖSTER (TAM EKRAN)"):
        st.markdown("""
            <div style="background-color:#ff4b4b; padding:40px; border-radius:20px; border: 8px solid #ffffff; text-align:center; box-shadow: 0 0 50px rgba(148,163,184,0.22);">
                <h1 style="color:white; font-size:50px; margin-bottom:10px;">KRİTİK UYARI</h1>
                <h2 style="color:white; border-bottom: 2px solid white; padding-bottom:15px;">HASTA DMD (DUCHENNE) TANILIDIR</h2>
                <div style="text-align:left; color:white; font-size:24px; margin-top:20px; line-height:1.6;">
                    <p><b>1. ANESTEZİ:</b> SÜKSİNİLKOLİN VE TÜM GAZLAR (İnhalanlar) <b>KESİNLİKLE YASAK!</b> Sadece TIVA (Propofol vb.) kullanılabilir.</p>
                    <p><b>2. OKSİJEN:</b> Hedef %92-95. Kontrolsüz yüksek oksijen solunum durmasına yol açabilir!</p>
                    <p><b>3. KALP:</b> Kardiyomiyopati riski nedeniyle EKG ve Troponin takibi yapılmalıdır.</p>
                    <p><b>4. STEROİD:</b> Düzenli steroid alıyorsa, stres dozu (hidrokortizon) gerekebilir.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Solunum ve Kırık Yönetimi Detayları
    col_em1, col_em2 = st.columns(2)
    with col_em1:
        st.subheader("Solunum Desteği")
        st.info("""
        - **Öksürük Desteği:** Manuel veya mekanik (Cough Assist) mutlaka sağlanmalı.
        - **NIV/BiPAP:** Solunum sıkıntısı varsa oksijenden önce cihaz desteği düşünülmeli.
        """)
    with col_em2:
        st.subheader("Kırık ve Travma")
        st.error("""
        - **Yağ Embolisi:** Uzun kemik kırıklarından sonra solunum sıkıntısı başlarsa acil müdahale gerekir.
        - **Hareketsizlik:** Uzun süreli yatak istirahatinden kas yıkımı (atrofi) riski nedeniyle kaçınılmalıdır.
        """)

    st.divider()
    st.subheader("Acil Durum Kartı")
    with st.form("emergency_card_form"):
        ec1, ec2 = st.columns(2)
        with ec1:
            e_patient = st.text_input("Hasta Adı", value=str(st.session_state.get("emergency_card", {}).get("patient_name", "")))
            e_age = st.text_input("Yaş", value=str(st.session_state.get("emergency_card", {}).get("age", st.session_state.get("yas", ""))))
            e_weight = st.text_input("Kilo", value=str(st.session_state.get("emergency_card", {}).get("weight", st.session_state.get("kilo", ""))))
            e_phone = st.text_input("Yakını Telefon", value=str(st.session_state.get("emergency_card", {}).get("contact_phone", "")))
        with ec2:
            e_doc = st.text_input("Sorumlu Hekim", value=str(st.session_state.get("emergency_card", {}).get("doctor_name", "")))
            e_hosp = st.text_input("Hastane", value=str(st.session_state.get("emergency_card", {}).get("hospital", "")))
            e_steroid = st.text_input("Steroid Bilgisi", value=str(st.session_state.get("emergency_card", {}).get("steroid", "")))
            e_allergy = st.text_input("Alerji Notu", value=str(st.session_state.get("emergency_card", {}).get("allergy", "")))
        ec_save = st.form_submit_button("Acil Kartı Kaydet")
        if ec_save:
            st.session_state["emergency_card"] = {
                "patient_name": e_patient.strip(),
                "age": e_age.strip(),
                "weight": e_weight.strip(),
                "contact_phone": e_phone.strip(),
                "doctor_name": e_doc.strip(),
                "hospital": e_hosp.strip(),
                "steroid": e_steroid.strip(),
                "allergy": e_allergy.strip(),
            }
            _save_pipeline("emergency_card_saved", e_patient.strip())
            st.success("Acil kart bilgisi kaydedildi.")

    card_text = _build_emergency_card_text()
    st.download_button(
        "Acil Durum Kartını İndir (TXT)",
        data=card_text.encode("utf-8"),
        file_name="dmd_acil_durum_karti.txt",
        mime="text/plain",
    )

# --- SAYFA 5: KLİNİK TAKVİM & YASAL HAKLAR (EKSİKSİZ) ---
elif page == D['nav'][PAGE_CALENDAR]:
    st.title(D['cal_h'])
    
    tab_cal, tab_law, tab_guide = st.tabs([" Randevu Takvimi", " Devlet Hakları", " Başvuru Rehberi"])
    
    with tab_cal:
        st.subheader(" Periyodik Kontrol Listesi")
        st.write("DMD yönetiminde zamanlama her şeydir. Lütfen aşağıdaki kontrolleri aksatmayın:")
        
        c1, c2 = st.columns(2)
        with c1:
            st.success("**6 Ayda Bir Yapılacaklar:**")
            st.checkbox("Nöroloji / Kas Hastalıkları Muayenesi")
            st.checkbox("Fizyoterapi (Eklem Açıklığı & NSAA)")
            st.checkbox("SFT (Solunum Fonksiyon Testi)")
        with c2:
            st.warning("**Yılda Bir Yapılacaklar:**")
            st.checkbox("Kardiyoloji (EKO ve mümkünse Kardiyak MR)")
            st.checkbox("DEXA (Kemik Yoğunluğu Ölçümü)")
            st.checkbox("Göz Muayenesi (Katarakt Kontrolü - Steroid kaynaklı)")
            
        next_apt = st.date_input("Bir Sonraki Kritik Randevunuzu Not Edin:", help="Randevularınızı buraya kaydederek takip edebilirsiniz.")
        apt_title = st.text_input("Randevu Başlığı", value="Kritik Kontrol")
        if st.button("Randevuyu Hatırlatıcıya Ekle", use_container_width=True):
            rec = {"date": str(next_apt), "title": apt_title.strip() or "Kritik Kontrol", "note": "Takvim sayfasından eklendi"}
            st.session_state["reminders"].append(rec)
            _sync_session_to_active_patient()
            save_current_session_profile()
            _add_audit("calendar_reminder_add", rec["title"])
            st.success("Randevu hatırlatıcı listesine eklendi.")

    with tab_law:
        st.subheader(" Yasal Haklar ve Sosyal Destekler")
        with st.expander(" ÖTV Muafiyetli Araç Alımı"):
            st.write("ÇÖZGER raporunda 'Özel Koşul Gereksinimi Vardır (ÖKGV)' ibaresi bulunan bireyler, 5 yılda bir ÖTV muafiyetli araç alabilirler.")
        
        with st.expander(" Maaş ve Maddi Destekler"):
            st.write("- **Engelli Maaşı:** Hane gelirine bağlı olarak bağlanabilir.")
            st.write("- **Evde Bakım Aylığı:** Tam bağımlı raporu olan bireylerin bakıcılarına ödenir.")
        
        with st.expander(" Eğitim ve RAM Destekleri"):
            st.write("Rehabilitasyon merkezlerinde haftalık seans desteği ve okulda 'BEP' (Bireyselleştirilmiş Eğitim Planı) hakkınız mevcuttur.")
        with st.expander(" Psikososyal Destek ve Bakım Veren Desteği"):
            st.write("- **Psikolojik destek:** Uzun dönem bakım yükü için aile ve birey odaklı danışmanlık planlanabilir.")
            st.write("- **Okul/iş uyumu:** Eğitim kurumu ile bireyselleştirilmiş uyum planı yapılmalıdır.")
            st.write("- **Bakım veren tükenmişliği:** Düzenli mola planı ve sosyal hizmet yönlendirmesi önerilir.")

    with tab_guide:
        st.subheader(" Rapor ve Başvuru Adımları")
        st.markdown("""
        1. **ÇÖZGER Raporu:** Üniversite hastanelerinden veya tam teşekküllü devlet hastanelerinden alınır.
        2. **RAM Raporu:** Fizik tedavi desteği için Rehberlik Araştırma Merkezi'nden randevu alınmalıdır.
        3. **İlaç Raporları:** Steroid ve kalp ilaçları için nöroloji/kardiyoloji tarafından periyodik yenilenmelidir.
        """)
        st.info(" **İpucu:** Tüm raporlarınızın aslı gibidir onaylı fotokopilerini her zaman yanınızda bulundurun.")
# --- SAYFA 6: GÜNCEL DMD HABERLERİ ---
elif page == D['nav'][PAGE_NEWS]:
    st.title(D["news_h"])
    st.caption("Kaynak: Google News RSS. Başlıklara tıklayarak haberin tamamını açabilirsiniz.")

    c_news1, c_news2 = st.columns([1, 1])
    with c_news1:
        if st.button("Haberleri Yenile", use_container_width=True):
            fetch_dmd_news.clear()
            st.rerun()
    with c_news2:
        all_news_url = (
            "https://news.google.com/search?q="
            + quote_plus("duchenne muscular dystrophy")
            + ("&hl=en-US&gl=US&ceid=US:en" if st.session_state.lang == "EN" else "&hl=tr&gl=TR&ceid=TR:tr")
        )
        st.link_button(D["news_all"], all_news_url, use_container_width=True)

    lang_code = "EN" if st.session_state.lang == "EN" else "TR"
    raw_news_items = fetch_dmd_news(lang=lang_code, limit=25)
    news_items = list(raw_news_items)
    key_filter = st.text_input("Haber içinde ara", value="").strip().lower()
    sources = sorted({n.get("source", "Google News") for n in news_items}) if news_items else []
    source_pick = st.selectbox("Kaynak filtresi", ["Tümü"] + sources, index=0)
    if news_items:
        news_items = [
            n for n in news_items
            if (not key_filter or key_filter in str(n.get("title", "")).lower())
            and (source_pick == "Tümü" or n.get("source", "") == source_pick)
        ]

    if not raw_news_items:
        st.warning("Haberler şu an alınamadı. İnternet bağlantınızı ve RSS erişimini kontrol edip tekrar deneyin.")
    elif not news_items:
        st.info("Filtrelere uygun haber bulunamadı. Arama ifadesini veya kaynak filtresini değiştirin.")
    else:
        for i, item in enumerate(news_items, start=1):
            published = item.get("published", "")
            source = item.get("source", "Google News")
            title = html.escape(item.get("title", "Haber başlığı"))
            link = _safe_link(item.get("link", "#"))
            source = html.escape(source)
            published = html.escape(published)
            with st.container(border=True):
                st.markdown(f"**{i}. [{title}]({link})**")
                st.caption(f"Kaynak: {source} | Tarih: {published}")

# --- NEW/UPDATED --- SAYFA 7: KLINIK OPERASYON MERKEZI (ROL BAZLI)
elif page == D['nav'][PAGE_OPS]:
    st.title(D.get("advanced_h", "Gelişmiş Modüller"))
    _ensure_patient_state()
    _ensure_patient_visits_schema()
    _sync_session_to_active_patient()
    role = st.session_state.get("user_role", "family")

    patients_all = st.session_state.get("patients", {})
    active_pid = st.session_state.get("active_patient_id", "")
    active_name = patients_all.get(active_pid, {}).get("name", active_pid or "-")
    reminders_all = st.session_state.get("reminders", [])
    visits_all = sorted([_normalize_visit_record(v) for v in st.session_state.get("visits", [])], key=lambda x: x.get("date", ""))
    last_visit_date = visits_all[-1].get("date", "-") if visits_all else "-"

    due_next7 = 0
    today_dt = datetime.now().date()
    for r in reminders_all:
        try:
            d = datetime.strptime(str(r.get("date", "")), "%Y-%m-%d").date()
            if 0 <= (d - today_dt).days <= 7:
                due_next7 += 1
        except Exception:
            continue

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Toplam Hasta", len(patients_all))
    with m2:
        st.metric("Aktif Hasta", active_name if role != "researcher" else "Anonim")
    with m3:
        st.metric("7 Gün İçinde Randevu", due_next7)
    with m4:
        st.metric("Son Ziyaret", last_visit_date)

    tab_labels = [
        "A) Hasta Yönetimi",
        "B) Ziyaretler (Visit Timeline)",
        "C) Trend & Rapor",
        "D) İlaç & Yan Etki",
        "E) Hatırlatıcı",
        "F) Doktor Notları",
        "I) Araştırma & Anonim Export",
    ]
    if role == "admin":
        tab_labels = [
            "A) Hasta Yönetimi",
            "B) Ziyaretler (Visit Timeline)",
            "C) Trend & Rapor",
            "D) İlaç & Yan Etki",
            "E) Hatırlatıcı",
            "F) Doktor Notları",
            "G) Veri Kalitesi & Audit",
            "H) Yedekleme",
            "I) Araştırma & Anonim Export",
        ]
    if role == "researcher":
        tab_labels = ["I) Araştırma & Anonim Export"]

    tab_objs = st.tabs(tab_labels)
    tabs = dict(zip(tab_labels, tab_objs))

    if "A) Hasta Yönetimi" in tabs:
        with tabs["A) Hasta Yönetimi"]:
            st.subheader("Hasta Yönetimi")
            patients = st.session_state.get("patients", {})
            with st.form("add_patient_form_v3", clear_on_submit=True):
                new_patient_name = st.text_input("Yeni Hasta Adı")
                add_ok = st.form_submit_button("Hasta Ekle")
                if add_ok and new_patient_name.strip():
                    pid = f"hasta_{len(patients) + 1}"
                    patients[pid] = {
                        "name": new_patient_name.strip(),
                        "kilo": 30.0,
                        "yas": 6,
                        "nsaa_total": 0,
                        "nsaa_prev_total": None,
                        "nsaa_history": [],
                        "pul_score": 0,
                        "vignos_score": 1,
                        "pul_history": [],
                        "vignos_history": [],
                        "medications": [],
                        "side_effects": [],
                        "reminders": [],
                        "doctor_notes": [],
                        "emergency_card": {},
                        "molecular_profile": {},
                        "care_plan": {},
                        "visits": [],
                        "documents": [],
                    }
                    st.session_state["active_patient_id"] = pid
                    _sync_active_patient_to_session()
                    _add_audit("patient_add", pid)
                    save_current_session_profile()
                    st.success(f"Hasta eklendi: {new_patient_name}")
                    st.rerun()

    if "B) Ziyaretler (Visit Timeline)" in tabs:
        with tabs["B) Ziyaretler (Visit Timeline)"]:
            st.subheader("Ziyaretler (Visit Timeline)")
            visits = sorted([_normalize_visit_record(v) for v in st.session_state.get("visits", [])], key=lambda x: x.get("date", ""))
            if not visits:
                st.info("Henüz ziyaret kaydı yok.")
            else:
                vdf = pd.DataFrame(visits)
                st.dataframe(vdf[["date", "age", "weight", "nsaa", "pul", "vignos", "ef", "fvc", "notes"]], use_container_width=True, hide_index=True)
                c_left, c_right = st.columns([1, 1])
                with c_left:
                    selected_date = st.selectbox("Visit seç", options=[v.get("date") for v in visits], index=len(visits) - 1)
                current_visit = next((v for v in visits if v.get("date") == selected_date), visits[-1])
                with c_right:
                    st.markdown("**Visit Detayı**")
                    st.write(current_visit)
                st.markdown("#### Iki ziyaret karsilastirma")
                cmp1, cmp2 = st.columns(2)
                with cmp1:
                    d1 = st.selectbox("Ilk tarih", options=[v.get("date") for v in visits], index=max(0, len(visits) - 2), key="cmp_date_1")
                with cmp2:
                    d2 = st.selectbox("Ikinci tarih", options=[v.get("date") for v in visits], index=len(visits) - 1, key="cmp_date_2")
                v1 = next((v for v in visits if v.get("date") == d1), current_visit)
                v2 = next((v for v in visits if v.get("date") == d2), current_visit)
                delta_rows = _visit_delta(v1, v2)
                st.dataframe(pd.DataFrame(delta_rows), use_container_width=True, hide_index=True)
                with st.form("visit_edit_form_v3"):
                    e_age = st.number_input("Yaş", 0, 90, int(current_visit.get("age") or 0))
                    e_weight = st.number_input("Kilo", 0.0, 250.0, float(current_visit.get("weight") or 0.0), step=0.1)
                    e_nsaa = st.number_input("NSAA", 0, 34, int(current_visit.get("nsaa") or 0))
                    e_pul = st.number_input("PUL", 0, 42, int(current_visit.get("pul") or 0))
                    e_vignos = st.number_input("Vignos", 1, 10, int(current_visit.get("vignos") or 1))
                    e_ef = st.number_input("EF", 0, 100, int(current_visit.get("ef") or 0))
                    e_fvc = st.number_input("FVC", 0, 150, int(current_visit.get("fvc") or 0))
                    e_notes = st.text_area("Not", value=str(current_visit.get("notes") or ""))
                    e_ok = st.form_submit_button("Düzenle")
                    if e_ok:
                        updated = []
                        for rec in visits:
                            if rec.get("date") == selected_date:
                                rec = {
                                    **rec,
                                    "age": int(e_age),
                                    "weight": float(e_weight),
                                    "nsaa": int(e_nsaa),
                                    "pul": int(e_pul),
                                    "vignos": int(e_vignos),
                                    "ef": int(e_ef),
                                    "fvc": int(e_fvc),
                                    "notes": e_notes.strip(),
                                }
                            updated.append(_normalize_visit_record(rec))
                        st.session_state["visits"] = updated
                        _sync_session_to_active_patient()
                        save_current_session_profile()
                        _add_audit("visit_update", selected_date)
                        st.success("Visit güncellendi.")
                        st.rerun()

    if "C) Trend & Rapor" in tabs:
        with tabs["C) Trend & Rapor"]:
            st.subheader("Trend & Rapor")
            if visits_all:
                cdf = pd.DataFrame(visits_all).sort_values("date")
                st.line_chart(cdf.set_index("date")[["nsaa", "pul", "vignos"]], use_container_width=True)
            st.markdown("#### Klinik Alarm Motoru")
            alerts = _clinical_alerts()
            if not alerts:
                st.success("Aktif kritik alarm yok.")
            else:
                for a in alerts:
                    if a.get("level") == "critical":
                        st.error(a.get("text", "Kritik alarm"))
                    else:
                        st.warning(a.get("text", "Klinik uyari"))
            report_text = _build_text_report()
            st.download_button("Klinik Raporu İndir (TXT)", data=report_text.encode("utf-8"), file_name="dmd_rapor.txt", mime="text/plain")

    if "D) İlaç & Yan Etki" in tabs:
        with tabs["D) İlaç & Yan Etki"]:
            st.subheader("İlaç & Yan Etki")
            d1, d2 = st.columns(2)
            with d1:
                with st.form("med_form_v3", clear_on_submit=True):
                    m_date = st.date_input("İlaç Tarihi", key="med_date_v3")
                    m_name = st.text_input("İlaç Adı")
                    m_dose = st.text_input("Doz")
                    m_note = st.text_area("Not", key="med_note_v3")
                    if st.form_submit_button("İlaç Kaydı Ekle") and m_name.strip():
                        st.session_state["medications"].append({"date": str(m_date), "name": m_name.strip(), "dose": m_dose.strip(), "note": m_note.strip()})
                        _add_audit("medication_add", m_name.strip())
                        save_current_session_profile()
                        st.success("İlaç kaydı eklendi.")
            with d2:
                with st.form("side_form_v3", clear_on_submit=True):
                    s_date = st.date_input("Yan Etki Tarihi", key="side_date_v3")
                    s_name = st.text_input("Yan Etki")
                    s_sev = st.selectbox("Şiddet", ["Hafif", "Orta", "Yüksek"])
                    s_note = st.text_area("Detay", key="side_note_v3")
                    if st.form_submit_button("Yan Etki Ekle") and s_name.strip():
                        st.session_state["side_effects"].append({"date": str(s_date), "effect": s_name.strip(), "severity": s_sev, "note": s_note.strip()})
                        _add_audit("side_effect_add", s_name.strip())
                        save_current_session_profile()
                        st.success("Yan etki kaydı eklendi.")

    if "E) Hatırlatıcı" in tabs:
        with tabs["E) Hatırlatıcı"]:
            st.subheader("Hatırlatıcı")
            with st.form("reminder_form_v3", clear_on_submit=True):
                r_date = st.date_input("Tarih", key="rem_date_v3")
                r_title = st.text_input("Başlık", key="rem_title_v3")
                r_note = st.text_area("Not", key="rem_note_v3")
                if st.form_submit_button("Hatırlatıcı Ekle") and r_title.strip():
                    st.session_state["reminders"].append({"date": str(r_date), "title": r_title.strip(), "note": r_note.strip()})
                    _add_audit("reminder_add", r_title.strip())
                    save_current_session_profile()
                    st.success("Hatırlatıcı eklendi.")

    if "F) Doktor Notları" in tabs:
        with tabs["F) Doktor Notları"]:
            st.subheader("Doktor Notları")
            with st.form("doctor_note_form_v3", clear_on_submit=True):
                n_date = st.date_input("Not Tarihi", key="doc_note_date_v3")
                n_branch = st.text_input("Branş / Bölüm")
                n_doctor = st.text_input("Doktor Adı")
                n_text = st.text_area("Klinik Not")
                if st.form_submit_button("Notu Kaydet") and n_text.strip():
                    st.session_state["doctor_notes"].append({"date": str(n_date), "branch": n_branch.strip(), "doctor": n_doctor.strip(), "note": n_text.strip()})
                    _sync_session_to_active_patient()
                    save_current_session_profile()
                    _add_audit("doctor_note_add", n_branch.strip() or "genel")
                    st.success("Doktor notu kaydedildi.")
            st.markdown("#### Dokuman Metadata Yukleme")
            up = st.file_uploader(
                "Rapor/PDF/Gorsel yukle (metadata kaydi)",
                type=["pdf", "png", "jpg", "jpeg", "txt", "doc", "docx"],
                key="doc_upload_v3",
            )
            if st.button("Dokumani kaydet", key="save_doc_meta_btn") and up is not None:
                up_size = int(getattr(up, "size", 0) or 0)
                if up_size > MAX_UPLOAD_BYTES:
                    st.error(f"Dosya boyutu cok buyuk. En fazla {MAX_UPLOAD_BYTES // (1024 * 1024)} MB yukleyebilirsiniz.")
                else:
                    pid = st.session_state.get("active_patient_id", "hasta")
                    meta = _store_uploaded_document(up, str(pid))
                    if meta is None:
                        st.error("Dosya kaydi basarisiz.")
                    else:
                        docs = st.session_state.get("documents", [])
                        if not isinstance(docs, list):
                            docs = []
                        docs.append(meta)
                        st.session_state["documents"] = docs[-200:]
                        _sync_session_to_active_patient()
                        save_current_session_profile()
                        _add_audit("document_uploaded", meta.get("name", ""))
                        st.success("Dokuman metadata kaydedildi.")
            docs = st.session_state.get("documents", [])
            if docs:
                st.dataframe(pd.DataFrame(docs[-20:]), use_container_width=True, hide_index=True)

    if "G) Veri Kalitesi & Audit" in tabs:
        with tabs["G) Veri Kalitesi & Audit"]:
            st.subheader("Veri Kalitesi & Audit")
            issues = _quality_checks()
            if issues:
                for i in issues:
                    st.warning(i)
            for rec in st.session_state.get("audits", [])[-80:]:
                st.write(f"- {rec.get('time')} | {rec.get('user')} | {rec.get('event')} | {rec.get('detail')}")

    if "H) Yedekleme" in tabs:
        with tabs["H) Yedekleme"]:
            st.subheader("Yedekleme")
            if not _can_export_system_backup(role):
                st.warning("Bu rolde tam yedek export izni yok.")
            else:
                backup_json = json.dumps({"patients": st.session_state.get("patients", {}), "active_patient_id": st.session_state.get("active_patient_id"), "audits": st.session_state.get("audits", [])}, ensure_ascii=False, indent=2).encode("utf-8")
                st.download_button("Tam Yedek İndir (JSON)", data=backup_json, file_name="dmd_backup.json", mime="application/json")

    if "I) Araştırma & Anonim Export" in tabs:
        with tabs["I) Araştırma & Anonim Export"]:
            st.subheader("Araştırma & Anonim Export")
            if not _can_export_research_data(role):
                st.warning("Bu rolde anonim araştırma export izni yok.")
            else:
                salt = _research_salt()
                if not salt:
                    st.warning("research_salt tanımlı değil. Export kapalı.")
                else:
                    rows = _build_research_export_data()
                    if not rows:
                        st.info("Export edilecek veri yok.")
                    else:
                        summary_df = pd.DataFrame(rows)
                        st.dataframe(summary_df, use_container_width=True, hide_index=True)
                        csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
                        json_bytes = json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8")
                        st.download_button("Anonim Export CSV", data=csv_bytes, file_name="dmd_research_export.csv", mime="text/csv")
                        st.download_button("Anonim Export JSON", data=json_bytes, file_name="dmd_research_export.json", mime="application/json")

# --- SAYFA 8: AI SORU-CEVAP ---
elif page == D['nav'][PAGE_AI]:
    st.title(D.get("ai_h", "AI Destekli Soru-Cevap"))
    st.caption("Bu bölüm OpenAI API anahtarı varsa çalışır. Yanıtlar bilgilendirme amaçlıdır.")

    if "ai_chat_history" not in st.session_state or not isinstance(st.session_state.get("ai_chat_history"), list):
        st.session_state["ai_chat_history"] = []
    if "openai_api_key" not in st.session_state:
        st.session_state["openai_api_key"] = ""

    with st.expander("AI Ayarları"):
        runtime_key_input = st.text_input(
            "OpenAI API Key",
            value=st.session_state.get("openai_api_key", ""),
            type="password",
            placeholder="sk-...",
            help="Boş bırakırsan önce st.secrets, sonra ortam değişkeni kullanılır.",
        )
        st.session_state["openai_api_key"] = (runtime_key_input or "").strip()
        has_key = bool(_get_openai_api_key())
        st.caption(f"API anahtar durumu: {'Hazır' if has_key else 'Tanımlı değil'}")
        if not has_key:
            st.warning("AI için API anahtarı tanımlı değil. Ayarlardan anahtar girin.")

    consent_ok = bool((st.session_state.get("privacy_settings", {}) or {}).get("consent_accepted", False))
    use_context = st.checkbox("Hasta bağlamını soruya ekle", value=consent_ok)
    question = st.text_area("Sorunuz", placeholder="Örn: 8 yaş, NSAA 22 olan hasta için takipte nelere dikkat edilmeli")

    cqa1, cqa2 = st.columns([1, 1])
    with cqa1:
        ask_btn = st.button("AI'ya Sor", use_container_width=True)
    with cqa2:
        if st.button("Sohbet Geçmişini Temizle", use_container_width=True):
            st.session_state["ai_chat_history"] = []
            st.rerun()

    if ask_btn:
        q_clean = (question or "").strip()
        if not q_clean:
            st.error("Lütfen bir soru yazın.")
        elif not _get_openai_api_key():
            st.error("OpenAI API anahtarı bulunamadı. AI Ayarları bölümünden anahtar girin.")
        else:
            ctx = ""
            if use_context and consent_ok:
                ctx = (
                    f"Kilo: {st.session_state.get('kilo')}\n"
                    f"Yaş: {st.session_state.get('yas')}\n"
                    f"NSAA: {st.session_state.get('nsaa_total')}/34\n"
                    f"PUL: {st.session_state.get('pul_score', 0)}\n"
                    f"Vignos: {st.session_state.get('vignos_score', 1)}\n"
                    f"Mod: {st.session_state.get('user_mode', 'Aile')}"
                )
            elif use_context and not consent_ok:
                st.warning("Hasta bağlamı için gizlilik onayı gerekli. KVKK bölümünden onay verin.")
            with st.spinner("AI yanıtı alınıyor..."):
                answer = ask_openai_medical_assistant(q_clean, ctx)
            st.session_state["ai_chat_history"].append(
                {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "q": q_clean,
                    "a": answer,
                }
            )
            st.session_state["ai_chat_history"] = st.session_state["ai_chat_history"][-20:]
            _add_audit("ai_question", q_clean[:120])

    if st.session_state["ai_chat_history"]:
        st.markdown("#### Son Sorular")
        for rec in reversed(st.session_state["ai_chat_history"][-10:]):
            with st.expander(f"{rec.get('time')} | {rec.get('q', '')[:80]}"):
                st.markdown(f"**Soru:** {rec.get('q', '')}")
                st.markdown(f"**Yanıt:** {rec.get('a', '')}")
# --- SAYFA 9: VİZYON & STRATEJİK LİDERLİK (ELITE DESIGN) ---
elif page == D['nav'][PAGE_VISION]:
    st.markdown("""
        <style>
        .vision-hero {
            background:
                radial-gradient(circle at 12% 18%, rgba(28,131,225,0.16), transparent 42%),
                radial-gradient(circle at 88% 20%, rgba(22,163,74,0.12), transparent 38%),
                #ffffff;
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 24px;
            padding: 36px 28px;
            box-shadow: 0 16px 40px rgba(148, 163, 184, 0.24);
            text-align: center;
            margin-bottom: 22px;
        }
        .vision-kicker {
            margin: 0 0 8px 0;
            font-size: 0.75rem;
            letter-spacing: 2px;
            color: #1c83e1;
            font-weight: 700;
        }
        .vision-title {
            margin: 0;
            font-size: clamp(1.9rem, 4vw, 3rem);
            line-height: 1.1;
            color: #0f172a;
            font-weight: 800;
            font-family: 'Poppins', sans-serif;
        }
        .vision-sub {
            margin: 12px auto 0 auto;
            max-width: 780px;
            color: #334155;
            font-size: 1.04rem;
            line-height: 1.7;
        }
        .vision-card {
            background: #ffffff;
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 20px;
            padding: 24px 20px;
            min-height: 300px;
            box-shadow: 0 10px 24px rgba(148, 163, 184, 0.22);
        }
        .vision-card h3 {
            margin: 0 0 10px 0;
            font-size: 1.15rem;
            color: #0f172a;
        }
        .vision-card p {
            margin: 0;
            color: #475569;
            line-height: 1.7;
            font-size: 0.95rem;
        }
        .vision-icon {
            font-size: 2.1rem;
            margin-bottom: 12px;
        }
        .vision-note {
            background: linear-gradient(90deg, rgba(28,131,225,0.08), rgba(16,185,129,0.08));
            border: 1px solid rgba(28,131,225,0.18);
            border-left: 8px solid #1c83e1;
            border-radius: 16px;
            padding: 18px 18px;
            margin-top: 16px;
        }
        .signature-panel {
            margin-top: 20px;
            background:
                linear-gradient(145deg, rgba(255,255,255,0.98), rgba(247,250,255,0.98));
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 26px;
            padding: 34px 22px;
            text-align: center;
            box-shadow: 0 14px 30px rgba(148, 163, 184, 0.28);
            position: relative;
            overflow: hidden;
        }
        .signature-panel::before {
            content: "";
            position: absolute;
            inset: -80px auto auto -90px;
            width: 220px;
            height: 220px;
            border-radius: 999px;
            background: rgba(28,131,225,0.08);
        }
        .signature-panel::after {
            content: "";
            position: absolute;
            inset: auto -70px -90px auto;
            width: 220px;
            height: 220px;
            border-radius: 999px;
            background: rgba(239,68,68,0.08);
        }
        .signature-content { position: relative; z-index: 1; }
        .signature-role {
            margin: 0;
            font-size: 0.74rem;
            letter-spacing: 2.6px;
            font-weight: 700;
            color: #1c83e1;
        }
        .signature-name {
            margin: 12px 0 8px 0;
            font-size: clamp(2rem, 6vw, 3.2rem);
            line-height: 1.1;
            font-weight: 900;
            color: #0f172a;
            font-family: 'Poppins', sans-serif;
        }
        .signature-quote {
            margin: 0;
            color: #334155;
            font-size: 1.05rem;
            font-style: italic;
        }

        /* ✅ NEW: email style */
        .signature-email{
            margin: 8px 0 14px 0;
            font-size: 0.92rem;
            font-weight: 700;
            color: #0f172a;
            opacity: 0.85;
        }
        .signature-email a{
            color: #1c83e1;
            text-decoration: none;
            border-bottom: 1px dashed rgba(28,131,225,0.45);
            padding-bottom: 2px;
        }
        .signature-email a:hover{
            opacity: 0.9;
        }

        .signature-tags {
            margin-top: 18px;
            display: flex;
            justify-content: center;
            gap: 8px;
            flex-wrap: wrap;
        }
        .signature-tag {
            border-radius: 999px;
            padding: 6px 12px;
            font-size: 0.73rem;
            font-weight: 700;
            border: 1px solid rgba(148,163,184,0.32);
            color: #0f172a;
            background: rgba(241,245,249,0.95);
        }
        .signature-foot {
            margin-top: 16px;
            font-size: 0.68rem;
            letter-spacing: 1.4px;
            color: #64748b;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="vision-hero">
            <p class="vision-kicker">STRATEJİK VİZYON</p>
            <h1 class="vision-title">Neurodegenerative Clinical Platform</h1>
            <p class="vision-sub">
                Nadir hastalık yönetiminde klinik doğruluk, erişilebilir dijital deneyim ve insan odaklı bakım
                standardını aynı platformda birleştiriyoruz.
            </p>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
            <div class="vision-card">
                <div class="vision-icon">🌍</div>
                <h3>Global Klinik Çerçeve</h3>
                <p>WDO ve TREAT-NMD rehberlerini tek akışta birleştirerek, ekiplerin kararlarını ülke bağımsız ve standardize biçimde destekliyoruz.</p>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
            <div class="vision-card">
                <div class="vision-icon">⚡</div>
                <h3>Anlık Müdahale Desteği</h3>
                <p>Acil anlarda kritik riskleri sade ve görünür bir formatta sunarak zaman kaybını azaltıyor, bakım kalitesini yükseltiyoruz.</p>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
            <div class="vision-card">
                <div class="vision-icon">🧬</div>
                <h3>Kişiselleştirilmiş Gelecek</h3>
                <p>Mutasyon temelli takip, tedavi yanıtı öngörüsü ve uzun dönem progresyon analitiği için yapay zeka odaklı altyapı geliştiriyoruz.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("""
        <div class="vision-note">
            <h4 style="margin:0 0 8px 0; color:#0f172a;"> Gizlilik ve Etik Protokolü</h4>
            <p style="margin:0; color:#334155; line-height:1.7;">
                Platform Privacy by Design yaklaşımıyla geliştirildi. Veriler yerel oturumda güvenle tutulur,
                zorunlu olmayan hiçbir üçüncü taraf paylaşımı yapılmaz. KVKK ve GDPR ilkeleri temel tasarım
                kriteridir.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # ✅ NEW: Role-based visibility for founder/signature panel
    role = st.session_state.get("user_role", "family")
    show_signature = role in {"admin", "doctor"}

    if show_signature:
        st.markdown("""
            <div class="signature-panel">
                <div class="signature-content">
                    <p class="signature-role">KURUCU VE BAŞ VİZYONER</p>
                    <h2 class="signature-name">BERFİN NİDA ÖZTÜRK</h2>
                    <p class="signature-email"><a href="mailto:berfinida@gmail.com">berfinida@gmail.com</a></p>
                    <p class="signature-quote">"Nadir yaşamları kodla güçlendiriyoruz."</p>
                    <div class="signature-tags">
                        <span class="signature-tag">DMD ODAKLI GELİŞTİRİCİ</span>
                        <span class="signature-tag">KLİNİK ÜRÜN TASARIMI</span>
                        <span class="signature-tag">GELİŞTİRİCİ</span>
                    </div>
                    <p class="signature-foot">Neurodegenerative Clinical Platform © 2026 TÜM HAKLARI SAKLIDIR</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Bu bölüm (kurucu imzası) yalnızca doktor/admin görünümünde gösterilir.")
# --- CALL TO ACTION (ROLE-BASED PREMIUM FOOTER) ---
st.markdown("""
<style>
.cta-panel{
    margin-top: 26px;
    padding: 28px 22px;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(28,131,225,0.12), rgba(16,185,129,0.10));
    border: 1px solid rgba(28,131,225,0.22);
    text-align: center;
    box-shadow: 0 14px 32px rgba(148,163,184,0.24);
}
.cta-title{
    margin:0;
    font-size:1.35rem;
    font-weight:800;
    color:#0f172a;
}
.cta-sub{
    margin:10px auto 18px auto;
    max-width:680px;
    color:#334155;
    font-size:0.95rem;
    line-height:1.7;
}
</style>
<div class="cta-panel">
    <h3 class="cta-title">🚀 Birlikte Daha Güçlü Bir Klinik Gelecek</h3>
    <p class="cta-sub">
        DMD Guardian Global Pro sürekli gelişen bir klinik platformdur.
        Rolünüze göre en doğru kanaldan geri bildirim ve iş birliği akışını açıyoruz.
    </p>
</div>
""", unsafe_allow_html=True)

role = st.session_state.get("user_role", "family")

# Role'a göre buton seti
if role in {"doctor", "admin"}:
    b1, b2, b3 = st.columns(3)

    with b1:
        if st.button(" Feedback Gönder"):
            st.success("Feedback kaydı alındı (demo). Yakında form + kayıt sistemi eklenecek.")
    with b2:
        if st.button("Partnership"):
            st.info("İş birliği için iletişim: berfinida@gmail.com")
    with b3:
        if st.button("Platformu Destekle"):
            st.success("Desteğiniz için teşekkür ederiz 🙌")

elif role == "researcher":
    b1, b2, b3 = st.columns(3)

    with b1:
        if st.button("Research Collaboration"):
            st.info("Araştırma iş birliği için: berfinida@gmail.com")
    with b2:
        if st.button("Feedback Gönder"):
            st.success("Feedback kaydı alındı (demo).")
    with b3:
        st.caption("Bu mod anonim araştırma odaklıdır.")

else:  # family (default)
    b1, b2, b3 = st.columns(3)

    with b1:
        if st.button("Feedback Gönder"):
            st.success("Feedback kaydı alındı (demo). Yakında form + kayıt sistemi eklenecek.")
    with b2:
        if st.button("Platformu Destekle"):
            st.success("Desteğiniz için teşekkür ederiz 🙌")
    with b3:
        st.caption("")






