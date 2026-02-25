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
import importlib
from dataclasses import dataclass, field
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from datetime import datetime, timedelta, timezone
px = None
try:
    # Optional dependency: load dynamically to avoid static analyzer missing-import warnings.
    px = importlib.import_module("plotly.express")
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

A4 = (595.2755905511812, 841.8897637795277)  # fallback A4 size in points
canvas = None
try:
    # Optional dependency: load dynamically to avoid static analyzer missing-source warnings.
    _reportlab_pagesizes = importlib.import_module("reportlab.lib.pagesizes")
    _reportlab_canvas = importlib.import_module("reportlab.pdfgen.canvas")
    A4 = _reportlab_pagesizes.A4
    canvas = _reportlab_canvas
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
    # Optional dependency: load dynamically so static analyzers don't flag missing import.
    bcrypt = importlib.import_module("bcrypt")
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
    page_icon="N",
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
    except (OSError, sqlite3.Error):
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
    except sqlite3.Error as e:
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
    except sqlite3.Error:
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
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def _read_json_file(path: Path, default):
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as e:
        _debug_log(f"JSON read failed: {path}", e)
    return default


def _write_json_file(path: Path, data) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except (OSError, TypeError, ValueError) as e:
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
    except (json.JSONDecodeError, TypeError, ValueError):
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
    except (json.JSONDecodeError, TypeError, ValueError):
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
    except sqlite3.Error as e:
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


@st.cache_data(ttl=600)
def fetch_disease_news(query: str, lang: str = "TR", limit: int = 20) -> list[dict]:
    q = str(query or "").strip() or "neurodegenerative disease"
    rss_url = (
        "https://news.google.com/rss/search?q="
        + quote_plus(q)
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
                items.append({"title": title, "link": link, "source": source, "published": published})
            if len(items) >= limit:
                break
    except Exception:
        return []

    seen = set()
    out = []
    for it in items:
        key = (it["title"], it["link"])
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


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


def _render_ai_key_settings(scope_key: str = "global") -> bool:
    if "openai_api_key" not in st.session_state:
        st.session_state["openai_api_key"] = ""
    with st.expander("AI Ayarları"):
        runtime_key_input = st.text_input(
            "OpenAI API Key",
            value=st.session_state.get("openai_api_key", ""),
            type="password",
            placeholder="sk-...",
            key=f"openai_api_key_input::{scope_key}",
            help="Boş bırakırsan önce st.secrets, sonra ortam değişkeni kullanılır.",
        )
        st.session_state["openai_api_key"] = (runtime_key_input or "").strip()
        has_key = bool(_get_openai_api_key())
        st.caption(f"API anahtar durumu: {'Hazır' if has_key else 'Tanımlı değil'}")
        if st.button("AI anahtarını temizle", key=f"clear_openai_api_key::{scope_key}", use_container_width=True):
            st.session_state["openai_api_key"] = ""
            st.rerun()
        if not has_key:
            st.warning("AI için API anahtarı tanımlı değil. Ayarlardan anahtar girin.")
    return bool(_get_openai_api_key())


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
            except (json.JSONDecodeError, TypeError, ValueError):
                payload = {}
            out[user] = payload if isinstance(payload, dict) else {}
        return out
    except sqlite3.Error:
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
    except (sqlite3.Error, json.JSONDecodeError, TypeError, ValueError):
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
        except sqlite3.Error as e:
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
    if "neuro_dynamic_records" not in st.session_state or not isinstance(st.session_state.get("neuro_dynamic_records"), dict):
        st.session_state["neuro_dynamic_records"] = {}
    if "neuro_workspace_notes" not in st.session_state or not isinstance(st.session_state.get("neuro_workspace_notes"), dict):
        st.session_state["neuro_workspace_notes"] = {}


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
        "neuro_dynamic_records": st.session_state.get("neuro_dynamic_records", {}),
        "neuro_workspace_notes": st.session_state.get("neuro_workspace_notes", {}),
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
        st.session_state["neuro_dynamic_records"] = profile.get("neuro_dynamic_records", st.session_state.get("neuro_dynamic_records", {}))
        st.session_state["neuro_workspace_notes"] = profile.get("neuro_workspace_notes", st.session_state.get("neuro_workspace_notes", {}))
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
    if st.session_state.get("_suppress_autosave_once", False):
        st.session_state["_suppress_autosave_once"] = False
    else:
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

WORKSPACE_I18N_REPLACE = {
    "EN": {
        "Klinik Vizyon": "Clinical Vision",
        "Stratejik Odaklar": "Strategic Focus Areas",
        "2026 Hedef Metrikleri": "2026 Target Metrics",
        "Hedef:": "Target:",
        "Durum:": "Status:",
        "Klinik Yük": "Clinical Burden",
        "Test Toplam": "Total Test Score",
        "Test Modeli": "Test Model",
        "Takip aralığını bireyselleştir (risk parametrelerine göre)": "Personalize follow-up interval (based on risk parameters)",
        "Önerilen bir sonraki kontrol:": "Recommended next follow-up:",
        "içinde.": "within.",
        "Bu hastalık için tam ölçekli test modeli tanımlı değil.": "No full-scale test model is defined for this disease.",
        "Bu hastalık için hedef metrik seti tanımlı değil.": "No target metric set is defined for this disease.",
        "Bu bölümdeki skorlar izlem amaçlıdır; tanı ve tedavi kararı tek başına bu testten verilmez.": "Scores in this section are for follow-up support; diagnosis and treatment decisions must not rely on this test alone.",
        "Vizyon sayfası ürün yönü ve klinik kalite hedeflerini özetler; tanı/tedavi kararı yerine geçmez.": "The vision page summarizes product direction and clinical quality goals; it does not replace diagnosis/treatment decisions.",
        "Hedefe Uygun": "On Target",
        "Dikkat": "Caution",
        "Kritik": "Critical",
        "Bilgi yok": "No data",
        "Acil Uyarı": "Emergency Notice",
        "Tam Ölçekli Test": "Full-Scale Test",
        "Tam ölçekli test kaydedildi.": "Full-scale test saved.",
    },
    "DE": {
        "Klinik Vizyon": "Klinische Vision",
        "Stratejik Odaklar": "Strategische Schwerpunkte",
        "2026 Hedef Metrikleri": "Zielmetriken 2026",
        "Hedef:": "Ziel:",
        "Durum:": "Status:",
        "Klinik Yük": "Klinische Belastung",
        "Test Toplam": "Gesamttestwert",
        "Test Modeli": "Testmodell",
        "Takip aralığını bireyselleştir (risk parametrelerine göre)": "Kontrollintervall personalisieren (nach Risikoparametern)",
        "Önerilen bir sonraki kontrol:": "Empfohlene nächste Kontrolle:",
        "içinde.": "innerhalb von.",
        "Bu hastalık için tam ölçekli test modeli tanımlı değil.": "Für diese Erkrankung ist kein Volltest-Modell definiert.",
        "Bu hastalık için hedef metrik seti tanımlı değil.": "Für diese Erkrankung ist kein Zielmetriksatz definiert.",
        "Bu bölümdeki skorlar izlem amaçlıdır; tanı ve tedavi kararı tek başına bu testten verilmez.": "Die Werte in diesem Abschnitt dienen der Verlaufskontrolle; Diagnose- und Therapieentscheidungen dürfen nicht allein darauf basieren.",
        "Vizyon sayfası ürün yönü ve klinik kalite hedeflerini özetler; tanı/tedavi kararı yerine geçmez.": "Die Visionsseite fasst Produktausrichtung und klinische Qualitätsziele zusammen; sie ersetzt keine Diagnose-/Therapieentscheidung.",
        "Hedefe Uygun": "Ziel erreicht",
        "Dikkat": "Achtung",
        "Kritik": "Kritisch",
        "Bilgi yok": "Keine Daten",
        "Acil Uyarı": "Notfallhinweis",
        "Tam Ölçekli Test": "Vollständiger Test",
        "Tam ölçekli test kaydedildi.": "Vollständiger Test gespeichert.",
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
        ("", ""),
        ("", ""),
        ("", ""),
        ("", ""),
        ("⬇", "⬇"),
        ("⬆", "⬆"),
        ("", ""),
        ("", ""),
        ("", ""),
        ("", ""),
        ("", ""),
        ("", ""),
        ("", ""),
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
    raw.update(WORKSPACE_I18N_REPLACE.get(lang, {}))
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
    st.session_state["neuro_dynamic_records"] = {}
    st.session_state["neuro_workspace_notes"] = {}
    st.session_state["profile_loaded_for"] = None
    st.session_state["_last_profile_sig"] = ""
    st.session_state["_last_profile_save_ts"] = 0.0
    # Bu turda autosave'in silinen profili geri yazmasını engelle.
    st.session_state["_suppress_autosave_once"] = True
    return ok_local


WORKSPACE_DMD = "DMD (Duchenne Musküler Distrofi)"
WORKSPACE_ALS = "ALS (Amyotrofik Lateral Skleroz)"
WORKSPACE_ALZHEIMER = "Alzheimer Hastalığı"
WORKSPACE_PARKINSON = "Parkinson Hastalığı"
WORKSPACE_HUNTINGTON = "Huntington Hastalığı"
WORKSPACE_LEWY = "Lewy Cisimcikli Demans"
WORKSPACE_FTD = "Frontotemporal Demans (FTD)"
WORKSPACE_SMA = "Spinal Müsküler Atrofi (SMA)"

WORKSPACE_OPTIONS = [
    WORKSPACE_DMD,
    WORKSPACE_ALS,
    WORKSPACE_ALZHEIMER,
    WORKSPACE_PARKINSON,
    WORKSPACE_HUNTINGTON,
    WORKSPACE_LEWY,
    WORKSPACE_FTD,
    WORKSPACE_SMA,
]

# Ortak nörodejeneratif veri kolonları (DMD disi moduller icin hazir sema)
NEURO_COMMON_DATA_COLUMNS = [
    "patient_id",
    "visit_date",
    "age",
    "sex",
    "onset_age",
    "disease_duration_years",
    "functional_score",
    "cognitive_score",
    "motor_score",
    "respiratory_score",
    "notes",
    "updated_at",
]
NEURO_DISEASE_DATA_SCHEMA = {
    WORKSPACE_ALS: list(NEURO_COMMON_DATA_COLUMNS),
    WORKSPACE_ALZHEIMER: list(NEURO_COMMON_DATA_COLUMNS),
    WORKSPACE_PARKINSON: list(NEURO_COMMON_DATA_COLUMNS),
    WORKSPACE_HUNTINGTON: list(NEURO_COMMON_DATA_COLUMNS),
    WORKSPACE_LEWY: list(NEURO_COMMON_DATA_COLUMNS),
    WORKSPACE_FTD: list(NEURO_COMMON_DATA_COLUMNS),
    WORKSPACE_SMA: list(NEURO_COMMON_DATA_COLUMNS),
}

# Leiden uyumlu DMD exon faz haritasi (Dp427, exon 1..79)
# Her exon icin start/end fazlari (0,1,2) tutulur.
LEIDEN_DMD_EXON_PHASES: dict[int, dict[str, int]] = {
    1: {"start": 0, "end": 1}, 2: {"start": 1, "end": 0}, 3: {"start": 0, "end": 0},
    4: {"start": 0, "end": 0}, 5: {"start": 0, "end": 0}, 6: {"start": 0, "end": 2},
    7: {"start": 2, "end": 1}, 8: {"start": 1, "end": 0}, 9: {"start": 0, "end": 0},
    10: {"start": 0, "end": 0}, 11: {"start": 0, "end": 2}, 12: {"start": 2, "end": 0},
    13: {"start": 0, "end": 0}, 14: {"start": 0, "end": 0}, 15: {"start": 0, "end": 0},
    16: {"start": 0, "end": 0}, 17: {"start": 0, "end": 2}, 18: {"start": 2, "end": 0},
    19: {"start": 0, "end": 1}, 20: {"start": 1, "end": 0}, 21: {"start": 0, "end": 1},
    22: {"start": 1, "end": 0}, 23: {"start": 0, "end": 0}, 24: {"start": 0, "end": 0},
    25: {"start": 0, "end": 0}, 26: {"start": 0, "end": 0}, 27: {"start": 0, "end": 0},
    28: {"start": 0, "end": 0}, 29: {"start": 0, "end": 0}, 30: {"start": 0, "end": 0},
    31: {"start": 0, "end": 0}, 32: {"start": 0, "end": 0}, 33: {"start": 0, "end": 0},
    34: {"start": 0, "end": 0}, 35: {"start": 0, "end": 0}, 36: {"start": 0, "end": 0},
    37: {"start": 0, "end": 0}, 38: {"start": 0, "end": 0}, 39: {"start": 0, "end": 0},
    40: {"start": 0, "end": 0}, 41: {"start": 0, "end": 0}, 42: {"start": 0, "end": 0},
    43: {"start": 0, "end": 2}, 44: {"start": 2, "end": 0}, 45: {"start": 0, "end": 2},
    46: {"start": 2, "end": 0}, 47: {"start": 0, "end": 0}, 48: {"start": 0, "end": 0},
    49: {"start": 0, "end": 0}, 50: {"start": 0, "end": 1}, 51: {"start": 1, "end": 0},
    52: {"start": 0, "end": 1}, 53: {"start": 1, "end": 0}, 54: {"start": 0, "end": 2},
    55: {"start": 2, "end": 0}, 56: {"start": 0, "end": 2}, 57: {"start": 2, "end": 0},
    58: {"start": 0, "end": 1}, 59: {"start": 1, "end": 0}, 60: {"start": 0, "end": 0},
    61: {"start": 0, "end": 1}, 62: {"start": 1, "end": 2}, 63: {"start": 2, "end": 1},
    64: {"start": 1, "end": 1}, 65: {"start": 1, "end": 2}, 66: {"start": 2, "end": 1},
    67: {"start": 1, "end": 0}, 68: {"start": 0, "end": 2}, 69: {"start": 2, "end": 0},
    70: {"start": 0, "end": 2}, 71: {"start": 2, "end": 2}, 72: {"start": 2, "end": 2},
    73: {"start": 2, "end": 2}, 74: {"start": 2, "end": 2}, 75: {"start": 2, "end": 0},
    76: {"start": 0, "end": 1}, 77: {"start": 1, "end": 1}, 78: {"start": 1, "end": 0},
    79: {"start": 0, "end": 0},
}

# Geriye donuk adlandirma uyumu.
exon_phases = LEIDEN_DMD_EXON_PHASES


def _validate_exon_phase_map(phase_map: dict[int, dict[str, int]]) -> bool:
    if not isinstance(phase_map, dict) or len(phase_map) != 79:
        return False
    for exon in range(1, 80):
        row = phase_map.get(exon)
        if not isinstance(row, dict):
            return False
        s = row.get("start")
        e = row.get("end")
        if s not in (0, 1, 2) or e not in (0, 1, 2):
            return False
    return True


def _normalize_deleted_exons(deleted_exons: list[int]) -> list[int]:
    return sorted({int(x) for x in (deleted_exons or []) if 1 <= int(x) <= 79})


def _contiguous_blocks(exons: list[int]) -> list[tuple[int, int]]:
    normalized = _normalize_deleted_exons(exons)
    if not normalized:
        return []
    blocks: list[tuple[int, int]] = []
    block_start = normalized[0]
    block_end = normalized[0]
    for exon in normalized[1:]:
        if exon == block_end + 1:
            block_end = exon
        else:
            blocks.append((block_start, block_end))
            block_start = exon
            block_end = exon
    blocks.append((block_start, block_end))
    return blocks


def _analyze_deleted_exon_block(start_exon: int, end_exon: int) -> dict:
    prev_exon = start_exon - 1
    next_exon = end_exon + 1
    prev_end_phase = int(LEIDEN_DMD_EXON_PHASES.get(prev_exon, {}).get("end", 0))
    next_start_phase = int(LEIDEN_DMD_EXON_PHASES.get(next_exon, {}).get("start", 0))
    # Kural: silinen bloğun solundaki end phase ile sağındaki start phase eslesirse in-frame.
    frame_ok = (prev_end_phase == next_start_phase)
    return {
        "start": int(start_exon),
        "end": int(end_exon),
        "prev_exon": prev_exon if prev_exon >= 1 else None,
        "next_exon": next_exon if next_exon <= 79 else None,
        "left_end_phase": prev_end_phase,
        "right_start_phase": next_start_phase,
        "ok": bool(frame_ok),
    }


def _analyze_deleted_exons(deleted_exons: list[int]) -> dict:
    exons = _normalize_deleted_exons(deleted_exons)
    if not exons:
        return {"ok": None, "exons": [], "blocks": []}
    blocks = _contiguous_blocks(exons)
    block_rows = [_analyze_deleted_exon_block(s, e) for s, e in blocks]
    return {"ok": all(r["ok"] for r in block_rows), "exons": exons, "blocks": block_rows}


def _build_dna_track_df(deleted_exons: list[int]) -> pd.DataFrame:
    deleted = set(_normalize_deleted_exons(deleted_exons))
    return pd.DataFrame(
        {
            "x": list(range(1, 80)),
            "y": [1] * 79,
            "Durum": ["Silinmiş" if i in deleted else "Mevcut" for i in range(1, 80)],
        }
    )


def dmd_dashboard() -> None:
    st.subheader("DMD Genetik Analiz Modülü")
    _clinical_evidence_caption("Karar-Destek", "Reading-frame analizi olası fenotipi destekler; kesin klinik yorum için genetik ve klinik bulgular birlikte değerlendirilir.")
    if not _validate_exon_phase_map(LEIDEN_DMD_EXON_PHASES):
        st.error("Faz veritabanı hatalı: 79 exon start/end (0,1,2) formatı doğrulanamadı.")
        return
    selected_exons = st.multiselect(
        "Silinen Eksonlar (1-79)",
        options=list(range(1, 80)),
        default=st.session_state.get("dmd_deleted_exons", []),
        key="dmd_deleted_exons",
    )
    analysis = _analyze_deleted_exons(selected_exons)
    exons = analysis["exons"]
    if analysis["ok"] is None:
        st.info("Frame-shift analizi için en az bir ekson seçin.")
    else:
        if analysis["ok"]:
            st.success(
                "Analiz Sonucu: In-frame (Okuma Çerçevesi Korunmuş). Klinik tablo muhtemelen Becker (BMD) ile uyumludur."
            )
        else:
            st.error(
                "Analiz Sonucu: Out-of-frame (Okuma Çerçevesi Bozulmuş). Klinik tablo muhtemelen Duchenne (DMD) ile uyumludur."
            )
        for r in analysis["blocks"]:
            st.caption(
                f"Blok {r['start']}-{r['end']} | Önceki ekson: {r['prev_exon'] or '5-prime sınır'} "
                f"(end phase={r['left_end_phase']}) | "
                f"Sonraki ekson: {r['next_exon'] or '3-prime sınır'} "
                f"(start phase={r['right_start_phase']}) | "
                f"Durum: {'In-frame' if r['ok'] else 'Out-of-frame'}"
            )

    viz_df = _build_dna_track_df(exons)
    if PLOTLY_OK:
        fig = px.scatter(
            viz_df,
            x="x",
            y="y",
            color="Durum",
            color_discrete_map={"Silinmiş": "#ef4444", "Mevcut": "#22c55e"},
            title="Ekson Şeridi (1-79)",
        )
        fig.update_traces(marker_symbol="square", marker_size=18)
        fig.update_layout(height=220, xaxis_title="Ekson", yaxis_title="", showlegend=True)
        fig.update_yaxes(visible=False, showticklabels=False)
        fig.update_xaxes(tickmode="linear", dtick=2, range=[0.5, 79.5])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Plotly bulunamadı; görsel özet tablo olarak gösteriliyor.")
        st.dataframe(viz_df, use_container_width=True, hide_index=True)


NEURO_DYNAMIC_CONFIG = {
    WORKSPACE_ALS: {
        "title": "ALS Klinik İzlem",
        "desc": "Motor, solunum ve fonksiyonel kayıp takibi",
        "fields": [
            {"key": "alsfrs_r", "label": "ALSFRS-R", "min": 0, "max": 48, "default": 40, "better_high": True},
            {"key": "fvc_pct", "label": "FVC (%)", "min": 0, "max": 150, "default": 85, "better_high": True},
            {"key": "bulbar_score", "label": "Bulbar Etkilenim", "min": 0, "max": 10, "default": 2, "better_high": False},
        ],
    },
    WORKSPACE_ALZHEIMER: {
        "title": "Alzheimer Klinik İzlem",
        "desc": "Bilişsel ve fonksiyonel ilerleme takibi",
        "fields": [
            {"key": "mmse", "label": "MMSE", "min": 0, "max": 30, "default": 24, "better_high": True},
            {"key": "adas_cog", "label": "ADAS-Cog", "min": 0, "max": 70, "default": 20, "better_high": False},
            {"key": "adl_loss", "label": "ADL Kayıp Düzeyi", "min": 0, "max": 10, "default": 2, "better_high": False},
        ],
    },
    WORKSPACE_PARKINSON: {
        "title": "Parkinson Klinik İzlem",
        "desc": "Motor şiddet, denge ve günlük yaşam etkisi",
        "fields": [
            {"key": "updrs_total", "label": "UPDRS Toplam", "min": 0, "max": 199, "default": 35, "better_high": False},
            {"key": "hn_stage", "label": "Hoehn-Yahr Evresi", "min": 1, "max": 5, "default": 2, "better_high": False},
            {"key": "fall_risk", "label": "Düşme Riski", "min": 0, "max": 10, "default": 3, "better_high": False},
        ],
    },
    WORKSPACE_HUNTINGTON: {
        "title": "Huntington Klinik İzlem",
        "desc": "Motor-kognitif-fonksiyonel etkilenim takibi",
        "fields": [
            {"key": "uhdrs_motor", "label": "UHDRS Motor", "min": 0, "max": 124, "default": 28, "better_high": False},
            {"key": "sdmt", "label": "SDMT", "min": 0, "max": 110, "default": 40, "better_high": True},
            {"key": "tfc", "label": "TFC", "min": 0, "max": 13, "default": 9, "better_high": True},
        ],
    },
    WORKSPACE_LEWY: {
        "title": "Lewy Cisimcikli Demans Klinik İzlem",
        "desc": "Bilişsel dalgalanma, nörops?kiyatrik ve parkinsonizm etkisi",
        "fields": [
            {"key": "mmse", "label": "MMSE", "min": 0, "max": 30, "default": 22, "better_high": True},
            {"key": "hallucination", "label": "Halüsinasyon Şiddeti", "min": 0, "max": 12, "default": 3, "better_high": False},
            {"key": "fluctuation", "label": "Bilişsel Dalgalanma", "min": 0, "max": 10, "default": 4, "better_high": False},
        ],
    },
    WORKSPACE_FTD: {
        "title": "Frontotemporal Demans Klinik İzlem",
        "desc": "Davranış, dil ve yürütücü işlev etkilenimi",
        "fields": [
            {"key": "fbi", "label": "Frontal Behavioral Inventory", "min": 0, "max": 72, "default": 20, "better_high": False},
            {"key": "language_impairment", "label": "Dil Etkilenimi", "min": 0, "max": 10, "default": 3, "better_high": False},
            {"key": "adl_loss", "label": "ADL Kayıp Düzeyi", "min": 0, "max": 10, "default": 3, "better_high": False},
        ],
    },
    WORKSPACE_SMA: {
        "title": "SMA Klinik İzlem",
        "desc": "Motor fonksiyon ve solunum performansı",
        "fields": [
            {"key": "hfmse", "label": "HFMSE", "min": 0, "max": 66, "default": 38, "better_high": True},
            {"key": "rulm", "label": "RULM", "min": 0, "max": 37, "default": 24, "better_high": True},
            {"key": "fvc_pct", "label": "FVC (%)", "min": 0, "max": 150, "default": 80, "better_high": True},
        ],
    },
}


def _sync_neuro_schema_from_config() -> None:
    for disease, cfg in NEURO_DYNAMIC_CONFIG.items():
        base_cols = list(NEURO_COMMON_DATA_COLUMNS)
        for f in cfg.get("fields", []):
            k = str(f.get("key", "")).strip()
            if k and k not in base_cols:
                base_cols.append(k)
        if "progression_index" not in base_cols:
            base_cols.append("progression_index")
        NEURO_DISEASE_DATA_SCHEMA[disease] = base_cols


_sync_neuro_schema_from_config()


def _active_patient_scope_key() -> str:
    pid = str(st.session_state.get("active_patient_id", "")).strip()
    return pid or "_global"


def _neuro_input_state_key(hastalik_turu: str) -> str:
    return f"neuro_inputs::{_active_patient_scope_key()}::{hastalik_turu}"


def _neuro_history_state_key(hastalik_turu: str) -> str:
    return f"neuro_history::{_active_patient_scope_key()}::{hastalik_turu}"


def _ensure_neuro_dynamic_state(hastalik_turu: str) -> None:
    cfg = NEURO_DYNAMIC_CONFIG.get(hastalik_turu, {})
    fields = cfg.get("fields", [])
    state_key = _neuro_input_state_key(hastalik_turu)
    history_key = _neuro_history_state_key(hastalik_turu)
    if state_key not in st.session_state or not isinstance(st.session_state.get(state_key), dict):
        st.session_state[state_key] = {}
    for f in fields:
        k = str(f.get("key"))
        dv = float(f.get("default", 0))
        st.session_state[state_key].setdefault(k, dv)
    if history_key not in st.session_state or not isinstance(st.session_state.get(history_key), list):
        st.session_state[history_key] = []
    if not st.session_state.get(history_key):
        store = st.session_state.get("neuro_dynamic_records", {})
        if isinstance(store, dict):
            patient_key = _active_patient_scope_key()
            saved_hist = []
            patient_bucket = store.get(patient_key, {})
            if isinstance(patient_bucket, dict):
                saved_hist = patient_bucket.get(hastalik_turu, [])
            # Legacy fallback: flat disease -> history
            if not saved_hist:
                saved_hist = store.get(hastalik_turu, [])
            if isinstance(saved_hist, list):
                st.session_state[history_key] = saved_hist[-120:]


def _metric_burden(value: float, min_v: float, max_v: float, better_high: bool) -> float:
    span = max(float(max_v) - float(min_v), 1.0)
    norm = (float(value) - float(min_v)) / span
    norm = max(0.0, min(norm, 1.0))
    return (1.0 - norm) * 100.0 if better_high else norm * 100.0


def analiz_yap(hastalik_turu: str, girdiler: dict) -> dict:
    cfg = NEURO_DYNAMIC_CONFIG.get(hastalik_turu, {})
    fields = cfg.get("fields", [])
    if not fields:
        return {"progression_index": 0.0, "stage": "NA", "metric_rows": []}
    metric_rows = []
    burdens = []
    for f in fields:
        key = str(f.get("key"))
        val = float(girdiler.get(key, f.get("default", 0)))
        min_v = float(f.get("min", 0))
        max_v = float(f.get("max", 100))
        better_high = bool(f.get("better_high", True))
        burden = _metric_burden(val, min_v, max_v, better_high)
        burdens.append(burden)
        metric_rows.append({"metric": str(f.get("label", key)), "value": val, "burden": round(burden, 2)})
    progression_index = round(sum(burdens) / len(burdens), 2) if burdens else 0.0
    if progression_index < 33:
        stage = "Düşük Etkilenim"
    elif progression_index < 66:
        stage = "Orta Etkilenim"
    else:
        stage = "Yüksek Etkilenim"
    return {"progression_index": progression_index, "stage": stage, "metric_rows": metric_rows}


def _render_neuro_charts(hastalik_turu: str, analiz: dict) -> None:
    metric_rows = analiz.get("metric_rows", [])
    history = st.session_state.get(_neuro_history_state_key(hastalik_turu), [])
    if PLOTLY_OK and metric_rows:
        cur_df = pd.DataFrame(metric_rows)
        fig_cur = px.bar(
            cur_df,
            x="metric",
            y="burden",
            title="Güncel Kayıp Yükü Dağılımı",
            color="burden",
            color_continuous_scale="RdYlGn_r",
        )
        fig_cur.update_layout(height=320, yaxis_title="Kayıp Yükü (%)", xaxis_title="")
        st.plotly_chart(fig_cur, use_container_width=True)
    if history:
        hist_df = pd.DataFrame(history)
        if "time" in hist_df.columns and "progression_index" in hist_df.columns:
            if PLOTLY_OK:
                fig_line = px.line(
                    hist_df,
                    x="time",
                    y="progression_index",
                    markers=True,
                    title="Progresyon Eğrisi",
                )
                fig_line.update_layout(height=320, yaxis_title="Progresyon İndeksi")
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.line_chart(hist_df.set_index("time")[["progression_index"]], use_container_width=True)


def _save_neuro_history_record(hastalik_turu: str, rec: dict) -> None:
    hist_key = _neuro_history_state_key(hastalik_turu)
    hist = st.session_state.get(hist_key, [])
    if not isinstance(hist, list):
        hist = []
    hist.append(rec if isinstance(rec, dict) else {})
    st.session_state[hist_key] = hist[-120:]
    store = st.session_state.get("neuro_dynamic_records", {})
    if not isinstance(store, dict):
        store = {}
    patient_key = _active_patient_scope_key()
    patient_bucket = store.get(patient_key, {})
    if not isinstance(patient_bucket, dict):
        patient_bucket = {}
    patient_bucket[hastalik_turu] = st.session_state[hist_key]
    store[patient_key] = patient_bucket
    st.session_state["neuro_dynamic_records"] = store


def _render_dynamic_neuro_dashboard(hastalik_turu: str) -> None:
    cfg = NEURO_DYNAMIC_CONFIG.get(hastalik_turu)
    if not cfg:
        st.warning("Bu hastalık için dinamik modül çok yakında eklenecektir.")
        return
    _ensure_neuro_dynamic_state(hastalik_turu)
    state_key = _neuro_input_state_key(hastalik_turu)
    state_vals = st.session_state.get(state_key, {})
    st.subheader(cfg.get("title", "Klinik İzlem"))
    st.caption(cfg.get("desc", ""))

    fields = cfg.get("fields", [])
    cols = st.columns(len(fields)) if fields else []
    for i, f in enumerate(fields):
        with cols[i]:
            key = str(f.get("key"))
            ui_key = f"{_active_patient_scope_key()}::{hastalik_turu}::{key}"
            val = st.number_input(
                str(f.get("label", key)),
                min_value=float(f.get("min", 0)),
                max_value=float(f.get("max", 100)),
                value=float(state_vals.get(key, f.get("default", 0))),
                key=ui_key,
            )
            state_vals[key] = float(val)
    st.session_state[state_key] = state_vals

    analiz = analiz_yap(hastalik_turu, state_vals)
    pidx = float(analiz.get("progression_index", 0.0))
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Progresyon İndeksi", f"{pidx:.1f} / 100")
    with c2:
        st.metric("Klinik Etkilenim", str(analiz.get("stage", "-")))
    if pidx < 33:
        st.success("Klinik tablo düşük etkilenim düzeyinde görünüyor.")
    elif pidx < 66:
        st.warning("Klinik tablo orta etkilenim düzeyinde görünüyor.")
    else:
        st.error("Klinik tablo yüksek etkilenim düzeyinde görünüyor.")

    if st.button("Ölçümü Kaydet", key=f"save_neuro_{hastalik_turu}", use_container_width=True):
        rec = {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "progression_index": pidx, **state_vals}
        _save_neuro_history_record(hastalik_turu, rec)
        save_current_session_profile()
        st.success("Ölçüm kaydedildi.")

    _render_neuro_charts(hastalik_turu, analiz)


ALSFRS_ITEMS = [
    "1. Konuşma",
    "2. Salivasyon",
    "3. Yutma",
    "4. El yazısı",
    "5. Yemek kesme / kap kullanımı",
    "6. Giyinme ve hijyen",
    "7. Yatakta dönme / örtü düzenleme",
    "8. Yürüme",
    "9. Merdiven çıkma",
    "10. Dispne",
    "11. Ortopne",
    "12. Solunum desteği",
]


def _alsfrs_state_key() -> str:
    return f"alsfrs_items::{_active_patient_scope_key()}"


def _ensure_alsfrs_state() -> None:
    key = _alsfrs_state_key()
    values = st.session_state.get(key, [])
    if not isinstance(values, list) or len(values) != 12:
        values = [4] * 12
    values = [max(0, min(4, int(v))) for v in values]
    st.session_state[key] = values


def _als_domain_scores(item_scores: list[int]) -> dict[str, int]:
    s = [max(0, min(4, int(v))) for v in (item_scores or [0] * 12)]
    return {
        "Bulber": sum(s[0:3]),
        "İnce Motor": sum(s[3:6]),
        "Kaba Motor": sum(s[6:9]),
        "Solunum": sum(s[9:12]),
    }


def _als_progression_summary(history: list[dict]) -> tuple[float | None, str]:
    if not isinstance(history, list) or len(history) < 2:
        return (None, "Yeterli ALSFRS-R geçmişi yok.")
    df = pd.DataFrame(history)
    if "alsfrs_r" not in df.columns:
        return (None, "Yeterli ALSFRS-R geçmişi yok.")
    df = df.dropna(subset=["alsfrs_r"])
    if len(df) < 2:
        return (None, "Yeterli ALSFRS-R geçmişi yok.")
    first = float(df.iloc[0]["alsfrs_r"])
    last = float(df.iloc[-1]["alsfrs_r"])
    n = len(df) - 1
    decline_per_visit = (last - first) / n if n > 0 else 0.0
    if decline_per_visit <= -2:
        msg = "Hızlı progresyon olasılığı: yakın aralıklı multidisipliner izlem önerilir."
    elif decline_per_visit <= -1:
        msg = "Orta progresyon eğilimi mevcut."
    else:
        msg = "Stabil/yavaş progresyon eğilimi."
    return (round(decline_per_visit, 2), msg)


def als_dashboard() -> None:
    _ensure_neuro_dynamic_state(WORKSPACE_ALS)
    _ensure_alsfrs_state()
    scope = _active_patient_scope_key()
    als_items = list(st.session_state.get(_alsfrs_state_key(), [4] * 12))
    state_key = _neuro_input_state_key(WORKSPACE_ALS)
    state_vals = st.session_state.get(state_key, {})

    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #0891b2 0%, #2563eb 100%); padding: 24px; border-radius: 18px; color: white; margin-bottom: 20px;">
            <h2 style="margin:0;">ALS Klinik Operasyon Paneli</h2>
            <p style="margin:6px 0 0 0; opacity:0.92;">ALSFRS-R + Solunum/Bulber Risk Tabanlı İzlem</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("ℹ ALSFRS-R Puanlama Rehberi"):
        st.markdown(
            "- **4:** Normal fonksiyon\n"
            "- **3:** Hafif etkilenim\n"
            "- **2:** Orta etkilenim\n"
            "- **1:** Belirgin etkilenim\n"
            "- **0:** Fonksiyon kaybı"
        )

    a1, a2, a3 = st.columns(3)
    if a1.button("Tümünü 4 Yap", key=f"als_all4_{scope}", use_container_width=True):
        st.session_state[_alsfrs_state_key()] = [4] * 12
        st.rerun()
    if a2.button("Tümünü 2 Yap", key=f"als_all2_{scope}", use_container_width=True):
        st.session_state[_alsfrs_state_key()] = [2] * 12
        st.rerun()
    if a3.button("Sıfırla", key=f"als_reset_{scope}", use_container_width=True):
        st.session_state[_alsfrs_state_key()] = [0] * 12
        st.rerun()

    c1, c2 = st.columns(2)
    for i, label in enumerate(ALSFRS_ITEMS):
        target_col = c1 if i < 6 else c2
        with target_col:
            st.markdown(f"**{label}**")
            widget_key = f"als_item::{scope}::{i}"
            if hasattr(st, "segmented_control"):
                score_i = st.segmented_control(
                    label=f"ALS Item {i + 1}",
                    options=[0, 1, 2, 3, 4],
                    default=int(als_items[i]),
                    key=widget_key,
                    label_visibility="collapsed",
                )
            else:
                score_i = st.radio(
                    label=f"ALS Item {i + 1}",
                    options=[0, 1, 2, 3, 4],
                    index=int(als_items[i]),
                    key=widget_key,
                    horizontal=True,
                    label_visibility="collapsed",
                )
            als_items[i] = int(score_i if score_i is not None else 0)

    st.session_state[_alsfrs_state_key()] = als_items
    alsfrs_total = int(sum(als_items))
    domain_scores = _als_domain_scores(als_items)

    s1, s2, s3 = st.columns(3)
    with s1:
        fvc_pct = st.number_input(
            "FVC (%)",
            min_value=0.0,
            max_value=150.0,
            value=float(state_vals.get("fvc_pct", 85.0)),
            step=1.0,
            key=f"als_fvc::{scope}",
        )
    with s2:
        snip = st.number_input(
            "SNIP (cmH2O)",
            min_value=0.0,
            max_value=200.0,
            value=float(state_vals.get("snip", 60.0)),
            step=1.0,
            key=f"als_snip::{scope}",
        )
    with s3:
        cough_peak = st.number_input(
            "Peak Cough Flow (L/dk)",
            min_value=0.0,
            max_value=800.0,
            value=float(state_vals.get("cough_peak", 280.0)),
            step=5.0,
            key=f"als_cough::{scope}",
        )

    bulbar_impairment = max(0, 12 - int(domain_scores["Bulber"]))
    state_vals.update(
        {
            "alsfrs_r": float(alsfrs_total),
            "fvc_pct": float(fvc_pct),
            "bulbar_score": float(bulbar_impairment),
            "snip": float(snip),
            "cough_peak": float(cough_peak),
        }
    )
    st.session_state[state_key] = state_vals
    analiz = analiz_yap(WORKSPACE_ALS, state_vals)

    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("ALSFRS-R", f"{alsfrs_total} / 48")
    with r2:
        st.metric("FVC", f"{float(fvc_pct):.0f}%")
    with r3:
        st.metric("Bulber Etkilenim", f"{bulbar_impairment} / 12")

    if fvc_pct < 50 or cough_peak < 160:
        st.error("Solunum riski artmış olabilir: NIV ve sekresyon yönetimi açısından acil klinik değerlendirme önerilir.")
    elif fvc_pct < 70 or cough_peak < 240:
        st.warning("Solunum riski orta olabilir: takip sıklığı ve değerlendirme planı artırılmalı.")
    else:
        st.success("Solunum parametreleri göreceli stabil.")
    st.caption("Not: FVC/CPF eşikleri nöromüsküler literatürde karar desteği için kullanılır; tek başına tanı/tedavi kararı yerine geçmez.")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Bulber", f"{domain_scores['Bulber']} / 12")
    d2.metric("İnce Motor", f"{domain_scores['İnce Motor']} / 12")
    d3.metric("Kaba Motor", f"{domain_scores['Kaba Motor']} / 12")
    d4.metric("Solunum", f"{domain_scores['Solunum']} / 12")

    hist_key = _neuro_history_state_key(WORKSPACE_ALS)
    als_history = st.session_state.get(hist_key, [])
    slope, slope_msg = _als_progression_summary(als_history)
    if slope is None:
        st.info(slope_msg)
    else:
        st.caption(f"ALSFRS-R ortalama değişim/ziyaret: {slope:+.2f} | {slope_msg}")

    if st.button("ALS Ölçümünü Kaydet", key=f"save_als_detailed::{scope}", use_container_width=True):
        rec = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "alsfrs_r": float(alsfrs_total),
            "fvc_pct": float(fvc_pct),
            "bulbar_score": float(bulbar_impairment),
            "snip": float(snip),
            "cough_peak": float(cough_peak),
            "progression_index": float(analiz.get("progression_index", 0.0)),
            "alsfrs_items": list(als_items),
            "domain_scores": dict(domain_scores),
        }
        _save_neuro_history_record(WORKSPACE_ALS, rec)
        save_current_session_profile()
        st.success("ALS ölçümü kaydedildi.")

    if PLOTLY_OK:
        dom_df = pd.DataFrame(
            [{"domain": k, "score": v, "max": 12} for k, v in domain_scores.items()]
        )
        fig_dom = px.bar(
            dom_df,
            x="domain",
            y="score",
            title="ALSFRS-R Alt Domain Skorları",
            color="score",
            color_continuous_scale="Blues",
        )
        fig_dom.update_layout(height=300, xaxis_title="", yaxis_title="Skor")
        st.plotly_chart(fig_dom, use_container_width=True)
    else:
        st.dataframe(
            pd.DataFrame([{"Domain": k, "Skor": v} for k, v in domain_scores.items()]),
            use_container_width=True,
            hide_index=True,
        )

    if als_history:
        df_hist = pd.DataFrame(als_history)
        plot_cols = [c for c in ["alsfrs_r", "fvc_pct", "progression_index"] if c in df_hist.columns]
        if "time" in df_hist.columns and plot_cols:
            if PLOTLY_OK:
                fig_line = px.line(
                    df_hist,
                    x="time",
                    y=plot_cols,
                    markers=True,
                    title="ALS Progresyon Eğrileri",
                )
                fig_line.update_layout(height=350, xaxis_title="", yaxis_title="Skor")
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.line_chart(df_hist.set_index("time")[plot_cols], use_container_width=True)
        with st.expander("Son ALS Kayıtları (5)"):
            for rec in reversed(als_history[-5:]):
                st.write(
                    f"- {rec.get('time','-')} | ALSFRS-R: {rec.get('alsfrs_r','-')} | "
                    f"FVC: {rec.get('fvc_pct','-')}% | PI: {rec.get('progression_index','-')}"
                )


def _trend_per_visit(history: list[dict], col: str) -> float | None:
    if not isinstance(history, list) or len(history) < 2:
        return None
    df = pd.DataFrame(history)
    if col not in df.columns:
        return None
    df = df.dropna(subset=[col])
    if len(df) < 2:
        return None
    first = float(df.iloc[0][col])
    last = float(df.iloc[-1][col])
    n = len(df) - 1
    return (last - first) / n if n > 0 else 0.0


def alzheimer_dashboard() -> None:
    _ensure_neuro_dynamic_state(WORKSPACE_ALZHEIMER)
    scope = _active_patient_scope_key()
    state_key = _neuro_input_state_key(WORKSPACE_ALZHEIMER)
    state_vals = st.session_state.get(state_key, {})

    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #0ea5a4 0%, #1d4ed8 100%); padding: 24px; border-radius: 18px; color: white; margin-bottom: 20px;">
            <h2 style="margin:0;">Alzheimer Klinik Operasyon Paneli</h2>
            <p style="margin:6px 0 0 0; opacity:0.92;">Bilişsel Fonksiyon, Davranış ve Günlük Yaşam İzlemi</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        mmse = st.number_input("MMSE", 0, 30, int(state_vals.get("mmse", 24)), key=f"ad_mmse::{scope}")
        moca = st.number_input("MoCA", 0, 30, int(state_vals.get("moca", 21)), key=f"ad_moca::{scope}")
    with c2:
        cdr_sum = st.number_input("CDR-SB", 0.0, 18.0, float(state_vals.get("cdr_sum", 4.0)), step=0.5, key=f"ad_cdr::{scope}")
        adas_cog = st.number_input("ADAS-Cog", 0.0, 70.0, float(state_vals.get("adas_cog", 20.0)), step=1.0, key=f"ad_adas::{scope}")
    with c3:
        adl_ind = st.number_input("ADL Bağımsızlık (%)", 0.0, 100.0, float(state_vals.get("adl_independence", 75.0)), step=1.0, key=f"ad_adl::{scope}")
        npi = st.number_input("NPI Davranış Skoru", 0.0, 144.0, float(state_vals.get("npi", 10.0)), step=1.0, key=f"ad_npi::{scope}")

    # Dynamic config ile uyumlu anahtarları da güncelle.
    state_vals.update(
        {
            "mmse": float(mmse),
            "adas_cog": float(adas_cog),
            "adl_loss": float(max(0.0, 100.0 - float(adl_ind))) / 10.0,
            "moca": float(moca),
            "cdr_sum": float(cdr_sum),
            "adl_independence": float(adl_ind),
            "npi": float(npi),
        }
    )
    st.session_state[state_key] = state_vals

    analiz = analiz_yap(WORKSPACE_ALZHEIMER, state_vals)
    pidx = float(analiz.get("progression_index", 0.0))
    m1, m2, m3 = st.columns(3)
    m1.metric("MMSE", f"{int(mmse)} / 30")
    m2.metric("CDR-SB", f"{float(cdr_sum):.1f} / 18")
    m3.metric("Progresyon İndeksi", f"{pidx:.1f} / 100")

    if mmse < 10 or cdr_sum >= 12:
        st.error("İleri düzey bilişsel etkilenim olasılığı: bakım yükü ve güvenlik planı yakın izlenmeli.")
    elif mmse < 20 or cdr_sum >= 6:
        st.warning("Orta düzey bilişsel etkilenim olasılığı: fonksiyonel destek planı güçlendirilmeli.")
    else:
        st.success("Erken/hafif evre profili: düzenli bilişsel izlem sürdürülmeli.")
    st.caption("Not: MMSE/CDR-SB aralıkları klinik destek amaçlıdır; kesin evreleme uzman değerlendirmesiyle yapılmalıdır.")

    if st.button("Alzheimer Ölçümünü Kaydet", key=f"save_ad_detailed::{scope}", use_container_width=True):
        rec = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "mmse": float(mmse),
            "moca": float(moca),
            "cdr_sum": float(cdr_sum),
            "adas_cog": float(adas_cog),
            "adl_independence": float(adl_ind),
            "npi": float(npi),
            "progression_index": pidx,
        }
        _save_neuro_history_record(WORKSPACE_ALZHEIMER, rec)
        save_current_session_profile()
        st.success("Alzheimer ölçümü kaydedildi.")

    if PLOTLY_OK:
        cur_df = pd.DataFrame(
            [
                {"metric": "MMSE", "value": float(mmse), "max": 30.0},
                {"metric": "MoCA", "value": float(moca), "max": 30.0},
                {"metric": "CDR-SB", "value": float(cdr_sum), "max": 18.0},
                {"metric": "ADAS-Cog", "value": float(adas_cog), "max": 70.0},
                {"metric": "ADL %", "value": float(adl_ind), "max": 100.0},
            ]
        )
        fig = px.bar(cur_df, x="metric", y="value", title="Alzheimer Güncel Skor Özeti", color="value", color_continuous_scale="Teal")
        fig.update_layout(height=320, xaxis_title="", yaxis_title="Skor")
        st.plotly_chart(fig, use_container_width=True)

    ad_history = st.session_state.get(_neuro_history_state_key(WORKSPACE_ALZHEIMER), [])
    if ad_history:
        df = pd.DataFrame(ad_history)
        cols = [c for c in ["mmse", "cdr_sum", "progression_index"] if c in df.columns]
        if "time" in df.columns and cols:
            if PLOTLY_OK:
                fig_line = px.line(df, x="time", y=cols, markers=True, title="Alzheimer Progresyon Eğrileri")
                fig_line.update_layout(height=340, xaxis_title="", yaxis_title="Skor")
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.line_chart(df.set_index("time")[cols], use_container_width=True)
        slope = _trend_per_visit(ad_history, "mmse")
        if slope is not None:
            st.caption(f"MMSE ortalama değişim/ziyaret: {slope:+.2f}")
        with st.expander("Son Alzheimer Kayıtları (5)"):
            for rec in reversed(ad_history[-5:]):
                st.write(
                    f"- {rec.get('time','-')} | MMSE: {rec.get('mmse','-')} | "
                    f"CDR-SB: {rec.get('cdr_sum','-')} | PI: {rec.get('progression_index','-')}"
                )


def parkinson_dashboard() -> None:
    _ensure_neuro_dynamic_state(WORKSPACE_PARKINSON)
    scope = _active_patient_scope_key()
    state_key = _neuro_input_state_key(WORKSPACE_PARKINSON)
    state_vals = st.session_state.get(state_key, {})

    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #0f766e 0%, #7c3aed 100%); padding: 24px; border-radius: 18px; color: white; margin-bottom: 20px;">
            <h2 style="margin:0;">Parkinson Klinik Operasyon Paneli</h2>
            <p style="margin:6px 0 0 0; opacity:0.92;">UPDRS, Evreleme ve Denge/Fall Riski İzlemi</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        updrs_1 = st.number_input("MDS-UPDRS I", 0, 52, int(state_vals.get("updrs_1", 8)), key=f"pd_up1::{scope}")
        updrs_2 = st.number_input("MDS-UPDRS II", 0, 52, int(state_vals.get("updrs_2", 12)), key=f"pd_up2::{scope}")
    with c2:
        updrs_3 = st.number_input("MDS-UPDRS III", 0, 132, int(state_vals.get("updrs_3", 28)), key=f"pd_up3::{scope}")
        updrs_4 = st.number_input("MDS-UPDRS IV", 0, 24, int(state_vals.get("updrs_4", 4)), key=f"pd_up4::{scope}")
    with c3:
        hn_stage = st.number_input("Hoehn-Yahr", 1.0, 5.0, float(state_vals.get("hn_stage", 2.0)), step=0.5, key=f"pd_hy::{scope}")
        tug_sec = st.number_input("TUG (sn)", 0.0, 180.0, float(state_vals.get("tug_sec", 12.0)), step=0.5, key=f"pd_tug::{scope}")

    p2, p3 = st.columns(2)
    with p2:
        falls_month = st.number_input("Aylık Düşme Sayısı", 0, 100, int(state_vals.get("falls_month", 1)), key=f"pd_falls::{scope}")
    with p3:
        freezing = st.number_input("Freezing Şiddeti", 0, 10, int(state_vals.get("freezing", 2)), key=f"pd_freeze::{scope}")

    updrs_total = int(updrs_1 + updrs_2 + updrs_3 + updrs_4)
    fall_risk = float(min(10, falls_month + (1 if tug_sec >= 13.5 else 0) + (1 if freezing >= 4 else 0)))
    state_vals.update(
        {
            "updrs_1": float(updrs_1),
            "updrs_2": float(updrs_2),
            "updrs_3": float(updrs_3),
            "updrs_4": float(updrs_4),
            "updrs_total": float(updrs_total),
            "hn_stage": float(hn_stage),
            "tug_sec": float(tug_sec),
            "falls_month": float(falls_month),
            "freezing": float(freezing),
            "fall_risk": float(fall_risk),
        }
    )
    st.session_state[state_key] = state_vals
    analiz = analiz_yap(WORKSPACE_PARKINSON, state_vals)
    pidx = float(analiz.get("progression_index", 0.0))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("UPDRS Toplam", f"{updrs_total}")
    m2.metric("Hoehn-Yahr", f"{hn_stage:.1f}")
    m3.metric("Düşme Riski", f"{fall_risk:.1f} / 10")
    m4.metric("Progresyon İndeksi", f"{pidx:.1f} / 100")

    if hn_stage >= 4 or falls_month >= 4:
        st.error("Yüksek denge/mobilite riski olasılığı: düşme önleme ve destek cihaz planı acil gözden geçirilmeli.")
    elif hn_stage >= 3 or falls_month >= 2:
        st.warning("Orta risk profili olasılığı: denge rehabilitasyonu ve ev içi güvenlik güçlendirilmeli.")
    else:
        st.success("Mobilite riski göreceli düşük/stabil.")
    st.caption("Not: H&Y, düşme sayısı ve TUG birlikte yorumlanır; tek ölçütle tedavi kararı verilmez.")

    if st.button("Parkinson Ölçümünü Kaydet", key=f"save_pd_detailed::{scope}", use_container_width=True):
        rec = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "updrs_total": float(updrs_total),
            "updrs_1": float(updrs_1),
            "updrs_2": float(updrs_2),
            "updrs_3": float(updrs_3),
            "updrs_4": float(updrs_4),
            "hn_stage": float(hn_stage),
            "tug_sec": float(tug_sec),
            "falls_month": float(falls_month),
            "freezing": float(freezing),
            "fall_risk": float(fall_risk),
            "progression_index": pidx,
        }
        _save_neuro_history_record(WORKSPACE_PARKINSON, rec)
        save_current_session_profile()
        st.success("Parkinson ölçümü kaydedildi.")

    if PLOTLY_OK:
        dom_df = pd.DataFrame(
            [
                {"domain": "UPDRS I", "score": updrs_1},
                {"domain": "UPDRS II", "score": updrs_2},
                {"domain": "UPDRS III", "score": updrs_3},
                {"domain": "UPDRS IV", "score": updrs_4},
            ]
        )
        fig_dom = px.bar(dom_df, x="domain", y="score", title="UPDRS Alt Domain Dağılımı", color="score", color_continuous_scale="Viridis")
        fig_dom.update_layout(height=320, xaxis_title="", yaxis_title="Skor")
        st.plotly_chart(fig_dom, use_container_width=True)

    pd_history = st.session_state.get(_neuro_history_state_key(WORKSPACE_PARKINSON), [])
    if pd_history:
        df = pd.DataFrame(pd_history)
        cols = [c for c in ["updrs_total", "hn_stage", "fall_risk", "progression_index"] if c in df.columns]
        if "time" in df.columns and cols:
            if PLOTLY_OK:
                fig_line = px.line(df, x="time", y=cols, markers=True, title="Parkinson Progresyon Eğrileri")
                fig_line.update_layout(height=340, xaxis_title="", yaxis_title="Skor")
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.line_chart(df.set_index("time")[cols], use_container_width=True)
        slope = _trend_per_visit(pd_history, "updrs_total")
        if slope is not None:
            st.caption(f"UPDRS toplam ortalama değişim/ziyaret: {slope:+.2f}")
        with st.expander("Son Parkinson Kayıtları (5)"):
            for rec in reversed(pd_history[-5:]):
                st.write(
                    f"- {rec.get('time','-')} | UPDRS: {rec.get('updrs_total','-')} | "
                    f"H&Y: {rec.get('hn_stage','-')} | PI: {rec.get('progression_index','-')}"
                )


def huntington_dashboard() -> None:
    _ensure_neuro_dynamic_state(WORKSPACE_HUNTINGTON)
    scope = _active_patient_scope_key()
    state_key = _neuro_input_state_key(WORKSPACE_HUNTINGTON)
    state_vals = st.session_state.get(state_key, {})

    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #0f766e 0%, #1d4ed8 100%); padding: 24px; border-radius: 18px; color: white; margin-bottom: 20px;">
            <h2 style="margin:0;">Huntington Klinik Operasyon Paneli</h2>
            <p style="margin:6px 0 0 0; opacity:0.92;">UHDRS Motor + Kognitif + Fonksiyonel İzlem</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        uhdrs_motor = st.number_input("UHDRS Motor", 0, 124, int(state_vals.get("uhdrs_motor", 28)), key=f"hd_motor::{scope}")
        chorea = st.number_input("Kore Şiddeti", 0, 28, int(state_vals.get("chorea", 8)), key=f"hd_chorea::{scope}")
    with c2:
        sdmt = st.number_input("SDMT", 0, 110, int(state_vals.get("sdmt", 40)), key=f"hd_sdmt::{scope}")
        stroop = st.number_input("Stroop Word", 0, 120, int(state_vals.get("stroop", 55)), key=f"hd_stroop::{scope}")
    with c3:
        tfc = st.number_input("TFC", 0, 13, int(state_vals.get("tfc", 9)), key=f"hd_tfc::{scope}")
        adl_ind = st.number_input("ADL Bağımsızlık (%)", 0.0, 100.0, float(state_vals.get("adl_independence", 70.0)), step=1.0, key=f"hd_adl::{scope}")

    state_vals.update(
        {
            "uhdrs_motor": float(uhdrs_motor),
            "chorea": float(chorea),
            "sdmt": float(sdmt),
            "stroop": float(stroop),
            "tfc": float(tfc),
            "adl_independence": float(adl_ind),
            "progression_hint": float(uhdrs_motor + chorea),
        }
    )
    st.session_state[state_key] = state_vals
    analiz = analiz_yap(WORKSPACE_HUNTINGTON, state_vals)
    pidx = float(analiz.get("progression_index", 0.0))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("UHDRS Motor", str(int(uhdrs_motor)))
    m2.metric("TFC", f"{int(tfc)} / 13")
    m3.metric("SDMT", str(int(sdmt)))
    m4.metric("Progresyon İndeksi", f"{pidx:.1f} / 100")

    if tfc <= 4 or uhdrs_motor >= 60:
        st.error("İleri fonksiyonel etkilenim olasılığı: güvenlik ve bakım planı sıkılaştırılmalı.")
    elif tfc <= 8 or uhdrs_motor >= 35:
        st.warning("Orta etkilenim olasılığı: takip aralığı ve rehabilitasyon planı güncellenmeli.")
    else:
        st.success("Fonksiyonel etkilenim düşük/orta-alt düzeyde.")
    st.caption("Not: UHDRS-TMS ve TFC birlikte değerlendirilmelidir; klinik tablo multidisipliner olarak yorumlanır.")

    if st.button("Huntington Ölçümünü Kaydet", key=f"save_hd_detailed::{scope}", use_container_width=True):
        rec = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "uhdrs_motor": float(uhdrs_motor),
            "chorea": float(chorea),
            "sdmt": float(sdmt),
            "stroop": float(stroop),
            "tfc": float(tfc),
            "adl_independence": float(adl_ind),
            "progression_index": pidx,
        }
        _save_neuro_history_record(WORKSPACE_HUNTINGTON, rec)
        save_current_session_profile()
        st.success("Huntington ölçümü kaydedildi.")

    if PLOTLY_OK:
        dom_df = pd.DataFrame(
            [
                {"metric": "UHDRS Motor", "value": uhdrs_motor},
                {"metric": "Kore", "value": chorea},
                {"metric": "SDMT", "value": sdmt},
                {"metric": "TFC", "value": tfc},
            ]
        )
        fig = px.bar(dom_df, x="metric", y="value", title="Huntington Güncel Skor Özeti", color="value", color_continuous_scale="Teal")
        fig.update_layout(height=320, xaxis_title="", yaxis_title="Skor")
        st.plotly_chart(fig, use_container_width=True)

    hist = st.session_state.get(_neuro_history_state_key(WORKSPACE_HUNTINGTON), [])
    if hist:
        df = pd.DataFrame(hist)
        cols = [c for c in ["uhdrs_motor", "tfc", "progression_index"] if c in df.columns]
        if "time" in df.columns and cols:
            if PLOTLY_OK:
                fig_line = px.line(df, x="time", y=cols, markers=True, title="Huntington Progresyon Eğrileri")
                fig_line.update_layout(height=340, xaxis_title="", yaxis_title="Skor")
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.line_chart(df.set_index("time")[cols], use_container_width=True)


def lewy_body_dementia_dashboard() -> None:
    _ensure_neuro_dynamic_state(WORKSPACE_LEWY)
    scope = _active_patient_scope_key()
    state_key = _neuro_input_state_key(WORKSPACE_LEWY)
    state_vals = st.session_state.get(state_key, {})

    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #0f766e 0%, #6366f1 100%); padding: 24px; border-radius: 18px; color: white; margin-bottom: 20px;">
            <h2 style="margin:0;">Lewy Cisimcikli Demans Klinik Operasyon Paneli</h2>
            <p style="margin:6px 0 0 0; opacity:0.92;">Dalgalanma, Halüsinasyon ve Parkinsonizm Odaklı İzlem</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        mmse = st.number_input("MMSE", 0, 30, int(state_vals.get("mmse", 22)), key=f"lbd_mmse::{scope}")
        fluctuation = st.number_input("Bilişsel Dalgalanma", 0, 10, int(state_vals.get("fluctuation", 4)), key=f"lbd_fluct::{scope}")
    with c2:
        hallucination = st.number_input("Halüsinasyon Şiddeti", 0, 12, int(state_vals.get("hallucination", 3)), key=f"lbd_hall::{scope}")
        parkinsonism = st.number_input("Parkinsonizm Şiddeti", 0, 20, int(state_vals.get("parkinsonism", 6)), key=f"lbd_parkinson::{scope}")
    with c3:
        rem = st.number_input("REM Uyku Davranış Bozukluğu", 0, 10, int(state_vals.get("rbd", 3)), key=f"lbd_rbd::{scope}")
        adl_ind = st.number_input("ADL Bağımsızlık (%)", 0.0, 100.0, float(state_vals.get("adl_independence", 65.0)), step=1.0, key=f"lbd_adl::{scope}")

    state_vals.update(
        {
            "mmse": float(mmse),
            "fluctuation": float(fluctuation),
            "hallucination": float(hallucination),
            "parkinsonism": float(parkinsonism),
            "rbd": float(rem),
            "adl_independence": float(adl_ind),
        }
    )
    st.session_state[state_key] = state_vals
    analiz = analiz_yap(WORKSPACE_LEWY, state_vals)
    pidx = float(analiz.get("progression_index", 0.0))

    m1, m2, m3 = st.columns(3)
    m1.metric("MMSE", f"{int(mmse)} / 30")
    m2.metric("Nörops?kiyatrik Yük", f"{int(hallucination + fluctuation)}")
    m3.metric("Progresyon İndeksi", f"{pidx:.1f} / 100")

    if mmse < 12 or hallucination >= 8:
        st.error("Yüksek nörops?kiyatrik risk: güvenlik ve bakım planı yakın izlenmeli.")
    elif mmse < 20 or fluctuation >= 6:
        st.warning("Orta risk profili: tedavi/izlem planı sıklaştırılmalı.")
    else:
        st.success("Klinik etkilenim düşük-orta seviyede.")

    if st.button("Lewy Ölçümünü Kaydet", key=f"save_lbd_detailed::{scope}", use_container_width=True):
        rec = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "mmse": float(mmse),
            "fluctuation": float(fluctuation),
            "hallucination": float(hallucination),
            "parkinsonism": float(parkinsonism),
            "rbd": float(rem),
            "adl_independence": float(adl_ind),
            "progression_index": pidx,
        }
        _save_neuro_history_record(WORKSPACE_LEWY, rec)
        save_current_session_profile()
        st.success("Lewy ölçümü kaydedildi.")

    if PLOTLY_OK:
        cur_df = pd.DataFrame(
            [
                {"metric": "MMSE", "value": mmse},
                {"metric": "Dalgalanma", "value": fluctuation},
                {"metric": "Halüsinasyon", "value": hallucination},
                {"metric": "Parkinsonizm", "value": parkinsonism},
            ]
        )
        fig = px.bar(cur_df, x="metric", y="value", title="Lewy Güncel Skor Özeti", color="value", color_continuous_scale="Viridis")
        fig.update_layout(height=320, xaxis_title="", yaxis_title="Skor")
        st.plotly_chart(fig, use_container_width=True)

    hist = st.session_state.get(_neuro_history_state_key(WORKSPACE_LEWY), [])
    if hist:
        df = pd.DataFrame(hist)
        cols = [c for c in ["mmse", "hallucination", "fluctuation", "progression_index"] if c in df.columns]
        if "time" in df.columns and cols:
            if PLOTLY_OK:
                fig_line = px.line(df, x="time", y=cols, markers=True, title="Lewy Progresyon Eğrileri")
                fig_line.update_layout(height=340, xaxis_title="", yaxis_title="Skor")
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.line_chart(df.set_index("time")[cols], use_container_width=True)


def ftd_dashboard() -> None:
    _ensure_neuro_dynamic_state(WORKSPACE_FTD)
    scope = _active_patient_scope_key()
    state_key = _neuro_input_state_key(WORKSPACE_FTD)
    state_vals = st.session_state.get(state_key, {})

    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #0ea5a4 0%, #7c3aed 100%); padding: 24px; border-radius: 18px; color: white; margin-bottom: 20px;">
            <h2 style="margin:0;">FTD Klinik Operasyon Paneli</h2>
            <p style="margin:6px 0 0 0; opacity:0.92;">Davranışsal ve Dil Alt Tiplerine Yönelik İzlem</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        fbi = st.number_input("FBI", 0, 72, int(state_vals.get("fbi", 20)), key=f"ftd_fbi::{scope}")
        apathy = st.number_input("Apati", 0, 10, int(state_vals.get("apathy", 4)), key=f"ftd_apathy::{scope}")
    with c2:
        disinhibition = st.number_input("Disinhibisyon", 0, 10, int(state_vals.get("disinhibition", 4)), key=f"ftd_disin::{scope}")
        language = st.number_input("Dil Etkilenimi", 0, 10, int(state_vals.get("language_impairment", 3)), key=f"ftd_lang::{scope}")
    with c3:
        executive = st.number_input("Yürütücü İşlev Etkilenimi", 0, 10, int(state_vals.get("executive", 4)), key=f"ftd_exec::{scope}")
        adl_loss = st.number_input("ADL Kayıp Düzeyi", 0, 10, int(state_vals.get("adl_loss", 3)), key=f"ftd_adl_loss::{scope}")

    state_vals.update(
        {
            "fbi": float(fbi),
            "apathy": float(apathy),
            "disinhibition": float(disinhibition),
            "language_impairment": float(language),
            "executive": float(executive),
            "adl_loss": float(adl_loss),
        }
    )
    st.session_state[state_key] = state_vals
    analiz = analiz_yap(WORKSPACE_FTD, state_vals)
    pidx = float(analiz.get("progression_index", 0.0))

    m1, m2, m3 = st.columns(3)
    m1.metric("FBI", f"{int(fbi)} / 72")
    m2.metric("Davranışsal Yük", f"{int(apathy + disinhibition)} / 20")
    m3.metric("Progresyon İndeksi", f"{pidx:.1f} / 100")

    if fbi >= 45 or adl_loss >= 7:
        st.error("Yüksek davranışsal/fonksiyonel etkilenim olasılığı: bakım planı ve güvenlik önlemleri artırılmalı.")
    elif fbi >= 25 or adl_loss >= 4:
        st.warning("Orta etkilenim olasılığı: davranışsal müdahale ve takip sıklığı artırılmalı.")
    else:
        st.success("Etkilenim düşük-orta seviyede.")
    st.caption("Not: FTD skorlama eşikleri merkezler arasında değişebilir; sonuçlar klinik görüşmeyle birlikte ele alınmalıdır.")

    if st.button("FTD Ölçümünü Kaydet", key=f"save_ftd_detailed::{scope}", use_container_width=True):
        rec = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "fbi": float(fbi),
            "apathy": float(apathy),
            "disinhibition": float(disinhibition),
            "language_impairment": float(language),
            "executive": float(executive),
            "adl_loss": float(adl_loss),
            "progression_index": pidx,
        }
        _save_neuro_history_record(WORKSPACE_FTD, rec)
        save_current_session_profile()
        st.success("FTD ölçümü kaydedildi.")

    if PLOTLY_OK:
        cur_df = pd.DataFrame(
            [
                {"metric": "FBI", "value": fbi},
                {"metric": "Apati", "value": apathy},
                {"metric": "Disinhibisyon", "value": disinhibition},
                {"metric": "Dil", "value": language},
                {"metric": "Yürütücü", "value": executive},
            ]
        )
        fig = px.bar(cur_df, x="metric", y="value", title="FTD Güncel Skor Özeti", color="value", color_continuous_scale="Magma")
        fig.update_layout(height=320, xaxis_title="", yaxis_title="Skor")
        st.plotly_chart(fig, use_container_width=True)

    hist = st.session_state.get(_neuro_history_state_key(WORKSPACE_FTD), [])
    if hist:
        df = pd.DataFrame(hist)
        cols = [c for c in ["fbi", "adl_loss", "progression_index"] if c in df.columns]
        if "time" in df.columns and cols:
            if PLOTLY_OK:
                fig_line = px.line(df, x="time", y=cols, markers=True, title="FTD Progresyon Eğrileri")
                fig_line.update_layout(height=340, xaxis_title="", yaxis_title="Skor")
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.line_chart(df.set_index("time")[cols], use_container_width=True)


def sma_dashboard() -> None:
    _render_dynamic_neuro_dashboard(WORKSPACE_SMA)


WORKSPACE_DASHBOARD_HANDLERS = {
    WORKSPACE_ALS: als_dashboard,
    WORKSPACE_ALZHEIMER: alzheimer_dashboard,
    WORKSPACE_PARKINSON: parkinson_dashboard,
    WORKSPACE_HUNTINGTON: huntington_dashboard,
    WORKSPACE_LEWY: lewy_body_dementia_dashboard,
    WORKSPACE_FTD: ftd_dashboard,
    WORKSPACE_SMA: sma_dashboard,
}

DISEASE_WORKSPACE_NAV_PAGES = [
    "Ana Panel",
    "Klinik Hesaplayıcı",
    "Tam Ölçekli Test",
    "Klinik Operasyon Merkezi",
    "Klinik Takvim & Haklar",
    "Acil Durum & Kritik Bakım",
    "Sıkça Sorulan Sorular",
    "Güncel Haberler",
    "AI'ya Sor",
    "Vizyon & İmza",
]

WORKSPACE_NEWS_QUERY = {
    WORKSPACE_ALS: "amyotrophic lateral sclerosis",
    WORKSPACE_ALZHEIMER: "alzheimer disease",
    WORKSPACE_PARKINSON: "parkinson disease",
    WORKSPACE_HUNTINGTON: "huntington disease",
    WORKSPACE_LEWY: "dementia with lewy bodies",
    WORKSPACE_FTD: "frontotemporal dementia",
    WORKSPACE_SMA: "spinal muscular atrophy",
}

WORKSPACE_FULL_TEST_MODELS: dict[str, dict] = {
    WORKSPACE_ALS: {
        "name": "ALSFRS-R (12 Madde)",
        "higher_is_better": True,
        "items": [{"label": x, "min": 0, "max": 4} for x in ALSFRS_ITEMS],
    },
    WORKSPACE_ALZHEIMER: {
        "name": "MMSE Domain Testi",
        "higher_is_better": True,
        "items": [
            {"label": "Yonelim (Zaman)", "min": 0, "max": 5},
            {"label": "Yonelim (Yer)", "min": 0, "max": 5},
            {"label": "Kayıt / İsim Tekrarı", "min": 0, "max": 3},
            {"label": "Dikkat / Hesaplama", "min": 0, "max": 5},
            {"label": "Gecikmeli Hatirlama", "min": 0, "max": 3},
            {"label": "Dil ve Komutlar", "min": 0, "max": 9},
        ],
    },
    WORKSPACE_PARKINSON: {
        "name": "UPDRS Hizli Tarama",
        "higher_is_better": False,
        "items": [
            {"label": "Tremor", "min": 0, "max": 4},
            {"label": "Bradikinezi", "min": 0, "max": 4},
            {"label": "Rijidite", "min": 0, "max": 4},
            {"label": "Postural Instabilite", "min": 0, "max": 4},
            {"label": "Yürüme", "min": 0, "max": 4},
            {"label": "Donakalma", "min": 0, "max": 4},
            {"label": "Konuşma", "min": 0, "max": 4},
            {"label": "Yutma", "min": 0, "max": 4},
        ],
    },
    WORKSPACE_HUNTINGTON: {
        "name": "UHDRS Motor+Fonksiyon Hizli",
        "higher_is_better": False,
        "items": [
            {"label": "Kore", "min": 0, "max": 4},
            {"label": "Distoni", "min": 0, "max": 4},
            {"label": "Sakkad Takibi", "min": 0, "max": 4},
            {"label": "Dizartri", "min": 0, "max": 4},
            {"label": "Yuruyus", "min": 0, "max": 4},
            {"label": "Denge", "min": 0, "max": 4},
            {"label": "Fonksiyonel Bagimsizlik", "min": 0, "max": 4},
            {"label": "ADL Destek Ihtiyaci", "min": 0, "max": 4},
        ],
    },
    WORKSPACE_LEWY: {
        "name": "Lewy Core Semptom Testi",
        "higher_is_better": False,
        "items": [
            {"label": "Bilissel Dalgalanma", "min": 0, "max": 4},
            {"label": "Görsel Halüsinasyon", "min": 0, "max": 4},
            {"label": "Parkinsonizm", "min": 0, "max": 4},
            {"label": "REM Uyku Davranis Bozuklugu", "min": 0, "max": 4},
            {"label": "Dikkat ve Islemleme", "min": 0, "max": 4},
            {"label": "ADL Etkilenimi", "min": 0, "max": 4},
        ],
    },
    WORKSPACE_FTD: {
        "name": "FTD Davranissal-Dil Testi",
        "higher_is_better": False,
        "items": [
            {"label": "Apati", "min": 0, "max": 4},
            {"label": "Disinhibisyon", "min": 0, "max": 4},
            {"label": "Empati Kaybi", "min": 0, "max": 4},
            {"label": "Dil Bozulmasi", "min": 0, "max": 4},
            {"label": "Yürütücü İşlev", "min": 0, "max": 4},
            {"label": "Sosyal Icgoru", "min": 0, "max": 4},
            {"label": "ADL Etkilenimi", "min": 0, "max": 4},
        ],
    },
    WORKSPACE_SMA: {
        "name": "HFMSE (33 Madde)",
        "higher_is_better": True,
        "items": [{"label": f"HFMSE Madde {i}", "min": 0, "max": 2} for i in range(1, 34)],
    },
}


def _clinical_evidence_caption(level: str, text: str) -> None:
    lang = str(st.session_state.get("lang", "TR"))
    raw_tag = str(level or "Bilgilendirme").strip()

    tag_map = {
        "TR": {
            "Bilgilendirme": "Bilgilendirme",
            "Karar-Destek": "Karar-Destek",
            "Acil Uyarı": "Acil Uyarı",
        },
        "EN": {
            "Bilgilendirme": "Information",
            "Karar-Destek": "Decision Support",
            "Acil Uyarı": "Emergency Notice",
        },
        "DE": {
            "Bilgilendirme": "Information",
            "Karar-Destek": "Entscheidungsunterstützung",
            "Acil Uyarı": "Notfallhinweis",
        },
    }
    label_map = {
        "TR": "Kanıt Seviyesi",
        "EN": "Evidence Level",
        "DE": "Evidenzniveau",
    }

    label = label_map.get(lang, "Kanıt Seviyesi")
    tag = tag_map.get(lang, tag_map["TR"]).get(raw_tag, raw_tag)
    body = _i18n_text(str(text or ""))
    st.caption(f"[{label}: {tag}] {body}")


def _workspace_full_test_interpretation(workspace_mode: str, burden_pct: float) -> tuple[str, str]:
    lvl = "low"
    if burden_pct >= 66:
        lvl = "high"
    elif burden_pct >= 33:
        lvl = "mid"

    msg_map = {
        WORKSPACE_ALS: {
            "high": "ALS: Fonksiyonel yük belirgin. Solunum, bulber ve beslenme izlem planı hızlandırılmalı.",
            "mid": "ALS: Orta düzey etkilenim. ALSFRS-R trendi ve solunum parametreleri daha sık izlenmeli.",
            "low": "ALS: Etkilenim düşük. Standart multidisipliner takip sürdürülebilir.",
        },
        WORKSPACE_ALZHEIMER: {
            "high": "Alzheimer: Bilişsel/fonksiyonel etkilenim yüksek olabilir. Güvenlik ve bakım planı güçlendirilmeli.",
            "mid": "Alzheimer: Orta düzey etkilenim. Günlük yaşam desteği ve davranışsal izlem sıklaştırılmalı.",
            "low": "Alzheimer: Erken/hafif etkilenim profili. Düzenli bilişsel takip sürdürülmeli.",
        },
        WORKSPACE_PARKINSON: {
            "high": "Parkinson: Motor yük yüksek olabilir. Düşme önleme, rehabilitasyon ve ilaç zamanlaması gözden geçirilmeli.",
            "mid": "Parkinson: Orta motor etkilenim. Yürüme-denge odaklı izlem artırılmalı.",
            "low": "Parkinson: Motor etkilenim göreceli düşük. Rutin izlem sürdürülebilir.",
        },
        WORKSPACE_HUNTINGTON: {
            "high": "Huntington: Motor/fonksiyonel etkilenim yüksek olabilir. Güvenlik ve bakım planı önceliklendirilmeli.",
            "mid": "Huntington: Orta etkilenim. Kognitif-fonksiyonel destek planı güncellenmeli.",
            "low": "Huntington: Etkilenim düşük-orta profilde. Düzenli takip sürdürülebilir.",
        },
        WORKSPACE_LEWY: {
            "high": "Lewy: Core semptom yükü yüksek olabilir. Halüsinasyon-dalgalanma güvenlik planı yakından izlenmeli.",
            "mid": "Lewy: Orta semptom yükü. Nöro-ps?kiyatrik izlem sıklaştırılmalı.",
            "low": "Lewy: Semptom yükü düşük-orta. Planlı takip devam edebilir.",
        },
        WORKSPACE_FTD: {
            "high": "FTD: Davranışsal/fonksiyonel yük yüksek olabilir. Bakım veren güvenliği ve davranış yönetimi güçlendirilmeli.",
            "mid": "FTD: Orta düzey etkilenim. Davranışsal ve dil odaklı takip artırılmalı.",
            "low": "FTD: Etkilenim düşük-orta. Planlı izlem sürdürülebilir.",
        },
        WORKSPACE_SMA: {
            "high": "SMA: Motor etkilenim yüksek olabilir. Solunum ve fonksiyonel hedefler açısından yakın takip önemli.",
            "mid": "SMA: Orta düzey etkilenim. HFMSE/RULM trendine göre rehabilitasyon planı güncellenmeli.",
            "low": "SMA: Etkilenim düşük. Düzenli motor-solunum takibi sürdürülebilir.",
        },
    }
    default_map = {
        "high": "Yüksek etkilenim profili: takip sıklığı ve güvenlik planı güçlendirilmeli.",
        "mid": "Orta etkilenim profili: izlem planı sıklaştırılmalıdır.",
        "low": "Düşük etkilenim profili.",
    }
    selected = msg_map.get(workspace_mode, default_map)
    return lvl, selected.get(lvl, default_map[lvl])


def _workspace_next_control_plan(workspace_mode: str, severity: str) -> dict[str, str]:
    sev = str(severity or "mid").strip().lower()
    if sev not in {"low", "mid", "high"}:
        sev = "mid"

    # Intervals with explicit guideline support where available (ALS, Parkinson, SMA).
    explicit_map = {
        WORKSPACE_ALS: {
            "high": "2-6 hafta",
            "mid": "1-2 ay",
            "low": "2-3 ay",
            "basis": "NICE NG42 ALS izleminde düzenli multidisipliner değerlendirme (genellikle 2-3 ay) vurgular.",
            "evidence": "Kılavuz + klinik bireyselleştirme",
        },
        WORKSPACE_PARKINSON: {
            "high": "1-2 ay",
            "mid": "2-4 ay",
            "low": "6-12 ay",
            "basis": "NICE NG71 stabil olguda periyodik izlem, değişken semptomlarda daha sık kontrol önerir.",
            "evidence": "Kılavuz + klinik bireyselleştirme",
        },
        WORKSPACE_SMA: {
            "high": "1-3 ay",
            "mid": "3-6 ay",
            "low": "6 ay",
            "basis": "SMA bakım standartlarında fonksiyonel/solunumsal izlemin evreye göre 3-6 ay aralıkla planlanması yaygındır.",
            "evidence": "Kılavuz/uzman konsensusu + bireyselleştirme",
        },
    }

    # For diseases without a single fixed interval guideline, use conservative clinical cadence.
    consensus_map = {
        WORKSPACE_ALZHEIMER: {"high": "1-3 ay", "mid": "3 ay", "low": "6 ay"},
        WORKSPACE_HUNTINGTON: {"high": "1-3 ay", "mid": "3 ay", "low": "3-6 ay"},
        WORKSPACE_LEWY: {"high": "1-3 ay", "mid": "3 ay", "low": "3-6 ay"},
        WORKSPACE_FTD: {"high": "1-3 ay", "mid": "3 ay", "low": "3-6 ay"},
    }

    if workspace_mode in explicit_map:
        row = explicit_map[workspace_mode]
        return {
            "interval": row.get(sev, "3 ay"),
            "basis": row.get("basis", ""),
            "evidence": row.get("evidence", "Kılavuz"),
            "personalized": False,
            "risk_hits": [],
        }

    row = consensus_map.get(
        workspace_mode,
        {"high": "1-3 ay", "mid": "3 ay", "low": "6 ay"},
    )
    return {
        "interval": row.get(sev, "3 ay"),
        "basis": "Bu hastalık grubunda takip aralığı semptom hızı, bakım yükü ve güvenlik riskine göre bireyselleştirilir.",
        "evidence": "Uzman konsensusu + bireyselleştirme",
        "personalized": False,
        "risk_hits": [],
    }


def _workspace_personal_risk_hits(workspace_mode: str) -> tuple[list[str], bool]:
    state = st.session_state.get(_neuro_input_state_key(workspace_mode), {})
    if not isinstance(state, dict):
        state = {}
    hits: list[str] = []
    urgent = False

    def f(key: str, default: float = 0.0) -> float:
        try:
            return float(state.get(key, default))
        except Exception:
            return float(default)

    if workspace_mode == WORKSPACE_ALS:
        fvc = f("fvc_pct", 100.0)
        cough = f("cough_peak", 300.0)
        if fvc < 50:
            hits.append("FVC <%50")
            urgent = True
        elif fvc < 70:
            hits.append("FVC <%70")
        if cough < 160:
            hits.append("Cough peak <160 L/dk")
            urgent = True
        elif cough < 240:
            hits.append("Cough peak <240 L/dk")
    elif workspace_mode == WORKSPACE_PARKINSON:
        falls = f("falls_month", 0.0)
        hy = f("hn_stage", 1.0)
        tug = f("tug_sec", 10.0)
        if falls >= 4:
            hits.append("Aylık düşme >=4")
            urgent = True
        elif falls >= 2:
            hits.append("Aylık düşme >=2")
        if hy >= 4:
            hits.append("Hoehn-Yahr >=4")
            urgent = True
        elif hy >= 3:
            hits.append("Hoehn-Yahr >=3")
        if tug >= 13.5:
            hits.append("TUG >=13.5 sn")
    elif workspace_mode == WORKSPACE_SMA:
        fvc = f("fvc_pct", 100.0)
        hfmse = f("hfmse", 66.0)
        if fvc < 50:
            hits.append("FVC <%50")
            urgent = True
        elif fvc < 70:
            hits.append("FVC <%70")
        if hfmse <= 20:
            hits.append("HFMSE <=20")
    elif workspace_mode == WORKSPACE_ALZHEIMER:
        mmse = f("mmse", 30.0)
        adl = f("adl_independence", 100.0)
        if mmse < 10:
            hits.append("MMSE <10")
            urgent = True
        elif mmse < 20:
            hits.append("MMSE <20")
        if adl < 50:
            hits.append("ADL bağımsızlık <%50")
    elif workspace_mode == WORKSPACE_HUNTINGTON:
        tfc = f("tfc", 13.0)
        uhdrs = f("uhdrs_motor", 0.0)
        if tfc <= 4:
            hits.append("TFC <=4")
            urgent = True
        elif tfc <= 8:
            hits.append("TFC <=8")
        if uhdrs >= 60:
            hits.append("UHDRS motor >=60")
            urgent = True
        elif uhdrs >= 35:
            hits.append("UHDRS motor >=35")
    elif workspace_mode == WORKSPACE_LEWY:
        mmse = f("mmse", 30.0)
        hall = f("hallucination", 0.0)
        fluc = f("fluctuation", 0.0)
        if mmse < 12:
            hits.append("MMSE <12")
            urgent = True
        elif mmse < 20:
            hits.append("MMSE <20")
        if hall >= 8:
            hits.append("Halüsinasyon >=8")
            urgent = True
        elif fluc >= 6:
            hits.append("Dalgalanma >=6")
    elif workspace_mode == WORKSPACE_FTD:
        fbi = f("fbi", 0.0)
        adl_loss = f("adl_loss", 0.0)
        if fbi >= 45:
            hits.append("FBI >=45")
            urgent = True
        elif fbi >= 25:
            hits.append("FBI >=25")
        if adl_loss >= 7:
            hits.append("ADL kaybi >=7")
            urgent = True
        elif adl_loss >= 4:
            hits.append("ADL kaybi >=4")

    return hits, urgent


def _workspace_next_control_plan_personalized(
    workspace_mode: str,
    severity: str,
    enabled: bool = True,
) -> dict[str, str]:
    base = _workspace_next_control_plan(workspace_mode, severity)
    if not enabled:
        return base

    sev = str(severity or "mid").strip().lower()
    if sev not in {"low", "mid", "high"}:
        sev = "mid"
    rank = {"low": 0, "mid": 1, "high": 2}
    rank_rev = {0: "low", 1: "mid", 2: "high"}

    hits, urgent = _workspace_personal_risk_hits(workspace_mode)
    r = rank.get(sev, 1)
    if urgent:
        r = 2
    elif len(hits) >= 2:
        r = min(2, r + 1)
    elif len(hits) == 1 and r == 0:
        r = 1

    adj_sev = rank_rev[r]
    if urgent:
        interval = "1-2 hafta"
    else:
        interval = _workspace_next_control_plan(workspace_mode, adj_sev).get("interval", base.get("interval", "3 ay"))

    note = "Bireyselleştirilmiş takip aralığı aktif."
    if hits:
        note += " Dikkate alinan riskler: " + ", ".join(hits[:4]) + "."
    else:
        note += " Ek yüksek risk sinyali saptanmadı."

    return {
        **base,
        "interval": interval,
        "personalized": True,
        "risk_hits": hits,
        "basis": f"{base.get('basis', '')} {note}".strip(),
    }


WORKSPACE_FAQ_DATA: dict[str, list[dict[str, str]]] = {
    WORKSPACE_ALS: [
        {"c": "Temel", "q": "ALS nedir?", "a": "ALS, üst ve alt motor nöronları etkileyen ilerleyici bir nörolojik hastalıktır.", "l": "https://www.cdc.gov/als/about/faqs.html"},
        {"c": "Temel", "q": "ALS'nin en sık erken belirtileri nelerdir?", "a": "Kas güçsüzlüğü, konuşma-yutma değişiklikleri ve ince motor becerilerde zorlanma sık başlangıç bulgularıdır.", "l": "https://www.cdc.gov/als/about/faqs.html"},
        {"c": "Neden", "q": "ALS'nin nedeni kesin olarak biliniyor mu?", "a": "Vaka çoğunluğunda tek bir kesin neden gösterilemez; genetik ve çevresel faktörler birlikte rol oynayabilir.", "l": "https://www.cdc.gov/als/about/faqs.html"},
        {"c": "Genetik", "q": "ALS kalıtsal olabilir mi?", "a": "Evet, bazı olgular aileseldir; genetik danışmanlık uygun hastalarda önemlidir.", "l": "https://www.mda.org/disease/amyotrophic-lateral-sclerosis/registry/faqs"},
        {"c": "Tanı", "q": "ALS tanısı nasıl konur?", "a": "Klinik muayene, nörolojik testler ve benzer hastalıkları dışlama yaklaşımı birlikte kullanılır.", "l": "https://www.cdc.gov/als/about/faqs.html"},
        {"c": "Seyir", "q": "Hastalık seyri herkeste aynı mıdır?", "a": "Hayır, progresyon hızı kişiden kişiye belirgin farklılık gösterebilir.", "l": "https://www.als.org/understanding-als/what-is-als"},
        {"c": "Tedavi", "q": "ALS için kesin kür var mı?", "a": "Şu anda kesin kür yoktur; tedavi semptom kontrolü ve yaşam kalitesini artırmaya odaklanır.", "l": "https://www.ninds.nih.gov/health-information/disorders/amyotrophic-lateral-sclerosis-als"},
        {"c": "Solunum", "q": "Solunum takibi neden kritik?", "a": "Solunum kasları etkilenebileceği için düzenli solunum izlemi komplikasyonları erken yakalamayı sağlar.", "l": "https://www.mndassociation.org"},
        {"c": "Solunum", "q": "NIV ne zaman gündeme gelir?", "a": "Solunum kapasitesi düşüşü veya gece hipoventilasyon bulgularında uzman ekip tarafından değerlendirilir.", "l": "https://www.ninds.nih.gov/health-information/disorders/amyotrophic-lateral-sclerosis-als"},
        {"c": "Beslenme", "q": "Yutma güçlüğünde ne yapılmalı?", "a": "Yutma değerlendirmesi, beslenme planı ve aspirasyon riskine yönelik önlemler erken planlanmalıdır.", "l": "https://www.als.org"},
        {"c": "Bakım", "q": "Fizyoterapi ALS'de işe yarar mı?", "a": "Uygun dozda rehabilitasyon, hareketliliği ve günlük yaşam fonksiyonunu destekleyebilir.", "l": "https://www.als.org"},
        {"c": "Bakım", "q": "Multidisipliner klinik neden önerilir?", "a": "Nöroloji, solunum, beslenme ve rehabilitasyonun birlikte yönetimi daha bütüncül bakım sağlar.", "l": "https://www.als.org"},
        {"c": "Araştırma", "q": "Klinik araştırmalara nasıl katılabilirim?", "a": "Uygunluk kriterleri için klinik merkezler ve ClinicalTrials kayıtları takip edilmelidir.", "l": "https://clinicaltrials.gov"},
        {"c": "Destek", "q": "Hasta yakını desteği nereden alınır?", "a": "Yerel dernekler, destek grupları ve sosyal hizmetler bakım veren yükünü azaltmada yardımcı olur.", "l": "https://www.als.org"},
        {"c": "Güvenlik", "q": "Hangi durumda acil başvuru gerekir?", "a": "Ani nefes darlığı, sekresyon yönetememe veya hızlı kötüleşmede acil değerlendirme gerekir.", "l": "https://www.cdc.gov/als/about/faqs.html"},
    ],
    WORKSPACE_ALZHEIMER: [
        {"c": "Temel", "q": "Alzheimer hastalığı nedir?", "a": "Alzheimer, ilerleyici bellek ve bilişsel işlev kaybına yol açan en sık demans nedenidir.", "l": "https://www.alz.org/alzheimers-dementia/what-is-alzheimers"},
        {"c": "Temel", "q": "Normal yaşlanma ile Alzheimer nasıl ayrılır?", "a": "Günlük yaşamı bozan ilerleyici bellek ve işlev kaybı normal yaşlanmadan farklıdır.", "l": "https://www.alz.org/alzheimers-dementia/10_signs"},
        {"c": "Belirti", "q": "Erken belirtiler nelerdir?", "a": "Yeni bilgiyi hatırlamada zorlanma, yönelim sorunları ve planlama güçlüğü sık görülen erken bulgulardır.", "l": "https://www.alz.org/alzheimers-dementia/10_signs"},
        {"c": "Tanı", "q": "Tanı nasıl konur?", "a": "Öykü, bilişsel testler, laboratuvar ve gerekirse görüntüleme birlikte değerlendirilir.", "l": "https://www.nia.nih.gov/health/alzheimers-disease-fact-sheet"},
        {"c": "Skor", "q": "MMSE ve MoCA ne işe yarar?", "a": "Bu testler bilişsel alanları tarar; tek başına tanı değil klinik değerlendirmeyi destekler.", "l": "https://www.alz.org"},
        {"c": "Tedavi", "q": "Tamamen iyileştiren tedavi var mı?", "a": "Kesin kür yoktur; bazı tedaviler semptomları ve hastalık seyrini belirli düzeyde etkileyebilir.", "l": "https://www.nia.nih.gov/health/alzheimers-disease-fact-sheet"},
        {"c": "Günlük Yaşam", "q": "Ev güvenliği nasıl artırılır?", "a": "Düşme önleme, ilaç düzeni, kapı-ocak güvenliği ve rutin planlama temel adımlardır.", "l": "https://www.alz.org/help-support/caregiving"},
        {"c": "Davranış", "q": "Ajitasyon ve davranış değişiklikleri nasıl yönetilir?", "a": "Öncelik çevresel düzenleme, iletişim stratejileri ve tetikleyicileri azaltmaktır.", "l": "https://www.alz.org/help-support/caregiving"},
        {"c": "İletişim", "q": "Hasta ile iletişimde nelere dikkat edilmeli?", "a": "Kısa, net cümleler ve sakin yaklaşım iletişim başarısını artırır.", "l": "https://www.alz.org/help-support/caregiving"},
        {"c": "Bakım", "q": "Ne zaman profesyonel bakım düşünülmeli?", "a": "Güvenlik riski, ciddi işlev kaybı veya bakım veren tükenmişliğinde profesyonel destek planlanmalıdır.", "l": "https://www.alz.org/help-support/resources/helpline"},
        {"c": "Risk", "q": "Aile öyküsü riski artırır mı?", "a": "Bazı genetik ve ailesel faktörler riski etkileyebilir; bu durum kişiye göre değerlendirilmelidir.", "l": "https://www.nia.nih.gov/health/alzheimers-and-genetics"},
        {"c": "Yaşam Tarzı", "q": "Risk azaltmak için ne yapılabilir?", "a": "Fiziksel aktivite, kardiyovasküler risk kontrolü, uyku ve sosyal-bilişsel aktivite destekleyicidir.", "l": "https://www.alz.org/alzheimers-dementia/research_progress/prevention"},
        {"c": "Destek", "q": "7/24 destek hattı var mı?", "a": "Alzheimer's Association 24/7 destek hattı ailelere bilgi ve kriz desteği sağlar.", "l": "https://www.alz.org/help-support/resources/helpline"},
        {"c": "İleri Evre", "q": "Hastalığın ileri evresinde hangi sorunlar beklenir?", "a": "İletişim güçlüğü, yutma sorunları ve tam bakım gereksinimi artabilir.", "l": "https://www.nia.nih.gov/health/alzheimers-disease-fact-sheet"},
        {"c": "Acil", "q": "Ne zaman acil yardım gerekir?", "a": "Ani bilinç değişikliği, düşme, susuzluk veya enfeksiyon şüphesinde acil değerlendirme gerekir.", "l": "https://www.alz.org/help-support/resources/helpline"},
    ],
    WORKSPACE_PARKINSON: [
        {"c": "Temel", "q": "Parkinson hastalığı nedir?", "a": "Parkinson, hareket başta olmak üzere motor ve motor dışı belirtilerle seyreden ilerleyici nörolojik bir hastalıktır.", "l": "https://www.parkinson.org/understanding-parkinsons/what-is-parkinsons"},
        {"c": "Belirti", "q": "En sık belirtiler nelerdir?", "a": "Titreme, yavaşlık, katılık ve denge sorunları temel motor belirtilerdir.", "l": "https://www.parkinson.org/understanding-parkinsons/symptoms"},
        {"c": "Belirti", "q": "Motor dışı belirtiler olur mu?", "a": "Evet; uyku, kabızlık, depresyon, koku kaybı ve bilişsel değişiklikler görülebilir.", "l": "https://www.parkinson.org/understanding-parkinsons/non-movement-symptoms"},
        {"c": "Tanı", "q": "Tanı için tek bir kesin test var mı?", "a": "Hayır; tanı klinik değerlendirme ile konur ve benzer nedenler dışlanır.", "l": "https://www.parkinson.org/understanding-parkinsons/diagnosis"},
        {"c": "Skor", "q": "UPDRS neyi ölçer?", "a": "UPDRS, motor bulgular ve günlük yaşam etkilenimini yapılandırılmış biçimde izlemeye yardımcı olur.", "l": "https://www.parkinson.org/library/books/faq"},
        {"c": "Skor", "q": "Hoehn-Yahr evresi neden kullanılır?", "a": "Hastalığın fonksiyonel evresini özetleyerek takip planına katkı sağlar.", "l": "https://www.parkinson.org/library/books/faq"},
        {"c": "Tedavi", "q": "Parkinson tedavi edilebilir mi?", "a": "Kesin kür yoktur; ilaç, rehabilitasyon ve gerektiğinde girişimsel yöntemlerle semptom yönetilir.", "l": "https://www.parkinson.org/understanding-parkinsons/treatment"},
        {"c": "İlaç", "q": "İlaç saatleri neden önemlidir?", "a": "Doz-zaman uyumu, semptom dalgalanmalarını azaltmada kritik rol oynar.", "l": "https://www.parkinson.org/library/books/faq"},
        {"c": "Egzersiz", "q": "Egzersiz gerçekten faydalı mı?", "a": "Düzenli egzersiz mobilite, denge ve yaşam kalitesini destekler.", "l": "https://www.parkinson.org/living-with-parkinsons/exercise"},
        {"c": "Risk", "q": "Düşme riski nasıl azaltılır?", "a": "Denge egzersizi, çevresel düzenleme ve ilaç planının gözden geçirilmesi önerilir.", "l": "https://www.parkinson.org/library/books/faq"},
        {"c": "Günlük Yaşam", "q": "Beslenmede nelere dikkat edilmeli?", "a": "Yutma, kabızlık ve ilaç-etkileşimleri açısından kişiselleştirilmiş beslenme planı önemlidir.", "l": "https://www.parkinson.org/library/books/faq"},
        {"c": "İleri Tedavi", "q": "DBS ne zaman düşünülür?", "a": "Uygun hastalarda uzman merkezlerde ilaçla yeterli kontrol sağlanamadığında değerlendirilir.", "l": "https://www.parkinson.org/understanding-parkinsons/treatment/surgical-treatment-options"},
        {"c": "Kognitif", "q": "Parkinson'da demans gelişir mi?", "a": "Bazı hastalarda bilişsel etkilenim gelişebilir; düzenli bilişsel izlem önemlidir.", "l": "https://www.parkinson.org/library/books/faq"},
        {"c": "Destek", "q": "Destek hizmetlerine nasıl ulaşırım?", "a": "Parkinson Foundation kaynakları, destek grupları ve helpline yönlendirmeleri yardımcı olur.", "l": "https://www.parkinson.org/blog/awareness/helpline-faq"},
        {"c": "Acil", "q": "Hangi durumda acile başvurmalıyım?", "a": "Şiddetli düşme, ani bilinç değişikliği, ciddi yutma-solunum sorunu varsa acil değerlendirme gerekir.", "l": "https://www.parkinson.org/library/books/faq"},
    ],
    WORKSPACE_SMA: [
        {"c": "Temel", "q": "SMA nedir?", "a": "SMA, motor nöronları etkileyen genetik bir nöromüsküler hastalıktır.", "l": "https://www.curesma.org/faqs/"},
        {"c": "Tipler", "q": "SMA'nın tipleri nelerdir?", "a": "SMA farklı başlangıç yaşları ve fonksiyonel düzeylerle sınıflandırılan alt tiplere ayrılır.", "l": "https://www.curesma.org/faqs/"},
        {"c": "Belirti", "q": "En sık belirtiler nelerdir?", "a": "Kas güçsüzlüğü, motor gecikme ve bazı olgularda solunum-yutma sorunları görülebilir.", "l": "https://www.curesma.org/faqs/"},
        {"c": "Genetik", "q": "SMA nasıl kalıtılır?", "a": "Sıklıkla otozomal resesif geçiş gösterir; taşıyıcılık değerlendirmesi önemlidir.", "l": "https://www.curesma.org/faqs/"},
        {"c": "Genetik", "q": "SMN1 ve SMN2 neden önemlidir?", "a": "SMN1 mutasyonu hastalığın temel nedenidir; SMN2 kopya sayısı fenotipi etkileyebilir.", "l": "https://www.curesma.org/faqs/"},
        {"c": "Tanı", "q": "SMA tanısı nasıl konur?", "a": "Genetik test tanıda temel yaklaşımdır ve klinik bulgularla birlikte değerlendirilir.", "l": "https://www.curesma.org/faqs/"},
        {"c": "Tedavi", "q": "SMA'da tedavi var mı?", "a": "Hastalığın seyrini etkileyebilen tedaviler ve destekleyici bakım seçenekleri mevcuttur.", "l": "https://www.curesma.org/faqs/"},
        {"c": "Tedavi", "q": "Erken tedavi neden kritik?", "a": "Erken dönemde tedavi ve izlem, motor fonksiyon korunumu açısından daha iyi sonuçlarla ilişkilidir.", "l": "https://www.curesma.org/faqs/"},
        {"c": "Solunum", "q": "Solunum izlemi ne sıklıkla yapılmalı?", "a": "Fonksiyonel duruma göre düzenli solunum değerlendirmesi komplikasyonları azaltmaya yardımcı olur.", "l": "https://www.mda.org"},
        {"c": "Rehabilitasyon", "q": "Fizyoterapi SMA'da gerekli mi?", "a": "Evet; eklem hareket açıklığı, postür ve fonksiyon korunumu için düzenli rehabilitasyon önerilir.", "l": "https://www.curesma.org/faqs/"},
        {"c": "Ortopedi", "q": "Skolyoz takibi neden önemli?", "a": "Omurga eğriliği solunum ve oturma dengesini etkileyebildiği için düzenli takip gerekir.", "l": "https://www.curesma.org/faqs/"},
        {"c": "Beslenme", "q": "Beslenme ve yutma desteği gerekir mi?", "a": "Bazı hastalarda yutma-güvenli beslenme planı ve beslenme desteği önemli olabilir.", "l": "https://www.curesma.org/faqs/"},
        {"c": "Takip", "q": "HFMSE ve RULM neden takip edilir?", "a": "Bu ölçekler alt ve üst ekstremite fonksiyon değişimini izlemeye yardımcı olur.", "l": "https://www.curesma.org/faqs/"},
        {"c": "Destek", "q": "Aileler için destek kaynakları var mı?", "a": "Hasta dernekleri, bakım rehberleri ve akran toplulukları uzun dönem bakımda önemli destek sağlar.", "l": "https://www.curesma.org/faqs/"},
        {"c": "Acil", "q": "Hangi bulgular acil değerlendirme gerektirir?", "a": "Artan solunum sıkıntısı, sekresyon birikimi veya beslenememe durumunda acil başvuru gerekir.", "l": "https://www.mda.org"},
    ],
    WORKSPACE_HUNTINGTON: [
        {"c": "Temel", "q": "Huntington hastalığı nedir?", "a": "Huntington, kalıtsal ve ilerleyici bir nörodejeneratif hastalıktır; motor, bilişsel ve psikiyatrik belirtilerle seyreder.", "l": "https://www.ninds.nih.gov/health-information/disorders/huntingtons-disease"},
        {"c": "Genetik", "q": "Huntington kalıtsal mıdır?", "a": "Evet; otozomal dominant geçiş gösterir ve aile bireyleri için genetik danışmanlık önemlidir.", "l": "https://www.ninds.nih.gov/health-information/disorders/huntingtons-disease"},
        {"c": "Genetik", "q": "Prediktif genetik test ne zaman düşünülür?", "a": "At-risk bireylerde psikososyal ve etik danışmanlıkla birlikte uzman merkezlerde değerlendirilir.", "l": "https://hdsa.org/hdsa-helpline/"},
        {"c": "Belirti", "q": "En sık erken belirtiler nelerdir?", "a": "Denge-motor değişiklikler, yürütücü işlev bozulması ve davranış-duygu değişiklikleri görülebilir.", "l": "https://www.ninds.nih.gov/health-information/disorders/huntingtons-disease"},
        {"c": "Belirti", "q": "Korea her hastada olur mu?", "a": "Sık görülse de her hastada aynı düzeyde olmayabilir; bazı hastalarda rijidite-akinezi baskın olabilir.", "l": "https://www.ninds.nih.gov/health-information/disorders/huntingtons-disease"},
        {"c": "Tanı", "q": "Tanı nasıl konur?", "a": "Nörolojik değerlendirme, aile öyküsü ve genetik test sonuçları birlikte ele alınır.", "l": "https://www.ninds.nih.gov/health-information/disorders/huntingtons-disease"},
        {"c": "Seyir", "q": "Hastalığın seyri değişken midir?", "a": "Evet, başlangıç yaşı ve progresyon hızı bireyler arasında farklılık gösterebilir.", "l": "https://www.ninds.nih.gov/health-information/disorders/huntingtons-disease"},
        {"c": "Tedavi", "q": "Kesin tedavi var mı?", "a": "Şu an hastalığı durduran kesin tedavi yoktur; semptom odaklı yaklaşım uygulanır.", "l": "https://www.ninds.nih.gov/health-information/disorders/huntingtons-disease"},
        {"c": "Tedavi", "q": "Motor belirtiler için ilaç seçenekleri var mı?", "a": "Bazı ilaçlar korea ve davranışsal belirtilerin yönetiminde klinik olarak kullanılabilir.", "l": "https://www.ninds.nih.gov/health-information/disorders/huntingtons-disease"},
        {"c": "Psikiyatri", "q": "Depresyon ve intihar riski neden yakından izlenir?", "a": "Psikiyatrik belirtiler hastalık yükünü artırabilir; erken tanı ve destek güvenlik açısından kritiktir.", "l": "https://www.ninds.nih.gov/health-information/disorders/huntingtons-disease"},
        {"c": "Beslenme", "q": "Kilo kaybı ve yutma sorunları nasıl yönetilir?", "a": "Beslenme danışmanlığı ve yutma değerlendirmesi aspirasyon riskini azaltmada önemlidir.", "l": "https://www.ninds.nih.gov/health-information/disorders/huntingtons-disease"},
        {"c": "Bakım", "q": "Aile için bakım planı nasıl yapılır?", "a": "İşlevsel hedefler, davranış yönetimi ve bakım veren desteği birlikte planlanmalıdır.", "l": "https://hdsa.org/hdsa-helpline/"},
        {"c": "Destek", "q": "HDSA helpline hangi konularda destek verir?", "a": "Semptom yönetimi, genetik test yönlendirmesi, yerel kaynaklar ve destek grupları için rehberlik sağlar.", "l": "https://hdsa.org/hdsa-helpline/"},
        {"c": "Araştırma", "q": "Klinik araştırmalara katılım mümkün mü?", "a": "Uygun adaylar için klinik araştırma merkezleri ve resmi kayıt platformları takip edilmelidir.", "l": "https://clinicaltrials.gov"},
        {"c": "Acil", "q": "Hangi durumda acil yardım gerekir?", "a": "Ciddi davranışsal kriz, intihar düşüncesi, yutma-solunum riski veya travmatik düşmelerde acil başvuru gerekir.", "l": "https://hdsa.org/hdsa-helpline/"},
    ],
    WORKSPACE_LEWY: [
        {"c": "Temel", "q": "Lewy cisimcikli demans (LBD) nedir?", "a": "LBD, DLB ve Parkinson hastalığı demansını kapsayan, bilişsel ve motor belirtilerle seyreden bir demans grubudur.", "l": "https://www.lewybody.org/information-and-support/faqs/"},
        {"c": "Tanı", "q": "DLB ile Parkinson demansı farkı nedir?", "a": "Kognitif belirtilerin motor bulgulara göre zamanlaması tanısal ayrımda temel yaklaşımdır.", "l": "https://www.lewybody.org/information-and-support/faqs/"},
        {"c": "Belirti", "q": "En tipik belirtiler hangileridir?", "a": "Bilişsel dalgalanma, görsel halüsinasyon, parkinsonizm ve REM uyku davranış bozukluğu sık görülür.", "l": "https://www.mayoclinic.org/diseases-conditions/lewy-body-dementia/diagnosis-treatment/drc-20352030"},
        {"c": "Belirti", "q": "Neden yanlış tanı sık olur?", "a": "LBD belirtileri Alzheimer ve Parkinson ile örtüşebildiği için erken dönemde karışabilir.", "l": "https://www.lewybody.org/information-and-support/faqs/"},
        {"c": "Tanı", "q": "Kesin tanı için tek test var mı?", "a": "Hayır; tanı klinik özelliklerin birlikte değerlendirilmesiyle konur.", "l": "https://www.mayoclinic.org/diseases-conditions/lewy-body-dementia/diagnosis-treatment/drc-20352030"},
        {"c": "Tedavi", "q": "LBD'nin küratif tedavisi var mı?", "a": "Şu anda kesin kür yoktur; semptom odaklı ve kişiselleştirilmiş tedavi uygulanır.", "l": "https://www.mayoclinic.org/diseases-conditions/lewy-body-dementia/diagnosis-treatment/drc-20352030"},
        {"c": "İlaç", "q": "Antipsikotik duyarlılığı neden önemli?", "a": "Bazı ilaçlar LBD'de ciddi yan etki oluşturabileceğinden tedavi uzman gözetiminde yapılmalıdır.", "l": "https://www.mayoclinic.org/diseases-conditions/lewy-body-dementia/diagnosis-treatment/drc-20352030"},
        {"c": "Uyku", "q": "REM uyku davranış bozukluğu nasıl ele alınır?", "a": "Uyku güvenliği ve medikal yaklaşım birlikte planlanmalı, düşme-travma riski azaltılmalıdır.", "l": "https://www.mayoclinic.org/diseases-conditions/lewy-body-dementia/diagnosis-treatment/drc-20352030"},
        {"c": "Günlük Yaşam", "q": "Dalgalanan bilişsel durumla günlük plan nasıl yapılır?", "a": "Kısa görevler, düşük uyaranlı ortam ve düzenli rutinler işlevselliği artırabilir.", "l": "https://www.lbda.org"},
        {"c": "Motor", "q": "Yürüme ve düşme riski nasıl yönetilir?", "a": "Fizyoterapi, çevre düzenlemesi ve ortostatik semptom takibi birlikte yapılmalıdır.", "l": "https://www.mayoclinic.org/diseases-conditions/lewy-body-dementia/diagnosis-treatment/drc-20352030"},
        {"c": "Otonom", "q": "Tansiyon dalgalanması ve bayılma görülebilir mi?", "a": "Evet, otonom disfonksiyon LBD'de görülebilir ve düzenli değerlendirme gerektirir.", "l": "https://www.mayoclinic.org/diseases-conditions/lewy-body-dementia/diagnosis-treatment/drc-20352030"},
        {"c": "Bakım", "q": "Bakım verenler için en önemli öncelikler nelerdir?", "a": "İlaç güvenliği, düşme önleme, uyku düzeni ve davranış yönetimi temel önceliklerdir.", "l": "https://www.lbda.org"},
        {"c": "Destek", "q": "Hasta ve aile destek kaynakları nerede bulunur?", "a": "Lewy Body dernekleri ve demans destek hatları yönlendirme ve eğitim sağlar.", "l": "https://www.lewybody.org/information-and-support/faqs/"},
        {"c": "Seyir", "q": "Hastalık zamanla kötüleşir mi?", "a": "LBD ilerleyici bir hastalıktır; semptomların şiddeti ve hızı bireye göre değişebilir.", "l": "https://www.lewybody.org/information-and-support/faqs/"},
        {"c": "Acil", "q": "Hangi durumda acil başvuru yapılmalı?", "a": "Ani bilinç değişikliği, ciddi halüsinasyon krizi, düşme veya yutma-solunum sorunu varsa acil değerlendirme gerekir.", "l": "https://www.lbda.org"},
    ],
    WORKSPACE_FTD: [
        {"c": "Temel", "q": "Frontotemporal demans (FTD) nedir?", "a": "FTD, frontal ve temporal beyin bölgelerini etkileyen, davranış ve dil değişiklikleriyle öne çıkan bir demans grubudur.", "l": "https://www.theaftd.org/aftd-helpline/"},
        {"c": "Belirti", "q": "FTD'nin en sık erken belirtileri nelerdir?", "a": "Davranış değişikliği, empati azalması, dürtüsellik veya dilde bozulma sık erken bulgulardır.", "l": "https://www.theaftd.org/aftd-helpline/"},
        {"c": "Belirti", "q": "FTD ile Alzheimer arasındaki temel fark nedir?", "a": "FTD'de başlangıçta bellekten çok davranış ve dil alanları daha belirgin etkilenebilir.", "l": "https://www.theaftd.org/aftd-helpline/"},
        {"c": "Alt Tip", "q": "Pick hastalığı FTD ile aynı mıdır?", "a": "Pick hastalığı, FTD spektrumundaki belirli patolojik alt tiplerden biri olarak değerlendirilir.", "l": "https://www.theaftd.org/aftd-helpline/"},
        {"c": "Genetik", "q": "FTD kalıtsal olabilir mi?", "a": "Evet, bazı FTD olguları ailesel geçiş gösterebilir; genetik danışmanlık uygun olabilir.", "l": "https://www.theaftd.org/aftd-helpline/"},
        {"c": "Tanı", "q": "FTD tanısı nasıl konur?", "a": "Nörolojik değerlendirme, nöropsikolojik testler ve görüntüleme bulguları birlikte değerlendirilir.", "l": "https://www.theaftd.org/aftd-helpline/"},
        {"c": "Tedavi", "q": "FTD için kesin tedavi var mı?", "a": "Şu an küratif tedavi yoktur; semptom yönetimi ve güvenlik odaklı bakım esastır.", "l": "https://www.theaftd.org/aftd-helpline/"},
        {"c": "Davranış", "q": "Dürtüsellik ve uygunsuz davranışlar nasıl yönetilir?", "a": "Yapılandırılmış rutin, çevresel sınırlar ve bakım veren eğitimi temel yaklaşımı oluşturur.", "l": "https://www.theaftd.org/aftd-helpline/"},
        {"c": "Dil", "q": "Dil bozukluğu olan FTD alt tiplerinde ne yapılır?", "a": "Konuşma-dil terapisi ve iletişim stratejileri günlük işlevi destekleyebilir.", "l": "https://www.theaftd.org/aftd-helpline/"},
        {"c": "Güvenlik", "q": "Araç kullanımı ne zaman bırakılmalı?", "a": "Güvenlik riski geliştiğinde hekim değerlendirmesiyle araç kullanımı sonlandırılmalıdır.", "l": "https://www.theaftd.org/aftd-helpline/"},
        {"c": "Bakım", "q": "Aile bakım planı nasıl oluşturulur?", "a": "Davranış profili, günlük işlev kaybı ve bakım veren kapasitesi birlikte ele alınmalıdır.", "l": "https://www.theaftd.org/aftd-helpline/"},
        {"c": "Psikososyal", "q": "Bakım veren tükenmişliği nasıl azaltılır?", "a": "Mola bakımı, destek grupları ve profesyonel danışmanlık bakım yükünü azaltabilir.", "l": "https://www.theaftd.org/aftd-helpline/"},
        {"c": "Seyir", "q": "Hastalık seyri öngörülebilir mi?", "a": "Seyir bireye göre değişkendir; düzenli izlem ve bakım planı güncellemesi gerekir.", "l": "https://www.theaftd.org/aftd-helpline/"},
        {"c": "Destek", "q": "AFTD HelpLine hangi konularda yardımcı olur?", "a": "Alt tip bilgisi, yeni tanı yönetimi, kaynak yönlendirmesi ve duygusal destek sağlar.", "l": "https://www.theaftd.org/aftd-helpline/"},
        {"c": "Acil", "q": "Ne zaman acil destek alınmalı?", "a": "Kendine-çevreye zarar riski, ciddi davranışsal kriz veya ani tıbbi kötüleşmede acil başvuru gerekir.", "l": "https://www.theaftd.org/aftd-helpline/"},
    ],
}
def _workspace_filter_faq_items(
    workspace_mode: str,
    query: str = "",
    category: str = "Tümü",
) -> list[dict[str, str]]:
    items = WORKSPACE_FAQ_DATA.get(workspace_mode, [])
    q = str(query or "").strip().lower()
    cat = str(category or "Tümü").strip()
    out: list[dict[str, str]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        it_cat = str(it.get("c", "Genel") or "Genel")
        if cat != "Tümü" and it_cat != cat:
            continue
        text_blob = f"{it.get('q', '')} {it.get('a', '')} {it_cat}".lower()
        if q and q not in text_blob:
            continue
        out.append(it)
    return out


def _render_workspace_faq_page(workspace_mode: str) -> None:
    st.subheader(f"{workspace_mode} | Sıkça Sorulan Sorular")
    base_items = WORKSPACE_FAQ_DATA.get(workspace_mode, [])
    categories = sorted({str(it.get("c", "Genel") or "Genel") for it in base_items if isinstance(it, dict)})
    c1, c2 = st.columns([2, 1])
    with c1:
        q = st.text_input("SSS içinde ara", "", key=f"ws_faq_search::{workspace_mode}")
    with c2:
        selected_category = st.selectbox(
            "Kategori",
            options=["Tümü"] + categories,
            index=0,
            key=f"ws_faq_category::{workspace_mode}",
        )

    faq_items = _workspace_filter_faq_items(workspace_mode, query=q, category=selected_category)
    st.caption(f"Toplam sonuç: {len(faq_items)}")
    if not faq_items:
        st.info("Bu hastalık için filtreye uyan SSS kaydı bulunmuyor.")
        return

    for idx, item in enumerate(faq_items, start=1):
        question = str(item.get("q", "")).strip()
        answer = str(item.get("a", "")).strip()
        category = str(item.get("c", "Genel") or "Genel").strip()
        link = _safe_link(str(item.get("l", "")).strip())
        title = f"{idx}. [{category}] {question or 'Soru'}"
        with st.expander(title):
            st.write(answer or "-")
            if link != "#":
                st.markdown(f"[Kaynak]({link})")

def _workspace_history_records(workspace_mode: str) -> list[dict]:
    hist_key = _neuro_history_state_key(workspace_mode)
    hist = st.session_state.get(hist_key, [])
    return hist if isinstance(hist, list) else []


def _workspace_note_key(workspace_mode: str) -> str:
    return f"{_active_patient_scope_key()}::{workspace_mode}"


def _workspace_history_df(workspace_mode: str) -> pd.DataFrame:
    hist = _workspace_history_records(workspace_mode)
    return pd.DataFrame(hist) if hist else pd.DataFrame()


def _workspace_latest_record(workspace_mode: str) -> dict:
    hist = _workspace_history_records(workspace_mode)
    if not hist:
        return {}
    return hist[-1] if isinstance(hist[-1], dict) else {}


def _render_disease_workspace_dashboard(workspace_mode: str) -> None:
    st.subheader(f"{workspace_mode} | Ana Panel")
    _clinical_evidence_caption("Bilgilendirme", "Trend g?r?n?m? ?zet ama?l?d?r; klinik karar multidisipliner de?erlendirme ile verilmelidir.")
    latest = _workspace_latest_record(workspace_mode)
    hist = _workspace_history_records(workspace_mode)
    pidx = float(latest.get("progression_index", 0.0) or 0.0)
    c1, c2, c3 = st.columns(3)
    c1.metric("Son Progresyon İndeksi", f"{pidx:.1f} / 100")
    c2.metric("Toplam Ölçüm", str(len(hist)))
    c3.metric("Son Kayıt", str(latest.get("time", "-")))
    if pidx >= 66:
        st.error("Yüksek etkilenim düzeyi: yakın izlem önerilir.")
    elif pidx >= 33:
        st.warning("Orta etkilenim düzeyi: takip planı sıklaştırılmalı.")
    else:
        st.success("Düşük etkilenim düzeyi.")
    df = _workspace_history_df(workspace_mode)
    if not df.empty and "time" in df.columns and "progression_index" in df.columns:
        if PLOTLY_OK:
            fig = px.line(df, x="time", y="progression_index", markers=True, title=f"{workspace_mode} Progresyon Eğrisi")
            fig.update_layout(height=320, xaxis_title="", yaxis_title="Progresyon İndeksi")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(df.set_index("time")[["progression_index"]], use_container_width=True)


def _render_disease_workspace_calculator(workspace_mode: str) -> None:
    handler = WORKSPACE_DASHBOARD_HANDLERS.get(workspace_mode)
    if handler is not None:
        _clinical_evidence_caption("Karar-Destek", "Skorlar izlem amaclidir; tedavi karari tek bir olcume dayandirilmamalidir.")
        handler()
        return

    st.subheader(f"{workspace_mode} | Klinik Hesaplayıcı")
    _ensure_neuro_dynamic_state(workspace_mode)
    cfg = NEURO_DYNAMIC_CONFIG.get(workspace_mode, {})
    fields = cfg.get("fields", [])
    state_key = _neuro_input_state_key(workspace_mode)
    state_vals = st.session_state.get(state_key, {})
    cols = st.columns(len(fields)) if fields else []
    for i, f in enumerate(fields):
        with cols[i]:
            k = str(f.get("key"))
            vv = st.number_input(
                str(f.get("label", k)),
                min_value=float(f.get("min", 0)),
                max_value=float(f.get("max", 100)),
                value=float(state_vals.get(k, f.get("default", 0))),
                key=f"calc::{_active_patient_scope_key()}::{workspace_mode}::{k}",
            )
            state_vals[k] = float(vv)
    st.session_state[state_key] = state_vals
    analiz = analiz_yap(workspace_mode, state_vals)
    pidx = float(analiz.get("progression_index", 0.0))
    st.metric("Hesaplanan Progresyon İndeksi", f"{pidx:.1f} / 100")
    if st.button("Hesaplayıcı Ölçümünü Kaydet", key=f"save_calc::{workspace_mode}", use_container_width=True):
        rec = {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "progression_index": pidx, **state_vals}
        _save_neuro_history_record(workspace_mode, rec)
        save_current_session_profile()
        st.success("Kayıt eklendi.")


def _render_disease_workspace_full_test(workspace_mode: str) -> None:
    st.markdown(
        f"<h3 style='text-align:center; margin-bottom: 8px;'>{workspace_mode} | Tam ?l?ekli Test</h3>",
        unsafe_allow_html=True,
    )
    _clinical_evidence_caption(
        "Karar-Destek",
        "Bu b?l?mdeki skorlar izlem ama?l?d?r; tan? ve tedavi karar? tek ba??na bu testten verilmez.",
    )

    model = WORKSPACE_FULL_TEST_MODELS.get(workspace_mode, {})
    items = model.get("items", [])
    if not items:
        st.info("Bu hastal?k i?in tam ?l?ekli test modeli tan?ml? de?il.")
        return

    scope = _active_patient_scope_key()
    scores_key = f"fulltest::{scope}::{workspace_mode}"

    min_defaults = [int(it.get("min", 0)) for it in items]
    scores = st.session_state.get(scores_key, min_defaults)
    if not isinstance(scores, list) or len(scores) != len(items):
        scores = list(min_defaults)

    c1, c2 = st.columns(2)
    total = 0
    max_total = 0
    for i, it in enumerate(items):
        label = str(it.get("label", f"Madde {i+1}"))
        min_v = int(it.get("min", 0))
        max_v = int(it.get("max", 4))
        max_total += max_v
        target_col = c1 if i % 2 == 0 else c2
        with target_col:
            st.markdown(f"**{i+1}. {label}**")
            v = st.slider(
                f"{workspace_mode}_test_{i}",
                min_value=min_v,
                max_value=max_v,
                value=int(scores[i]),
                key=f"fulltest_slider::{scope}::{workspace_mode}::{i}",
                label_visibility="collapsed",
            )
            scores[i] = int(v)
            total += int(v)

    st.session_state[scores_key] = scores
    max_total = max(1, int(max_total))

    higher_is_better = bool(model.get("higher_is_better", True))
    raw_pct = (float(total) / float(max_total)) * 100.0
    burden_pct = (100.0 - raw_pct) if higher_is_better else raw_pct
    burden_pct = max(0.0, min(100.0, burden_pct))

    m1, m2, m3 = st.columns(3)
    m1.metric("Test Toplam", f"{total} / {max_total}")
    m2.metric("Klinik Y?k", f"{burden_pct:.1f} / 100")
    m3.metric("Test Modeli", str(model.get("name", "-")))

    st.progress(min(1.0, max(0.0, burden_pct / 100.0)))

    sev, sev_msg = _workspace_full_test_interpretation(workspace_mode, burden_pct)
    if sev == "high":
        st.error(sev_msg)
    elif sev == "mid":
        st.warning(sev_msg)
    else:
        st.success(sev_msg)

    personalize_key = f"fulltest_personalize::{scope}::{workspace_mode}"
    personalize_followup = st.checkbox(
        "Takip aral???n? bireyselle?tir (risk parametrelerine g?re)",
        value=True,
        key=personalize_key,
    )
    plan = _workspace_next_control_plan_personalized(workspace_mode, sev, enabled=bool(personalize_followup))
    st.info(f"?nerilen bir sonraki kontrol: {plan.get('interval', '3 ay')} i?inde.")
    _clinical_evidence_caption(plan.get("evidence", "Bilgilendirme"), plan.get("basis", ""))

    if PLOTLY_OK:
        df_items = pd.DataFrame(
            [
                {
                    "item": str(items[idx].get("label", idx + 1)),
                    "score": float(scores[idx]),
                    "max": float(items[idx].get("max", 4)),
                }
                for idx in range(len(items))
            ]
        )
        if not df_items.empty:
            df_items["ratio"] = (df_items["score"] / df_items["max"]).fillna(0.0)
            fig = px.bar(
                df_items,
                x="item",
                y="score",
                color="ratio",
                color_continuous_scale="RdYlGn" if higher_is_better else "RdYlGn_r",
                title=f"{workspace_mode} Tam ?l?ekli Test Da??l?m?",
            )
            fig.update_layout(height=340, xaxis_title="", yaxis_title="Skor")
            st.plotly_chart(fig, use_container_width=True)

    if st.button("Tam ?l?ekli Testi Kaydet", key=f"save_fulltest::{workspace_mode}", use_container_width=True):
        rec = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "full_test_name": str(model.get("name", "custom")),
            "full_test_total": float(total),
            "full_test_pct": float(round(raw_pct, 2)),
            "progression_index": float(round(burden_pct, 2)),
            "full_test_items": list(scores),
            "full_test_item_labels": [str(it.get("label", "")) for it in items],
        }
        _save_neuro_history_record(workspace_mode, rec)
        save_current_session_profile()
        st.success("Tam ?l?ekli test kaydedildi.")


def _render_disease_workspace_ops(workspace_mode: str) -> None:
    st.subheader(f"{workspace_mode} | Klinik Operasyon Merkezi")
    df = _workspace_history_df(workspace_mode)
    if df.empty:
        st.info("Hen?z operasyon kayd? bulunmuyor.")
        return
    st.dataframe(df.tail(30), use_container_width=True, hide_index=True)
    out = io.StringIO()
    df.tail(200).to_csv(out, index=False)
    st.download_button(
        "Kay?tlar? CSV ?ndir",
        data=out.getvalue().encode("utf-8"),
        file_name=f"{workspace_mode.lower().replace(' ', '_')}_records.csv",
        mime="text/csv",
        use_container_width=True,
    )


def _render_disease_workspace_calendar(workspace_mode: str) -> None:
    st.subheader(f"{workspace_mode} | Klinik Takvim & Haklar")
    with st.form(f"ws_reminder_form::{workspace_mode}", clear_on_submit=True):
        rd = st.date_input("Tarih", key=f"ws_reminder_date::{workspace_mode}")
        rt = st.text_input("Ba?l?k", key=f"ws_reminder_title::{workspace_mode}")
        rn = st.text_input("Not", key=f"ws_reminder_note::{workspace_mode}")
        ok = st.form_submit_button("Randevu Ekle")
        if ok and rt.strip():
            arr = st.session_state.get("reminders", [])
            if not isinstance(arr, list):
                arr = []
            arr.append({"date": str(rd), "title": f"[{workspace_mode}] {rt.strip()}", "note": rn.strip()})
            st.session_state["reminders"] = arr
            save_current_session_profile()
            st.success("Randevu eklendi.")
    ws_rem = [r for r in st.session_state.get("reminders", []) if f"[{workspace_mode}]" in str(r.get("title", ""))]
    if ws_rem:
        st.dataframe(pd.DataFrame(ws_rem).tail(20), use_container_width=True, hide_index=True)
    else:
        st.info("Bu hastal?k i?in randevu kayd? yok.")


def _render_disease_workspace_emergency(workspace_mode: str) -> None:
    st.subheader(f"{workspace_mode} | Acil Durum & Kritik Bak?m")
    _clinical_evidence_caption("Acil Uyar?", "Bu b?l?m acil triyaj hat?rlat?c?s?d?r; lokal acil protokoller ve hekim de?erlendirmesi esast?r.")
    alerts = {
        WORKSPACE_ALS: ["Ani solunum s?k?nt?s?", "Sekresyon temizleyememe", "H?zl? bulber k?t?le?me"],
        WORKSPACE_ALZHEIMER: ["Ani bilin? de?i?ikli?i", "Yeni n?rolojik defisit", "?iddetli ajitasyon/dehidratasyon"],
        WORKSPACE_PARKINSON: ["Ani donakalma/d??me", "Aspirasyon riski", "?la? kesintisine ba?l? k?t?le?me"],
        WORKSPACE_HUNTINGTON: ["Yutma bozuklu?u ve aspirasyon", "Ciddi davran??sal kriz", "H?zl? fonksiyon kayb?"],
        WORKSPACE_LEWY: ["?iddetli hal?sinasyon-konf?zyon", "D??me/senkop", "Ani mobilite kayb?"],
        WORKSPACE_FTD: ["Tehlikeli davran??sal dezorganizasyon", "Beslenme/hidrasyon bozulmas?", "Bak?m veren g?venlik riski"],
        WORKSPACE_SMA: ["Solunum efor art???", "Sekresyon retansiyonu", "Ani g?? kayb?"],
    }
    for a in alerts.get(workspace_mode, []):
        st.error(f"- {a}")


def _render_disease_workspace_vision(workspace_mode: str) -> None:
    st.markdown(
        f"<h3 style='text-align:center; margin-bottom: 8px;'>{workspace_mode} | Vizyon</h3>",
        unsafe_allow_html=True,
    )

    vision_map = {
        WORKSPACE_ALS: {
            "mission": "ALS bak?m?nda fonksiyonel kayb? erken yakalay?p solunum odakl? riskleri daha ?nce g?r?n?r k?lmak.",
            "pillars": [
                "ALSFRS-R ve solunum trendlerinin tek panelde birle?tirilmesi",
                "Bulber ve sekresyon riskinde erken alarm mekanizmas?",
                "Bak?m veren y?k?n? azaltan sade takip ak???",
            ],
        },
        WORKSPACE_ALZHEIMER: {
            "mission": "Bili?sel gerilemeyi sadece skorla de?il g?nl?k ya?am etkisiyle birlikte izleyen etik bir dijital klinik ak??? kurmak.",
            "pillars": [
                "MMSE/MoCA + ADL birlikteli?i ile evreye duyarl? takip",
                "Davran??sal belirtiler i?in g?venlik odakl? planlama",
                "Aile ve bak?m veren ileti?imi i?in standart raporlama",
            ],
        },
        WORKSPACE_PARKINSON: {
            "mission": "Motor semptom, d??me riski ve tedavi zamanlamas?n? tek bir klinik ritimde y?netmek.",
            "pillars": [
                "UPDRS ve denge metrikleriyle progresyon izleme",
                "D??me ?nleme odakl? erken m?dahale i?aretleri",
                "Fonksiyon kayb?n? geciktirmeye y?nelik multidisipliner plan",
            ],
        },
        WORKSPACE_HUNTINGTON: {
            "mission": "Motor, kognitif ve fonksiyonel de?i?imi birlikte de?erlendiren s?rekli ve ?zenli bir takip modeli sunmak.",
            "pillars": [
                "UHDRS/TFC ekseninde progresyonun sade g?r?n?m?",
                "G?venlik ve ba??ms?zl?k hedeflerinin erken g?ncellenmesi",
                "Aile ile uzun d?nem bak?m plan?n?n e?it zamanl? y?netimi",
            ],
        },
        WORKSPACE_LEWY: {
            "mission": "Dalgalanma ve hal?sinasyon a??rl???n? g?venlik odakl? ?ekilde erken fark eden bir klinik koordinasyon sistemi kurmak.",
            "pillars": [
                "Core semptomlar?n d?zenli ve yap?sal takibi",
                "Geceleri artan riskler i?in erken uyar? modeli",
                "N?ro-psikiyatrik ve motor izlemi tek ?at? alt?nda toplamak",
            ],
        },
        WORKSPACE_FTD: {
            "mission": "Davran??sal ve dil bozukluklar?n? bak?m veren g?venli?i ile birlikte izleyen insan-merkezli bir sistem geli?tirmek.",
            "pillars": [
                "Davran??sal y?k ve ADL kayb?n?n birlikte takibi",
                "Dil alt tiplerinde hedefli izlem notasyonu",
                "Bak?m veren t?keni?ini azaltan pratik klinik aksiyonlar",
            ],
        },
        WORKSPACE_SMA: {
            "mission": "Motor performans ve solunum kapasitesini ?ocuk-ergen-eri?kin s?re?lerinde tutarl? bir kaliteyle izlemek.",
            "pillars": [
                "HFMSE/RULM/FVC trendlerinin ortak klinik paneli",
                "Tedavi yan?t? ve fonksiyon korunumu i?in erken fark sinyalleri",
                "Rehabilitasyon ve solunum plan?n?n d?nemsel optimizasyonu",
            ],
        },
    }

    data = vision_map.get(
        workspace_mode,
        {
            "mission": "N?ro-dejeneratif hastal?klarda klinik kalite, veri b?t?nl??? ve hasta g?venli?ini birlikte y?kselten bir platform geli?tirmek.",
            "pillars": [
                "Risk odakl? izlem",
                "Yap?sal klinik raporlama",
                "Multidisipliner karar deste?i",
            ],
        },
    )

    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #0ea5e9 0%, #22c55e 100%); padding: 22px; border-radius: 16px; color: white; margin-bottom: 16px; text-align:center;">
            <h3 style="margin:0 0 8px 0;">Klinik Vizyon</h3>
            <p style="margin:0; opacity:0.95;">{data.get('mission','')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<h4 style='text-align:center;'>Stratejik Odaklar</h4>", unsafe_allow_html=True)
    for ptxt in data.get("pillars", []):
        st.write(f"- {ptxt}")

    target_map = {
        WORKSPACE_ALS: [
            {"label": "ALSFRS-R y?ll?k d????", "target": "<= 6 puan/y?l", "key": "alsfrs_r", "op": "le", "thr": 6.0},
            {"label": "FVC korunumu", "target": ">= %70", "key": "fvc_pct", "op": "ge", "thr": 70.0},
            {"label": "Acil solunum ba?vurusu", "target": "<= 1/y?l", "key": ""},
        ],
        WORKSPACE_ALZHEIMER: [
            {"label": "MMSE korunumu", "target": ">= 18", "key": "mmse", "op": "ge", "thr": 18.0},
            {"label": "ADL ba??ms?zl?k", "target": ">= %60", "key": "adl_independence", "op": "ge", "thr": 60.0},
            {"label": "Bak?m krizi ba?vurusu", "target": "<= 1/y?l", "key": ""},
        ],
        WORKSPACE_PARKINSON: [
            {"label": "UPDRS toplam", "target": "<= 40", "key": "updrs_total", "op": "le", "thr": 40.0},
            {"label": "Ayl?k d??me say?s?", "target": "<= 1", "key": "falls_month", "op": "le", "thr": 1.0},
            {"label": "TUG", "target": "< 13.5 sn", "key": "tug_sec", "op": "lt", "thr": 13.5},
        ],
        WORKSPACE_HUNTINGTON: [
            {"label": "UHDRS motor", "target": "<= 40", "key": "uhdrs_motor", "op": "le", "thr": 40.0},
            {"label": "TFC korunumu", "target": ">= 7", "key": "tfc", "op": "ge", "thr": 7.0},
            {"label": "Acil davran??sal kriz", "target": "<= 1/y?l", "key": ""},
        ],
        WORKSPACE_LEWY: [
            {"label": "MMSE korunumu", "target": ">= 18", "key": "mmse", "op": "ge", "thr": 18.0},
            {"label": "Hal?sinasyon y?k?", "target": "<= 4", "key": "hallucination", "op": "le", "thr": 4.0},
            {"label": "G?venlik olay?", "target": "<= 1/y?l", "key": ""},
        ],
        WORKSPACE_FTD: [
            {"label": "FBI seviyesi", "target": "<= 30", "key": "fbi", "op": "le", "thr": 30.0},
            {"label": "ADL kay?p seviyesi", "target": "<= 5", "key": "adl_loss", "op": "le", "thr": 5.0},
            {"label": "Bak?m veren risk olay?", "target": "<= 1/y?l", "key": ""},
        ],
        WORKSPACE_SMA: [
            {"label": "HFMSE korunumu", "target": ">= 30", "key": "hfmse", "op": "ge", "thr": 30.0},
            {"label": "RULM korunumu", "target": ">= 20", "key": "rulm", "op": "ge", "thr": 20.0},
            {"label": "FVC korunumu", "target": ">= %70", "key": "fvc_pct", "op": "ge", "thr": 70.0},
        ],
    }

    st.markdown("<h4 style='text-align:center;'>2026 Hedef Metrikleri</h4>", unsafe_allow_html=True)
    latest = _workspace_latest_record(workspace_mode)
    goals = target_map.get(workspace_mode, [])
    if goals:
        gcols = st.columns(len(goals))
        for i, g in enumerate(goals):
            k = str(g.get("key", "")).strip()
            cur = "-"
            if k and isinstance(latest, dict) and k in latest:
                try:
                    val = float(latest.get(k))
                    cur = f"{val:.1f}"
                except Exception:
                    cur = str(latest.get(k))
            op = str(g.get("op", "")).strip().lower()
            thr = g.get("thr", None)
            status = "Bilgi yok"
            status_color = "#64748b"
            if cur != "-" and op and isinstance(thr, (int, float)):
                try:
                    vcur = float(cur)
                    vthr = float(thr)
                    if op == "ge":
                        if vcur >= vthr:
                            status, status_color = "Hedefe Uygun", "#16a34a"
                        elif vcur >= (0.9 * vthr):
                            status, status_color = "Dikkat", "#d97706"
                        else:
                            status, status_color = "Kritik", "#dc2626"
                    elif op == "le":
                        if vcur <= vthr:
                            status, status_color = "Hedefe Uygun", "#16a34a"
                        elif vcur <= (1.25 * vthr):
                            status, status_color = "Dikkat", "#d97706"
                        else:
                            status, status_color = "Kritik", "#dc2626"
                    elif op == "lt":
                        if vcur < vthr:
                            status, status_color = "Hedefe Uygun", "#16a34a"
                        elif vcur <= (1.15 * vthr):
                            status, status_color = "Dikkat", "#d97706"
                        else:
                            status, status_color = "Kritik", "#dc2626"
                except Exception:
                    status, status_color = "Bilgi yok", "#64748b"

            with gcols[i]:
                st.metric(str(g.get("label", "Hedef")), cur)
                st.markdown(f"<div style='text-align:center; font-size:0.85rem; color:#64748b;'>Hedef: {g.get('target', '-')}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align:center; font-size:0.85rem; color:{status_color};'><b>Durum:</b> {status}</div>", unsafe_allow_html=True)
    else:
        st.info("Bu hastal?k i?in hedef metrik seti tan?ml? de?il.")

    _clinical_evidence_caption(
        "Bilgilendirme",
        "Vizyon sayfas? ?r?n y?n? ve klinik kalite hedeflerini ?zetler; tan?/tedavi karar? yerine ge?mez.",
    )


def _render_disease_workspace_page(workspace_mode: str, page_label: str) -> None:
    if page_label == DISEASE_WORKSPACE_NAV_PAGES[0]:
        _render_disease_workspace_dashboard(workspace_mode)
        return

    if page_label == DISEASE_WORKSPACE_NAV_PAGES[1]:
        _render_disease_workspace_calculator(workspace_mode)
        return

    if page_label == DISEASE_WORKSPACE_NAV_PAGES[2]:
        _render_disease_workspace_full_test(workspace_mode)
        return

    if page_label == DISEASE_WORKSPACE_NAV_PAGES[3]:
        _render_disease_workspace_ops(workspace_mode)
        return

    if page_label == DISEASE_WORKSPACE_NAV_PAGES[4]:
        _render_disease_workspace_calendar(workspace_mode)
        return

    if page_label == DISEASE_WORKSPACE_NAV_PAGES[5]:
        _render_disease_workspace_emergency(workspace_mode)
        return

    if page_label == DISEASE_WORKSPACE_NAV_PAGES[6]:
        _render_workspace_faq_page(workspace_mode)
        return

    if page_label == DISEASE_WORKSPACE_NAV_PAGES[7]:
        st.subheader(f"{workspace_mode} | Güncel Haberler")
        st.caption("Kaynak: Google News RSS. Başlıklara tıklayarak haberin tamamını açabilirsiniz.")

        lang_v = st.session_state.get("lang", "TR")
        query = WORKSPACE_NEWS_QUERY.get(workspace_mode, workspace_mode)

        c_news1, c_news2 = st.columns([1, 1])
        with c_news1:
            if st.button("Haberleri Yenile", key=f"ws_news_refresh::{workspace_mode}", use_container_width=True):
                fetch_disease_news.clear()
                st.rerun()
        with c_news2:
            all_news_url = (
                "https://news.google.com/search?q="
                + quote_plus(query)
                + ("&hl=en-US&gl=US&ceid=US:en" if lang_v == "EN" else "&hl=tr&gl=TR&ceid=TR:tr")
            )
            st.link_button("Tüm Haberleri Google News'te Gör", all_news_url, use_container_width=True)

        raw_news_items = fetch_disease_news(query, lang=lang_v, limit=25)
        news_items = list(raw_news_items)

        key_filter = st.text_input("Haber içinde ara", value="", key=f"ws_news_search::{workspace_mode}").strip().lower()
        sources = sorted({n.get("source", "Google News") for n in news_items}) if news_items else []
        source_pick = st.selectbox("Kaynak filtresi", ["Tümü"] + sources, index=0, key=f"ws_news_source::{workspace_mode}")

        if news_items:
            news_items = [
                n for n in news_items
                if (not key_filter or key_filter in str(n.get("title", "")).lower())
                and (source_pick == "Tümü" or str(n.get("source", "")) == source_pick)
            ]

        if not raw_news_items:
            st.info("Haber akışı şu anda boş. Daha sonra tekrar deneyin.")
        elif not news_items:
            st.info("Filtrelere uygun haber bulunamadı. Arama ifadesini veya kaynak filtresini değiştirin.")
        else:
            for i, item in enumerate(news_items, start=1):
                title = html.escape(item.get("title", "Haber"))
                link = _safe_link(item.get("link", "#"))
                source = html.escape(str(item.get("source", "Google News")))
                published = html.escape(str(item.get("published", "")))
                st.markdown(f"{i}. **{title}**")
                meta_parts = [f"Kaynak: {source}"]
                if published:
                    meta_parts.append(f"Tarih: {published}")
                st.caption(" | ".join(meta_parts))
                if link != "#":
                    st.markdown(f"[Haberi Aç]({link})")
                st.markdown("---")
        return

    if page_label == DISEASE_WORKSPACE_NAV_PAGES[8]:
        st.subheader(f"{workspace_mode} | AI Yardım")
        _render_ai_key_settings(scope_key=f"workspace::{workspace_mode}")
        q = st.text_area("Sorunuz", placeholder=f"{workspace_mode} ile ilgili klinik soruyu yazın...", key=f"ai_q::{workspace_mode}")
        if st.button("AI Yanıtı Oluştur", key=f"ai_btn::{workspace_mode}", use_container_width=True):
            q_clean = str(q or "").strip()
            if not q_clean:
                st.error("Lütfen bir soru yazın.")
            elif not _get_openai_api_key():
                st.error("OpenAI API anahtarı bulunamadı. AI Ayarları bölümünden anahtar girin.")
            else:
                context = f"Çalışma alanı: {workspace_mode}\nHasta: {_active_patient_scope_key()}"
                ans = ask_openai_medical_assistant(q_clean, context_text=context)
                st.markdown(f"**Yanıt:** {ans}")
        return

    if page_label == DISEASE_WORKSPACE_NAV_PAGES[9]:
        _render_disease_workspace_vision(workspace_mode)
        return


    st.warning("Geçersiz sayfa seçimi.")

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
    selected_workspace = st.selectbox(
        "Çalışma Alanı Seçin",
        options=WORKSPACE_OPTIONS,
        index=(
            WORKSPACE_OPTIONS.index(
                st.session_state.get("workspace_mode", WORKSPACE_DMD)
            )
            if st.session_state.get("workspace_mode", WORKSPACE_DMD) in WORKSPACE_OPTIONS
            else 0
        ),
        key="workspace_mode_select",
    )
    st.session_state["workspace_mode"] = selected_workspace

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

    if selected_workspace == WORKSPACE_DMD:
        visible_pages = _role_page_labels(D, st.session_state.get("user_role", "family"))
        default_page = visible_pages[0] if visible_pages else D["nav"][0]
        page_options = visible_pages if visible_pages else [default_page]
    else:
        default_page = DISEASE_WORKSPACE_NAV_PAGES[0]
        page_options = list(DISEASE_WORKSPACE_NAV_PAGES)
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

workspace_mode = st.session_state.get("workspace_mode", WORKSPACE_DMD)
if workspace_mode != WORKSPACE_DMD:
    _render_disease_workspace_page(workspace_mode, page)
    st.stop()

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
                direction = "⬇ Declining"
            elif delta > 0:
                direction = "⬆ Improving"
            else:
                direction = " Stable"

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
            alerts.append(" NSAA düşüşü tespit edildi.")
    except Exception:
        pass

    try:
        ef = mp.get("ef")
        if str(ef).strip() != "" and int(float(ef)) < 55:
            alerts.append(" EF düşük olabilir (kardiyak kontrol önerilir).")
    except Exception:
        pass

    try:
        fvc = mp.get("fvc")
        if str(fvc).strip() != "" and int(float(fvc)) < 80:
            alerts.append(" FVC düşük olabilir (solunum değerlendirmesi önerilir).")
    except Exception:
        pass

    if alerts:
        for a in alerts:
            st.warning(a)
    else:
        st.success(" Kritik uyarı yok")

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
    st.markdown("### DMD Genetik Modül")
    dmd_dashboard()
    st.divider()

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
                ["Yürüme dayanıklılığı", "Üst ekstremite fonksiyonu", "Solunum egzersizi", "Postür/skolyoz takibi", "Okul/iş uyumu", "Ps?kososyal destek"],
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

    # Arama + kategori filtresi (workspace SSS formatı ile uyumlu)
    faq_search = st.text_input("SSS içinde ara", "", key="dmd_faq_search").strip().lower()

    # --- GENİŞLETİLMİŞ VERİ SETİ ---
    faq_data = [
        {
            "cat": " GENETİK VE TEŞHİS",
            "q": "DMD ve BMD Arasındaki Fark Nedir",
            "a": "DMD (Duchenne), distrofin proteininin tamamen eks?k olduğu daha ağır seyreden tiptir. BMD (Becker) ise proteinin az veya kusurlu olduğu, semptomların daha geç ve hafif başladığı formdur.",
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
            "a": "DMD hastalarında kalp kası (miyokard) distrofin eks?kliğinden etkilenir. Belirti olmasa dahi 10 yaşından önce ACE inhibitörleri gibi koruyucu tedavilere başlamak hayati olabilir.",
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

    categories = sorted({str(item.get("cat", "Genel")).strip() for item in faq_data if isinstance(item, dict)})
    _, c_faq_2 = st.columns([2, 1])
    with c_faq_2:
        selected_category = st.selectbox(
            "Kategori",
            options=["Tümü"] + categories,
            index=0,
            key="dmd_faq_category",
        )

    def _match_dmd_faq(item: dict, query: str, category: str) -> bool:
        cat = str(item.get("cat", "Genel")).strip()
        if category != "Tümü" and cat != category:
            return False
        blob = f"{item.get('q', '')} {item.get('a', '')} {cat}".lower()
        return (not query) or (query in blob)

    filtered_faq = [item for item in faq_data if isinstance(item, dict) and _match_dmd_faq(item, faq_search, selected_category)]

    st.caption("Kaynak standardı: Her madde resmi kurum/dernek veya klinik rehber bağlantısı içerir.")
    st.caption(f"Toplam sonuç: {len(filtered_faq)}")

    if not filtered_faq:
        st.info("Filtreye uyan SSS kaydı bulunamadı.")

    for idx, item in enumerate(filtered_faq, start=1):
        category = str(item.get("cat", "Genel")).strip()
        question = str(item.get("q", "")).strip()
        answer = str(item.get("a", "")).strip()
        link = _safe_link(str(item.get("l", "")).strip())
        with st.expander(f"{idx}. [{category}] {question or 'Soru'}"):
            st.write(answer or "-")
            if link != "#":
                st.markdown(f"[Kaynak]({link})")

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
    st.warning("**Tıbbi Personel İçin Özet:** Bu hasta Distrofin eks?kliği (DMD) tanılıdır. Succinylcholine kaçınılmalı, volatil ajanlar dikkatle değerlendirilmelidir.")

    # Acil Servis Butonu ve Genişletilmiş Görünüm
    if st.button("ACİL SERVİS: DOKTORA GÖSTER (TAM EKRAN)"):
        st.markdown("""
            <div style="background-color:#ff4b4b; padding:40px; border-radius:20px; border: 8px solid #ffffff; text-align:center; box-shadow: 0 0 50px rgba(148,163,184,0.22);">
                <h1 style="color:white; font-size:50px; margin-bottom:10px;">KRİTİK UYARI</h1>
                <h2 style="color:white; border-bottom: 2px solid white; padding-bottom:15px;">HASTA DMD (DUCHENNE) TANILIDIR</h2>
                <div style="text-align:left; color:white; font-size:24px; margin-top:20px; line-height:1.6;">
                    <p><b>1. ANESTEZİ:</b> SÜKSİNİLKOLİN VE TÜM GAZLAR (İnhalanlar) <b>KESİNLİKLE YASAK!</b> Sadece TIVA (Propofol vb.) kullanılabilir.</p>
                    <p><b>2. SOLUNUM DESTEĞİ:</b> Oksijen gereksinimi ventilasyon ve CO2 izlemi ile birlikte değerlendirilmelidir; oksijenin tek başına verilmesi hipoventilasyonu maskeleyebilir.</p>
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
        with st.expander(" Ps?kososyal Destek ve Bakım Veren Desteği"):
            st.write("- **Ps?kolojik destek:** Uzun dönem bakım yükü için aile ve birey odaklı danışmanlık planlanabilir.")
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
    _render_ai_key_settings(scope_key="dmd")

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

        /*  NEW: email style */
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
                <div class="vision-icon"></div>
                <h3>Global Klinik Çerçeve</h3>
                <p>WDO ve TREAT-NMD rehberlerini tek akışta birleştirerek, ekiplerin kararlarını ülke bağımsız ve standardize biçimde destekliyoruz.</p>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
            <div class="vision-card">
                <div class="vision-icon"></div>
                <h3>Anlık Müdahale Desteği</h3>
                <p>Acil anlarda kritik riskleri sade ve görünür bir formatta sunarak zaman kaybını azaltıyor, bakım kalitesini yükseltiyoruz.</p>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
            <div class="vision-card">
                <div class="vision-icon"></div>
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

    #  NEW: Role-based visibility for founder/signature panel
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
    <h3 class="cta-title"> Birlikte Daha Güçlü Bir Klinik Gelecek</h3>
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
            st.success("Desteğiniz için teşekkür ederiz ")

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
            st.success("Desteğiniz için teşekkür ederiz ")
    with b3:
        st.caption("")






