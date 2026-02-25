# NIZEN Clinical Platform (`final_v50.py`)

Streamlit tabanlı, çok hastalıklı (DMD + nörodejeneratif workspaceler) klinik karar-destek ve takip uygulaması.

## Kapsam

- DMD ana modülleri: dashboard, klinik hesaplayıcı, NSAA, acil durum, takvim/haklar, SSS, haberler, AI destekli soru-cevap.
- Nöro workspace modları: ALS, Alzheimer, Parkinson, Huntington, Lewy Body Demans, FTD, SMA.
- Rol tabanlı erişim: `family`, `doctor`, `researcher`, `admin`.
- Yerel kalıcılık: SQLite (`data/dmd_local.db`) + JSON kuyruk/yedek dosyaları.
- Opsiyonel bulut senkron: Google Sheets (`streamlit-gsheets`).
- Opsiyonel AI entegrasyonu: OpenAI Responses API.

## Proje Yapısı

- Uygulama: `final_v50.py`
- Test: `tests/test_dmd_exon_phase_map.py`
- Veri klasörü (runtime): `data/`

## Gereksinimler

Minimum:

- Python 3.10+
- `streamlit`
- `pandas`

Opsiyonel:

- `streamlit-gsheets` (Google Sheets senkronu)
- `bcrypt` (şifre hash için)
- `plotly` (grafikler)
- `reportlab` (PDF rapor)

## Çalıştırma

```powershell
streamlit run final_v50.py
```

## Konfigürasyon

### 1) `st.secrets` / environment

Desteklenen ana ayarlar:

- `SHEET_URL` veya env `SHEET_URL`
- `OPENAI_API_KEY` veya env `OPENAI_API_KEY`
- `auth_secret` veya env `AUTH_SECRET`
- `persistent_login_via_query` veya env `DMD_PERSISTENT_LOGIN_QUERY`
- env `DMD_PERSISTENT_LOGIN_TTL_SEC`
- `i18n_patch_enabled` veya env `DMD_ENABLE_ST_I18N_PATCH`
- `admin_owner_username`, `admin_owner_password_hash` (veya `admin_owner_password`)
- env: `DMD_ADMIN_OWNER_USERNAME`, `DMD_ADMIN_OWNER_PASSWORD_HASH`, `DMD_ADMIN_OWNER_PASSWORD`
- `user_roles` veya env `DMD_USER_ROLES_JSON`
- `research_salt` (anonim araştırma exportu için gerekli)
- env `DMD_DEBUG`

### 2) Örnek `.streamlit/secrets.toml`

```toml
SHEET_URL = "https://docs.google.com/spreadsheets/d/<ID>"
OPENAI_API_KEY = "sk-..."
auth_secret = "<32+ byte entropy secret>"
persistent_login_via_query = true
i18n_patch_enabled = true
research_salt = "change-me"

[user_roles]
demo_doctor = "doctor"
demo_research = "researcher"
```

## Kimlik Doğrulama ve Güvenlik

- Şifreler `bcrypt` (varsa) veya `PBKDF2-HMAC-SHA256` ile saklanır.
- Legacy şifre formatlarından yeni hash formatına otomatik migrasyon vardır.
- Geçici hesap kilitleme (başarısız deneme sayacı) bulunur.
- Kalıcı giriş (query token) HMAC imzalıdır; güçlü `auth_secret` gerektirir.

## Veri Katmanı

- Yerel DB tabloları: `users`, `profiles`, `system_kv`
- Legacy JSON dosyalarından DB’ye tek seferlik migrasyon yapılır.
- Bulut yazımı başarısız olursa sync queue’ya alınır ve sonra drain edilir.

## Test ve Doğrulama

```powershell
python -B -m py_compile final_v50.py
python -m unittest tests/test_dmd_exon_phase_map.py -v
```

## Notlar

- `git` bulunmayan ortamlarda uygulama yine çalışır; sürüm kontrolü için ayrıca kurulmalıdır.
- Google Sheets için servis hesabı paylaşım izinleri ve worksheet adları (`users`, `profiles`) doğru olmalıdır.
