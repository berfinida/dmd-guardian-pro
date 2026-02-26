# NIZEN Clinical Platform

Streamlit tabanli klinik takip ve karar-destek uygulamasi.
Ana uygulama dosyasi: `final_v50.py`

## Ozellikler

- DMD odakli moduller: dashboard, klinik hesaplayici, NSAA, acil durum, takvim, haberler, AI destekli soru-cevap
- Diger norodejeneratif workspace'ler: ALS, Alzheimer, Parkinson, Huntington, Lewy, FTD, SMA
- Rol tabanli erisim: `family`, `doctor`, `researcher`, `admin`
- Yerel kalicilik: SQLite (`data/dmd_local.db`)
- Opsiyonel bulut senkron: Google Sheets (`users`, `profiles`)
- Bulut kesintisinde local fallback + sync queue

## Proje Yapisi

- Uygulama: `final_v50.py`
- Gereksinimler: `requirements.txt`
- Testler: `tests/`
- Runtime verisi: `data/`

## Kurulum

1. Python 3.10+ kurun.
2. Sanal ortam olusturun ve aktif edin.
3. Bagimliliklari kurun:

```powershell
pip install -r requirements.txt
```

## Calistirma

```powershell
streamlit run final_v50.py
```

## Konfigurasyon

Uygulama ayarlari `st.secrets` veya environment variable ile verilebilir.

### Kritik ayarlar

- `SHEET_URL` (Google Sheet URL)
- `OPENAI_API_KEY`
- `auth_secret` veya `AUTH_SECRET`
- `persistent_login_via_query` veya `DMD_PERSISTENT_LOGIN_QUERY`
- `DMD_PERSISTENT_LOGIN_TTL_SEC`
- `research_salt`

### Ornek `.streamlit/secrets.toml`

```toml
SHEET_URL = "https://docs.google.com/spreadsheets/d/<ID>"
OPENAI_API_KEY = "sk-..."
auth_secret = "<strong-random-secret>"
persistent_login_via_query = true
research_salt = "change-me"

[user_roles]
demo_doctor = "doctor"
demo_researcher = "researcher"
```

## Bulut Senkron Mantigi

- Uygulama once local DB ile calisir.
- Google Sheets ulasilabilirse local veriler buluta push edilir.
- Bulut yazimi basarisiz olursa islem `sync_queue` icine alinir.
- Baglanti geri geldiginde queue otomatik drain edilir.
- Doktor/Admin sidebar uzerinden manuel "Verileri Buluta Gonder" tetikleyebilir.

## Guvenlik Notlari

- Sifreler `bcrypt` (varsa) veya `PBKDF2-HMAC-SHA256` ile saklanir.
- Login deneme limiti ve gecici kilitleme uygulanir.
- Kalici login token'i HMAC imzalidir; guclu secret kullanin.

## Dogrulama

```powershell
python -B -m py_compile final_v50.py
python -m pyflakes final_v50.py
python -m unittest -v
```

## Sorun Giderme

- `403 Forbidden`: Service account, sheet ile paylasilmamis olabilir.
- `429`: API rate/limit asimi; bir sure sonra tekrar deneyin.
- `users/profiles` sekme adlari birebir dogru olmali.
- `SHEET_URL` formati: `https://docs.google.com/spreadsheets/d/<ID>`
