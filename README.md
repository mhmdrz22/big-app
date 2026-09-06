# آرام‌نگار (AramNegar) — v6

سامانه‌ی **غربالگری حمایتی استرس از متن** — نه تشخیص پزشکی.

## اجرای سریع

### لوکال (Windows)
```cmd
setup-local.bat
```

### لوکال (Linux/macOS)
```bash
chmod +x setup-local.sh
./setup-local.sh
```

### Docker
```bash
docker compose up -d
```

## گیت‌هاب

```bash
git init
git add .
git commit -m "AramNegar v6"
git remote add origin https://github.com/YOUR_USERNAME/aramnegar.git
git push -u origin main
```

## فایل‌های مهم

| فایل | کاربرد |
|------|--------|
| `.gitignore` | حذف فایل‌های حجیم/موقت از گیت |
| `.gitattributes` | line endings + LFS-ready |
| `docker-compose.yml` | اجرای محلی با Docker |
| `compose.prod.yaml` | اجرای production (Nginx+PostgreSQL) |
| `Dockerfile` | ایمیج سبک Python 3.11 |
| `.dockerignore` | حذف فایل‌های غیرضروری از ایمیج |

## حجم ریپو

| بخش | حجم تقریبی |
|-----|-----------|
| کد Python | ~۲ مگابایت |
| مدل (joblib) | ~۳ مگابایت (در .gitignore) |
| داده (CSV) | ~۳ مگابایت (در .gitignore) |
| **کل ریپو** | **~۲-۳ مگابایت** |

## API

| Endpoint | Method | توضیح |
|----------|--------|-------|
| `/api/v1/analyze` | POST | تحلیل متن → stress_score 0-100 |
| `/api/v1/feedback` | POST | بازخورد سه‌جهته |
| `/api/v1/admin/report/stakeholder.pdf` | GET | گزارش PDF |

## لایسنس

MIT
