# run.py Usage Guide

این فایل توضیح می‌دهد چطور `run.py` را تنظیم کنید.

## 1) محدود کردن پردازش همزمان درخواست‌ها
در `run.py` این مقدار را تنظیم کنید:

- `MAX_CONCURRENT_REQUESTS = 4`

اگر 20 درخواست همزمان بیاید، فقط تا سقف این عدد با هم پردازش می‌شوند و بقیه در صف ThreadPool می‌مانند.

## 2) انتخاب CPU یا GPU
در `run.py`:

- `DEVICE_MODE = "gpu"` یا `DEVICE_MODE = "cpu"`

همچنین می‌توانید در زمان اجرا override کنید:

- `REACTOR_DEVICE=gpu python run.py`
- `REACTOR_DEVICE=cpu python run.py`

## 3) کش نتیجه نهایی پردازش
در `run.py` مسیر کش قابل تنظیم است:

- `CACHE_DIR = REPO_ROOT / "cache"`

نتیجه نهایی در این مسیر ذخیره می‌شود:

- `cache/results/<sha256>.jpg`

نام فایل خروجی بر اساس هش ترکیب این موارد ساخته می‌شود:

- `source_url`
- `target_url`
- پارامترهای swap
- مدل انتخاب‌شده

اگر همین ترکیب قبلاً پردازش شده باشد، پردازش دوباره انجام نمی‌شود و فایل cached برگردانده می‌شود.

## 4) عدم دانلود مجدد URL های تکراری
برای ورودی‌های URL:

- اگر قبلاً دانلود شده باشند، از دیسک خوانده می‌شوند.
- در غیر این صورت یکبار دانلود و ذخیره می‌شوند.

## 5) ساختار فولدر کش ورودی‌ها
فایل‌ها داخل `CACHE_DIR` در دو فولدر جدا ذخیره می‌شوند:

- `cache/sources/<sha256_of_source_url>.<ext>`
- `cache/targets/<sha256_of_target_url>.<ext>`

## 6) تنظیم پارامترهای node از طریق URL
پارامترهای قابل تنظیم در URL (در `run.py` پیاده شده‌اند):

- `source_faces_index` (مثال: `0` یا `0,1`)
- `faces_index` (مثال: `0` یا `1,2`)
- `gender_source` (عدد)
- `gender_target` (عدد)
- `faces_order` (مثال: `large-small,large-small`)
- `face_boost_enabled` (`true/false` یا `1/0`)

### مثال URL کامل

```text
/swap?source_url=https%3A%2F%2Fexample.com%2Fs.jpg&target_url=https%3A%2F%2Fexample.com%2Ft.jpg&source_faces_index=0,1&faces_index=0&gender_source=0&gender_target=0&faces_order=large-small,large-small&face_boost_enabled=true
```

## 7) نکته اجرایی
با اجرای `python run.py`، سرور روی `REACTOR_HOST` و `REACTOR_PORT` بالا می‌آید (پیش‌فرض `0.0.0.0:8004`).
