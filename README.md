# YOLOv3 Car Counter 🚗

سیستم تشخیص و شمارش خودرو در تصاویر دوربین‌های ترافیکی با YOLOv3 و OpenCV.

## ویژگی‌ها

- تشخیص خودرو (car, truck, bus) با YOLOv3
- شمارش تعداد خودرو در هر فریم
- ردیابی خودرو بین فریم‌ها (IoU-based)
- رسم باکس دور هر خودرو با label و confidence
- خروجی ویدیوی پردازش‌شده
- خروجی CSV با ستون‌های `frame_number`, `car_count`

## پیش‌نیازها

- Python 3.8+
- OpenCV
- NumPy

## نصب

```bash
pip install -r requirements.txt
```

## دانلود وزن‌های مدل

```bash
# ویندوز
download_weights.bat

# یا دستی:
# دانلود از: https://pjreddie.com/media/files/yolov3.weights
# و قرار دادن در پوشه models/
```

## استفاده

```bash
python src/main.py --input data/input/traffic.mp4
```

### آرگومان‌ها

| آرگومان | پیش‌فرض | توضیح |
|---------|---------|-------|
| `--input` | `data/input/traffic.mp4` | مسیر ویدیوی ورودی |
| `--output` | `data/output/output.mp4` | مسیر ویدیوی خروجی |
| `--csv` | `data/output/car_count.csv` | مسیر فایل CSV |
| `--width` | `416` | عرض ورودی مدل (مضرب ۳۲) |
| `--height` | `416` | ارتفاع ورودی مدل (مضرب ۳۲) |
| `--display` | `False` | نمایش زنده حین پردازش |

### نمونه

```bash
python src/main.py --input data/input/highway.mp4 --output data/output/result.mp4 --display
```

## تنظیمات

فایل `config.py`:

```python
INPUT_WIDTH = 416           # سایز ورودی YOLO
INPUT_HEIGHT = 416
CONFIDENCE_THRESHOLD = 0.5  # آستانه اطمینان
NMS_THRESHOLD = 0.4         # آستانه NMS
VEHICLE_CLASSES = {"car", "truck", "bus"}
```

## ساختار پروژه

```
yolov3-car-counter/
├── data/
│   ├── input/              # ویدیوهای ورودی
│   └── output/             # نتایج + CSV
├── models/
│   ├── yolov3.cfg          # تنظیمات معماری YOLO
│   ├── yolov3.weights      # وزن‌های آموزش‌دیده (دانلود جدا)
│   └── coco.names          # نام کلاس‌های COCO
├── src/
│   ├── detector.py         # بارگذاری و اجرای مدل
│   ├── tracker.py          # ردیابی بین فریم‌ها
│   ├── utils.py            # NMS, رسم باکس, توابع کمکی
│   └── main.py             # پایپ‌لاین اصلی
├── config.py               # تنظیمات
├── download_weights.bat    # اسکریپت دانلود وزن‌ها
└── requirements.txt
```

## Docker

```bash
docker compose up
```

## لایسنس

MIT
