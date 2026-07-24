@echo off
echo ============================================
echo   YOLOv3 Car Counter - دانلود وزن‌های مدل
echo ============================================
echo.
echo در حال دانلود yolov3.weights (~236MB)...
echo.

curl -L -o models\yolov3.weights https://pjreddie.com/media/files/yolov3.weights

if %errorlevel% equ 0 (
    echo.
    echo ============================================
    echo   دانلود با موفقیت انجام شد!
    echo ============================================
) else (
    echo.
    echo خطا در دانلود! لطفا فایل را دستی دانلود کنید:
    echo https://pjreddie.com/media/files/yolov3.weights
    echo و در پوشه models/ قرار دهید.
)
pause
