@echo off
TITLE Chinese-English OCR Subtitles (GPU)
cd /d "c:\Users\divap\ocr subtitles"

echo Starting OCR Subtitle Tool with GPU acceleration...
echo Environment: C:\subtitles_env
echo.

:: Run using the dedicated GPU-enabled environment
"C:\subtitles_env\Scripts\python.exe" ocr_translator.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Tool exited with an error. 
    pause
)
