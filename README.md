# 🎬 Chinese → English OCR Subtitle Tool v2 (GPU Enhanced)

Reads Chinese subtitles directly from your screen and displays English translation instantly using high-performance OCR and aggressive image preprocessing.

**No audio processing. Near real-time. High-accuracy v2 engine.**

---

## 🚀 Key Features in v2
- **Aggressive Preprocessing**: Automatically upscales images (3x), denoises, and boosts contrast to make small/blurry text readable.
- **Smart Multi-pass OCR**: Runs 5 different image treatments (white-text mask, yellow-text mask, contrast boost, etc.) and picks the result with the highest confidence.
- **Gibberish Filtering**: A built-in text scorer rejects "garbage" results and repeated character symbols.
- **Full GPU Acceleration**: Optimized for NVIDIA RTX cards (like your RTX 4050) using CUDA.
- **Smart Translation Caching**: Avoids re-translating static subtitles, saving bandwidth and improving speed.

---

## 🛠 Prerequisites

This tool is optimized for **Python 3.11** to ensure full GPU (CUDA) compatibility and stability.

- **OS**: Windows (tested on Windows 11)
- **GPU**: NVIDIA RTX Graphics Card (recommended for peak speed)

---

## 📦 Setup & Installation

We use a dedicated environment to ensure zero compatibility issues with other Python versions.

### 1. Create & Activate Environment
```powershell
# Create dedicated environment
py -3.11 -m venv C:\subtitles_env

# Activate it
C:\subtitles_env\Scripts\activate
```

### 2. Install Dependencies
```powershell
pip install easyocr opencv-python pyautogui pillow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install googletrans==4.0.0-rc1 legacy-cgi
```

---

## ▶️ Usage

Whenever you want to run the tool:

1. **Activate the environment**:
   ```powershell
   C:\subtitles_env\Scripts\activate
   ```
2. **Run the script**:
   ```powershell
   python ocr_translator.py
   ```

### On Startup: Select Your Region
1. Your screen will dim.
2. **Click and drag** tightly over the area where Chinese subtitles appear.
3. Release — scanning begins instantly on your GPU.

---

## ⌨️ Controls

| Action | How |
|--------|-----|
| **Reselect region** | Click **"⊹ reselect"** on the overlay |
| **Move overlay** | Click and drag the subtitle bar |
| **Hide / Show** | `Ctrl + Shift + T` |
| **Close** | Click `✕` |

---

## 🔴 Status Indicators (The "Dot")

- 🔵 **Blue**: Scanning screen & Preprocessing
- 🟡 **Amber**: Translating Chinese → English
- 🟢 **Green**: Translation displayed
- ⚫ **Gray**: Idle (no text detected in region)

---

## ⚙️ Advanced Tuning

Edit these values in `ocr_translator.py` `Config` class:

| Setting | Default | Notes |
|---------|---------|-------|
| `CAPTURE_INTERVAL` | `0.7s` | Time between scans. Lower = snappier, more CPU/GPU. |
| `MIN_CONFIDENCE` | `0.3` | Lower = more permissive (useful for blurry text). |
| `UPSCALE_FACTOR` | `3`   | 3x enlargement. Change to `2` for slower PCs. |
| `FONT_SIZE` | `22` | Changes the size of the English overlay text. |

---

## 🛠 Troubleshooting

**GPU not being used?**
Verify your installation by running:
```powershell
python -c "import torch; print(torch.cuda.is_available())"
```
If it returns `False`, ensure you installed the `cu121` version of PyTorch in the 3.11 environment.

**Translation stops working?**
The free Google Translate API has occasional rate limits. If the tool stays stuck on "translating" (Amber dot), wait 30 seconds or restart the script.
"# orc-screen-sub" 
