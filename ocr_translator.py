#!/usr/bin/env python3
"""
Chinese → English Real-Time Subtitle Tool (OCR Mode) - Enhanced
- Aggressive image preprocessing: upscale, denoise, contrast boost
- Multi-pass OCR: tries multiple image treatments and picks best result
- Smart text filtering: removes garbage/gibberish results
- Translation with caching

Requirements:
    pip install easyocr pyautogui pillow googletrans==4.0.0-rc1 opencv-python
"""

import sys
import time
import threading
import queue
import re
import tkinter as tk
from tkinter import font as tkfont
from typing import Optional, Tuple, List
from PIL import ImageGrab, Image, ImageEnhance, ImageFilter
import numpy as np
import easyocr
import cv2
import ctypes

# Windows High DPI awareness
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
class Config:
    CAPTURE_INTERVAL: float = 0.7
    OVERLAY_OPACITY: float = 0.92
    TEXT_COLOR: str = "#FFFFFF"
    BG_COLOR: str = "#1a1a1a"
    FONT_SIZE: int = 22
    MAX_WIDTH: int = 950
    WINDOW_HEIGHT: int = 95
    BOTTOM_MARGIN: int = 60
    MIN_CONFIDENCE: float = 0.3      # Lower = more permissive OCR
    TARGET_LANG: str = "en"
    SOURCE_LANG: str = "zh-cn"
    UPSCALE_FACTOR: int = 3          # How much to enlarge image before OCR (3x = big improvement)


# ─────────────────────────────────────────────
#  IMAGE PREPROCESSOR
#  The core fix — makes subtitle text readable for OCR
# ─────────────────────────────────────────────
class ImagePreprocessor:
    """
    Applies multiple preprocessing treatments to a screenshot.
    Small/blurry subtitle text becomes clean and large before OCR sees it.
    """

    @staticmethod
    def upscale(img: np.ndarray, factor: int = Config.UPSCALE_FACTOR) -> np.ndarray:
        """Enlarge image using INTER_CUBIC (best for text)."""
        h, w = img.shape[:2]
        return cv2.resize(img, (w * factor, h * factor), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def sharpen(img: np.ndarray) -> np.ndarray:
        """Sharpen edges to make character strokes clearer."""
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ])
        return cv2.filter2D(img, -1, kernel)

    @staticmethod
    def denoise(img: np.ndarray) -> np.ndarray:
        """Remove noise while preserving text edges."""
        return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    @staticmethod
    def boost_contrast(img: np.ndarray) -> np.ndarray:
        """CLAHE contrast boost — works on any background color."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def white_text_mask(img: np.ndarray) -> np.ndarray:
        """Isolate white/light text on dark background."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        result = np.zeros_like(img)
        result[mask == 255] = [0, 0, 0]        # Text → black
        result[mask != 255] = [255, 255, 255]  # Background → white
        return result

    @staticmethod
    def yellow_text_mask(img: np.ndarray) -> np.ndarray:
        """Isolate yellow text (common in Chinese streaming sites)."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([15, 80, 80])
        upper_yellow = np.array([40, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        result = np.full_like(img, 255)         # White background
        result[mask > 0] = [0, 0, 0]            # Yellow text → black
        return result

    @staticmethod
    def dark_text_mask(img: np.ndarray) -> np.ndarray:
        """Isolate dark text on light background."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        result = np.full_like(img, 255)
        result[mask == 255] = [0, 0, 0]
        return result

    @classmethod
    def get_all_variants(cls, screenshot: Image.Image) -> List[np.ndarray]:
        """
        Returns multiple preprocessed versions of the screenshot.
        OCR will run on ALL of them and pick the best result.
        """
        # Convert PIL → OpenCV
        img_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

        # Step 1: Always upscale first
        img_up = cls.upscale(img_cv)

        # Step 2: Generate variants
        variants = []

        # Variant 1: Upscaled + contrast boost + sharpen (general purpose)
        v1 = cls.sharpen(cls.boost_contrast(img_up))
        variants.append(v1)

        # Variant 2: Denoised + upscaled (for blurry/compressed video)
        v2 = cls.sharpen(cls.upscale(cls.denoise(img_cv)))
        variants.append(v2)

        # Variant 3: White text isolation (most streaming sites)
        v3 = cls.white_text_mask(img_up)
        variants.append(v3)

        # Variant 4: Yellow text isolation
        v4 = cls.yellow_text_mask(img_up)
        variants.append(v4)

        # Variant 5: Dark text isolation
        v5 = cls.dark_text_mask(img_up)
        variants.append(v5)

        return variants


# ─────────────────────────────────────────────
#  TEXT QUALITY SCORER
#  Picks the best OCR result from multiple attempts
# ─────────────────────────────────────────────
class TextScorer:
    # Chinese unicode range
    CHINESE_RE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')
    # Gibberish indicators
    GARBAGE_RE = re.compile(r'[^\w\s\u4e00-\u9fff\u3400-\u4dbf.,!?:;\'\"()\-]')

    @classmethod
    def score(cls, text: str, confidence: float) -> float:
        """
        Returns a quality score. Higher = better.
        Rewards: Chinese characters, reasonable length, high confidence
        Penalizes: garbage symbols, all-latin short strings, repeated chars
        """
        if not text or len(text.strip()) < 2:
            return 0.0

        chinese_chars = len(cls.CHINESE_RE.findall(text))
        garbage_chars = len(cls.GARBAGE_RE.findall(text))
        total_chars = len(text.strip())

        # Must have at least some Chinese characters
        if chinese_chars == 0:
            return 0.0

        # Penalize garbage symbols heavily
        garbage_ratio = garbage_chars / max(total_chars, 1)
        if garbage_ratio > 0.3:
            return 0.0

        # Reward Chinese character density
        chinese_ratio = chinese_chars / max(total_chars, 1)

        # Detect repeated character spam (e.g. "的的的的的")
        if total_chars > 3:
            unique_ratio = len(set(text)) / total_chars
            if unique_ratio < 0.2:
                return 0.0

        score = (chinese_ratio * 0.5) + (confidence * 0.4) + (min(total_chars, 30) / 30 * 0.1)
        return score

    @classmethod
    def is_valid(cls, text: str) -> bool:
        return cls.score(text, 1.0) > 0.0


# ─────────────────────────────────────────────
#  OCR ENGINE (EasyOCR + Multi-pass)
# ─────────────────────────────────────────────
class OCREngine:
    def __init__(self):
        print("[OCR] Loading EasyOCR Chinese model...")
        self.reader = easyocr.Reader(
            ['ch_sim', 'en'],
            gpu=True,
            verbose=False
        )
        self.preprocessor = ImagePreprocessor()
        print("[OCR] Model ready.")

    def _run_ocr(self, img: np.ndarray) -> Tuple[str, float]:
        """Run OCR on a single image variant. Returns (text, avg_confidence)."""
        try:
            results = self.reader.readtext(
                img,
                detail=1,
                paragraph=False,        # Non-paragraph mode = more granular results
                text_threshold=0.5,     # Minimum text confidence
                low_text=0.3,
                width_ths=0.8,
                decoder='beamsearch',   # More accurate than greedy
            )

            if not results:
                return "", 0.0

            texts = []
            confidences = []
            for (_, text, conf) in results:
                if conf >= Config.MIN_CONFIDENCE and text.strip():
                    texts.append(text.strip())
                    confidences.append(conf)

            if not texts:
                return "", 0.0

            combined = " ".join(texts)
            avg_conf = sum(confidences) / len(confidences)
            return combined, avg_conf

        except Exception as e:
            return "", 0.0

    def read(self, region: Tuple[int, int, int, int]) -> Optional[str]:
        """
        Multi-pass OCR: run on all image variants, pick the best result.
        """
        try:
            screenshot = ImageGrab.grab(bbox=region, all_screens=True)
            variants = self.preprocessor.get_all_variants(screenshot)

            best_text = ""
            best_score = 0.0

            for i, variant in enumerate(variants):
                text, confidence = self._run_ocr(variant)
                score = TextScorer.score(text, confidence)

                if score > best_score:
                    best_score = score
                    best_text = text

            if best_score > 0 and best_text:
                # Clean up spacing around Chinese text
                best_text = re.sub(r'\s+', ' ', best_text).strip()
                return best_text

            return None

        except Exception as e:
            print(f"[OCR] Error: {e}")
            return None


# ─────────────────────────────────────────────
#  TRANSLATOR (Google Translate, free)
# ─────────────────────────────────────────────
class Translator:
    def __init__(self):
        from googletrans import Translator as GTranslator
        self.gt = GTranslator()
        self._cache: dict = {}

    def translate(self, text: str) -> Optional[str]:
        if text in self._cache:
            return self._cache[text]
        try:
            result = self.gt.translate(text, dest=Config.TARGET_LANG, src=Config.SOURCE_LANG)
            translated = result.text
            self._cache[text] = translated
            if len(self._cache) > 300:
                del self._cache[next(iter(self._cache))]
            return translated
        except Exception as e:
            print(f"[Translate] Error: {e}")
            return None


# ─────────────────────────────────────────────
#  REGION SELECTOR
# ─────────────────────────────────────────────
class RegionSelector:
    def __init__(self, master):
        self.master = master
        self.region = None
        self._start_x = 0
        self._start_y = 0
        self._rect_id = None

    def select(self) -> Optional[Tuple[int, int, int, int]]:
        self.top = tk.Toplevel(self.master)
        sw = self.top.winfo_screenwidth()
        sh = self.top.winfo_screenheight()
        self.top.geometry(f"{sw}x{sh}+0+0")
        self.top.attributes('-topmost', True)
        self.top.configure(bg='black')
        self.top.update()
        self.top.after(300, lambda: self.top.attributes('-alpha', 0.45))

        self.canvas = tk.Canvas(self.top, cursor="cross", bg='black', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)

        self.canvas.create_text(
            sw // 2, 55,
            text="🖱  Click and drag over the subtitle area, then release",
            fill="white", font=("Segoe UI", 22, "bold")
        )
        self.canvas.create_text(
            sw // 2, 95,
            text="Select ONLY the subtitle bar — tightly around the text",
            fill="#aaaaaa", font=("Segoe UI", 14)
        )

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.top.bind("<Escape>", lambda e: self.top.destroy())
        self.top.wait_window()
        return self.region

    def _on_press(self, e):
        self._start_x, self._start_y = e.x, e.y
        if self._rect_id:
            self.canvas.delete(self._rect_id)

    def _on_drag(self, e):
        if self._rect_id:
            self.canvas.delete(self._rect_id)
        self._rect_id = self.canvas.create_rectangle(
            self._start_x, self._start_y, e.x, e.y,
            outline="#00FF88", width=2, fill="#00FF88", stipple="gray25"
        )

    def _on_release(self, e):
        x1, y1 = min(self._start_x, e.x), min(self._start_y, e.y)
        x2, y2 = max(self._start_x, e.x), max(self._start_y, e.y)
        if (x2 - x1) < 30 or (y2 - y1) < 10:
            return
        self.region = (x1, y1, x2, y2)
        self.top.destroy()


# ─────────────────────────────────────────────
#  SUBTITLE OVERLAY
# ─────────────────────────────────────────────
class SubtitleOverlay:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Subtitles")
        self.root.configure(bg=Config.BG_COLOR)
        self.root.overrideredirect(True)
        self.root.attributes('-topmost', True)
        self.root.withdraw()

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x = (sw - Config.MAX_WIDTH) // 2
        y = sh - Config.WINDOW_HEIGHT - Config.BOTTOM_MARGIN
        self.root.geometry(f"{Config.MAX_WIDTH}x{Config.WINDOW_HEIGHT}+{x}+{y}")

        self.frame = tk.Frame(self.root, bg=Config.BG_COLOR)
        self.frame.pack(expand=True, fill='both', padx=16, pady=10)

        self.subtitle_font = tkfont.Font(family="Segoe UI", size=Config.FONT_SIZE, weight="bold")
        self.subtitle_label = tk.Label(
            self.frame, text="⏳ Initializing...",
            font=self.subtitle_font, fg=Config.TEXT_COLOR, bg=Config.BG_COLOR,
            wraplength=Config.MAX_WIDTH - 40, justify='center', anchor='center'
        )
        self.subtitle_label.pack(expand=True)

        self.status_dot = tk.Label(self.root, text="●", font=tkfont.Font(size=10),
                                   fg="#444444", bg=Config.BG_COLOR)
        self.status_dot.place(relx=1.0, x=-28, rely=1.0, y=-18, anchor='se')

        tk.Button(self.root, text="✕", command=self.root.quit,
                  bg=Config.BG_COLOR, fg="#666666", bd=0,
                  font=tkfont.Font(size=12), activebackground=Config.BG_COLOR,
                  cursor="hand2").place(relx=1.0, x=-8, y=6, anchor='ne')

        self.reselect_btn = tk.Button(
            self.root, text="⊹ reselect", command=self._request_reselect,
            bg=Config.BG_COLOR, fg="#555555", bd=0,
            font=tkfont.Font(size=10), activebackground=Config.BG_COLOR, cursor="hand2"
        )
        self.reselect_btn.place(x=8, y=6, anchor='nw')

        self._drag = {"x": 0, "y": 0}
        self.root.bind("<Button-1>", lambda e: self._drag.__setitem__("x", e.x) or self._drag.__setitem__("y", e.y))
        self.root.bind("<B1-Motion>", lambda e: self.root.geometry(
            f"+{self.root.winfo_x() + e.x - self._drag['x']}+{self.root.winfo_y() + e.y - self._drag['y']}"))
        self.root.bind_all("<Control-Shift-T>", lambda e: self.toggle())

        self._queue: queue.Queue = queue.Queue()
        self._reselect_requested = False
        self._poll()

    def _request_reselect(self):
        self._reselect_requested = True

    def _poll(self):
        try:
            while True:
                msg = self._queue.get_nowait()
                if msg["type"] == "text":
                    self.subtitle_label.config(text=msg["text"], fg=msg["color"])
                elif msg["type"] == "status":
                    self.status_dot.config(fg=msg["color"])
        except queue.Empty:
            pass
        self.root.after(80, self._poll)

    def set_text(self, text: str, is_error: bool = False):
        self._queue.put({"type": "text", "text": text,
                         "color": "#FF5555" if is_error else Config.TEXT_COLOR})

    def set_status(self, status: str):
        self._queue.put({"type": "status", "color": {
            "scanning": "#00AAFF", "translating": "#FFAA00",
            "done": "#00CC44", "idle": "#333333"
        }.get(status, "#333333")})

    def toggle(self):
        self.root.withdraw() if self.root.winfo_viewable() else self.root.deiconify()

    def run(self):
        self.root.mainloop()


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────
class OCRPipeline:
    def __init__(self):
        print("[System] Initializing...")
        self.root = tk.Tk()
        self.root.withdraw()
        self.ocr = OCREngine()
        self.translator = Translator()
        self.overlay = SubtitleOverlay(self.root)
        self.region = None
        self.running = False
        self._last_text = ""
        print("[System] Ready.")

    def select_region(self):
        selector = RegionSelector(self.root)
        self.region = selector.select()
        if self.region:
            self.overlay.root.deiconify()
            self.overlay.root.attributes('-alpha', Config.OVERLAY_OPACITY)
            self.overlay.set_text("✅ Region set! Scanning for subtitles...")
            print(f"[Setup] Region: {self.region}")
        else:
            print("[Setup] No region selected, try again.")
            self.select_region()

    def _loop(self):
        while self.running:
            if self.overlay._reselect_requested:
                self.overlay._reselect_requested = False
                self.root.after(0, self.select_region)
                time.sleep(0.5)
                continue

            if not self.region:
                time.sleep(0.2)
                continue

            try:
                self.overlay.set_status("scanning")
                chinese_text = self.ocr.read(self.region)

                if not chinese_text:
                    self.overlay.set_status("idle")
                    time.sleep(Config.CAPTURE_INTERVAL)
                    continue

                if chinese_text == self._last_text:
                    time.sleep(Config.CAPTURE_INTERVAL)
                    continue

                self._last_text = chinese_text
                print(f"[OCR] {chinese_text}")

                self.overlay.set_status("translating")
                english = self.translator.translate(chinese_text)

                if not english:
                    time.sleep(Config.CAPTURE_INTERVAL)
                    continue

                print(f"[EN]  {english}")
                self.overlay.set_text(english)
                self.overlay.set_status("done")

            except Exception as e:
                print(f"[Error] {e}")
                self.overlay.set_text(f"Error: {e}", is_error=True)

            time.sleep(Config.CAPTURE_INTERVAL)

    def start(self):
        print("=" * 52)
        print("  Chinese → English OCR Subtitle Tool v2")
        print("  Ctrl+Shift+T  : toggle overlay")
        print("  ⊹ reselect    : pick new subtitle region")
        print("  ✕             : close")
        print("=" * 52)
        self.select_region()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        self.overlay.run()
        self.running = False
        print("[Done] Exited.")


if __name__ == "__main__":
    pipeline = OCRPipeline()
    pipeline.start()