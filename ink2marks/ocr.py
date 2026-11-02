"""Local image/PDF preprocessing and OCR utilities for Ink2Marks."""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np
import pytesseract
from PIL import Image

SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into",
    "is", "it", "of", "on", "or", "that", "the", "their", "this", "to", "was", "were", "with",
}


def configure_tesseract(command: str | None = None) -> str:
    """Resolve Tesseract executable from explicit config, env, or PATH."""
    requested = (command or os.getenv("TESSERACT_CMD", "")).strip()
    if requested:
        if not Path(requested).is_file():
            raise FileNotFoundError(f"Tesseract executable not found at: {requested}")
        pytesseract.pytesseract.tesseract_cmd = requested
        return requested
    located = shutil.which("tesseract")
    if not located:
        raise FileNotFoundError(
            "Tesseract OCR was not found. Install Tesseract and add it to PATH, "
            "or set TESSERACT_CMD in your .env file."
        )
    pytesseract.pytesseract.tesseract_cmd = located
    return located


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _deskew(binary: np.ndarray) -> np.ndarray:
    """Correct small page rotations using foreground pixel orientation."""
    foreground = cv2.bitwise_not(binary)
    coords = np.column_stack(np.where(foreground > 0))
    if len(coords) < 20:
        return binary
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.3 or abs(angle) > 15:
        return binary
    height, width = binary.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        binary, matrix, (width, height), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )


def preprocess_image(image: Image.Image, deskew: bool = True) -> Image.Image:
    """Convert to clean black-on-white text image for OCR."""
    bgr = _pil_to_bgr(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Upscale small scans before denoising and thresholding.
    if min(gray.shape[:2]) < 1200:
        scale = min(2.0, 1200 / min(gray.shape[:2]))
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11
    )
    if deskew:
        binary = _deskew(binary)
    return Image.fromarray(binary)


def _pdf_pages(data: bytes, dpi_scale: float = 2.0) -> list[Image.Image]:
    pages: list[Image.Image] = []
    with fitz.open(stream=data, filetype="pdf") as document:
        if document.page_count == 0:
            raise ValueError("The PDF contains no pages.")
        for page in document:
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi_scale, dpi_scale), alpha=False)
            pages.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    return pages


def load_pages(filename: str, data: bytes) -> list[Image.Image]:
    """Decode uploaded PDF pages or a single uploaded image."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        try:
            return _pdf_pages(data)
        except Exception as exc:
            raise ValueError(f"Could not render PDF pages: {exc}") from exc
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError("Upload a PDF, PNG, JPG, TIFF, or BMP file.")
    try:
        with Image.open(__import__("io").BytesIO(data)) as image:
            return [image.convert("RGB")]
    except Exception as exc:
        raise ValueError(f"Could not read the image: {exc}") from exc


def extract_text(image: Image.Image, language: str = "eng", preprocess: bool = True) -> tuple[str, Image.Image]:
    """Preprocess a page and return OCR text plus the processed preview."""
    configure_tesseract()
    prepared = preprocess_image(image) if preprocess else image.convert("L")
    text = pytesseract.image_to_string(prepared, lang=language, config="--psm 6")
    return text.strip(), prepared


def _keywords(text: str) -> set[str]:
    return {
        word for word in re.findall(r"[a-zA-Z0-9]+", text.casefold())
        if len(word) > 1 and word not in STOPWORDS
    }


def compare_with_reference(ocr_text: str, reference_text: str) -> dict[str, Any]:
    """Compute transparent reference-keyword coverage, not semantic grading."""
    expected = _keywords(reference_text)
    found = _keywords(ocr_text)
    matched = sorted(expected & found)
    missing = sorted(expected - found)
    coverage = (len(matched) / len(expected) * 100) if expected else 0.0
    return {
        "coverage_percent": round(coverage, 1),
        "matched_keywords": matched,
        "missing_keywords": missing,
        "reference_keyword_count": len(expected),
    }
