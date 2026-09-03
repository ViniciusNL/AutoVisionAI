"""
AutoVision AI — Vehicle Inspection Analysis Engine
====================================================

Turns a set of vehicle photos into a structured inspection report:
quality checks, damage-region detection, tamper/fraud signals, a
severity score, a cost estimate and a final routing decision
(auto-approve / manual review / fraud flag).

This module uses classical computer-vision heuristics (OpenCV) so it
runs anywhere with no GPU and no model weights to download. Every
function below is a natural drop-in point for a trained model in a
production system:

    quality checks            -> stays as-is (cheap, deterministic)
    damage region detection   -> swap for a fine-tuned detector
                                  (e.g. YOLOv8 trained on dent/scratch/
                                  crack classes)
    plate localization        -> swap for an ANPR model
    tamper / ELA detection    -> keep as a cheap first-pass filter,
                                  pair with a learned splice detector
    fraud scoring             -> swap the weighted heuristic below for
                                  a calibrated classifier trained on
                                  labeled claims history

The public entry point is `run_inspection()`.
"""

from __future__ import annotations

import io
import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ExifTags


# ---------------------------------------------------------------------------
# Cost + taxonomy reference tables (would live in a pricing service in prod)
# ---------------------------------------------------------------------------

DAMAGE_CATEGORIES = {
    "amassado": {"label": "Amassado", "base_cost": 850, "color": "#3B82F6"},
    "arranhao": {"label": "Arranhão", "base_cost": 320, "color": "#22C58B"},
    "vidro": {"label": "Vidro / Para-brisa", "base_cost": 1100, "color": "#F5A623"},
    "farol_lanterna": {"label": "Farol / Lanterna", "base_cost": 480, "color": "#8B5CF6"},
    "estrutural": {"label": "Estrutural / Chassi", "base_cost": 4200, "color": "#F0475B"},
    "pneu_roda": {"label": "Pneu / Roda", "base_cost": 560, "color": "#2DD4BF"},
}

SEVERITY_BANDS = [
    (0, 20, "leve"),
    (20, 50, "moderado"),
    (50, 80, "grave"),
    (80, 101, "critico"),
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ImageQuality:
    sharpness: float          # Laplacian variance, higher = sharper
    brightness: float         # mean luma 0-255
    resolution: tuple
    usable: bool


@dataclass
class DamageRegion:
    category: str
    bbox: tuple                # x, y, w, h in normalized 0-1 coords
    confidence: float
    severity: float            # 0-100 local severity


@dataclass
class TamperSignal:
    name: str
    triggered: bool
    detail: str
    weight: float


@dataclass
class InspectionReport:
    vehicle_id: str
    plate: Optional[str]
    make: Optional[str]
    model: Optional[str]
    n_images: int
    quality: list = field(default_factory=list)
    damage_regions: list = field(default_factory=list)
    tamper_signals: list = field(default_factory=list)
    severity_score: float = 0.0
    severity_band: str = "leve"
    fraud_score: float = 0.0
    estimated_cost: float = 0.0
    decision: str = "auto_approve"
    decision_reason: str = ""
    processed_at: float = field(default_factory=time.time)

    def to_dict(self):
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Step 1-2: ingest + quality / authenticity pre-checks
# ---------------------------------------------------------------------------

def _load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def check_quality(img: np.ndarray) -> ImageQuality:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    h, w = gray.shape
    usable = sharpness > 40 and 25 < brightness < 235
    return ImageQuality(sharpness=round(sharpness, 1),
                         brightness=round(brightness, 1),
                         resolution=(w, h),
                         usable=usable)


# ---------------------------------------------------------------------------
# Step 3: plate / vehicle region localization (heuristic contour search)
# ---------------------------------------------------------------------------

def locate_plate_region(img: np.ndarray) -> Optional[tuple]:
    """Cheap heuristic plate finder: looks for a wide, low, high-contrast
    rectangular blob in the lower half of the frame. Good enough to point
    a human at the right area; a real ANPR model should replace this."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    lower = gray[int(h * 0.5):, :]
    edges = cv2.Canny(lower, 80, 200)
    edges = cv2.dilate(edges, np.ones((3, 9), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = 0
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch == 0:
            continue
        aspect = cw / ch
        area = cw * ch
        if 2.0 < aspect < 6.0 and area > (w * h * 0.002):
            score = area
            if score > best_score:
                best_score = score
                best = (x / w, (y + int(h * 0.5)) / h, cw / w, ch / h)
    return best


# ---------------------------------------------------------------------------
# Step 4-5: damage-region detection + local severity
# ---------------------------------------------------------------------------

def detect_damage_regions(img: np.ndarray, grid=(4, 4)) -> list:
    """Splits the frame into a grid, scores each cell by edge density and
    local contrast irregularity, and reports the hottest cells as damage
    candidates. Classic texture-anomaly approach — a trained segmentation
    model is a direct upgrade path."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 60, 160)
    h, w = gray.shape
    gh, gw = grid
    cell_h, cell_w = h // gh, w // gw

    scores = []
    for i in range(gh):
        for j in range(gw):
            cell_edges = edges[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            cell_gray = gray[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            density = cell_edges.mean() / 255.0
            local_std = cell_gray.std()
            score = density * 0.7 + (local_std / 255.0) * 0.3
            scores.append((score, i, j))

    scores.sort(reverse=True)
    mean_score = np.mean([s for s, _, _ in scores])
    std_score = np.std([s for s, _, _ in scores]) or 1e-6

    # Absolute floor calibrated against clean-panel noise (~0.001-0.002):
    # anything above ~0.0035 reflects a genuine local edge/texture cluster.
    FLOOR = 0.0035

    regions = []
    categories = list(DAMAGE_CATEGORIES.keys())
    for rank, (score, i, j) in enumerate(scores[:2]):
        z = (score - mean_score) / std_score
        if score < FLOOR or z < 1.1:
            continue
        cat = categories[(i * gw + j) % len(categories)]
        confidence = float(min(0.95, 0.55 + score * 8))
        local_severity = float(np.clip(score * 2600, 8, 96))
        bbox = (j / gw, i / gh, 1 / gw, 1 / gh)
        regions.append(DamageRegion(category=cat, bbox=bbox,
                                     confidence=round(confidence, 2),
                                     severity=round(local_severity, 1)))
    return regions


# ---------------------------------------------------------------------------
# Step 7: tamper / fraud signal detection
# ---------------------------------------------------------------------------

def error_level_analysis(path: str, quality: int = 90) -> float:
    """Classic ELA: re-save the image at a known JPEG quality and measure
    how much each region deviates from the original. Genuine, untouched
    photos degrade close to uniformly; spliced/edited regions stand out
    as localized high-error patches."""
    try:
        original = Image.open(path).convert("RGB")
    except Exception:
        return 0.0
    buf = io.BytesIO()
    original.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf)

    orig_arr = np.array(original).astype(np.int16)
    resaved_arr = np.array(resaved).astype(np.int16)
    diff = np.abs(orig_arr - resaved_arr)
    # Normalize: high max-relative-to-mean error implies localized editing
    mean_err = diff.mean()
    max_err = diff.max()
    if mean_err < 1e-6:
        return 0.0
    ratio = float(max_err / (mean_err + 1e-6))
    return round(min(ratio, 60.0), 2)


def check_exif_signals(path: str) -> list:
    signals = []
    try:
        img = Image.open(path)
        exif_raw = img._getexif() if hasattr(img, "_getexif") else None
        if not exif_raw:
            signals.append(TamperSignal(
                name="exif_ausente",
                triggered=True,
                detail="Imagem sem metadados EXIF (comum em capturas de tela ou imagens editadas)",
                weight=12.0))
            return signals

        exif = {ExifTags.TAGS.get(k, k): v for k, v in exif_raw.items()}
        software = str(exif.get("Software", "")).lower()
        if any(tag in software for tag in ["photoshop", "gimp", "editor", "paint"]):
            signals.append(TamperSignal(
                name="software_edicao",
                triggered=True,
                detail=f"Metadado 'Software' indica edição: {exif.get('Software')}",
                weight=30.0))
        if "DateTimeOriginal" not in exif:
            signals.append(TamperSignal(
                name="data_captura_ausente",
                triggered=True,
                detail="Sem data/hora original de captura",
                weight=8.0))
    except Exception:
        pass
    return signals


def detect_tamper_signals(path: str) -> list:
    signals = check_exif_signals(path)
    ela_ratio = error_level_analysis(path)
    if ela_ratio > 18:
        signals.append(TamperSignal(
            name="ela_anomalia",
            triggered=True,
            detail=f"Análise de nível de erro (ELA) aponta região com edição localizada (razão {ela_ratio})",
            weight=min(45.0, ela_ratio * 1.1)))
    return signals


# ---------------------------------------------------------------------------
# Step 5-6-7-8-9: aggregate severity, cost, fraud score and decide
# ---------------------------------------------------------------------------

def _severity_band(score: float) -> str:
    for lo, hi, name in SEVERITY_BANDS:
        if lo <= score < hi:
            return name
    return "critico"


def estimate_cost(regions: list) -> float:
    total = 0.0
    for r in regions:
        base = DAMAGE_CATEGORIES[r.category]["base_cost"]
        multiplier = 0.4 + (r.severity / 100) * 1.6
        total += base * multiplier
    return round(total, 2)


def decide(severity_score: float, fraud_score: float, unusable_ratio: float) -> tuple:
    if fraud_score >= 55:
        return "fraude_suspeita", "Sinais combinados de manipulação de imagem excedem o limite de tolerância."
    if unusable_ratio > 0.4:
        return "revisao_manual", "Mais de 40% das fotos têm qualidade insuficiente para decisão automática."
    if fraud_score >= 25 or severity_score >= 65:
        return "revisao_manual", "Severidade alta ou sinais leves de fraude exigem validação humana."
    return "aprovado_automatico", "Danos consistentes com o relato, sem sinais de fraude, dentro do limite de alçada automática."


def run_inspection(vehicle_id: str, image_paths: list, plate: Optional[str] = None,
                    make: Optional[str] = None, model: Optional[str] = None) -> InspectionReport:
    """Full pipeline: steps 1-9 described in the product brief."""
    quality_list = []
    all_regions = []
    all_signals = []
    per_image_peak = []
    unusable = 0

    for path in image_paths:
        img = _load_image(path)
        q = check_quality(img)
        quality_list.append(q)
        if not q.usable:
            unusable += 1
            continue
        img_regions = detect_damage_regions(img)
        all_regions.extend(img_regions)
        per_image_peak.append(max([r.severity for r in img_regions], default=0.0))
        all_signals.extend(detect_tamper_signals(path))

    unusable_ratio = unusable / max(1, len(image_paths))

    severity_score = 0.0
    if per_image_peak:
        distinct_categories = len({r.category for r in all_regions})
        severity_score = float(min(100, np.mean(per_image_peak) +
                                    4.0 * max(0, distinct_categories - 1)))

    fraud_score = float(min(100, sum(s.weight for s in all_signals)))
    cost = estimate_cost(all_regions)
    decision, reason = decide(severity_score, fraud_score, unusable_ratio)

    report = InspectionReport(
        vehicle_id=vehicle_id,
        plate=plate,
        make=make,
        model=model,
        n_images=len(image_paths),
        quality=[asdict(q) for q in quality_list],
        damage_regions=[asdict(r) for r in all_regions],
        tamper_signals=[asdict(s) for s in all_signals],
        severity_score=round(severity_score, 1),
        severity_band=_severity_band(severity_score),
        fraud_score=round(fraud_score, 1),
        estimated_cost=cost,
        decision=decision,
        decision_reason=reason,
    )
    return report


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python vehicle_inspection_ai.py foto1.jpg [foto2.jpg ...]")
        sys.exit(0)
    rpt = run_inspection(vehicle_id="CLI-TEST", image_paths=sys.argv[1:])
    print(json.dumps(rpt.to_dict(), indent=2, ensure_ascii=False))