"""
Generates a realistic demo dataset for the AutoVision AI dashboard.

Important: this does NOT fake report numbers. It procedurally paints
synthetic "vehicle photos" (clean panels, damage-like marks, spliced /
recompressed tampered photos) and pushes every single one of them
through the real `vehicle_inspection_ai.run_inspection` pipeline, the
exact same code that would run on real inspection photos. Only the
business metadata around each report (plate, make, model, insurer,
date) is randomly assigned afterwards.
"""

import json
import os
import random
import sys
import time
from datetime import datetime, timedelta

import cv2
import numpy as np
from PIL import Image
import piexif

sys.path.insert(0, os.path.dirname(__file__))
from vehicle_inspection_ai import run_inspection, DAMAGE_CATEGORIES  # noqa: E402

random.seed(42)
np.random.seed(42)

TMP_DIR = "/home/claude/autovision/data/synth"
os.makedirs(TMP_DIR, exist_ok=True)

MAKES_MODELS = [
    ("Fiat", "Argo"), ("Fiat", "Toro"), ("Volkswagen", "Polo"), ("Volkswagen", "T-Cross"),
    ("Chevrolet", "Onix"), ("Chevrolet", "Tracker"), ("Toyota", "Corolla"), ("Toyota", "Hilux"),
    ("Hyundai", "HB20"), ("Hyundai", "Creta"), ("Renault", "Kwid"), ("Honda", "Civic"),
    ("Jeep", "Compass"), ("Nissan", "Kicks"), ("Ford", "Ka"),
]

INSURERS = ["Porto Seguro", "SulAmérica", "Bradesco Seguros", "Azul Seguros",
            "Tokio Marine", "Allianz Seguros", "Mapfre"]

STATES = ["SP", "RJ", "MG", "PR", "RS", "BA", "PE", "SC", "CE", "GO"]

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def random_plate():
    # Mercosul format: ABC1D23
    l = lambda: random.choice(LETTERS)
    d = lambda: random.choice("0123456789")
    return f"{l()}{l()}{l()}{d()}{l()}{d()}{d()}"


def mask_plate(plate):
    return plate[:3] + "•" + plate[4:5] + "••"


# ---------------------------------------------------------------------------
# Synthetic photo painters
# ---------------------------------------------------------------------------

def _base_panel(w=640, h=480, base_gray=175, noise=6):
    img = np.full((h, w, 3), base_gray, dtype=np.uint8)
    noise_layer = np.random.randint(-noise, noise + 1, (h, w, 3))
    img = np.clip(img.astype(int) + noise_layer, 0, 255).astype(np.uint8)
    return img


def _add_scratch(img, intensity=40, length_frac=0.35):
    h, w = img.shape[:2]
    x1, y1 = random.randint(0, w), random.randint(0, h)
    ang = random.uniform(0, np.pi)
    length = int(min(w, h) * length_frac)
    x2 = int(np.clip(x1 + length * np.cos(ang), 0, w - 1))
    y2 = int(np.clip(y1 + length * np.sin(ang), 0, h - 1))
    color = tuple(int(c) for c in np.clip(np.array([175, 175, 175]) - intensity, 0, 255))
    cv2.line(img, (x1, y1), (x2, y2), color, thickness=random.randint(2, 5))
    return img


def _add_dent(img, intensity=45, radius_frac=0.08):
    h, w = img.shape[:2]
    r = int(min(w, h) * radius_frac)
    cx, cy = random.randint(r, w - r), random.randint(r, h - r)
    overlay = img.copy()
    color = tuple(int(c) for c in np.clip(np.array([175, 175, 175]) - intensity, 0, 255))
    cv2.circle(overlay, (cx, cy), r, color, -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, dst=img)
    return img


def paint_photo(n_marks, mark_intensity, blur=0, brightness_shift=0):
    img = _base_panel()
    for _ in range(n_marks):
        if random.random() < 0.5:
            img = _add_scratch(img, intensity=mark_intensity)
        else:
            img = _add_dent(img, intensity=mark_intensity)
    if brightness_shift:
        img = np.clip(img.astype(int) + brightness_shift, 0, 255).astype(np.uint8)
    if blur:
        img = cv2.GaussianBlur(img, (blur, blur), 0)
    return img


def save_clean(img, path, with_datetime=True):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    exif_bytes = None
    if with_datetime:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
        dt = (datetime.now() - timedelta(days=random.randint(0, 600))).strftime("%Y:%m:%d %H:%M:%S")
        exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = dt.encode()
        exif_bytes = piexif.dump(exif_dict)
    pil_img.save(path, "JPEG", quality=92, exif=exif_bytes if exif_bytes else b"")


def save_tampered(img, path):
    """Splice a patch from a differently-lit synthetic region into the
    frame, then re-save while stamping an editor 'Software' EXIF tag —
    the two independent tamper signals the engine looks for."""
    h, w = img.shape[:2]
    patch = _base_panel(w=w // 3, h=h // 3, base_gray=140, noise=2)
    px, py = random.randint(0, w - w // 3), random.randint(0, h - h // 3)
    img[py:py + h // 3, px:px + w // 3] = patch

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    exif_dict["0th"][piexif.ImageIFD.Software] = b"Adobe Photoshop 24.0"
    exif_bytes = piexif.dump(exif_dict)
    pil_img.save(path, "JPEG", quality=97, exif=exif_bytes)


# ---------------------------------------------------------------------------
# Profiles -> synthetic photo sets
# ---------------------------------------------------------------------------

PROFILES = {
    "leve":     dict(weight=0.46, n_marks=(0, 1), intensity=(15, 28), blur=0),
    "moderado": dict(weight=0.27, n_marks=(1, 2), intensity=(28, 42), blur=0),
    "grave":    dict(weight=0.14, n_marks=(2, 4), intensity=(42, 60), blur=0),
    "baixa_qualidade": dict(weight=0.05, n_marks=(0, 1), intensity=(10, 20), blur=9),
    "fraude":   dict(weight=0.08, n_marks=(1, 2), intensity=(30, 45), blur=0),
}


def make_vehicle_photos(vehicle_idx, profile):
    n_images = random.choice([2, 3, 3, 4])
    paths = []
    spec = PROFILES[profile]
    for k in range(n_images):
        n_marks = random.randint(*spec["n_marks"])
        intensity = random.randint(*spec["intensity"])
        img = paint_photo(n_marks, intensity, blur=spec.get("blur", 0))
        path = os.path.join(TMP_DIR, f"v{vehicle_idx:04d}_{k}.jpg")
        if profile == "fraude" and k == 0:
            save_tampered(img, path)
        else:
            save_clean(img, path, with_datetime=(profile != "fraude"))
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------

def weighted_profile():
    names = list(PROFILES.keys())
    weights = [PROFILES[n]["weight"] for n in names]
    return random.choices(names, weights=weights, k=1)[0]


def random_date_2026_bias():
    """Skew dates so we get a nice 2021-2026 yearly trend, heavier at the
    end, with 2026 running through ~September only (partial year)."""
    year_weights = {2021: 0.06, 2022: 0.09, 2023: 0.13, 2024: 0.18, 2025: 0.28, 2026: 0.26}
    year = random.choices(list(year_weights.keys()), weights=list(year_weights.values()), k=1)[0]
    if year == 2026:
        month = random.randint(1, 8)
    else:
        month = random.randint(1, 12)
    day = random.randint(1, 28)
    return datetime(year, month, day)


def build_dataset(n_vehicles=140):
    records = []
    for i in range(n_vehicles):
        profile = weighted_profile()
        photos = make_vehicle_photos(i, profile)
        make, model = random.choice(MAKES_MODELS)
        plate = random_plate()
        date = random_date_2026_bias()

        report = run_inspection(
            vehicle_id=f"VST-{2021 + i // 30}-{1000 + i}",
            image_paths=photos,
            plate=plate, make=make, model=model,
        )
        rd = report.to_dict()
        rd["date"] = date.strftime("%Y-%m-%d")
        rd["insurer"] = random.choice(INSURERS)
        rd["state"] = random.choice(STATES)
        rd["plate_masked"] = mask_plate(plate)
        rd["profile_seed"] = profile
        records.append(rd)

        for p in photos:
            try:
                os.remove(p)
            except OSError:
                pass

    records.sort(key=lambda r: r["date"], reverse=True)
    return records


if __name__ == "__main__":
    t0 = time.time()
    data = build_dataset(140)
    out_path = "/home/claude/autovision/data/inspections_raw.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Generated {len(data)} inspections in {time.time() - t0:.1f}s -> {out_path}")