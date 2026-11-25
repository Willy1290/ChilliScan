"""
fitur_cabai.py - SATU-SATUNYA tempat logika praproses + ekstraksi fitur.
Dipakai oleh siapkan_dataset.py, latih_svm.py, DAN aplikasi (app.py), supaya
pelatihan dan aplikasi pasti memakai proses yang sama (tanpa blur tambahan).

Masukan : citra BGR 256x256, latar sudah PUTIH (hasil rembg).
Keluaran: vektor 69 fitur = 12 morfologi + 57 HSV.
"""
import cv2, numpy as np
from skimage.measure import label, regionprops

UKURAN = 256
BATAS_SEHAT, BATAS_RINGAN, BATAS_SEDANG = 0.01, 0.15, 0.35   # bawaan: R<1% sehat, <=15% ringan, <=35% sedang, >35% berat
import json as _json, os as _os
_p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "ambang.json")
if _os.path.exists(_p):        # dibuat oleh cari_ambang.py (ambang dari data penilaian manual)
    _a = _json.load(open(_p, encoding="utf8"))
    BATAS_SEHAT, BATAS_RINGAN, BATAS_SEDANG = _a["sehat"], _a["ringan"], _a["sedang"]
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
IDX_RASIO = 11
HIJAU_BAWAH, HIJAU_ATAS = (25, 35, 35), (90, 255, 255)   # rentang HSV 'hijau sehat'; nekrosis = daun yang BUKAN hijau
NEKROSIS = "bukan_hijau"     # "bukan_hijau" (kode rasio_train.py kamu) atau "H0_35" (kode lama rasio_dataset.py)

def hapus_latar(img_bgr, sesi):
    """rembg (u2netp) -> latar putih -> resize 256x256."""
    from rembg import remove
    rgba = remove(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), session=sesi, post_process_mask=True)
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = (rgba[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
    return cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (UKURAN, UKURAN))

def sudah_latar_putih(img_bgr):
    """True bila bingkai tepi citra sudah putih seragam (mis. hasil Roboflow 'backremoved') -> rembg tidak perlu/merusak."""
    h, w = img_bgr.shape[:2]; b = max(2, min(h, w) // 40)
    f = np.concatenate([img_bgr[:b].reshape(-1, 3), img_bgr[-b:].reshape(-1, 3), img_bgr[:, :b].reshape(-1, 3), img_bgr[:, -b:].reshape(-1, 3)])
    return bool(f.min(1).mean() > 240 and f.std(0).max() < 12)

def kualitas_segmentasi(img256):
    """(ok, alasan, luas_daun_persen, median_S). Gagal bila daun nyaris hilang atau hanya bayangan pucat sisa rembg."""
    hsv = cv2.cvtColor(img256, cv2.COLOR_BGR2HSV)
    daun = cv2.bitwise_not(cv2.inRange(hsv, (0, 0, 240), (180, 20, 255)))
    frac = float((daun > 0).mean()); L = hsv[..., 1][daun > 0]; ms = float(np.median(L)) if len(L) else 0.0
    if frac < 0.04: return False, "daun terlalu kecil/hilang", frac * 100, ms
    if ms < 30: return False, "daun pucat/abu-abu (sisa rembg)", frac * 100, ms
    return True, "ok", frac * 100, ms

def siapkan_citra(img_bgr, sesi=None):
    """rembg (kecuali sudah berlatar putih atau sesi=None) -> 256x256 -> cek kualitas. Mengembalikan (citra, status, alasan)."""
    if sesi is None or sudah_latar_putih(img_bgr):
        out = cv2.resize(img_bgr, (UKURAN, UKURAN)); st = "tanpa_rembg"
    else:
        out = hapus_latar(img_bgr, sesi); st = "rembg"
    ok, alasan, _, _ = kualitas_segmentasi(out)
    return out, (st if ok else "gagal"), alasan

def clahe(img_bgr):
    l, a, b = cv2.split(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def mask_daun_nekrosis(img_bgr):
    """Mask daun (kontur terbesar, lubang diisi) dan mask nekrosis, keduanya
    dibersihkan dengan operasi morfologi (opening lalu closing, elips 3x3)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    daun = cv2.bitwise_not(cv2.inRange(hsv, (0, 0, 240), (180, 20, 255)))
    daun = cv2.morphologyEx(daun, cv2.MORPH_OPEN, KERNEL)
    daun = cv2.morphologyEx(daun, cv2.MORPH_CLOSE, KERNEL)
    cnts, _ = cv2.findContours(daun, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return hsv, np.zeros_like(daun), np.zeros_like(daun), None
    c = max(cnts, key=cv2.contourArea)
    solid = np.zeros_like(daun); cv2.drawContours(solid, [c], -1, 255, -1)
    if NEKROSIS == "bukan_hijau":      # nekrosis = bagian daun yang BUKAN hijau (termasuk pusat bercak abu-abu/putih)
        nek = cv2.bitwise_not(cv2.inRange(hsv, HIJAU_BAWAH, HIJAU_ATAS))
    else:                              # definisi lama: rona coklat/kuning saja (H 0-35)
        nek = cv2.inRange(hsv, (0, 40, 20), (35, 255, 220))
    nek = cv2.bitwise_and(nek, solid)
    nek = cv2.morphologyEx(nek, cv2.MORPH_OPEN, KERNEL)
    nek = cv2.morphologyEx(nek, cv2.MORPH_CLOSE, KERNEL)
    nek = cv2.bitwise_and(nek, solid)
    return hsv, solid, nek, c

def rasio_nekrosis(img_bgr):
    """Dipakai untuk PELABELAN (citra tanpa CLAHE)."""
    _, solid, nek, c = mask_daun_nekrosis(img_bgr)
    if c is None or (solid > 0).sum() == 0: return 0.0
    return float((nek > 0).sum() / (solid > 0).sum())

def label_dari_rasio(r):
    return "ringan" if r <= BATAS_RINGAN else ("sedang" if r <= BATAS_SEDANG else "berat")

def kelas_dari_rasio(r):
    """Empat kelas dari rasio saja (termasuk sehat)."""
    return "sehat" if r < BATAS_SEHAT else label_dari_rasio(r)

def _g(p, baru, lama):
    if p is None: return 0
    return getattr(p, baru) if hasattr(p, baru) else getattr(p, lama)

def fitur_morfologi(solid, nek, c):
    if c is None: return np.zeros(12, np.float32)
    area_px = float((solid > 0).sum())
    area = cv2.contourArea(c)
    if area == 0: return np.zeros(12, np.float32)
    per = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)
    props = regionprops(label(solid))
    p = max(props, key=lambda r: r.area) if props else None
    ratio = float((nek > 0).sum() / (area_px + 1e-6))
    return np.array([area, per, w / (h + 1e-6), area / (w * h + 1e-6),
                     per ** 2 / (4 * np.pi * area + 1e-6), 4 * np.pi * area / (per ** 2 + 1e-6),
                     p.eccentricity if p else 0, p.solidity if p else 0,
                     _g(p, 'axis_major_length', 'major_axis_length'), _g(p, 'axis_minor_length', 'minor_axis_length'),
                     _g(p, 'equivalent_diameter_area', 'equivalent_diameter'), ratio], np.float32)

def fitur_hsv(hsv, solid):
    f = []
    for i in range(3):
        v = hsv[..., i][solid > 0]
        if len(v) == 0: f += [0, 0, 0]; continue
        m, s = v.mean(), v.std()
        f += [m, s, np.mean(((v - m) / (s + 1e-6)) ** 3)]
    for i, rng in enumerate(([0, 180], [0, 256], [0, 256])):
        h = cv2.calcHist([hsv], [i], solid, [16], rng).flatten()
        f += (h / (h.sum() + 1e-6)).tolist()
    return np.array(f, np.float32)

def ekstrak_69(img_putih_bgr):
    """img_putih_bgr: citra 256x256 latar putih (tanpa CLAHE). CLAHE diterapkan di sini."""
    img = clahe(img_putih_bgr)
    hsv, solid, nek, c = mask_daun_nekrosis(img)
    return np.concatenate([fitur_morfologi(solid, nek, c), fitur_hsv(hsv, solid)])
