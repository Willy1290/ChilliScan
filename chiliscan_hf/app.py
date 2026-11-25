import streamlit as st
import cv2
import numpy as np
import joblib
import pandas as pd
from PIL import Image
import fitur_cabai as fc
import io
import os
from rembg import new_session

st.set_page_config(page_title="ChiliScan Dashboard", page_icon="🌶️", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# KONFIGURASI
# ============================================================
HASIL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
SK = "70_30"                       # skenario yang dipakai aplikasi
AMBANG_YAKIN = 0.60                # probabilitas maksimum minimal (sesuai Gambar 3.2)

OPSI_MODE = {"4 Kelas": "kelas4", "3 Kelas": "kelas3"}
URUT_KELAS = {"kelas4": ["sehat", "ringan", "sedang", "berat"],
              "kelas3": ["sehat", "ringan_sedang", "berat"]}

def path_model(mode): return os.path.join(HASIL, mode, f"svm_model_{SK}.pkl")
def path_cm(mode):    return os.path.join(HASIL, mode, f"confusion_matrix_{SK}.png")

# Satu sumber data untuk nama, warna, rekomendasi, dan panduan tiap kelas
INFO = {
    "sehat": dict(nama="Sehat", ikon="🟢", warna="#22c55e",
        judul="Daun Sehat", kondisi="Daun berfungsi normal dan tidak ada infeksi berarti.",
        tindakan="Lanjutkan perawatan rutin (penyiraman dan pemupukan). Tidak perlu pestisida.",
        visual="Daun berwarna hijau merata, tidak ada bercak, bentuk daun utuh.",
        agro="Daun berfungsi normal untuk fotosintesis. Tidak diperlukan pestisida."),
    "ringan": dict(nama="Ringan", ikon="🟡", warna="#eab308",
        judul="Nekrosis Ringan", kondisi="Bercak mulai muncul. Fotosintesis mulai sedikit terganggu.",
        tindakan="Segera petik daun yang berbercak agar tidak menular. Aplikasikan pestisida/fungisida organik dosis rendah.",
        visual="Muncul bercak coklat/kuning kecil pada pinggiran atau permukaan daun.",
        agro="Tahap awal infeksi. Petani dapat melakukan pencegahan dini sebelum menyebar."),
    "sedang": dict(nama="Sedang", ikon="🟠", warna="#f97316",
        judul="Nekrosis Sedang", kondisi="Infeksi mulai menyebar luas.",
        tindakan="Diperlukan penyemprotan fungisida/bakterisida kimia dengan dosis yang tepat. Pantau tanaman di sekitarnya.",
        visual="Bercak menyebar dan menyatu, warna daun berubah kekuningan kusam.",
        agro="Fotosintesis terganggu signifikan. Memerlukan penyemprotan fungisida dosis tepat agar panen tidak gagal."),
    "ringan_sedang": dict(nama="Ringan–Sedang", ikon="🟠", warna="#f59e0b",
        judul="Nekrosis Ringan–Sedang", kondisi="Bercak sudah muncul.",
        tindakan="Petik daun berbercak dan lakukan penyemprotan fungisida sesuai anjuran. Pantau perkembangan tanaman sekitar.",
        visual="Bercak coklat/kuning kecil sampai menyebar dan mulai menyatu pada permukaan daun.",
        agro="Infeksi sudah ada tetapi daun masih berfungsi. Petik daun berbercak, lakukan penyemprotan sesuai anjuran, dan pantau perkembangannya."),
    "berat": dict(nama="Berat", ikon="🔴", warna="#ef4444",
        judul="Nekrosis Berat", kondisi="Jaringan daun sudah banyak yang mati dan mengering.",
        tindakan="Pangkas total daun yang sakit atau cabut tanaman. Lakukan penyemprotan masif pada tanaman sehat di sekitarnya.",
        visual="Jaringan daun mati (kering, coklat/hitam). Daun menggulung dan bentuknya rusak parah.",
        agro="Tindakan darurat (pemangkasan/pencabutan) diperlukan untuk mencegah penularan masif."),
}
DEFAULT_INFO = dict(nama="?", ikon="⚪", warna="#0ea5e9", judul="Hasil", kondisi="", tindakan="", visual="", agro="")

# ============================================================
# TAMPILAN (CSS)
# ============================================================
def html(s):
    """Render HTML satu baris (menghindari markdown mengira indentasi = blok kode)."""
    st.markdown("".join(line.strip() for line in s.strip().splitlines()), unsafe_allow_html=True)

html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root { --primary:#0ea5e9; --primary-d:#0369a1; --bg:#f1f5f9; --card:#ffffff; --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; }
html, body, [class*="css"] { font-family:'Inter',sans-serif; }
.stApp { background:var(--bg); }
.block-container { padding-top:1.6rem; padding-bottom:3rem; max-width:1250px; }
#MainMenu, footer { visibility:hidden; }
[data-testid="stHeader"] { background:transparent; }

[data-testid="stSidebar"] { background:linear-gradient(180deg,#0b1220 0%,#0f172a 100%) !important; border-right:1px solid #1e293b; }
[data-testid="stSidebar"] hr { border-color:#1e293b; margin:1.1rem 0; }
[data-testid="stSidebar"] .stRadio > div { gap:4px; }
[data-testid="stSidebar"] .stRadio label { padding:11px 14px; border-radius:10px; cursor:pointer; transition:all .18s ease; background:transparent; width:100%; }
[data-testid="stSidebar"] .stRadio label, [data-testid="stSidebar"] .stRadio label * { color:#cbd5e1 !important; font-weight:500; font-size:14.5px; }
[data-testid="stSidebar"] .stRadio label:hover { background:rgba(255,255,255,.07); }
[data-testid="stSidebar"] .stRadio label:has(input:checked) { background:linear-gradient(135deg,#0ea5e9,#0284c7); box-shadow:0 6px 14px -4px rgba(14,165,233,.55); }
[data-testid="stSidebar"] .stRadio label:has(input:checked), [data-testid="stSidebar"] .stRadio label:has(input:checked) * { color:#fff !important; font-weight:600; }
[data-testid="stSidebar"] .stRadio label > div:first-child { display:none; }
.st-key-mode_kelas [role="radiogroup"] { flex-direction:row; background:rgba(255,255,255,.06); border-radius:12px; padding:4px; gap:4px; }
.st-key-mode_kelas label { flex:1; justify-content:center; text-align:center; padding:9px 6px !important; }

.brand { display:flex; align-items:center; gap:12px; margin-bottom:22px; }
.brand-ico { width:44px; height:44px; border-radius:12px; background:linear-gradient(135deg,#0ea5e9,#6366f1); display:flex; align-items:center; justify-content:center; font-size:24px; }
.brand-t { font-size:20px; font-weight:800; color:#f8fafc; line-height:1.1; }
.brand-s { font-size:12px; color:#94a3b8; margin-top:2px; }
.side-lbl { font-size:11px; font-weight:700; letter-spacing:1px; text-transform:uppercase; color:#64748b; margin:0 0 8px 2px; }
.model-card { padding:14px 16px; border-radius:12px; background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.09); }
.model-card .k { font-size:11px; color:#94a3b8; text-transform:uppercase; letter-spacing:.8px; }
.model-card .v { font-size:15px; font-weight:700; color:#f8fafc; margin-top:4px; }
.model-card .s { font-size:12.5px; color:#38bdf8; margin-top:4px; }
.kelas-note { font-size:12px; color:#94a3b8; margin-top:8px; line-height:1.5; }

.hero { background:linear-gradient(120deg,#0369a1 0%,#0ea5e9 55%,#38bdf8 100%); border-radius:18px; padding:26px 30px; margin-bottom:22px; display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:14px; box-shadow:0 14px 30px -12px rgba(2,132,199,.55); }
.hero-t { font-size:28px; font-weight:800; color:#fff; letter-spacing:-.4px; }
.hero-s { font-size:14.5px; color:#e0f2fe; margin-top:4px; }
.chips { display:flex; flex-wrap:wrap; gap:8px; }
.chip { background:rgba(255,255,255,.18); color:#fff; padding:6px 13px; border-radius:999px; font-size:12.5px; font-weight:600; border:1px solid rgba(255,255,255,.25); }
.chip-on { background:#fff; color:#0369a1; }

.sec { display:flex; align-items:center; gap:10px; font-size:17px; font-weight:700; color:var(--ink); margin:4px 0 12px; }
.sec-no { width:26px; height:26px; border-radius:8px; background:var(--primary); color:#fff; display:inline-flex; align-items:center; justify-content:center; font-size:13px; font-weight:700; }

.card { background:var(--card); border:1px solid var(--line); border-radius:16px; padding:20px 22px; margin-bottom:16px; box-shadow:0 2px 6px rgba(15,23,42,.04); }
.card-t { font-size:13px; font-weight:700; color:var(--muted); text-transform:uppercase; letter-spacing:.7px; margin-bottom:14px; }

.res-card { border-radius:16px; padding:24px 26px; color:#fff; margin-bottom:16px; }
.res-top { display:flex; justify-content:space-between; align-items:center; }
.res-tag { font-size:12px; text-transform:uppercase; letter-spacing:1.2px; opacity:.92; font-weight:600; }
.res-ico { font-size:26px; }
.res-name { font-size:38px; font-weight:800; letter-spacing:-.8px; margin:6px 0 14px; color:#fff; }
.res-conf { display:flex; justify-content:space-between; font-size:13px; opacity:.95; margin-bottom:6px; }
.res-conf b { font-size:15px; }
.res-bar { height:8px; border-radius:99px; background:rgba(255,255,255,.3); overflow:hidden; }
.res-bar > div { height:100%; background:#fff; border-radius:99px; }

.pr-row { display:flex; align-items:center; gap:12px; margin-bottom:11px; }
.pr-name { width:110px; font-size:13.5px; color:var(--muted); font-weight:500; }
.pr-track { flex:1; height:11px; background:#f1f5f9; border-radius:99px; overflow:hidden; }
.pr-fill { height:100%; border-radius:99px; }
.pr-val { width:52px; text-align:right; font-size:13px; font-weight:600; color:var(--muted); }
.pr-row.on .pr-name, .pr-row.on .pr-val { color:var(--ink); font-weight:700; }

.rec { border-radius:16px; padding:20px 22px; margin-bottom:16px; border:1px solid var(--line); border-left:6px solid var(--c); background:var(--card); }
.rec-h { display:flex; align-items:center; gap:10px; font-size:18px; font-weight:700; color:var(--ink); }
.rec-d { font-size:14px; color:var(--muted); margin:8px 0 14px; line-height:1.55; }
.rec-a { background:#f8fafc; border-radius:10px; padding:12px 14px; font-size:14px; color:var(--ink); line-height:1.55; }
.rec-a b { color:var(--c); }

.kpi-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:16px; margin-bottom:18px; }
.kpi { background:var(--card); border:1px solid var(--line); border-top:4px solid var(--a); border-radius:14px; padding:20px 22px; box-shadow:0 2px 6px rgba(15,23,42,.04); transition:transform .2s; }
.kpi:hover { transform:translateY(-3px); }
.kpi-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; }
.kpi-title { font-size:12px; color:var(--muted); font-weight:700; text-transform:uppercase; letter-spacing:.7px; }
.kpi-ico { font-size:20px; }
.kpi-value { font-size:32px; font-weight:800; color:var(--ink); line-height:1; letter-spacing:-.5px; }
.kpi-sub { font-size:12px; color:var(--muted); margin-top:8px; min-height:16px; }

.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(270px,1fr)); gap:16px; margin-bottom:16px; }
.gcard { background:var(--card); border:1px solid var(--line); border-top:5px solid var(--c); border-radius:14px; padding:20px 22px; box-shadow:0 2px 6px rgba(15,23,42,.04); }
.gh { display:flex; align-items:center; gap:10px; font-size:17px; font-weight:800; color:var(--ink); margin-bottom:12px; }
.gl { font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.8px; color:var(--c); margin-top:10px; }
.gt { font-size:14px; color:var(--muted); line-height:1.55; margin-top:3px; }
.note { background:#e0f2fe; border:1px solid #bae6fd; border-radius:14px; padding:18px 22px; color:#075985; font-size:14px; line-height:1.6; }
.note b { color:#0c4a6e; }

.empty { background:var(--card); border:1px dashed #cbd5e1; border-radius:18px; padding:34px 24px; text-align:center; margin-top:16px; }
.empty-ico { font-size:44px; }
.empty-t { font-size:20px; font-weight:700; color:var(--ink); margin-top:6px; }
.empty-s { font-size:14px; color:var(--muted); margin:6px 0 22px; }
.steps { display:flex; flex-wrap:wrap; justify-content:center; gap:12px; }
.step { background:#f8fafc; border:1px solid var(--line); border-radius:12px; padding:12px 16px; font-size:13.5px; color:var(--ink); display:flex; align-items:center; gap:10px; font-weight:500; }
.step b { width:24px; height:24px; border-radius:50%; background:var(--primary); color:#fff; display:inline-flex; align-items:center; justify-content:center; font-size:12px; }

.warn { background:#fef2f2; border:1px solid #fecaca; border-left:6px solid #ef4444; border-radius:14px; padding:18px 20px; color:#7f1d1d; font-size:14.5px; line-height:1.55; }
.warn.amber { background:#fffbeb; border-color:#fde68a; border-left-color:#f59e0b; color:#78350f; }

[data-testid="stFileUploader"] section { border:2px dashed #cbd5e1; border-radius:16px; background:#fff; padding:22px; transition:all .2s; }
[data-testid="stFileUploader"] section:hover { border-color:var(--primary); background:#f0f9ff; }
[data-testid="stImage"] img { border-radius:10px; }
[data-testid="stExpander"] { background:#fff; border-radius:14px; border:1px solid var(--line); }
.stTabs [data-baseweb="tab-list"] { gap:8px; }
.stTabs [data-baseweb="tab"] { padding:8px 16px; background:#fff; border-radius:10px; color:var(--muted); font-weight:600; font-size:14px; border:1px solid var(--line); }
.stTabs [aria-selected="true"] { background:var(--primary) !important; color:#fff !important; border-color:var(--primary) !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }
</style>
""")

# ============================================================
# MODEL & PROSES (logika tidak diubah)
# ============================================================
@st.cache_resource
def load_model(mode):
    try:
        bundle = joblib.load(path_model(mode))
        return bundle["model"], bundle["scaler"], bundle["encoder"]
    except Exception:
        return None, None, None

@st.cache_resource
def load_rembg():
    return new_session("u2netp")

@st.cache_data(show_spinner=False)
def hapus_background(image_bytes):
    """Sama persis dengan pembuatan dataset (fitur_cabai.siapkan_citra). Di-cache agar ganti mode tidak mengulang proses."""
    arr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    img, status, alasan = fc.siapkan_citra(arr, load_rembg())
    return img

@st.cache_data
def ambil_metrik(mode):
    p = os.path.join(HASIL, mode, "ringkasan_semua.csv")
    if not os.path.exists(p): return None
    d = pd.read_csv(p); d = d[d.skenario == SK]
    return None if d.empty else d.iloc[0]

def ekstrak(img_bgr):
    """CLAHE -> mask + morfologi -> 12 fitur morfologi + 57 fitur HSV = 69 fitur."""
    img_pre = fc.clahe(img_bgr)
    hsv, solid, nek, c = fc.mask_daun_nekrosis(img_pre)
    fitur = np.concatenate([fc.fitur_morfologi(solid, nek, c), fc.fitur_hsv(hsv, solid)])
    return fitur, img_pre, solid, nek

def cek_daun_valid(img_clean_bgr):
    hsv = cv2.cvtColor(img_clean_bgr, cv2.COLOR_BGR2HSV)
    mask_bg = cv2.inRange(hsv, np.array([0, 0, 230]), np.array([180, 30, 255]))
    mask_leaf = cv2.bitwise_not(mask_bg)
    contours, _ = cv2.findContours(mask_leaf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return False, "Tidak ada objek yang terdeteksi."
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    total_pixel = img_clean_bgr.shape[0] * img_clean_bgr.shape[1]
    if area < (0.03 * total_pixel): return False, "Objek terlalu kecil atau gambar tidak jelas."
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(min(w, h)) / max(w, h)
    if aspect_ratio > 0.85: return False, "Bentuk objek terlalu membulat/kotak."
    mask_hijau = cv2.inRange(hsv, np.array([25, 30, 30]), np.array([90, 255, 255]))
    mask_coklat = cv2.inRange(hsv, np.array([5, 40, 20]), np.array([25, 255, 200]))
    mask_gelap = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 60]))
    mask_valid_color = cv2.bitwise_or(mask_hijau, cv2.bitwise_or(mask_coklat, mask_gelap))
    mask_valid_leaf = cv2.bitwise_and(mask_valid_color, mask_leaf)
    rasio_valid = np.sum(mask_valid_leaf > 0) / (area + 1e-6)
    if rasio_valid < 0.30: return False, "Warna objek tidak sesuai dengan karakteristik daun cabai."
    return True, "Valid"

def prediksi_gambar(image_bgr, model, scaler, le):
    fitur, img_pre, mask_leaf, mask_nec = ekstrak(image_bgr)
    proba = model.predict_proba(scaler.transform(fitur.reshape(1, -1)))[0]
    label_pred = le.inverse_transform([int(np.argmax(proba))])[0]   # murni keluaran SVM
    return label_pred, proba, le.classes_, img_pre, mask_leaf, mask_nec, fitur

# ============================================================
# KOMPONEN UI
# ============================================================
def hero(pilih):
    html(f"""<div class="hero"><div><div class="hero-t">Dashboard Analisis Daun Cabai</div>
    <div class="hero-s">Deteksi tingkat keparahan nekrosis berbasis AI</div></div>
    <div class="chips"><span class="chip">SVM · RBF</span><span class="chip">69 fitur</span>
    <span class="chip">Skenario {SK.replace('_', ':')}</span><span class="chip chip-on">{pilih}</span></div></div>""")

def bagian(no, judul):
    html(f'<div class="sec"><span class="sec-no">{no}</span>{judul}</div>')

def kartu_hasil(label, proba, pilih):
    i = INFO.get(label, DEFAULT_INFO); w = i["warna"]; conf = float(np.max(proba)) * 100
    html(f"""<div class="res-card" style="background:linear-gradient(135deg,{w},{w}cc); box-shadow:0 14px 26px -12px {w};">
    <div class="res-top"><span class="res-tag">Prediksi · {pilih}</span><span class="res-ico">{i['ikon']}</span></div>
    <div class="res-name">{i['nama']}</div>
    <div class="res-conf"><span>Tingkat keyakinan model</span><b>{conf:.1f}%</b></div>
    <div class="res-bar"><div style="width:{conf:.1f}%"></div></div></div>""")

def kartu_probabilitas(classes, proba, pred):
    baris = ""
    for k, p in zip(classes, proba):
        i = INFO.get(k, DEFAULT_INFO)
        on = " on" if k == pred else ""
        baris += (f'<div class="pr-row{on}"><div class="pr-name">{i["nama"]}</div>'
                  f'<div class="pr-track"><div class="pr-fill" style="width:{p*100:.1f}%; background:{i["warna"]};"></div></div>'
                  f'<div class="pr-val">{p*100:.1f}%</div></div>')
    html(f'<div class="card"><div class="card-t">Distribusi probabilitas</div>{baris}</div>')

def kartu_rekomendasi(label):
    i = INFO.get(label, DEFAULT_INFO)
    html(f"""<div class="rec" style="--c:{i['warna']};">
    <div class="rec-h"><span>{i['ikon']}</span>{i['judul']}</div>
    <div class="rec-d">{i['kondisi']}</div>
    <div class="rec-a"><b>Tindakan:</b> {i['tindakan']}</div></div>""")

def kpi(judul, nilai, ikon, warna, sub=""):
    return (f'<div class="kpi" style="--a:{warna};"><div class="kpi-top"><span class="kpi-title">{judul}</span>'
            f'<span class="kpi-ico">{ikon}</span></div><div class="kpi-value">{nilai}</div><div class="kpi-sub">{sub}</div></div>')

# ============================================================
# HALAMAN
# ============================================================
def halaman_deteksi(model, scaler, le, pilih):
    up = st.file_uploader("Upload Gambar (JPG/PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if up is None:
        html("""<div class="empty"><div class="empty-ico">🍃</div>
        <div class="empty-t">Unggah foto daun cabai untuk memulai</div>
        <div class="empty-s">Format JPG/PNG. Gunakan foto satu daun dengan pencahayaan merata.</div>
        <div class="steps"><div class="step"><b>1</b>Unggah gambar</div><div class="step"><b>2</b>Hapus background</div>
        <div class="step"><b>3</b>Ekstraksi 69 fitur</div><div class="step"><b>4</b>Klasifikasi SVM</div></div></div>""")
        return

    file_bytes = up.getvalue()
    img_asli = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    col_img, col_res = st.columns([1, 1], gap="large")

    with col_img:
        bagian("1", "Input Gambar")
        with st.container(border=True):
            st.image(img_asli, caption="Gambar Asli", use_container_width=True)
        with st.spinner("AI sedang menghapus background..."):
            img_clean_bgr = hapus_background(file_bytes)
        with st.container(border=True):
            st.image(cv2.cvtColor(img_clean_bgr, cv2.COLOR_BGR2RGB), caption="Setelah Background Dihapus (AI)", use_container_width=True)

    hasil = None
    with col_res:
        bagian("2", "Hasil Analisis")
        valid, pesan = cek_daun_valid(img_clean_bgr)
        if not valid:
            html(f'<div class="warn">❌ <b>Deteksi gagal:</b> {pesan}</div>')
        else:
            with st.spinner("Memprediksi kelas..."):
                label, proba, classes, img_pre, mask_leaf, mask_nec, fitur = prediksi_gambar(img_clean_bgr, model, scaler, le)
            if float(np.max(proba)) < AMBANG_YAKIN:
                html(f'<div class="warn amber">⚠️ <b>Keyakinan model rendah</b> ({np.max(proba)*100:.1f}% &lt; {AMBANG_YAKIN*100:.0f}%). '
                     f'Citra mungkin bukan kondisi yang dikenali sistem.</div>')
            else:
                kartu_hasil(label, proba, pilih)
                kartu_probabilitas(classes, proba, label)
                kartu_rekomendasi(label)
                hasil = (img_pre, mask_leaf, mask_nec, fitur)

    if hasil is not None:
        img_pre, mask_leaf, mask_nec, fitur = hasil
        with st.expander("🔍 Detail Ekstraksi Fitur (Masking, Morfologi & HSV)"):
            dt1, dt2, dt3, dt4 = st.tabs(["Masking & Morfologi", "Pemisahan HSV", "Tabel Morfologi", "Tabel Warna (HSV)"])
            with dt1:
                c1, c2 = st.columns(2)
                with c1: st.image(mask_leaf, caption="Mask Daun (Putih = Area Daun)", use_container_width=True, clamp=True)
                with c2: st.image(mask_nec, caption="Mask Nekrosis (Putih = Bercak Mati)", use_container_width=True, clamp=True)
            with dt2:
                h, s, v = cv2.split(cv2.cvtColor(img_pre, cv2.COLOR_BGR2HSV))
                c1, c2, c3 = st.columns(3)
                with c1: st.image(h, caption="Hue (Warna)", use_container_width=True, clamp=True)
                with c2: st.image(s, caption="Saturation (Kejenuhan)", use_container_width=True, clamp=True)
                with c3: st.image(v, caption="Value (Kecerahan)", use_container_width=True, clamp=True)
            with dt3:
                st.dataframe(pd.DataFrame({
                    "Fitur Morfologi": ["Area (Luas)", "Perimeter (Keliling)", "Aspect Ratio", "Extent", "Compactness", "Roundness", "Eccentricity", "Solidity", "Major Axis", "Minor Axis", "Eq. Diameter", "Rasio Nekrosis (%)"],
                    "Nilai": [f"{fitur[i]:.4f}" for i in range(11)] + [f"{fitur[11]*100:.2f}%"]
                }), use_container_width=True, hide_index=True)
            with dt4:
                st.dataframe(pd.DataFrame({
                    "Fitur Warna (HSV)": ["H Mean", "H Std", "H Skew", "S Mean", "S Std", "S Skew", "V Mean", "V Std", "V Skew"],
                    "Nilai": [f"{fitur[i]:.2f}" for i in range(12, 21)]
                }), use_container_width=True, hide_index=True)
                import plotly.express as px
                hist_df = pd.DataFrame({"Bin": list(range(1, 17)) * 3, "Kanal": ["H"] * 16 + ["S"] * 16 + ["V"] * 16, "Proporsi": fitur[21:69]})
                fig = px.bar(hist_df, x="Bin", y="Proporsi", color="Kanal", barmode="group", height=280,
                             color_discrete_map={"H": "#0ea5e9", "S": "#22c55e", "V": "#f59e0b"})
                fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  legend=dict(orientation="h", y=1.1, x=0, title=None), yaxis=dict(gridcolor="#f1f5f9"))
                st.plotly_chart(fig, use_container_width=True)

def halaman_evaluasi(mode, pilih):
    bagian("📊", f"Evaluasi Performa Model ({pilih})")
    html(f'<div style="color:#64748b; font-size:14px; margin:-4px 0 16px;">Hasil evaluasi pada data uji skenario {SK.replace("_", ":")} '
         f'(split per grup, tanpa citra kembar antara data latih dan uji).</div>')
    m = ambil_metrik(mode)
    if m is None:
        html('<div class="warn amber">ringkasan_semua.csv belum ditemukan. Jalankan latih_svm.py terlebih dahulu.</div>')
    else:
        sub_acc = ""
        if "acc_dilipat3" in m.index and mode == "kelas4":
            sub_acc = f"Jika dilipat ke 3 kelas: {m.acc_dilipat3*100:.2f}%"
        n_tr, n_te = m.get("n_train"), m.get("n_test")
        data_txt = f"{int(n_tr)} / {int(n_te)}" if n_tr is not None and n_te is not None else "-"
        html('<div class="kpi-grid">'
             + kpi("Akurasi Test", f"{m.test_acc*100:.2f}%", "🎯", "#0ea5e9", sub_acc)
             + kpi("Macro F1-Score", f"{m.macro_f1*100:.2f}%", "⚖️", "#22c55e", "Rata-rata F1 antar kelas")
             + kpi("Weighted F1", f"{m.weighted_f1*100:.2f}%", "📈", "#f59e0b", "Dibobot jumlah data tiap kelas")
             + kpi("Data Latih / Uji", data_txt, "🗂️", "#6366f1", "Kernel RBF · 69 fitur")
             + '</div>')

    col_cm, col_lap = st.columns([1, 1], gap="large")
    with col_cm:
        html(f'<div class="card-t">Confusion Matrix ({SK.replace("_", ":")}, {pilih})</div>')
        with st.container(border=True):
            if os.path.exists(path_cm(mode)): st.image(path_cm(mode), use_container_width=True, caption=f"Confusion Matrix SVM ({pilih})")
            else: st.warning("Gambar Confusion Matrix tidak ditemukan.")
    with col_lap:
        html('<div class="card-t">Laporan Klasifikasi</div>')
        lap = os.path.join(HASIL, mode, f"laporan_{SK}.txt")
        if os.path.exists(lap): st.code(open(lap, encoding="utf8").read())
        else: st.warning("File laporan tidak ditemukan.")
        html('<div class="note"><b>Fitur yang dipakai:</b> 69 fitur, terdiri dari 12 fitur morfologi (bentuk dan rasio nekrosis) '
             'dan 57 fitur warna HSV (9 statistik + 48 histogram).</div>')

def halaman_panduan(mode, pilih):
    bagian("📖", f"Panduan Tingkat Keparahan Nekrosis ({pilih})")
    html('<div style="color:#64748b; font-size:14px; margin:-4px 0 16px;">Alasan ilmiah dan agronomis di balik pembagian kelas pada model SVM.</div>')
    kartu = "".join(
        f'<div class="gcard" style="--c:{INFO[k]["warna"]};"><div class="gh"><span>{INFO[k]["ikon"]}</span>{INFO[k]["nama"].upper()}</div>'
        f'<div class="gl">Karakteristik Visual</div><div class="gt">{INFO[k]["visual"]}</div>'
        f'<div class="gl">Alasan Agronomis</div><div class="gt">{INFO[k]["agro"]}</div></div>'
        for k in URUT_KELAS[mode])
    html(f'<div class="grid">{kartu}</div>')
    html('<div class="note"><b>Bagaimana kelas ditentukan?</b><br>Kelas ditentukan penulis lewat penilaian visual (jumlah, ukuran, dan sebaran bercak). '
         'Rasio nekrosis ditampilkan sebagai informasi pendukung dan menjadi salah satu fitur SVM, tetapi bukan penentu kelas: '
         'pada data penelitian ini rasio saja hanya mencapai sekitar separuh akurasi, terutama karena kelas Ringan dan Sedang banyak tumpang tindih. '
         'Kelas yang bersebelahan (terutama Ringan dan Sedang) lebih rawan tertukar.</div>')

# ============================================================
# MAIN
# ============================================================
def main():
    with st.sidebar:
        html('<div class="brand"><div class="brand-ico">🌶️</div><div><div class="brand-t">ChiliScan</div>'
             '<div class="brand-s">Necrosis Detection System</div></div></div>')
        html('<div class="side-lbl">Menu</div>')
        page = st.radio("Navigasi", ["🔍 Deteksi Gambar", "📊 Evaluasi Model", "📖 Panduan Klasifikasi"], label_visibility="collapsed")
        st.divider()

        html('<div class="side-lbl">Jumlah Kelas</div>')
        pilih = st.radio("Mode Kelas", list(OPSI_MODE.keys()), key="mode_kelas", horizontal=True, label_visibility="collapsed")
        MODE = OPSI_MODE[pilih]
        daftar = " · ".join(INFO[k]["nama"] for k in URUT_KELAS[MODE])
        html(f'<div class="kelas-note">{daftar}</div>')
        st.divider()

        html(f'<div class="model-card"><div class="k">Model Aktif</div><div class="v">SVM (RBF) · 69 Fitur</div>'
             f'<div class="s">Skenario {SK.replace("_", ":")} · {pilih}</div></div>')

    model, scaler, le = load_model(MODE)
    if model is None:
        st.error(f"Gagal memuat model. Pastikan path `{path_model(MODE)}` benar.")
        return

    hero(pilih)
    if page == "🔍 Deteksi Gambar":
        halaman_deteksi(model, scaler, le, pilih)
    elif page == "📊 Evaluasi Model":
        halaman_evaluasi(MODE, pilih)
    else:
        halaman_panduan(MODE, pilih)

if __name__ == "__main__":
    main()
