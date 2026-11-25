import streamlit as st
import cv2
import numpy as np
import joblib
import pandas as pd
import plotly.express as px
from PIL import Image
import fitur_cabai as fc
import io
import os
from rembg import new_session

st.set_page_config(page_title="ChiliScan Dashboard", page_icon="🌶️", layout="wide", initial_sidebar_state="expanded")

# Folder keluaran latih_svm.py
HASIL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")  # folder model/ di repo
SK = "70_30"                       # skenario yang dipakai aplikasi
MODE = "kelas4"                    # "kelas4" (sehat/ringan/sedang/berat) atau "kelas3" (sehat/ringan_sedang/berat)
MODEL_PATH = os.path.join(HASIL, f"svm_model_{SK}.pkl")
CM_PATH = os.path.join(HASIL, f"confusion_matrix_{SK}.png")
AMBANG_YAKIN = 0.60                # probabilitas maksimum minimal (sesuai Gambar 3.2)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    :root { --primary: #0ea5e9; --bg-color: #f8fafc; --sidebar-bg: #0f172a; --text-main: #1e293b; --text-muted: #64748b; }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: var(--bg-color); }
    [data-testid="stSidebar"] { background-color: var(--sidebar-bg) !important; border-right: 1px solid #1e293b; }
    [data-testid="stSidebar"] * { color: #f8fafc !important; }
    [data-testid="stSidebar"] .stRadio > div { gap: 5px; }
    [data-testid="stSidebar"] .stRadio label { padding: 12px 15px; border-radius: 8px; cursor: pointer; font-weight: 500; font-size: 15px; transition: all 0.2s ease; background-color: transparent; }
    [data-testid="stSidebar"] .stRadio label:hover { background-color: rgba(255, 255, 255, 0.08); }
    [data-testid="stSidebar"] .stRadio label[data-checked="true"] { background-color: var(--primary) !important; }
    [data-testid="stSidebar"] .stRadio label div:first-child { display: none; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} [data-testid="stHeader"] { background: transparent; }
    .kpi-card { background-color: #ffffff; padding: 24px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); transition: transform 0.2s; }
    .kpi-card:hover { transform: translateY(-2px); }
    .kpi-title { font-size: 13px; color: var(--text-muted); margin-bottom: 8px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;}
    .kpi-value { font-size: 32px; color: var(--text-main); font-weight: 700; margin: 0; line-height: 1;}
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 16px; background-color: #ffffff; border-radius: 8px; color: var(--text-muted); font-weight: 600; font-size: 14px; border: 1px solid #e2e8f0; }
    .stTabs [aria-selected="true"] { background-color: var(--primary) !important; color: white !important; border-color: var(--primary) !important; }
    .stTabs [data-baseweb="tab-highlight"] { display: none; } .stTabs [data-baseweb="tab-border"] { display: none; }
    .info-box { padding: 20px; border-radius: 12px; margin-bottom: 15px; border-left: 6px solid; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
    .info-box h4 { font-size: 18px; margin-bottom: 8px; } .info-box p { font-size: 14px; color: var(--text-muted); margin-bottom: 6px; line-height: 1.5; }
    .info-sehat { border-color: #22c55e; } .info-ringan { border-color: #eab308; } .info-sedang { border-color: #f97316; } .info-berat { border-color: #ef4444; }
    [data-testid="stExpander"] { background-color: white; border-radius: 12px; border: 1px solid #e2e8f0; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        bundle = joblib.load(MODEL_PATH)
        return bundle["model"], bundle["scaler"], bundle["encoder"]
    except Exception as e:
        return None, None, None

@st.cache_resource
def load_rembg():
    return new_session("u2netp")

def hapus_background(image_bytes):
    """Sama persis dengan pembuatan dataset (fitur_cabai.hapus_latar)."""
    arr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    img, status, alasan = fc.siapkan_citra(arr, load_rembg())
    return img

@st.cache_data
def ambil_metrik():
    """Metrik asli hasil latih_svm.py (bukan angka ketikan)."""
    p = os.path.join(HASIL, "ringkasan_semua.csv")
    if not os.path.exists(p): return None
    d = pd.read_csv(p); d = d[d.skenario == SK]
    return None if d.empty else d.iloc[0]

def ekstrak(img_bgr):
    """CLAHE -> mask + morfologi (opening/closing) -> 12 fitur morfologi + 57 fitur HSV = 69 fitur."""
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
    label_pred = le.inverse_transform([int(np.argmax(proba))])[0]   # murni keluaran SVM, tanpa penimpaan
    return label_pred, proba, le.classes_, img_pre, mask_leaf, mask_nec, fitur

def main():
    with st.sidebar:
        st.markdown("<h2 style='font-size: 26px; font-weight: 800; margin-bottom: 5px; color: #0ea5e9;'>🌶️ ChiliScan</h2>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 13px; color: #94a3b8; margin-bottom: 25px;'>Necrosis Detection System</p>", unsafe_allow_html=True)
        page = st.radio("Navigasi", ["🔍 Deteksi Gambar", "📊 Evaluasi Model", "📖 Panduan Klasifikasi"], label_visibility="collapsed")
        st.write("---")
        st.markdown(f"""
            <div style='padding: 15px; border-radius: 8px; background-color: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);'>
                <p style='margin: 0; font-size: 12px; color: #94a3b8;'>Model Aktif</p>
                <p style='margin: 5px 0 0 0; font-size: 16px; font-weight: 700; color: #f8fafc;'>SVM (RBF) - 69 Fitur</p>
                <p style='margin: 5px 0 0 0; font-size: 13px; color: #38bdf8;'>Skenario {SK}</p>
            </div>
        """, unsafe_allow_html=True)

    model, scaler, le = load_model()
    if model is None:
        st.error(f"Gagal memuat model. Pastikan path `{MODEL_PATH}` benar.")
        return

    st.markdown("<h1 style='color: #1e293b; font-size: 32px; margin-bottom: 5px;'>Dashboard Analisis Daun Cabai</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748b; font-size: 15px; margin-bottom: 25px;'>Sistem cerdas untuk mendeteksi tingkat keparahan nekrosis berbasis AI.</p>", unsafe_allow_html=True)

    if page == "🔍 Deteksi Gambar":
        uploaded_file = st.file_uploader("Upload Gambar (JPG/PNG)", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            img_asli = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            
            col_img, col_res = st.columns([1, 1], gap="large")
            
            with col_img:
                st.markdown("#### 1. Input Gambar")
                st.image(img_asli, caption="Gambar Asli", use_container_width=True)
                with st.spinner("AI sedang menghapus background..."):
                    img_clean_bgr = hapus_background(file_bytes)
                    img_clean_rgb = cv2.cvtColor(img_clean_bgr, cv2.COLOR_BGR2RGB)
                    st.image(img_clean_rgb, caption="Setelah Background Dihapus (AI)", use_container_width=True)

            with col_res:
                st.markdown("#### 2. Hasil Analisis")
                valid, pesan = cek_daun_valid(img_clean_bgr)
                if not valid:
                    st.error(f"❌ **Deteksi Gagal:** {pesan}")
                    st.stop()
                
                with st.spinner("Memprediksi kelas..."):
                    label, proba, classes, img_pre, mask_leaf, mask_nec, fitur = prediksi_gambar(img_clean_bgr, model, scaler, le) 
                    if float(np.max(proba)) < AMBANG_YAKIN:
                        st.warning(f"Keyakinan model rendah ({np.max(proba)*100:.1f}% < {AMBANG_YAKIN*100:.0f}%). Citra mungkin bukan kondisi yang dikenali sistem.")
                        st.stop()
                    colors = {"sehat": "#22c55e", "ringan": "#eab308", "sedang": "#f97316", "ringan_sedang": "#f59e0b", "berat": "#ef4444"}
                    bg_color = colors.get(label, "#0ea5e9")
                    
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, {bg_color}, {bg_color}dd); padding: 25px; border-radius: 12px; text-align: center; color: white; margin-bottom: 20px; box-shadow: 0 10px 15px -3px {bg_color}55;'>
                            <p style='margin: 0; font-size: 14px; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px;'>Prediksi Kelas</p>
                            <h2 style='margin: 5px 0; color: white; font-size: 36px; font-weight: 800;'>{label.upper()}</h2>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    prob_df = pd.DataFrame({"Kelas": classes, "Probabilitas": proba, "Probabilitas (%)": proba * 100})
                    
                    fig_donut = px.pie(prob_df, names="Kelas", values="Probabilitas", hole=0.7, color="Kelas", color_discrete_map=colors)
                    fig_donut.update_traces(textposition='inside', textinfo='percent', hoverinfo='label+percent', marker=dict(line=dict(color='#ffffff', width=2)))
                    fig_donut.update_layout(showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, title=None, font=dict(size=12)), height=250, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_donut, use_container_width=True)
                    
                    fig_bar = px.bar(prob_df, x="Kelas", y="Probabilitas (%)", color="Kelas", color_discrete_map=colors, text_auto='.1f')
                    fig_bar.update_layout(showlegend=False, height=250, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_title=None, yaxis_title="Probabilitas (%)", font=dict(size=12), yaxis=dict(showgrid=True, gridcolor='#f1f5f9'))
                    st.plotly_chart(fig_bar, use_container_width=True)

                st.write("---")
                st.markdown("#### 💡 Rekomendasi Tindakan")
                if label == "sehat":
                    st.success("**Daun Sehat**\n\nDaun berfungsi normal dan tidak ada infeksi berarti. **Tindakan:** Lanjutkan perawatan rutin (penyiraman dan pemupukan). Tidak perlu pestisida.")
                elif label == "ringan":
                    st.warning("**Nekrosis Ringan**\n\nBercak mulai muncul. Fotosintesis mulai sedikit terganggu. **Tindakan:** Segera petik daun yang berbercak agar tidak menular. Aplikasikan pestisida/fungisida organik dosis rendah.")
                elif label == "sedang":
                    st.error("**Nekrosis Sedang**\n\nInfeksi mulai menyebar luas. **Tindakan:** Diperlukan penyemprotan fungisida/bakterisida kimia dengan dosis yang tepat. Pantau tanaman di sekitarnya.")
                elif label == "ringan_sedang":
                    st.warning("**Nekrosis Ringan–Sedang**\n\nBercak sudah muncul. **Tindakan:** Petik daun berbercak dan lakukan penyemprotan fungisida sesuai anjuran. Pantau perkembangan tanaman sekitar.")
                elif label == "berat":
                    st.error("🚨 **Nekrosis Berat**\n\nJaringan daun sudah banyak yang mati dan mengering. **Tindakan:** Pangkas total daun yang sakit atau cabut tanaman. Lakukan penyemprotan masif pada tanaman sehat di sekitarnya.")

                st.write("---")
                with st.expander("🔍 Lihat Detail Ekstraksi Fitur (Masking, Morfologi & HSV)"):
                    dt1, dt2, dt3, dt4 = st.tabs(["Masking & Morfologi", "Pemisahan HSV", "Tabel Morfologi", "Tabel Warna (HSV)"])
                    with dt1:
                        st.markdown("**Hasil Segmentasi Daun dan Nekrosis**")
                        c1, c2 = st.columns(2)
                        with c1: st.image(mask_leaf, caption="Mask Daun (Putih = Area Daun)", use_container_width=True, clamp=True)
                        with c2: st.image(mask_nec, caption="Mask Nekrosis (Putih = Bercak Mati)", use_container_width=True, clamp=True)
                    with dt2:
                        st.markdown("**Pemisahan Channel Warna HSV**")
                        hsv_img = cv2.cvtColor(img_pre, cv2.COLOR_BGR2HSV)
                        h, s, v = cv2.split(hsv_img)
                        c1, c2, c3 = st.columns(3)
                        with c1: st.image(h, caption="Hue (Warna)", use_container_width=True, clamp=True)
                        with c2: st.image(s, caption="Saturation (Kejenuhan)", use_container_width=True, clamp=True)
                        with c3: st.image(v, caption="Value (Kecerahan)", use_container_width=True, clamp=True)
                    with dt3:
                        st.markdown("**Nilai Ekstraksi 12 Fitur Morfologi**")
                        fitur_df = pd.DataFrame({
                            "Fitur Morfologi": ["Area (Luas)", "Perimeter (Keliling)", "Aspect Ratio", "Extent", "Compactness", "Roundness", "Eccentricity", "Solidity", "Major Axis", "Minor Axis", "Eq. Diameter", "Rasio Nekrosis (%)"],
                            "Nilai": [f"{fitur[i]:.4f}" for i in range(11)] + [f"{fitur[11]*100:.2f}%"]
                        })
                        st.dataframe(fitur_df, use_container_width=True, hide_index=True)
                    with dt4:
                        st.markdown("**Nilai Ekstraksi 57 Fitur Warna (HSV): 9 statistik + 48 histogram**")
                        fitur_hsv_df = pd.DataFrame({
                            "Fitur Warna (HSV)": ["H Mean", "H Std", "H Skew", "S Mean", "S Std", "S Skew", "V Mean", "V Std", "V Skew"],
                            "Nilai": [f"{fitur[i]:.2f}" for i in range(12, 21)]
                        })
                        st.dataframe(fitur_hsv_df, use_container_width=True, hide_index=True)
                        hist_df = pd.DataFrame({"Bin": list(range(1, 17)) * 3,
                                                "Kanal": ["H"] * 16 + ["S"] * 16 + ["V"] * 16,
                                                "Proporsi": fitur[21:69]})
                        st.plotly_chart(px.bar(hist_df, x="Bin", y="Proporsi", color="Kanal", barmode="group", height=260), use_container_width=True)
        else:
            st.info("Silakan unggah gambar daun cabai untuk memulai prediksi.")

    elif page == "📊 Evaluasi Model":
        st.markdown("### Evaluasi Performa Model")
        st.markdown("<p style='color: #64748b; font-size: 15px;'>Hasil evaluasi pada data uji skenario 70:30 (split per grup, tanpa citra kembar antara data latih dan uji).</p>", unsafe_allow_html=True)
        
        m = ambil_metrik()
        if m is None:
            st.warning("ringkasan_semua.csv belum ditemukan. Jalankan latih_svm.py terlebih dahulu.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Akurasi Test</div><div class='kpi-value' style='color: #0ea5e9;'>{m.test_acc*100:.2f}%</div></div>", unsafe_allow_html=True)
            with col2: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Macro F1-Score</div><div class='kpi-value'>{m.macro_f1*100:.2f}%</div></div>", unsafe_allow_html=True)
            with col3: st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Weighted F1</div><div class='kpi-value'>{m.weighted_f1*100:.2f}%</div></div>", unsafe_allow_html=True)
            with col4: st.markdown("<div class='kpi-card'><div class='kpi-title'>Kernel SVM</div><div class='kpi-value' style='color: #64748b;'>RBF</div></div>", unsafe_allow_html=True)
        lap = os.path.join(HASIL, f"laporan_{SK}.txt")
        if os.path.exists(lap): st.code(open(lap, encoding="utf8").read())
        st.write("---")
        col_table, col_cm = st.columns([1, 1], gap="large")
        with col_table:
            st.markdown("#### Keterangan")
            st.info("Model menggunakan 69 fitur: 12 fitur morfologi (bentuk dan rasio nekrosis) + 57 fitur warna HSV.")
        with col_cm:
            st.markdown("#### Confusion Matrix (70:30)")
            if os.path.exists(CM_PATH): st.image(CM_PATH, use_container_width=True, caption="Confusion Matrix Model SVM (69 Fitur)")
            else: st.warning("Gambar Confusion Matrix tidak ditemukan.")

    elif page == "📖 Panduan Klasifikasi":
        st.markdown("### Panduan Tingkat Keparahan Nekrosis")
        st.markdown("<p style='color: #64748b; font-size: 15px;'>Alasan ilmiah dan agronomis di balik pembagian kelas pada model SVM.</p>", unsafe_allow_html=True)
        _kotak = {
            "sehat": """<div class="info-box info-sehat"><h4 style="color: #15803d; margin-top:0;">🟢 SEHAT</h4><p><b>Karakteristik Visual:</b> Daun berwarna hijau merata, tidak ada bercak, bentuk daun utuh.</p><p><b>Alasan Agronomis:</b> Daun berfungsi normal untuk fotosintesis. Tidak diperlukan pestisida.</p></div>""",
            "ringan": """<div class="info-box info-ringan"><h4 style="color: #ca8a04; margin-top:0;">🟡 RINGAN</h4><p><b>Karakteristik Visual:</b> Muncul bercak coklat/kuning kecil pada pinggiran atau permukaan daun.</p><p><b>Alasan Agronomis:</b> Tahap awal infeksi. Petani dapat melakukan pencegahan dini sebelum menyebar.</p></div>""",
            "sedang": """<div class="info-box info-sedang"><h4 style="color: #ea580c; margin-top:0;">🟠 SEDANG</h4><p><b>Karakteristik Visual:</b> Bercak menyebar dan menyatu, warna daun berubah kekuningan kusam.</p><p><b>Alasan Agronomis:</b> Fotosintesis terganggu signifikan. Memerlukan penyemprotan fungisida dosis tepat agar panen tidak gagal.</p></div>""",
            "berat": """<div class="info-box info-berat"><h4 style="color: #dc2626; margin-top:0;">🔴 BERAT</h4><p><b>Karakteristik Visual:</b> Jaringan daun mati (kering, coklat/hitam). Daun menggulung dan bentuknya rusak parah.</p><p><b>Alasan Agronomis:</b> Tindakan darurat (pemangkasan/pencabutan) diperlukan untuk mencegah penularan masif.</p></div>""",
            "ringan_sedang": """<div class="info-box info-sedang"><h4 style="color: #ea580c; margin-top:0;">🟠 RINGAN–SEDANG</h4><p><b>Karakteristik Visual:</b> Bercak coklat/kuning kecil sampai menyebar dan mulai menyatu pada permukaan daun.</p><p><b>Alasan Agronomis:</b> Infeksi sudah ada tetapi daun masih berfungsi. Petik daun berbercak, lakukan penyemprotan sesuai anjuran, dan pantau perkembangannya.</p></div>""",
        }
        _urut = ["sehat", "ringan_sedang", "berat"] if MODE == "kelas3" else ["sehat", "ringan", "sedang", "berat"]
        st.markdown("".join(_kotak[k] for k in _urut), unsafe_allow_html=True)
        st.write("---")
        st.info("**Bagaimana kelas ditentukan?**\n\nKelas ditentukan penulis lewat penilaian visual (jumlah, ukuran, dan sebaran bercak). Rasio nekrosis ditampilkan sebagai informasi pendukung dan menjadi salah satu fitur SVM, tetapi bukan penentu kelas: pada data penelitian ini rasio saja hanya mencapai sekitar separuh akurasi, terutama karena kelas Ringan dan Sedang banyak tumpang tindih. Kelas yang bersebelahan (terutama Ringan dan Sedang) lebih rawan tertukar.")

if __name__ == "__main__":
    main()