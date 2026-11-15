import importlib.util
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Öğrenci Başarı Tahmini", page_icon="🎓", layout="centered")

# --- TEŞHİS: gerekli paketler kurulu mu? Değilse nedenini ekranda göster ---
_eksik = [m for m in ("joblib", "sklearn", "numpy", "pandas") if importlib.util.find_spec(m) is None]
if _eksik:
    _klasor = Path(__file__).parent
    _req = _klasor / "requirements.txt"
    st.error(f"Eksik paketler: {_eksik}")
    st.write("Python sürümü:", sys.version)
    st.write("Klasördeki dosyalar:", sorted(p.name for p in _klasor.iterdir()))
    st.code(_req.read_text(encoding="utf-8", errors="replace") if _req.exists() else "requirements.txt BU KLASÖRDE YOK")
    st.stop()

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import BaseEstimator, TransformerMixin  # noqa: E402

MODEL_YOLU = Path(__file__).parent / "ogrenci_basari_production_model.pkl"


# --- Özel sınıf: pkl bu sınıfı '__main__.StudentFeatureEngineer' olarak arıyor ---
# joblib.load'dan ÖNCE tanımlanmalı. Colab'daki sınıfla birebir aynı olmalı.
class StudentFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        d = X.copy()
        d['ortalama_akademik_puan'] = d[['vize_puani', 'quiz_puani', 'odev_puani']].mean(axis=1, skipna=True)
        d['calisma_uyku_orani'] = d['gunluk_calisma_saati'] / (d['gunluk_uyku_saati'] + 1)
        d['basari_devam_skoru'] = d['onceki_ortalama'] * (d['devam_yuzdesi'] / 100)
        d['toplam_zaman_yuku'] = d[['gunluk_calisma_saati', 'ders_disi_aktivite_saati']].sum(axis=1, skipna=True)
        d['akademik_risk_skoru'] = d['alttan_ders_sayisi'] / d['onceki_ortalama'].replace(0, 0.1)
        return d.replace([np.inf, -np.inf], np.nan)


@st.cache_resource
def model_yukle():
    paket = joblib.load(MODEL_YOLU)
    return paket['model'], paket['threshold'], paket.get('model_name', 'Model'), paket['feature_names']


# ============================================================
# GÖRÜNÜM (CSS)
# ============================================================
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.8rem; padding-bottom: 3rem; max-width: 960px;}

.hero {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 55%, #EC4899 130%);
    color: #fff; padding: 30px 34px; border-radius: 22px; margin-bottom: 1.4rem;
    box-shadow: 0 12px 32px rgba(79, 70, 229, .28);
}
.hero h1 {margin: 0; font-size: 2rem; font-weight: 800; color: #fff; padding: 0;}
.hero p {margin: .45rem 0 0 0; opacity: .92; font-size: 1.02rem;}
.hero .rozet {
    display: inline-block; margin-top: .9rem; padding: .25rem .8rem; border-radius: 999px;
    background: rgba(255,255,255,.18); font-size: .8rem; font-weight: 600; letter-spacing: .02em;
}

.bolum {font-size: 1.02rem; font-weight: 700; color: #4F46E5; margin: 0 0 .4rem 0;}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 18px !important; background: #FFFFFF;
    box-shadow: 0 2px 14px rgba(31, 41, 55, .06);
}

.stButton > button {
    width: 100%; border: 0; border-radius: 14px; padding: .8rem 1rem;
    font-size: 1.05rem; font-weight: 700; color: #fff;
    background: linear-gradient(135deg, #4F46E5, #7C3AED);
    box-shadow: 0 8px 20px rgba(79, 70, 229, .30); transition: all .15s ease;
}
.stButton > button:hover {transform: translateY(-2px); box-shadow: 0 12px 26px rgba(79, 70, 229, .38); color: #fff;}
.stButton > button:active {transform: translateY(0);}

.sonuc {border-radius: 22px; padding: 26px 30px; margin: 1.2rem 0 .8rem 0; color: #fff;}
.sonuc.dusuk {background: linear-gradient(135deg, #059669, #10B981); box-shadow: 0 12px 30px rgba(16,185,129,.30);}
.sonuc.sinirda {background: linear-gradient(135deg, #D97706, #F59E0B); box-shadow: 0 12px 30px rgba(245,158,11,.30);}
.sonuc.yuksek {background: linear-gradient(135deg, #DC2626, #F43F5E); box-shadow: 0 12px 30px rgba(244,63,94,.30);}
.sonuc .etiket {font-size: 1.05rem; font-weight: 700; opacity: .95;}
.sonuc .yuzde {font-size: 3.4rem; font-weight: 800; line-height: 1.1; margin: .2rem 0;}
.sonuc .alt {font-size: .92rem; opacity: .92;}
.bar {position: relative; height: 12px; background: rgba(255,255,255,.28); border-radius: 999px; margin-top: 1.1rem;}
.bar .dolgu {height: 100%; background: #fff; border-radius: 999px;}
.bar .isaret {position: absolute; top: -6px; width: 3px; height: 24px; background: rgba(0,0,0,.55); border-radius: 2px;}
.bar-not {display: flex; justify-content: space-between; font-size: .75rem; opacity: .85; margin-top: .45rem;}

.ipucu {padding: .65rem .9rem; border-radius: 12px; background: #EEF2FF; color: #3730A3; margin: .35rem 0; font-size: .95rem;}
.kucuk {color: #6B7280; font-size: .85rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🎓 Öğrenci Başarı Tahmini</h1>
  <p>Öğrencinin bilgilerini girin; model kalma riskini saniyeler içinde hesaplasın.</p>
  <span class="rozet">Makine öğrenmesi destekli karar aracı</span>
</div>
""", unsafe_allow_html=True)

# ============================================================
# MODEL
# ============================================================
if not MODEL_YOLU.exists():
    st.error(f"Model dosyası bulunamadı: {MODEL_YOLU.name}")
    st.write("Bu klasördeki dosyalar:", sorted(p.name for p in MODEL_YOLU.parent.iterdir()))
    st.stop()

try:
    model, threshold, model_name, kolonlar = model_yukle()
except Exception as e:
    st.error(f"⚠️ Model yüklenirken hata oluştu: {type(e).__name__}: {e}")
    st.info("Çoğunlukla scikit-learn sürüm uyuşmazlığıdır: requirements.txt içinde scikit-learn==1.6.1 olmalı.")
    st.stop()

# ============================================================
# FORM (kategori seçenekleri eğitimdekiyle birebir aynı olmalı)
# ============================================================
sol, sag = st.columns(2, gap="medium")

with sol:
    with st.container(border=True):
        st.markdown('<p class="bolum">📚 Akademik performans</p>', unsafe_allow_html=True)
        vize_puani = st.slider("Vize puanı", 0.0, 100.0, 50.0, 1.0)
        quiz_puani = st.slider("Quiz puanı", 0.0, 100.0, 50.0, 1.0)
        odev_puani = st.slider("Ödev puanı", 0.0, 100.0, 60.0, 1.0)
        onceki_ortalama = st.slider("Önceki dönem ortalaması", 0.0, 4.0, 2.5, 0.05)
        alttan_ders_sayisi = st.number_input("Alttan alınan ders sayısı", 0, 10, 0, 1)

with sag:
    with st.container(border=True):
        st.markdown('<p class="bolum">⏰ Günlük alışkanlıklar</p>', unsafe_allow_html=True)
        gunluk_calisma_saati = st.slider("Günlük çalışma (saat)", 0.0, 12.0, 3.0, 0.5)
        gunluk_uyku_saati = st.slider("Günlük uyku (saat)", 0.0, 14.0, 7.0, 0.5)
        devam_yuzdesi = st.slider("Derse devam (%)", 0, 100, 80)
        ders_disi_aktivite_saati = st.slider("Ders dışı aktivite (saat)", 0.0, 10.0, 1.0, 0.5)

with st.container(border=True):
    st.markdown('<p class="bolum">👤 Kişisel bilgiler</p>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    yas = k1.number_input("Yaş", 15, 60, 20, 1)
    cinsiyet = k2.selectbox("Cinsiyet", ["Kadın", "Erkek", "Diğer"])
    internet_erisimi = k3.selectbox("İnternet erişimi", ["Evet", "Hayır"])
    ebeveyn_egitim_durumu = k4.selectbox(
        "Ebeveyn eğitimi", ["Lise", "Diploma", "Lisans", "Yüksek Lisans", "Doktora"])

st.write("")
tahmin_et = st.button("🔮 Tahmin et")

# ============================================================
# SONUÇ
# ============================================================
if tahmin_et:
    girdi = {
        'yas': yas, 'cinsiyet': cinsiyet,
        'gunluk_calisma_saati': gunluk_calisma_saati, 'devam_yuzdesi': devam_yuzdesi,
        'gunluk_uyku_saati': gunluk_uyku_saati, 'odev_puani': odev_puani,
        'quiz_puani': quiz_puani, 'vize_puani': vize_puani,
        'onceki_ortalama': onceki_ortalama, 'alttan_ders_sayisi': alttan_ders_sayisi,
        'internet_erisimi': internet_erisimi,
        'ders_disi_aktivite_saati': ders_disi_aktivite_saati,
        'ebeveyn_egitim_durumu': ebeveyn_egitim_durumu,
    }
    df_yeni = pd.DataFrame([girdi])[kolonlar]  # sütun sırası eğitimdekiyle aynı
    proba = float(model.predict_proba(df_yeni)[:, 1][0])

    # Seviye: eşiğin altı düşük/sınırda, eşik ve üstü yüksek
    if proba >= threshold:
        sinif, emoji, baslik = "yuksek", "🚨", "Yüksek kalma riski"
    elif proba >= threshold * 0.5:
        sinif, emoji, baslik = "sinirda", "⚠️", "Sınırda: yakından takip edilmeli"
    else:
        sinif, emoji, baslik = "dusuk", "✅", "Düşük risk: geçmesi bekleniyor"

    yuzde = min(max(proba * 100, 0), 100)
    esik_yuzde = min(max(threshold * 100, 0), 100)

    st.markdown(f"""
    <div class="sonuc {sinif}">
      <div class="etiket">{emoji} {baslik}</div>
      <div class="yuzde">%{yuzde:.1f}</div>
      <div class="alt">Kalma risk skoru · Karar eşiği %{esik_yuzde:.1f} (siyah çizgi)</div>
      <div class="bar"><div class="dolgu" style="width:{yuzde:.1f}%"></div><div class="isaret" style="left:{esik_yuzde:.1f}%"></div></div>
      <div class="bar-not"><span>0%</span><span>100%</span></div>
    </div>
    """, unsafe_allow_html=True)

    ort_puan = (vize_puani + quiz_puani + odev_puani) / 3
    m1, m2, m3 = st.columns(3)
    m1.metric("Ortalama akademik puan", f"{ort_puan:.1f}")
    m2.metric("Çalışma / uyku oranı", f"{gunluk_calisma_saati / (gunluk_uyku_saati + 1):.2f}")
    m3.metric("Derse devam", f"%{devam_yuzdesi}")

    ipuclari = []
    if devam_yuzdesi < 75:
        ipuclari.append("Devam oranı düşük görünüyor.")
    if gunluk_calisma_saati < 2:
        ipuclari.append("Günlük çalışma süresi düşük.")
    if gunluk_uyku_saati < 6:
        ipuclari.append("Uyku süresi 6 saatin altında.")
    if alttan_ders_sayisi > 0:
        ipuclari.append(f"{alttan_ders_sayisi} alttan ders var.")
    if ort_puan < 50:
        ipuclari.append("Ara sınav, quiz ve ödev ortalaması 50'nin altında.")
    if ipuclari:
        st.markdown("**Dikkat çeken noktalar**")
        for ip in ipuclari:
            st.markdown(f'<div class="ipucu">• {ip}</div>', unsafe_allow_html=True)

    st.markdown(
        f'<p class="kucuk">Model: {model_name}. Bu sonuç bir karar destek aracıdır, kesin hüküm değildir; '
        f'"dikkat çeken noktalar" girilen değerlere dayalı basit kontrollerdir, modelin açıklaması değildir.</p>',
        unsafe_allow_html=True)
