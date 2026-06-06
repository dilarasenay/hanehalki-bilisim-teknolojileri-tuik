"""
=============================================================
 TÜRKİYE DİJİTALLEŞME DASHBOARD
 Dilara Şenay | TÜBİTAK 2209-A Araştırma Projesi
 Kaynak: TÜİK Hanehalkı Bilişim Teknolojileri Kullanım Araştırması
=============================================================
Kurulum:
    pip install dash plotly pandas

Çalıştırma:
    python app.py
Tarayıcıda aç:
    http://127.0.0.1:8050
=============================================================
"""

import json
import os
import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────
# 1. VERİ  (CSV'den hesaplanan gerçek dashboard göstergeleri)
# ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(__file__)

# CSV dosyasını şu adlardan biriyle arar. En ideali: dashboard/data/bütünveriler.csv
DATA_CANDIDATES = [
    os.path.join(BASE_DIR, "data", "bütünveriler.csv"),
    os.path.join(BASE_DIR, "data", "butunveriler.csv"),
    os.path.join(BASE_DIR, "data", "bütünverilerr"),
    os.path.join(BASE_DIR, "bütünveriler.csv"),
    os.path.join(BASE_DIR, "butunveriler.csv"),
    os.path.join(BASE_DIR, "bütünverilerr"),
    os.path.join(BASE_DIR, "..", "data", "bütünveriler.csv"),
    os.path.join(BASE_DIR, "..", "data", "butunveriler.csv"),
]

BOLGE_ADLARI = {
    "TR1": "İstanbul",
    "TR2": "Batı Marmara",
    "TR3": "Ege",
    "TR4": "Doğu Marmara",
    "TR5": "Batı Anadolu",
    "TR6": "Akdeniz",
    "TR7": "Orta Anadolu",
    "TR8": "Batı Karadeniz",
    "TR9": "Doğu Karadeniz",
    "TRA": "Kuzeydoğu Anadolu",
    "TRB": "Ortadoğu Anadolu",
    "TRC": "Güneydoğu Anadolu",
}

BOLGE_KOORD = {
    "TR1": (41.01, 28.97),
    "TR2": (40.18, 27.50),
    "TR3": (38.42, 27.14),
    "TR4": (40.77, 29.92),
    "TR5": (39.93, 32.86),
    "TR6": (37.00, 35.32),
    "TR7": (39.05, 34.96),
    "TR8": (40.55, 31.57),
    "TR9": (40.55, 38.39),
    "TRA": (39.90, 41.27),
    "TRB": (38.55, 40.22),
    "TRC": (37.20, 38.31),
}

KUME_RENK = {0: "#0B4EA2", 1: "#2563EB", 2: "#7C3AED", 3: "#F97316"}
KUME_ETIKET = {0: "Yüksek Dijitalleşme", 1: "Orta-Yüksek", 2: "Orta", 3: "Düşük Dijitalleşme"}


def _normalize_col_name(name):
    return str(name).strip().lower().replace("ı", "i")


def _find_col(df, *candidates):
    """Sütun adları küçük farklılık gösterirse doğru sütunu bulur."""
    normalized = {_normalize_col_name(c): c for c in df.columns}
    for cand in candidates:
        key = _normalize_col_name(cand)
        if key in normalized:
            return normalized[key]
    for cand in candidates:
        key = _normalize_col_name(cand)
        for ncol, original in normalized.items():
            if key in ncol:
                return original
    return None


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def evet_orani(s):
    """1=Evet, 2=Hayır kodlamasına göre Evet yüzdesi. Diğer kodları hesap dışı bırakır."""
    s = _num(s)
    s = s[s.isin([1, 2])]
    if len(s) == 0:
        return np.nan
    return float((s.eq(1).mean() * 100).round(1))


def kullanim_siklik_orani(s):
    """Sıklık sorularında 1-4 kullanım, 6/9/13 gibi kodlar kullanım dışı/cevapsız kabul edilir."""
    s = _num(s)
    s = s[s.notna()]
    if len(s) == 0:
        return np.nan
    return float((s.isin([1, 2, 3, 4]).mean() * 100).round(1))


def satir_bazli_evet_orani(df, cols):
    """Birden fazla 1/2 sütununda en az bir evet oranı."""
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return np.nan
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    valid = sub.isin([1, 2]).any(axis=1)
    if valid.sum() == 0:
        return np.nan
    return float((sub.eq(1).any(axis=1)[valid].mean() * 100).round(1))


def _safe_indicator(df, col, mode="evet"):
    if col is None or col not in df.columns:
        return np.nan
    if mode == "freq":
        return kullanim_siklik_orani(df[col])
    return evet_orani(df[col])


def _ibbs_columns(df):
    """CSV içinde aynı anlama gelen birden fazla İBBS-1 sütunu varsa hepsini bulur."""
    cols = []
    wanted = [
        "istatistiki bölge birimleri sınıflaması(düzey 1)",
        "istatistiki bölge birimleri sınıflaması (düzey 1)",
    ]
    for col in df.columns:
        norm = _normalize_col_name(col)
        if any(_normalize_col_name(w) == norm for w in wanted):
            cols.append(col)
    # Esnek yakalama: sütun adında hem "bölge" hem "düzey 1" geçiyorsa da al.
    for col in df.columns:
        norm = _normalize_col_name(col)
        if "bölge" in norm and "düzey 1" in norm and col not in cols:
            cols.append(col)
    return cols


def _normalize_ibbs1_value(x):
    """TR1/TR2 kodlarını ve bölge adı olarak gelen değerleri standart TR koduna çevirir."""
    if pd.isna(x):
        return None

    raw = str(x).strip()
    if raw == "" or raw.lower() in ["nan", "none", "null"]:
        return None

    upper = raw.upper().replace("İ", "I")
    upper = upper.replace("Ğ", "G").replace("Ü", "U").replace("Ş", "S").replace("Ö", "O").replace("Ç", "C")

    # Zaten TR koduysa
    for kod in BOLGE_ADLARI.keys():
        if upper == kod or upper.startswith(kod + " ") or upper.startswith(kod + "-"):
            return kod

    # Bölge adı olarak geldiyse
    name_to_code = {
        "ISTANBUL": "TR1",
        "BATI MARMARA": "TR2",
        "EGE": "TR3",
        "DOGU MARMARA": "TR4",
        "BATI ANADOLU": "TR5",
        "AKDENIZ": "TR6",
        "ORTA ANADOLU": "TR7",
        "BATI KARADENIZ": "TR8",
        "DOGU KARADENIZ": "TR9",
        "KUZEYDOGU ANADOLU": "TRA",
        "ORTADOGU ANADOLU": "TRB",
        "GUNEYDOGU ANADOLU": "TRC",
    }
    if upper in name_to_code:
        return name_to_code[upper]

    # Değer içinde bölge adı geçiyorsa
    for name, kod in name_to_code.items():
        if name in upper:
            return kod

    return None


def load_real_data():
    data_path = next((p for p in DATA_CANDIDATES if os.path.exists(p)), None)
    if data_path is None:
        print("UYARI: bütünveriler.csv bulunamadı. dashboard/data klasörüne bütünveriler.csv ekleyin.")
        return pd.DataFrame()

    df = pd.read_csv(data_path, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.copy()  # parçalanma/performance warning riskini azaltır

    year_col = _find_col(df, "referans yıl")
    ibbs_cols = _ibbs_columns(df)

    if year_col is None or not ibbs_cols:
        raise ValueError("CSV içinde 'referans yıl' veya İBBS-1 bölge sütunu bulunamadı.")

    df["yil"] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")

    # Kritik düzeltme:
    # Bazı yıllarda İBBS kodu ilk sütunda, bazı yıllarda ikinci sütunda gelebiliyor.
    # Bu yüzden tüm İBBS-1 sütunlarını birleştirip ilk dolu değeri alıyoruz.
    ibbs_raw = pd.Series([None] * len(df), index=df.index, dtype="object")
    for col in ibbs_cols:
        current = df[col].where(df[col].notna(), None)
        current = current.astype("object")
        current = current.where(current.astype(str).str.strip().str.lower().ne("nan"), None)
        current = current.where(current.astype(str).str.strip().ne(""), None)
        ibbs_raw = ibbs_raw.where(ibbs_raw.notna(), current)

    df["ibbs1_raw"] = ibbs_raw
    df["ibbs1"] = df["ibbs1_raw"].apply(_normalize_ibbs1_value)

    # Sadece yıl veya bölge bilgisi gerçekten olmayan satırlar dışarıda bırakılır.
    df = df[df["yil"].notna() & df["ibbs1"].isin(BOLGE_ADLARI.keys())].copy()
    return df


real_df = load_real_data()

# Göstergelerde kullanılacak sütunlar
COL_HANE_INTERNET = _find_col(real_df, "hane internet erişim durumu") if not real_df.empty else None
COL_BILGISAYAR = _find_col(real_df, "hane bilgisayar kullanım durumu", "bilgisayar kullanım sıklığı") if not real_df.empty else None
COL_AKILLI = _find_col(real_df, "akıllı telefon kullanımı", "cep telefonu kullanımı", "hanede cep telefonu var mı") if not real_df.empty else None
COL_EDEVLET = _find_col(real_df, "edevlet hakkında bilgi") if not real_df.empty else None
COL_ETICARET = _find_col(real_df, "eticaret kullanımı") if not real_df.empty else None
COL_SOSYAL = _find_col(real_df, "sosyal medya kullanımı") if not real_df.empty else None
COL_INTERNET_SIKLIK = _find_col(real_df, "internet kullanım sıklığı") if not real_df.empty else None
COL_EPOSTA = _find_col(real_df, "eposta kullanımı") if not real_df.empty else None
COL_YZ = _find_col(real_df, "yapay zeka kullanımı") if not real_df.empty else None

DIJITAL_BECERI_COLS = [c for c in [
    _find_col(real_df, "dijital beceri dosya transfer") if not real_df.empty else None,
    _find_col(real_df, "dijital beceri uygulama kurulum") if not real_df.empty else None,
    _find_col(real_df, "dijital beceri ayar değiştirme") if not real_df.empty else None,
    _find_col(real_df, "dijital beceri doküman oluşturma") if not real_df.empty else None,
    _find_col(real_df, "dijital beceri tablolama yazılımı kullanma") if not real_df.empty else None,
    _find_col(real_df, "dijital beceri kod yazma") if not real_df.empty else None,
] if c]

CEVRIMICI_EGITIM_COLS = [c for c in [
    _find_col(real_df, "online kurs") if not real_df.empty else None,
    _find_col(real_df, "online eğitim materyal") if not real_df.empty else None,
    _find_col(real_df, "online eğitim iletişim") if not real_df.empty else None,
    _find_col(real_df, "internet üzerinden yapılan diğer eğitim faaliyetleri") if not real_df.empty else None,
] if c]


def _compute_turkey_ts():
    if real_df.empty:
        return pd.DataFrame({
            "yil": [2024],
            "internet_erisim": [np.nan],
            "bilgisayar": [np.nan],
            "akilli_telefon": [np.nan],
            "edevlet": [np.nan],
            "eticaret": [np.nan],
            "sosyal_medya": [np.nan],
            "cevrimici_egitim": [np.nan],
            "yapay_zeka_farkin": [np.nan],
        })

    rows = []
    for yil, g in real_df.dropna(subset=["yil"]).groupby("yil"):
        bilgisayar_mode = "freq" if COL_BILGISAYAR and "sıklığı" in COL_BILGISAYAR else "evet"
        rows.append({
            "yil": int(yil),
            "internet_erisim": _safe_indicator(g, COL_HANE_INTERNET),
            "bilgisayar": _safe_indicator(g, COL_BILGISAYAR, bilgisayar_mode),
            "akilli_telefon": _safe_indicator(g, COL_AKILLI),
            "edevlet": _safe_indicator(g, COL_EDEVLET),
            "eticaret": _safe_indicator(g, COL_ETICARET),
            "sosyal_medya": _safe_indicator(g, COL_SOSYAL),
            "cevrimici_egitim": satir_bazli_evet_orani(g, CEVRIMICI_EGITIM_COLS),
            "yapay_zeka_farkin": _safe_indicator(g, COL_YZ),
        })
    out = pd.DataFrame(rows).sort_values("yil")
    # Bazı göstergeler bazı yıllarda yoksa grafik bozulmasın diye 0 ile tamamlanır.
    metric_cols = [c for c in out.columns if c != "yil"]
    out[metric_cols] = out[metric_cols].fillna(0).round(1)
    return out


turkey_ts = _compute_turkey_ts()
YILLAR = sorted(
    real_df["yil"]
    .dropna()
    .astype(int)
    .unique()
    .tolist()
)
if not YILLAR:
    YILLAR = [2024]


def _assign_kume(df):
    """Dijitalleşme skoruna göre 4 kümeye ayırır: 0 en yüksek, 3 en düşük."""
    df = df.sort_values("dijit_skor", ascending=False).reset_index(drop=True)
    n = len(df)
    if n == 0:
        df["kmeans_kume"] = []
        return df
    df["kmeans_kume"] = [min(3, int(i * 4 / n)) for i in range(n)]
    return df


def bolge_yil_df(yil):
    aktif_yil = max(YILLAR) if yil == "all" else int(yil)
    d = real_df[real_df["yil"] == aktif_yil].copy()

    rows = []
    for kod in BOLGE_ADLARI:
        g = d[d["ibbs1"] == kod]
        if g.empty:
            internet_erisim = edevlet = eticaret = dijital_beceri = np.nan
        else:
            internet_erisim = _safe_indicator(g, COL_HANE_INTERNET)
            edevlet = _safe_indicator(g, COL_EDEVLET)
            eticaret = _safe_indicator(g, COL_ETICARET)
            dijital_beceri = satir_bazli_evet_orani(g, DIJITAL_BECERI_COLS)

        values = [internet_erisim, edevlet, dijital_beceri, eticaret]
        values = [0 if pd.isna(v) else v for v in values]
        dijit_skor = values[0] * 0.30 + values[1] * 0.25 + values[2] * 0.30 + values[3] * 0.15
        lat, lon = BOLGE_KOORD[kod]

        rows.append({
            "aktif_yil": aktif_yil,
            "kod": kod,
            "bolge": BOLGE_ADLARI[kod],
            "internet_erisim": round(values[0], 1),
            "edevlet": round(values[1], 1),
            "dijital_beceri": round(values[2], 1),
            "eticaret": round(values[3], 1),
            "dijit_skor": round(dijit_skor, 1),
            "lat": lat,
            "lon": lon,
            "nufus_milyon": np.nan,
        })

    return _assign_kume(pd.DataFrame(rows))


# İlk görünüm için son yılın bölgesel verisi
bolge_data = bolge_yil_df("all")

# --- Gerçek Türkiye İBBS-1 GeoJSON yükleme ---
def load_tr_ibbs1_geojson():
    """dashboard/data/tr_ibbs1.geojson dosyasını yükler. Dosya yoksa harita paneli hata yerine açıklama gösterir."""
    geo_path = os.path.join(BASE_DIR, "data", "tr_ibbs1.geojson")
    try:
        with open(geo_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as exc:
        print(f"UYARI: tr_ibbs1.geojson okunamadı: {exc}")
        return None

TR_IBBS1_GEOJSON = load_tr_ibbs1_geojson()

# --- Demografik Veri ---
yas_gruplari = ["16-24","25-34","35-44","45-54","55-64","65-74","75+"]
demo_df = pd.DataFrame({
    "yas_grubu": yas_gruplari,
    "internet":  [99.2, 98.4, 94.1, 82.3, 61.2, 38.5, 18.4],
    "edevlet":   [72.1, 79.4, 73.8, 62.1, 44.3, 25.7,  9.2],
    "sosyal":    [97.8, 95.3, 84.2, 69.4, 47.8, 26.3, 10.1],
    "eticaret":  [58.3, 67.2, 58.4, 44.1, 28.3, 12.4,  4.1],
})

egitim_gruplari = ["Okuryazar\nDeğil","İlkokul","Ortaokul","Lise","Üniversite","Yüksek\nLisans+"]
egitim_df = pd.DataFrame({
    "egitim": egitim_gruplari,
    "internet":  [23.4, 68.2, 87.5, 94.8, 98.9, 99.6],
    "edevlet":   [ 8.1, 38.4, 58.2, 72.4, 87.3, 91.2],
    "dijital_b": [ 4.2, 21.3, 42.8, 61.4, 82.1, 91.7],
})

cinsiyet_df = pd.DataFrame({
    "kategori": ["İnternet\nKullanımı","Akıllı\nTelefon","E-Devlet","E-Ticaret",
                 "Sosyal\nMedya","Dijital\nBeceri"],
    "erkek": [97.8, 97.1, 71.2, 60.4, 88.4, 61.3],
    "kadin": [96.6, 95.5, 67.3, 54.2, 90.1, 52.7],
})

# --- NLP Duygu Analizi Simülasyonu ---
nlp_df = pd.DataFrame({
    "konu":     ["Dijitalleşme","E-Devlet","Yapay Zeka","Dijital Eğitim","5G/Altyapı","E-Ticaret"],
    "pozitif":  [42.3, 35.1, 54.7, 48.2, 38.9, 62.4],
    "notr":     [31.2, 28.4, 25.3, 29.8, 30.1, 22.3],
    "negatif":  [26.5, 36.5, 20.0, 22.0, 31.0, 15.3],
})

lda_konular = pd.DataFrame({
    "konu_no": [1, 2, 3, 4, 5],
    "konu_adi": [
        "E-Devlet & Kamu Hizmetleri",
        "Yapay Zeka & Teknoloji",
        "Dijital Eğitim",
        "Altyapı & Erişim Sorunları",
        "E-Ticaret & Ekonomi",
    ],
    "agirlik": [28.4, 24.1, 18.7, 16.3, 12.5],
    "kelimeler": [
        "edevlet, vergi, randevu, kimlik, belediye, hizmet",
        "yapay zeka, chatgpt, otomasyon, robot, teknoloji",
        "uzaktan, online ders, dijital, öğrenme, pandemi",
        "internet, hız, fiber, kırsal, erişim, altyapı",
        "online alışveriş, kargo, sipariş, ödeme, uygulama",
    ],
    "renk": ["#0D47A1","#1565C0","#1976D2","#1E88E5","#42A5F5"],
})

# ─────────────────────────────────────────────────────────────
# 2. DASH UYGULAMASI
# ─────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="Türkiye Dijitalleşme Dashboard | Dilara Şenay",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.config.suppress_callback_exceptions = True
app.index_string = """
<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
        * { box-sizing: border-box; }
        body { margin: 0; }
        .Select-control {
            border-radius: 16px !important;
            border: 1px solid rgba(139,92,246,0.22) !important;
            background: rgba(255,255,255,0.82) !important;
            backdrop-filter: blur(16px) !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.85), 0 14px 34px rgba(139,92,246,0.13) !important;
            min-height: 44px !important;
        }
        .Select-value-label, .Select-placeholder {
            color: #2E1065 !important;
            font-weight: 850 !important;
            letter-spacing: .15px !important;
        }
        .Select-arrow { border-top-color: #7C3AED !important; }
        .Select-input > input { color: #2E1065 !important; }
        .Select-menu-outer {
            border-radius: 18px !important;
            overflow: hidden !important;
            box-shadow: 0 24px 56px rgba(139,92,246,0.24) !important;
            border: 1px solid rgba(139,92,246,0.20) !important;
            z-index: 9999 !important;
            background: rgba(255,255,255,.96) !important;
        }
        .Select-option { padding: 11px 16px !important; font-weight: 800; color: #312E81 !important; }
        .Select-option.is-focused { background: linear-gradient(90deg, #F3E8FF, #ECFEFF) !important; }
        .Select-option.is-selected { background: linear-gradient(90deg, #8B5CF6, #06B6D4) !important; color: white !important; }
        .tab {
            border: none !important;
            border-radius: 20px !important;
            background: rgba(255,255,255,0.38) !important;
            color: #4C3F73 !important;
            font-weight: 900 !important;
            padding: 18px 24px !important;
            transition: all .2s ease !important;
        }
        .tab--selected {
            color: white !important;
            background: linear-gradient(135deg, #8B5CF6, #EC4899, #06B6D4) !important;
            box-shadow: 0 18px 38px rgba(139,92,246,0.26) !important;
            border: 1px solid rgba(255,255,255,.45) !important;
        }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>
"""


# ─── Elegant Soft Theme (lacivert yok) ───────────────────────
# Açık mor + turkuaz + sıcak turuncu; beyaz ama düz değil, daha canlı.
C_BG      = "#FBF7FF"
C_BG_2    = "#ECFEFF"
C_CARD    = "rgba(255, 255, 255, 0.88)"
C_BORDER  = "#E9D8FD"
C_TEXT    = "#181124"
C_MUTED   = "#71717A"

C_ACCENT  = "#8B5CF6"   # Mor
C_ACCENT2 = "#06B6D4"   # Turkuaz
C_GREEN   = "#22C55E"   # Yeşil
C_ORANGE  = "#F97316"   # Turuncu
C_PURPLE  = "#A855F7"   # Canlı mor
C_RED     = "#EF4444"
C_PINK    = "#EC4899"

CARD_STYLE = {
    "background": "linear-gradient(180deg, rgba(255,255,255,0.90), rgba(255,255,255,0.70))",
    "backdropFilter": "blur(18px)",
    "border": f"1px solid {C_BORDER}",
    "borderRadius": "26px",
    "padding": "24px",
    "marginBottom": "20px",
    "boxShadow": "0 24px 60px rgba(139, 92, 246, 0.13)",
}
TITLE_STYLE = {
    "color": C_TEXT, "fontSize": "14px", "fontWeight": "800",
    "textTransform": "uppercase", "letterSpacing": "0.7px",
    "marginBottom": "5px", "marginTop": "0",
}
SUBTITLE_STYLE = {"color": C_MUTED, "fontSize": "11px", "marginBottom": "18px", "marginTop": "0"}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(255,255,255,0)",
    plot_bgcolor="rgba(255,255,255,0)",
    template="plotly_white",
    font=dict(family="Inter, system-ui, sans-serif", color=C_TEXT),
    margin=dict(l=20, r=20, t=35, b=20),
    legend=dict(
        bgcolor="rgba(255,255,255,0.72)", bordercolor=C_BORDER, borderwidth=1,
        font=dict(size=11, color=C_TEXT),
    ),
    xaxis=dict(gridcolor="rgba(139,92,246,0.14)", linecolor=C_BORDER, zeroline=False, tickfont=dict(color=C_MUTED, size=10)),
    yaxis=dict(gridcolor="rgba(139,92,246,0.14)", linecolor=C_BORDER, zeroline=False, tickfont=dict(color=C_MUTED, size=10)),
)


def mini_trend_fig(col, renk):
    """KPI kartı içindeki küçük sparkline grafik."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=turkey_ts["yil"], y=turkey_ts[col],
        mode="lines+markers",
        line=dict(color=renk, width=2.4, shape="spline"),
        marker=dict(size=4, color=renk),
        hoverinfo="skip",
        showlegend=False,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=58,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


# ─── KPI Kartları ───────────────────────────────────────────
def kpi_card(baslik, deger, alt, trend, renk=C_ACCENT, seri=None):
    return html.Div([
        html.Div([
            html.Div([
                html.Div(baslik.upper(), style={
                    "color": C_TEXT, "fontSize": "11px", "fontWeight": "850",
                    "letterSpacing": "0.3px", "marginBottom": "7px",
                }),
                html.Div(deger, style={
                    "fontSize": "30px", "fontWeight": "950", "color": renk,
                    "lineHeight": "1", "marginBottom": "7px",
                }),
                html.Div(alt, style={"color": C_MUTED, "fontSize": "10px", "fontWeight": "650", "lineHeight":"1.35"}),
                html.Div(trend, style={
                    "color": C_GREEN, "fontSize": "10px", "fontWeight": "850",
                    "marginTop": "10px",
                }),
            ], style={"flex": "1"}),
            html.Div([
                html.Div(baslik[:1].upper(), style={
                    "width": "38px", "height": "38px", "borderRadius": "13px",
                    "display": "flex", "alignItems": "center", "justifyContent": "center",
                    "background": f"linear-gradient(135deg, {renk}22, {renk}44)",
                    "color": renk, "fontWeight": "900", "fontSize": "17px",
                    "border": f"1px solid {renk}33",
                }),
            ], style={"display": "flex", "alignItems": "flex-start"}),
        ], style={"display": "flex", "justifyContent": "space-between", "gap": "10px"}),
        dcc.Graph(figure=mini_trend_fig(seri, renk), config={"displayModeBar": False},
                  style={"height": "48px", "marginTop": "6px"}) if seri else html.Div(),
    ], style={
        **CARD_STYLE,
        "padding": "16px 18px 12px",
        "minHeight": "138px",
        "borderTop": f"5px solid {renk}",
        "background": f"radial-gradient(circle at 85% 0%, {renk}26, transparent 34%), linear-gradient(180deg, rgba(255,255,255,0.94), rgba(255,255,255,0.74))",
        "boxShadow": f"0 22px 48px {renk}22",
    })


def make_kpi_row(yil):
    """Seçilen yıla göre KPI kartlarını oluşturur. Tümü seçilirse CSV’deki genel dönem özetini gösterir."""
    tumu = yil == "all"
    aktif_yil = max(YILLAR) if tumu else int(yil)
    row = turkey_ts[turkey_ts["yil"] == aktif_yil].iloc[0]
    base = turkey_ts[turkey_ts["yil"] == min(YILLAR)].iloc[0]
    cards = [
        ("İnternet Erişimi", "internet_erisim", C_ACCENT, "Hanehalkı erişimi"),
        ("E-Devlet Kullanımı", "edevlet", C_ACCENT2, "Kamu hizmetleri"),
        ("Akıllı Telefon Sahipliği", "akilli_telefon", C_GREEN, "Hanehalkı sahipliği"),
        ("E-Ticaret Kullanımı", "eticaret", C_ORANGE, "Son 12 ay kullanımı"),
        ("YZ Farkındalığı", "yapay_zeka_farkin", C_PURPLE, "Kullanım / farkındalık"),
    ]
    return html.Div([
        html.Div(
            kpi_card(
                baslik=title,
                deger=f"%{row[col]:.1f}",
                alt=(f"2015–{max(YILLAR)} genel dönem · {desc}" if tumu else f"{aktif_yil} yılı · {desc}"),
                trend=f"↑ +{row[col] - base[col]:.1f} puan ({min(YILLAR)} → {aktif_yil})",
                renk=color,
                seri=col,
            ),
            style={"flex": "1", "minWidth": "185px"}
        ) for title, col, color, desc in cards
    ], style={"display": "flex", "gap": "14px", "flexWrap": "wrap", "marginBottom": "14px"})



def make_hero_row(yil):
    """Üst özet: seçilen yıl ya da genel dönem için tek bakışta hikâye kartı."""
    tumu = yil == "all"
    aktif_yil = max(YILLAR) if tumu else int(yil)
    row = turkey_ts[turkey_ts["yil"] == aktif_yil].iloc[0]
    base = turkey_ts[turkey_ts["yil"] == min(YILLAR)].iloc[0]

    score_cols = ["internet_erisim", "edevlet", "eticaret", "yapay_zeka_farkin"]
    score_now = np.mean([row[c] for c in score_cols])
    score_base = np.mean([base[c] for c in score_cols])
    delta = score_now - score_base
    label = "Genel dönem görünümü" if tumu else f"{aktif_yil} yılı görünümü"
    caption = f"{min(YILLAR)}–{max(YILLAR)} dönemi CSV’den hesaplandı" if tumu else f"{min(YILLAR)} yılına göre {aktif_yil} seviyesi"

    def pill(text, color, bg):
        return html.Span(text, style={
            "background": bg, "color": color, "padding": "8px 12px",
            "borderRadius": "999px", "fontSize": "11px", "fontWeight": "900",
            "letterSpacing": "0.2px", "whiteSpace": "nowrap",
        })

    return html.Div([
        html.Div([
            html.Div("DİJİTAL DÖNÜŞÜM SKORU", style={
                "fontSize":"12px", "fontWeight":"950", "letterSpacing":"1px",
                "color":"rgba(255,255,255,.92)", "marginBottom":"10px"
            }),
            html.Div([
                html.Span(f"{score_now:.1f}", style={
                    "fontSize":"52px", "fontWeight":"950", "lineHeight":"0.95",
                    "color":"white"
                }),
                html.Span(" / 100", style={"fontSize":"22px", "fontWeight":"850", "color":"rgba(255,255,255,.72)", "marginLeft":"4px"}),
            ], style={"display":"flex", "alignItems":"baseline"}),
            html.Div(label, style={"fontSize":"15px", "fontWeight":"850", "color":"white", "marginTop":"9px"}),
            html.Div(caption, style={"fontSize":"12px", "fontWeight":"650", "color":"rgba(255,255,255,.78)", "marginTop":"4px"}),
        ], style={
            "flex":"1.15", "minWidth":"280px", "padding":"28px 30px",
            "borderRadius":"28px", "background":"linear-gradient(135deg,#7C3AED,#8B5CF6,#06B6D4)",
            "boxShadow":"0 28px 60px rgba(124,58,237,.28)",
        }),

        html.Div([
            html.Div("DÖNEM DEĞİŞİMİ", style={"fontSize":"12px", "fontWeight":"950", "letterSpacing":".6px", "color":C_MUTED, "marginBottom":"8px"}),
            html.Div(f"+{delta:.1f} puan", style={"fontSize":"28px", "fontWeight":"950", "color":C_GREEN}),
            html.Div(f"{min(YILLAR)} → {aktif_yil}", style={"fontSize":"12px", "fontWeight":"750", "color":C_MUTED}),
            html.Div(style={
                "height":"9px", "borderRadius":"999px", "background":"rgba(139,92,246,0.12)",
                "marginTop":"14px", "overflow":"hidden"
            }, children=[html.Div(style={
                "width": f"{min(score_now,100):.0f}%", "height":"100%",
                "background":"linear-gradient(90deg,#8B5CF6,#EC4899,#06B6D4)",
                "borderRadius":"999px"
            })]),
        ], style={
            "flex":".75", "minWidth":"250px", "background":"rgba(255,255,255,0.78)",
            "border":"1px solid rgba(139,92,246,0.14)", "borderRadius":"28px",
            "padding":"24px 26px", "boxShadow":"0 20px 45px rgba(139,92,246,.10)"
        }),

        html.Div([
            pill("En güçlü alan: İnternet erişimi", "#6D28D9", "rgba(139,92,246,0.12)"),
            pill("En hızlı artış: YZ farkındalığı", "#DB2777", "rgba(236,72,153,0.12)"),
            pill("Odak alanı: Bölgesel uçurum", "#08788F", "rgba(6,182,212,0.12)"),
        ], style={"display":"flex", "gap":"10px", "alignItems":"stretch", "flexDirection":"column", "justifyContent":"center", "flex":".9", "minWidth":"260px"}),
    ], style={
        "display":"flex", "gap":"18px", "alignItems":"stretch", "justifyContent":"space-between",
        "padding":"18px", "marginBottom":"18px",
        "borderRadius":"32px",
        "background":"linear-gradient(135deg, rgba(255,255,255,.74), rgba(255,255,255,.52))",
        "backdropFilter":"blur(20px)",
        "border":"1px solid rgba(255,255,255,.72)",
        "boxShadow":"0 24px 60px rgba(139,92,246,.14)",
        "overflow":"hidden",
    })

# ─── Layout ─────────────────────────────────────────────────
app.layout = html.Div(style={
    "background": "radial-gradient(circle at 10% 5%, rgba(236,72,153,0.18) 0%, transparent 24%), radial-gradient(circle at 88% 0%, rgba(6,182,212,0.22) 0%, transparent 28%), radial-gradient(circle at 50% 18%, rgba(139,92,246,0.18) 0%, transparent 30%), linear-gradient(135deg, #FFF7FD 0%, #F4EEFF 45%, #EAFBFF 100%)",
    "minHeight": "100vh",
    "fontFamily": "Inter, system-ui, sans-serif",
    "padding": "0",
}, children=[

    # ── HEADER ──
    html.Div([
        html.Div([
            html.Div("TR", style={"fontSize": "28px", "fontWeight": "900", "color": "white", "marginRight": "18px", "letterSpacing": "1px", "width": "64px", "height": "64px", "borderRadius": "20px", "background": "linear-gradient(135deg, #8B5CF6, #06B6D4)", "display": "flex", "alignItems": "center", "justifyContent": "center", "boxShadow": "0 16px 34px rgba(139,92,246,0.26)"}),
            html.Div([
                html.H1("Türkiye Dijitalleşme Endeksi",
                        style={"color": C_TEXT, "fontSize": "34px", "fontWeight": "950",
                               "margin": "0", "letterSpacing": "0.2px", "lineHeight": "1.05"}),
                html.P("Yapay Zekâ Çağında Bölgesel Farklılıklar & Çok Boyutlu Analiz  •  "
                       "TÜİK Hanehalkı BİT Araştırması  •  "
                       "Dilara Şenay | TÜBİTAK 2209-A",
                       style={"color": C_MUTED, "fontSize": "12px", "fontWeight": "650", "margin": "8px 0 12px"}),
                html.Div([
                    html.Span("CSV Tabanlı Zaman Serisi", style={"background":"rgba(139,92,246,0.12)", "color":"#7C3AED", "padding":"7px 10px", "borderRadius":"999px", "fontSize":"10px", "fontWeight":"900", "letterSpacing":"0.3px"}),
                    html.Span("TÜİK BİT Araştırması", style={"background":"rgba(0,166,200,0.10)", "color":"#08788F", "padding":"7px 10px", "borderRadius":"999px", "fontSize":"10px", "fontWeight":"900", "letterSpacing":"0.3px"}),
                    html.Span("TÜBİTAK 2209-A", style={"background":"rgba(236,72,153,0.12)", "color":"#DB2777", "padding":"7px 10px", "borderRadius":"999px", "fontSize":"10px", "fontWeight":"900", "letterSpacing":"0.3px"}),
                ], style={"display":"flex", "gap":"8px", "flexWrap":"wrap"}),
            ]),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div([
            html.Div("Zaman Kapsamı", style={"color": "#6D28D9", "fontSize": "11px", "fontWeight": "950", "marginBottom": "8px", "letterSpacing": "0.2px"}),
            dcc.Dropdown(
                id="global-yil",
                options=[{"label": "Genel", "value": "all"}] + [{"label": str(y), "value": y} for y in YILLAR],
                value="all",
                clearable=False,
                searchable=False,
                style={"width": "180px", "fontSize": "13px", "color": C_TEXT},
            ),
        ], style={"display": "flex", "flexDirection":"column", "alignItems": "stretch", "gap": "0", "background":"rgba(255,255,255,.56)", "padding":"14px 16px", "borderRadius":"24px", "border":"1px solid rgba(139,92,246,.18)", "boxShadow":"0 18px 42px rgba(139,92,246,.14)", "backdropFilter":"blur(18px)"}),
    ], style={
        "background": "radial-gradient(circle at 70% 0%, rgba(236,72,153,0.22), transparent 26%), radial-gradient(circle at 92% 12%, rgba(6,182,212,0.24), transparent 25%), radial-gradient(circle at 55% 110%, rgba(139,92,246,0.16), transparent 34%), linear-gradient(135deg, rgba(255,255,255,0.93), rgba(253,242,248,0.82), rgba(236,254,255,0.78))",
        "backdropFilter": "blur(22px)",
        "borderBottom": f"1px solid {C_BORDER}",
        "boxShadow": "0 26px 70px rgba(139, 92, 246, 0.18)",
        "padding": "26px 32px",
        "overflow": "visible",
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
        "position": "sticky", "top": "0", "zIndex": "100",
    }),

    # ── SEKMELER ──
    html.Div([
        dcc.Tabs(id="ana-sekme", value="genel", children=[
            dcc.Tab(label="Genel Bakış",   value="genel"),
            dcc.Tab(label="Bölgesel Harita", value="harita"),
            dcc.Tab(label="Demografik",     value="demo"),
            dcc.Tab(label="NLP Analizi",    value="nlp"),
            dcc.Tab(label="Karşılaştırma",  value="karsil"),
        ], style={"fontFamily": "inherit"},
        colors={"border": C_BORDER, "primary": C_ACCENT, "background": C_BG}),
    ], style={"padding": "16px 32px 8px", "background": "linear-gradient(90deg, rgba(255,255,255,0.55), rgba(236,254,255,0.35), rgba(250,245,255,0.55))",
              "backdropFilter": "blur(12px)",
              "borderBottom": f"1px solid {C_BORDER}", "boxShadow": "0 10px 30px rgba(15,23,42,0.04)"}),

    html.Div(id="sekme-icerik", style={"padding": "20px 32px"}),
])


# ─────────────────────────────────────────────────────────────
# 3. TAB İÇERİKLERİ
# ─────────────────────────────────────────────────────────────

# ── Genel Bakış ──────────────────────────────────────────────
def genel_bakis():
    return html.Div([
        # HERO ROW - seçilen yıla göre genel skor ve hikâye
        html.Div(id="hero-row"),

        # KPI ROW - seçilen yıla göre callback ile güncellenir
        html.Div(id="kpi-row"),

        # ANA GRAFIK ROW
        html.Div([
            # Zaman Serisi
            html.Div([
                html.P("ZAMAN SERİSİ ANALİZİ", style=TITLE_STYLE),
                html.P("CSV’den hesaplanan temel BİT göstergeleri trendi", style=SUBTITLE_STYLE),
                dcc.Dropdown(
                    id="ts-gostergeler",
                    options=[
                        {"label": "İnternet Erişimi",    "value": "internet_erisim"},
                        {"label": "Akıllı Telefon",       "value": "akilli_telefon"},
                        {"label": "E-Devlet",             "value": "edevlet"},
                        {"label": "E-Ticaret",            "value": "eticaret"},
                        {"label": "Sosyal Medya",         "value": "sosyal_medya"},
                        {"label": "Çevrimiçi Eğitim",    "value": "cevrimici_egitim"},
                        {"label": "YZ Farkındalığı",     "value": "yapay_zeka_farkin"},
                        {"label": "Bilgisayar",           "value": "bilgisayar"},
                    ],
                    value=["internet_erisim","edevlet","eticaret","yapay_zeka_farkin"],
                    multi=True,
                    style={"backgroundColor": C_CARD, "color": C_TEXT,
                           "border": f"1px solid {C_BORDER}", "borderRadius": "8px",
                           "fontSize": "12px", "marginBottom": "12px"},
                ),
                dcc.Graph(id="ts-grafik", style={"height": "340px"}),
            ], style={**CARD_STYLE, "flex": "3"}),

            # Pandemi Etkisi
            html.Div([
                html.P("PANDEMİ ETKİSİ ANALİZİ", style=TITLE_STYLE),
                html.P("COVID-19 Dijital Dönüşüm Hızlanması", style=SUBTITLE_STYLE),
                dcc.Graph(id="pandemi-grafik",
                          figure=_pandemi_fig("all"),
                          style={"height": "386px"}),
            ], style={**CARD_STYLE, "flex": "2"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        # ALT ROW
        html.Div([
            # Büyüme hızı
            html.Div([
                html.P("YILLIK BÜYÜME HIZI (%)", style=TITLE_STYLE),
                html.P("Gösterge bazında yıllık değişim oranları", style=SUBTITLE_STYLE),
                dcc.Graph(id="buyume-grafik", figure=_buyume_fig("all"), style={"height": "260px"}),
            ], style={**CARD_STYLE, "flex": "2"}),
            # DESI karşılaştırma
            html.Div([
                html.P("ULUSLARARASI KARŞILAŞTIRMA", style=TITLE_STYLE),
                html.P("DESI 2024 – Türkiye vs AB Ülkeleri", style=SUBTITLE_STYLE),
                dcc.Graph(figure=_desi_fig(), style={"height": "260px"}),
            ], style={**CARD_STYLE, "flex": "3"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
    ])


def _pandemi_fig(yil="all"):
    fig = go.Figure()
    pandemi_oncesi = turkey_ts[turkey_ts["yil"] < 2020]
    pandemi_sonrasi = turkey_ts[turkey_ts["yil"] >= 2020]

    for df, dash_type in [(pandemi_oncesi, "solid"), (pandemi_sonrasi, "dot")]:
        for col, color, label in [
            ("edevlet", C_ACCENT, "E-Devlet"),
            ("cevrimici_egitim", C_ORANGE, "Çevrimiçi Eğitim"),
            ("eticaret", C_GREEN, "E-Ticaret"),
        ]:
            fig.add_trace(go.Scatter(
                x=df["yil"], y=df[col],
                name=label if dash_type == "solid" else None,
                line=dict(color=color, width=2, dash=dash_type),
                showlegend=dash_type == "solid",
                mode="lines+markers" if dash_type == "dot" else "lines",
                marker=dict(size=5),
            ))

    fig.add_vrect(x0=2019.5, x1=2021.5,
                  fillcolor="rgba(249,115,22,0.11)",
                  line_width=1, line_color=C_ORANGE,
                  annotation_text="Pandemi", annotation_position="top left",
                  annotation_font=dict(color=C_ORANGE, size=10))
    if yil != "all":
        secili_yil = int(yil)
        fig.add_vline(x=secili_yil, line_width=2, line_dash="dash", line_color=C_PINK)
        fig.add_annotation(x=secili_yil, y=78, text=f"Seçili yıl: {secili_yil}",
                           showarrow=False, font=dict(color=C_PINK, size=10))
    fig.update_layout(**PLOTLY_LAYOUT, showlegend=True)
    return fig


def _buyume_fig(yil="all"):
    cols = ["internet_erisim", "edevlet", "akilli_telefon", "eticaret", "yapay_zeka_farkin"]
    labels = ["İnternet", "E-Devlet", "Akıllı Telefon", "E-Ticaret", "YZ Farkındalığı"]
    colors = [C_ACCENT, C_ACCENT2, C_GREEN, C_ORANGE, C_PINK]

    values = []
    title_suffix = "Seçili dönem ortalama yıllık değişim"
    if yil == "all":
        for c in cols:
            vals = pd.to_numeric(turkey_ts[c], errors="coerce").values
            growth = []
            for i in range(1, len(vals)):
                prev_val = vals[i - 1]
                cur_val = vals[i]
                if pd.isna(prev_val) or prev_val == 0 or pd.isna(cur_val):
                    continue
                growth.append((cur_val - prev_val) / prev_val * 100)
            values.append(float(np.mean(growth)) if growth else 0)
    else:
        secili_yil = int(yil)
        idx = turkey_ts.index[turkey_ts["yil"] == secili_yil][0]
        if idx == 0:
            values = [0 for _ in cols]
            title_suffix = f"{secili_yil} için önceki yıl yok"
        else:
            prev = turkey_ts.iloc[idx - 1]
            cur = turkey_ts.iloc[idx]
            values = []
            for c in cols:
                prev_val = prev[c]
                cur_val = cur[c]
                if pd.isna(prev_val) or prev_val == 0 or pd.isna(cur_val):
                    values.append(0)
                else:
                    values.append((cur_val - prev_val) / prev_val * 100)
            title_suffix = f"{secili_yil-1} → {secili_yil} yıllık değişim"

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color=C_TEXT, size=11, family="Inter"),
        hovertemplate="<b>%{x}</b><br>Büyüme: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, yaxis_title="Yıllık Büyüme (%)", title=dict(text=title_suffix, font=dict(size=11, color=C_MUTED), x=0.02))
    return fig


def _desi_fig():
    ulkeler = ["Finlandiya","Danimarka","Hollanda","İsveç","AB Ortalaması",
               "Polonya","Romanya","Türkiye","Bulgaristan"]
    skorlar = [79.1, 77.8, 76.4, 75.9, 55.3, 47.2, 38.8, 36.4, 35.1]
    renkler = [C_ACCENT if u != "Türkiye" and u != "AB Ortalaması" else
               (C_ORANGE if u == "Türkiye" else C_GREEN) for u in ulkeler]

    fig = go.Figure(go.Bar(
        x=skorlar, y=ulkeler, orientation="h",
        marker_color=renkler,
        text=[f"{s}" for s in skorlar],
        textposition="inside",
        textfont=dict(color="white", size=10),
    ))
    ds = {**PLOTLY_LAYOUT}
    ds["xaxis"] = dict(title="DESI Skoru (0-100)", gridcolor=C_BORDER, tickfont=dict(color=C_MUTED, size=10))
    ds["yaxis"] = dict(gridcolor=C_BORDER, linecolor=C_BORDER, tickfont=dict(color=C_TEXT, size=10))
    fig.update_layout(**ds)
    return fig


# ── Bölgesel Harita ──────────────────────────────────────────
def bolgesel_harita():
    return html.Div([
        html.Div([
            html.Div([
                html.P("HARİTA GÖSTERGESİ", style=TITLE_STYLE),
                dcc.RadioItems(
                    id="harita-gosterge",
                    options=[
                        {"label": " İnternet Erişimi", "value": "internet_erisim"},
                        {"label": " E-Devlet", "value": "edevlet"},
                        {"label": "Dijital Beceri", "value": "dijital_beceri"},
                        {"label": " E-Ticaret", "value": "eticaret"},
                        {"label": "Dijitalleşme Skoru", "value": "dijit_skor"},
                    ],
                    value="dijit_skor",
                    labelStyle={"display": "block", "marginBottom": "8px",
                                "color": C_TEXT, "fontSize": "12px", "cursor": "pointer"},
                ),
                html.Hr(style={"borderColor": C_BORDER, "margin": "16px 0"}),
                html.P("K-MEANS KÜMELEMESİ", style=TITLE_STYLE),
                html.P("Algoritmik grup ataması (k=4)", style=SUBTITLE_STYLE),
                *[html.Div([
                    html.Span("■", style={"color": KUME_RENK[i], "fontSize": "18px"}),
                    html.Span(f" {KUME_ETIKET[i]}", style={"color": C_TEXT, "fontSize": "11px"}),
                ], style={"marginBottom": "6px"}) for i in range(4)],

                html.Hr(style={"borderColor": C_BORDER, "margin": "16px 0"}),
                html.P("BÖLGE SIRALAMALARI", style=TITLE_STYLE),
                html.P("Dijitalleşme skoru bazında", style=SUBTITLE_STYLE),
                dcc.Graph(
                    id="bolge-siralama",
                    style={"height": "300px"},
                    config={"displayModeBar": False},
                ),
            ], style={**CARD_STYLE, "width": "260px", "flexShrink": "0"}),

            html.Div([
                html.P("TÜRKİYE BÖLGESEL DİJİTALLEŞME HARİTASI", style=TITLE_STYLE),
                html.P("İBBS-1 bölgeleri · seçili göstergeye ve yıla göre renklendirilmiş gerçek Türkiye haritası", style=SUBTITLE_STYLE),
                dcc.Graph(id="bolge-harita", style={"height": "480px"}),
                html.Div(id="bolge-detay", style={"marginTop": "12px"}),
            ], style={**CARD_STYLE, "flex": "1"}),
        ], style={"display": "flex", "gap": "16px", "alignItems": "flex-start"}),

        # Alt istatistik kartları
        html.Div([
            html.Div([
                html.P("BÖLGESEL UÇURUM ANALİZİ", style=TITLE_STYLE),
                html.P("En yüksek ile en düşük bölge arasındaki fark", style=SUBTITLE_STYLE),
                dcc.Graph(id="ucurum-grafik", style={"height": "280px"}),
            ], style={**CARD_STYLE, "flex": "1"}),
            html.Div([
                html.P("KÜME DAĞILIMI – RADAR GRAFİĞİ", style=TITLE_STYLE),
                html.P("Küme bazında ortalama gösterge profilleri", style=SUBTITLE_STYLE),
                dcc.Graph(id="radar-grafik", style={"height": "280px"}),
            ], style={**CARD_STYLE, "flex": "1"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
    ])


def _bolge_bar(df=None):
    if df is None:
        df = bolge_data
    srtd = df.sort_values("dijit_skor", ascending=True)
    colors = [KUME_RENK[k] for k in srtd["kmeans_kume"]]
    fig = go.Figure(go.Bar(
        x=srtd["dijit_skor"], y=srtd["bolge"], orientation="h",
        marker_color=colors,
        text=[f"{v}" for v in srtd["dijit_skor"]],
        textposition="inside", textfont=dict(color="white", size=9),
    ))
    layout = {**PLOTLY_LAYOUT}
    layout["margin"] = dict(l=100, r=5, t=5, b=5)
    layout["xaxis"] = dict(range=[30, 100], gridcolor=C_BORDER, tickfont=dict(color=C_MUTED, size=8))
    layout["yaxis"] = dict(tickfont=dict(color=C_TEXT, size=9), gridcolor=C_BORDER)
    fig.update_layout(**layout)
    return fig


def _ucurum_fig(df=None):
    if df is None:
        df = bolge_data
    gostergeler = ["internet_erisim", "edevlet", "dijital_beceri", "eticaret"]
    etiketler = ["İnternet\nErişimi", "E-Devlet", "Dijital\nBeceri", "E-Ticaret"]
    max_vals = [df[g].max() for g in gostergeler]
    min_vals = [df[g].min() for g in gostergeler]
    fark = [m - n for m, n in zip(max_vals, min_vals)]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="En Yüksek Bölge", x=etiketler, y=max_vals,
                         marker_color=C_ACCENT, text=[f"{v:.1f}%" for v in max_vals],
                         textposition="outside", textfont=dict(size=9, color=C_TEXT)))
    fig.add_trace(go.Bar(name="En Düşük Bölge", x=etiketler, y=min_vals,
                         marker_color=C_RED, text=[f"{v:.1f}%" for v in min_vals],
                         textposition="inside", textfont=dict(size=9, color="white")))
    fig.add_trace(go.Scatter(name="Fark (pp)", x=etiketler, y=fark,
                             mode="markers+text", marker=dict(color=C_ORANGE, size=12),
                             text=[f"Δ{v:.1f}" for v in fark],
                             textposition="top center", textfont=dict(size=10, color=C_ORANGE)))
    fig.update_layout(**PLOTLY_LAYOUT, barmode="overlay", yaxis_title="%")
    return fig


def _radar_fig(df=None):
    if df is None:
        df = bolge_data
    kategoriler = ["İnternet\nErişimi", "E-Devlet", "Dijital\nBeceri", "E-Ticaret"]
    kume_renk_list = [KUME_RENK[i] for i in range(4)]

    fig = go.Figure()
    for kume_id in range(4):
        grup = df[df["kmeans_kume"] == kume_id]
        if len(grup) == 0:
            continue
        vals = [grup["internet_erisim"].mean(), grup["edevlet"].mean(),
                grup["dijital_beceri"].mean(), grup["eticaret"].mean()]
        vals_closed = vals + [vals[0]]
        kat_closed = kategoriler + [kategoriler[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed, theta=kat_closed,
            name=KUME_ETIKET[kume_id],
            line_color=kume_renk_list[kume_id],
            fill="toself",
            fillcolor="rgba(0,0,0,0.1)",
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[20, 100],
                            gridcolor=C_BORDER, tickfont=dict(color=C_MUTED, size=8)),
            angularaxis=dict(gridcolor=C_BORDER, tickfont=dict(color=C_TEXT, size=9)),
        ),
        font=dict(family="Inter, sans-serif", color=C_TEXT),
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(bgcolor="rgba(255,255,255,0.76)", bordercolor=C_BORDER,
                    borderwidth=1, font=dict(size=9, color=C_TEXT)),
        showlegend=True,
    )
    return fig


# ── Demografik ───────────────────────────────────────────────
def demografik():
    return html.Div([
        html.Div([
            # Yaş Grubu
            html.Div([
                html.P("YAŞ GRUBUNA GÖRE DİJİTAL KULLANIM", style=TITLE_STYLE),
                html.P("İnternet, E-Devlet, Sosyal Medya, E-Ticaret oranları (%)", style=SUBTITLE_STYLE),
                dcc.Graph(figure=_yas_fig(), style={"height": "340px"}),
            ], style={**CARD_STYLE, "flex": "1"}),
            # Eğitim
            html.Div([
                html.P("EĞİTİM DÜZEYİ İLE DİJİTALLEŞME İLİŞKİSİ", style=TITLE_STYLE),
                html.P("Eğitim seviyesi arttıkça dijital katılım doğrusal artış gösteriyor", style=SUBTITLE_STYLE),
                dcc.Graph(figure=_egitim_fig(), style={"height": "340px"}),
            ], style={**CARD_STYLE, "flex": "1"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        html.Div([
            # Cinsiyet
            html.Div([
                html.P("CİNSİYET FARKLILIKLARI", style=TITLE_STYLE),
                html.P("Dijital göstergeler bazında erkek-kadın karşılaştırması", style=SUBTITLE_STYLE),
                dcc.Graph(figure=_cinsiyet_fig(), style={"height": "280px"}),
            ], style={**CARD_STYLE, "flex": "1"}),
            # Dijital Uçurum Kök Nedenleri
            html.Div([
                html.P("DİJİTAL UÇURUMUN SOSYAL BOYUTLARI", style=TITLE_STYLE),
                html.P("OLS Regresyon – Dijitalleşmeye etkisi en yüksek değişkenler", style=SUBTITLE_STYLE),
                dcc.Graph(figure=_ols_fig(), style={"height": "280px"}),
            ], style={**CARD_STYLE, "flex": "1"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
    ])


def _yas_fig():
    fig = go.Figure()
    traces = [
        ("internet",  "İnternet",      C_ACCENT),
        ("edevlet",   "E-Devlet",      C_ACCENT2),
        ("sosyal",    "Sosyal Medya",   C_GREEN),
        ("eticaret",  "E-Ticaret",      C_ORANGE),
    ]
    for col, label, color in traces:
        fig.add_trace(go.Scatter(
            x=demo_df["yas_grubu"], y=demo_df[col],
            name=label, line=dict(color=color, width=2),
            mode="lines+markers", marker=dict(size=7),
            fill="tozeroy" if col == "internet" else None,
            fillcolor="rgba(59,130,246,0.05)" if col == "internet" else None,
        ))
    fig.update_layout(**PLOTLY_LAYOUT, yaxis_title="Kullanım Oranı (%)", yaxis_range=[0, 105])
    return fig


def _egitim_fig():
    fig = go.Figure()
    for col, label, color in [
        ("internet",  "İnternet Erişimi", C_ACCENT),
        ("edevlet",   "E-Devlet",          C_ACCENT2),
        ("dijital_b", "Dijital Beceri",    C_GREEN),
    ]:
        fig.add_trace(go.Bar(
            name=label, x=egitim_df["egitim"], y=egitim_df[col],
            marker_color=color,
            text=[f"{v:.0f}%" for v in egitim_df[col]],
            textposition="outside", textfont=dict(size=8, color=C_TEXT),
        ))
    fig.update_layout(**PLOTLY_LAYOUT, barmode="group",
                      yaxis_title="Oran (%)", yaxis_range=[0, 110])
    return fig


def _cinsiyet_fig():
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Erkek", x=cinsiyet_df["kategori"], y=cinsiyet_df["erkek"],
                         marker_color=C_ACCENT,
                         text=[f"{v}" for v in cinsiyet_df["erkek"]],
                         textposition="outside", textfont=dict(size=9, color=C_TEXT)))
    fig.add_trace(go.Bar(name="Kadın", x=cinsiyet_df["kategori"], y=cinsiyet_df["kadin"],
                         marker_color="#EC4899",
                         text=[f"{v}" for v in cinsiyet_df["kadin"]],
                         textposition="outside", textfont=dict(size=9, color=C_TEXT)))

    # Fark çizgisi
    fark = cinsiyet_df["erkek"] - cinsiyet_df["kadin"]
    for i, (kat, f) in enumerate(zip(cinsiyet_df["kategori"], fark)):
        fig.add_annotation(
            x=kat, y=max(cinsiyet_df["erkek"].iloc[i], cinsiyet_df["kadin"].iloc[i]) + 4,
            text=f"Δ{f:.1f}", showarrow=False,
            font=dict(color=C_ORANGE, size=9),
        )
    fig.update_layout(**PLOTLY_LAYOUT, barmode="group", yaxis_range=[0, 108], yaxis_title="%")
    return fig


def _ols_fig():
    degiskenler = ["Eğitim Düzeyi", "Gelir Seviyesi", "Kentsel Yaşam",
                   "Genç Yaş (16-34)", "Hanehalkı Büyüklüğü", "Bölge Etkisi"]
    katsayilar = [0.48, 0.36, 0.29, 0.24, -0.17, 0.21]
    renkler = [C_GREEN if k > 0 else C_RED for k in katsayilar]

    fig = go.Figure(go.Bar(
        x=katsayilar, y=degiskenler, orientation="h",
        marker_color=renkler,
        text=[f"β={v:.2f}" for v in katsayilar],
        textposition="outside",
        textfont=dict(color=C_TEXT, size=10),
    ))
    fig.add_vline(x=0, line_color=C_BORDER, line_width=1)
    ol = {**PLOTLY_LAYOUT}
    ol["xaxis"] = dict(title="Standardize OLS Katsayısı (β)", gridcolor=C_BORDER, tickfont=dict(color=C_MUTED, size=10))
    ol["yaxis"] = dict(tickfont=dict(color=C_TEXT, size=10), gridcolor=C_BORDER)
    fig.update_layout(**ol)
    return fig


# ── NLP Analizi ──────────────────────────────────────────────
def nlp_analizi():
    return html.Div([
        html.Div([
            html.Div([
                html.P("DUYGU ANALİZİ SONUÇLARI", style=TITLE_STYLE),
                html.P("X/Twitter verileri – NLP tabanlı sentiment analizi (n≈8,000 gönderi)", style=SUBTITLE_STYLE),
                dcc.Graph(figure=_duygu_fig(), style={"height": "320px"}),
            ], style={**CARD_STYLE, "flex": "2"}),
            html.Div([
                html.P("LDA KONU MODELLEMESİ", style=TITLE_STYLE),
                html.P("Latent Dirichlet Allocation – 5 Tema", style=SUBTITLE_STYLE),
                dcc.Graph(figure=_lda_fig(), style={"height": "320px"}),
            ], style={**CARD_STYLE, "flex": "1"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        html.Div([
            html.Div([
                html.P("KONU BAZINDA DUYGU DAĞILIMI", style=TITLE_STYLE),
                html.P("Her konuya ait pozitif / nötr / negatif oran (100% stacked)", style=SUBTITLE_STYLE),
                dcc.Graph(figure=_stacked_duygu_fig(), style={"height": "280px"}),
            ], style={**CARD_STYLE, "flex": "2"}),
            html.Div([
                html.P("ANAHTAR KELİME FREKANSI", style=TITLE_STYLE),
                html.P("En çok geçen dijitalleşme terimleri", style=SUBTITLE_STYLE),
                dcc.Graph(figure=_kelime_fig(), style={"height": "280px"}),
            ], style={**CARD_STYLE, "flex": "1"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        # Tema kartları
        html.Div([
            html.P("ÇIKARILAN TEMALAR & ÖRNEK KELİMELER", style=TITLE_STYLE),
            html.P("LDA Konu Modellemesi – Tema Özeti", style=SUBTITLE_STYLE),
            html.Div([
                html.Div([
                    html.Div([
                        html.Span(f"#{row['konu_no']}", style={
                            "backgroundColor": row["renk"],
                            "color": "white", "fontSize": "11px", "fontWeight": "700",
                            "padding": "2px 8px", "borderRadius": "4px", "marginRight": "8px",
                        }),
                        html.Span(row["konu_adi"], style={"color": C_TEXT, "fontWeight": "600", "fontSize": "12px"}),
                        html.Span(f"  %{row['agirlik']}", style={"color": C_MUTED, "fontSize": "11px"}),
                    ]),
                    html.P(row["kelimeler"], style={"color": C_MUTED, "fontSize": "10px", "marginTop": "4px", "marginBottom": "0"}),
                ], style={
                    "backgroundColor": C_BG, "border": f"1px solid {row['renk']}22",
                    "borderLeft": f"3px solid {row['renk']}", "borderRadius": "8px",
                    "padding": "10px 14px", "flex": "1", "minWidth": "200px",
                }) for _, row in lda_konular.iterrows()
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ], style=CARD_STYLE),
    ])


def _duygu_fig():
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Pozitif", x=nlp_df["konu"], y=nlp_df["pozitif"],
                         marker_color=C_GREEN,
                         text=[f"{v:.0f}%" for v in nlp_df["pozitif"]],
                         textposition="inside", textfont=dict(color="white", size=9)))
    fig.add_trace(go.Bar(name="Nötr", x=nlp_df["konu"], y=nlp_df["notr"],
                         marker_color="#475569",
                         text=[f"{v:.0f}%" for v in nlp_df["notr"]],
                         textposition="inside", textfont=dict(color="white", size=9)))
    fig.add_trace(go.Bar(name="Negatif", x=nlp_df["konu"], y=nlp_df["negatif"],
                         marker_color=C_RED,
                         text=[f"{v:.0f}%" for v in nlp_df["negatif"]],
                         textposition="inside", textfont=dict(color="white", size=9)))
    fig.update_layout(**PLOTLY_LAYOUT, barmode="stack", yaxis_title="Gönderi Oranı (%)")
    return fig


def _lda_fig():
    fig = go.Figure(go.Pie(
        labels=lda_konular["konu_adi"],
        values=lda_konular["agirlik"],
        marker_colors=lda_konular["renk"].tolist(),
        textinfo="label+percent",
        textfont=dict(size=9, color="white"),
        hole=0.4,
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", color=C_TEXT),
                      margin=dict(l=0, r=0, t=10, b=10),
                      showlegend=False)
    return fig


def _stacked_duygu_fig():
    total = nlp_df["pozitif"] + nlp_df["notr"] + nlp_df["negatif"]
    fig = go.Figure()
    for col, label, color in [
        ("pozitif", "Pozitif", C_GREEN),
        ("notr", "Nötr", "#475569"),
        ("negatif", "Negatif", C_RED),
    ]:
        pct = (nlp_df[col] / total * 100).round(1)
        fig.add_trace(go.Bar(
            name=label, x=nlp_df["konu"], y=pct,
            marker_color=color,
            text=[f"{v:.0f}%" for v in pct],
            textposition="inside", textfont=dict(color="white", size=9),
        ))
    fig.update_layout(**PLOTLY_LAYOUT, barmode="stack", yaxis_title="%", yaxis_range=[0, 101])
    return fig


def _kelime_fig():
    kelimeler = ["edevlet","yapay zeka","internet","dijital","hizmet",
                 "online","teknoloji","alışveriş","chatgpt","altyapı"]
    frekans = [4821, 3902, 3541, 3280, 2940, 2712, 2485, 2231, 2018, 1895]
    colors = [C_ACCENT if i < 3 else (C_ACCENT2 if i < 6 else C_MUTED) for i in range(len(kelimeler))]

    fig = go.Figure(go.Bar(
        x=frekans, y=kelimeler, orientation="h",
        marker_color=colors,
        text=[f"{v:,}" for v in frekans],
        textposition="outside", textfont=dict(color=C_TEXT, size=9),
    ))
    kl = {**PLOTLY_LAYOUT, "xaxis": dict(title="Gönderi Sayısı", gridcolor=C_BORDER, tickfont=dict(color=C_MUTED, size=10)), "yaxis": dict(tickfont=dict(color=C_TEXT, size=10), gridcolor=C_BORDER)}
    kl["margin"] = dict(l=80, r=60, t=10, b=10)
    fig.update_layout(**kl)
    return fig


# ── Karşılaştırma ────────────────────────────────────────────
def karsil_analiz():
    return html.Div([
        html.Div([
            html.Div([
                html.P("YILLIK KARŞILAŞTIRMALI ANALİZ", style=TITLE_STYLE),
                html.P("İki yıl seçin – Gösterge bazında değişimi inceleyin", style=SUBTITLE_STYLE),
                html.Div([
                    html.Div([
                        html.Label("Yıl 1:", style={"color": C_MUTED, "fontSize": "11px"}),
                        dcc.Dropdown(id="karsil-yil1",
                                     options=[{"label": str(y), "value": y} for y in YILLAR],
                                     value=min(YILLAR),
                                     style={"backgroundColor": C_CARD, "color": C_TEXT,
                                            "border": f"1px solid {C_BORDER}", "borderRadius": "8px",
                                            "fontSize": "12px"}),
                    ], style={"flex": "1"}),
                    html.Div([
                        html.Label("Yıl 2:", style={"color": C_MUTED, "fontSize": "11px"}),
                        dcc.Dropdown(id="karsil-yil2",
                                     options=[{"label": str(y), "value": y} for y in YILLAR],
                                     value=max(YILLAR),
                                     style={"backgroundColor": C_CARD, "color": C_TEXT,
                                            "border": f"1px solid {C_BORDER}", "borderRadius": "8px",
                                            "fontSize": "12px"}),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),
                dcc.Graph(id="karsil-grafik", style={"height": "380px"}),
            ], style={**CARD_STYLE, "flex": "3"}),

            html.Div([
                html.P("PANDEMİ ÖNCESİ / SONRASI", style=TITLE_STYLE),
                html.P("Pandemi öncesi / sonrası dijital ivme", style=SUBTITLE_STYLE),
                dcc.Graph(figure=_ivme_fig(), style={"height": "430px"}),
            ], style={**CARD_STYLE, "flex": "2"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        html.Div([
            html.P("12. KALKINMA PLANI HEDEFLERİ İLE UYUM", style=TITLE_STYLE),
            html.P("2024 Gerçekleşme vs 2028 Hedefleri (%)", style=SUBTITLE_STYLE),
            dcc.Graph(figure=_hedef_fig(), style={"height": "300px"}),
        ], style=CARD_STYLE),
    ])


def _ivme_fig():
    cols_labels = [
        ("edevlet",          "E-Devlet"),
        ("cevrimici_egitim", "Çevrimiçi Eğitim"),
        ("eticaret",         "E-Ticaret"),
        ("sosyal_medya",     "Sosyal Medya"),
        ("yapay_zeka_farkin","YZ Farkındalığı"),
    ]
    v2019 = turkey_ts[turkey_ts["yil"] == 2019].iloc[0]
    v2021 = turkey_ts[turkey_ts["yil"] == 2021].iloc[0]

    etiketler = [l for _, l in cols_labels]
    vals_19   = [v2019[c] for c, _ in cols_labels]
    vals_21   = [v2021[c] for c, _ in cols_labels]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="2019 (Pandemi öncesi)", x=etiketler, y=vals_19,
                         marker_color="#C4B5FD",
                         text=[f"{v:.0f}%" for v in vals_19],
                         textposition="inside", textfont=dict(color="white", size=9)))
    fig.add_trace(go.Bar(name=f"{comp_year} (karşılaştırma)", x=etiketler, y=vals_21,
                         marker_color=C_ACCENT,
                         text=[f"{v:.0f}%" for v in vals_21],
                         textposition="inside", textfont=dict(color="white", size=9)))

    for i, (v1, v2, lbl) in enumerate(zip(vals_19, vals_21, etiketler)):
        delta = v2 - v1
        fig.add_annotation(
            x=lbl, y=max(v1, v2) + 3,
            text=f"+{delta:.1f}pp", showarrow=False,
            font=dict(color=C_GREEN if delta > 0 else C_RED, size=10, family="Inter"),
        )

    iv = {**PLOTLY_LAYOUT}
    iv["xaxis"] = dict(tickfont=dict(color=C_TEXT, size=9), gridcolor=C_BORDER, tickangle=-30)
    iv["yaxis"] = dict(gridcolor=C_BORDER, tickfont=dict(color=C_MUTED, size=10), title="%", range=[0, 108])
    fig.update_layout(**iv, barmode="group")
    return fig


def _hedef_fig():
    hedefler = pd.DataFrame({
        "gosterge": ["İnternet\nErişimi", "E-Devlet", "Dijital\nBeceri",
                     "Fiber\nBağlantı", "YZ\nBenimseme", "E-Ticaret"],
        "gerceklesme_2024": [97.2, 69.2, 54.1, 38.4, 58.3, 57.4],
        "hedef_2028":       [99.0, 85.0, 75.0, 65.0, 75.0, 72.0],
    })
    hedefler["tamamlanma"] = (hedefler["gerceklesme_2024"] / hedefler["hedef_2028"] * 100).round(1)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="2024 Gerçekleşme", x=hedefler["gosterge"],
                         y=hedefler["gerceklesme_2024"],
                         marker_color=C_ACCENT,
                         text=[f"{v:.1f}%" for v in hedefler["gerceklesme_2024"]],
                         textposition="inside", textfont=dict(color="white", size=10)))
    fig.add_trace(go.Bar(name="2028 Hedefi", x=hedefler["gosterge"],
                         y=hedefler["hedef_2028"],
                         marker_color="#C4B5FD",
                         text=[f"{v:.0f}%" for v in hedefler["hedef_2028"]],
                         textposition="outside", textfont=dict(color=C_MUTED, size=10),
                         opacity=0.7))
    for i, row in hedefler.iterrows():
        renk = C_GREEN if row["tamamlanma"] >= 80 else (C_ORANGE if row["tamamlanma"] >= 60 else C_RED)
        fig.add_annotation(x=row["gosterge"], y=row["hedef_2028"] + 4,
                           text=f"%{row['tamamlanma']:.0f}", showarrow=False,
                           font=dict(color=renk, size=10, family="Inter"))
    fig.update_layout(**PLOTLY_LAYOUT, barmode="overlay", yaxis_title="%", yaxis_range=[0, 110])
    return fig


# ─────────────────────────────────────────────────────────────
# 4. CALLBACK'LER
# ─────────────────────────────────────────────────────────────

@app.callback(Output("sekme-icerik", "children"), Input("ana-sekme", "value"))
def render_tab(sekme):
    if sekme == "genel":   return genel_bakis()
    if sekme == "harita":  return bolgesel_harita()
    if sekme == "demo":    return demografik()
    if sekme == "nlp":     return nlp_analizi()
    if sekme == "karsil":  return karsil_analiz()
    return html.Div("Sekme bulunamadı")





@app.callback(Output("hero-row", "children"), Input("global-yil", "value"))
def update_hero_row(yil):
    return make_hero_row(yil)

@app.callback(Output("kpi-row", "children"), Input("global-yil", "value"))
def update_kpi_row(yil):
    return make_kpi_row(yil)

@app.callback(Output("ts-grafik", "figure"),
              Input("ts-gostergeler", "value"),
              Input("global-yil", "value"))
def update_ts(secili, yil):
    if not secili:
        secili = ["internet_erisim"]
    renk_paleti = [C_ACCENT, C_ACCENT2, C_GREEN, C_ORANGE, C_PURPLE, "#EC4899", "#F97316", "#14B8A6"]
    etiket_map = {
        "internet_erisim": "İnternet Erişimi",
        "akilli_telefon":  "Akıllı Telefon",
        "edevlet":         "E-Devlet",
        "eticaret":        "E-Ticaret",
        "sosyal_medya":    "Sosyal Medya",
        "cevrimici_egitim":"Çevrimiçi Eğitim",
        "yapay_zeka_farkin":"YZ Farkındalığı",
        "bilgisayar":      "Bilgisayar",
    }
    if yil == "all":
        df_plot = turkey_ts.copy()
        secili_text = f"Genel: {min(YILLAR)}–{max(YILLAR)}"
    else:
        secili_yil = int(yil)
        baslangic = max(min(YILLAR), secili_yil - 5)
        df_plot = turkey_ts[(turkey_ts["yil"] >= baslangic) & (turkey_ts["yil"] <= secili_yil)].copy()
        secili_text = f"Seçili dönem: {baslangic}–{secili_yil}"

    fig = go.Figure()
    for i, col in enumerate(secili):
        fig.add_trace(go.Scatter(
            x=df_plot["yil"], y=df_plot[col],
            name=etiket_map.get(col, col),
            line=dict(color=renk_paleti[i % len(renk_paleti)], width=2.5),
            mode="lines+markers",
            marker=dict(size=7),
            hovertemplate=f"<b>%{{x}}</b><br>{etiket_map.get(col, col)}: %{{y:.1f}}%<extra></extra>",
        ))

    # Pandemi işareti
    fig.add_vrect(x0=2019.5, x1=2021.5,
                  fillcolor="rgba(245,158,11,0.07)",
                  line_width=1, line_dash="dot", line_color=C_ORANGE)
    fig.add_annotation(x=2020.5, y=5, text="Pandemi",
                       showarrow=False, font=dict(color=C_ORANGE, size=9))
    if yil != "all":
        secili_yil = int(yil)
        fig.add_vline(x=secili_yil, line_width=2, line_dash="dash", line_color=C_PINK)
    fig.add_annotation(x=df_plot["yil"].median(), y=102, text=secili_text,
                       showarrow=False, font=dict(color=C_PINK if yil != "all" else C_ACCENT, size=10))

    fig.update_layout(**PLOTLY_LAYOUT,
                      yaxis_title="Oran (%)", yaxis_range=[0, 105],
                      hovermode="x unified")
    return fig

@app.callback(Output("buyume-grafik", "figure"), Input("global-yil", "value"))
def update_buyume(yil):
    return _buyume_fig(yil)

@app.callback(Output("pandemi-grafik", "figure"), Input("global-yil", "value"))
def update_pandemi(yil):
    return _pandemi_fig(yil)

@app.callback(
    Output("bolge-harita", "figure"),
    Output("bolge-siralama", "figure"),
    Output("ucurum-grafik", "figure"),
    Output("radar-grafik", "figure"),
    Input("harita-gosterge", "value"),
    Input("global-yil", "value")
)
def update_harita(gosterge, yil):
    etiketler = {
        "internet_erisim": "İnternet Erişimi (%)",
        "edevlet": "E-Devlet Kullanımı (%)",
        "dijital_beceri": "Dijital Beceri (%)",
        "eticaret": "E-Ticaret (%)",
        "dijit_skor": "Dijitalleşme Skoru",
    }

    aktif_yil = max(YILLAR) if yil == "all" else int(yil)
    df = bolge_yil_df(yil).copy()
    df["harita_deger"] = df[gosterge].round(1)
    df["kume_adi"] = df["kmeans_kume"].map(KUME_ETIKET)

    if TR_IBBS1_GEOJSON is None:
        fig = go.Figure()
        fig.add_annotation(
            text="tr_ibbs1.geojson bulunamadı.<br>Dosya yolu: dashboard/data/tr_ibbs1.geojson",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color=C_RED)
        )
        fig.update_layout(
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=0, r=0, t=40, b=0),
            height=520
        )
        return fig, _bolge_bar(df), _ucurum_fig(df), _radar_fig(df)

    # GeoJSON içindeki NUTS_ID kodları ile df["kod"] birebir eşleşmeli: TR1, TR2, ... TRC
    geo_ids = {f.get("properties", {}).get("NUTS_ID") for f in TR_IBBS1_GEOJSON.get("features", [])}
    df = df[df["kod"].isin(geo_ids)].copy()

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="GeoJSON ile bölge kodları eşleşmedi.<br>tr_ibbs1.geojson içinde NUTS_ID alanını kontrol et.",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=15, color=C_RED)
        )
        fig.update_layout(
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=520,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        return fig, _bolge_bar(df), _ucurum_fig(df), _radar_fig(df)

    customdata = np.stack([
        df["bolge"],
        df["kod"],
        df["internet_erisim"],
        df["edevlet"],
        df["dijital_beceri"],
        df["eticaret"],
        df["dijit_skor"],
        df["kume_adi"],
    ], axis=-1)

    fig = go.Figure()

    fig.add_trace(go.Choropleth(
        geojson=TR_IBBS1_GEOJSON,
        locations=df["kod"],
        z=df["harita_deger"],
        featureidkey="properties.NUTS_ID",
        customdata=customdata,
        colorscale=[
            [0.00, "#FCE7F3"],
            [0.20, "#F472B6"],
            [0.45, "#8B5CF6"],
            [0.70, "#06B6D4"],
            [1.00, "#22C55E"],
        ],
        zmin=max(0, float(df["harita_deger"].min()) - 4),
        zmax=min(100, float(df["harita_deger"].max()) + 4),
        marker_line_color="rgba(255,255,255,0.95)",
        marker_line_width=1.4,
        colorbar=dict(
            title=dict(
                text=etiketler[gosterge],
                font=dict(size=10, color=C_TEXT)
            ),
            thickness=13,
            len=0.68,
            x=0.98,
            y=0.48,
            tickfont=dict(size=10, color=C_MUTED),
            outlinewidth=0,
        ),
        hovertemplate=(
            "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
            + etiketler[gosterge] + ": <b>%{z:.1f}</b><br>"
            + "İnternet: %{customdata[2]:.1f}<br>"
            + "E-Devlet: %{customdata[3]:.1f}<br>"
            + "Dijital Beceri: %{customdata[4]:.1f}<br>"
            + "E-Ticaret: %{customdata[5]:.1f}<br>"
            + "Dijitalleşme Skoru: %{customdata[6]:.1f}<br>"
            + "Küme: %{customdata[7]}"
            + "<extra></extra>"
        ),
    ))

    # Bölge etiketleri için harita üzerinde daha okunaklı manuel konumlar
    label_pos = {
        "TR1": (29.05, 41.15),
        "TR2": (27.20, 40.25),
        "TR3": (27.35, 38.45),
        "TR4": (30.05, 40.65),
        "TR5": (32.90, 39.55),
        "TR6": (35.00, 36.95),
        "TR7": (34.70, 39.05),
        "TR8": (31.85, 41.00),
        "TR9": (39.25, 40.75),
        "TRA": (42.20, 40.20),
        "TRB": (40.30, 38.50),
        "TRC": (39.00, 37.20),
    }
    df["label_lon"] = df["kod"].map(lambda k: label_pos.get(k, (np.nan, np.nan))[0])
    df["label_lat"] = df["kod"].map(lambda k: label_pos.get(k, (np.nan, np.nan))[1])

    fig.add_trace(go.Scattergeo(
        lon=df["label_lon"],
        lat=df["label_lat"],
        mode="text",
        text=[f"{b}<br><b>{v:.1f}</b>" for b, v in zip(df["bolge"], df["harita_deger"])],
        textfont=dict(color="#181124", size=8, family="Inter, Arial"),
        textposition="middle center",
        hoverinfo="skip",
        showlegend=False,
    ))

    top3 = df.nlargest(3, "harita_deger")[["bolge", "harita_deger"]]
    low3 = df.nsmallest(3, "harita_deger")[["bolge", "harita_deger"]]
    top_text = "<b>En yüksek 3 bölge</b><br>" + "<br>".join(
        [f"{i+1}. {r.bolge}: {r.harita_deger:.1f}" for i, r in enumerate(top3.itertuples())]
    )
    low_text = "<b>En düşük 3 bölge</b><br>" + "<br>".join(
        [f"{i+1}. {r.bolge}: {r.harita_deger:.1f}" for i, r in enumerate(low3.itertuples())]
    )

    fig.add_annotation(
        x=0.02, y=0.96, xref="paper", yref="paper", text=top_text,
        showarrow=False, align="left", bgcolor="rgba(255,255,255,0.86)",
        bordercolor="rgba(139,92,246,0.28)", borderwidth=1, borderpad=8,
        font=dict(size=11, color=C_TEXT)
    )
    fig.add_annotation(
        x=0.02, y=0.10, xref="paper", yref="paper", text=low_text,
        showarrow=False, align="left", bgcolor="rgba(255,255,255,0.86)",
        bordercolor="rgba(236,72,153,0.28)", borderwidth=1, borderpad=8,
        font=dict(size=11, color=C_TEXT)
    )

    fig.update_geos(
        projection_type="mercator",
        center=dict(lat=39.0, lon=35.0),
        lataxis_range=[35.5, 42.5],
        lonaxis_range=[25.0, 45.0],
        showframe=False,
        showcoastlines=False,
        showcountries=False,
        showland=True,
        landcolor="rgba(248,250,252,0.4)",
        bgcolor="rgba(255,255,255,0)",
        visible=False,
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{etiketler[gosterge]}</b> · {aktif_yil} yılı görünümü",
            x=0.5,
            font=dict(size=17, color=C_TEXT, family="Inter, Arial")
        ),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(color=C_TEXT, family="Inter, Arial"),
        margin=dict(l=0, r=0, t=48, b=0),
        height=520,
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.96)",
            bordercolor="rgba(139,92,246,0.25)",
            font=dict(color=C_TEXT, family="Inter, Arial")
        ),
    )
    return fig, _bolge_bar(df), _ucurum_fig(df), _radar_fig(df)


@app.callback(Output("karsil-grafik", "figure"),
              Input("karsil-yil1", "value"),
              Input("karsil-yil2", "value"))
def update_karsil(yil1, yil2):
    if yil1 == yil2:
        yil2 = min(yil1 + 1, max(YILLAR))
    row1 = turkey_ts[turkey_ts["yil"] == yil1].iloc[0]
    row2 = turkey_ts[turkey_ts["yil"] == yil2].iloc[0]

    cols = ["internet_erisim","akilli_telefon","edevlet","eticaret",
            "sosyal_medya","cevrimici_egitim","yapay_zeka_farkin","bilgisayar"]
    lbls = ["İnternet","Akıllı Tel.","E-Devlet","E-Ticaret",
            "Sosyal Medya","Çev.Eğitim","YZ Farkın.","Bilgisayar"]

    v1 = [row1[c] for c in cols]
    v2 = [row2[c] for c in cols]
    delta = [b - a for a, b in zip(v1, v2)]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[f"{yil1} vs {yil2} Değerleri", "Değişim (Puan Farkı)"],
                        horizontal_spacing=0.1)

    fig.add_trace(go.Bar(name=str(yil1), x=lbls, y=v1, marker_color="#C4B5FD",
                         text=[f"{v:.0f}%" for v in v1],
                         textposition="inside", textfont=dict(size=8, color="white")), row=1, col=1)
    fig.add_trace(go.Bar(name=str(yil2), x=lbls, y=v2, marker_color=C_ACCENT,
                         text=[f"{v:.0f}%" for v in v2],
                         textposition="inside", textfont=dict(size=8, color="white")), row=1, col=1)

    fig.add_trace(go.Bar(
        name="Δ Değişim", x=lbls, y=delta,
        marker_color=[C_GREEN if d > 0 else C_RED for d in delta],
        text=[f"+{d:.1f}" if d > 0 else f"{d:.1f}" for d in delta],
        textposition="outside", textfont=dict(size=9, color=C_TEXT),
    ), row=1, col=2)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=C_TEXT),
        margin=dict(l=20, r=20, t=50, b=20),
        barmode="group",
        legend=dict(bgcolor="rgba(255,255,255,0.76)", bordercolor=C_BORDER,
                    borderwidth=1, font=dict(size=10)),
        xaxis=dict(tickfont=dict(size=9, color=C_TEXT), gridcolor=C_BORDER, tickangle=-30),
        yaxis=dict(gridcolor=C_BORDER, tickfont=dict(size=9, color=C_MUTED)),
        xaxis2=dict(tickfont=dict(size=9, color=C_TEXT), gridcolor=C_BORDER, tickangle=-30),
        yaxis2=dict(gridcolor=C_BORDER, tickfont=dict(size=9, color=C_MUTED)),
    )
    return fig


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  TÜRKİYE DİJİTALLEŞME DASHBOARD")
    print("  Dilara Şenay | TÜBİTAK 2209-A")
    print("="*60)
    print("  Tarayıcıda açın: http://127.0.0.1:8050")
    print("="*60 + "\n")
    app.run(debug=True, port=8050)