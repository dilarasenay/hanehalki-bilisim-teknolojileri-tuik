"""
=============================================================
 TÜRKİYE DİJİTALLEŞME DASHBOARD + YOUTUBE NLP
 Dilara Şenay | TÜBİTAK 2209-A Araştırma Projesi
 Kaynak: TÜİK Hanehalkı Bilişim Teknolojileri Kullanım Araştırması
 NLP Kaynak: YouTube yorumları
=============================================================
Kurulum:
    pip install dash plotly pandas numpy statsmodels

Çalıştırma:
    python app.py

Tarayıcıda aç:
    http://127.0.0.1:8050
=============================================================

Klasör yapısı önerisi:
    app.py
    data/
        bütünveriler.csv
        tr_ibbs1.geojson
    gorseller/
        youtube_nlp_tam.csv
        tfidf_sonuclar.csv
        lda_konular.csv
        temsili_yorumlar.csv
    assets/
        03_kelime_bulutu.png
        06_lda_konular.png
        08_zaman_serisi.png
        09_top_engagement.png

Not:
- YouTube PNG görsellerinin dashboardda görünmesi için ilgili png dosyalarını assets klasörüne kopyalayın.
- YouTube CSV dosyaları gorseller klasöründe yoksa dashboard çökmez, uyarı gösterir.
"""

import json
import os
from collections import Counter

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

try:
    import statsmodels.api as sm
except Exception:
    sm = None


# ─────────────────────────────────────────────────────────────
# 1. GENEL AYARLAR
# ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(__file__)

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

KUME_RENK = {0: "#8B5CF6", 1: "#06B6D4", 2: "#EC4899", 3: "#F97316"}
KUME_ETIKET = {0: "Yüksek Dijitalleşme", 1: "Orta-Yüksek", 2: "Orta", 3: "Düşük Dijitalleşme"}

# Tema renkleri: lacivert yok
C_BG      = "#FBF7FF"
C_BG_2    = "#ECFEFF"
C_CARD    = "rgba(255, 255, 255, 0.88)"
C_BORDER  = "#E9D8FD"
C_TEXT    = "#181124"
C_MUTED   = "#71717A"
C_ACCENT  = "#8B5CF6"
C_ACCENT2 = "#06B6D4"
C_GREEN   = "#22C55E"
C_ORANGE  = "#F97316"
C_PURPLE  = "#A855F7"
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
    "color": C_TEXT,
    "fontSize": "14px",
    "fontWeight": "800",
    "textTransform": "uppercase",
    "letterSpacing": "0.7px",
    "marginBottom": "5px",
    "marginTop": "0",
}

SUBTITLE_STYLE = {
    "color": C_MUTED,
    "fontSize": "11px",
    "marginBottom": "18px",
    "marginTop": "0",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(255,255,255,0)",
    plot_bgcolor="rgba(255,255,255,0)",
    template="plotly_white",
    font=dict(family="Inter, system-ui, sans-serif", color=C_TEXT),
    margin=dict(l=20, r=20, t=35, b=20),
    legend=dict(
        bgcolor="rgba(255,255,255,0.72)",
        bordercolor=C_BORDER,
        borderwidth=1,
        font=dict(size=11, color=C_TEXT),
    ),
    xaxis=dict(gridcolor="rgba(139,92,246,0.14)", linecolor=C_BORDER, zeroline=False, tickfont=dict(color=C_MUTED, size=10)),
    yaxis=dict(gridcolor="rgba(139,92,246,0.14)", linecolor=C_BORDER, zeroline=False, tickfont=dict(color=C_MUTED, size=10)),
)


# ─────────────────────────────────────────────────────────────
# 2. YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────────────────────

def _normalize_col_name(name):
    return str(name).strip().lower().replace("ı", "i")


def _find_col(df, *candidates):
    if df is None or df.empty:
        return None

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
    s = _num(s)
    s = s[s.isin([1, 2])]
    if len(s) == 0:
        return np.nan
    return float((s.eq(1).mean() * 100).round(1))


def kullanim_siklik_orani(s):
    s = _num(s)
    s = s[s.notna()]
    if len(s) == 0:
        return np.nan
    return float((s.isin([1, 2, 3, 4]).mean() * 100).round(1))


def satir_bazli_evet_orani(df, cols):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return np.nan
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    valid = sub.isin([1, 2]).any(axis=1)
    if valid.sum() == 0:
        return np.nan
    return float((sub.eq(1).any(axis=1)[valid].mean() * 100).round(1))


def _safe_indicator(df, col, mode="evet"):
    if df is None or df.empty or col is None or col not in df.columns:
        return np.nan
    if mode == "freq":
        return kullanim_siklik_orani(df[col])
    return evet_orani(df[col])


def fmt_pct(v):
    try:
        if pd.isna(v):
            return "Veri yok"
        return f"%{float(v):.1f}"
    except Exception:
        return "Veri yok"


def fmt_num(v, digits=1):
    try:
        if pd.isna(v):
            return "Veri yok"
        return f"{float(v):.{digits}f}"
    except Exception:
        return "Veri yok"


def fmt_delta(v):
    try:
        if pd.isna(v):
            return "Veri yok"
        sign = "+" if float(v) >= 0 else ""
        return f"↑ {sign}{float(v):.1f} puan"
    except Exception:
        return "Veri yok"


def safe_nanmean(values):
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if vals.empty:
        return np.nan
    return float(vals.mean())


def _empty_fig(message):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, font=dict(size=14, color=C_MUTED))
    layout = {**PLOTLY_LAYOUT}
    layout["xaxis"] = dict(visible=False)
    layout["yaxis"] = dict(visible=False)
    fig.update_layout(**layout)
    return fig


# ─────────────────────────────────────────────────────────────
# 3. TÜİK VERİ YÜKLEME
# ─────────────────────────────────────────────────────────────

def _ibbs_columns(df):
    cols = []
    wanted = [
        "istatistiki bölge birimleri sınıflaması(düzey 1)",
        "istatistiki bölge birimleri sınıflaması (düzey 1)",
    ]

    for col in df.columns:
        norm = _normalize_col_name(col)
        if any(_normalize_col_name(w) == norm for w in wanted):
            cols.append(col)

    for col in df.columns:
        norm = _normalize_col_name(col)
        if "bölge" in norm and "düzey 1" in norm and col not in cols:
            cols.append(col)

    return cols


def _normalize_ibbs1_value(x):
    if pd.isna(x):
        return None

    raw = str(x).strip()
    if raw == "" or raw.lower() in ["nan", "none", "null"]:
        return None

    upper = raw.upper().replace("İ", "I")
    upper = upper.replace("Ğ", "G").replace("Ü", "U").replace("Ş", "S").replace("Ö", "O").replace("Ç", "C")

    for kod in BOLGE_ADLARI.keys():
        if upper == kod or upper.startswith(kod + " ") or upper.startswith(kod + "-"):
            return kod

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
    df = df.copy()

    year_col = _find_col(df, "referans yıl")
    ibbs_cols = _ibbs_columns(df)

    if year_col is None or not ibbs_cols:
        print("UYARI: CSV içinde 'referans yıl' veya İBBS-1 bölge sütunu bulunamadı.")
        return pd.DataFrame()

    df["yil"] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")

    ibbs_raw = pd.Series([None] * len(df), index=df.index, dtype="object")
    for col in ibbs_cols:
        current = df[col].where(df[col].notna(), None).astype("object")
        current = current.where(current.astype(str).str.strip().str.lower().ne("nan"), None)
        current = current.where(current.astype(str).str.strip().ne(""), None)
        ibbs_raw = ibbs_raw.where(ibbs_raw.notna(), current)

    df["ibbs1_raw"] = ibbs_raw
    df["ibbs1"] = df["ibbs1_raw"].apply(_normalize_ibbs1_value)

    df = df[df["yil"].notna() & df["ibbs1"].isin(BOLGE_ADLARI.keys())].copy()
    return df


real_df = load_real_data()

COL_HANE_INTERNET = _find_col(real_df, "hane internet erişim durumu") if not real_df.empty else None
COL_BILGISAYAR = _find_col(real_df, "hane bilgisayar kullanım durumu", "bilgisayar kullanım sıklığı") if not real_df.empty else None
COL_AKILLI = _find_col(real_df, "akıllı telefon kullanımı", "cep telefonu kullanımı", "hanede cep telefonu var mı") if not real_df.empty else None
COL_EDEVLET = _find_col(real_df, "edevlet hakkında bilgi") if not real_df.empty else None
COL_ETICARET = _find_col(real_df, "eticaret kullanımı") if not real_df.empty else None
COL_SOSYAL = _find_col(real_df, "sosyal medya kullanımı") if not real_df.empty else None
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

COL_YAS = _find_col(real_df, "yaş", "yas") if not real_df.empty else None
COL_CINSIYET = _find_col(real_df, "cinsiyet") if not real_df.empty else None
COL_EGITIM = _find_col(real_df, "eğitim", "okul biten", "okul", "egitim") if not real_df.empty else None
COL_GELIR_5LI_STD = _find_col(real_df, "gelir_5li_std") if not real_df.empty else None
COL_GELIR_RAW = _find_col(
    real_df,
    "hanenin aylık toplam net geliri (5'li gelir grubu)",
    "hanenin aylık toplam net geliri",
    "gelir grubu",
    "gelir"
) if not real_df.empty else None


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
            "dijital_beceri": [np.nan],
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
            "dijital_beceri": satir_bazli_evet_orani(g, DIJITAL_BECERI_COLS),
            "cevrimici_egitim": satir_bazli_evet_orani(g, CEVRIMICI_EGITIM_COLS),
            "yapay_zeka_farkin": _safe_indicator(g, COL_YZ),
        })

    out = pd.DataFrame(rows).sort_values("yil")
    metric_cols = [c for c in out.columns if c != "yil"]
    out[metric_cols] = out[metric_cols].round(1)
    return out


turkey_ts = _compute_turkey_ts()
YILLAR = sorted(real_df["yil"].dropna().astype(int).unique().tolist()) if not real_df.empty else [2024]
if not YILLAR:
    YILLAR = [2024]


def _assign_kume(df):
    df = df.sort_values("dijit_skor", ascending=False).reset_index(drop=True)
    n = len(df)
    if n == 0:
        df["kmeans_kume"] = []
        return df
    df["kmeans_kume"] = [min(3, int(i * 4 / n)) for i in range(n)]
    return df


def bolge_yil_df(yil):
    aktif_yil = max(YILLAR) if yil == "all" else int(yil)
    d = real_df[real_df["yil"] == aktif_yil].copy() if not real_df.empty else pd.DataFrame()

    rows = []
    for kod in BOLGE_ADLARI:
        g = d[d["ibbs1"] == kod] if not d.empty else pd.DataFrame()

        if g.empty:
            internet_erisim = edevlet = eticaret = dijital_beceri = np.nan
        else:
            internet_erisim = _safe_indicator(g, COL_HANE_INTERNET)
            edevlet = _safe_indicator(g, COL_EDEVLET)
            eticaret = _safe_indicator(g, COL_ETICARET)
            dijital_beceri = satir_bazli_evet_orani(g, DIJITAL_BECERI_COLS)

        values = [internet_erisim, edevlet, dijital_beceri, eticaret]
        clean_values = [v for v in values if not pd.isna(v)]
        dijit_skor = float(np.mean(clean_values)) if clean_values else np.nan
        values = [0 if pd.isna(v) else v for v in values]
        lat, lon = BOLGE_KOORD[kod]

        rows.append({
            "aktif_yil": aktif_yil,
            "kod": kod,
            "bolge": BOLGE_ADLARI[kod],
            "internet_erisim": round(values[0], 1),
            "edevlet": round(values[1], 1),
            "dijital_beceri": round(values[2], 1),
            "eticaret": round(values[3], 1),
            "dijit_skor": round(dijit_skor, 1) if not pd.isna(dijit_skor) else 0,
            "lat": lat,
            "lon": lon,
        })

    return _assign_kume(pd.DataFrame(rows))


bolge_data = bolge_yil_df("all")


def load_tr_ibbs1_geojson():
    geo_path = os.path.join(BASE_DIR, "data", "tr_ibbs1.geojson")
    try:
        with open(geo_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"UYARI: tr_ibbs1.geojson okunamadı: {exc}")
        return None


TR_IBBS1_GEOJSON = load_tr_ibbs1_geojson()


def _filtered_df_by_year(yil="all"):
    df = real_df.copy()
    if yil != "all" and "yil" in df.columns:
        df = df[df["yil"] == int(yil)].copy()
    return df


def standardize_income_5li(series):
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return s
    max_val = valid.max()
    if max_val <= 5:
        return s.clip(1, 5)
    if max_val <= 20:
        return np.ceil(s / 4).clip(1, 5)
    try:
        return pd.qcut(s, q=5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(float)
    except Exception:
        return pd.Series(np.nan, index=s.index)


def education_group(series):
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(index=s.index, dtype="object")
    out[s.isin([1, 2])] = "İlkokul ve Altı"
    out[s.isin([3, 4, 5])] = "Ortaokul/Lise"
    out[s.isin([6, 51, 511, 512])] = "Üniversite"
    out[s.isin([7, 52, 53])] = "Üniversite Üstü"
    return out


def _income_5li_for_df(df):
    if COL_GELIR_5LI_STD and COL_GELIR_5LI_STD in df.columns:
        return pd.to_numeric(df[COL_GELIR_5LI_STD], errors="coerce")
    if COL_GELIR_RAW and COL_GELIR_RAW in df.columns:
        return standardize_income_5li(df[COL_GELIR_RAW])
    return pd.Series(np.nan, index=df.index)


def create_demographic_tables(yil="all"):
    df = _filtered_df_by_year(yil)
    if df.empty or COL_YAS is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = df.copy()
    df["yas_num"] = pd.to_numeric(df[COL_YAS], errors="coerce")
    df["yas_grubu"] = pd.cut(df["yas_num"], bins=[15, 30, 45, 60, 120], labels=["16-30", "31-45", "46-60", "60+"])

    yas_rows = []
    for grup, g in df.groupby("yas_grubu", observed=False):
        yas_rows.append({
            "yas_grubu": str(grup),
            "internet": _safe_indicator(g, COL_HANE_INTERNET),
            "edevlet": _safe_indicator(g, COL_EDEVLET),
            "sosyal": _safe_indicator(g, COL_SOSYAL),
            "eticaret": _safe_indicator(g, COL_ETICARET),
        })
    demo_df = pd.DataFrame(yas_rows)

    if COL_EGITIM:
        df["egitim_grup"] = education_group(df[COL_EGITIM])
        egitim_order = ["İlkokul ve Altı", "Ortaokul/Lise", "Üniversite", "Üniversite Üstü"]
        egitim_rows = []
        for egitim in egitim_order:
            g = df[df["egitim_grup"] == egitim]
            egitim_rows.append({
                "egitim": egitim,
                "internet": _safe_indicator(g, COL_HANE_INTERNET),
                "edevlet": _safe_indicator(g, COL_EDEVLET),
                "dijital_b": satir_bazli_evet_orani(g, DIJITAL_BECERI_COLS),
            })
        egitim_df = pd.DataFrame(egitim_rows)
    else:
        egitim_df = pd.DataFrame()

    gender_rows = []
    if COL_CINSIYET:
        cnum = pd.to_numeric(df[COL_CINSIYET], errors="coerce")
        for kod, ad in {1: "Erkek", 2: "Kadın"}.items():
            g = df[cnum == kod]
            gender_rows.append({
                "cinsiyet": ad,
                "internet": _safe_indicator(g, COL_HANE_INTERNET),
                "edevlet": _safe_indicator(g, COL_EDEVLET),
                "eticaret": _safe_indicator(g, COL_ETICARET),
                "sosyal": _safe_indicator(g, COL_SOSYAL),
                "dijital": satir_bazli_evet_orani(g, DIJITAL_BECERI_COLS),
            })

    return demo_df, egitim_df, pd.DataFrame(gender_rows)


def _row_digital_score(df):
    cols = [COL_HANE_INTERNET, COL_EDEVLET, COL_ETICARET, COL_SOSYAL, COL_AKILLI]
    score_cols = []
    for col in cols:
        if col and col in df.columns:
            score_cols.append(pd.to_numeric(df[col], errors="coerce").map({1: 1, 2: 0}))
    if not score_cols:
        return pd.Series(np.nan, index=df.index)
    return pd.concat(score_cols, axis=1).mean(axis=1)


def calculate_real_ols(yil="all"):
    df = _filtered_df_by_year(yil).copy()
    if df.empty or COL_YAS is None or COL_CINSIYET is None or COL_EGITIM is None:
        return pd.Series(dtype=float)

    df["dijital_skor"] = _row_digital_score(df)
    df["yas_num"] = pd.to_numeric(df[COL_YAS], errors="coerce")
    df["cinsiyet_kadin"] = (pd.to_numeric(df[COL_CINSIYET], errors="coerce") == 2).astype(float)
    df["gelir_num"] = _income_5li_for_df(df)
    df["egitim_grup"] = education_group(df[COL_EGITIM])

    model_df = df[["dijital_skor", "yas_num", "cinsiyet_kadin", "egitim_grup", "gelir_num"]].dropna().copy()
    if len(model_df) < 20:
        return pd.Series(dtype=float)

    edu_order = ["İlkokul ve Altı", "Ortaokul/Lise", "Üniversite", "Üniversite Üstü"]
    model_df["egitim_grup"] = pd.Categorical(model_df["egitim_grup"], categories=edu_order, ordered=True)
    edu_dummies = pd.get_dummies(model_df["egitim_grup"], prefix="egitim", drop_first=True, dtype=float)

    gelir_clean = pd.to_numeric(model_df["gelir_num"], errors="coerce").round().clip(1, 5)
    model_df["gelir_grup"] = pd.Categorical(gelir_clean, categories=[1, 2, 3, 4, 5], ordered=True)
    gelir_dummies = pd.get_dummies(model_df["gelir_grup"], prefix="gelir", drop_first=True, dtype=float)

    X = pd.concat([model_df[["yas_num", "cinsiyet_kadin"]], gelir_dummies, edu_dummies], axis=1)
    y = pd.to_numeric(model_df["dijital_skor"], errors="coerce")
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.notna().any(axis=0)]
    X = X.loc[:, X.std(ddof=0, skipna=True).fillna(0) > 0]
    if X.empty:
        return pd.Series(dtype=float)

    X = (X - X.mean()) / X.std(ddof=0).replace(0, np.nan)
    valid = X.replace([np.inf, -np.inf], np.nan).notna().all(axis=1) & y.notna()
    X = X.loc[valid]
    y = y.loc[valid]
    if len(X) < 20:
        return pd.Series(dtype=float)

    try:
        if sm is not None:
            X_model = sm.add_constant(X, has_constant="add")
            model = sm.OLS(y, X_model).fit()
            params = model.params.drop("const", errors="ignore")
        else:
            X_np = np.column_stack([np.ones(len(X)), X.values.astype(float)])
            beta = np.linalg.lstsq(X_np, y.values.astype(float), rcond=None)[0]
            params = pd.Series(beta[1:], index=X.columns)
    except Exception:
        return pd.Series(dtype=float)

    rename = {
        "yas_num": "Yaş",
        "cinsiyet_kadin": "Kadın",
        "gelir_2": "Gelir 2",
        "gelir_3": "Gelir 3",
        "gelir_4": "Gelir 4",
        "gelir_5": "Gelir 5",
        "egitim_Ortaokul/Lise": "Ortaokul/Lise",
        "egitim_Üniversite": "Üniversite",
        "egitim_Üniversite Üstü": "Üniversite Üstü",
    }
    params.index = [rename.get(i, str(i).replace("egitim_", "")) for i in params.index]
    expected = ["Yaş", "Kadın", "Gelir 2", "Gelir 3", "Gelir 4", "Gelir 5", "Ortaokul/Lise", "Üniversite", "Üniversite Üstü"]
    return params.reindex(expected).fillna(0)


def yz_beceri_tables(yil="all"):
    df = _filtered_df_by_year(yil)
    if df.empty:
        return pd.DataFrame(columns=["yil", "yz", "beceri"])

    rows = []
    for yil_degeri, g in df.groupby("yil"):
        rows.append({
            "yil": int(yil_degeri),
            "yz": _safe_indicator(g, COL_YZ),
            "beceri": satir_bazli_evet_orani(g, DIJITAL_BECERI_COLS),
        })
    return pd.DataFrame(rows).sort_values("yil")


# ─────────────────────────────────────────────────────────────
# 4. YOUTUBE NLP VERİ YÜKLEME VE GRAFİKLER
# ─────────────────────────────────────────────────────────────

# Dashboard app.py dosyası bazen dashboard/ klasörü içinde,
# NLP çıktıları ise proje kökünde gorseller/ klasöründe olabiliyor.
# Bu yüzden hem BASE_DIR/gorseller hem de BASE_DIR/../gorseller aranır.
YOUTUBE_NLP_CANDIDATES = [
    os.path.join(BASE_DIR, "gorseller", "youtube_nlp_tam.csv"),
    os.path.join(BASE_DIR, "..", "gorseller", "youtube_nlp_tam.csv"),
    os.path.join(BASE_DIR, "youtube_nlp_tam.csv"),
    os.path.join(BASE_DIR, "youtube_duygu.csv"),
    os.path.join(BASE_DIR, "..", "youtube_duygu.csv"),
    os.path.join(BASE_DIR, "youtube_temiz.csv"),
    os.path.join(BASE_DIR, "..", "youtube_temiz.csv"),
]

YOUTUBE_TFIDF_CANDIDATES = [
    os.path.join(BASE_DIR, "gorseller", "tfidf_sonuclar.csv"),
    os.path.join(BASE_DIR, "..", "gorseller", "tfidf_sonuclar.csv"),
]
YOUTUBE_LDA_CANDIDATES = [
    os.path.join(BASE_DIR, "gorseller", "lda_konular.csv"),
    os.path.join(BASE_DIR, "..", "gorseller", "lda_konular.csv"),
]
YOUTUBE_REP_CANDIDATES = [
    os.path.join(BASE_DIR, "gorseller", "temsili_yorumlar.csv"),
    os.path.join(BASE_DIR, "..", "gorseller", "temsili_yorumlar.csv"),
]


def read_csv_safe(path):
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()


def _first_existing(paths):
    return next((p for p in paths if os.path.exists(p)), None)


def load_youtube_nlp_data():
    main_path = _first_existing(YOUTUBE_NLP_CANDIDATES)
    tfidf_path = _first_existing(YOUTUBE_TFIDF_CANDIDATES)
    lda_path = _first_existing(YOUTUBE_LDA_CANDIDATES)
    rep_path = _first_existing(YOUTUBE_REP_CANDIDATES)

    df = read_csv_safe(main_path) if main_path else pd.DataFrame()
    tfidf_df = read_csv_safe(tfidf_path) if tfidf_path else pd.DataFrame()
    lda_df = read_csv_safe(lda_path) if lda_path else pd.DataFrame()
    rep_df = read_csv_safe(rep_path) if rep_path else pd.DataFrame()
    return df, tfidf_df, lda_df, rep_df


def youtube_kpi_cards(df):
    toplam = len(df)
    temiz = df["temiz_yorum"].notna().sum() if "temiz_yorum" in df.columns else toplam

    if "duygu" in df.columns and toplam > 0:
        duygu_counts = df["duygu"].value_counts(normalize=True) * 100
        pozitif = duygu_counts.get("Pozitif", 0)
        negatif = duygu_counts.get("Negatif", 0)
        notr = duygu_counts.get("Nötr", 0)
    else:
        pozitif = negatif = notr = 0

    cards = [
        ("Toplam Yorum", f"{toplam:,}".replace(",", "."), "YouTube’dan toplanan yorum", C_ACCENT),
        ("Temiz Yorum", f"{temiz:,}".replace(",", "."), "Analize giren yorum", C_ACCENT2),
        ("Pozitif", f"%{pozitif:.1f}", "Olumlu algı", C_GREEN),
        ("Negatif", f"%{negatif:.1f}", "Olumsuz algı", C_RED),
        ("Nötr", f"%{notr:.1f}", "Bilgilendirici / yorumsal", C_ORANGE),
    ]

    return html.Div([
        html.Div(kpi_card(baslik=t, deger=d, alt=a, trend="YouTube NLP çıktısı", renk=r), style={"flex": "1", "minWidth": "180px"})
        for t, d, a, r in cards
    ], style={"display": "flex", "gap": "14px", "flexWrap": "wrap", "marginBottom": "14px"})


def youtube_sentiment_fig(df):
    if df.empty or "duygu" not in df.columns:
        return _empty_fig("Duygu analizi verisi bulunamadı.")

    counts = df["duygu"].value_counts().reset_index()
    counts.columns = ["duygu", "sayi"]
    color_map = {"Pozitif": C_GREEN, "Negatif": C_RED, "Nötr": C_ORANGE}

    fig = go.Figure(go.Bar(
        x=counts["duygu"],
        y=counts["sayi"],
        marker_color=[color_map.get(x, C_ACCENT) for x in counts["duygu"]],
        text=counts["sayi"],
        textposition="outside",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, yaxis_title="Yorum Sayısı")
    return fig


def youtube_keyword_sentiment_fig(df):
    if df.empty or "keyword" not in df.columns or "duygu" not in df.columns:
        return _empty_fig("Anahtar kelimeye göre duygu verisi bulunamadı.")

    pivot = df.groupby(["keyword", "duygu"]).size().reset_index(name="sayi")
    fig = px.bar(
        pivot,
        x="keyword",
        y="sayi",
        color="duygu",
        barmode="group",
        color_discrete_map={"Pozitif": C_GREEN, "Negatif": C_RED, "Nötr": C_ORANGE},
    )
    fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="", yaxis_title="Yorum Sayısı")
    return fig


def youtube_word_freq_fig(df):
    if df.empty or "temiz_yorum" not in df.columns:
        return _empty_fig("Kelime frekansı için temiz yorum verisi bulunamadı.")

    words = " ".join(df["temiz_yorum"].dropna().astype(str)).split()
    freq = pd.DataFrame(Counter(words).most_common(25), columns=["kelime", "frekans"])

    if freq.empty:
        return _empty_fig("Kelime frekansı hesaplanamadı.")

    fig = go.Figure(go.Bar(
        x=freq["frekans"],
        y=freq["kelime"],
        orientation="h",
        marker_color=C_ACCENT,
        text=freq["frekans"],
        textposition="outside",
    ))
    layout = {**PLOTLY_LAYOUT}
    layout["yaxis"] = dict(autorange="reversed", tickfont=dict(color=C_TEXT, size=10))
    fig.update_layout(**layout)
    return fig


def youtube_tfidf_fig(tfidf_df):
    if tfidf_df.empty:
        return _empty_fig("TF-IDF çıktısı bulunamadı.")

    term_col = "terim" if "terim" in tfidf_df.columns else tfidf_df.columns[0]
    score_col = "tfidf_ortalama" if "tfidf_ortalama" in tfidf_df.columns else tfidf_df.columns[-1]
    data = tfidf_df.head(20).copy()

    fig = go.Figure(go.Bar(
        x=data[score_col],
        y=data[term_col],
        orientation="h",
        marker_color=C_ACCENT2,
        text=[f"{v:.3f}" for v in data[score_col]],
        textposition="outside",
    ))
    layout = {**PLOTLY_LAYOUT}
    layout["yaxis"] = dict(autorange="reversed", tickfont=dict(color=C_TEXT, size=10))
    fig.update_layout(**layout)
    return fig


def youtube_lda_table(lda_df):
    if lda_df.empty:
        return html.Div("LDA konu çıktısı bulunamadı.", style={"color": C_MUTED})

    show_cols = [c for c in ["konu_no", "kelimeler", "agirliklar"] if c in lda_df.columns]
    if not show_cols:
        show_cols = lda_df.columns[:3].tolist()

    return html.Div([
        html.Table([
            html.Thead(html.Tr([
                html.Th(col, style={"textAlign": "left", "padding": "10px", "color": C_TEXT})
                for col in show_cols
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(str(row[col])[:220], style={
                        "padding": "10px",
                        "borderTop": f"1px solid {C_BORDER}",
                        "fontSize": "12px",
                        "color": C_TEXT,
                    })
                    for col in show_cols
                ])
                for _, row in lda_df.iterrows()
            ])
        ], style={"width": "100%", "borderCollapse": "collapse"})
    ], style={"maxHeight": "340px", "overflowY": "auto"})


def youtube_comments_table(df, title):
    if df.empty:
        return html.Div(f"{title} bulunamadı.", style={**CARD_STYLE, "color": C_MUTED})

    cols = [c for c in ["duygu", "yorum", "begeni", "keyword"] if c in df.columns]
    if not cols:
        cols = df.columns[:4].tolist()

    data = df.copy()
    if "begeni" in data.columns:
        data = data.sort_values("begeni", ascending=False)
    data = data.head(10)

    return html.Div([
        html.P(title, style=TITLE_STYLE),
        html.Table([
            html.Thead(html.Tr([
                html.Th(col, style={"textAlign": "left", "padding": "10px", "color": C_TEXT})
                for col in cols
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(str(row[col])[:230], style={
                        "padding": "10px",
                        "borderTop": f"1px solid {C_BORDER}",
                        "fontSize": "12px",
                        "color": C_TEXT,
                    })
                    for col in cols
                ])
                for _, row in data.iterrows()
            ])
        ], style={"width": "100%", "borderCollapse": "collapse"})
    ], style={**CARD_STYLE, "overflowX": "auto", "flex": "1", "minWidth": "360px"})


def youtube_image_card(title, filename):
    asset_path = os.path.join(BASE_DIR, "assets", filename)
    if not os.path.exists(asset_path):
        return html.Div()

    return html.Div([
        html.P(title, style=TITLE_STYLE),
        html.Img(src=f"/assets/{filename}", style={
            "width": "100%",
            "borderRadius": "18px",
            "border": f"1px solid {C_BORDER}",
        })
    ], style={**CARD_STYLE, "flex": "1", "minWidth": "360px"})


# ─────────────────────────────────────────────────────────────
# 5. DASH UYGULAMASI
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


def mini_trend_fig(col, renk):
    fig = go.Figure()
    if col in turkey_ts.columns:
        fig.add_trace(go.Scatter(
            x=turkey_ts["yil"],
            y=turkey_ts[col],
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


def kpi_card(baslik, deger, alt, trend, renk=C_ACCENT, seri=None):
    return html.Div([
        html.Div([
            html.Div([
                html.Div(baslik.upper(), style={
                    "color": C_TEXT,
                    "fontSize": "11px",
                    "fontWeight": "850",
                    "letterSpacing": "0.3px",
                    "marginBottom": "7px",
                }),
                html.Div(deger, style={
                    "fontSize": "30px",
                    "fontWeight": "950",
                    "color": renk,
                    "lineHeight": "1",
                    "marginBottom": "7px",
                }),
                html.Div(alt, style={"color": C_MUTED, "fontSize": "10px", "fontWeight": "650", "lineHeight": "1.35"}),
                html.Div(trend, style={
                    "color": C_GREEN,
                    "fontSize": "10px",
                    "fontWeight": "850",
                    "marginTop": "10px",
                }),
            ], style={"flex": "1"}),
            html.Div([
                html.Div(baslik[:1].upper(), style={
                    "width": "38px",
                    "height": "38px",
                    "borderRadius": "13px",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "background": f"linear-gradient(135deg, {renk}22, {renk}44)",
                    "color": renk,
                    "fontWeight": "900",
                    "fontSize": "17px",
                    "border": f"1px solid {renk}33",
                }),
            ], style={"display": "flex", "alignItems": "flex-start"}),
        ], style={"display": "flex", "justifyContent": "space-between", "gap": "10px"}),
        dcc.Graph(figure=mini_trend_fig(seri, renk), config={"displayModeBar": False}, style={"height": "48px", "marginTop": "6px"}) if seri else html.Div(),
    ], style={
        **CARD_STYLE,
        "padding": "16px 18px 12px",
        "minHeight": "138px",
        "borderTop": f"5px solid {renk}",
        "background": f"radial-gradient(circle at 85% 0%, {renk}26, transparent 34%), linear-gradient(180deg, rgba(255,255,255,0.94), rgba(255,255,255,0.74))",
        "boxShadow": f"0 22px 48px {renk}22",
    })


def make_kpi_row(yil):
    tumu = yil == "all"
    aktif_yil = max(YILLAR) if tumu else int(yil)
    row = turkey_ts[turkey_ts["yil"] == aktif_yil].iloc[0]
    base = turkey_ts[turkey_ts["yil"] == min(YILLAR)].iloc[0]

    cards = [
        ("İnternet Erişimi", "internet_erisim", C_ACCENT, "Hanehalkı erişimi"),
        ("E-Devlet Kullanımı", "edevlet", C_ACCENT2, "Kamu hizmetleri"),
        ("Dijital Beceri", "dijital_beceri", C_GREEN, "Temel dijital yetkinlik"),
        ("E-Ticaret Kullanımı", "eticaret", C_ORANGE, "Son 12 ay kullanımı"),
        ("YZ Farkındalığı", "yapay_zeka_farkin", C_PURPLE, "Kullanım / farkındalık"),
    ]

    return html.Div([
        html.Div(
            kpi_card(
                baslik=title,
                deger=fmt_pct(row[col]),
                alt=(f"{min(YILLAR)}–{max(YILLAR)} genel dönem · {desc}" if tumu else f"{aktif_yil} yılı · {desc}"),
                trend=f"{fmt_delta(row[col] - base[col])} ({min(YILLAR)} → {aktif_yil})",
                renk=color,
                seri=col,
            ),
            style={"flex": "1", "minWidth": "185px"}
        )
        for title, col, color, desc in cards
    ], style={"display": "flex", "gap": "14px", "flexWrap": "wrap", "marginBottom": "14px"})


def make_hero_row(yil):
    tumu = yil == "all"
    aktif_yil = max(YILLAR) if tumu else int(yil)
    row = turkey_ts[turkey_ts["yil"] == aktif_yil].iloc[0]
    base = turkey_ts[turkey_ts["yil"] == min(YILLAR)].iloc[0]

    score_cols = ["internet_erisim", "edevlet", "eticaret", "dijital_beceri"]
    score_now = safe_nanmean([row[c] for c in score_cols])
    score_base = safe_nanmean([base[c] for c in score_cols])
    delta = score_now - score_base if not pd.isna(score_now) and not pd.isna(score_base) else np.nan
    label = "Genel dönem görünümü" if tumu else f"{aktif_yil} yılı görünümü"
    caption = f"{min(YILLAR)}–{max(YILLAR)} dönemi CSV’den hesaplandı" if tumu else f"{min(YILLAR)} yılına göre {aktif_yil} seviyesi"

    def pill(text, color, bg):
        return html.Span(text, style={
            "background": bg,
            "color": color,
            "padding": "8px 12px",
            "borderRadius": "999px",
            "fontSize": "11px",
            "fontWeight": "900",
            "letterSpacing": "0.2px",
            "whiteSpace": "nowrap",
        })

    return html.Div([
        html.Div([
            html.Div("DİJİTAL DÖNÜŞÜM SKORU", style={
                "fontSize": "12px",
                "fontWeight": "950",
                "letterSpacing": "1px",
                "color": "rgba(255,255,255,.92)",
                "marginBottom": "10px",
            }),
            html.Div([
                html.Span(fmt_num(score_now), style={"fontSize": "52px", "fontWeight": "950", "lineHeight": "0.95", "color": "white"}),
                html.Span(" / 100", style={"fontSize": "22px", "fontWeight": "850", "color": "rgba(255,255,255,.72)", "marginLeft": "4px"}),
            ], style={"display": "flex", "alignItems": "baseline"}),
            html.Div(label, style={"fontSize": "15px", "fontWeight": "850", "color": "white", "marginTop": "9px"}),
            html.Div(caption, style={"fontSize": "12px", "fontWeight": "650", "color": "rgba(255,255,255,.78)", "marginTop": "4px"}),
        ], style={
            "flex": "1.15",
            "minWidth": "280px",
            "padding": "28px 30px",
            "borderRadius": "28px",
            "background": "linear-gradient(135deg,#7C3AED,#8B5CF6,#06B6D4)",
            "boxShadow": "0 28px 60px rgba(124,58,237,.28)",
        }),

        html.Div([
            html.Div("DÖNEM DEĞİŞİMİ", style={"fontSize": "12px", "fontWeight": "950", "letterSpacing": ".6px", "color": C_MUTED, "marginBottom": "8px"}),
            html.Div(fmt_delta(delta).replace("↑ ", ""), style={"fontSize": "28px", "fontWeight": "950", "color": C_GREEN}),
            html.Div(f"{min(YILLAR)} → {aktif_yil}", style={"fontSize": "12px", "fontWeight": "750", "color": C_MUTED}),
            html.Div(style={"height": "9px", "borderRadius": "999px", "background": "rgba(139,92,246,0.12)", "marginTop": "14px", "overflow": "hidden"}, children=[
                html.Div(style={
                    "width": f"{0 if pd.isna(score_now) else min(score_now,100):.0f}%",
                    "height": "100%",
                    "background": "linear-gradient(90deg,#8B5CF6,#EC4899,#06B6D4)",
                    "borderRadius": "999px",
                })
            ]),
        ], style={
            "flex": ".75",
            "minWidth": "250px",
            "background": "rgba(255,255,255,0.78)",
            "border": "1px solid rgba(139,92,246,0.14)",
            "borderRadius": "28px",
            "padding": "24px 26px",
            "boxShadow": "0 20px 45px rgba(139,92,246,.10)",
        }),

        html.Div([
            pill("Nicel veri: TÜİK", "#6D28D9", "rgba(139,92,246,0.12)"),
            pill("Nitel veri: YouTube NLP", "#DB2777", "rgba(236,72,153,0.12)"),
            pill("Odak alanı: Bölgesel uçurum", "#08788F", "rgba(6,182,212,0.12)"),
        ], style={"display": "flex", "gap": "10px", "alignItems": "stretch", "flexDirection": "column", "justifyContent": "center", "flex": ".9", "minWidth": "260px"}),
    ], style={
        "display": "flex",
        "gap": "18px",
        "alignItems": "stretch",
        "justifyContent": "space-between",
        "padding": "18px",
        "marginBottom": "18px",
        "borderRadius": "32px",
        "background": "linear-gradient(135deg, rgba(255,255,255,.74), rgba(255,255,255,.52))",
        "backdropFilter": "blur(20px)",
        "border": "1px solid rgba(255,255,255,.72)",
        "boxShadow": "0 24px 60px rgba(139,92,246,.14)",
        "overflow": "hidden",
    })


# ─────────────────────────────────────────────────────────────
# 6. TAB İÇERİKLERİ
# ─────────────────────────────────────────────────────────────

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
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df["yil"],
                    y=df[col],
                    name=label if dash_type == "solid" else None,
                    line=dict(color=color, width=2, dash=dash_type),
                    showlegend=dash_type == "solid",
                    mode="lines+markers" if dash_type == "dot" else "lines",
                    marker=dict(size=5),
                ))

    fig.add_vrect(
        x0=2019.5,
        x1=2021.5,
        fillcolor="rgba(249,115,22,0.11)",
        line_width=1,
        line_color=C_ORANGE,
        annotation_text="Pandemi",
        annotation_position="top left",
        annotation_font=dict(color=C_ORANGE, size=10),
    )
    if yil != "all":
        secili_yil = int(yil)
        fig.add_vline(x=secili_yil, line_width=2, line_dash="dash", line_color=C_PINK)
    fig.update_layout(**PLOTLY_LAYOUT, showlegend=True)
    return fig


def _buyume_fig(yil="all"):
    cols = ["internet_erisim", "edevlet", "dijital_beceri", "eticaret", "yapay_zeka_farkin"]
    labels = ["İnternet", "E-Devlet", "Dijital Beceri", "E-Ticaret", "YZ Farkındalığı"]
    colors = [C_ACCENT, C_ACCENT2, C_GREEN, C_ORANGE, C_PINK]

    values = []
    title_suffix = "Seçili dönem ortalama yıllık değişim"

    if yil == "all":
        for c in cols:
            vals = pd.to_numeric(turkey_ts[c], errors="coerce").values
            growth = []
            for i in range(1, len(vals)):
                prev_val, cur_val = vals[i - 1], vals[i]
                if pd.isna(prev_val) or prev_val == 0 or pd.isna(cur_val):
                    continue
                growth.append((cur_val - prev_val) / prev_val * 100)
            values.append(float(np.mean(growth)) if growth else 0)
    else:
        secili_yil = int(yil)
        idxs = turkey_ts.index[turkey_ts["yil"] == secili_yil]
        if len(idxs) == 0 or idxs[0] == 0:
            values = [0 for _ in cols]
            title_suffix = f"{secili_yil} için önceki yıl yok"
        else:
            idx = idxs[0]
            prev, cur = turkey_ts.iloc[idx - 1], turkey_ts.iloc[idx]
            for c in cols:
                prev_val, cur_val = prev[c], cur[c]
                if pd.isna(prev_val) or prev_val == 0 or pd.isna(cur_val):
                    values.append(0)
                else:
                    values.append((cur_val - prev_val) / prev_val * 100)
            title_suffix = f"{secili_yil-1} → {secili_yil} yıllık değişim"

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color=C_TEXT, size=11, family="Inter"),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, yaxis_title="Yıllık Büyüme (%)", title=dict(text=title_suffix, font=dict(size=11, color=C_MUTED), x=0.02))
    return fig


def _desi_fig():
    ulkeler = ["Finlandiya", "Danimarka", "Hollanda", "İsveç", "AB Ortalaması", "Polonya", "Romanya", "Türkiye", "Bulgaristan"]
    skorlar = [79.1, 77.8, 76.4, 75.9, 55.3, 47.2, 38.8, 36.4, 35.1]
    renkler = [C_ACCENT if u != "Türkiye" and u != "AB Ortalaması" else (C_ORANGE if u == "Türkiye" else C_GREEN) for u in ulkeler]

    fig = go.Figure(go.Bar(
        x=skorlar,
        y=ulkeler,
        orientation="h",
        marker_color=renkler,
        text=[f"{s}" for s in skorlar],
        textposition="inside",
        textfont=dict(color="white", size=10),
    ))
    layout = {**PLOTLY_LAYOUT}
    layout["xaxis"] = dict(title="DESI Skoru (0-100)", gridcolor=C_BORDER, tickfont=dict(color=C_MUTED, size=10))
    layout["yaxis"] = dict(gridcolor=C_BORDER, linecolor=C_BORDER, tickfont=dict(color=C_TEXT, size=10))
    fig.update_layout(**layout)
    return fig


def genel_bakis():
    return html.Div([
        html.Div(id="hero-row"),
        html.Div(id="kpi-row"),

        html.Div([
            html.Div([
                html.P("ZAMAN SERİSİ ANALİZİ", style=TITLE_STYLE),
                html.P("CSV’den hesaplanan temel BİT göstergeleri trendi", style=SUBTITLE_STYLE),
                dcc.Dropdown(
                    id="ts-gostergeler",
                    options=[
                        {"label": "İnternet Erişimi", "value": "internet_erisim"},
                        {"label": "Dijital Beceri", "value": "dijital_beceri"},
                        {"label": "E-Devlet", "value": "edevlet"},
                        {"label": "E-Ticaret", "value": "eticaret"},
                        {"label": "Sosyal Medya", "value": "sosyal_medya"},
                        {"label": "Çevrimiçi Eğitim", "value": "cevrimici_egitim"},
                        {"label": "YZ Farkındalığı", "value": "yapay_zeka_farkin"},
                        {"label": "Bilgisayar", "value": "bilgisayar"},
                    ],
                    value=["internet_erisim", "edevlet", "eticaret", "yapay_zeka_farkin"],
                    multi=True,
                    style={"fontSize": "12px", "marginBottom": "12px"},
                ),
                dcc.Graph(id="ts-grafik", style={"height": "340px"}),
            ], style={**CARD_STYLE, "flex": "3"}),

            html.Div([
                html.P("PANDEMİ ETKİSİ ANALİZİ", style=TITLE_STYLE),
                html.P("COVID-19 dijital dönüşüm hızlanması", style=SUBTITLE_STYLE),
                dcc.Graph(id="pandemi-grafik", figure=_pandemi_fig("all"), style={"height": "386px"}),
            ], style={**CARD_STYLE, "flex": "2"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        html.Div([
            html.Div([
                html.P("YILLIK BÜYÜME HIZI (%)", style=TITLE_STYLE),
                html.P("Gösterge bazında yıllık değişim oranları", style=SUBTITLE_STYLE),
                dcc.Graph(id="buyume-grafik", figure=_buyume_fig("all"), style={"height": "260px"}),
            ], style={**CARD_STYLE, "flex": "2"}),
            html.Div([
                html.P("ULUSLARARASI KARŞILAŞTIRMA", style=TITLE_STYLE),
                html.P("DESI 2024 – Türkiye vs AB ülkeleri", style=SUBTITLE_STYLE),
                dcc.Graph(figure=_desi_fig(), style={"height": "260px"}),
            ], style={**CARD_STYLE, "flex": "3"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
    ])


def _bolge_bar(df=None):
    if df is None:
        df = bolge_data
    srtd = df.sort_values("dijit_skor", ascending=True)
    colors = [KUME_RENK.get(k, C_ACCENT) for k in srtd["kmeans_kume"]]

    fig = go.Figure(go.Bar(
        x=srtd["dijit_skor"],
        y=srtd["bolge"],
        orientation="h",
        marker_color=colors,
        text=[f"{v}" for v in srtd["dijit_skor"]],
        textposition="inside",
        textfont=dict(color="white", size=9),
    ))
    layout = {**PLOTLY_LAYOUT}
    layout["margin"] = dict(l=100, r=5, t=5, b=5)
    layout["xaxis"] = dict(range=[0, 100], gridcolor=C_BORDER, tickfont=dict(color=C_MUTED, size=8))
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

    fig = go.Figure()
    fig.add_trace(go.Bar(name="En Yüksek Bölge", x=etiketler, y=max_vals, marker_color=C_ACCENT, text=[f"{v:.1f}%" for v in max_vals], textposition="outside"))
    fig.add_trace(go.Bar(name="En Düşük Bölge", x=etiketler, y=min_vals, marker_color=C_RED, text=[f"{v:.1f}%" for v in min_vals], textposition="inside"))
    fig.update_layout(**PLOTLY_LAYOUT, barmode="overlay", yaxis_title="%")
    return fig


def _radar_fig(df=None):
    if df is None:
        df = bolge_data

    kategoriler = ["İnternet\nErişimi", "E-Devlet", "Dijital\nBeceri", "E-Ticaret"]
    fig = go.Figure()

    for kume_id in range(4):
        grup = df[df["kmeans_kume"] == kume_id]
        if len(grup) == 0:
            continue
        vals = [
            grup["internet_erisim"].mean(),
            grup["edevlet"].mean(),
            grup["dijital_beceri"].mean(),
            grup["eticaret"].mean(),
        ]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=kategoriler + [kategoriler[0]],
            name=KUME_ETIKET[kume_id],
            line_color=KUME_RENK[kume_id],
            fill="toself",
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=C_BORDER, tickfont=dict(color=C_MUTED, size=8)),
            angularaxis=dict(gridcolor=C_BORDER, tickfont=dict(color=C_TEXT, size=9)),
        ),
        font=dict(family="Inter, sans-serif", color=C_TEXT),
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(bgcolor="rgba(255,255,255,0.76)", bordercolor=C_BORDER, borderwidth=1, font=dict(size=9, color=C_TEXT)),
        showlegend=True,
    )
    return fig


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
                        {"label": " Dijital Beceri", "value": "dijital_beceri"},
                        {"label": " E-Ticaret", "value": "eticaret"},
                        {"label": " Dijitalleşme Skoru", "value": "dijit_skor"},
                    ],
                    value="dijit_skor",
                    labelStyle={"display": "block", "marginBottom": "8px", "color": C_TEXT, "fontSize": "12px", "cursor": "pointer"},
                ),
                html.Hr(style={"borderColor": C_BORDER, "margin": "16px 0"}),
                html.P("BÖLGE SIRALAMALARI", style=TITLE_STYLE),
                dcc.Graph(id="bolge-siralama", style={"height": "300px"}, config={"displayModeBar": False}),
            ], style={**CARD_STYLE, "width": "260px", "flexShrink": "0"}),

            html.Div([
                html.P("TÜRKİYE BÖLGESEL DİJİTALLEŞME HARİTASI", style=TITLE_STYLE),
                html.P("İBBS-1 bölgeleri · seçili göstergeye ve yıla göre renklendirilmiş gerçek Türkiye haritası", style=SUBTITLE_STYLE),
                dcc.Graph(id="bolge-harita", style={"height": "480px"}),
            ], style={**CARD_STYLE, "flex": "1"}),
        ], style={"display": "flex", "gap": "16px", "alignItems": "flex-start"}),

        html.Div([
            html.Div([
                html.P("BÖLGESEL UÇURUM ANALİZİ", style=TITLE_STYLE),
                dcc.Graph(id="ucurum-grafik", style={"height": "280px"}),
            ], style={**CARD_STYLE, "flex": "1"}),
            html.Div([
                html.P("KÜME DAĞILIMI – RADAR GRAFİĞİ", style=TITLE_STYLE),
                dcc.Graph(id="radar-grafik", style={"height": "280px"}),
            ], style={**CARD_STYLE, "flex": "1"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
    ])


def _yas_fig(yil="all"):
    demo_df, _, _ = create_demographic_tables(yil)
    if demo_df.empty:
        return _empty_fig("Yaş grafiği için yeterli veri bulunamadı.")
    fig = go.Figure()
    for col, label, color in [
        ("internet", "İnternet", C_ACCENT),
        ("edevlet", "E-Devlet", C_ACCENT2),
        ("sosyal", "Sosyal Medya", C_GREEN),
        ("eticaret", "E-Ticaret", C_ORANGE),
    ]:
        if col in demo_df.columns:
            fig.add_trace(go.Scatter(x=demo_df["yas_grubu"], y=demo_df[col], name=label, line=dict(color=color, width=2.7), mode="lines+markers", marker=dict(size=7)))
    fig.update_layout(**PLOTLY_LAYOUT, yaxis_title="Kullanım Oranı (%)", yaxis_range=[0, 105])
    return fig


def _egitim_fig(yil="all"):
    _, egitim_df, _ = create_demographic_tables(yil)
    if egitim_df.empty:
        return _empty_fig("Eğitim grafiği için yeterli veri bulunamadı.")
    fig = go.Figure()
    for col, label, color in [
        ("internet", "İnternet", C_ACCENT),
        ("edevlet", "E-Devlet", C_ACCENT2),
        ("dijital_b", "Dijital Beceri", C_GREEN),
    ]:
        if col in egitim_df.columns:
            fig.add_trace(go.Bar(name=label, x=egitim_df["egitim"], y=egitim_df[col], marker_color=color, text=["" if pd.isna(v) else f"{v:.0f}%" for v in egitim_df[col]], textposition="outside"))
    fig.update_layout(**PLOTLY_LAYOUT, barmode="group", yaxis_title="Oran (%)", yaxis_range=[0, 110])
    return fig


def _cinsiyet_fig(yil="all"):
    _, _, cinsiyet_df = create_demographic_tables(yil)
    if cinsiyet_df.empty or "cinsiyet" not in cinsiyet_df.columns:
        return _empty_fig("Cinsiyet grafiği için yeterli veri bulunamadı.")

    erkek_df = cinsiyet_df[cinsiyet_df["cinsiyet"] == "Erkek"]
    kadin_df = cinsiyet_df[cinsiyet_df["cinsiyet"] == "Kadın"]
    if erkek_df.empty or kadin_df.empty:
        return _empty_fig("Erkek/Kadın kırılımı bu yıl için bulunamadı.")

    kategoriler = ["internet", "edevlet", "eticaret", "sosyal", "dijital"]
    etiketler = ["İnternet", "E-Devlet", "E-Ticaret", "Sosyal Medya", "Dijital Beceri"]
    erkek, kadin = erkek_df.iloc[0], kadin_df.iloc[0]
    erkek_vals = [erkek.get(k, np.nan) for k in kategoriler]
    kadin_vals = [kadin.get(k, np.nan) for k in kategoriler]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Erkek", x=etiketler, y=erkek_vals, marker_color=C_ACCENT, text=["" if pd.isna(v) else f"{v:.1f}" for v in erkek_vals], textposition="outside"))
    fig.add_trace(go.Bar(name="Kadın", x=etiketler, y=kadin_vals, marker_color=C_PINK, text=["" if pd.isna(v) else f"{v:.1f}" for v in kadin_vals], textposition="outside"))
    fig.update_layout(**PLOTLY_LAYOUT, barmode="group", yaxis_range=[0, 108], yaxis_title="%")
    return fig


def _ols_fig(yil="all"):
    katsayilar = calculate_real_ols(yil)
    katsayilar = pd.to_numeric(katsayilar, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if katsayilar.empty:
        return _empty_fig("OLS için yeterli veri bulunamadı veya statsmodels kurulu değil.")
    renkler = [C_GREEN if k > 0 else C_RED for k in katsayilar]
    fig = go.Figure(go.Bar(x=katsayilar.values, y=katsayilar.index, orientation="h", marker_color=renkler, text=[f"β={v:.3f}" for v in katsayilar.values], textposition="outside"))
    fig.add_vline(x=0, line_color=C_BORDER, line_width=1)
    layout = {**PLOTLY_LAYOUT}
    layout["xaxis"] = dict(title="Standardize OLS Katsayısı", gridcolor=C_BORDER, tickfont=dict(color=C_MUTED, size=10))
    layout["yaxis"] = dict(tickfont=dict(color=C_TEXT, size=10), gridcolor=C_BORDER)
    fig.update_layout(**layout)
    return fig


def demografik(yil="all"):
    return html.Div([
        html.Div([
            html.Div([
                html.P("YAŞ GRUBUNA GÖRE DİJİTAL KULLANIM", style=TITLE_STYLE),
                dcc.Graph(figure=_yas_fig(yil), style={"height": "340px"}),
            ], style={**CARD_STYLE, "flex": "1"}),
            html.Div([
                html.P("EĞİTİM DÜZEYİ İLE DİJİTALLEŞME İLİŞKİSİ", style=TITLE_STYLE),
                dcc.Graph(figure=_egitim_fig(yil), style={"height": "340px"}),
            ], style={**CARD_STYLE, "flex": "1"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
        html.Div([
            html.Div([
                html.P("CİNSİYET FARKLILIKLARI", style=TITLE_STYLE),
                dcc.Graph(figure=_cinsiyet_fig(yil), style={"height": "280px"}),
            ], style={**CARD_STYLE, "flex": "1"}),
            html.Div([
                html.P("DİJİTALLEŞMEYİ AÇIKLAYAN DEĞİŞKENLER", style=TITLE_STYLE),
                dcc.Graph(figure=_ols_fig(yil), style={"height": "280px"}),
            ], style={**CARD_STYLE, "flex": "1"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
    ])


def _yz_trend_fig(yil="all"):
    yz_df = yz_beceri_tables("all" if yil == "all" else yil)
    if yz_df.empty:
        return _empty_fig("Yapay zekâ göstergesi için veri bulunamadı.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yz_df["yil"], y=yz_df["yz"], mode="lines+markers", name="Yapay Zekâ", line=dict(color=C_PURPLE, width=3.5), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=yz_df["yil"], y=yz_df["beceri"], mode="lines+markers", name="Dijital Beceri", line=dict(color=C_GREEN, width=2.8), marker=dict(size=7)))
    if yil != "all":
        fig.add_vline(x=int(yil), line_width=2, line_dash="dash", line_color=C_PINK)
    fig.update_layout(**PLOTLY_LAYOUT, yaxis_title="%", yaxis_range=[0, 105], hovermode="x unified")
    return fig


def _beceri_fig(yil="all"):
    df = _filtered_df_by_year(yil)
    rows = []
    for col in DIJITAL_BECERI_COLS:
        label = col.replace("dijital beceri", "").strip().title()
        rows.append({"beceri": label, "oran": _safe_indicator(df, col)})
    beceri_df = pd.DataFrame(rows).dropna(subset=["oran"])
    if beceri_df.empty:
        return _empty_fig("Dijital beceri sütunları için yeterli veri bulunamadı.")
    fig = go.Figure(go.Bar(x=beceri_df["oran"], y=beceri_df["beceri"], orientation="h", marker_color=C_ACCENT2, text=[f"{v:.1f}%" for v in beceri_df["oran"]], textposition="outside"))
    fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="Oran (%)", yaxis_title="")
    return fig


def _yz_by_age_fig(yil="all"):
    df = _filtered_df_by_year(yil).copy()
    if df.empty or COL_YAS is None or COL_YZ is None:
        return _empty_fig("Yaşa göre yapay zekâ için yeterli veri bulunamadı.")
    df["yas_num"] = pd.to_numeric(df[COL_YAS], errors="coerce")
    df["yas_grubu"] = pd.cut(df["yas_num"], bins=[15, 30, 45, 60, 120], labels=["16-30", "31-45", "46-60", "60+"])
    rows = []
    for grup, g in df.groupby("yas_grubu", observed=False):
        rows.append({"yas_grubu": str(grup), "yz": _safe_indicator(g, COL_YZ)})
    out = pd.DataFrame(rows)
    fig = go.Figure(go.Bar(x=out["yas_grubu"], y=out["yz"], marker_color=C_PURPLE, text=["" if pd.isna(v) else f"{v:.1f}%" for v in out["yz"]], textposition="outside"))
    fig.update_layout(**PLOTLY_LAYOUT, yaxis_title="%", yaxis_range=[0, 105])
    return fig


def _gelir_dijital_fig(yil="all"):
    df = _filtered_df_by_year(yil).copy()
    if df.empty:
        return _empty_fig("Gelir analizi için yeterli veri bulunamadı.")
    df["gelir_5li"] = _income_5li_for_df(df)
    df["dijital_skor"] = _row_digital_score(df) * 100
    labels = {1: "1-En Düşük", 2: "2-Düşük", 3: "3-Orta", 4: "4-Orta Üstü", 5: "5-Yüksek"}
    rows = []
    for gval, g in df.dropna(subset=["gelir_5li"]).groupby("gelir_5li"):
        rows.append({"gelir": labels.get(int(gval), str(gval)), "skor": g["dijital_skor"].mean()})
    out = pd.DataFrame(rows)
    if out.empty:
        return _empty_fig("Standart 5'li gelir değişkeni bulunamadı.")
    fig = go.Figure(go.Bar(x=out["gelir"], y=out["skor"], marker_color=C_ORANGE, text=[f"{v:.1f}" for v in out["skor"]], textposition="outside"))
    fig.update_layout(**PLOTLY_LAYOUT, yaxis_title="Dijital Skor (%)", yaxis_range=[0, 105])
    return fig


def yz_beceri_analizi(yil="all"):
    return html.Div([
        html.Div([
            html.Div([
                html.P("YAPAY ZEKÂ KULLANIMI", style=TITLE_STYLE),
                dcc.Graph(figure=_yz_trend_fig(yil), style={"height": "340px"}),
            ], style={**CARD_STYLE, "flex": "1"}),
            html.Div([
                html.P("DİJİTAL BECERİ PROFİLİ", style=TITLE_STYLE),
                dcc.Graph(figure=_beceri_fig(yil), style={"height": "340px"}),
            ], style={**CARD_STYLE, "flex": "1"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
        html.Div([
            html.Div([
                html.P("YAŞ GRUBUNA GÖRE YAPAY ZEKÂ", style=TITLE_STYLE),
                dcc.Graph(figure=_yz_by_age_fig(yil), style={"height": "300px"}),
            ], style={**CARD_STYLE, "flex": "1"}),
            html.Div([
                html.P("GELİR GRUBUNA GÖRE DİJİTALLEŞME", style=TITLE_STYLE),
                dcc.Graph(figure=_gelir_dijital_fig(yil), style={"height": "300px"}),
            ], style={**CARD_STYLE, "flex": "1"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
    ])


def _ivme_fig():
    cols_labels = [
        ("edevlet", "E-Devlet"),
        ("cevrimici_egitim", "Çevrimiçi Eğitim"),
        ("eticaret", "E-Ticaret"),
        ("sosyal_medya", "Sosyal Medya"),
        ("yapay_zeka_farkin", "YZ Farkındalığı"),
    ]
    if turkey_ts.empty or len(turkey_ts) < 2:
        return _empty_fig("Pandemi öncesi/sonrası karşılaştırması için yeterli yıl yok.")

    years = sorted(turkey_ts["yil"].dropna().astype(int).unique().tolist())
    y_before = 2019 if 2019 in years else years[0]
    y_after = 2021 if 2021 in years else years[-1]
    if y_before == y_after and len(years) >= 2:
        y_before, y_after = years[0], years[-1]

    v_before = turkey_ts[turkey_ts["yil"] == y_before].iloc[0]
    v_after = turkey_ts[turkey_ts["yil"] == y_after].iloc[0]

    available = [(c, l) for c, l in cols_labels if c in turkey_ts.columns]
    etiketler = [l for _, l in available]
    vals_before = [v_before[c] for c, _ in available]
    vals_after = [v_after[c] for c, _ in available]

    fig = go.Figure()
    fig.add_trace(go.Bar(name=f"{y_before} (öncesi)", x=etiketler, y=vals_before, marker_color="#C4B5FD", text=["" if pd.isna(v) else f"{v:.0f}%" for v in vals_before], textposition="inside"))
    fig.add_trace(go.Bar(name=f"{y_after} (sonrası)", x=etiketler, y=vals_after, marker_color=C_ACCENT, text=["" if pd.isna(v) else f"{v:.0f}%" for v in vals_after], textposition="inside"))
    fig.update_layout(**PLOTLY_LAYOUT, barmode="group", yaxis_title="%", yaxis_range=[0, 108])
    return fig


def _hedef_fig():
    if turkey_ts.empty or len(turkey_ts) < 2:
        return _empty_fig("Karşılaştırma için yeterli yıl yok.")
    first = turkey_ts.iloc[0]
    last = turkey_ts.iloc[-1]
    cols = ["internet_erisim", "edevlet", "dijital_beceri", "eticaret", "sosyal_medya", "yapay_zeka_farkin"]
    labels = {
        "internet_erisim": "İnternet\nErişimi",
        "edevlet": "E-Devlet",
        "dijital_beceri": "Dijital\nBeceri",
        "eticaret": "E-Ticaret",
        "sosyal_medya": "Sosyal\nMedya",
        "yapay_zeka_farkin": "YZ\nFarkındalığı",
    }
    available = [c for c in cols if c in turkey_ts.columns]
    fig = go.Figure()
    fig.add_trace(go.Bar(name=str(int(first["yil"])), x=[labels.get(c, c) for c in available], y=[first[c] for c in available], marker_color="#C4B5FD"))
    fig.add_trace(go.Bar(name=str(int(last["yil"])), x=[labels.get(c, c) for c in available], y=[last[c] for c in available], marker_color=C_ACCENT))
    fig.update_layout(**PLOTLY_LAYOUT, barmode="group", yaxis_title="%", yaxis_range=[0, 110])
    return fig


def karsil_analiz():
    return html.Div([
        html.Div([
            html.Div([
                html.P("YILLIK KARŞILAŞTIRMALI ANALİZ", style=TITLE_STYLE),
                html.Div([
                    html.Div([
                        html.Label("Yıl 1:", style={"color": C_MUTED, "fontSize": "11px"}),
                        dcc.Dropdown(id="karsil-yil1", options=[{"label": str(y), "value": y} for y in YILLAR], value=min(YILLAR), style={"fontSize": "12px"}),
                    ], style={"flex": "1"}),
                    html.Div([
                        html.Label("Yıl 2:", style={"color": C_MUTED, "fontSize": "11px"}),
                        dcc.Dropdown(id="karsil-yil2", options=[{"label": str(y), "value": y} for y in YILLAR], value=max(YILLAR), style={"fontSize": "12px"}),
                    ], style={"flex": "1"}),
                ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),
                dcc.Graph(id="karsil-grafik", style={"height": "380px"}),
            ], style={**CARD_STYLE, "flex": "3"}),

            html.Div([
                html.P("PANDEMİ ÖNCESİ / SONRASI", style=TITLE_STYLE),
                dcc.Graph(figure=_ivme_fig(), style={"height": "430px"}),
            ], style={**CARD_STYLE, "flex": "2"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        html.Div([
            html.P("BAŞLANGIÇ – SON YIL DEĞİŞİMİ", style=TITLE_STYLE),
            dcc.Graph(figure=_hedef_fig(), style={"height": "300px"}),
        ], style=CARD_STYLE),
    ])


def youtube_nlp_analizi():
    df, tfidf_df, lda_df, rep_df = load_youtube_nlp_data()

    if df.empty:
        return html.Div([
            html.Div([
                html.P("YOUTUBE NLP ÇIKTILARI BULUNAMADI", style=TITLE_STYLE),
                html.P("youtube_nlp_tam.csv dosyasını proje kökündeki gorseller klasörüne veya dashboard/gorseller klasörüne ekleyin.", style=SUBTITLE_STYLE),
            ], style=CARD_STYLE)
        ])

    top_comments = df.copy()
    if "begeni" in top_comments.columns:
        top_comments = top_comments.sort_values("begeni", ascending=False)

    return html.Div([
        html.Div([
            html.Div("YOUTUBE NLP", style={
                "fontSize": "12px",
                "fontWeight": "950",
                "letterSpacing": "1px",
                "color": "rgba(255,255,255,.90)",
                "marginBottom": "10px",
            }),
            html.Div("Dijitalleşmeye Yönelik Toplumsal Algı", style={
                "fontSize": "34px",
                "fontWeight": "950",
                "color": "white",
                "lineHeight": "1.05",
            }),
            html.Div("YouTube yorumları üzerinden duygu analizi, kelime frekansı, TF-IDF ve LDA konu modelleme çıktıları", style={
                "fontSize": "13px",
                "fontWeight": "650",
                "color": "rgba(255,255,255,.78)",
                "marginTop": "8px",
            }),
        ], style={
            "padding": "28px 30px",
            "borderRadius": "30px",
            "marginBottom": "18px",
            "background": "linear-gradient(135deg,#8B5CF6,#EC4899,#06B6D4)",
            "boxShadow": "0 28px 60px rgba(139,92,246,.28)",
        }),

        youtube_kpi_cards(df),

        html.Div([
            html.Div([
                html.P("DUYGU DAĞILIMI", style=TITLE_STYLE),
                html.P("Pozitif, negatif ve nötr yorum sayıları", style=SUBTITLE_STYLE),
                dcc.Graph(figure=youtube_sentiment_fig(df), style={"height": "330px"}),
            ], style={**CARD_STYLE, "flex": "1"}),

            html.Div([
                html.P("ANAHTAR KELİMEYE GÖRE DUYGU", style=TITLE_STYLE),
                html.P("Arama kelimeleri bazında toplumsal algı dağılımı", style=SUBTITLE_STYLE),
                dcc.Graph(figure=youtube_keyword_sentiment_fig(df), style={"height": "330px"}),
            ], style={**CARD_STYLE, "flex": "1"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        html.Div([
            html.Div([
                html.P("EN SIK KULLANILAN KELİMELER", style=TITLE_STYLE),
                html.P("Temizlenmiş yorumlar üzerinden kelime frekansı", style=SUBTITLE_STYLE),
                dcc.Graph(figure=youtube_word_freq_fig(df), style={"height": "360px"}),
            ], style={**CARD_STYLE, "flex": "1"}),

            html.Div([
                html.P("TF-IDF TERİMLERİ", style=TITLE_STYLE),
                html.P("Dijitalleşme söylemini ayıran güçlü terimler", style=SUBTITLE_STYLE),
                dcc.Graph(figure=youtube_tfidf_fig(tfidf_df), style={"height": "360px"}),
            ], style={**CARD_STYLE, "flex": "1"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        html.Div([
            html.P("LDA KONU MODELLEME ÖZETİ", style=TITLE_STYLE),
            html.P("YouTube yorumlarından çıkarılan temel konu kümeleri", style=SUBTITLE_STYLE),
            youtube_lda_table(lda_df),
        ], style=CARD_STYLE),

        html.Div([
            youtube_comments_table(rep_df, "TEMSİLİ YORUMLAR"),
            youtube_comments_table(top_comments, "EN ÇOK BEĞENİ ALAN YORUMLAR"),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        html.Div([
            youtube_image_card("Kelime Bulutu", "03_kelime_bulutu.png"),
            youtube_image_card("LDA Konuları", "06_lda_konular.png"),
            youtube_image_card("Zaman Serisi", "08_zaman_serisi.png"),
            youtube_image_card("Etkileşim Analizi", "09_top_engagement.png"),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
    ])


# ─────────────────────────────────────────────────────────────
# 7. LAYOUT
# ─────────────────────────────────────────────────────────────

app.layout = html.Div(style={
    "background": "radial-gradient(circle at 10% 5%, rgba(236,72,153,0.18) 0%, transparent 24%), radial-gradient(circle at 88% 0%, rgba(6,182,212,0.22) 0%, transparent 28%), radial-gradient(circle at 50% 18%, rgba(139,92,246,0.18) 0%, transparent 30%), linear-gradient(135deg, #FFF7FD 0%, #F4EEFF 45%, #EAFBFF 100%)",
    "minHeight": "100vh",
    "fontFamily": "Inter, system-ui, sans-serif",
    "padding": "0",
}, children=[
    html.Div([
        html.Div([
            html.Div("TR", style={
                "fontSize": "28px",
                "fontWeight": "900",
                "color": "white",
                "marginRight": "18px",
                "letterSpacing": "1px",
                "width": "64px",
                "height": "64px",
                "borderRadius": "20px",
                "background": "linear-gradient(135deg, #8B5CF6, #06B6D4)",
                "display": "flex",
                "alignItems": "center",
                "justifyContent": "center",
                "boxShadow": "0 16px 34px rgba(139,92,246,0.26)",
            }),
            html.Div([
                html.H1("Türkiye Dijitalleşme Endeksi", style={
                    "color": C_TEXT,
                    "fontSize": "34px",
                    "fontWeight": "950",
                    "margin": "0",
                    "letterSpacing": "0.2px",
                    "lineHeight": "1.05",
                }),
                html.P(
                    "Yapay Zekâ Çağında Bölgesel Farklılıklar & Çok Boyutlu Analiz • TÜİK BİT Araştırması • YouTube NLP • Dilara Şenay | TÜBİTAK 2209-A",
                    style={"color": C_MUTED, "fontSize": "12px", "fontWeight": "650", "margin": "8px 0 12px"},
                ),
                html.Div([
                    html.Span("CSV Tabanlı Zaman Serisi", style={"background": "rgba(139,92,246,0.12)", "color": "#7C3AED", "padding": "7px 10px", "borderRadius": "999px", "fontSize": "10px", "fontWeight": "900", "letterSpacing": "0.3px"}),
                    html.Span("TÜİK BİT Araştırması", style={"background": "rgba(0,166,200,0.10)", "color": "#08788F", "padding": "7px 10px", "borderRadius": "999px", "fontSize": "10px", "fontWeight": "900", "letterSpacing": "0.3px"}),
                    html.Span("YouTube NLP", style={"background": "rgba(236,72,153,0.12)", "color": "#DB2777", "padding": "7px 10px", "borderRadius": "999px", "fontSize": "10px", "fontWeight": "900", "letterSpacing": "0.3px"}),
                    html.Span("TÜBİTAK 2209-A", style={"background": "rgba(249,115,22,0.12)", "color": "#EA580C", "padding": "7px 10px", "borderRadius": "999px", "fontSize": "10px", "fontWeight": "900", "letterSpacing": "0.3px"}),
                ], style={"display": "flex", "gap": "8px", "flexWrap": "wrap"}),
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
        ], style={
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "stretch",
            "background": "rgba(255,255,255,.56)",
            "padding": "14px 16px",
            "borderRadius": "24px",
            "border": "1px solid rgba(139,92,246,.18)",
            "boxShadow": "0 18px 42px rgba(139,92,246,.14)",
            "backdropFilter": "blur(18px)",
        }),
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
        "position": "sticky",
        "top": "0",
        "zIndex": "100",
    }),

    html.Div([
        dcc.Tabs(id="ana-sekme", value="genel", children=[
            dcc.Tab(label="Genel Bakış", value="genel"),
            dcc.Tab(label="Bölgesel Harita", value="harita"),
            dcc.Tab(label="Demografik", value="demo"),
            dcc.Tab(label="YZ & Dijital Beceri", value="yz"),
            dcc.Tab(label="Karşılaştırma", value="karsil"),
            dcc.Tab(label="YouTube NLP", value="youtube_nlp"),
        ], style={"fontFamily": "inherit"}, colors={"border": C_BORDER, "primary": C_ACCENT, "background": C_BG}),
    ], style={
        "padding": "16px 32px 8px",
        "background": "linear-gradient(90deg, rgba(255,255,255,0.55), rgba(236,254,255,0.35), rgba(250,245,255,0.55))",
        "backdropFilter": "blur(12px)",
        "borderBottom": f"1px solid {C_BORDER}",
        "boxShadow": "0 10px 30px rgba(15,23,42,0.04)",
    }),

    html.Div(id="sekme-icerik", style={"padding": "20px 32px"}),
])


# ─────────────────────────────────────────────────────────────
# 8. CALLBACK'LER
# ─────────────────────────────────────────────────────────────

@app.callback(
    Output("sekme-icerik", "children"),
    Input("ana-sekme", "value"),
    Input("global-yil", "value"),
)
def render_tab(sekme, yil):
    if sekme == "genel":
        return genel_bakis()
    if sekme == "harita":
        return bolgesel_harita()
    if sekme == "demo":
        return demografik(yil)
    if sekme == "yz":
        return yz_beceri_analizi(yil)
    if sekme == "karsil":
        return karsil_analiz()
    if sekme == "youtube_nlp":
        return youtube_nlp_analizi()
    return html.Div("Sekme bulunamadı")


@app.callback(Output("hero-row", "children"), Input("global-yil", "value"))
def update_hero_row(yil):
    return make_hero_row(yil)


@app.callback(Output("kpi-row", "children"), Input("global-yil", "value"))
def update_kpi_row(yil):
    return make_kpi_row(yil)


@app.callback(
    Output("ts-grafik", "figure"),
    Input("ts-gostergeler", "value"),
    Input("global-yil", "value"),
)
def update_ts(secili, yil):
    if not secili:
        secili = ["internet_erisim"]

    renk_paleti = [C_ACCENT, C_ACCENT2, C_GREEN, C_ORANGE, C_PURPLE, C_PINK, "#F97316", "#14B8A6"]
    etiket_map = {
        "internet_erisim": "İnternet Erişimi",
        "dijital_beceri": "Dijital Beceri",
        "edevlet": "E-Devlet",
        "eticaret": "E-Ticaret",
        "sosyal_medya": "Sosyal Medya",
        "cevrimici_egitim": "Çevrimiçi Eğitim",
        "yapay_zeka_farkin": "YZ Farkındalığı",
        "bilgisayar": "Bilgisayar",
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
        if col not in df_plot.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df_plot["yil"],
            y=df_plot[col],
            name=etiket_map.get(col, col),
            line=dict(color=renk_paleti[i % len(renk_paleti)], width=2.5),
            mode="lines+markers",
            marker=dict(size=7),
            hovertemplate=f"<b>%{{x}}</b><br>{etiket_map.get(col, col)}: %{{y:.1f}}%<extra></extra>",
        ))

    fig.add_vrect(x0=2019.5, x1=2021.5, fillcolor="rgba(245,158,11,0.07)", line_width=1, line_dash="dot", line_color=C_ORANGE)
    fig.add_annotation(x=2020.5, y=5, text="Pandemi", showarrow=False, font=dict(color=C_ORANGE, size=9))
    if yil != "all":
        fig.add_vline(x=int(yil), line_width=2, line_dash="dash", line_color=C_PINK)

    fig.add_annotation(x=df_plot["yil"].median(), y=102, text=secili_text, showarrow=False, font=dict(color=C_PINK if yil != "all" else C_ACCENT, size=10))
    fig.update_layout(**PLOTLY_LAYOUT, yaxis_title="Oran (%)", yaxis_range=[0, 105], hovermode="x unified")
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
    Input("global-yil", "value"),
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
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=C_RED),
        )
        fig.update_layout(
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=0, r=0, t=40, b=0),
            height=520,
        )
        return fig, _bolge_bar(df), _ucurum_fig(df), _radar_fig(df)

    geo_ids = {f.get("properties", {}).get("NUTS_ID") for f in TR_IBBS1_GEOJSON.get("features", [])}
    df = df[df["kod"].isin(geo_ids)].copy()

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="GeoJSON ile bölge kodları eşleşmedi.<br>tr_ibbs1.geojson içinde NUTS_ID alanını kontrol et.",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=15, color=C_RED),
        )
        fig.update_layout(
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=520,
            margin=dict(l=0, r=0, t=40, b=0),
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
            title=dict(text=etiketler[gosterge], font=dict(size=10, color=C_TEXT)),
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
            + "Seviye: %{customdata[7]}"
            + "<extra></extra>"
        ),
    ))

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
        title=dict(text=f"<b>{etiketler[gosterge]}</b> · {aktif_yil} yılı görünümü", x=0.5, font=dict(size=17, color=C_TEXT, family="Inter, Arial")),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(color=C_TEXT, family="Inter, Arial"),
        margin=dict(l=0, r=0, t=48, b=0),
        height=520,
        hoverlabel=dict(bgcolor="rgba(255,255,255,0.96)", bordercolor="rgba(139,92,246,0.25)", font=dict(color=C_TEXT, family="Inter, Arial")),
    )

    return fig, _bolge_bar(df), _ucurum_fig(df), _radar_fig(df)


@app.callback(
    Output("karsil-grafik", "figure"),
    Input("karsil-yil1", "value"),
    Input("karsil-yil2", "value"),
)
def update_karsil(yil1, yil2):
    if yil1 == yil2:
        yil2 = min(yil1 + 1, max(YILLAR))

    row1 = turkey_ts[turkey_ts["yil"] == yil1].iloc[0]
    row2 = turkey_ts[turkey_ts["yil"] == yil2].iloc[0]

    cols = ["internet_erisim", "dijital_beceri", "edevlet", "eticaret", "sosyal_medya", "cevrimici_egitim", "yapay_zeka_farkin", "bilgisayar"]
    lbls = ["İnternet", "Dijital Beceri", "E-Devlet", "E-Ticaret", "Sosyal Medya", "Çev.Eğitim", "YZ Farkın.", "Bilgisayar"]

    v1 = [row1[c] if c in row1 else np.nan for c in cols]
    v2 = [row2[c] if c in row2 else np.nan for c in cols]
    delta = [b - a if not pd.isna(a) and not pd.isna(b) else 0 for a, b in zip(v1, v2)]

    fig = make_subplots(rows=1, cols=2, subplot_titles=[f"{yil1} vs {yil2} Değerleri", "Değişim (Puan Farkı)"], horizontal_spacing=0.1)
    fig.add_trace(go.Bar(name=str(yil1), x=lbls, y=v1, marker_color="#C4B5FD", text=[f"{v:.0f}%" if not pd.isna(v) else "" for v in v1], textposition="inside"), row=1, col=1)
    fig.add_trace(go.Bar(name=str(yil2), x=lbls, y=v2, marker_color=C_ACCENT, text=[f"{v:.0f}%" if not pd.isna(v) else "" for v in v2], textposition="inside"), row=1, col=1)
    fig.add_trace(go.Bar(name="Δ Değişim", x=lbls, y=delta, marker_color=[C_GREEN if d > 0 else C_RED for d in delta], text=[f"+{d:.1f}" if d > 0 else f"{d:.1f}" for d in delta], textposition="outside"), row=1, col=2)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=C_TEXT),
        margin=dict(l=20, r=20, t=50, b=20),
        barmode="group",
        legend=dict(bgcolor="rgba(255,255,255,0.76)", bordercolor=C_BORDER, borderwidth=1, font=dict(size=10)),
        xaxis=dict(tickfont=dict(size=9, color=C_TEXT), gridcolor=C_BORDER, tickangle=-30),
        yaxis=dict(gridcolor=C_BORDER, tickfont=dict(size=9, color=C_MUTED)),
        xaxis2=dict(tickfont=dict(size=9, color=C_TEXT), gridcolor=C_BORDER, tickangle=-30),
        yaxis2=dict(gridcolor=C_BORDER, tickfont=dict(size=9, color=C_MUTED)),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# 9. ÇALIŞTIRMA
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  TÜRKİYE DİJİTALLEŞME DASHBOARD + YOUTUBE NLP")
    print("  Dilara Şenay | TÜBİTAK 2209-A")
    print("=" * 60)
    print("  Tarayıcıda açın: http://127.0.0.1:8050")
    print("=" * 60 + "\n")
    app.run(debug=True, port=8050)
