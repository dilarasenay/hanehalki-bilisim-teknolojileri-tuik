import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="Çocuk Dijital Yaşam 2024",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent

CSV_PATH = BASE_DIR / "2024cocukbilisim.csv"
CHILD_IMG = BASE_DIR / "assets" / "child.png"
CHILD_CLEAN_IMG = BASE_DIR / "assets" / "child_clean.png"

PLOT_CONFIG = {
    "displayModeBar": False,
    "displaylogo": False,
    "responsive": True
}

PURPLE = "#8B7CFF"
HOT_PINK = "#FF6FAE"
PINK = "#FF8FB3"
BLUE = "#7AB8FF"
MINT = "#7EDDC3"
YELLOW = "#FFD66B"
ORANGE = "#FFA36C"
GREEN = "#A6DA70"
RED = "#FF7A7A"
TEXT = "#24145F"
SOFT_TEXT = "#7B6FA8"

PALETTE = [PURPLE, HOT_PINK, BLUE, MINT, YELLOW, ORANGE, GREEN, RED]


@st.cache_data
def load_data():
    return pd.read_csv(CSV_PATH, encoding="utf-8-sig")


df = load_data()


def yes_pct(data, col):
    s = pd.to_numeric(data[col], errors="coerce")
    return round((s == 1).mean() * 100, 1)


def fmt_pct(x):
    return str(x).replace(".", ",")


def make_child_transparent(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        return None

    img = Image.open(input_path).convert("RGBA")
    new_data = []

    for r, g, b, a in img.getdata():
        if r > 218 and g > 218 and b > 218:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append((r, g, b, a))

    img.putdata(new_data)
    img.save(output_path)
    return output_path


def get_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


clean_path = make_child_transparent(CHILD_IMG, CHILD_CLEAN_IMG)
img_base64 = get_base64(clean_path if clean_path else CHILD_IMG)


def section_title(title):
    st.markdown(f"""
    <div class="section-wrap">
        <div class="bubble b1"></div>
        <div class="bubble b2"></div>
        <div class="bubble b3"></div>
        <div class="section-header">
            <span class="section-dot"></span>
            <span>{title}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def apply_theme(fig, height=350):
    if fig.layout.title.text is None:
        fig.update_layout(title="")

    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Nunito", color=TEXT, size=13),
        title=dict(font=dict(size=15, color=TEXT), x=0.02),
        margin=dict(t=42, b=32, l=32, r=32),
        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            orientation="h",
            y=-0.18,
            x=0.5,
            xanchor="center"
        ),
        xaxis=dict(
            gridcolor="rgba(139,124,255,0.13)",
            zeroline=False,
            linecolor="rgba(139,124,255,0.18)",
            tickfont=dict(color=SOFT_TEXT),
            title_font=dict(color=SOFT_TEXT)
        ),
        yaxis=dict(
            gridcolor="rgba(139,124,255,0.13)",
            zeroline=False,
            linecolor="rgba(139,124,255,0.18)",
            tickfont=dict(color=SOFT_TEXT),
            title_font=dict(color=SOFT_TEXT)
        )
    )
    return fig


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif !important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"],
[data-testid="stStatusWidget"] {
    display: none !important;
    visibility: hidden !important;
}

section[data-testid="stSidebar"] {
    width: 315px !important;
    min-width: 315px !important;
    max-width: 315px !important;
    transform: translateX(0px) !important;
    visibility: visible !important;
}

button[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

[data-testid="stSidebar"][aria-expanded="false"] {
    transform: translateX(0px) !important;
    margin-left: 0 !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 8% 8%, rgba(255,143,179,0.26), transparent 30%),
        radial-gradient(circle at 92% 6%, rgba(122,184,255,0.27), transparent 30%),
        radial-gradient(circle at 16% 94%, rgba(126,221,195,0.22), transparent 32%),
        radial-gradient(circle at 86% 88%, rgba(255,214,107,0.24), transparent 30%),
        linear-gradient(135deg, #FFF8EC 0%, #FFF2F8 45%, #F1FBFF 100%);
}

[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at 18% 8%, rgba(255,255,255,0.24), transparent 18%),
        radial-gradient(circle at 80% 30%, rgba(255,255,255,0.13), transparent 24%),
        linear-gradient(180deg, #7D6BFF 0%, #C681E8 42%, #FF6FAE 100%);
    border-right: 4px solid rgba(255,255,255,0.35);
    box-shadow: 18px 0 45px rgba(80,65,170,0.18);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

[data-testid="stSidebar"] label {
    font-weight: 900 !important;
    font-size: 13px !important;
}

[data-testid="stSidebar"] .stMultiSelect span {
    background: rgba(255,255,255,0.25) !important;
    border-radius: 999px !important;
    color: white !important;
    font-weight: 900 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.17) !important;
    border: 1px solid rgba(255,255,255,0.34) !important;
    border-radius: 18px !important;
}

[data-testid="stSidebar"] hr {
    background: rgba(255,255,255,0.25) !important;
}

.block-container {
    padding-top: 0.8rem !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
    padding-bottom: 2.5rem !important;
}

.hero {
    background:
        radial-gradient(circle at 88% 20%, rgba(255,255,255,0.22), transparent 18%),
        linear-gradient(135deg, #8B7CFF 0%, #FF8FB3 52%, #FFD66B 100%);
    border-radius: 34px;
    padding: 38px 44px;
    margin-bottom: 28px;
    color: white;
    position: relative;
    overflow: hidden;
    box-shadow: 0 26px 64px rgba(139,124,255,0.25);
    border: 4px solid rgba(255,255,255,0.62);
    min-height: 205px;
}

.hero::before {
    content: "";
    position: absolute;
    width: 230px;
    height: 230px;
    right: -58px;
    top: -78px;
    background: rgba(255,255,255,0.23);
    border-radius: 50%;
}

.hero::after {
    content: "★";
    position: absolute;
    left: 34px;
    top: 28px;
    color: #FFD66B;
    font-size: 44px;
    text-shadow: 0 8px 22px rgba(0,0,0,0.14);
    z-index: 3;
}

.hero h1 {
    font-size: 39px;
    font-weight: 900;
    margin: 0 300px 10px 75px;
    letter-spacing: -0.8px;
    position: relative;
    z-index: 2;
}

.hero p {
    margin: 0 300px 0 75px;
    font-size: 15px;
    font-weight: 900;
    opacity: 0.97;
    position: relative;
    z-index: 2;
}

.hero-child {
    position: absolute;
    right: 20px;
    bottom: -24px;
    width: 265px;
    z-index: 4;
    background: transparent !important;
    filter: drop-shadow(0 18px 30px rgba(60,52,137,0.24));
    animation: floatChild 3.2s ease-in-out infinite;
}

@keyframes floatChild {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}

.kpi {
    border-radius: 30px;
    padding: 23px 23px;
    min-height: 142px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 18px 38px rgba(60,52,137,0.11);
    border: 4px solid rgba(255,255,255,0.86);
    transition: all 0.20s ease;
}

.kpi:hover {
    transform: translateY(-5px) scale(1.01);
    box-shadow: 0 26px 52px rgba(60,52,137,0.16);
}

.kpi-label {
    font-size: 14px;
    font-weight: 900;
    opacity: 0.80;
}

.kpi-val {
    font-size: 43px;
    font-weight: 900;
    line-height: 1;
    letter-spacing: -1px;
}

.kpi-sub {
    font-size: 12px;
    font-weight: 900;
    opacity: 0.62;
}

.section-wrap {
    position: relative;
    display: inline-block;
    margin: 44px 0 20px;
}

.section-header {
    position: relative;
    z-index: 2;
    display: inline-flex;
    align-items: center;
    gap: 12px;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.97), rgba(255,244,252,0.84));
    color: #24145F;
    border-radius: 999px;
    padding: 16px 27px 16px 18px;
    font-size: 21px;
    font-weight: 900;
    box-shadow:
        0 18px 40px rgba(60,52,137,0.12),
        inset 0 1px 0 rgba(255,255,255,0.90);
    border: 3px solid rgba(255,255,255,0.92);
}

.section-dot {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    display: inline-flex;
    background: linear-gradient(135deg, #8B7CFF, #FF6FAE, #FFD66B);
    box-shadow: 0 6px 16px rgba(139,124,255,0.35);
}

.bubble {
    position: absolute;
    border-radius: 999px;
    z-index: 1;
    opacity: 0.95;
}

.b1 {
    width: 26px;
    height: 26px;
    background: linear-gradient(135deg, #8B7CFF, #7AB8FF);
    left: -8px;
    top: -5px;
}

.b2 {
    width: 18px;
    height: 18px;
    background: linear-gradient(135deg, #FF6FAE, #FFD66B);
    right: -6px;
    top: 6px;
}

.b3 {
    width: 13px;
    height: 13px;
    background: linear-gradient(135deg, #7EDDC3, #A6DA70);
    left: 18px;
    bottom: -5px;
}

hr {
    border: none;
    height: 3px;
    background: linear-gradient(90deg, transparent, rgba(139,124,255,0.34), rgba(255,143,179,0.34), transparent);
    margin: 32px 0;
}

[data-testid="stExpander"] {
    border-radius: 24px !important;
    background: rgba(255,255,255,0.75) !important;
    border: 2px solid rgba(255,255,255,0.92) !important;
}

div[data-baseweb="select"] {
    border-radius: 18px !important;
}

.stDownloadButton button {
    border-radius: 18px !important;
    font-weight: 900 !important;
}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("## Filtreler")
    st.caption("Seçimler tüm grafiklere anında yansır.")
    st.divider()

    age_range = st.slider(
        "Yaş aralığı",
        int(df["yaş"].min()),
        int(df["yaş"].max()),
        (int(df["yaş"].min()), int(df["yaş"].max()))
    )

    gender_map = {1: "Erkek", 2: "Kız"}

    gender_opts = st.multiselect(
        "Cinsiyet",
        [1, 2],
        default=[1, 2],
        format_func=lambda x: gender_map[x]
    )

    internet_opts = st.multiselect(
        "İnternet kullanıyor mu?",
        [1, 2],
        default=[1, 2],
        format_func=lambda x: "Evet" if x == 1 else "Hayır"
    )

    phone_opts = st.multiselect(
        "Cep telefonu kullanıyor mu?",
        [1, 2],
        default=[1, 2],
        format_func=lambda x: "Evet" if x == 1 else "Hayır"
    )

    social_opts = st.multiselect(
        "Sosyal medyada mı?",
        [1, 2],
        default=[1, 2],
        format_func=lambda x: "Evet" if x == 1 else "Hayır"
    )

    game_opts = st.multiselect(
        "Dijital oyun oynuyor mu?",
        [1, 2],
        default=[1, 2],
        format_func=lambda x: "Evet" if x == 1 else "Hayır"
    )

    school_vals = sorted(df["eğitime devam etme durumu"].dropna().unique())

    school_opts = st.multiselect(
        "Eğitime devam etme",
        school_vals,
        default=school_vals,
        format_func=lambda x: "Devam ediyor" if x == 1 else "Etmiyor"
    )

    st.divider()

    compare_by = st.selectbox(
        "Karşılaştırma ekseni",
        ["yaş", "cinsiyet", "eğitime devam etme durumu"],
        format_func=lambda x: {
            "yaş": "Yaş",
            "cinsiyet": "Cinsiyet",
            "eğitime devam etme durumu": "Okul durumu"
        }[x]
    )


flt = df.copy()

flt = flt[
    flt["yaş"].between(*age_range) &
    flt["cinsiyet"].isin(gender_opts) &
    flt["internet kullanım durumu"].isin(internet_opts) &
    flt["cep telefonu kullanım durumu"].isin(phone_opts) &
    flt["sosyal medya kullanım durumu"].isin(social_opts) &
    flt["dijital oyun oynama"].isin(game_opts) &
    flt["eğitime devam etme durumu"].isin(school_opts)
]

gender_label = (
    "Tümü"
    if len(gender_opts) == 2
    else gender_map.get(gender_opts[0], "?")
    if gender_opts
    else "Seçilmedi"
)

st.markdown(f"""
<div class="hero">
  <img src="data:image/png;base64,{img_base64}" class="hero-child"/>
  <h1>Çocuklar Dijital Dünyada — TÜİK 2024</h1>
  <p>
    Seçili gözlem: <strong>{len(flt):,}</strong> / {len(df):,} çocuk ·
    Yaş: <strong>{age_range[0]}–{age_range[1]}</strong> ·
    Cinsiyet: <strong>{gender_label}</strong>
  </p>
</div>
""", unsafe_allow_html=True)

if flt.empty:
    st.warning("Seçilen filtrelerle eşleşen veri yok. Lütfen filtreleri genişlet.")
    st.stop()

internet_r = yes_pct(flt, "internet kullanım durumu")
social_r = yes_pct(flt, "sosyal medya kullanım durumu")
phone_r = yes_pct(flt, "cep telefonu kullanım durumu")
computer_r = yes_pct(flt, "bilgisayar kullanım durumu")
game_r = yes_pct(flt, "dijital oyun oynama")

k1, k2, k3, k4, k5 = st.columns(5)

kpi_data = [
    (k1, "#EDEAFF", "#5847C6", "İnternet", internet_r),
    (k2, "#FFE4EE", "#D03F7A", "Sosyal medya", social_r),
    (k3, "#FFF0C4", "#BA7100", "Cep telefonu", phone_r),
    (k4, "#DDF8F1", "#118C70", "Bilgisayar", computer_r),
    (k5, "#DFF0FF", "#1973CF", "Dijital oyun", game_r),
]

for col, bg, fg, label, val in kpi_data:
    with col:
        st.markdown(f"""
        <div class="kpi" style="background:{bg};color:{fg}">
          <div class="kpi-label">{label}</div>
          <div class="kpi-val">%{fmt_pct(val)}</div>
          <div class="kpi-sub">kullanan çocuk oranı</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

section_title("Yaşa Göre Dijital Kullanım Eğrileri")

usage_cols_map = {
    "İnternet": "internet kullanım durumu",
    "Sosyal medya": "sosyal medya kullanım durumu",
    "Cep telefonu": "cep telefonu kullanım durumu",
    "Dijital oyun": "dijital oyun oynama",
}

age_trend = pd.DataFrame({
    label: flt.groupby("yaş")[col].apply(
        lambda x: round((pd.to_numeric(x, errors="coerce") == 1).mean() * 100, 1)
    )
    for label, col in usage_cols_map.items()
}).reset_index()

fig_trend = go.Figure()
colors_trend = [PURPLE, HOT_PINK, BLUE, ORANGE]

for i, label in enumerate(usage_cols_map.keys()):
    fig_trend.add_trace(go.Scatter(
        x=age_trend["yaş"],
        y=age_trend[label],
        name=label,
        mode="lines+markers",
        line=dict(color=colors_trend[i], width=4, shape="spline"),
        marker=dict(size=12, color=colors_trend[i], line=dict(color="white", width=3)),
        fill="tozeroy" if i == 0 else None,
        fillcolor="rgba(139,124,255,0.10)" if i == 0 else None,
    ))

fig_trend.update_layout(
    title="",
    yaxis=dict(title="Oran (%)", range=[0, 105]),
    xaxis=dict(title="Yaş", dtick=1),
    hovermode="x unified"
)

fig_trend = apply_theme(fig_trend, height=375)
st.plotly_chart(fig_trend, use_container_width=True, config=PLOT_CONFIG)

section_title("Platform ve Cihaz Dağılımı")

col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    platforms = {
        "YouTube": "youtube kullanımı",
        "Instagram": "instagram kullanımı",
        "TikTok": "tiktok kullanımı",
        "Snapchat": "snapchat kullanımı",
        "Pinterest": "pinterest kullanımı",
        "Facebook": "facebook kullanımı",
        "Twitter/X": "twitter kullanımı",
    }

    platform_colors = [RED, HOT_PINK, "#3D3D3D", YELLOW, "#E94A8A", BLUE, "#4CA3FF"]

    plat_df = pd.DataFrame({
        "Platform": list(platforms.keys()),
        "Oran": [yes_pct(flt, c) for c in platforms.values()],
        "Renk": platform_colors,
    }).sort_values("Oran", ascending=True)

    fig_plat = go.Figure()

    for _, row in plat_df.iterrows():
        fig_plat.add_trace(go.Scatter(
            x=[0, row["Oran"]],
            y=[row["Platform"], row["Platform"]],
            mode="lines",
            line=dict(color=row["Renk"], width=9),
            opacity=0.33,
            showlegend=False,
            hoverinfo="skip"
        ))

        fig_plat.add_trace(go.Scatter(
            x=[row["Oran"]],
            y=[row["Platform"]],
            mode="markers+text",
            marker=dict(color=row["Renk"], size=27, line=dict(color="white", width=3)),
            text=[f"{fmt_pct(row['Oran'])}%"],
            textposition="middle right",
            textfont=dict(size=13, color=TEXT),
            showlegend=False,
            name=row["Platform"],
        ))

    fig_plat.update_layout(
        title="Sosyal medya platformları kullanımı",
        xaxis=dict(range=[0, 115], ticksuffix="%"),
        yaxis=dict(title="")
    )

    fig_plat = apply_theme(fig_plat, height=355)
    st.plotly_chart(fig_plat, use_container_width=True, config=PLOT_CONFIG)

with col_right:
    devices = {
        "Cep telefonu": "çocuğun kendi cep telefonu",
        "Tablet": "çocuğun kendi tableti",
        "Akıllı saat": "çocuğun kendi akıllı saati",
        "Dizüstü": "çocuğun kendi dizüstü bilgisayarı",
        "Masaüstü": "çocuğun kendi masaüstü bilgisayarı",
        "Oyun konsolu": "çocuğun kendi oyun konsolu",
    }

    dev_df = pd.DataFrame({
        "Cihaz": list(devices.keys()),
        "Oran": [yes_pct(flt, c) for c in devices.values()],
    })

    fig_dev = px.pie(
        dev_df,
        names="Cihaz",
        values="Oran",
        hole=0.56,
        color_discrete_sequence=[PURPLE, HOT_PINK, BLUE, MINT, YELLOW, ORANGE],
        title="Kişisel cihaz sahipliği"
    )

    fig_dev.update_traces(
        textposition="outside",
        textinfo="label+percent",
        textfont_size=12,
        marker=dict(line=dict(color="white", width=4))
    )

    fig_dev.update_layout(
        annotations=[dict(
            text=f"<b>{len(flt):,}</b><br>çocuk",
            x=0.5,
            y=0.5,
            font_size=15,
            showarrow=False,
            font_color=TEXT
        )],
        showlegend=False
    )

    fig_dev = apply_theme(fig_dev, height=355)
    st.plotly_chart(fig_dev, use_container_width=True, config=PLOT_CONFIG)

section_title("Ekran Süresi Dağılımı")

c1, c2 = st.columns(2)

SAAT_MAP = {
    1.0: "1 saat",
    2.0: "2 saat",
    3.0: "3 saat",
    4.0: "4 saat",
    5.0: "5+ saat"
}

with c1:
    wi = flt["hafta içi cep telefonu ortalama kullanımı"].value_counts().reindex([1, 2, 3, 4, 5]).fillna(0)
    we = flt["hafta sonu cep telefonu ortalama kullanımı"].value_counts().reindex([1, 2, 3, 4, 5]).fillna(0)

    fig_phone = go.Figure()
    fig_phone.add_trace(go.Bar(
        name="Hafta içi",
        x=list(SAAT_MAP.values()),
        y=wi.values,
        marker=dict(color=BLUE, line=dict(color="white", width=2)),
        text=[f"{v:.0f}" for v in wi.values],
        textposition="outside",
    ))
    fig_phone.add_trace(go.Bar(
        name="Hafta sonu",
        x=list(SAAT_MAP.values()),
        y=we.values,
        marker=dict(color=HOT_PINK, line=dict(color="white", width=2)),
        text=[f"{v:.0f}" for v in we.values],
        textposition="outside",
    ))

    fig_phone.update_layout(
        title="Cep telefonu günlük kullanım süresi",
        barmode="group",
        yaxis=dict(title="Kişi sayısı"),
        xaxis=dict(title="")
    )

    fig_phone = apply_theme(fig_phone, height=355)
    st.plotly_chart(fig_phone, use_container_width=True, config=PLOT_CONFIG)

with c2:
    wi2 = flt["hafta içi ortalama bilgisayar kullanımı"].value_counts().reindex([1, 2, 3, 4, 5]).fillna(0)
    we2 = flt["hafta sonu ortalama bilgisayar kullanımı"].value_counts().reindex([1, 2, 3, 4, 5]).fillna(0)

    fig_pc = go.Figure()
    fig_pc.add_trace(go.Bar(
        name="Hafta içi",
        x=list(SAAT_MAP.values()),
        y=wi2.values,
        marker=dict(color=PURPLE, line=dict(color="white", width=2)),
        text=[f"{v:.0f}" for v in wi2.values],
        textposition="outside",
    ))
    fig_pc.add_trace(go.Bar(
        name="Hafta sonu",
        x=list(SAAT_MAP.values()),
        y=we2.values,
        marker=dict(color=MINT, line=dict(color="white", width=2)),
        text=[f"{v:.0f}" for v in we2.values],
        textposition="outside",
    ))

    fig_pc.update_layout(
        title="Bilgisayar günlük kullanım süresi",
        barmode="group",
        yaxis=dict(title="Kişi sayısı"),
        xaxis=dict(title="")
    )

    fig_pc = apply_theme(fig_pc, height=355)
    st.plotly_chart(fig_pc, use_container_width=True, config=PLOT_CONFIG)

section_title("İnternet Kullanım Amaçları")

internet_amac = {
    "Ödev/ders": "interneti ödev/ders amaçlı kullanma",
    "Video izleme": "interneti video izleme amaçlı kullanma",
    "Oyun": "interneti oyun indirme/oynama amaçlı kullanma",
    "Sosyal medya": "interneti sosyal medya amaçlı kullanma",
    "Müzik": "interneti müzik dinleme amaçlı kullanma",
    "Arama/mesaj": "interneti arama/mesaj amaçlı kullanma",
    "Alışveriş": "internet üzerinden alışveriş yapma",
    "E-posta": "interneti eposta gönderme/alma amaçlı kullanma",
}

amac_df = pd.DataFrame({
    "Amaç": list(internet_amac.keys()),
    "Oran": [yes_pct(flt, c) for c in internet_amac.values()],
}).sort_values("Oran", ascending=False)

fig_amac = px.bar(
    amac_df,
    x="Amaç",
    y="Oran",
    text="Oran",
    color="Amaç",
    color_discrete_sequence=PALETTE
)

fig_amac.update_traces(
    texttemplate="%{text}%",
    textposition="outside",
    marker_line=dict(color="white", width=2)
)

fig_amac.update_layout(
    title="",
    yaxis=dict(title="Oran (%)", range=[0, 110]),
    xaxis=dict(title=""),
    showlegend=False
)

fig_amac = apply_theme(fig_amac, height=375)
st.plotly_chart(fig_amac, use_container_width=True, config=PLOT_CONFIG)

section_title("Dijital Oyun Dünyası")

oc1, oc2 = st.columns(2)

with oc1:
    oyun_turleri = {
        "Macera": "dijital oyun türü macera",
        "Strateji": "dijital oyun türü strateji",
        "Savaş": "dijital oyun türü savaş",
        "Spor": "dijital oyun türü spor",
        "Simülasyon": "dijital oyun türü simülasyon",
        "Rol yapma": "dijital oyun türü rol yapma",
    }

    oyun_df = pd.DataFrame({
        "Tür": list(oyun_turleri.keys()),
        "Oran": [yes_pct(flt, c) for c in oyun_turleri.values()],
    })

    cats = oyun_df["Tür"].tolist() + [oyun_df["Tür"].tolist()[0]]
    vals = oyun_df["Oran"].tolist() + [oyun_df["Oran"].tolist()[0]]

    fig_radar = go.Figure(go.Scatterpolar(
        r=vals,
        theta=cats,
        fill="toself",
        fillcolor="rgba(139,124,255,0.25)",
        line=dict(color=PURPLE, width=4),
        marker=dict(color=HOT_PINK, size=10, line=dict(color="white", width=2)),
    ))

    fig_radar.update_layout(
        title="Oyun türü tercihleri (%)",
        polar=dict(
            bgcolor="rgba(255,255,255,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 50],
                ticksuffix="%",
                gridcolor="rgba(139,124,255,0.18)"
            ),
            angularaxis=dict(gridcolor="rgba(139,124,255,0.18)")
        ),
        showlegend=False
    )

    fig_radar = apply_theme(fig_radar, height=375)
    st.plotly_chart(fig_radar, use_container_width=True, config=PLOT_CONFIG)

with oc2:
    bagimlilik = {
        "Çok fazla zaman harcıyor": "dijital oyunda çok fazla zaman harcama",
        "Ailesi fazla oynadığını düşünüyor": "ailesi çok fazla dijital oyun oynadığını düşünüyor",
        "Planlandan fazla oynuyor": "planlanan süreden fazla dijital oyun oynama",
        "Sorumlulukları aksatıyor": "fazla dijital oyun oynadığı için sorumluluklarını aksatıyor",
        "Oynamayınca mutsuz": "dijital oyun oynamadığı zaman kendini mutsuz hissediyor",
    }

    bag_df = pd.DataFrame({
        "Durum": list(bagimlilik.keys()),
        "Oran": [yes_pct(flt, c) for c in bagimlilik.values()],
    }).sort_values("Oran", ascending=True)

    fig_bag = px.bar(
        bag_df,
        x="Oran",
        y="Durum",
        orientation="h",
        text="Oran",
        color="Durum",
        color_discrete_sequence=[GREEN, MINT, YELLOW, ORANGE, RED]
    )

    fig_bag.update_traces(
        texttemplate="%{text}%",
        textposition="outside",
        marker_line=dict(color="white", width=2)
    )

    fig_bag.update_layout(
        title="Dijital oyun davranış göstergeleri",
        xaxis=dict(range=[0, 45], ticksuffix="%"),
        yaxis=dict(title=""),
        showlegend=False
    )

    fig_bag = apply_theme(fig_bag, height=375)
    st.plotly_chart(fig_bag, use_container_width=True, config=PLOT_CONFIG)

section_title("Ekranın Bedeli")

etkiler = {
    "Az ders çalışma": "ekran başında fazla zaman geçirmekten az ders çalışma",
    "Az kitap okuma": "ekran başında fazla zaman geçirmekten az kitap okuma",
    "Aile ile az zaman": "ekran başında fazla zaman geçirmekten aile ile daha az zaman geçirme",
    "Arkadaşlarla az görüşme": "ekran başında fazla zaman geçirmekten arkadasları ile az görüşme",
    "Az uyku": "ekran başında fazla zaman geçirmekten az uyku",
}

etki_df = pd.DataFrame({
    "Etki": list(etkiler.keys()),
    "Oran": [yes_pct(flt, c) for c in etkiler.values()],
})

compare_col = st.selectbox(
    "Ekran etkilerini hangi değişkene göre karşılaştıralım?",
    list(etkiler.keys())
)

etki_selected_col = etkiler[compare_col]

grouped_etki = (
    flt.groupby(compare_by)[etki_selected_col]
    .apply(lambda x: round((pd.to_numeric(x, errors="coerce") == 1).mean() * 100, 1))
    .reset_index(name="Oran")
)

if compare_by == "cinsiyet":
    grouped_etki[compare_by] = grouped_etki[compare_by].map({1: "Erkek", 2: "Kız"})

ec1, ec2 = st.columns([1, 1.2])

with ec1:
    fig_etki = px.bar(
        etki_df.sort_values("Oran", ascending=True),
        x="Oran",
        y="Etki",
        orientation="h",
        text="Oran",
        color="Etki",
        color_discrete_sequence=[YELLOW, ORANGE, HOT_PINK, MINT, BLUE]
    )

    fig_etki.update_traces(
        texttemplate="%{text}%",
        textposition="outside",
        marker_line=dict(color="white", width=2)
    )

    fig_etki.update_layout(
        title="Ekranın olumsuz etkileri",
        xaxis=dict(title="Oran (%)", range=[0, 50]),
        yaxis=dict(title=""),
        showlegend=False
    )

    fig_etki = apply_theme(fig_etki, height=345)
    st.plotly_chart(fig_etki, use_container_width=True, config=PLOT_CONFIG)

with ec2:
    is_numeric = pd.api.types.is_numeric_dtype(grouped_etki[compare_by])
    chart_type = "line" if (compare_by == "yaş" and is_numeric) else "bar"

    if chart_type == "line":
        fig_compare = px.area(
            grouped_etki,
            x=compare_by,
            y="Oran",
            color_discrete_sequence=[HOT_PINK],
            markers=True,
            title=f"{compare_col} — {compare_by} bazında"
        )

        fig_compare.update_traces(
            line=dict(width=4, shape="spline"),
            marker=dict(size=10, color=HOT_PINK, line=dict(color="white", width=3)),
            fillcolor="rgba(255,111,174,0.20)"
        )

    else:
        fig_compare = px.bar(
            grouped_etki,
            x=compare_by,
            y="Oran",
            text="Oran",
            color=compare_by,
            color_discrete_sequence=[PURPLE, HOT_PINK, BLUE],
            title=f"{compare_col} — {compare_by} bazında"
        )

        fig_compare.update_traces(
            texttemplate="%{text}%",
            textposition="outside",
            marker_line=dict(color="white", width=2)
        )

    fig_compare.update_layout(
        yaxis=dict(title="Oran (%)", range=[0, 55]),
        xaxis=dict(title="", dtick=1 if compare_by == "yaş" else None),
        showlegend=False
    )

    fig_compare = apply_theme(fig_compare, height=345)
    st.plotly_chart(fig_compare, use_container_width=True, config=PLOT_CONFIG)

section_title("Telefon Kullanım Alışkanlıkları")

bagimlilik_tel = {
    "Her yarım saatte kontrol": "çocuk yarım saatte bir telefonunu kontrol ediyor",
    "Uyumadan önce kontrol": "uyumadan son iş olarak telefon kontrol eder",
    "Uyanınca ilk kontrol": "uyanınca ilk iş olarak telefon kontrol eder",
    "TV izlerken kullanma": "tv izlerken telefon kullanma",
    "Yemekte kullanma": "başkaları ile yemek yerken telefon kullanma",
}

tel_df = pd.DataFrame({
    "Davranış": list(bagimlilik_tel.keys()),
    "Oran": [yes_pct(flt, c) for c in bagimlilik_tel.values()],
})

cols_bag = st.columns(5)

habit_cards = [
    ("#EDEAFF", "#5847C6"),
    ("#FFE4EE", "#D03F7A"),
    ("#DFF0FF", "#1973CF"),
    ("#DDF8F1", "#118C70"),
    ("#FFF0C4", "#BA7100"),
]

for i, (_, row) in enumerate(tel_df.iterrows()):
    bg, fg = habit_cards[i]
    with cols_bag[i]:
        st.markdown(f"""
        <div style="
            background:{bg};
            color:{fg};
            border-radius:30px;
            padding:24px 14px;
            text-align:center;
            min-height:145px;
            display:flex;
            flex-direction:column;
            justify-content:center;
            gap:10px;
            box-shadow:0 18px 36px rgba(60,52,137,0.10);
            border:4px solid rgba(255,255,255,0.82);
        ">
          <div style="font-size:36px;font-weight:900;">%{fmt_pct(row['Oran'])}</div>
          <div style="font-size:12px;font-weight:900;opacity:0.76;">{row['Davranış']}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

with st.expander("Filtrelenmiş ham veriyi göster"):
    st.dataframe(flt, use_container_width=True, height=300)
    csv = flt.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "CSV olarak indir",
        csv,
        "filtrelenmis_veri.csv",
        "text/csv"
    )