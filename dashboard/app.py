import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Türkiye Dijitalleşme Dashboardu", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("../data/bütünveriler.csv", low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    return df

df = load_data()

year_col = "referans yıl"
age_col = "yaş"

region_candidates = [c for c in df.columns if "düzey 1" in c or "bölge" in c]
region_col = region_candidates[0] if region_candidates else None

digital_vars = [
    "en son internet ne zaman kullanıldı",
    "internet kullanım sıklığı",
    "bilgisayar son kullanım zamanı",
    "sosyal medya kullanımı",
    "eposta kullanımı",
    "internet araması",
    "online haber okuma",
    "ürün hizmet arama",
    "edevlet hakkında bilgi",
    "eticaret kullanımı"
]

for col in digital_vars:
    if col in df.columns:
        df[col] = df[col].replace([97, 98, 99, 999], np.nan)

maps = {
    "en son internet ne zaman kullanıldı": {1: 1, 2: 0.75, 3: 0.5, 4: 0},
    "internet kullanım sıklığı": {1: 1, 2: 0.75, 3: 0.5, 4: 0.25, 5: 0},
    "bilgisayar son kullanım zamanı": {1: 1, 2: 0.75, 3: 0.5, 4: 0}
}

for col, mp in maps.items():
    if col in df.columns:
        df[col] = df[col].map(mp)

binary_cols = [
    "sosyal medya kullanımı",
    "eposta kullanımı",
    "internet araması",
    "online haber okuma",
    "ürün hizmet arama",
    "edevlet hakkında bilgi",
    "eticaret kullanımı"
]

for col in binary_cols:
    if col in df.columns:
        df[col] = np.where(df[col] == 1, 1, 0)

available_vars = [c for c in digital_vars if c in df.columns]
df["digitalization_index"] = df[available_vars].mean(axis=1)

df["digitalization_level"] = pd.cut(
    df["digitalization_index"],
    bins=[0, 0.33, 0.66, 1],
    labels=["Düşük", "Orta", "Yüksek"],
    include_lowest=True
)

df["age_category"] = pd.cut(
    df[age_col],
    bins=[0, 18, 30, 45, 60, 120],
    labels=["0-18", "19-30", "31-45", "46-60", "60+"],
    include_lowest=True
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
h1, h2, h3 {
    color: #17233c;
}
[data-testid="stSidebar"] {
    background-color: #eef3f8;
}
</style>
""", unsafe_allow_html=True)

st.title("Türkiye Dijitalleşme Gösterge Paneli")
st.caption("TÜİK Hanehalkı Bilişim Teknolojileri Araştırması")

st.sidebar.header("Filtreler")

years = sorted(df[year_col].dropna().unique())
selected_year = st.sidebar.selectbox("Yıl seç", years, index=len(years)-1)

filtered = df[df[year_col] == selected_year].copy()

if region_col:
    valid_regions = sorted(filtered[region_col].dropna().astype(str).unique())
    selected_region = st.sidebar.selectbox("Bölge seç", ["Tümü"] + valid_regions)

    if selected_region != "Tümü":
        filtered = filtered[filtered[region_col].astype(str) == selected_region]

level_filter = st.sidebar.selectbox(
    "Dijitalleşme seviyesi",
    ["Tümü", "Düşük", "Orta", "Yüksek"]
)

if level_filter != "Tümü":
    filtered = filtered[filtered["digitalization_level"].astype(str) == level_filter]

st.subheader(f"{selected_year} Yılı Genel Özet")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric("Ortalama Endeks", f"{filtered['digitalization_index'].mean():.2f}")
kpi2.metric("Ortalama Yaş", f"{filtered[age_col].mean():.1f}")
kpi3.metric("Gözlem Sayısı", f"{len(filtered):,}")

if region_col:
    kpi4.metric("Bölge Sayısı", filtered[region_col].nunique())
else:
    kpi4.metric("Bölge Sayısı", "-")

st.divider()

st.subheader("Yıllara Göre Dijitalleşme Trendi")

yearly_df = (
    df.groupby(year_col)["digitalization_index"]
    .mean()
    .reset_index()
)

fig_year = px.line(
    yearly_df,
    x=year_col,
    y="digitalization_index",
    markers=True,
    text=yearly_df["digitalization_index"].round(2)
)

fig_year.update_traces(textposition="top center")
fig_year.update_layout(
    height=430,
    xaxis_title="Yıl",
    yaxis_title="Ortalama Dijitalleşme Endeksi"
)

st.plotly_chart(fig_year, use_container_width=True)

st.divider()

if region_col and filtered[region_col].notna().sum() > 0:
    st.subheader("Bölgelere Göre Dijitalleşme")

    regional_df = (
        filtered.groupby(region_col)["digitalization_index"]
        .mean()
        .reset_index()
        .sort_values("digitalization_index", ascending=False)
    )

    fig_region = px.bar(
        regional_df,
        x="digitalization_index",
        y=region_col,
        orientation="h",
        text="digitalization_index",
        color="digitalization_index",
        color_continuous_scale="Blues"
    )

    fig_region.update_traces(texttemplate="%{text:.2f}")
    fig_region.update_layout(
        height=520,
        xaxis_title="Ortalama Endeks",
        yaxis_title="Bölge",
        yaxis=dict(autorange="reversed"),
        showlegend=False
    )

    st.plotly_chart(fig_region, use_container_width=True)

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Yaş Gruplarına Göre Dijitalleşme")

    age_df = (
        filtered.groupby("age_category")["digitalization_index"]
        .mean()
        .reset_index()
    )

    fig_age = px.bar(
        age_df,
        x="age_category",
        y="digitalization_index",
        text="digitalization_index",
        color="digitalization_index",
        color_continuous_scale="Viridis"
    )

    fig_age.update_traces(texttemplate="%{text:.2f}")
    fig_age.update_layout(
        height=420,
        xaxis_title="Yaş Grubu",
        yaxis_title="Ortalama Endeks",
        showlegend=False
    )

    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    st.subheader("Dijitalleşme Seviyesi Dağılımı")

    level_df = (
        filtered["digitalization_level"]
        .value_counts()
        .reset_index()
    )

    level_df.columns = ["Seviye", "Kişi Sayısı"]

    fig_level = px.pie(
        level_df,
        names="Seviye",
        values="Kişi Sayısı",
        hole=0.45
    )

    fig_level.update_layout(height=420)

    st.plotly_chart(fig_level, use_container_width=True)

st.divider()

st.subheader("Dijital Göstergelerin Ortalama Kullanımı")

indicator_df = (
    filtered[available_vars]
    .mean()
    .reset_index()
)

indicator_df.columns = ["Gösterge", "Ortalama"]

fig_indicators = px.bar(
    indicator_df.sort_values("Ortalama", ascending=True),
    x="Ortalama",
    y="Gösterge",
    orientation="h",
    text="Ortalama",
    color="Ortalama",
    color_continuous_scale="Blues"
)

fig_indicators.update_traces(texttemplate="%{text:.2f}")
fig_indicators.update_layout(
    height=550,
    xaxis_title="Ortalama Kullanım / Puan",
    yaxis_title="",
    showlegend=False
)

st.plotly_chart(fig_indicators, use_container_width=True)

st.divider()

st.subheader("Veri Tablosu")

show_cols = [year_col, age_col, "age_category", "digitalization_index", "digitalization_level"]

if region_col:
    show_cols.insert(1, region_col)

st.dataframe(
    filtered[show_cols].head(500),
    use_container_width=True
)