
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


st.set_page_config(
    page_title="Happiness 2023",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global style ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

[data-testid="stSidebar"] {
    background: #0f1923;
    border-right: 1px solid #1e2d3d;
}
[data-testid="stSidebar"] * { color: #c8d8e8 !important; }
[data-testid="stSidebar"] .stRadio label { 
    padding: 6px 10px; border-radius: 6px; cursor: pointer;
    transition: background 0.15s;
}
[data-testid="stSidebar"] .stRadio label:hover { background: #1e2d3d; }

.metric-card {
    background: #0f1923;
    border: 1px solid #1e2d3d;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
}
.metric-card .val { font-size: 2rem; font-weight: 700; color: #4fc3f7; font-family: 'DM Mono'; }
.metric-card .lbl { font-size: 0.78rem; color: #7a9bb5; text-transform: uppercase; letter-spacing: .08em; margin-top: 4px; }

.page-title {
    font-size: 1.8rem; font-weight: 700; color: #e8f4fd;
    border-left: 4px solid #4fc3f7; padding-left: 14px;
    margin-bottom: 24px;
}
.insight-box {
    background: #0d1f2d;
    border-left: 3px solid #4fc3f7;
    border-radius: 0 8px 8px 0;
    padding: 12px 18px;
    font-size: 0.9rem;
    color: #b0cfe0;
    margin-top: 12px;
}

section[data-testid="stMain"] { background: #07111a; }
.stSelectbox label, .stSlider label { color: #b0cfe0 !important; font-size: 0.85rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib theme ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1f2d',
    'axes.facecolor':   '#0d1f2d',
    'axes.edgecolor':   '#1e3a50',
    'axes.labelcolor':  '#b0cfe0',
    'xtick.color':      '#7a9bb5',
    'ytick.color':      '#7a9bb5',
    'text.color':       '#c8d8e8',
    'grid.color':       '#1e3a50',
    'grid.linestyle':   '--',
    'grid.alpha':       0.5,
    'font.family':      'DejaVu Sans',
})
ACCENT  = '#4fc3f7'
RED     = '#ef5350'
GREEN   = '#66bb6a'
PALETTE = [ACCENT, '#f48fb1', '#ce93d8', '#ffcc80', '#80cbc4', '#a5d6a7']

# ── Load & prepare data ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('/mnt/user-data/uploads/predictorsOfHappiness2023.csv')
    df['temp_group'] = df['avg_temp'].apply(
        lambda x: 'Cold (<10°C)' if x < 10 else ('Hot (>25°C)' if x > 25 else 'Moderate')
    )
    df['burden_index'] = (
        (df['out_of_pocket_health_expenditure_percent'] / df['out_of_pocket_health_expenditure_percent'].max()) +
        (df['working_hours']           / df['working_hours'].max()) +
        (df['unemployment_rate_percent'] / df['unemployment_rate_percent'].max())
    ) / 3
    df['log_gdp'] = np.log1p(df['gdp_per_capita'])
    return df

df = load_data()
num_cols = df.select_dtypes(include='number').columns.tolist()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 😊 Happiness 2023")
    st.markdown("---")
    page = st.radio("", [
        "🏠  Overview",
        "📊  Distributions",
        "🔍  Outliers",
        "🔗  Correlations",
        "📈  Feature vs Target",
        "⚖️  Burden Index",
        "🌍  Country Explorer",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<span style='font-size:.75rem;color:#3a5a70'>130 countries · 24 variables · 2023</span>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown('<div class="page-title">World Happiness 2023</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#7a9bb5;margin-top:-16px;margin-bottom:24px'>Exploratory analysis of socioeconomic predictors across 130 countries.</p>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in zip(
        [c1, c2, c3, c4],
        ["130", "24", "0", "0"],
        ["Countries", "Variables", "Missing Values", "Duplicates"]
    ):
        col.markdown(f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**🏆 Top 10 Happiest**")
        top10 = df.nsmallest(10, 'rank')[['country', 'happiness_score']].reset_index(drop=True)
        top10.index += 1
        st.dataframe(top10, use_container_width=True, height=370)
    with col_r:
        st.markdown("**😔 Bottom 10 Least Happy**")
        bot10 = df.nlargest(10, 'rank')[['country', 'happiness_score']].reset_index(drop=True)
        bot10.index += 1
        st.dataframe(bot10, use_container_width=True, height=370)

    st.markdown("---")
    st.markdown("**📋 Full Dataset**")
    st.dataframe(df, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Distributions
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊  Distributions":
    st.markdown('<div class="page-title">Distributions</div>', unsafe_allow_html=True)

    selected = st.selectbox("Variable", num_cols, index=num_cols.index('happiness_score'))

    fig, ax = plt.subplots(figsize=(11, 4))
    sns.histplot(df[selected], bins=20, kde=True, color=ACCENT, edgecolor='#07111a', alpha=0.8, ax=ax)
    ax.axvline(df[selected].mean(),   color=RED,   linestyle='--', lw=1.5, label=f'Mean {df[selected].mean():.2f}')
    ax.axvline(df[selected].median(), color=GREEN, linestyle='--', lw=1.5, label=f'Median {df[selected].median():.2f}')
    ax.set_title(f'Distribution — {selected.replace("_"," ").title()}', pad=12)
    ax.legend(framealpha=0)
    ax.set_xlabel("")
    fig.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("**Summary Statistics**")
    st.dataframe(df[[selected]].describe().T.style.format("{:.3f}"), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Outliers
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Outliers":
    st.markdown('<div class="page-title">Outlier Detection</div>', unsafe_allow_html=True)

    selected = st.selectbox("Variable", num_cols, index=num_cols.index('happiness_score'))

    Q1, Q3 = df[selected].quantile(0.25), df[selected].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    outliers = df[(df[selected] < lower) | (df[selected] > upper)]

    col1, col2 = st.columns([1, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(5, 6))
        bp = ax.boxplot(df[selected].dropna(), patch_artist=True,
                        medianprops=dict(color=ACCENT, lw=2),
                        boxprops=dict(facecolor='#1e3a50', color='#4a7a9b'),
                        whiskerprops=dict(color='#4a7a9b'),
                        capprops=dict(color='#4a7a9b'),
                        flierprops=dict(marker='o', color=RED, markersize=6, alpha=0.7))
        ax.set_title(selected.replace('_',' ').title())
        ax.set_xticks([])
        fig.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        for lbl, val in [("Q1", Q1), ("Q3", Q3), ("IQR", IQR), ("Lower bound", lower), ("Upper bound", upper)]:
            st.metric(lbl, f"{val:.3f}")
        st.metric("Outliers found", len(outliers))

    if len(outliers):
        st.markdown("**Outlier countries:**")
        st.dataframe(outliers[['country', selected]].reset_index(drop=True), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Correlations
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔗  Correlations":
    st.markdown('<div class="page-title">Correlation Analysis</div>', unsafe_allow_html=True)

    corr = df[num_cols].corr()

    st.markdown("**Heatmap**")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', mask=mask,
                linewidths=0.4, annot_kws={'size': 6.5}, ax=ax,
                cbar_kws={'shrink': 0.6})
    ax.set_title('Correlation Heatmap', pad=14)
    fig.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("**Correlation with Happiness Score**")
    ch = corr['happiness_score'].drop('happiness_score').sort_values()
    colors = [RED if x < 0 else GREEN for x in ch]
    fig, ax = plt.subplots(figsize=(10, 8))
    ch.plot(kind='barh', color=colors, ax=ax, edgecolor='none')
    ax.axvline(0, color='#4a7a9b', lw=0.8)
    ax.set_title('Predictors of Happiness (Pearson r)', pad=12)
    ax.set_xlabel('Correlation coefficient')
    fig.tight_layout()
    st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Feature vs Target
# ════════════════════════════════════════════════════════════════════════════
elif page == "📈  Feature vs Target":
    st.markdown('<div class="page-title">Feature vs Happiness Score</div>', unsafe_allow_html=True)

    feat_cols = [c for c in num_cols if c not in ['happiness_score', 'rank']]
    selected = st.selectbox("Feature", feat_cols, index=feat_cols.index('gdp_per_capita'))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df[selected], df['happiness_score'], color=ACCENT, alpha=0.6, s=55, edgecolors='none')
    m, b, r, p, _ = stats.linregress(df[selected], df['happiness_score'])
    xs = np.linspace(df[selected].min(), df[selected].max(), 100)
    ax.plot(xs, m*xs+b, color=RED, lw=1.8)
    ax.set_xlabel(selected.replace('_',' ').title())
    ax.set_ylabel('Happiness Score')
    ax.set_title(f'{selected.replace("_"," ").title()} vs Happiness Score')
    fig.tight_layout()
    st.pyplot(fig); plt.close()

    direction = "positive 🟢" if r > 0 else "negative 🔴"
    strength  = "strong" if abs(r) > 0.5 else ("moderate" if abs(r) > 0.3 else "weak")
    st.markdown(f'<div class="insight-box">r = <b>{r:.3f}</b> — {strength} {direction} relationship</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Climate Zone vs Happiness**")
    fig, ax = plt.subplots(figsize=(8, 5))
    order = ['Cold (<10°C)', 'Moderate', 'Hot (>25°C)']
    pal   = ['#4A90D9', '#F5A623', '#E24B4A']
    sns.boxplot(data=df, x='temp_group', y='happiness_score', order=order, palette=pal, ax=ax)
    sns.stripplot(data=df, x='temp_group', y='happiness_score', order=order, palette=pal,
                  alpha=0.35, jitter=True, size=4, ax=ax)
    means = df.groupby('temp_group')['happiness_score'].mean()
    for i, z in enumerate(order):
        ax.text(i, means[z]+0.08, f"avg {means[z]:.2f}", ha='center', fontsize=8, color='white', fontweight='bold')
    ax.set_xlabel(""); ax.set_title("Happiness by Climate Zone")
    fig.tight_layout()
    st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Burden Index
# ════════════════════════════════════════════════════════════════════════════
elif page == "⚖️  Burden Index":
    st.markdown('<div class="page-title">Burden Index</div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#7a9bb5;margin-top:-16px'>Composite feature = average of normalized health cost + working hours + unemployment.</p>", unsafe_allow_html=True)

    r_val = df['burden_index'].corr(df['happiness_score'])

    fig, ax = plt.subplots(figsize=(11, 6))
    sc = ax.scatter(df['burden_index'], df['happiness_score'],
                    c=df['happiness_score'], cmap='RdYlGn', s=70, alpha=0.85, edgecolors='none', vmin=1, vmax=8)
    plt.colorbar(sc, ax=ax, label='Happiness Score')
    m, b, *_ = stats.linregress(df['burden_index'], df['happiness_score'])
    xs = np.linspace(df['burden_index'].min(), df['burden_index'].max(), 100)
    ax.plot(xs, m*xs+b, color=RED, lw=1.8, linestyle='--')

    highlight = ['Armenia', 'Finland', 'Afghanistan', 'Lebanon', 'Saudi Arabia', 'Bangladesh']
    for _, row in df[df['country'].isin(highlight)].iterrows():
        ax.annotate(row['country'], (row['burden_index'], row['happiness_score']),
                    fontsize=8, xytext=(6,3), textcoords='offset points', color='#c8d8e8')

    ax.set_xlabel('Burden Index  (0 = least burdened → 1 = most burdened)')
    ax.set_ylabel('Happiness Score')
    ax.set_title(f'Burden Index vs Happiness Score  (r = {r_val:.2f})')
    fig.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown(f'<div class="insight-box">Correlation r = <b>{r_val:.3f}</b> — countries with higher burden (health costs + overwork + unemployment) score significantly lower on happiness.</div>', unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Most Burdened**")
        st.dataframe(df[['country','burden_index','happiness_score']].sort_values('burden_index', ascending=False).head(10).reset_index(drop=True), use_container_width=True)
    with col_b:
        st.markdown("**Least Burdened**")
        st.dataframe(df[['country','burden_index','happiness_score']].sort_values('burden_index').head(10).reset_index(drop=True), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# PAGE 7 — Country Explorer
# ════════════════════════════════════════════════════════════════════════════
elif page == "🌍  Country Explorer":
    st.markdown('<div class="page-title">Country Explorer</div>', unsafe_allow_html=True)

    selected_country = st.selectbox("Country", sorted(df['country'].tolist()))
    row = df[df['country'] == selected_country].iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, lbl in zip(
        [c1, c2, c3, c4, c5],
        [f"#{int(row['rank'])}", f"{row['happiness_score']:.2f}",
         f"${row['gdp_per_capita']:,.0f}", f"{row['working_hours']:,.0f}", f"{row['burden_index']:.3f}"],
        ["Global Rank", "Happiness", "GDP / capita", "Working hrs/yr", "Burden Index"]
    ):
        col.markdown(f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    compare_vars = ['happiness_score','gdp_per_capita','working_hours',
                    'fertility_rate','physicians_per_thousand',
                    'gross_tertiary_education_enrollment_percent','burden_index']
    compare_labels = ['Happiness','GDP/capita','Work hrs','Fertility','Physicians/1k','Tertiary Edu %','Burden Index']

    # Normalize for radar-style bar comparison
    compare_df = pd.DataFrame({
        'Variable': compare_labels,
        selected_country: [row[v] for v in compare_vars],
        'Global Avg':     [df[v].mean() for v in compare_vars]
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(compare_df))
    w = 0.35
    ax.bar(x - w/2, compare_df[selected_country], w, label=selected_country, color=ACCENT, alpha=0.85)
    ax.bar(x + w/2, compare_df['Global Avg'],     w, label='Global Average',  color='#546e7a', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(compare_df['Variable'], rotation=20, ha='right', fontsize=9)
    ax.legend(framealpha=0)
    ax.set_title(f'{selected_country} vs Global Average')
    fig.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("**All Indicators**")
    country_data = row[num_cols].reset_index()
    country_data.columns = ['Indicator', 'Value']
    country_data['Indicator'] = country_data['Indicator'].str.replace('_',' ').str.title()
    st.dataframe(country_data, use_container_width=True)
