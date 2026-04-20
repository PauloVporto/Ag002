from __future__ import annotations

import streamlit as st

from src.penguin_model import ISLAND_MAP, SEX_MAP, predict_species, train_model


st.set_page_config(
    page_title="Classificador de Pinguins",
    page_icon="🐧",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_training_artifacts():
    return train_model()


artifacts = get_training_artifacts()

SPECIES_LABELS = {0: "Adeline", 1: "Chinstrap", 2: "Gentoo"}


def inject_css() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=DM+Mono:wght@400;500&display=swap');

            html, body, [class*="css"] {
                font-family: 'DM Sans', sans-serif;
                color: #1a2211;
            }

            .stApp { background: #f0f4ee; }

            /* ── Hero ── */
            .hero {
                background: #1b4332;
                border-radius: 20px;
                padding: 28px 32px;
                margin-bottom: 1.4rem;
                position: relative;
                overflow: hidden;
            }
            .hero::before {
                content: '';
                position: absolute;
                top: -40px; right: -40px;
                width: 220px; height: 220px;
                border-radius: 50%;
                background: rgba(255,255,255,0.04);
            }
            .hero::after {
                content: '🐧';
                font-size: 100px;
                position: absolute;
                right: 28px; bottom: -14px;
                opacity: 0.18;
                pointer-events: none;
            }
            .hero-title {
                font-size: 2rem;
                font-weight: 700;
                color: #fff;
                letter-spacing: -0.4px;
                line-height: 1.2;
            }
            .hero-sub {
                font-size: 0.9rem;
                color: #95d5b2;
                margin-top: 6px;
            }
            .hero-pill {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                background: rgba(255,255,255,0.1);
                border: 1px solid rgba(255,255,255,0.15);
                border-radius: 100px;
                padding: 5px 14px;
                font-size: 0.8rem;
                color: #d8f3dc;
                margin-top: 14px;
            }

            /* ── Metric cards ── */
            .metric-card {
                background: #fff;
                border: 1px solid #dde5d8;
                border-radius: 14px;
                padding: 20px 18px;
                display: flex;
                flex-direction: column;
                gap: 2px;
            }
            .metric-icon { font-size: 1.3rem; margin-bottom: 4px; }
            .metric-label {
                font-size: 0.72rem;
                color: #7d9070;
                text-transform: uppercase;
                letter-spacing: 0.9px;
            }
            .metric-value {
                font-size: 2.1rem;
                font-weight: 700;
                color: #2d6a4f;
                line-height: 1.1;
            }
            .metric-sub { font-size: 0.75rem; color: #9aab8e; }

            /* ── Tabs ── */
            .stTabs [data-baseweb="tab-list"] {
                gap: 4px;
                background: transparent;
                border-bottom: 2px solid #dde5d8;
                padding-bottom: 0;
            }
            .stTabs [data-baseweb="tab"] {
                background: transparent !important;
                border: none !important;
                border-bottom: 2px solid transparent !important;
                border-radius: 0 !important;
                color: #7d9070;
                font-weight: 500;
                padding: 0.5rem 1.1rem;
                font-size: 0.9rem;
                margin-bottom: -2px;
            }
            .stTabs [aria-selected="true"] {
                color: #2d6a4f !important;
                border-bottom: 2px solid #2d6a4f !important;
            }
            .stTabs [data-baseweb="tab-panel"] { padding-top: 1.2rem; }

            /* ── Cards ── */
            .card {
                background: #fff;
                border: 1px solid #dde5d8;
                border-radius: 16px;
                padding: 22px 24px;
            }
            .card-title {
                font-size: 0.95rem;
                font-weight: 600;
                color: #1b4332;
                margin-bottom: 14px;
                display: flex;
                align-items: center;
                gap: 8px;
            }

            /* ── Species badges ── */
            .badge {
                display: inline-block;
                padding: 2px 10px;
                border-radius: 100px;
                font-size: 0.75rem;
                font-weight: 500;
            }
            .badge-0 { background: #fde8d0; color: #7c3510; }
            .badge-1 { background: #d4e8f8; color: #1a457a; }
            .badge-2 { background: #d4f0e4; color: #165c38; }

            /* ── Form ── */
            .stSelectbox label,
            .stSlider label {
                color: #1a2211 !important;
                font-weight: 600 !important;
                font-size: 0.85rem !important;
            }

            /* ── Buttons ── */
            .stButton > button {
                width: 100%;
                background: #2d6a4f;
                color: #fff;
                border: none;
                border-radius: 12px;
                padding: 0.8rem 1rem;
                font-weight: 600;
                font-size: 0.95rem;
                letter-spacing: 0.1px;
                transition: background 0.15s, transform 0.1s;
            }
            .stButton > button:hover {
                background: #1b4332;
                border: none;
                transform: translateY(-1px);
            }
            .stButton > button:active { transform: translateY(0); }

            /* ── Result ── */
            .result-wrap {
                background: #1b4332;
                border-radius: 14px;
                padding: 22px;
                text-align: center;
                margin-top: 14px;
                animation: pop .25s ease;
            }
            @keyframes pop {
                from { opacity: 0; transform: scale(.96); }
                to   { opacity: 1; transform: scale(1); }
            }
            .result-emoji { font-size: 2.6rem; }
            .result-eyebrow {
                font-size: 0.68rem;
                text-transform: uppercase;
                letter-spacing: 1.2px;
                color: #95d5b2;
                margin-top: 8px;
            }
            .result-name {
                font-size: 2rem;
                font-weight: 700;
                color: #fff;
                margin-top: 2px;
            }
            .result-desc {
                font-size: 0.8rem;
                color: #95d5b2;
                margin-top: 6px;
                line-height: 1.5;
            }

            /* ── Info callout ── */
            .callout {
                background: #eaf5ee;
                border-left: 3px solid #52b788;
                border-radius: 0 10px 10px 0;
                padding: 10px 14px;
                font-size: 0.82rem;
                color: #1b4d2e;
                margin-top: 14px;
                line-height: 1.5;
            }

            /* ── Tips table ── */
            .tip-row {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 9px 0;
                border-bottom: 1px solid #f0f4ee;
                font-size: 0.85rem;
                gap: 12px;
            }
            .tip-row:last-child { border-bottom: none; }
            .tip-hint { color: #7d9070; font-size: 0.8rem; }

            h2, h3 { color: #1b4332 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Classificador de Pinguins</div>
        <div class="hero-sub">Decision Tree · Palmer Penguins Dataset · AG2</div>
        <div class="hero-pill">
            👤 Paulo Vicente de Carvalho Porto &nbsp;·&nbsp; Romulo Coutinho
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Metrics ───────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
metrics = [
    ("🗂️", "Amostras válidas", len(artifacts.processed_data), "após limpeza do CSV"),
    ("📚", "Treinamento", len(artifacts.x_train), "80% do dataset"),
    ("🎯", "Acurácia", f"{artifacts.accuracy * 100:.1f}%", "no conjunto de teste"),
]
for col, (icon, label, value, sub) in zip([c1, c2, c3], metrics):
    with col:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-icon">{icon}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-sub">{sub}</div>
            </div>""",
            unsafe_allow_html=True,
        )

st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Distribuição", "🔬 Dados", "📋 Avaliação", "🐧 Classificar"]
)

# ── Tab 1: Distribution ───────────────────────────────────────────────────────
with tab1:
    left, right = st.columns(2, gap="small")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Espécies no dataset</div>', unsafe_allow_html=True)

        counts = artifacts.data.dropna()["species"].value_counts()
        total = counts.sum()
        bar_cfg = {
            "Adelie":    ("#d97b3d", "Adeline"),
            "Chinstrap": ("#3d82c8", "Chinstrap"),
            "Gentoo":    ("#2d9e6e", "Gentoo"),
        }
        bars = ""
        for sp, cnt in counts.items():
            color, label = bar_cfg.get(sp, ("#888", sp))
            pct = int(cnt / total * 100)
            bars += f"""
            <div style="margin-bottom:16px">
                <div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:0.85rem">
                    <span style="font-weight:500">{label}</span>
                    <span style="color:#7d9070">{cnt} · {pct}%</span>
                </div>
                <div style="background:#f0f4ee;border-radius:6px;height:10px;overflow:hidden">
                    <div style="width:{pct}%;height:100%;background:{color};border-radius:6px"></div>
                </div>
            </div>"""
        st.markdown(bars, unsafe_allow_html=True)
        st.markdown(
            '<div class="callout">Adeline representa ~44% das amostras. A Decision Tree lida bem com esse leve desbalanceamento.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🗺️ Distribuição por ilha</div>', unsafe_allow_html=True)

        island_counts = artifacts.data.dropna()["island"].value_counts()
        total_i = island_counts.sum()
        island_colors = {"Biscoe": "#5b7fa6", "Dream": "#7a6ea6", "Torgersen": "#a6836e"}
        ibars = ""
        for isl, cnt in island_counts.items():
            color = island_colors.get(isl, "#888")
            pct = int(cnt / total_i * 100)
            ibars += f"""
            <div style="margin-bottom:16px">
                <div style="display:flex;justify-content:space-between;margin-bottom:6px;font-size:0.85rem">
                    <span style="font-weight:500">{isl}</span>
                    <span style="color:#7d9070">{cnt} · {pct}%</span>
                </div>
                <div style="background:#f0f4ee;border-radius:6px;height:10px;overflow:hidden">
                    <div style="width:{pct}%;height:100%;background:{color};border-radius:6px"></div>
                </div>
            </div>"""
        st.markdown(ibars, unsafe_allow_html=True)
        st.markdown(
            '<div class="callout">Gentoo habita quase exclusivamente a ilha Biscoe. Chinstrap é predominante em Dream.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ── Tab 2: Data ───────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📄 Dados originais (10 primeiras linhas)</div>', unsafe_allow_html=True)
    st.dataframe(artifacts.data.head(10), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">✅ Após limpeza e conversão numérica</div>', unsafe_allow_html=True)
    st.dataframe(artifacts.processed_data.head(10), use_container_width=True)
    st.markdown(
        '<div class="callout"><strong>Nota:</strong> "Adeline" (enunciado) = "Adelie" no dataset — mesma classe, código 0.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ── Tab 3: Evaluation ─────────────────────────────────────────────────────────
with tab3:
    rl, rr = st.columns(2, gap="small")

    with rl:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📋 Relatório de classificação</div>', unsafe_allow_html=True)
        st.code(artifacts.report_text, language="text")
        st.markdown("</div>", unsafe_allow_html=True)

    with rr:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔍 Real vs Previsto</div>', unsafe_allow_html=True)

        comparison = artifacts.x_test.copy()
        comparison["Real"] = artifacts.y_test.map(SPECIES_LABELS)
        comparison["Previsto"] = artifacts.predictions.map(SPECIES_LABELS)
        comparison["Acerto"] = comparison["Real"] == comparison["Previsto"]

        st.dataframe(
            comparison.head(15),
            use_container_width=True,
            column_config={"Acerto": st.column_config.CheckboxColumn("✓")},
        )
        st.markdown("</div>", unsafe_allow_html=True)

# ── Tab 4: Classify ───────────────────────────────────────────────────────────
with tab4:
    fl, fr = st.columns([1.15, 0.85], gap="small")

    with fl:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🐧 Dados do pinguim</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            island_name = st.selectbox("Ilha", list(ISLAND_MAP.keys()))
        with col_b:
            sex_name = st.selectbox("Sexo", list(SEX_MAP.keys()))

        culmen_length = st.slider("Culmen length (mm)", 30.0, 65.0, 45.2, 0.1, format="%.1f mm")
        culmen_depth = st.slider("Culmen depth (mm)", 13.0, 25.0, 17.1, 0.1, format="%.1f mm")
        flipper_length = st.slider("Flipper length (mm)", 170.0, 240.0, 210.0, 1.0, format="%.0f mm")
        body_mass = st.slider("Body mass (g)", 2500.0, 6500.0, 4200.0, 50.0, format="%.0f g")

        if st.button("Classificar →", use_container_width=True):
            _, species_name = predict_species(
                artifacts.model,
                island=ISLAND_MAP[island_name],
                sex=SEX_MAP[sex_name],
                culmen_length=culmen_length,
                culmen_depth=culmen_depth,
                flipper_length=flipper_length,
                body_mass=body_mass,
            )

            descs = {
                "Adeline":   "Bico curto e largo. Encontrada nas três ilhas do arquipélago.",
                "Chinstrap": "Listra preta sob o queixo. Prefere a ilha Dream.",
                "Gentoo":    "A maior das três. Nadadeiras longas, quase exclusiva da Biscoe.",
            }

            st.markdown(
                f"""
                <div class="result-wrap">
                    <div class="result-emoji">🐧</div>
                    <div class="result-eyebrow">Espécie identificada</div>
                    <div class="result-name">{species_name}</div>
                    <div class="result-desc">{descs.get(species_name, '')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    with fr:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">💡 Dicas de identificação</div>', unsafe_allow_html=True)

        tips = [
            ("Adeline",   "badge-0", "Bico curto · profundidade > 17mm"),
            ("Chinstrap", "badge-1", "Ilha Dream · bico fino"),
            ("Gentoo",    "badge-2", "Nadadeira > 210mm · Biscoe"),
        ]
        tips_html = "".join(
            f'<div class="tip-row"><span class="badge {cls}">{name}</span>'
            f'<span class="tip-hint">{hint}</span></div>'
            for name, cls, hint in tips
        )
        st.markdown(tips_html, unsafe_allow_html=True)

        st.markdown(
            '<div class="callout">Ajuste os sliders e veja como o modelo reage a diferentes combinações.</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="card-title">🗺️ Mapeamentos numéricos</div>', unsafe_allow_html=True)

        mappings = [
            ("Biscoe / Dream / Torgersen", "0 / 1 / 2"),
            ("FEMALE / MALE", "0 / 1"),
            ("Adeline / Chinstrap / Gentoo", "0 / 1 / 2"),
        ]
        map_html = "".join(
            f"""<div style="display:flex;justify-content:space-between;align-items:center;
                            padding:8px 0;border-bottom:1px solid #f0f4ee;font-size:0.82rem">
                <span style="color:#5a6b4e">{k}</span>
                <span style="font-family:'DM Mono',monospace;color:#1b4332;font-weight:500">{v}</span>
            </div>"""
            for k, v in mappings
        )
        st.markdown(map_html, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)