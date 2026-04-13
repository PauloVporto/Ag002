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


def inject_css() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

            html, body, [class*="css"] {
                font-family: 'Poppins', sans-serif;
                color: #243b53;
            }

            .stApp {
                background: linear-gradient(180deg, #f8fbff 0%, #eef3f8 100%);
            }

            .hero-box {
                background: #ffffff;
                border: 1px solid #d9e2ec;
                border-left: 6px solid #2f855a;
                border-radius: 18px;
                padding: 1.4rem 1.5rem;
                box-shadow: 0 8px 18px rgba(15, 23, 42, 0.05);
                margin-bottom: 1rem;
            }

            .main-title {
                font-size: 2.2rem;
                font-weight: 700;
                color: #102a43;
                margin-bottom: 0.35rem;
            }

            .subtitle {
                font-size: 1rem;
                line-height: 1.7;
                color: #486581;
            }

            .card {
                background: #ffffff;
                border: 1px solid #d9e2ec;
                border-radius: 16px;
                padding: 1rem 1.1rem;
                box-shadow: 0 6px 16px rgba(15, 23, 42, 0.04);
                margin-bottom: 0.5rem;
            }

            .metric-label {
                color: #627d98;
                font-size: 0.92rem;
                margin-bottom: 0.2rem;
            }

            .metric-value {
                color: #2f855a;
                font-size: 2rem;
                font-weight: 700;
            }

            .content-panel {
                background: #ffffff;
                border: 1px solid #d9e2ec;
                border-radius: 16px;
                padding: 1.2rem;
                box-shadow: 0 6px 16px rgba(15, 23, 42, 0.04);
            }

            .prediction-box {
                background: #2f855a;
                color: white;
                border-radius: 14px;
                padding: 1rem 1.15rem;
                margin-top: 1rem;
            }

            .prediction-title {
                font-size: 0.85rem;
                text-transform: uppercase;
                letter-spacing: 0.08rem;
                opacity: 0.85;
            }

            .prediction-species {
                font-size: 1.8rem;
                font-weight: 700;
                margin-top: 0.2rem;
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 0.4rem;
                margin-top: 0.75rem;
            }

            .stTabs [data-baseweb="tab"] {
                background: #ffffff;
                border: 1px solid #d9e2ec;
                border-radius: 10px 10px 0 0;
                color: #486581;
                font-weight: 600;
                padding: 0.65rem 1rem;
            }

            .stTabs [aria-selected="true"] {
                background: #2f855a !important;
                border-color: #2f855a !important;
                color: white !important;
            }

            h2, h3 {
                color: #102a43 !important;
            }

            .stSelectbox label,
            .stNumberInput label {
                color: #102a43 !important;
                font-weight: 600;
            }

            .stButton > button {
                width: 100%;
                background: #2f855a;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 0.78rem 1rem;
                font-weight: 600;
            }

            .stButton > button:hover {
                background: #276749;
            }

            .small-note {
                font-size: 0.92rem;
                color: #627d98;
                margin-top: 0.35rem;
            }

            .step-box {
                background: #f7fafc;
                border: 1px dashed #bcccdc;
                border-radius: 12px;
                padding: 0.85rem 0.95rem;
                margin-top: 0.75rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()

st.markdown(
    """
    <div class="hero-box">
        <div class="main-title">Classificador de Especies de Pinguins</div>
        <div class="subtitle">
            Projeto pratico da AG2 com leitura do CSV, tratamento dos dados,
            treinamento de um modelo Decision Tree e classificacao de novas amostras.
        </div>
        <div class="subtitle">
            Alunos: Paulo Vicente de Carvalho Porto e Romulo Coutinho
        </div>
        <div class="small-note">
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_col1, metric_col2, metric_col3 = st.columns(3)

with metric_col1:
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-label">Total de amostras validas</div>
            <div class="metric-value">{len(artifacts.processed_data)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metric_col2:
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-label">Amostras de treinamento</div>
            <div class="metric-value">{len(artifacts.x_train)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metric_col3:
    st.markdown(
        f"""
        <div class="card">
            <div class="metric-label">Acuracia do modelo</div>
            <div class="metric-value">{artifacts.accuracy * 100:.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Visao geral",
        "Pre-processamento",
        "Avaliacao",
        "Nova classificacao",
    ]
)

with tab1:
    left, right = st.columns([1.05, 0.95])

    with left:
        st.markdown('<div class="content-panel">', unsafe_allow_html=True)
        st.subheader("Resumo do projeto")
        st.write(
            """
            Nesta aplicacao, organizamos o trabalho em uma sequencia bem direta:
            carregar os dados, preparar o dataset, treinar o modelo e permitir
            que o usuario faca uma nova classificacao.
            """
        )
        st.markdown(
            """
            <div class="step-box">
                <strong>Passo 1:</strong> ler e tratar o arquivo CSV<br>
                <strong>Passo 2:</strong> converter os valores categoricos para inteiros<br>
                <strong>Passo 3:</strong> dividir os dados em treino e teste<br>
                <strong>Passo 4:</strong> treinar o modelo e analisar os resultados
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("Modelo escolhido: `Decision Tree`")
        st.write("Divisao do conjunto: `80% para treino` e `20% para teste`")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="content-panel">', unsafe_allow_html=True)
        st.subheader("Distribuicao das especies")
        st.write("Esse grafico mostra como as amostras estao divididas no dataset utilizado.")
        st.bar_chart(
            artifacts.data.dropna()["species"].value_counts().rename_axis("species")
        )
        st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="content-panel">', unsafe_allow_html=True)
    st.subheader("Dados originais")
    st.write("Aqui estao algumas linhas do CSV antes do tratamento.")
    st.dataframe(artifacts.data.head(10), use_container_width=True)

    st.subheader("Dados apos limpeza, conversao e reordenacao")
    st.write(
        "Depois da limpeza, os dados categoricos foram convertidos para numeros e as colunas foram colocadas na ordem pedida."
    )
    st.dataframe(artifacts.processed_data.head(10), use_container_width=True)

    st.caption(
        "Observacao: o dataset original usa 'Adelie'. Como o enunciado traz 'Adeline', "
        "o projeto considera os dois nomes como a mesma classe."
    )
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    result_left, result_right = st.columns(2)

    with result_left:
        st.markdown('<div class="content-panel">', unsafe_allow_html=True)
        st.subheader("Relatorio de classificacao")
        st.write("Essas metricas mostram o desempenho do modelo no conjunto de teste.")
        st.code(artifacts.report_text, language="text")
        st.markdown("</div>", unsafe_allow_html=True)

    with result_right:
        comparison = artifacts.x_test.copy()
        comparison["valor real"] = artifacts.y_test.map(
            {0: "Adeline", 1: "Chinstrap", 2: "Gentoo"}
        )
        comparison["previsao"] = artifacts.predictions.map(
            {0: "Adeline", 1: "Chinstrap", 2: "Gentoo"}
        )
        st.markdown('<div class="content-panel">', unsafe_allow_html=True)
        st.subheader("Exemplos do conjunto de teste")
        st.write("Comparacao entre o valor real e a previsao feita pelo modelo.")
        st.dataframe(comparison.head(12), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    form_col, info_col = st.columns([1.05, 0.95])

    with form_col:
        st.markdown('<div class="content-panel">', unsafe_allow_html=True)
        st.subheader("Teste uma nova classificacao")
        st.write(
            "Preencha os campos abaixo com os dados de um pinguim e clique no botao para ver a especie prevista."
        )

        island_name = st.selectbox("Ilha", list(ISLAND_MAP.keys()))
        sex_name = st.selectbox("Sexo", list(SEX_MAP.keys()))
        culmen_length = st.number_input(
            "Culmen length mm", min_value=30.0, max_value=65.0, value=45.2, step=0.1
        )
        culmen_depth = st.number_input(
            "Culmen depth mm", min_value=13.0, max_value=25.0, value=17.1, step=0.1
        )
        flipper_length = st.number_input(
            "Flipper length mm", min_value=170.0, max_value=240.0, value=210.0, step=1.0
        )
        body_mass = st.number_input(
            "Body mass g", min_value=2500.0, max_value=6500.0, value=4200.0, step=50.0
        )

        if st.button("Classificar pinguim", use_container_width=True):
            _, species_name = predict_species(
                artifacts.model,
                island=ISLAND_MAP[island_name],
                sex=SEX_MAP[sex_name],
                culmen_length=culmen_length,
                culmen_depth=culmen_depth,
                flipper_length=flipper_length,
                body_mass=body_mass,
            )

            st.markdown(
                f"""
                <div class="prediction-box">
                    <div class="prediction-title">Resultado previsto</div>
                    <div class="prediction-species">{species_name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    with info_col:
        st.markdown('<div class="content-panel">', unsafe_allow_html=True)
        st.subheader("Ajuda rapida")
        st.write("Os mapeamentos abaixo foram usados no pre-processamento do dataset.")
        st.write("`island` -> Biscoe = 0, Dream = 1, Torgersen = 2")
        st.write("`sex` -> FEMALE = 0, MALE = 1")
        st.write("`species` -> Adeline = 0, Chinstrap = 1, Gentoo = 2")
        st.info(
        )
        st.markdown("</div>", unsafe_allow_html=True)
