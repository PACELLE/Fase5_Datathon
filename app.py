import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ===============================
# CONFIGURAÇÃO INICIAL
# ===============================
st.set_page_config(
    page_title="Passos Mágicos | Risco de Defasagem",
    layout="wide"
)

# ===============================
# CARREGAMENTO
# ===============================
@st.cache_resource
def carregar_modelo():
    return joblib.load("model/modelo_risco_defasagem.pkl")

@st.cache_data
def carregar_dados():
    return pd.read_csv("data/raw/PEDE_consolidado_2022_2024.csv")

modelo = carregar_modelo()
df = carregar_dados()

# Padronização
df.columns = df.columns.str.upper()

features = ["IAA", "IEG", "IPS", "IDA", "IPP", "IPV", "IAN"]

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("🔎 Filtros")
ano = st.sidebar.selectbox(
    "Ano",
    sorted(df["ANO_DADOS"].unique(), reverse=True)
)

df_filtro = df[df["ANO_DADOS"] == ano]

# ===============================
# TÍTULO
# ===============================
st.title("🎓 Previsão de Risco de Defasagem Escolar")
st.markdown("""
Esta aplicação utiliza **Machine Learning** para identificar **alunos(as) com risco de entrar em defasagem** 
antes que a queda de desempenho ocorra.
""")

# ===============================
# PREVISÃO
# ===============================
X = df_filtro[features].dropna()
df_pred = df_filtro.loc[X.index].copy()

df_pred["PROB_RISCO"] = modelo.predict_proba(X)[:, 1]

# Classificação de risco
df_pred["NIVEL_RISCO"] = pd.cut(
    df_pred["PROB_RISCO"],
    bins=[0, 0.3, 0.6, 1],
    labels=["Baixo", "Médio", "Alto"]
)

# ===============================
# MÉTRICAS GERENCIAIS
# ===============================
col1, col2, col3 = st.columns(3)

col1.metric("Total de Alunos", len(df_pred))
col2.metric("Risco Alto", (df_pred["NIVEL_RISCO"] == "Alto").sum())
col3.metric(
    "% em Risco",
    f"{(df_pred['NIVEL_RISCO'] == 'Alto').mean() * 100:.1f}%"
)

# ===============================
# VISÃO GERENCIAL 1 – DISTRIBUIÇÃO DE RISCO
# ===============================
st.subheader("📊 Distribuição do Risco")

fig = px.histogram(
    df_pred,
    x="PROB_RISCO",
    nbins=20,
    color="NIVEL_RISCO",
    title="Distribuição da Probabilidade de Risco"
)
st.plotly_chart(fig, use_container_width=True)

# ===============================
# VISÃO GERENCIAL 2 – RISCO POR FASE
# ===============================
st.subheader("📚 Risco Médio por Fase")

fase_risco = (
    df_pred
    .groupby("FASE")["PROB_RISCO"]
    .mean()
    .reset_index()
)

fig2 = px.bar(
    fase_risco,
    x="FASE",
    y="PROB_RISCO",
    title="Risco Médio por Fase"
)
st.plotly_chart(fig2, use_container_width=True)

# ===============================
# VISÃO GERENCIAL 3 – LISTA PRIORITÁRIA
# ===============================
st.subheader("🚨 Alunos Prioritários para Intervenção")

st.dataframe(
    df_pred.sort_values("PROB_RISCO", ascending=False)[
        ["RA", "NOME", "FASE", "PROB_RISCO", "NIVEL_RISCO"]
    ].head(30),
    use_container_width=True
)