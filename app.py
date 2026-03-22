# app.py
# -*- coding: utf-8 -*-
"""
Passos Mágicos • Risco de Defasagem
- Página 1: Informação do Aluno (consulta por RA)
- Página 2: Relatórios Gerenciais

Requisitos de caminhos:
- Modelo: model/modelo_risco_defasagem.pkl
- Dados:  data/raw/PEDE_consolidado_2022_2024.csv
"""

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from typing import Optional

# ===============================
# CONFIGURAÇÃO INICIAL
# ===============================
st.set_page_config(
    page_title="Passos Mágicos • Risco de Defasagem",
    layout="wide"
)

# ===============================
# PARÂMETROS AJUSTÁVEIS
# ===============================
FEATURES = ["IAA", "IEG", "IPS", "IDA", "IPP", "IPV", "IAN"]
# Limiar binário para "Tem risco?"
RISK_THRESHOLD = 0.50  # 50%

# ===============================
# FUNÇÕES DE APOIO
# ===============================
@st.cache_resource
def carregar_modelo() -> Optional[object]:
    try:
        return joblib.load("model/modelo_risco_defasagem.pkl")
    except Exception as e:
        st.error(f"⚠️ Não foi possível carregar o modelo: {e}")
        return None

@st.cache_data
def carregar_dados() -> Optional[pd.DataFrame]:
    try:
        df_local = pd.read_csv("data/raw/PEDE_consolidado_2022_2024.csv")
        # Padronização de colunas
        df_local.columns = df_local.columns.str.upper()
        # Garante coluna auxiliar string do RA para matching
        if "RA" in df_local.columns:
            df_local["RA_STR"] = df_local["RA"].astype(str).str.strip()
        else:
            st.warning("⚠️ A coluna 'RA' não foi encontrada no dataset.")
        return df_local
    except Exception as e:
        st.error(f"⚠️ Não foi possível carregar os dados: {e}")
        return None

def normalizar_ra(valor: str) -> str:
    """Normaliza o RA digitado para comparação (texto, sem espaços)."""
    if valor is None:
        return ""
    return str(valor).strip()

def calcular_prob_e_nivel(modelo, row_features: pd.Series):
    """Calcula probabilidade e nível de risco (Baixo, Médio, Alto) para uma linha."""
    prob = float(modelo.predict_proba(row_features.values.reshape(1, -1))[:, 1][0])
    # Mesmo critério visual dos relatórios
    nivel = pd.cut(
        [prob],
        bins=[0, 0.3, 0.6, 1],
        labels=["Baixo", "Médio", "Alto"]
    )[0]
    return prob, str(nivel)

def desenhar_cartoes_metricas(df_pred: pd.DataFrame):
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Alunos", len(df_pred))
    col2.metric("Risco Alto", int((df_pred["NIVEL_RISCO"] == "Alto").sum()))
    perc_alto = (df_pred["NIVEL_RISCO"] == "Alto").mean() * 100 if len(df_pred) else 0.0
    col3.metric("% em Risco Alto", f"{perc_alto:.1f}%")

def desenhar_graficos(df_pred: pd.DataFrame):
    st.subheader("📊 Distribuição do Risco")
    fig = px.histogram(
        df_pred,
        x="PROB_RISCO",
        nbins=20,
        color="NIVEL_RISCO",
        color_discrete_map={"Baixo": "#2ca02c", "Médio": "#ff7f0e", "Alto": "#d62728"},
        title="Distribuição da Probabilidade de Risco"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📚 Risco Médio por Fase")
    if "FASE" in df_pred.columns:
        fase_risco = (
            df_pred
            .groupby("FASE")["PROB_RISCO"]
            .mean()
            .reset_index()
            .sort_values("PROB_RISCO", ascending=False)
        )
        fig2 = px.bar(
            fase_risco,
            x="FASE",
            y="PROB_RISCO",
            text="PROB_RISCO",
            title="Risco Médio por Fase"
        )
        fig2.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        fig2.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("ℹ️ A coluna 'FASE' não está disponível para este conjunto de dados.")

def construir_predicoes(df_filtro: pd.DataFrame, modelo) -> pd.DataFrame:
    """Retorna df com PROB_RISCO e NIVEL_RISCO (ignorando linhas sem features)."""
    X = df_filtro[FEATURES].dropna()
    df_pred = df_filtro.loc[X.index].copy()
    df_pred["PROB_RISCO"] = modelo.predict_proba(X)[:, 1]
    df_pred["NIVEL_RISCO"] = pd.cut(
        df_pred["PROB_RISCO"],
        bins=[0, 0.3, 0.6, 1],
        labels=["Baixo", "Médio", "Alto"]
    )
    return df_pred

# ===============================
# CARREGAMENTOS
# ===============================
modelo = carregar_modelo()
df = carregar_dados()

# Proteção se algo não carregou
if (modelo is None) or (df is None) or df.empty:
    st.stop()

# ===============================
# SIDEBAR (Filtros + Navegação)
# ===============================
st.sidebar.title("🔎 Filtros")
anos = sorted(df["ANO_DADOS"].unique(), reverse=True) if "ANO_DADOS" in df.columns else []
if not anos:
    st.error("⚠️ A coluna 'ANO_DADOS' não está presente ou não possui valores.")
    st.stop()

ano = st.sidebar.selectbox("Ano", anos)
pagina = st.sidebar.radio(
    "Navegação",
    options=["🧑‍🎓 Informação do Aluno", "📈 Relatórios Gerenciais"],
    index=0
)

# Filtro por ano
df_ano = df[df["ANO_DADOS"] == ano].copy()

# ===============================
# PÁGINA 1 – INFORMAÇÃO DO ALUNO
# ===============================
if pagina == "🧑‍🎓 Informação do Aluno":
    st.title("🎓 Previsão de Risco de Defasagem Escolar")
    st.markdown("""
    Informe o **RA do(a) aluno(a)** para verificar **probabilidade**, **nível de risco** e o resultado binário **“Tem risco?”**.
    """)

    with st.form(key="form_ra"):
        ra_input = st.text_input("RA do(a) aluno(a)", placeholder="Ex.: 123456")
        submitted = st.form_submit_button("Buscar aluno")

    if submitted:
        ra_proc = normalizar_ra(ra_input)
        if not ra_proc:
            st.warning("⚠️ Digite um RA válido.")
            st.stop()

        if "RA_STR" not in df_ano.columns:
            df_ano["RA_STR"] = df_ano["RA"].astype(str).str.strip()

        df_aluno = df_ano[df_ano["RA_STR"] == ra_proc]

        if df_aluno.empty:
            st.error("🙁 RA não encontrado para o ano selecionado.")
        else:
            # Considera o primeiro registro caso haja duplicidade
            aluno = df_aluno.iloc[0]

            # Cabeçalho com info do aluno (mostra apenas colunas existentes)
            cols_info = []
            for col in ["NOME", "RA", "FASE", "TURMA", "ESCOLA", "REDE", "CIDADE"]:
                if col in df_aluno.columns:
                    cols_info.append((col, aluno[col]))

            st.subheader("🧾 Informação do Aluno")
            if cols_info:
                info_cols = st.columns(min(4, len(cols_info)))
                for i, (rotulo, valor) in enumerate(cols_info):
                    info_cols[i % len(info_cols)].metric(rotulo.title(), str(valor))
            else:
                st.info("ℹ️ Não há colunas de identificação (NOME/FASE/etc.) disponíveis.")

            # Verifica se há dados suficientes para previsão
            if not all(f in df_aluno.columns for f in FEATURES):
                st.error(f"⚠️ As colunas de features necessárias não estão completas: {FEATURES}")
                st.stop()

            row_features = aluno[FEATURES]
            if row_features.isna().any():
                faltantes = list(row_features[row_features.isna()].index)
                st.error(
                    "⚠️ Não foi possível calcular a previsão porque há dados faltantes nestas features: "
                    + ", ".join(faltantes)
                )
                st.stop()

            # Cálculo da probabilidade / nível / binário
            prob, nivel = calcular_prob_e_nivel(modelo, row_features)
            tem_risco = "SIM" if prob >= RISK_THRESHOLD else "NÃO"

            st.subheader("🧪 Resultado")
            c1, c2, c3 = st.columns(3)
            c1.metric("Probabilidade de Risco", f"{prob:.1%}")
            c2.metric("Nível de Risco", nivel)
            c3.metric("Tem risco?", tem_risco)

            st.progress(min(max(prob, 0.0), 1.0))

            with st.expander("🔬 Detalhar features utilizadas", expanded=False):
                feat_df = pd.DataFrame({
                    "Feature": FEATURES,
                    "Valor": [aluno[f] for f in FEATURES]
                })
                st.dataframe(feat_df, use_container_width=True)

            st.caption(f"Critério binário: probabilidade ≥ {RISK_THRESHOLD:.0%} ⇒ **SIM**.")

# ===============================
# PÁGINA 2 – RELATÓRIOS GERENCIAIS
# ===============================
elif pagina == "📈 Relatórios Gerenciais":
    st.title("📈 Relatórios Gerenciais")
    st.markdown("""
    Esta visão apresenta indicadores consolidados e rankings para **priorização de intervenções**.
    """)

    # Constrói predições ignorando linhas sem features
    # (evita quebrar dashboards quando há NA em alguma feature)
    if not all(f in df_ano.columns for f in FEATURES):
        st.error(f"⚠️ As colunas de features necessárias não estão completas: {FEATURES}")
        st.stop()

    X = df_ano[FEATURES].dropna()
    if X.empty:
        st.warning("⚠️ Não há registros com todas as features preenchidas para este ano.")
        st.stop()

    df_pred = df_ano.loc[X.index].copy()
    df_pred["PROB_RISCO"] = modelo.predict_proba(X)[:, 1]
    df_pred["NIVEL_RISCO"] = pd.cut(
        df_pred["PROB_RISCO"],
        bins=[0, 0.3, 0.6, 1],
        labels=["Baixo", "Médio", "Alto"]
    )

    # Cartões
    desenhar_cartoes_metricas(df_pred)

    # Gráficos
    desenhar_graficos(df_pred)

    # Tabela prioritária + download
    st.subheader("🚨 Alunos Prioritários para Intervenção")
    cols_exibir = [c for c in ["RA", "NOME", "FASE", "PROB_RISCO", "NIVEL_RISCO"] if c in df_pred.columns]
    tabela_prioritaria = (
        df_pred
        .sort_values("PROB_RISCO", ascending=False)[cols_exibir]
        .head(30)
        .copy()
    )
    if "PROB_RISCO" in tabela_prioritaria.columns:
        tabela_prioritaria["PROB_RISCO"] = (tabela_prioritaria["PROB_RISCO"] * 100).round(1).astype(str) + "%"

    st.dataframe(tabela_prioritaria, use_container_width=True)

    # Download CSV completo com predições
    csv_bytes = df_pred.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Baixar predições (CSV)",
        data=csv_bytes,
        file_name=f"predicoes_risco_{ano}.csv",
        mime="text/csv"
    )
