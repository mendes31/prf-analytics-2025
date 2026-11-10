# ============================================================
# PRF ANALYTICS 2025
# DASHBOARD 1 - ANÁLISE DESCRITIVA DA POPULAÇÃO
#
# Versão 1.2 - Boxplots e análise de valores potencialmente atípicos
#
# Ajustes:
# - Eixos quantitativos com categorias discretas
# - Fontes e percentuais mais legíveis
# - Filtros e orientações em português
# - Informações interativas aprimoradas
# - Preservação dos cálculos estatísticos
# - Boxplots interativos de pessoas e feridos (Etapa 13.1)
# - Quartis, AIQ e identificação de valores potencialmente atípicos
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from pathlib import Path
from textwrap import wrap


# ============================================================
# 1. CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="População | PRF Analytics 2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

COR_PRINCIPAL = "#1677C8"
COR_DESTAQUE = "#E58A2B"
COR_TEXTO = "#253247"
COR_GRADE = "#E9EDF2"

TOTAL_ESPERADO = 72529

ARQUIVO = (
    Path(__file__).resolve().parents[1]
    / "datatran2025.csv"
)


# ============================================================
# 2. FUNÇÕES DE FORMATAÇÃO
# ============================================================

def numero_br(valor, casas=0):
    texto = f"{valor:,.{casas}f}"
    return (
        texto.replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


def percentual(parte, total):
    if total == 0:
        return 0.0

    return parte / total * 100


def quebrar_rotulo(texto, largura=42):
    partes = wrap(
        str(texto),
        width=largura,
        break_long_words=False,
        break_on_hyphens=False
    )

    return "<br>".join(partes) if partes else str(texto)


# ============================================================
# 3. CARREGAMENTO DA POPULAÇÃO
# ============================================================

@st.cache_data(show_spinner="Carregando dados da PRF...")
def carregar_dados():

    df = pd.read_csv(
        ARQUIVO,
        sep=";",
        encoding="latin1",
        low_memory=False
    )

    df.columns = df.columns.str.strip()
