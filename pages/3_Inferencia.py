
# ============================================================
# PRF ANALYTICS 2025
# DASHBOARD 3 - INFERENCIA ESTATISTICA E COMPARACAO
# VERSAO 1.1 - NAVEGACAO FUNCIONAL
#
# Populacao: 72.529
# Amostra: 383 (AAS sem reposicao)
# Semente original documentada: 42
# Confianca: 95% | Significancia: 5%
#
# A selecao do menu determina a secao exibida.
# Nenhum novo sorteio e realizado.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats


# ============================================================
# 1. CONFIGURACOES
# ============================================================

st.set_page_config(
    page_title="Inferência | PRF Analytics 2025",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

N_ESPERADO = 72529
n_ESPERADO = 383
ALFA = 0.05

AZUL = "#1677C8"
LARANJA = "#E58A2B"
VERDE = "#17846A"
VERMELHO = "#D44B48"
CINZA = "#64748B"

RAIZ = Path(__file__).resolve().parents[1]
CSV_POPULACAO = RAIZ / "datatran2025.csv"
CSV_AMOSTRA = RAIZ / "amostra_PRF_2025.csv"

CODIGOS = ["P1", "P2", "M1", "M2"]

HIPOTESES = {
    "P1": {
        "nome": "Acidentes com pelo menos uma morte",
        "tipo": "Proporção",
        "referencia": 0.10,
        "evento": "mortos > 0",
        "pergunta": (
            "A proporção de acidentes com pelo menos "
            "uma morte difere de 10%?"
        )
    },
    "P2": {
        "nome": "Acidentes em plena noite",
        "tipo": "Proporção",
        "referencia": 0.30,
        "evento": 'fase_dia == "Plena Noite"',
        "pergunta": (
            "A proporção de acidentes registrados "
            "em plena noite difere de 30%?"
        )
    },
    "M1": {
        "nome": "Pessoas envolvidas",
        "tipo": "Média",
        "referencia": 3.0,
        "evento": "pessoas",
        "pergunta": (
            "O número médio de pessoas envolvidas "
            "por acidente difere de três?"
        )
    },
    "M2": {
        "nome": "Pessoas feridas",
        "tipo": "Média",
        "referencia": 1.0,
        "evento": "feridos",
        "pergunta": (
            "O número médio de pessoas feridas "
            "por acidente difere de uma?"
        )
    }
}


# ============================================================
# 2. FORMATACAO
# ============================================================

def br(valor, casas=2):
    if not np.isfinite(float(valor)):
        return "N/D"

    texto = f"{valor:,.{casas}f}"
    return (
        texto.replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


def pct(valor, casas=2):
    return br(valor * 100, casas) + "%"


def formato_p(valor):
    if valor < 0.000001:
        return "< 0,000001"
    return br(valor, 6)


def decisao(valor):
    return (
        "Rejeitar H0"
        if valor < ALFA
        else "Não rejeitar H0"
    )


def formatar_valor(valor, tipo, casas=3):
    if tipo == "Proporção":
        return pct(valor, 2)
    return br(valor, casas)


# ============================================================
# 3. LEITURA DOS ARQUIVOS
# ============================================================

@st.cache_data(show_spinner="Carregando população...")
def carregar_populacao():
    dados = pd.read_csv(
        CSV_POPULACAO,
        sep=";",
        encoding="latin1",
        low_memory=False
    )
    dados.columns = dados.columns.str.strip()
    return dados


@st.cache_data(show_spinner="Carregando amostra...")
def carregar_amostra():
    for encoding in ["utf-8-sig", "latin1"]:
        for separador in [",", ";", "\t"]:
            try:
                dados = pd.read_csv(
                    CSV_AMOSTRA,
                    sep=separador,
                    encoding=encoding,
                    low_memory=False
                )
                dados.columns = dados.columns.str.strip()

                if (
                    len(dados) == n_ESPERADO
                    and len(dados.columns) == 30
                    and "id" in dados.columns
                ):
                    return dados

            except (UnicodeError, pd.errors.ParserError):
                continue

    raise ValueError(
        "Não foi possível reconhecer o formato "
        "do arquivo amostra_PRF_2025.csv."
    )


try:
    populacao = carregar_populacao()
    amostra = carregar_amostra()
except Exception as erro:
    st.error("Falha no carregamento dos dados.")
    st.exception(erro)
    st.stop()

N = len(populacao)
n = len(amostra)

if N != N_ESPERADO or n != n_ESPERADO:
    st.error(
        "As quantidades de registros não correspondem "
        "às bases originais do estudo."
    )
    st.stop()

obrigatorias = [
    "id", "mortos", "fase_dia", "pessoas", "feridos"
]

for coluna in obrigatorias:
    if (
        coluna not in populacao.columns
        or coluna not in amostra.columns
    ):
        st.error(f"Coluna ausente: {coluna}")
        st.stop()


# ============================================================
# 4. VALIDACAO DA AMOSTRA
# ============================================================

for dados in [populacao, amostra]:
    dados["id"] = pd.to_numeric(
        dados["id"], errors="coerce"
    )

    for coluna in ["mortos", "pessoas", "feridos"]:
        dados[coluna] = pd.to_numeric(
            dados[coluna], errors="coerce"
        )

    dados["fase_dia"] = (
        dados["fase_dia"]
        .astype("string")
        .str.strip()
    )

if (
    amostra["id"].isna().any()
    or amostra["id"].duplicated().any()
    or populacao["id"].isna().any()
    or populacao["id"].duplicated().any()
    or not amostra["id"].isin(populacao["id"]).all()
):
    st.error("Falha na validação dos IDs.")
    st.stop()

variaveis_conferencia = [
    "mortos", "fase_dia", "pessoas", "feridos"
]

if (
    amostra[["mortos", "pessoas", "feridos"]]
    .isna().any().any()
):
    st.error(
        "Existem valores ausentes nas variáveis "
        "quantitativas da amostra."
    )
    st.stop()

conferencia = amostra[
    ["id"] + variaveis_conferencia
].merge(
    populacao[["id"] + variaveis_conferencia],
    on="id",
    how="left",
    suffixes=("_a", "_p"),
    validate="one_to_one"
)

for coluna in variaveis_conferencia:
    iguais = (
        conferencia[f"{coluna}_a"]
        .eq(conferencia[f"{coluna}_p"])
        .fillna(False)
        .all()
    )

    if not iguais:
        st.error(
            f"Divergência entre população e amostra: {coluna}"
        )
        st.stop()


# ============================================================
# 5. CONSTANTES ESTATISTICAS
# ============================================================

FPC = np.sqrt((N - n) / (N - 1))
Z_CRITICO = stats.norm.ppf(1 - ALFA / 2)
T_CRITICO = stats.t.ppf(1 - ALFA / 2, df=n - 1)


def valores_variavel(dados, codigo):
    if codigo == "P1":
        return (
            dados["mortos"] > 0
        ).astype(float).to_numpy()

    if codigo == "P2":
        return (
            dados["fase_dia"]
            .eq("Plena Noite")
            .fillna(False)
        ).astype(float).to_numpy()

    if codigo == "M1":
        return dados["pessoas"].to_numpy(dtype=float)

    return dados["feridos"].to_numpy(dtype=float)


# ============================================================
# 6. ESTIMACAO E TESTES
# ============================================================

def inferencia(codigo):
    h = HIPOTESES[codigo]
    valores = valores_variavel(amostra, codigo)
    valores_pop = valores_variavel(populacao, codigo)

    estimativa = float(np.mean(valores))
    parametro = float(np.mean(valores_pop))
    h0 = h["referencia"]

    if h["tipo"] == "Proporção":
        erro_ic = (
            np.sqrt(estimativa * (1 - estimativa) / n)
            * FPC
        )
        margem = Z_CRITICO * erro_ic

        inferior = max(0.0, estimativa - margem)
        superior = min(1.0, estimativa + margem)

        erro_h0 = (
            np.sqrt(h0 * (1 - h0) / n)
            * FPC
        )
        estatistica = (estimativa - h0) / erro_h0
        valor_p = float(
            2 * stats.norm.sf(abs(estatistica))
        )

    else:
        desvio = float(np.std(valores, ddof=1))
        erro = desvio / np.sqrt(n) * FPC

        margem = T_CRITICO * erro
        inferior = estimativa - margem
        superior = estimativa + margem

        estatistica = (estimativa - h0) / erro
        valor_p = float(
            2 * stats.t.sf(
                abs(estatistica),
                df=n - 1
            )
        )

    return {
        "codigo": codigo,
        "nome": h["nome"],
        "tipo": h["tipo"],
        "estimativa": estimativa,
        "parametro": parametro,
        "h0": h0,
        "inferior": inferior,
        "superior": superior,
        "estatistica": float(estatistica),
        "p": valor_p,
        "decisao": decisao(valor_p),
        "contem": inferior <= parametro <= superior
    }


resultados = {
    c: inferencia(c) for c in CODIGOS
}


# ============================================================
# 7. AJUSTE DE HOLM
# ============================================================

def ajuste_holm(valores):
    p = np.asarray(valores, dtype=float)
    ordem = np.argsort(p)
    ajustados = np.empty(len(p))
    acumulado = 0.0

    for posicao, indice in enumerate(ordem):
        novo = min(
            1.0,
            (len(p) - posicao) * p[indice]
        )
        acumulado = max(acumulado, novo)
        ajustados[indice] = acumulado

    return ajustados


p_holm = ajuste_holm([
    resultados[c]["p"] for c in CODIGOS
])

for i, codigo in enumerate(CODIGOS):
    resultados[codigo]["p_holm"] = float(p_holm[i])
    resultados[codigo]["decisao_holm"] = decisao(
        p_holm[i]
    )


# ============================================================
# 8. ANALISE DE SENSIBILIDADE
# ============================================================

def calcular_sensibilidade(codigo):
    coluna = "pessoas" if codigo == "M1" else "feridos"

    valores = amostra[coluna].to_numpy(dtype=float)
    posicao_max = int(np.argmax(valores))
    valores_reduzidos = np.delete(valores, posicao_max)

    novo_n = len(valores_reduzidos)
    novo_fpc = np.sqrt((N - novo_n) / (N - 1))

    media = float(np.mean(valores_reduzidos))
    desvio = float(np.std(valores_reduzidos, ddof=1))

    t = (
        media - HIPOTESES[codigo]["referencia"]
    ) / (desvio / np.sqrt(novo_n) * novo_fpc)

    p = float(
        2 * stats.t.sf(abs(t), df=novo_n - 1)
    )

    return {
        "codigo": codigo,
        "id_extremo": int(amostra.iloc[posicao_max]["id"]),
        "maximo": float(valores[posicao_max]),
        "media_original": float(np.mean(valores)),
        "media_sem_maximo": media,
        "media_aparada": float(
            stats.trim_mean(valores, 0.05)
        ),
        "p_original": resultados[codigo]["p"],
        "p_sem_maximo": p,
        "decisao_original": resultados[codigo]["decisao"],
        "decisao_sem_maximo": decisao(p)
    }


sensibilidade = {
    c: calcular_sensibilidade(c)
    for c in ["M1", "M2"]
}


# ============================================================
# 9. FUNCOES DE VISUALIZACAO
# ============================================================

def grafico_intervalos(codigos, titulo):
    fig = go.Figure()

    for indice, codigo in enumerate(codigos):
        r = resultados[codigo]
        fator = 100 if r["tipo"] == "Proporção" else 1

        estimativa = r["estimativa"] * fator
        parametro = r["parametro"] * fator
        inferior = r["inferior"] * fator
        superior = r["superior"] * fator

        rotulo = f"{codigo} — {r['nome']}"

        fig.add_trace(go.Scatter(
            x=[estimativa],
            y=[rotulo],
            mode="markers",
            name="Estimativa amostral",
            marker=dict(size=13, color=AZUL),
            error_x=dict(
                type="data",
                symmetric=False,
                array=[superior - estimativa],
                arrayminus=[estimativa - inferior],
                thickness=3,
                width=7,
                color=AZUL
            ),
            hovertemplate=(
                "<b>Estimativa amostral</b>"
                "<br>Valor: %{x:.3f}"
                "<extra></extra>"
            ),
            showlegend=(indice == 0)
        ))

        fig.add_trace(go.Scatter(
            x=[parametro],
            y=[rotulo],
            mode="markers",
            name="Parâmetro populacional",
            marker=dict(
                size=14,
                symbol="diamond",
                color=LARANJA
            ),
            hovertemplate=(
                "<b>Parâmetro populacional</b>"
                "<br>Valor: %{x:.3f}"
                "<extra></extra>"
            ),
            showlegend=(indice == 0)
        ))

    fig.update_layout(
        title=titulo,
        template="plotly_white",
        height=320,
        font=dict(size=13),
        xaxis_title=(
            "Proporção (%)"
            if codigos[0].startswith("P")
            else "Pessoas por acidente"
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=[
                f"{c} — {resultados[c]['nome']}"
                for c in reversed(codigos)
            ]
        ),
        legend=dict(
            orientation="h",
            y=-0.35
        ),
        margin=dict(l=20, r=30, t=55, b=75)
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False}
    )


def tabela_comparacao():
    linhas = []

    for codigo in CODIGOS:
        r = resultados[codigo]

        diferenca = r["estimativa"] - r["parametro"]

        if r["tipo"] == "Proporção":
            diferenca_texto = (
                f"{br(diferenca * 100, 2)} p.p."
            )
        else:
            diferenca_texto = br(diferenca, 3)

        linhas.append({
            "Teste": codigo,
            "Indicador": r["nome"],
            "População": formatar_valor(
                r["parametro"], r["tipo"]
            ),
            "Amostra": formatar_valor(
                r["estimativa"], r["tipo"]
            ),
            "Diferença": diferenca_texto,
            "IC de 95%": (
                formatar_valor(r["inferior"], r["tipo"])
                + " a "
                + formatar_valor(r["superior"], r["tipo"])
            ),
            "Contém parâmetro": (
                "Sim" if r["contem"] else "Não"
            )
        })

    return pd.DataFrame(linhas)


def tabela_testes():
    linhas = []

    for codigo in CODIGOS:
        r = resultados[codigo]
        linhas.append({
            "Teste": codigo,
            "Indicador": r["nome"],
            "Estimativa": formatar_valor(
                r["estimativa"], r["tipo"]
            ),
            "H0": formatar_valor(
                r["h0"], r["tipo"], 2
            ),
            "Estatística": br(r["estatistica"], 4),
            "Valor-p": formato_p(r["p"]),
            "Decisão": r["decisao"]
        })

    return pd.DataFrame(linhas)


def tabela_holm():
    linhas = []

    for codigo in CODIGOS:
        r = resultados[codigo]
        linhas.append({
            "Teste": codigo,
            "Indicador": r["nome"],
            "Valor-p original": formato_p(r["p"]),
            "Decisão original": r["decisao"],
            "Valor-p ajustado Holm": formato_p(
                r["p_holm"]
            ),
            "Decisão Holm": r["decisao_holm"]
        })

    return pd.DataFrame(linhas)


# ============================================================
# 10. MENU LATERAL FUNCIONAL
# ============================================================

MENU = [
    "Visão geral",
    "Intervalos de confiança",
    "Testes de hipóteses",
    "Sensibilidade",
    "Ajuste de Holm",
    "Metodologia"
]

st.sidebar.title("PRF Analytics 2025")
st.sidebar.markdown("### Dashboard 3 — Inferência")

st.sidebar.info(
    "Os procedimentos inferenciais utilizam "
    "sempre os 383 registros originais, "
    "sem filtros ou novos sorteios."
)

secao = st.sidebar.radio(
    "Selecione a seção",
    MENU,
    key="menu_inferencia_v11"
)

st.sidebar.divider()
st.sidebar.caption(
    "AAS sem reposição | Semente original: 42"
)

# Diferentemente da versao anterior,
# a variavel 'secao' controla os blocos abaixo.


# ============================================================
# 11. CABECALHO COMUM
# ============================================================

st.title("📈 Inferência estatística e comparação")
st.caption("PRF Analytics 2025 — Dashboard 3")

st.markdown(
    """
    Estudo de **72.529 acidentes da PRF em 2025**,
    com amostra aleatória simples de **383 ocorrências**.

    Confiança: **95%** | Significância: **5%** |
    Testes: **bilaterais** | Amostra: **preservada**
    """
)

st.divider()

total_cobertura = sum(
    int(resultados[c]["contem"]) for c in CODIGOS
)
rejeicoes = sum(
    int(resultados[c]["p"] < ALFA)
    for c in CODIGOS
)
rejeicoes_holm = sum(
    int(resultados[c]["p_holm"] < ALFA)
    for c in CODIGOS
)


# ============================================================
# 12. SECAO: VISAO GERAL
# ============================================================

if secao == "Visão geral":

    st.header("Visão geral dos resultados")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Parâmetros analisados", "4")
    c2.metric(
        "ICs com cobertura",
        f"{total_cobertura} de 4"
    )
    c3.metric("Rejeições individuais", rejeicoes)
    c4.metric("Rejeições após Holm", rejeicoes_holm)

    st.caption(
        f"População: {br(N, 0)} | "
        f"Amostra: {br(n, 0)} | "
        f"Correção para população finita: {br(FPC, 6)}"
    )

    st.divider()

    st.subheader("Comparação entre população e amostra")

    st.dataframe(
        tabela_comparacao(),
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "Diferença = estimativa amostral − "
        "parâmetro populacional. "
        "Proporções em pontos percentuais; "
        "médias em pessoas por acidente."
    )

    st.subheader("Resumo dos testes")

    st.dataframe(
        tabela_testes(),
        hide_index=True,
        use_container_width=True
    )

    st.subheader("Principais conclusões")

    st.markdown(
        """
        **Intervalos de confiança:** os quatro
        intervalos de 95% incluíram os respectivos
        parâmetros populacionais conhecidos.

        **Testes individuais:** P1 e M2 rejeitaram
        suas hipóteses nulas. P2 e M1 não apresentaram
        evidências suficientes para rejeição.

        **Ajuste de Holm:** nenhuma hipótese nula
        permaneceu rejeitada após o controle
        do erro familiar de 5%.

        **Valores extremos:** a análise de
        sensibilidade identificou influência
        relevante da ocorrência extrema sobre
        os resultados das médias.

        **Interpretação:** significância estatística
        não implica necessariamente relevância
        prática ou relação causal.
        """
    )

    st.info(
        "Selecione uma seção no menu lateral "
        "para visualizar os gráficos e as análises "
        "detalhadas."
    )


# ============================================================
# 13. SECAO: INTERVALOS DE CONFIANCA
# ============================================================

elif secao == "Intervalos de confiança":

    st.header("Intervalos de confiança de 95%")

    st.markdown(
        """
        Os pontos azuis representam estimativas
        amostrais e as barras seus intervalos
        de confiança.

        Os losangos laranja representam os
        parâmetros calculados na população.
        """
    )

    grafico_intervalos(
        ["P1", "P2"],
        "Intervalos de confiança — Proporções"
    )

    grafico_intervalos(
        ["M1", "M2"],
        "Intervalos de confiança — Médias"
    )

    st.subheader("Resultados numéricos")

    st.dataframe(
        tabela_comparacao(),
        use_container_width=True,
        hide_index=True
    )

    st.success(
        "Os quatro intervalos de confiança "
        "abrangeram seus respectivos parâmetros "
        "populacionais nesta amostra."
    )

    st.markdown(
        """
        **Interpretação estatística**

        As estimativas amostrais não coincidiram
        exatamente com os parâmetros conhecidos,
        mas os intervalos apresentaram cobertura
        nesta realização específica.

        O nível de confiança de 95% refere-se
        à frequência esperada de cobertura do
        procedimento em repetidas amostragens,
        sob as condições do método.
        """
    )

    with st.expander("Fórmulas dos intervalos"):

        st.markdown("**Proporção — Wald**")

        st.latex(
            r"IC_{95\%}(\hat p)=\hat p\pm"
            r"z_{0,975}\sqrt{\frac{\hat p(1-\hat p)}{n}}"
            r"\sqrt{\frac{N-n}{N-1}}"
        )

        st.markdown("**Média — t de Student**")

        st.latex(
            r"IC_{95\%}(\bar x)=\bar x\pm"
            r"t_{0,975;n-1}\frac{s}{\sqrt n}"
            r"\sqrt{\frac{N-n}{N-1}}"
        )

        st.caption(
            "Proporções: intervalo de Wald. "
            "Médias: t de Student com 382 graus "
            "de liberdade. Aplicada a correção "
            "para população finita."
        )


# ============================================================
# 14. SECAO: TESTES DE HIPOTESES
# ============================================================

elif secao == "Testes de hipóteses":

    st.header("Testes de hipóteses estatísticas")

    st.markdown(
        """
        Foram realizados dois testes Z para
        proporções e dois testes t para médias,
        todos bilaterais, com α = 0,05.
        """
    )

    st.dataframe(
        tabela_testes(),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Hipóteses e interpretações")

    for codigo in CODIGOS:
        r = resultados[codigo]
        h = HIPOTESES[codigo]

        with st.expander(
            f"{codigo} — {r['nome']}",
            expanded=(codigo == "P1")
        ):

            st.write("**Questão:**", h["pergunta"])
            st.write("**Variável/evento:**", h["evento"])

            if r["tipo"] == "Proporção":
                st.latex(
                    rf"H_0:p={h['referencia']:.2f}"
                )
                st.latex(
                    rf"H_1:p\neq{h['referencia']:.2f}"
                )
            else:
                st.latex(
                    rf"H_0:\mu={h['referencia']:.0f}"
                )
                st.latex(
                    rf"H_1:\mu\neq{h['referencia']:.0f}"
                )

            x1, x2, x3 = st.columns(3)

            x1.metric(
                "Estimativa",
                formatar_valor(
                    r["estimativa"], r["tipo"]
                )
            )
            x2.metric(
                "Estatística",
                br(r["estatistica"], 4)
            )
            x3.metric(
                "Valor-p",
                formato_p(r["p"])
            )

            if r["p"] < ALFA:
                st.success(
                    "Rejeitar H0: evidência estatística "
                    "de diferença em relação à referência."
                )
            else:
                st.info(
                    "Não rejeitar H0: evidência insuficiente "
                    "para afirmar diferença da referência."
                )

            st.caption(
                "A não rejeição não comprova "
                "a igualdade proposta em H0."
            )

    st.subheader("Condições dos testes de proporção")

    linhas = []

    for codigo in ["P1", "P2"]:
        p0 = HIPOTESES[codigo]["referencia"]
        s = n * p0
        f = n * (1 - p0)

        linhas.append({
            "Teste": codigo,
            "Sucessos esperados sob H0": br(s, 1),
            "Insucessos esperados sob H0": br(f, 1),
            "Critério ≥ 10": (
                "Atendido"
                if min(s, f) >= 10
                else "Não atendido"
            )
        })

    st.dataframe(
        pd.DataFrame(linhas),
        hide_index=True,
        use_container_width=True
    )

    st.warning(
        "As distribuições de pessoas e feridos "
        "apresentaram assimetria e valores extremos. "
        "Os resultados dos testes t devem ser "
        "interpretados considerando essas limitações."
    )


# ============================================================
# 15. SECAO: SENSIBILIDADE
# ============================================================

elif secao == "Sensibilidade":

    st.header("Análise de sensibilidade")

    st.markdown(
        """
        Investigação complementar da influência
        da ocorrência extrema nas médias M1 e M2.

        A exclusão foi **hipotética**, feita
        apenas para diagnóstico.

        A amostra original permaneceu preservada.
        """
    )

    linhas = []

    for codigo in ["M1", "M2"]:
        s = sensibilidade[codigo]

        linhas.append({
            "Teste": codigo,
            "ID extremo": s["id_extremo"],
            "Máximo": br(s["maximo"], 0),
            "Média original": br(s["media_original"], 6),
            "Média sem máximo": br(
                s["media_sem_maximo"], 6
            ),
            "Média aparada 5%": br(
                s["media_aparada"], 6
            ),
            "p original": formato_p(s["p_original"]),
            "p sem máximo": formato_p(
                s["p_sem_maximo"]
            ),
            "Decisão original": s["decisao_original"],
            "Decisão sem máximo": s["decisao_sem_maximo"]
        })

    st.dataframe(
        pd.DataFrame(linhas),
        hide_index=True,
        use_container_width=True
    )

    fig = go.Figure()

    for codigo, cor in [
        ("M1", AZUL),
        ("M2", LARANJA)
    ]:
        s = sensibilidade[codigo]

        valores = [
            s["media_original"],
            s["media_sem_maximo"],
            s["media_aparada"]
        ]

        fig.add_trace(go.Bar(
            name=codigo,
            x=[
                "Média original",
                "Média sem máximo",
                "Média aparada 5%"
            ],
            y=valores,
            marker_color=cor,
            text=[br(v, 3) for v in valores],
            textposition="outside",
            hovertemplate=(
                "%{x}<br>Média: %{y:.4f}"
                "<extra></extra>"
            )
        ))

    fig.update_layout(
        title="Sensibilidade das médias",
        template="plotly_white",
        barmode="group",
        height=440,
        yaxis_title="Pessoas por acidente",
        font=dict(size=13),
        margin=dict(l=30, r=30, t=65, b=55)
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False}
    )

    st.subheader("Interpretação")

    st.markdown(
        """
        **M1 — Pessoas envolvidas**

        A média diminuiu de aproximadamente
        2,744 para 2,552 após a exclusão
        hipotética da ocorrência máxima.

        A decisão do teste mudou de não rejeitar
        para rejeitar H0.

        **M2 — Pessoas feridas**

        A média diminuiu de aproximadamente
        1,269 para 1,173 após a mesma exclusão.

        O teste continuou rejeitando H0,
        mas apresentou alteração no valor-p.
        """
    )

    st.warning(
        "Os valores-p após a exclusão são apenas "
        "diagnósticos de sensibilidade. "
        "Não constituem testes confirmatórios "
        "com o mesmo desenho amostral original."
    )


# ============================================================
# 16. SECAO: AJUSTE DE HOLM
# ============================================================

elif secao == "Ajuste de Holm":

    st.header("Comparações múltiplas — Holm")

    st.markdown(
        """
        Os quatro valores-p originais foram
        submetidos ao procedimento sequencial
        de Holm, adotando significância familiar
        de 5%.
        """
    )

    st.dataframe(
        tabela_holm(),
        hide_index=True,
        use_container_width=True
    )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Valor-p original",
        x=CODIGOS,
        y=[resultados[c]["p"] for c in CODIGOS],
        marker_color=AZUL
    ))

    fig.add_trace(go.Bar(
        name="Valor-p ajustado Holm",
        x=CODIGOS,
        y=[resultados[c]["p_holm"] for c in CODIGOS],
        marker_color=LARANJA
    ))

    fig.add_hline(
        y=ALFA,
        line_dash="dash",
        line_color=VERMELHO,
        annotation_text="α = 0,05"
    )

    fig.update_layout(
        title="Valores-p antes e depois do ajuste",
        template="plotly_white",
        barmode="group",
        height=440,
        xaxis_title="Teste",
        yaxis_title="Valor-p",
        font=dict(size=13),
        margin=dict(l=25, r=25, t=65, b=45)
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False}
    )

    st.info(
        "P1 e M2 rejeitaram H0 nos testes individuais. "
        "Após o ajuste de Holm, nenhuma hipótese "
        "permaneceu rejeitada ao nível familiar de 5%."
    )

    st.markdown(
        """
        **Interpretação**

        O ajuste de Holm não invalida os testes
        originais. Ele fornece uma avaliação
        complementar das evidências quando
        são consideradas conjuntamente
        as quatro comparações realizadas.
        """
    )


# ============================================================
# 17. SECAO: METODOLOGIA E AUDITORIA
# ============================================================

elif secao == "Metodologia":

    st.header("Metodologia e auditoria")

    st.subheader("Desenho amostral")

    st.markdown(
        """
        - **População:** 72.529 registros da PRF 2025.
        - **Amostra:** 383 ocorrências.
        - **Método:** AAS sem reposição.
        - **Semente original:** `random_state=42`.
        - **Nível de confiança:** 95%.
        - **Significância:** 5%.
        - **Testes:** bilaterais.
        """
    )

    st.subheader("Correção para população finita")

    st.latex(
        r"FPC=\sqrt{\frac{N-n}{N-1}}"
    )

    st.metric(
        "Fator utilizado",
        br(FPC, 6)
    )

    st.subheader("Estatísticas dos testes")

    st.markdown("**Teste Z para proporções**")

    st.latex(
        r"Z=\frac{\hat p-p_0}"
        r"{\sqrt{\frac{p_0(1-p_0)}{n}}\cdot FPC}"
    )

    st.markdown("**Teste t para médias**")

    st.latex(
        r"t=\frac{\bar x-\mu_0}"
        r"{\frac{s}{\sqrt{n}}\cdot FPC}"
    )

    st.subheader("Auditoria computacional")

    auditoria = pd.DataFrame([
        {
            "Verificação": "Registros populacionais",
            "Resultado": N,
            "Situação": "Aprovado"
        },
        {
            "Verificação": "Registros amostrais",
            "Resultado": n,
            "Situação": "Aprovado"
        },
        {
            "Verificação": "IDs únicos e pertencentes à população",
            "Resultado": "Sim",
            "Situação": "Aprovado"
        },
        {
            "Verificação": "Valores amostrais conferidos",
            "Resultado": "Sim",
            "Situação": "Aprovado"
        },
        {
            "Verificação": "Parâmetros analisados",
            "Resultado": len(CODIGOS),
            "Situação": "Aprovado"
        },
        {
            "Verificação": "ICs contendo parâmetro",
            "Resultado": total_cobertura,
            "Situação": (
                "Aprovado" if total_cobertura == 4
                else "Revisar"
            )
        },
        {
            "Verificação": "Amostra original",
            "Resultado": "Preservada",
            "Situação": "Aprovado"
        }
    ])

    st.dataframe(
        auditoria,
        hide_index=True,
        use_container_width=True
    )

    st.subheader("Limitações metodológicas")

    st.markdown(
        """
        A população operacional corresponde aos
        registros disponíveis na base utilizada,
        não sendo automaticamente generalizável
        a acidentes não registrados.

        Os intervalos para proporções utilizaram
        a aproximação normal de Wald.

        Os intervalos e testes para médias
        utilizaram a distribuição t de Student.
        A assimetria e os valores extremos podem
        afetar a adequação dessas aproximações.

        O diagnóstico de exclusão hipotética
        não representa um novo sorteio aleatório.

        As quatro hipóteses foram definidas
        para fins acadêmicos exploratórios,
        sem representar metas oficiais da PRF.

        Os testes individuais e o ajuste de Holm
        devem ser interpretados separadamente.
        """
    )


# ============================================================
# 18. RODAPE COMUM
# ============================================================

st.divider()

st.caption(
    "PRF Analytics 2025 | Dashboard 3 — "
    "Inferência estatística e comparação dos resultados | "
    "Probabilidade e Estatística para IA"
)
