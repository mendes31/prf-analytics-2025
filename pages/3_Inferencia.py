
# ============================================================
# PRF ANALYTICS 2025
# DASHBOARD 3 — INFERENCIA ESTATISTICA
# VERSAO 1.3 — GRAFICOS EXPLICATIVOS
#
# Populacao: 72.529 acidentes
# Amostra original: 383 acidentes
# AAS sem reposicao — semente 42
#
# Preserva:
# - Visao geral
# - Intervalos de confianca
# - Testes Z e t
# - Curvas teoricas
# - Comparacao populacao x amostra
# - Sensibilidade
# - Ajuste de Holm
# - Metodologia e auditoria
#
# Nenhum novo sorteio e realizado.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from scipy import stats


# ============================================================
# 1. CONFIGURACAO
# ============================================================

st.set_page_config(
    page_title="Inferência | PRF Analytics 2025",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
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
        "pergunta": "A proporção difere de 10%?",
    },
    "P2": {
        "nome": "Acidentes em plena noite",
        "tipo": "Proporção",
        "referencia": 0.30,
        "evento": 'fase_dia == "Plena Noite"',
        "pergunta": "A proporção difere de 30%?",
    },
    "M1": {
        "nome": "Pessoas envolvidas",
        "tipo": "Média",
        "referencia": 3.0,
        "evento": "pessoas",
        "pergunta": "A média difere de 3 pessoas?",
    },
    "M2": {
        "nome": "Pessoas feridas",
        "tipo": "Média",
        "referencia": 1.0,
        "evento": "feridos",
        "pergunta": "A média difere de 1 pessoa?",
    },
}


# ============================================================
# 2. FORMATACAO
# ============================================================

def br(valor, casas=2):
    try:
        numero = float(valor)

        if not np.isfinite(numero):
            return "N/D"

        texto = f"{numero:,.{casas}f}"

        return (
            texto.replace(",", "X")
            .replace(".", ",")
            .replace("X", ".")
        )

    except (TypeError, ValueError):
        return "N/D"


def pct(valor, casas=2):
    return br(valor * 100, casas) + "%"


def formato_p(valor):
    if valor < 0.000001:
        return "< 0,000001"

    return br(valor, 6)


def decisao(p):
    if p < ALFA:
        return "Rejeitar H0"

    return "Não rejeitar H0"


def formatar_valor(valor, tipo, casas=3):
    if tipo == "Proporção":
        return pct(valor, 2)

    return br(valor, casas)


def plotar(fig):
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displaylogo": False},
    )


# ============================================================
# 3. CARREGAMENTO DAS BASES
# ============================================================

@st.cache_data(show_spinner="Carregando população...")
def carregar_populacao():
    dados = pd.read_csv(
        CSV_POPULACAO,
        sep=";",
        encoding="latin1",
        low_memory=False,
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
                    low_memory=False,
                )

                dados.columns = dados.columns.str.strip()

                if (
                    dados.shape == (383, 30)
                    and "id" in dados.columns
                ):
                    return dados

            except (
                UnicodeError,
                pd.errors.ParserError,
            ):
                continue

    raise ValueError(
        "Não foi possível reconhecer o CSV da amostra."
    )


try:
    populacao = carregar_populacao()
    amostra = carregar_amostra()

except Exception as erro:
    st.error("Falha no carregamento das bases.")
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


# ============================================================
# 4. VALIDACAO DA AMOSTRA
# ============================================================

COLUNAS = [
    "id",
    "mortos",
    "fase_dia",
    "pessoas",
    "feridos",
]

for coluna in COLUNAS:
    if (
        coluna not in populacao.columns
        or coluna not in amostra.columns
    ):
        st.error(f"Coluna ausente: {coluna}")
        st.stop()


for dados in [populacao, amostra]:
    dados["id"] = pd.to_numeric(
        dados["id"],
        errors="coerce",
    )

    for coluna in ["mortos", "pessoas", "feridos"]:
        dados[coluna] = pd.to_numeric(
            dados[coluna],
            errors="coerce",
        )

    dados["fase_dia"] = (
        dados["fase_dia"]
        .astype("string")
        .str.strip()
    )


ids_validos = bool(
    populacao["id"].notna().all()
    and amostra["id"].notna().all()
    and populacao["id"].is_unique
    and amostra["id"].is_unique
    and amostra["id"].isin(populacao["id"]).all()
)

if not ids_validos:
    st.error("Falha na validação dos IDs.")
    st.stop()


if (
    amostra[["mortos", "pessoas", "feridos"]]
    .isna()
    .any()
    .any()
):
    st.error("Existem valores quantitativos ausentes.")
    st.stop()


conferencia = amostra[COLUNAS].merge(
    populacao[COLUNAS],
    on="id",
    how="left",
    suffixes=("_a", "_p"),
    validate="one_to_one",
)

valores_conferidos = True

for coluna in [
    "mortos",
    "fase_dia",
    "pessoas",
    "feridos",
]:
    iguais = bool(
        conferencia[f"{coluna}_a"]
        .eq(conferencia[f"{coluna}_p"])
        .fillna(False)
        .all()
    )

    if not iguais:
        valores_conferidos = False
        st.error(
            "Divergência entre população e amostra: "
            + coluna
        )
        st.stop()


contagens_validas = bool(
    int(amostra["pessoas"].sum()) == 1051
    and int(amostra["feridos"].sum()) == 486
    and int((amostra["mortos"] > 0).sum()) == 24
    and int(
        amostra["fase_dia"]
        .eq("Plena Noite")
        .fillna(False)
        .sum()
    ) == 121
)

if not contagens_validas:
    st.error(
        "Os indicadores da amostra divergem "
        "do Google Colab."
    )
    st.stop()


# ============================================================
# 5. CONSTANTES ESTATISTICAS
# ============================================================

FPC = float(
    np.sqrt((N - n) / (N - 1))
)

Z_CRITICO = float(
    stats.norm.ppf(1 - ALFA / 2)
)

T_CRITICO = float(
    stats.t.ppf(1 - ALFA / 2, df=n - 1)
)

GL = n - 1


def vetor(dados, codigo):
    if codigo == "P1":
        return (
            dados["mortos"]
            .gt(0)
            .astype(float)
            .to_numpy()
        )

    if codigo == "P2":
        return (
            dados["fase_dia"]
            .eq("Plena Noite")
            .fillna(False)
            .astype(float)
            .to_numpy()
        )

    coluna = (
        "pessoas"
        if codigo == "M1"
        else "feridos"
    )

    return dados[coluna].to_numpy(dtype=float)


# ============================================================
# 6. INFERENCIA ESTATISTICA
# ============================================================

def calcular_inferencia(codigo):
    hipotese = HIPOTESES[codigo]

    dados_amostrais = vetor(amostra, codigo)
    dados_populacionais = vetor(populacao, codigo)

    estimativa = float(
        np.mean(dados_amostrais)
    )

    parametro = float(
        np.mean(dados_populacionais)
    )

    h0 = float(hipotese["referencia"])

    desvio = None

    if hipotese["tipo"] == "Proporção":
        sucessos = int(
            np.sum(dados_amostrais)
        )

        erro_ic = (
            np.sqrt(
                estimativa * (1 - estimativa) / n
            )
            * FPC
        )

        critico = Z_CRITICO

        margem = critico * erro_ic

        inferior = max(
            0.0,
            estimativa - margem,
        )

        superior = min(
            1.0,
            estimativa + margem,
        )

        erro_h0 = (
            np.sqrt(h0 * (1 - h0) / n)
            * FPC
        )

        estatistica = (
            estimativa - h0
        ) / erro_h0

        valor_p = float(
            2 * stats.norm.sf(
                abs(estatistica)
            )
        )

    else:
        sucessos = None

        desvio = float(
            np.std(
                dados_amostrais,
                ddof=1,
            )
        )

        erro_ic = (
            desvio / np.sqrt(n)
        ) * FPC

        critico = T_CRITICO

        margem = critico * erro_ic

        inferior = estimativa - margem
        superior = estimativa + margem

        estatistica = (
            estimativa - h0
        ) / erro_ic

        valor_p = float(
            2 * stats.t.sf(
                abs(estatistica),
                df=GL,
            )
        )

    return {
        "codigo": codigo,
        "nome": hipotese["nome"],
        "tipo": hipotese["tipo"],
        "estimativa": estimativa,
        "parametro": parametro,
        "h0": h0,
        "inferior": float(inferior),
        "superior": float(superior),
        "margem": float(margem),
        "erro_ic": float(erro_ic),
        "critico": float(critico),
        "desvio": desvio,
        "sucessos": sucessos,
        "estatistica": float(estatistica),
        "p": valor_p,
        "decisao": decisao(valor_p),
        "contem": bool(
            inferior <= parametro <= superior
        ),
    }


resultados = {
    codigo: calcular_inferencia(codigo)
    for codigo in CODIGOS
}


# ============================================================
# 7. AJUSTE DE HOLM
# ============================================================

def ajuste_holm(valores):
    p = np.asarray(
        valores,
        dtype=float,
    )

    ordem = np.argsort(p)

    ajustados = np.empty(len(p))

    acumulado = 0.0

    for posicao, indice in enumerate(ordem):
        novo = min(
            1.0,
            (len(p) - posicao) * p[indice],
        )

        acumulado = max(
            acumulado,
            novo,
        )

        ajustados[indice] = acumulado

    return ajustados


p_holm = ajuste_holm([
    resultados[codigo]["p"]
    for codigo in CODIGOS
])

for indice, codigo in enumerate(CODIGOS):
    resultados[codigo]["p_holm"] = float(
        p_holm[indice]
    )

    resultados[codigo]["decisao_holm"] = decisao(
        float(p_holm[indice])
    )


# ============================================================
# 8. ANALISE DE SENSIBILIDADE
# ============================================================

def calcular_sensibilidade(codigo):
    coluna = (
        "pessoas"
        if codigo == "M1"
        else "feridos"
    )

    valores = amostra[coluna].to_numpy(
        dtype=float
    )

    posicao = int(
        np.argmax(valores)
    )

    reduzidos = np.delete(
        valores,
        posicao,
    )

    novo_n = len(reduzidos)

    novo_fpc = np.sqrt(
        (N - novo_n) / (N - 1)
    )

    media = float(
        np.mean(reduzidos)
    )

    desvio = float(
        np.std(reduzidos, ddof=1)
    )

    estatistica = (
        media - HIPOTESES[codigo]["referencia"]
    ) / (
        desvio / np.sqrt(novo_n) * novo_fpc
    )

    valor_p = float(
        2 * stats.t.sf(
            abs(estatistica),
            df=novo_n - 1,
        )
    )

    return {
        "codigo": codigo,
        "id_extremo": int(
            amostra.iloc[posicao]["id"]
        ),
        "maximo": float(valores[posicao]),
        "media_original": float(
            np.mean(valores)
        ),
        "media_sem_maximo": media,
        "media_aparada": float(
            stats.trim_mean(valores, 0.05)
        ),
        "p_original": resultados[codigo]["p"],
        "p_sem_maximo": valor_p,
        "decisao_original": resultados[codigo]["decisao"],
        "decisao_sem_maximo": decisao(valor_p),
    }


sensibilidade = {
    codigo: calcular_sensibilidade(codigo)
    for codigo in ["M1", "M2"]
}


# ============================================================
# 9. VALIDACAO COM RESULTADOS DO COLAB
# ============================================================

referencias = {
    "P1": {
        "estimativa": 24 / 383,
        "p": 0.014602,
    },
    "P2": {
        "estimativa": 121 / 383,
        "p": 0.495256,
    },
    "M1": {
        "estimativa": 1051 / 383,
        "p": 0.212285,
    },
    "M2": {
        "estimativa": 486 / 383,
        "p": 0.013189,
    },
}

resultados_conferidos = True

for codigo, esperado in referencias.items():
    r = resultados[codigo]

    if not np.isclose(
        r["estimativa"],
        esperado["estimativa"],
        atol=1e-8,
    ):
        resultados_conferidos = False

    if not np.isclose(
        r["p"],
        esperado["p"],
        atol=1e-5,
    ):
        resultados_conferidos = False

if not resultados_conferidos:
    st.error(
        "Os resultados divergem dos valores "
        "validados no Google Colab."
    )
    st.stop()


# ============================================================
# 10. TABELAS
# ============================================================

def tabela_comparacao():
    linhas = []

    for codigo in CODIGOS:
        r = resultados[codigo]

        diferenca = (
            r["estimativa"] - r["parametro"]
        )

        diferenca_texto = (
            br(diferenca * 100, 2) + " p.p."
            if r["tipo"] == "Proporção"
            else br(diferenca, 3)
        )

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
            "Limite inferior": formatar_valor(
                r["inferior"], r["tipo"]
            ),
            "Limite superior": formatar_valor(
                r["superior"], r["tipo"]
            ),
            "Contém parâmetro": (
                "Sim" if r["contem"] else "Não"
            ),
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
                r["h0"], r["tipo"]
            ),
            "Estatística": br(
                r["estatistica"], 4
            ),
            "Valor-p": formato_p(r["p"]),
            "Decisão": r["decisao"],
        })

    return pd.DataFrame(linhas)


def tabela_holm():
    return pd.DataFrame([
        {
            "Teste": codigo,
            "Indicador": resultados[codigo]["nome"],
            "Valor-p original": formato_p(
                resultados[codigo]["p"]
            ),
            "Decisão original": resultados[codigo]["decisao"],
            "Valor-p ajustado Holm": formato_p(
                resultados[codigo]["p_holm"]
            ),
            "Decisão Holm": resultados[codigo]["decisao_holm"],
        }
        for codigo in CODIGOS
    ])


# ============================================================
# 11. GRAFICOS INDIVIDUAIS DOS ICS
# ============================================================

def grafico_ic_individual(codigo):
    r = resultados[codigo]

    proporcao = r["tipo"] == "Proporção"

    fator = 100 if proporcao else 1

    estimativa = r["estimativa"] * fator
    parametro = r["parametro"] * fator
    inferior = r["inferior"] * fator
    superior = r["superior"] * fator

    margem = r["margem"] * fator

    unidade = "%" if proporcao else ""

    minimo = min(
        inferior,
        parametro,
        estimativa,
    )

    maximo = max(
        superior,
        parametro,
        estimativa,
    )

    amplitude = maximo - minimo

    folga = max(
        amplitude * 0.20,
        0.5 if proporcao else 0.05,
    )

    eixo_min = max(
        0,
        minimo - folga,
    )

    eixo_max = maximo + folga

    fig = go.Figure()

    # Intervalo de confianca.
    fig.add_trace(
        go.Scatter(
            x=[inferior, superior],
            y=[0, 0],
            mode="lines+markers",
            name="IC 95%",
            line=dict(
                color=AZUL,
                width=7,
            ),
            marker=dict(
                color=AZUL,
                size=10,
                symbol="line-ns-open",
                line=dict(width=3),
            ),
            hovertemplate=(
                "Limite do intervalo: %{x:.4f}"
                + unidade
                + "<extra></extra>"
            ),
        )
    )

    # Estimativa amostral.
    fig.add_trace(
        go.Scatter(
            x=[estimativa],
            y=[0],
            mode="markers+text",
            name="Estimativa amostral",
            marker=dict(
                color=AZUL,
                size=17,
                line=dict(
                    color="white",
                    width=2,
                ),
            ),
            text=[
                f"Amostra: {br(estimativa, 3)}{unidade}"
            ],
            textposition="top center",
            textfont=dict(
                color=AZUL,
                size=13,
            ),
            hovertemplate=(
                "Estimativa amostral: %{x:.4f}"
                + unidade
                + "<extra></extra>"
            ),
        )
    )

    # Parametro populacional.
    fig.add_trace(
        go.Scatter(
            x=[parametro],
            y=[0],
            mode="markers",
            name="Parâmetro populacional",
            marker=dict(
                symbol="diamond",
                size=18,
                color=LARANJA,
                line=dict(
                    color="white",
                    width=2,
                ),
            ),
            hovertemplate=(
                "Parâmetro populacional: %{x:.4f}"
                + unidade
                + "<extra></extra>"
            ),
        )
    )

    # Indicacao dos limites.
    for valor, rotulo in [
        (inferior, "Limite inferior"),
        (superior, "Limite superior"),
    ]:
        fig.add_annotation(
            x=valor,
            y=0,
            text=(
                rotulo
                + "<br><b>"
                + br(valor, 3)
                + unidade
                + "</b>"
            ),
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=60,
            font=dict(
                size=11,
                color=AZUL,
            ),
            bgcolor="rgba(255,255,255,0.92)",
        )

    # Identificacao do valor populacional.
    fig.add_annotation(
        x=parametro,
        y=0,
        text=(
            "População<br><b>"
            + br(parametro, 3)
            + unidade
            + "</b>"
        ),
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-85,
        font=dict(
            size=11,
            color=LARANJA,
        ),
        bgcolor="rgba(255,255,255,0.92)",
    )

    fig.update_layout(
        title=dict(
            text=(
                f"{codigo} — {r['nome']}"
                f"<br><sup>IC 95%: "
                f"{br(inferior, 3)}{unidade} a "
                f"{br(superior, 3)}{unidade}"
                f" | Margem: ±{br(margem, 4)}"
                f"{' p.p.' if proporcao else ' pessoa(s)'}</sup>"
            )
        ),
        template="plotly_white",
        height=370,
        xaxis=dict(
            title=(
                "Proporção de acidentes (%)"
                if proporcao
                else "Média de pessoas por acidente"
            ),
            range=[eixo_min, eixo_max],
            gridcolor="#E9EDF2",
            zeroline=False,
        ),
        yaxis=dict(
            visible=False,
            range=[-0.45, 0.45],
        ),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.32,
        ),
        margin=dict(
            l=40,
            r=40,
            t=85,
            b=80,
        ),
    )

    plotar(fig)


# ============================================================
# 12. MEMORIA DE CALCULO DOS ICS
# ============================================================

def memoria_calculo_ic(codigo):
    r = resultados[codigo]

    with st.expander(
        f"📐 Como foi calculado o IC de {codigo}?",
        expanded=False,
    ):
        st.write(
            "**Método:**",
            (
                "Intervalo de Wald para proporção"
                if r["tipo"] == "Proporção"
                else "Intervalo t de Student para média"
            ),
        )

        st.write(
            "**População:**", br(N, 0),
            "| **Amostra:**", br(n, 0),
            "| **FPC:**", br(FPC, 6),
        )

        if r["tipo"] == "Proporção":
            st.latex(
                r"\hat p=\frac{x}{n}"
            )

            st.latex(
                r"IC_{95\%}=\hat p\pm"
                r"z_{0,975}"
                r"\sqrt{\frac{\hat p(1-\hat p)}{n}}"
                r"\times FPC"
            )

            st.markdown(
                f"**Ocorrências com o evento:** "
                f"{r['sucessos']} de {n}"
            )

            st.markdown(
                f"**Estimativa:** "
                f"{r['sucessos']}/{n} = "
                f"{pct(r['estimativa'], 4)}"
            )

            st.markdown(
                f"**Valor crítico Z:** "
                f"{br(r['critico'], 6)}"
            )

            st.markdown(
                f"**Erro-padrão corrigido:** "
                f"{pct(r['erro_ic'], 6)}"
            )

            st.markdown(
                f"**Margem de erro do IC:** "
                f"±{br(r['margem'] * 100, 4)} "
                f"pontos percentuais"
            )

            st.latex(
                r"LI=\hat p-z_{0,975}\cdot EP"
            )

            st.latex(
                r"LS=\hat p+z_{0,975}\cdot EP"
            )

            st.write(
                "**Limite inferior:**",
                pct(r["inferior"], 4),
            )

            st.write(
                "**Limite superior:**",
                pct(r["superior"], 4),
            )

        else:
            st.latex(
                r"\bar x=\frac{\sum x_i}{n}"
            )

            st.latex(
                r"IC_{95\%}=\bar x\pm"
                r"t_{0,975;n-1}"
                r"\frac{s}{\sqrt n}\times FPC"
            )

            st.write(
                "**Média amostral:**",
                br(r["estimativa"], 6),
            )

            st.write(
                "**Desvio-padrão amostral:**",
                br(r["desvio"], 6),
            )

            st.write(
                "**Graus de liberdade:**",
                GL,
            )

            st.write(
                "**Valor crítico t:**",
                br(r["critico"], 6),
            )

            st.write(
                "**Erro-padrão corrigido:**",
                br(r["erro_ic"], 6),
            )

            st.write(
                "**Margem do IC:**",
                "±" + br(r["margem"], 6),
            )

            st.write(
                "**Limite inferior:**",
                br(r["inferior"], 6),
            )

            st.write(
                "**Limite superior:**",
                br(r["superior"], 6),
            )

        st.info(
            "A hipótese nula utilizada no teste "
            "não determina os limites deste IC. "
            "O intervalo é calculado a partir "
            "da estimativa e da variabilidade amostral."
        )


# ============================================================
# 13. INTERPRETACAO INDIVIDUAL DOS ICS
# ============================================================

def interpretacao_ic(codigo):
    r = resultados[codigo]

    diferenca = (
        r["estimativa"] - r["parametro"]
    )

    if r["tipo"] == "Proporção":
        unidade = "pontos percentuais"

        diferenca_texto = (
            br(abs(diferenca) * 100, 4)
        )

    else:
        unidade = "pessoas por acidente"

        diferenca_texto = br(
            abs(diferenca),
            4,
        )

    direcao = (
        "acima"
        if diferenca > 0
        else "abaixo"
    )

    if np.isclose(diferenca, 0):
        direcao = "igual ao"

    st.markdown(
        f"**Interpretação — {codigo}:** "
        f"A estimativa amostral foi "
        f"**{formatar_valor(r['estimativa'], r['tipo'])}**, "
        f"com IC de 95% entre "
        f"**{formatar_valor(r['inferior'], r['tipo'])}** "
        f"e **{formatar_valor(r['superior'], r['tipo'])}**. "
        f"O parâmetro populacional é "
        f"**{formatar_valor(r['parametro'], r['tipo'])}** "
        f"e está "
        f"**{'dentro' if r['contem'] else 'fora'}** "
        f"do intervalo."
    )

    st.caption(
        f"Diferença amostra − população: "
        f"{diferenca_texto} {unidade} "
        f"({direcao} do valor populacional)."
    )


# ============================================================
# 14. GRAFICOS COMPARATIVOS
# ============================================================

def grafico_comparativo(codigos, titulo):
    proporcao = (
        resultados[codigos[0]]["tipo"]
        == "Proporção"
    )

    fator = 100 if proporcao else 1

    categorias = [
        f"{c} — {resultados[c]['nome']}"
        for c in codigos
    ]

    pop = [
        resultados[c]["parametro"] * fator
        for c in codigos
    ]

    amo = [
        resultados[c]["estimativa"] * fator
        for c in codigos
    ]

    limites_inf = [
        resultados[c]["inferior"] * fator
        for c in codigos
    ]

    limites_sup = [
        resultados[c]["superior"] * fator
        for c in codigos
    ]

    erros_inf = [
        valor - limite
        for valor, limite in zip(amo, limites_inf)
    ]

    erros_sup = [
        limite - valor
        for valor, limite in zip(amo, limites_sup)
    ]

    unidade = "%" if proporcao else ""

    casas = 2 if proporcao else 3

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=categorias,
            y=pop,
            name="População",
            marker_color=AZUL,
            text=[
                f"{v:.{casas}f}{unidade}"
                for v in pop
            ],
            textposition="outside",
            cliponaxis=False,
        )
    )

    fig.add_trace(
        go.Bar(
            x=categorias,
            y=amo,
            name="Amostra (IC 95%)",
            marker_color=LARANJA,
            text=[
                f"{v:.{casas}f}{unidade}"
                for v in amo
            ],
            textposition="outside",
            cliponaxis=False,
            error_y=dict(
                type="data",
                symmetric=False,
                array=erros_sup,
                arrayminus=erros_inf,
                visible=True,
                color="#7C2D12",
                width=7,
                thickness=2,
            ),
            customdata=np.column_stack([
                limites_inf,
                limites_sup,
            ]),
            hovertemplate=(
                "Estimativa: %{y:.4f}"
                + unidade
                + "<br>IC 95%: "
                + "%{customdata[0]:.4f} a "
                + "%{customdata[1]:.4f}"
                + unidade
                + "<extra></extra>"
            ),
        )
    )

    maximo = max(
        max(pop),
        max(limites_sup),
    )

    fig.update_layout(
        title=titulo,
        template="plotly_white",
        barmode="group",
        height=470,
        yaxis=dict(
            title=(
                "Proporção (%)"
                if proporcao
                else "Pessoas por acidente"
            ),
            range=[0, maximo * 1.25],
        ),
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.25,
        ),
        margin=dict(
            l=45,
            r=35,
            t=75,
            b=90,
        ),
    )

    plotar(fig)


# ============================================================
# 15. CURVAS Z E T
# ============================================================

def grafico_curva_teste(codigo):
    r = resultados[codigo]

    proporcao = (
        r["tipo"] == "Proporção"
    )

    nome_teste = (
        "Z" if proporcao else "t"
    )

    critico = (
        Z_CRITICO
        if proporcao
        else T_CRITICO
    )

    observado = r["estatistica"]

    x = np.linspace(
        -4.5,
        4.5,
        1400,
    )

    def densidade(valores):
        if proporcao:
            return stats.norm.pdf(valores)

        return stats.t.pdf(
            valores,
            df=GL,
        )

    y = densidade(x)

    fig = go.Figure()

    for inicio, fim in [
        (-4.5, -critico),
        (critico, 4.5),
    ]:
        xx = np.linspace(
            inicio,
            fim,
            300,
        )

        fig.add_trace(
            go.Scatter(
                x=xx,
                y=densidade(xx),
                mode="lines",
                line=dict(width=0),
                fill="tozeroy",
                fillcolor="rgba(220,38,38,0.20)",
                name="Região crítica",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    for inicio, fim in [
        (-4.5, -abs(observado)),
        (abs(observado), 4.5),
    ]:
        xx = np.linspace(
            inicio,
            fim,
            300,
        )

        fig.add_trace(
            go.Scatter(
                x=xx,
                y=densidade(xx),
                mode="lines",
                line=dict(width=0),
                fill="tozeroy",
                fillcolor="rgba(245,158,11,0.38)",
                name="Área do valor-p",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name="Distribuição sob H0",
            line=dict(
                color=AZUL,
                width=3,
            ),
        )
    )

    for valor in [-critico, critico]:
        fig.add_vline(
            x=float(valor),
            line_dash="dash",
            line_color=VERMELHO,
            line_width=2,
        )

    fig.add_vline(
        x=float(observado),
        line_color=VERDE,
        line_width=3,
    )

    fig.add_annotation(
        x=float(observado),
        y=float(densidade(observado)),
        text=(
            f"{nome_teste} observado = "
            f"{br(observado, 4)}"
        ),
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-55,
        font=dict(color=VERDE),
        bgcolor="white",
    )

    fig.update_layout(
        title=dict(
            text=(
                f"{codigo} — {r['nome']}"
                "<br><sup>"
                f"Valor-p = {formato_p(r['p'])} | "
                f"{r['decisao']}"
                "</sup>"
            )
        ),
        template="plotly_white",
        height=480,
        xaxis=dict(
            title=f"Estatística {nome_teste}",
            range=[-4.5, 4.5],
        ),
        yaxis=dict(
            title="Densidade",
            range=[0, 0.48],
        ),
        margin=dict(
            l=40,
            r=30,
            t=95,
            b=60,
        ),
    )

    plotar(fig)


# ============================================================
# 16. EXPLICACAO DOS TESTES
# ============================================================

def explicacao_teste(codigo):
    r = resultados[codigo]

    h = HIPOTESES[codigo]

    st.markdown(
        f"**Questão investigada:** {h['pergunta']}"
    )

    if r["tipo"] == "Proporção":
        st.latex(
            rf"H_0:p={r['h0']:.2f}"
        )

        st.latex(
            rf"H_1:p\neq{r['h0']:.2f}"
        )

        st.latex(
            r"Z=\frac{\hat p-p_0}"
            r"{\sqrt{\frac{p_0(1-p_0)}{n}}\cdot FPC}"
        )

    else:
        st.latex(
            rf"H_0:\mu={r['h0']:.0f}"
        )

        st.latex(
            rf"H_1:\mu\neq{r['h0']:.0f}"
        )

        st.latex(
            r"t=\frac{\bar x-\mu_0}"
            r"{\frac{s}{\sqrt n}\cdot FPC}"
        )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Estimativa",
        formatar_valor(
            r["estimativa"],
            r["tipo"],
        ),
    )

    c2.metric(
        "Estatística de teste",
        br(r["estatistica"], 4),
    )

    c3.metric(
        "Valor-p",
        formato_p(r["p"]),
    )

    st.markdown(
        f"**Valor de referência de H0:** "
        f"{formatar_valor(r['h0'], r['tipo'])}"
    )

    st.markdown(
        f"**Nível de significância:** "
        f"{pct(ALFA, 0)}"
    )

    if r["p"] < ALFA:
        st.success(
            "O valor-p é inferior a 0,05. "
            "Rejeitamos H0 no teste individual."
        )

    else:
        st.info(
            "O valor-p é igual ou superior a 0,05. "
            "Não rejeitamos H0 no teste individual."
        )

    st.caption(
        "O valor-p não representa a probabilidade "
        "de H0 ser verdadeira. Não rejeitar H0 "
        "também não comprova igualdade."
    )


# ============================================================
# 17. MENU LATERAL
# ============================================================

MENU = [
    "Visão geral",
    "Intervalos de confiança",
    "Testes de hipóteses",
    "Sensibilidade",
    "Ajuste de Holm",
    "Metodologia",
]

st.sidebar.title("PRF Analytics 2025")

st.sidebar.markdown(
    "### Dashboard 3 — Inferência"
)

st.sidebar.info(
    "Os procedimentos inferenciais utilizam "
    "sempre os 383 registros originais, "
    "sem filtros e sem novos sorteios."
)

secao = st.sidebar.radio(
    "Selecione a seção",
    MENU,
    key="menu_inferencia_v13",
)

st.sidebar.divider()

st.sidebar.caption(
    "AAS sem reposição | Semente original: 42"
)


# ============================================================
# 18. CABECALHO
# ============================================================

st.title(
    "📈 Inferência estatística e comparação"
)

st.caption(
    "PRF Analytics 2025 — Dashboard 3"
)

st.markdown(
    """
    Estudo de **72.529 acidentes da PRF em 2025**,
    com amostra aleatória simples de **383 ocorrências**.

    **Confiança:** 95% |
    **Significância:** 5% |
    **Testes:** bilaterais |
    **Amostra:** preservada.
    """
)

st.divider()


total_cobertura = sum(
    int(resultados[c]["contem"])
    for c in CODIGOS
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
# 19. VISAO GERAL
# ============================================================

if secao == "Visão geral":
    st.header(
        "Visão geral dos resultados"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Parâmetros analisados",
        "4",
    )

    c2.metric(
        "ICs com cobertura",
        f"{total_cobertura} de 4",
    )

    c3.metric(
        "Rejeições individuais",
        rejeicoes,
    )

    c4.metric(
        "Rejeições após Holm",
        rejeicoes_holm,
    )

    st.caption(
        f"População: {br(N, 0)} | "
        f"Amostra: {br(n, 0)} | "
        f"FPC: {br(FPC, 6)}"
    )

    st.subheader(
        "Comparação entre população e amostra"
    )

    st.dataframe(
        tabela_comparacao(),
        hide_index=True,
        width="stretch",
    )

    st.subheader(
        "Resumo dos testes"
    )

    st.dataframe(
        tabela_testes(),
        hide_index=True,
        width="stretch",
    )

    st.subheader(
        "Principais conclusões"
    )

    st.markdown(
        """
        **Intervalos de confiança:** os quatro
        intervalos de 95% incluíram os respectivos
        parâmetros populacionais.

        **Testes individuais:** P1 e M2 rejeitaram
        H0. P2 e M1 não apresentaram evidência
        suficiente para rejeição.

        **Ajuste de Holm:** nenhuma hipótese nula
        permaneceu rejeitada após a correção.

        **Sensibilidade:** valores extremos
        influenciaram principalmente as médias.

        **Limitação:** os resultados referem-se
        à população operacional estudada.
        """
    )

    st.info(
        "Utilize o menu lateral para explorar "
        "os gráficos e a memória de cálculo."
    )


# ============================================================
# 20. INTERVALOS DE CONFIANCA
# ============================================================

elif secao == "Intervalos de confiança":
    st.header(
        "Intervalos de confiança de 95%"
    )

    st.markdown(
        """
        **Como interpretar os gráficos:**

        - A linha azul representa o intervalo de confiança.
        - O ponto azul representa a estimativa amostral.
        - O losango laranja representa o parâmetro populacional.
        - Os limites e a margem são calculados a partir da amostra.

        Cada indicador possui sua própria escala para
        facilitar a visualização.
        """
    )

    st.info(
        "O intervalo de confiança fornece uma faixa "
        "de valores plausíveis para o parâmetro. "
        "O nível de 95% representa a cobertura esperada "
        "do procedimento em repetidas amostragens."
    )

    for codigo in CODIGOS:
        st.divider()

        st.subheader(
            f"{codigo} — {resultados[codigo]['nome']}"
        )

        r = resultados[codigo]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Estimativa amostral",
            formatar_valor(
                r["estimativa"],
                r["tipo"],
            ),
        )

        c2.metric(
            "Limite inferior",
            formatar_valor(
                r["inferior"],
                r["tipo"],
            ),
        )

        c3.metric(
            "Limite superior",
            formatar_valor(
                r["superior"],
                r["tipo"],
            ),
        )

        c4.metric(
            "Parâmetro populacional",
            formatar_valor(
                r["parametro"],
                r["tipo"],
            ),
        )

        grafico_ic_individual(codigo)

        interpretacao_ic(codigo)

        memoria_calculo_ic(codigo)

    st.divider()

    st.subheader(
        "Resultados numéricos consolidados"
    )

    st.dataframe(
        tabela_comparacao(),
        hide_index=True,
        width="stretch",
    )

    st.success(
        "Os quatro intervalos de 95% incluíram "
        "seus respectivos parâmetros populacionais."
    )

    st.subheader(
        "Comparação gráfica — População x Amostra"
    )

    st.caption(
        "Os gráficos abaixo oferecem uma visão conjunta. "
        "As barras de erro aparecem somente nas "
        "estimativas amostrais."
    )

    grafico_comparativo(
        ["P1", "P2"],
        "Proporções — População x Amostra (IC 95%)",
    )

    grafico_comparativo(
        ["M1", "M2"],
        "Médias — População x Amostra (IC 95%)",
    )

    with st.expander(
        "Fórmulas gerais dos intervalos"
    ):
        st.markdown(
            "**Proporção — Wald com FPC**"
        )

        st.latex(
            r"IC_{95\%}(\hat p)=\hat p\pm"
            r"z_{0,975}"
            r"\sqrt{\frac{\hat p(1-\hat p)}{n}}"
            r"\sqrt{\frac{N-n}{N-1}}"
        )

        st.markdown(
            "**Média — t de Student com FPC**"
        )

        st.latex(
            r"IC_{95\%}(\bar x)=\bar x\pm"
            r"t_{0,975;n-1}"
            r"\frac{s}{\sqrt n}"
            r"\sqrt{\frac{N-n}{N-1}}"
        )


# ============================================================
# 21. TESTES DE HIPOTESES
# ============================================================

elif secao == "Testes de hipóteses":
    st.header(
        "Testes de hipóteses estatísticas"
    )

    st.markdown(
        """
        Foram realizados **dois testes Z para proporções**
        e **dois testes t para médias**.

        Todos são bilaterais, com significância de 5%.

        **Regra:** rejeitamos H0 quando o valor-p
        é inferior a 0,05.
        """
    )

    st.dataframe(
        tabela_testes(),
        hide_index=True,
        width="stretch",
    )

    st.subheader(
        "Hipóteses, estatísticas e interpretações"
    )

    for codigo in CODIGOS:
        with st.expander(
            f"{codigo} — {resultados[codigo]['nome']}",
            expanded=(codigo == "P1"),
        ):
            explicacao_teste(codigo)

    st.subheader(
        "Condições dos testes de proporção"
    )

    condicoes = []

    for codigo in ["P1", "P2"]:
        p0 = HIPOTESES[codigo]["referencia"]

        sucessos = n * p0
        insucessos = n * (1 - p0)

        condicoes.append({
            "Teste": codigo,
            "Sucessos esperados sob H0": br(
                sucessos, 1
            ),
            "Insucessos esperados sob H0": br(
                insucessos, 1
            ),
            "Critério ≥ 10": (
                "Atendido"
                if min(sucessos, insucessos) >= 10
                else "Não atendido"
            ),
        })

    st.dataframe(
        pd.DataFrame(condicoes),
        hide_index=True,
        width="stretch",
    )

    st.warning(
        "As distribuições de pessoas e feridos "
        "apresentam assimetria e valores extremos. "
        "Os testes t exigem interpretação cuidadosa."
    )

    st.divider()

    st.subheader(
        "Curvas normais — P1 e P2"
    )

    st.caption(
        "As curvas representam a distribuição "
        "teórica da estatística Z sob H0, "
        "não a distribuição dos acidentes."
    )

    grafico_curva_teste("P1")
    grafico_curva_teste("P2")

    st.divider()

    st.subheader(
        "Curvas t de Student — M1 e M2"
    )

    st.caption(
        f"Distribuição t de Student com "
        f"{GL} graus de liberdade."
    )

    grafico_curva_teste("M1")
    grafico_curva_teste("M2")

    st.info(
        "As linhas vermelhas indicam os valores "
        "críticos individuais. A linha verde "
        "representa a estatística observada. "
        "As áreas laranja correspondem ao valor-p "
        "bilateral e podem se sobrepor às áreas críticas."
    )

    st.warning(
        "P1 e M2 rejeitaram H0 individualmente, "
        "mas nenhuma hipótese permaneceu rejeitada "
        "após o ajuste de Holm."
    )


# ============================================================
# 22. SENSIBILIDADE
# ============================================================

elif secao == "Sensibilidade":
    st.header(
        "Análise de sensibilidade"
    )

    st.markdown(
        """
        Esta seção investiga a influência das
        ocorrências extremas nas médias M1 e M2.

        A exclusão é **hipotética** e utilizada
        apenas para diagnóstico.

        A amostra original permanece preservada.
        """
    )

    linhas = []

    for codigo in ["M1", "M2"]:
        s = sensibilidade[codigo]

        linhas.append({
            "Teste": codigo,
            "ID extremo": str(s["id_extremo"]),
            "Máximo": br(s["maximo"], 0),
            "Média original": br(
                s["media_original"], 6
            ),
            "Média sem máximo": br(
                s["media_sem_maximo"], 6
            ),
            "Média aparada 5%": br(
                s["media_aparada"], 6
            ),
            "p original": formato_p(
                s["p_original"]
            ),
            "p sem máximo": formato_p(
                s["p_sem_maximo"]
            ),
            "Decisão original": s["decisao_original"],
            "Decisão sem máximo": s["decisao_sem_maximo"],
        })

    st.dataframe(
        pd.DataFrame(linhas),
        hide_index=True,
        width="stretch",
    )

    fig = go.Figure()

    for codigo, cor in [
        ("M1", AZUL),
        ("M2", LARANJA),
    ]:
        s = sensibilidade[codigo]

        valores = [
            s["media_original"],
            s["media_sem_maximo"],
            s["media_aparada"],
        ]

        fig.add_trace(
            go.Bar(
                name=codigo,
                x=[
                    "Média original",
                    "Média sem máximo",
                    "Média aparada 5%",
                ],
                y=valores,
                marker_color=cor,
                text=[
                    br(valor, 3)
                    for valor in valores
                ],
                textposition="outside",
            )
        )

    fig.update_layout(
        title="Sensibilidade das médias",
        template="plotly_white",
        barmode="group",
        height=440,
        yaxis_title="Pessoas por acidente",
    )

    plotar(fig)

    st.subheader(
        "Interpretação"
    )

    st.markdown(
        """
        **M1 — Pessoas envolvidas**

        A média diminui de aproximadamente
        2,744 para 2,552 após a exclusão
        hipotética da ocorrência máxima.

        A decisão individual muda de
        não rejeitar para rejeitar H0.

        **M2 — Pessoas feridas**

        A média diminui de aproximadamente
        1,269 para 1,173.

        O teste continua rejeitando H0,
        mas seu valor-p muda.
        """
    )

    st.warning(
        "Esses resultados são diagnósticos "
        "de sensibilidade e não substituem "
        "os testes da amostra original."
    )


# ============================================================
# 23. HOLM
# ============================================================

elif secao == "Ajuste de Holm":
    st.header(
        "Comparações múltiplas — Holm"
    )

    st.markdown(
        """
        O ajuste de Holm controla o risco familiar
        de rejeições incorretas quando analisamos
        simultaneamente os quatro testes.

        **Antes do ajuste:** P1 e M2 rejeitavam H0.

        **Após o ajuste:** nenhuma hipótese
        permanece rejeitada ao nível de 5%.
        """
    )

    st.dataframe(
        tabela_holm(),
        hide_index=True,
        width="stretch",
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Valor-p original",
            x=CODIGOS,
            y=[
                resultados[c]["p"]
                for c in CODIGOS
            ],
            marker_color=AZUL,
            text=[
                formato_p(resultados[c]["p"])
                for c in CODIGOS
            ],
            textposition="outside",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Valor-p ajustado Holm",
            x=CODIGOS,
            y=[
                resultados[c]["p_holm"]
                for c in CODIGOS
            ],
            marker_color=LARANJA,
            text=[
                formato_p(resultados[c]["p_holm"])
                for c in CODIGOS
            ],
            textposition="outside",
        )
    )

    fig.add_hline(
        y=ALFA,
        line_dash="dash",
        line_color=VERMELHO,
        annotation_text="α = 0,05",
    )

    fig.update_layout(
        title="Valores-p antes e depois do ajuste",
        template="plotly_white",
        barmode="group",
        height=470,
        xaxis_title="Teste",
        yaxis_title="Valor-p",
        margin=dict(
            l=30,
            r=30,
            t=65,
            b=45,
        ),
    )

    plotar(fig)

    st.subheader(
        "Como funciona o procedimento de Holm?"
    )

    ordenados = sorted(
        CODIGOS,
        key=lambda c: resultados[c]["p"],
    )

    etapas = []

    total = len(ordenados)

    interrompido = False

    for posicao, codigo in enumerate(ordenados):
        limite = ALFA / (total - posicao)

        p_original = resultados[codigo]["p"]

        rejeitar = (
            not interrompido
            and p_original <= limite
        )

        if not rejeitar:
            interrompido = True

        etapas.append({
            "Ordem": posicao + 1,
            "Teste": codigo,
            "Valor-p": formato_p(p_original),
            "Limite Holm": br(limite, 6),
            "Decisão sequencial": (
                "Rejeitar H0"
                if rejeitar
                else "Não rejeitar H0"
            ),
        })

    st.dataframe(
        pd.DataFrame(etapas),
        hide_index=True,
        width="stretch",
    )

    st.info(
        "Os valores-p ajustados de P1 e M2 "
        "ficam próximos de 0,052755, "
        "ligeiramente acima de 0,05. "
        "Por isso, nenhuma hipótese permanece "
        "rejeitada após o ajuste."
    )

    st.caption(
        "O ajuste de Holm não invalida os "
        "testes individuais. Ele fornece uma "
        "interpretação complementar quando "
        "os quatro testes são considerados juntos."
    )


# ============================================================
# 24. METODOLOGIA E AUDITORIA
# ============================================================

elif secao == "Metodologia":
    st.header(
        "Metodologia e auditoria"
    )

    st.subheader(
        "Desenho amostral"
    )

    st.markdown(
        """
        - **População:** 72.529 registros da PRF 2025.
        - **Amostra:** 383 ocorrências.
        - **Método:** AAS sem reposição.
        - **Semente original:** `random_state=42`.
        - **Confiança:** 95%.
        - **Significância:** 5%.
        - **Testes:** bilaterais.
        """
    )

    st.subheader(
        "Correção para população finita"
    )

    st.latex(
        r"FPC=\sqrt{\frac{N-n}{N-1}}"
    )

    st.metric(
        "Fator utilizado",
        br(FPC, 6),
    )

    st.subheader(
        "Fórmulas dos intervalos"
    )

    st.markdown(
        "**Proporção — Wald**"
    )

    st.latex(
        r"IC_{95\%}=\hat p\pm"
        r"z_{0,975}"
        r"\sqrt{\frac{\hat p(1-\hat p)}{n}}"
        r"\cdot FPC"
    )

    st.markdown(
        "**Média — t de Student**"
    )

    st.latex(
        r"IC_{95\%}=\bar x\pm"
        r"t_{0,975;n-1}"
        r"\frac{s}{\sqrt n}"
        r"\cdot FPC"
    )

    st.subheader(
        "Estatísticas dos testes"
    )

    st.markdown(
        "**Teste Z para proporção**"
    )

    st.latex(
        r"Z=\frac{\hat p-p_0}"
        r"{\sqrt{\frac{p_0(1-p_0)}{n}}\cdot FPC}"
    )

    st.markdown(
        "**Teste t para média**"
    )

    st.latex(
        r"t=\frac{\bar x-\mu_0}"
        r"{\frac{s}{\sqrt n}\cdot FPC}"
    )

    st.subheader(
        "Auditoria computacional"
    )

    auditoria = pd.DataFrame([
        {
            "Verificação": "Registros populacionais",
            "Resultado": str(N),
            "Situação": "Aprovado",
        },
        {
            "Verificação": "Registros amostrais",
            "Resultado": str(n),
            "Situação": "Aprovado",
        },
        {
            "Verificação": "IDs únicos e pertencentes à população",
            "Resultado": "Sim",
            "Situação": "Aprovado",
        },
        {
            "Verificação": "Valores amostrais conferidos",
            "Resultado": (
                "Sim" if valores_conferidos else "Não"
            ),
            "Situação": "Aprovado",
        },
        {
            "Verificação": "Contagens da amostra",
            "Resultado": (
                "Conferidas" if contagens_validas
                else "Divergentes"
            ),
            "Situação": "Aprovado",
        },
        {
            "Verificação": "Parâmetros analisados",
            "Resultado": "4",
            "Situação": "Aprovado",
        },
        {
            "Verificação": "ICs contendo parâmetro",
            "Resultado": str(total_cobertura),
            "Situação": "Aprovado",
        },
        {
            "Verificação": "Resultados inferenciais",
            "Resultado": (
                "Conferidos" if resultados_conferidos
                else "Divergentes"
            ),
            "Situação": "Aprovado",
        },
        {
            "Verificação": "Rejeições individuais",
            "Resultado": str(rejeicoes),
            "Situação": "Conferido",
        },
        {
            "Verificação": "Rejeições após Holm",
            "Resultado": str(rejeicoes_holm),
            "Situação": "Conferido",
        },
        {
            "Verificação": "Amostra original",
            "Resultado": "Preservada",
            "Situação": "Aprovado",
        },
    ])

    st.dataframe(
        auditoria,
        hide_index=True,
        width="stretch",
    )

    st.subheader(
        "Visualizações complementares"
    )

    st.markdown(
        """
        **Curvas normais:** distribuições teóricas
        dos testes Z para P1 e P2.

        **Curvas t:** distribuições teóricas
        dos testes M1 e M2, com 382 graus
        de liberdade.

        **Gráficos comparativos:** parâmetros
        populacionais, estimativas amostrais
        e seus ICs de 95%.

        **Sensibilidade:** avaliação hipotética
        da influência de ocorrências extremas.

        **Holm:** controle complementar do erro
        familiar nas quatro comparações.
        """
    )

    st.subheader(
        "Limitações metodológicas"
    )

    st.markdown(
        """
        A população operacional corresponde
        aos registros disponíveis na base
        utilizada e não é automaticamente
        generalizável a acidentes não registrados.

        Os intervalos para proporções utilizam
        a aproximação normal de Wald.

        Os intervalos e testes para médias
        utilizam a distribuição t de Student.

        A assimetria e os valores extremos
        podem afetar a adequação dessas
        aproximações inferenciais.

        A distribuição teórica das estatísticas
        de teste não implica que as variáveis
        originais sejam normalmente distribuídas.

        A exclusão hipotética de ocorrências
        extremas não representa novo sorteio
        e não substitui os testes originais.

        As quatro hipóteses foram formuladas
        para fins acadêmicos exploratórios,
        não representando metas oficiais da PRF.

        A cobertura observada dos quatro ICs
        não garante cobertura em novas amostragens.

        Os testes individuais e o ajuste de Holm
        devem ser interpretados separadamente.
        """
    )


# ============================================================
# 25. RODAPE
# ============================================================

st.divider()

st.caption(
    "PRF Analytics 2025 | Dashboard 3 — "
    "Inferência estatística e comparação dos resultados | "
    "Probabilidade e Estatística para IA"
)
