
# ============================================================
# PRF ANALYTICS 2025
# DASHBOARD 3 - INFERENCIA ESTATISTICA
#
# Streamlit + Plotly
# Versao 1.0
#
# Populacao: 72.529 acidentes
# Amostra: 383 acidentes - AAS sem reposicao
# Semente original: 42
# Confianca: 95%
# Significancia: 5%
#
# Metodos:
# - Intervalos de Wald para proporcoes
# - Intervalos t de Student para medias
# - Correcao para populacao finita
# - Testes Z bilaterais para proporcoes
# - Testes t bilaterais para medias
# - Sensibilidade aos valores extremos
# - Ajuste de Holm
#
# Todos os procedimentos inferenciais utilizam
# exclusivamente a amostra original.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from scipy import stats
from pathlib import Path


# ============================================================
# 1. CONFIGURACAO
# ============================================================

st.set_page_config(
    page_title="Inferencia | PRF Analytics 2025",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

COR_AZUL = "#1677C8"
COR_VERDE = "#17846A"
COR_LARANJA = "#DF8731"
COR_VERMELHO = "#CB4545"
COR_CINZA = "#68798B"

N_ESPERADO = 72529
n_ESPERADO = 383

ALFA = 0.05
CONFIANCA = 0.95

DIRETORIO = Path(__file__).resolve().parents[1]

ARQUIVO_POPULACAO = (
    DIRETORIO / "datatran2025.csv"
)

ARQUIVO_AMOSTRA = (
    DIRETORIO / "amostra_PRF_2025.csv"
)


# ============================================================
# 2. FORMATACAO BRASILEIRA
# ============================================================

def numero_br(valor, casas=2):

    if pd.isna(valor):
        return "N/D"

    texto = f"{valor:,.{casas}f}"

    return (
        texto.replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


def percentual_br(valor, casas=2):

    return numero_br(valor * 100, casas) + "%"


def valor_p_br(valor):

    if not np.isfinite(valor):
        return "N/D"

    if valor < 0.000001:
        return "< 0,000001"

    return numero_br(valor, 6)


def decisao_teste(valor_p):

    if valor_p < ALFA:
        return "Rejeitar H0"

    return "Não rejeitar H0"


def nome_indicador(codigo):

    nomes = {
        "P1": "Acidentes com pelo menos uma morte",
        "P2": "Acidentes em plena noite",
        "M1": "Pessoas envolvidas",
        "M2": "Pessoas feridas"
    }

    return nomes[codigo]


# ============================================================
# 3. CARREGAMENTO DOS DADOS
# ============================================================

@st.cache_data(show_spinner="Carregando população...")
def carregar_populacao():

    df = pd.read_csv(
        ARQUIVO_POPULACAO,
        sep=";",
        encoding="latin1",
        low_memory=False
    )

    df.columns = df.columns.str.strip()

    return df


@st.cache_data(show_spinner="Carregando amostra...")
def carregar_amostra():

    tentativas = []

    for codificacao in ["utf-8-sig", "latin1"]:

        for separador in [",", ";", "\t"]:

            try:

                df = pd.read_csv(
                    ARQUIVO_AMOSTRA,
                    sep=separador,
                    encoding=codificacao,
                    low_memory=False
                )

                df.columns = df.columns.str.strip()

                if (
                    len(df) == n_ESPERADO
                    and len(df.columns) == 30
                    and "id" in df.columns
                ):
                    return df

                tentativas.append(
                    f"{codificacao} / {repr(separador)}: "
                    f"{df.shape}"
                )

            except (
                pd.errors.ParserError,
                UnicodeError
            ) as erro:

                tentativas.append(str(erro))

    raise ValueError(
        "Formato da amostra nao reconhecido. "
        + "; ".join(tentativas)
    )


try:

    populacao = carregar_populacao()
    amostra = carregar_amostra()

except Exception as erro:

    st.error("Erro no carregamento dos dados.")
    st.exception(erro)
    st.stop()


# ============================================================
# 4. VALIDACAO E PREPARACAO
# ============================================================

N = len(populacao)
n = len(amostra)

if N != N_ESPERADO or n != n_ESPERADO:

    st.error(
        "Os tamanhos dos arquivos nao correspondem "
        "aos utilizados no estudo."
    )

    st.stop()


colunas_necessarias = [
    "id",
    "mortos",
    "fase_dia",
    "pessoas",
    "feridos"
]

for coluna in colunas_necessarias:

    if (
        coluna not in populacao.columns
        or coluna not in amostra.columns
    ):

        st.error(
            f"Coluna obrigatoria ausente: {coluna}"
        )

        st.stop()


# Validar identificadores sem alterar o sorteio.

ids_populacao = pd.to_numeric(
    populacao["id"],
    errors="coerce"
)

ids_amostra = pd.to_numeric(
    amostra["id"],
    errors="coerce"
)

if (
    ids_amostra.isna().any()
    or ids_amostra.duplicated().any()
    or not ids_amostra.isin(ids_populacao).all()
):

    st.error(
        "Falha na validacao dos IDs da amostra."
    )

    st.stop()


for coluna in ["mortos", "pessoas", "feridos"]:

    populacao[coluna] = pd.to_numeric(
        populacao[coluna],
        errors="coerce"
    )

    amostra[coluna] = pd.to_numeric(
        amostra[coluna],
        errors="coerce"
    )


if (
    amostra[["mortos", "pessoas", "feridos"]]
    .isna().any().any()
):

    st.error(
        "Existem valores ausentes nas variaveis "
        "utilizadas nos testes."
    )

    st.stop()


for df in [populacao, amostra]:

    df["fase_dia"] = (
        df["fase_dia"]
        .astype("string")
        .str.strip()
    )


# Validacao dos valores amostrais contra a populacao.

colunas_conferencia = [
    "mortos",
    "pessoas",
    "feridos",
    "fase_dia"
]

pop_conf = populacao[
    ["id"] + colunas_conferencia
].copy()

amo_conf = amostra[
    ["id"] + colunas_conferencia
].copy()

pop_conf["id"] = ids_populacao
amo_conf["id"] = ids_amostra

if pop_conf["id"].duplicated().any():

    st.error(
        "IDs duplicados na população impedem "
        "a conferência integral da amostra."
    )
    st.stop()


conferencia = amo_conf.merge(
    pop_conf,
    on="id",
    how="left",
    suffixes=("_amostra", "_populacao"),
    validate="one_to_one"
)

valores_coincidentes = True

for coluna in colunas_conferencia:

    iguais = (
        conferencia[f"{coluna}_amostra"]
        .eq(conferencia[f"{coluna}_populacao"])
        .fillna(False)
        .all()
    )

    valores_coincidentes &= bool(iguais)


if not valores_coincidentes:

    st.error(
        "Foram encontradas divergências entre "
        "os valores da amostra e os registros "
        "correspondentes na população."
    )

    st.stop()


# ============================================================
# 5. CORRECAO PARA POPULACAO FINITA
# ============================================================

FPC = np.sqrt(
    (N - n) / (N - 1)
)

Z_CRITICO = stats.norm.ppf(
    1 - ALFA / 2
)

T_CRITICO = stats.t.ppf(
    1 - ALFA / 2,
    df=n - 1
)


# ============================================================
# 6. DEFINICAO DOS PARAMETROS
# ============================================================

def extrair_variavel(df, codigo):

    if codigo == "P1":

        return (
            df["mortos"] > 0
        ).astype(int).to_numpy()

    if codigo == "P2":

        return (
            df["fase_dia"] == "Plena Noite"
        ).astype(int).to_numpy()

    if codigo == "M1":

        return df["pessoas"].to_numpy(
            dtype=float
        )

    if codigo == "M2":

        return df["feridos"].to_numpy(
            dtype=float
        )

    raise ValueError(
        f"Parametro desconhecido: {codigo}"
    )


HIPOTESES = {
    "P1": {
        "tipo": "Proporção",
        "h0": 0.10,
        "evento": "mortos > 0",
        "pergunta": (
            "A proporção de acidentes com pelo "
            "menos uma morte difere de 10%?"
        )
    },
    "P2": {
        "tipo": "Proporção",
        "h0": 0.30,
        "evento": 'fase_dia == "Plena Noite"',
        "pergunta": (
            "A proporção de acidentes em plena "
            "noite difere de 30%?"
        )
    },
    "M1": {
        "tipo": "Média",
        "h0": 3.0,
        "evento": "pessoas",
        "pergunta": (
            "A média de pessoas envolvidas "
            "difere de três por acidente?"
        )
    },
    "M2": {
        "tipo": "Média",
        "h0": 1.0,
        "evento": "feridos",
        "pergunta": (
            "A média de feridos difere de uma "
            "pessoa por acidente?"
        )
    }
}


# ============================================================
# 7. INTERVALOS E TESTES
# ============================================================

def calcular_inferencia(codigo):

    configuracao = HIPOTESES[codigo]

    valores_amostra = extrair_variavel(
        amostra,
        codigo
    )

    valores_populacao = extrair_variavel(
        populacao,
        codigo
    )

    referencia = configuracao["h0"]

    estimativa = float(
        np.mean(valores_amostra)
    )

    parametro = float(
        np.mean(valores_populacao)
    )

    if configuracao["tipo"] == "Proporção":

        # IC bilateral de Wald
        ep_ic = np.sqrt(
            estimativa * (1 - estimativa) / n
        ) * FPC

        margem = Z_CRITICO * ep_ic

        ic_inferior = max(
            0.0,
            estimativa - margem
        )

        ic_superior = min(
            1.0,
            estimativa + margem
        )

        # Teste Z sob H0
        ep_h0 = np.sqrt(
            referencia * (1 - referencia) / n
        ) * FPC

        estatistica = (
            estimativa - referencia
        ) / ep_h0

        valor_p = 2 * stats.norm.sf(
            abs(estatistica)
        )

    else:

        # IC t bilateral
        desvio = float(
            np.std(valores_amostra, ddof=1)
        )

        ep = desvio / np.sqrt(n) * FPC

        margem = T_CRITICO * ep

        ic_inferior = estimativa - margem
        ic_superior = estimativa + margem

        # Teste t bilateral
        estatistica = (
            estimativa - referencia
        ) / ep

        valor_p = 2 * stats.t.sf(
            abs(estatistica),
            df=n - 1
        )

    return {
        "codigo": codigo,
        "nome": nome_indicador(codigo),
        "tipo": configuracao["tipo"],
        "referencia": referencia,
        "estimativa": estimativa,
        "populacao": parametro,
        "ic_inferior": ic_inferior,
        "ic_superior": ic_superior,
        "estatistica": float(estatistica),
        "valor_p": float(valor_p),
        "decisao": decisao_teste(valor_p),
        "contem": (
            ic_inferior
            <= parametro
            <= ic_superior
        )
    }


CODIGOS = ["P1", "P2", "M1", "M2"]

resultados = {
    codigo: calcular_inferencia(codigo)
    for codigo in CODIGOS
}


# ============================================================
# 8. AJUSTE DE HOLM
# ============================================================

def ajustar_holm(valores_p):

    valores = np.asarray(
        valores_p,
        dtype=float
    )

    quantidade = len(valores)

    ordem = np.argsort(valores)

    ajustados = np.empty(
        quantidade,
        dtype=float
    )

    maior_ajustado = 0.0

    for posicao, indice in enumerate(ordem):

        multiplicador = quantidade - posicao

        valor = min(
            1.0,
            valores[indice] * multiplicador
        )

        maior_ajustado = max(
            maior_ajustado,
            valor
        )

        ajustados[indice] = maior_ajustado

    return ajustados


p_originais = [
    resultados[codigo]["valor_p"]
    for codigo in CODIGOS
]

p_ajustados = ajustar_holm(
    p_originais
)

for indice, codigo in enumerate(CODIGOS):

    resultados[codigo]["p_holm"] = float(
        p_ajustados[indice]
    )

    resultados[codigo]["decisao_holm"] = (
        decisao_teste(p_ajustados[indice])
    )


# ============================================================
# 9. GRAFICO DOS INTERVALOS DE CONFIANCA
# ============================================================

def grafico_intervalos(codigos, titulo):

    fig = go.Figure()

    for codigo in codigos:

        r = resultados[codigo]

        estimativa = r["estimativa"]
        parametro = r["populacao"]

        inferior = r["ic_inferior"]
        superior = r["ic_superior"]

        multiplicador = (
            100
            if r["tipo"] == "Proporção"
            else 1
        )

        x = estimativa * multiplicador
        p = parametro * multiplicador

        erro_inferior = (
            estimativa - inferior
        ) * multiplicador

        erro_superior = (
            superior - estimativa
        ) * multiplicador

        y = f"{codigo} — {r['nome']}"

        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers",
                name="Estimativa amostral",
                marker=dict(
                    size=13,
                    color=COR_AZUL
                ),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[erro_superior],
                    arrayminus=[erro_inferior],
                    color=COR_AZUL,
                    thickness=3,
                    width=7
                ),
                hovertemplate=(
                    "<b>Estimativa amostral</b>"
                    "<br>%{x:.3f}"
                    "<extra></extra>"
                ),
                showlegend=(codigo == codigos[0])
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[p],
                y=[y],
                mode="markers",
                name="Parâmetro populacional",
                marker=dict(
                    size=14,
                    symbol="diamond",
                    color=COR_LARANJA
                ),
                hovertemplate=(
                    "<b>Parâmetro populacional</b>"
                    "<br>%{x:.3f}"
                    "<extra></extra>"
                ),
                showlegend=(codigo == codigos[0])
            )
        )

    unidade = (
        "Proporção (%)"
        if codigos[0].startswith("P")
        else "Média de pessoas por acidente"
    )

    fig.update_layout(
        title=titulo,
        template="plotly_white",
        height=330,
        xaxis_title=unidade,
        yaxis_title=None,
        font=dict(size=13),
        hovermode="closest",
        legend=dict(
            orientation="h",
            y=-0.35
        ),
        margin=dict(
            l=25,
            r=40,
            t=65,
            b=75
        )
    )

    fig.update_yaxes(
        categoryorder="array",
        categoryarray=[
            f"{codigo} — {resultados[codigo]['nome']}"
            for codigo in reversed(codigos)
        ]
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False}
    )


# ============================================================
# 10. SENSIBILIDADE A VALORES EXTREMOS
# ============================================================

def teste_t_vetor(valores, referencia, fpc):

    tamanho = len(valores)

    media = float(np.mean(valores))

    desvio = float(
        np.std(valores, ddof=1)
    )

    erro_padrao = (
        desvio / np.sqrt(tamanho) * fpc
    )

    estatistica = (
        media - referencia
    ) / erro_padrao

    valor_p = 2 * stats.t.sf(
        abs(estatistica),
        df=tamanho - 1
    )

    return (
        media,
        float(estatistica),
        float(valor_p)
    )


def calcular_sensibilidade(codigo):

    coluna = (
        "pessoas"
        if codigo == "M1"
        else "feridos"
    )

    referencia = HIPOTESES[codigo]["h0"]

    valores = amostra[coluna].to_numpy(
        dtype=float
    )

    indice_maximo = int(
        np.argmax(valores)
    )

    id_extremo = amostra.iloc[
        indice_maximo
    ]["id"]

    valor_maximo = valores[indice_maximo]

    # Copia temporaria para diagnostico.
    # A amostra original permanece preservada.

    valores_sem_maximo = np.delete(
        valores,
        indice_maximo
    )

    n_reduzido = len(
        valores_sem_maximo
    )

    fpc_reduzido = np.sqrt(
        (N - n_reduzido) / (N - 1)
    )

    media_original = float(
        np.mean(valores)
    )

    (
        media_sem_maximo,
        t_sem_maximo,
        p_sem_maximo
    ) = teste_t_vetor(
        valores_sem_maximo,
        referencia,
        fpc_reduzido
    )

    media_aparada = float(
        stats.trim_mean(
            valores,
            proportiontocut=0.05
        )
    )

    return {
        "codigo": codigo,
        "id_extremo": id_extremo,
        "maximo": valor_maximo,
        "media_original": media_original,
        "media_sem_maximo": media_sem_maximo,
        "media_aparada": media_aparada,
        "p_original": resultados[codigo]["valor_p"],
        "p_sem_maximo": p_sem_maximo,
        "t_sem_maximo": t_sem_maximo,
        "decisao_original": resultados[codigo]["decisao"],
        "decisao_sem_maximo": decisao_teste(
            p_sem_maximo
        )
    }


sensibilidade = {
    codigo: calcular_sensibilidade(codigo)
    for codigo in ["M1", "M2"]
}


# ============================================================
# 11. CABECALHO
# ============================================================

st.title(
    "📈 Inferência estatística e comparação"
)

st.markdown(
    """
    **PRF Analytics 2025 — Dashboard 3**

    Estimação de parâmetros populacionais a partir
    de uma amostra aleatória simples de
    **383 acidentes**, selecionada da população
    operacional de **72.529 ocorrências**.

    Os procedimentos apresentados foram definidos
    nas Etapas 10, 11 e 12 do estudo estatístico.

    **Nível de confiança:** 95%  
    **Nível de significância:** 5%  
    **Tipo de testes:** bilaterais  
    **Amostragem:** AAS sem reposição
    """
)

st.divider()


# ============================================================
# 12. NAVEGACAO
# ============================================================

st.sidebar.title("PRF Analytics 2025")

st.sidebar.markdown(
    "### Dashboard 3 — Inferência"
)

st.sidebar.info(
    "Os resultados inferenciais utilizam sempre "
    "os 383 acidentes originais. "
    "Não são aplicados filtros que alterem "
    "o desenho da amostra."
)

secao = st.sidebar.radio(
    "Ir para a seção",
    [
        "Visão geral",
        "Intervalos de confiança",
        "Testes de hipóteses",
        "Sensibilidade",
        "Ajuste de Holm",
        "Metodologia"
    ]
)

st.sidebar.caption(
    "Semente original: 42 | "
    "Amostra preservada"
)


# ============================================================
# 13. INDICADORES GERAIS
# ============================================================

total_cobertura = sum(
    resultados[c]["contem"]
    for c in CODIGOS
)

total_rejeicoes = sum(
    resultados[c]["valor_p"] < ALFA
    for c in CODIGOS
)

total_rejeicoes_holm = sum(
    resultados[c]["p_holm"] < ALFA
    for c in CODIGOS
)


c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Parâmetros analisados",
        "4"
    )

with c2:
    st.metric(
        "ICs contendo o parâmetro",
        f"{total_cobertura} de 4"
    )

with c3:
    st.metric(
        "Rejeições individuais",
        str(total_rejeicoes)
    )

with c4:
    st.metric(
        "Rejeições após Holm",
        str(total_rejeicoes_holm)
    )


st.caption(
    f"População: {numero_br(N, 0)} | "
    f"Amostra: {numero_br(n, 0)} | "
    f"FPC: {numero_br(FPC, 6)} | "
    f"Confiança: 95% | Alfa: 5%"
)

st.divider()


# ============================================================
# 14. COMPARACAO POPULACAO X AMOSTRA
# ============================================================

st.header(
    "1. Comparação entre população e amostra"
)

linhas_comparacao = []

for codigo in CODIGOS:

    r = resultados[codigo]

    if r["tipo"] == "Proporção":

        pop = percentual_br(r["populacao"])
        amo = percentual_br(r["estimativa"])

        diferenca = (
            r["estimativa"] - r["populacao"]
        ) * 100

        diff = (
            f"{numero_br(diferenca, 2)} p.p."
        )

        ic = (
            f"{percentual_br(r['ic_inferior'])} "
            f"a {percentual_br(r['ic_superior'])}"
        )

    else:

        pop = numero_br(r["populacao"], 3)
        amo = numero_br(r["estimativa"], 3)

        diferenca = (
            r["estimativa"] - r["populacao"]
        )

        diff = numero_br(diferenca, 3)

        ic = (
            f"{numero_br(r['ic_inferior'], 3)} "
            f"a {numero_br(r['ic_superior'], 3)}"
        )

    linhas_comparacao.append({
        "Teste": codigo,
        "Indicador": r["nome"],
        "População": pop,
        "Amostra": amo,
        "Diferença": diff,
        "IC de 95%": ic,
        "Contém parâmetro": (
            "Sim" if r["contem"] else "Não"
        )
    })


tabela_comparacao = pd.DataFrame(
    linhas_comparacao
)

st.dataframe(
    tabela_comparacao,
    use_container_width=True,
    hide_index=True
)

st.markdown(
    """
    As diferenças foram calculadas subtraindo o
    parâmetro populacional da estimativa amostral.

    Para proporções, são apresentadas em pontos
    percentuais. Para médias, são expressas em
    pessoas por acidente.
    """
)


# ============================================================
# 15. INTERVALOS DE CONFIANCA
# ============================================================

st.header(
    "2. Intervalos de confiança de 95%"
)

st.markdown(
    """
    Os pontos azuis representam as estimativas
    amostrais e as barras horizontais mostram os
    respectivos intervalos de confiança de 95%.

    Os losangos laranja representam os parâmetros
    calculados diretamente na população completa.
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

st.info(
    "Os quatro intervalos abrangem seus "
    "respectivos parâmetros populacionais "
    "nesta amostra específica."
)

st.caption(
    "Os ICs de proporção utilizam a aproximação "
    "normal de Wald. Os ICs de média utilizam "
    "t de Student. Ambos incluem correção para "
    "população finita."
)

st.divider()


# ============================================================
# 16. TESTES DE HIPOTESES
# ============================================================

st.header(
    "3. Testes de hipóteses"
)

st.markdown(
    """
    Foram realizados quatro testes bilaterais,
    com nível de significância de 5%.

    Os valores das hipóteses nulas foram
    definidos previamente como referências
    exploratórias e não representam metas
    oficiais da PRF.
    """
)


linhas_testes = []

for codigo in CODIGOS:

    r = resultados[codigo]

    referencia = (
        percentual_br(r["referencia"])
        if r["tipo"] == "Proporção"
        else numero_br(r["referencia"], 2)
    )

    estimativa = (
        percentual_br(r["estimativa"])
        if r["tipo"] == "Proporção"
        else numero_br(r["estimativa"], 3)
    )

    linhas_testes.append({
        "Teste": codigo,
        "Indicador": r["nome"],
        "Estimativa": estimativa,
        "H0": referencia,
        "Estatística": numero_br(
            r["estatistica"], 4
        ),
        "Valor-p": valor_p_br(
            r["valor_p"]
        ),
        "Decisão": r["decisao"]
    })


st.dataframe(
    pd.DataFrame(linhas_testes),
    use_container_width=True,
    hide_index=True
)


st.subheader("Hipóteses e interpretações")

for codigo in CODIGOS:

    r = resultados[codigo]
    h = HIPOTESES[codigo]

    with st.expander(
        f"{codigo} — {r['nome']}"
    ):

        st.markdown(
            f"**Questão de pesquisa:** {h['pergunta']}"
        )

        st.markdown(
            f"**Variável/evento:** `{h['evento']}`"
        )

        if r["tipo"] == "Proporção":

            st.latex(
                rf"H_0: p = {h['h0']:.2f}"
            )

            st.latex(
                rf"H_1: p \neq {h['h0']:.2f}"
            )

        else:

            st.latex(
                rf"H_0: \mu = {h['h0']:.0f}"
            )

            st.latex(
                rf"H_1: \mu \neq {h['h0']:.0f}"
            )

        st.write(
            "**Estatística de teste:**",
            numero_br(r["estatistica"], 6)
        )

        st.write(
            "**Valor-p:**",
            valor_p_br(r["valor_p"])
        )

        st.write(
            "**Decisão:**",
            r["decisao"]
        )

        if r["valor_p"] < ALFA:

            st.success(
                "Foram encontradas evidências "
                "estatísticas suficientes para "
                "rejeitar a hipótese nula ao "
                "nível de 5%."
            )

        else:

            st.info(
                "Não foram encontradas evidências "
                "estatísticas suficientes para "
                "rejeitar a hipótese nula ao "
                "nível de 5%."
            )

        st.caption(
            "Não rejeitar H0 não significa "
            "comprovar que H0 seja verdadeira."
        )


# ============================================================
# 17. VERIFICACAO DAS CONDICOES
# ============================================================

st.subheader(
    "Condições dos testes de proporção"
)

linhas_condicoes = []

for codigo in ["P1", "P2"]:

    p0 = HIPOTESES[codigo]["h0"]

    sucessos = n * p0
    insucessos = n * (1 - p0)

    linhas_condicoes.append({
        "Teste": codigo,
        "Sucessos esperados sob H0": numero_br(
            sucessos, 1
        ),
        "Insucessos esperados sob H0": numero_br(
            insucessos, 1
        ),
        "Critério ≥ 10": (
            "Atendido"
            if min(sucessos, insucessos) >= 10
            else "Não atendido"
        )
    })


st.dataframe(
    pd.DataFrame(linhas_condicoes),
    hide_index=True,
    use_container_width=True
)

st.caption(
    "Para médias, os testes t usam 382 graus "
    "de liberdade. A assimetria e os valores "
    "extremos exigem cautela na interpretação."
)

st.divider()


# ============================================================
# 18. ANALISE DE SENSIBILIDADE
# ============================================================

st.header(
    "4. Análise de sensibilidade"
)

st.markdown(
    """
    Foi investigada a influência da ocorrência
    máxima sobre as médias e os testes t.

    A exclusão foi apenas hipotética e realizada
    em cópias temporárias das variáveis.

    **A amostra original não foi modificada.**

    Os resultados recalculados sem o máximo
    são diagnósticos exploratórios, e não novos
    testes confirmatórios válidos sob o sorteio
    original.
    """
)


linhas_sensibilidade = []

for codigo in ["M1", "M2"]:

    s = sensibilidade[codigo]

    linhas_sensibilidade.append({
        "Teste": codigo,
        "ID extremo": str(s["id_extremo"]),
        "Máximo": numero_br(s["maximo"], 0),
        "Média original": numero_br(
            s["media_original"], 6
        ),
        "Média sem máximo": numero_br(
            s["media_sem_maximo"], 6
        ),
        "Média aparada 5%": numero_br(
            s["media_aparada"], 6
        ),
        "p original": valor_p_br(
            s["p_original"]
        ),
        "p sem máximo": valor_p_br(
            s["p_sem_maximo"]
        ),
        "Decisão original": s["decisao_original"],
        "Decisão sem máximo": s["decisao_sem_maximo"]
    })


st.dataframe(
    pd.DataFrame(linhas_sensibilidade),
    use_container_width=True,
    hide_index=True
)


# Grafico de sensibilidade

fig_sens = go.Figure()

for codigo in ["M1", "M2"]:

    s = sensibilidade[codigo]

    fig_sens.add_trace(
        go.Bar(
            name=codigo,
            x=[
                "Média original",
                "Média sem máximo",
                "Média aparada 5%"
            ],
            y=[
                s["media_original"],
                s["media_sem_maximo"],
                s["media_aparada"]
            ],
            text=[
                numero_br(
                    s["media_original"], 3
                ),
                numero_br(
                    s["media_sem_maximo"], 3
                ),
                numero_br(
                    s["media_aparada"], 3
                )
            ],
            textposition="outside"
        )
    )


fig_sens.update_layout(
    title="Sensibilidade das médias",
    template="plotly_white",
    barmode="group",
    height=440,
    yaxis_title="Pessoas por acidente",
    xaxis_title=None,
    font=dict(size=13),
    margin=dict(
        l=30,
        r=30,
        t=65,
        b=55
    )
)

st.plotly_chart(
    fig_sens,
    use_container_width=True,
    config={"displaylogo": False}
)

st.warning(
    "O teste M1 é sensível à observação extrema: "
    "sua decisão muda no cenário hipotético "
    "sem o valor máximo. "
    "Essa mudança não autoriza a exclusão "
    "do registro original."
)

st.divider()


# ============================================================
# 19. AJUSTE PARA COMPARACOES MULTIPLAS
# ============================================================

st.header(
    "5. Comparações múltiplas — Holm"
)

st.markdown(
    """
    Os quatro valores-p dos testes individuais
    foram submetidos ao ajuste de Holm.

    O procedimento controla a probabilidade de
    pelo menos uma rejeição indevida na família
    de hipóteses testadas, sob suas condições
    de aplicação.

    Foi mantido nível de significância familiar
    de 5%.
    """
)


linhas_holm = []

for codigo in CODIGOS:

    r = resultados[codigo]

    linhas_holm.append({
        "Teste": codigo,
        "Indicador": r["nome"],
        "Valor-p original": valor_p_br(
            r["valor_p"]
        ),
        "Decisão original": r["decisao"],
        "Valor-p ajustado Holm": valor_p_br(
            r["p_holm"]
        ),
        "Decisão Holm": r["decisao_holm"]
    })


st.dataframe(
    pd.DataFrame(linhas_holm),
    hide_index=True,
    use_container_width=True
)


# Grafico de valores-p

fig_holm = go.Figure()

fig_holm.add_trace(
    go.Bar(
        name="Valor-p original",
        x=CODIGOS,
        y=[
            resultados[c]["valor_p"]
            for c in CODIGOS
        ],
        marker_color=COR_AZUL
    )
)

fig_holm.add_trace(
    go.Bar(
        name="Valor-p ajustado (Holm)",
        x=CODIGOS,
        y=[
            resultados[c]["p_holm"]
            for c in CODIGOS
        ],
        marker_color=COR_LARANJA
    )
)

fig_holm.add_hline(
    y=ALFA,
    line_dash="dash",
    line_color=COR_VERMELHO,
    annotation_text="Alfa = 0,05"
)

fig_holm.update_layout(
    title="Valores-p antes e depois do ajuste de Holm",
    template="plotly_white",
    barmode="group",
    height=440,
    xaxis_title="Teste",
    yaxis_title="Valor-p",
    font=dict(size=13),
    margin=dict(
        l=25,
        r=25,
        t=65,
        b=45
    )
)

st.plotly_chart(
    fig_holm,
    use_container_width=True,
    config={"displaylogo": False}
)


st.info(
    "Nos testes individuais, P1 e M2 rejeitam H0. "
    "Após o ajuste de Holm, nenhuma das quatro "
    "hipóteses nulas é rejeitada ao nível "
    "familiar de 5%."
)

st.divider()


# ============================================================
# 20. SINTESE E INTERPRETACAO
# ============================================================

st.header(
    "6. Síntese inferencial"
)

st.markdown(
    """
    ### Principais conclusões

    **Intervalos de confiança**

    Os quatro intervalos bilaterais de 95%
    incluíram os respectivos parâmetros
    populacionais conhecidos nesta amostra.

    **Testes individuais**

    As hipóteses nulas P1 e M2 foram rejeitadas
    ao nível de significância de 5%.

    Não houve evidência suficiente para rejeitar
    as hipóteses nulas P2 e M1.

    **Comparações múltiplas**

    Após o ajuste de Holm, nenhuma das hipóteses
    nulas permaneceu rejeitada ao nível familiar
    de 5%.

    **Valores extremos**

    A análise de sensibilidade mostrou influência
    relevante das observações extremas sobre
    as estimativas das médias e os resultados
    dos testes t.

    **Variabilidade amostral**

    As diferenças entre estimativas e parâmetros
    populacionais são esperadas em consequência
    do processo de amostragem.

    A inclusão dos parâmetros nos intervalos
    observados não garante cobertura em todas
    as possíveis amostras.
    """
)

st.divider()


# ============================================================
# 21. METODOLOGIA E LIMITACOES
# ============================================================

with st.expander(
    "Metodologia estatística e limitações"
):

    st.markdown(
        """
        **População operacional**

        72.529 acidentes registrados pela PRF
        em 2025, conforme a base utilizada.

        **Amostra**

        383 ocorrências selecionadas anteriormente
        por amostragem aleatória simples sem reposição.

        **Semente do sorteio original**

        `random_state=42`

        **Nível de confiança**

        95%.

        **Significância**

        5%.

        **Intervalos para proporções**

        Método bilateral de Wald, com erro-padrão
        estimado pela proporção amostral.

        **Intervalos para médias**

        Distribuição t de Student, utilizando
        o desvio-padrão amostral com `ddof=1`.

        **Correção para população finita**

        Aplicada aos erros-padrão dos intervalos
        e dos testes em razão da amostragem sem
        reposição.

        **Testes para proporções**

        Testes Z bilaterais, com erros-padrão
        calculados sob as hipóteses nulas.

        **Testes para médias**

        Testes t bilaterais, com 382 graus
        de liberdade.

        **Comparações múltiplas**

        Método sequencial de Holm aplicado
        aos quatro testes originais.

        **Sensibilidade**

        Exclusão hipotética de uma ocorrência
        máxima por variável e cálculo de médias
        aparadas em 5%.

        Nenhuma observação foi removida da
        amostra utilizada nos resultados principais.

        **Limitações**

        A população operacional corresponde
        aos registros disponíveis na base utilizada.

        Os resultados não são automaticamente
        generalizáveis a acidentes não registrados
        ou a outros períodos.

        Os procedimentos para médias utilizam
        aproximações cuja adequação pode ser
        afetada por assimetria e valores extremos.

        Os intervalos de proporções utilizam
        a aproximação de Wald.

        A significância estatística não implica
        necessariamente relevância prática ou
        relação causal.

        Os valores das hipóteses nulas foram
        definidos para fins acadêmicos exploratórios.
        """
    )


# ============================================================
# 22. AUDITORIA COMPUTACIONAL
# ============================================================

with st.expander(
    "Auditoria e validação computacional"
):

    auditoria = pd.DataFrame([
        {
            "Verificação": "População",
            "Resultado": N,
            "Situação": (
                "Aprovado" if N == 72529
                else "Verificar"
            )
        },
        {
            "Verificação": "Amostra",
            "Resultado": n,
            "Situação": (
                "Aprovado" if n == 383
                else "Verificar"
            )
        },
        {
            "Verificação": "IDs da amostra",
            "Resultado": "Válidos e únicos",
            "Situação": "Aprovado"
        },
        {
            "Verificação": "Valores amostrais",
            "Resultado": "Coincidentes com a população",
            "Situação": (
                "Aprovado"
                if valores_coincidentes
                else "Verificar"
            )
        },
        {
            "Verificação": "Parâmetros analisados",
            "Resultado": len(resultados),
            "Situação": "Aprovado"
        },
        {
            "Verificação": "ICs contendo parâmetro",
            "Resultado": total_cobertura,
            "Situação": (
                "Aprovado"
                if total_cobertura == 4
                else "Verificar"
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
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Os cálculos são reconstruídos a partir "
        "dos arquivos CSV. A amostra não é "
        "sorteada novamente."
    )


# ============================================================
# 23. RODAPE
# ============================================================

st.divider()

st.caption(
    "PRF Analytics 2025 | "
    "Dashboard 3 — Inferência estatística e "
    "comparação dos resultados | "
    "Probabilidade e Estatística para IA"
)
