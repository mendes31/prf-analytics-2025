
# ============================================================
# PRF ANALYTICS 2025
# DASHBOARD 1 - ANÁLISE DESCRITIVA DA POPULAÇÃO
#
# Streamlit + Plotly
# População operacional: 72.529 acidentes
# Fonte: Polícia Rodoviária Federal - 2025
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from pathlib import Path


# ============================================================
# 1. CONFIGURAÇÃO
# ============================================================

st.set_page_config(
    page_title="População | PRF Analytics 2025",
    page_icon="📊",
    layout="wide"
)

COR_PRINCIPAL = "#1677C8"
COR_SECUNDARIA = "#0F466E"
COR_DESTAQUE = "#E58A2B"

ARQUIVO = (
    Path(__file__).resolve().parents[1]
    / "datatran2025.csv"
)

TOTAL_ESPERADO = 72529


# ============================================================
# 2. FUNÇÕES AUXILIARES
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


@st.cache_data(show_spinner="Carregando dados da PRF...")
def carregar_dados():

    df = pd.read_csv(
        ARQUIVO,
        sep=";",
        encoding="latin1",
        low_memory=False
    )

    df.columns = df.columns.str.strip()

    numericas = [
        "pessoas",
        "mortos",
        "feridos",
        "feridos_leves",
        "feridos_graves",
        "veiculos"
    ]

    for coluna in numericas:
        if coluna in df.columns:
            df[coluna] = pd.to_numeric(
                df[coluna],
                errors="coerce"
            )

    df["data_inversa"] = pd.to_datetime(
        df["data_inversa"],
        errors="coerce"
    )

    return df


def tabela_frequencia(df, coluna):

    serie = (
        df[coluna]
        .fillna("Não informado")
        .astype(str)
        .str.strip()
        .replace("", "Não informado")
    )

    resultado = (
        serie.value_counts(dropna=False)
        .rename_axis("Categoria")
        .reset_index(name="Frequência")
    )

    resultado["Percentual"] = (
        resultado["Frequência"]
        / len(df) * 100
        if len(df) > 0
        else 0.0
    )

    return resultado


def grafico_qualitativo(
    df,
    coluna,
    titulo,
    top_n=None
):

    tabela = tabela_frequencia(
        df,
        coluna
    )

    if top_n is not None:
        tabela = tabela.head(top_n).copy()

    tabela = tabela.sort_values(
        ["Percentual", "Categoria"],
        ascending=[True, True]
    )

    fig = px.bar(
        tabela,
        x="Percentual",
        y="Categoria",
        orientation="h",
        text="Percentual",
        color_discrete_sequence=[COR_PRINCIPAL],
        custom_data=["Frequência"]
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{y}</b>"
            "<br>Participação: %{x:.2f}%"
            "<br>Ocorrências: %{customdata[0]}"
            "<extra></extra>"
        )
    )

    maximo = tabela["Percentual"].max()

    fig.update_layout(
        title=titulo,
        xaxis_title="% das ocorrências selecionadas",
        yaxis_title=None,
        height=max(310, len(tabela) * 39 + 110),
        margin=dict(l=15, r=65, t=60, b=35),
        xaxis=dict(
            range=[0, max(1, maximo * 1.16)],
            rangemode="tozero",
            ticksuffix="%"
        ),
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


def grafico_quantitativo(
    df,
    coluna,
    titulo,
    limite
):

    serie = pd.to_numeric(
        df[coluna],
        errors="coerce"
    ).dropna()

    categorias = [
        str(i) for i in range(limite)
    ] + [f"{limite} ou mais"]

    grupos = serie.apply(
        lambda x: (
            str(int(x))
            if x < limite
            else f"{limite} ou mais"
        )
    )

    frequencias = (
        grupos.value_counts()
        .reindex(categorias, fill_value=0)
    )

    tabela = pd.DataFrame({
        "Categoria": categorias,
        "Frequência": frequencias.values
    })

    tabela["Percentual"] = (
        tabela["Frequência"] / len(df) * 100
        if len(df) > 0
        else 0.0
    )

    fig = px.bar(
        tabela,
        x="Categoria",
        y="Percentual",
        text="Percentual",
        color_discrete_sequence=[COR_PRINCIPAL],
        custom_data=["Frequência"]
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{x}</b>"
            "<br>Participação: %{y:.2f}%"
            "<br>Ocorrências: %{customdata[0]}"
            "<extra></extra>"
        )
    )

    maximo = tabela["Percentual"].max()

    fig.update_layout(
        title=titulo,
        xaxis_title="Quantidade por acidente",
        yaxis_title="% das ocorrências selecionadas",
        height=410,
        margin=dict(l=20, r=20, t=65, b=75),
        yaxis=dict(
            range=[0, max(1, maximo * 1.16)]
        ),
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# 3. CARREGAMENTO E VALIDAÇÃO
# ============================================================

try:
    populacao = carregar_dados()

except Exception as erro:
    st.error("Erro ao carregar a população.")
    st.exception(erro)
    st.stop()

if len(populacao) != TOTAL_ESPERADO:
    st.error(
        "A quantidade de registros não corresponde "
        "à população validada no Google Colab."
    )
    st.stop()

colunas_necessarias = [
    "id",
    "data_inversa",
    "uf",
    "fase_dia",
    "classificacao_acidente",
    "tipo_acidente",
    "causa_acidente",
    "condicao_metereologica",
    "pessoas",
    "mortos",
    "feridos",
    "feridos_leves",
    "feridos_graves",
    "veiculos"
]

ausentes = [
    c for c in colunas_necessarias
    if c not in populacao.columns
]

if ausentes:
    st.error(
        "Colunas não encontradas na base: "
        + ", ".join(ausentes)
    )
    st.stop()


# ============================================================
# 4. CABEÇALHO
# ============================================================

st.title("📊 Análise descritiva da população")

st.markdown(
    """
    **PRF Analytics 2025**

    Análise das ocorrências registradas nas
    rodovias federais brasileiras.

    **População operacional:** 72.529 acidentes.

    Explore os indicadores utilizando os filtros.
    Sem filtros, são apresentados os resultados
    da população completa.
    """
)

st.divider()


# ============================================================
# 5. FILTROS INTERATIVOS
# ============================================================

st.sidebar.title("PRF Analytics 2025")
st.sidebar.markdown("### Filtros da população")


def opcoes_filtro(coluna):
    return sorted(
        populacao[coluna]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


ufs = st.sidebar.multiselect(
    "Unidade federativa",
    options=opcoes_filtro("uf")
)

fases = st.sidebar.multiselect(
    "Fase do dia",
    options=opcoes_filtro("fase_dia")
)

classificacoes = st.sidebar.multiselect(
    "Classificação do acidente",
    options=opcoes_filtro("classificacao_acidente")
)

tipos = st.sidebar.multiselect(
    "Tipo de acidente",
    options=opcoes_filtro("tipo_acidente")
)

df = populacao.copy()

if ufs:
    df = df[df["uf"].isin(ufs)]

if fases:
    df = df[df["fase_dia"].isin(fases)]

if classificacoes:
    df = df[
        df["classificacao_acidente"].isin(
            classificacoes
        )
    ]

if tipos:
    df = df[df["tipo_acidente"].isin(tipos)]

filtros_ativos = any([
    bool(ufs),
    bool(fases),
    bool(classificacoes),
    bool(tipos)
])

if df.empty:
    st.warning(
        "Nenhuma ocorrência corresponde "
        "aos filtros selecionados."
    )
    st.stop()


# ============================================================
# 6. CONTEXTO DOS FILTROS
# ============================================================

if filtros_ativos:

    st.warning(
        "Visualização filtrada: os resultados abaixo "
        "descrevem apenas as ocorrências selecionadas. "
        "Não representam os parâmetros da população "
        "completa usados nos testes estatísticos."
    )

else:

    st.success(
        "Visualização da população completa: "
        "72.529 ocorrências."
    )

st.caption(
    f"Registros selecionados: "
    f"{numero_br(len(df))} de "
    f"{numero_br(len(populacao))} "
    f"({numero_br(percentual(len(df), len(populacao)), 2)}%)."
)


# ============================================================
# 7. INDICADORES PRINCIPAIS
# ============================================================

st.subheader("Indicadores gerais")

total_acidentes = len(df)
total_pessoas = int(df["pessoas"].sum())
total_mortos = int(df["mortos"].sum())
total_feridos = int(df["feridos"].sum())
total_veiculos = int(df["veiculos"].sum())

acidentes_fatais = int(
    (df["mortos"] > 0).sum()
)

acidentes_feridos = int(
    (df["feridos"] > 0).sum()
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Acidentes",
        numero_br(total_acidentes)
    )

with c2:
    st.metric(
        "Pessoas envolvidas",
        numero_br(total_pessoas)
    )

with c3:
    st.metric(
        "Vítimas fatais",
        numero_br(total_mortos)
    )

with c4:
    st.metric(
        "Pessoas feridas",
        numero_br(total_feridos)
    )

c5, c6, c7, c8 = st.columns(4)

with c5:
    st.metric(
        "Veículos envolvidos",
        numero_br(total_veiculos)
    )

with c6:
    st.metric(
        "Acidentes com mortes",
        numero_br(acidentes_fatais)
    )

with c7:
    st.metric(
        "Proporção com mortes",
        numero_br(
            percentual(acidentes_fatais, total_acidentes),
            2
        ) + "%"
    )

with c8:
    st.metric(
        "Acidentes com feridos",
        numero_br(acidentes_feridos)
    )

st.divider()


# ============================================================
# 8. ESTATÍSTICAS DESCRITIVAS
# ============================================================

st.subheader("Medidas estatísticas")

variaveis = [
    "pessoas",
    "mortos",
    "feridos",
    "feridos_leves",
    "feridos_graves",
    "veiculos"
]

linhas = []

for coluna in variaveis:

    serie = df[coluna].dropna()

    linhas.append({
        "Variável": coluna,
        "Observações válidas": len(serie),
        "Média": serie.mean(),
        "Mediana": serie.median(),
        "Desvio-padrão populacional": (
            serie.std(ddof=0)
        ),
        "Mínimo": serie.min(),
        "1º quartil": serie.quantile(0.25),
        "3º quartil": serie.quantile(0.75),
        "Máximo": serie.max()
    })

estatisticas = pd.DataFrame(linhas)

st.dataframe(
    estatisticas.style.format({
        "Média": "{:.3f}",
        "Mediana": "{:.2f}",
        "Desvio-padrão populacional": "{:.3f}",
        "Mínimo": "{:.0f}",
        "1º quartil": "{:.2f}",
        "3º quartil": "{:.2f}",
        "Máximo": "{:.0f}"
    }),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Desvio-padrão calculado com ddof=0, "
    "por se tratar da descrição do conjunto "
    "populacional selecionado."
)

st.divider()


# ============================================================
# 9. DISTRIBUIÇÕES QUANTITATIVAS
# ============================================================

st.subheader("Distribuições quantitativas")

st.markdown(
    """
    Os gráficos apresentam frequências relativas,
    respeitando a natureza discreta das variáveis.

    Valores elevados foram reunidos nas categorias
    finais, sem exclusão de registros.
    """
)

q1, q2 = st.columns(2)

with q1:
    grafico_quantitativo(
        df,
        "pessoas",
        "Pessoas envolvidas",
        limite=10
    )

with q2:
    grafico_quantitativo(
        df,
        "mortos",
        "Vítimas fatais",
        limite=4
    )

grafico_quantitativo(
    df,
    "feridos",
    "Pessoas feridas",
    limite=6
)

st.divider()


# ============================================================
# 10. DISTRIBUIÇÕES QUALITATIVAS
# ============================================================

st.subheader("Distribuições qualitativas")

st.markdown(
    """
    Os gráficos apresentam percentuais calculados
    sobre o total de ocorrências selecionadas.

    Para variáveis com muitas categorias, são exibidas
    as dez categorias mais frequentes no conjunto
    atualmente selecionado. As demais permanecem
    incluídas no denominador.
    """
)

grafico_qualitativo(
    df,
    "classificacao_acidente",
    "Classificação dos acidentes"
)

grafico_qualitativo(
    df,
    "fase_dia",
    "Fase do dia"
)

grafico_qualitativo(
    df,
    "tipo_acidente",
    "10 principais tipos de acidente",
    top_n=10
)

grafico_qualitativo(
    df,
    "causa_acidente",
    "10 principais causas registradas",
    top_n=10
)

grafico_qualitativo(
    df,
    "uf",
    "10 principais unidades federativas",
    top_n=10
)

grafico_qualitativo(
    df,
    "condicao_metereologica",
    "Condições meteorológicas"
)

st.divider()


# ============================================================
# 11. TABELA DE FREQUÊNCIAS
# ============================================================

st.subheader("Exploração das frequências")

colunas_categoricas = {
    "Classificação": "classificacao_acidente",
    "Fase do dia": "fase_dia",
    "Tipo de acidente": "tipo_acidente",
    "Causa registrada": "causa_acidente",
    "Unidade federativa": "uf",
    "Condição meteorológica": "condicao_metereologica"
}

selecionada = st.selectbox(
    "Selecione uma variável",
    options=list(colunas_categoricas.keys())
)

tabela = tabela_frequencia(
    df,
    colunas_categoricas[selecionada]
)

st.dataframe(
    tabela.style.format({
        "Percentual": "{:.2f}%"
    }),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "A tabela apresenta todas as categorias, "
    "inclusive aquelas omitidas dos gráficos Top 10."
)

st.divider()


# ============================================================
# 12. REGISTROS DA BASE
# ============================================================

with st.expander("Consultar registros selecionados"):

    st.dataframe(
        df[
            [
                "id",
                "data_inversa",
                "uf",
                "fase_dia",
                "tipo_acidente",
                "pessoas",
                "mortos",
                "feridos"
            ]
        ].head(1000),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Prévia limitada aos primeiros 1.000 registros "
        "do conjunto selecionado."
    )


# ============================================================
# 13. METODOLOGIA
# ============================================================

with st.expander("Metodologia e limitações"):

    st.markdown(
        """
        **Fonte**

        Polícia Rodoviária Federal — base de
        acidentes de trânsito de 2025.

        **População operacional**

        72.529 ocorrências utilizadas no estudo.

        **Unidade de análise**

        Uma ocorrência registrada.

        **Estatísticas descritivas**

        Médias, medianas, quartis, valores extremos
        e desvios-padrão populacionais.

        **Frequências**

        Frequências absolutas e relativas para
        variáveis qualitativas e quantitativas discretas.

        **Filtros**

        Os filtros permitem explorar subconjuntos
        da população operacional.

        Os indicadores filtrados não substituem
        os parâmetros populacionais definidos no
        estudo inferencial.

        **Limitações**

        As frequências representam a distribuição
        dos registros, não taxas de risco.

        Comparações entre estados, horários ou
        condições meteorológicas não consideram
        necessariamente a exposição ao tráfego.

        As causas são classificações registradas
        pela PRF e não demonstram, isoladamente,
        relações causais adicionais.
        """
    )


# ============================================================
# 14. RODAPÉ
# ============================================================

st.divider()

st.caption(
    "PRF Analytics 2025 | Dashboard 1 — População | "
    "Probabilidade e Estatística para IA"
)
