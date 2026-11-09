
# ============================================================
# PRF ANALYTICS 2025
# DASHBOARD 2 - ANALISE DESCRITIVA DA AMOSTRA
#
# Versao 1.0
# Streamlit + Plotly
#
# Populacao operacional: 72.529 acidentes
# Amostra aleatoria simples: 383 acidentes
# Semente do sorteio original: 42
#
# A amostra nao sera sorteada novamente.
# Todos os indicadores sao calculados com a amostra.
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from pathlib import Path
from textwrap import wrap


# ============================================================
# 1. CONFIGURACAO
# ============================================================

st.set_page_config(
    page_title="Amostra | PRF Analytics 2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

COR_PRINCIPAL = "#1677C8"
COR_TEXTO = "#253247"
COR_GRADE = "#E9EDF2"

N_POPULACAO = 72529
N_AMOSTRA = 383
SEMENTE_ORIGINAL = 42

DIRETORIO = Path(__file__).resolve().parents[1]

ARQUIVO_AMOSTRA = (
    DIRETORIO / "amostra_PRF_2025.csv"
)

ARQUIVO_POPULACAO = (
    DIRETORIO / "datatran2025.csv"
)


# ============================================================
# 2. FUNCOES AUXILIARES
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


def quebrar_rotulo(texto, largura=44):

    partes = wrap(
        str(texto),
        width=largura,
        break_long_words=False,
        break_on_hyphens=False
    )

    return "<br>".join(partes) if partes else str(texto)


def normalizar_categoria(serie):

    return (
        serie.fillna("Não informado")
        .astype(str)
        .str.strip()
        .replace("", "Não informado")
    )


def tabela_frequencia(df, coluna):

    serie = normalizar_categoria(df[coluna])

    resultado = (
        serie.value_counts(dropna=False)
        .rename_axis("Categoria")
        .reset_index(name="Frequência")
    )

    resultado["Percentual"] = (
        resultado["Frequência"] / len(df) * 100
        if len(df) > 0
        else 0.0
    )

    return resultado


# ============================================================
# 3. CARREGAMENTO DOS DADOS
# ============================================================

@st.cache_data(show_spinner="Carregando amostra PRF 2025...")
def carregar_amostra():

    erros = []

    for encoding in ["utf-8-sig", "latin1"]:

        for separador in [",", ";", "\t"]:

            try:

                dados = pd.read_csv(
                    ARQUIVO_AMOSTRA,
                    sep=separador,
                    encoding=encoding,
                    low_memory=False
                )

                dados.columns = dados.columns.str.strip()

                if (
                    len(dados) == N_AMOSTRA
                    and len(dados.columns) == 30
                    and "id" in dados.columns
                ):
                    return dados

                erros.append(
                    f"Separador {repr(separador)}, "
                    f"codificacao {encoding}: "
                    f"{len(dados)} registros e "
                    f"{len(dados.columns)} colunas."
                )

            except (
                pd.errors.ParserError,
                UnicodeError
            ) as erro:

                erros.append(str(erro))

    raise ValueError(
        "Formato do CSV da amostra nao identificado.\n"
        + "\n".join(erros)
    )


@st.cache_data(show_spinner="Carregando referências populacionais...")
def carregar_populacao():

    dados = pd.read_csv(
        ARQUIVO_POPULACAO,
        sep=";",
        encoding="latin1",
        low_memory=False
    )

    dados.columns = dados.columns.str.strip()

    return dados


# ============================================================
# 4. PREPARACAO E VALIDACAO
# ============================================================

try:

    amostra = carregar_amostra()
    populacao = carregar_populacao()

except Exception as erro:

    st.error("Erro ao carregar os arquivos CSV.")
    st.exception(erro)
    st.stop()


if len(amostra) != N_AMOSTRA:

    st.error(
        "A amostra nao possui os 383 registros "
        "esperados."
    )
    st.stop()


if len(populacao) != N_POPULACAO:

    st.error(
        "A populacao nao possui os 72.529 registros "
        "esperados."
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
    if c not in amostra.columns
    or c not in populacao.columns
]

if ausentes:

    st.error(
        "Variaveis ausentes nos arquivos: "
        + ", ".join(ausentes)
    )

    st.stop()


# Verificar IDs sem alterar a amostra.

ids_populacao = pd.to_numeric(
    populacao["id"],
    errors="coerce"
)

ids_amostra = pd.to_numeric(
    amostra["id"],
    errors="coerce"
)

ids_validos = (
    ids_amostra.notna().all()
    and ids_amostra.is_unique
    and ids_amostra.isin(ids_populacao).all()
)

if not ids_validos:

    st.error(
        "Falha na validacao dos identificadores "
        "da amostra."
    )

    st.stop()


# Conversao das variaveis quantitativas.
# Nao altera a selecao dos registros.

variaveis_numericas = [
    "pessoas",
    "mortos",
    "feridos",
    "feridos_leves",
    "feridos_graves",
    "veiculos"
]

for coluna in variaveis_numericas:

    amostra[coluna] = pd.to_numeric(
        amostra[coluna],
        errors="coerce"
    )


amostra["data_inversa"] = pd.to_datetime(
    amostra["data_inversa"],
    errors="coerce"
)


# ============================================================
# 5. ESTILO DOS GRAFICOS
# ============================================================

def aplicar_estilo(fig, altura=430):

    fig.update_layout(
        template="plotly_white",
        height=altura,
        font=dict(
            family="Arial",
            size=13,
            color=COR_TEXTO
        ),
        title_font=dict(
            size=17,
            color=COR_TEXTO
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial"
        ),
        margin=dict(
            l=20,
            r=65,
            t=65,
            b=75
        )
    )

    fig.update_xaxes(
        zeroline=False,
        tickfont=dict(size=12)
    )

    fig.update_yaxes(
        gridcolor=COR_GRADE,
        zeroline=False,
        tickfont=dict(size=12)
    )

    return fig


# ============================================================
# 6. GRAFICOS QUANTITATIVOS
# ============================================================

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
        lambda valor: (
            str(int(valor))
            if valor < limite
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

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=tabela["Categoria"],
            y=tabela["Percentual"],
            marker_color=COR_PRINCIPAL,
            text=[
                numero_br(valor, 1) + "%"
                for valor in tabela["Percentual"]
            ],
            textposition="outside",
            textfont=dict(
                size=13,
                color=COR_TEXTO
            ),
            customdata=tabela["Frequência"],
            hovertemplate=(
                "<b>Quantidade: %{x}</b>"
                "<br>Participação: %{y:.2f}%"
                "<br>Acidentes: %{customdata:,}"
                "<extra></extra>"
            ),
            cliponaxis=False
        )
    )

    maximo = tabela["Percentual"].max()

    fig = aplicar_estilo(fig)

    fig.update_layout(
        title=dict(
            text=titulo,
            x=0,
            xanchor="left"
        ),
        xaxis=dict(
            title="Quantidade por acidente",
            type="category",
            categoryorder="array",
            categoryarray=categorias,
            tickmode="array",
            tickvals=categorias,
            ticktext=categorias,
            tickangle=-35,
            showgrid=False
        ),
        yaxis=dict(
            title="% dos acidentes selecionados",
            range=[0, max(1, maximo * 1.17)],
            gridcolor=COR_GRADE,
            showgrid=True,
            rangemode="tozero"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False}
    )


# ============================================================
# 7. GRAFICOS QUALITATIVOS
# ============================================================

def grafico_qualitativo(
    df,
    coluna,
    titulo,
    top_n=None
):

    tabela_amostra = tabela_frequencia(
        df,
        coluna
    )

    # As categorias de referencia sao escolhidas
    # com base na populacao completa.
    # Os percentuais sao exclusivamente amostrais.

    tabela_populacao = tabela_frequencia(
        populacao,
        coluna
    )

    if top_n is not None:

        categorias_referencia = (
            tabela_populacao["Categoria"]
            .head(top_n)
            .tolist()
        )

    else:

        categorias_referencia = (
            tabela_populacao["Categoria"].tolist()
        )

    tabela = (
        tabela_amostra
        .set_index("Categoria")
        .reindex(
            categorias_referencia,
            fill_value=0
        )
        .reset_index()
    )

    # A ordenacao visual considera a AMOSTRA,
    # mas preserva as categorias populacionais.

    tabela = tabela.sort_values(
        ["Percentual", "Categoria"],
        ascending=[True, True]
    ).copy()

    tabela["Rotulo"] = tabela["Categoria"].apply(
        lambda x: quebrar_rotulo(x, 44)
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=tabela["Percentual"],
            y=tabela["Rotulo"],
            orientation="h",
            marker_color=COR_PRINCIPAL,
            text=[
                numero_br(valor, 2) + "%"
                for valor in tabela["Percentual"]
            ],
            textposition="outside",
            textfont=dict(
                size=13,
                color=COR_TEXTO
            ),
            customdata=[
                [categoria, int(frequencia)]
                for categoria, frequencia in zip(
                    tabela["Categoria"],
                    tabela["Frequência"]
                )
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b>"
                "<br>Participação: %{x:.2f}%"
                "<br>Acidentes: %{customdata[1]:,}"
                "<extra></extra>"
            ),
            cliponaxis=False
        )
    )

    maximo = tabela["Percentual"].max()

    fig = aplicar_estilo(
        fig,
        altura=max(340, len(tabela) * 47 + 115)
    )

    fig.update_layout(
        title=dict(
            text=titulo,
            x=0,
            xanchor="left"
        ),
        xaxis=dict(
            title="% dos acidentes selecionados",
            range=[0, max(1, maximo * 1.16)],
            ticksuffix="%",
            gridcolor=COR_GRADE,
            showgrid=True,
            rangemode="tozero"
        ),
        yaxis=dict(
            title=None,
            type="category",
            showgrid=False
        ),
        margin=dict(
            l=20,
            r=90,
            t=65,
            b=60
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False}
    )


# ============================================================
# 8. CABECALHO
# ============================================================

st.title("📊 Análise descritiva da amostra")

st.markdown(
    """
    **PRF Analytics 2025**

    Análise interativa dos **383 acidentes**
    selecionados por amostragem aleatória simples
    sem reposição.

    **População de origem:** 72.529 ocorrências.

    **Semente utilizada no sorteio original:** 42.

    Os resultados desta página são calculados
    exclusivamente a partir da amostra preservada.
    """
)

st.divider()


# ============================================================
# 9. FILTROS INTERATIVOS
# ============================================================

st.sidebar.title("PRF Analytics 2025")

st.sidebar.markdown("### Filtros da amostra")

st.sidebar.caption(
    "Os filtros podem ser combinados. "
    "Deixe todos sem seleção para visualizar "
    "os 383 acidentes da amostra original."
)


def opcoes_filtro(coluna):

    return sorted(
        amostra[coluna]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )


ufs = st.sidebar.multiselect(
    "Unidade federativa",
    options=opcoes_filtro("uf"),
    placeholder="Selecione uma ou mais opções"
)

fases = st.sidebar.multiselect(
    "Fase do dia",
    options=opcoes_filtro("fase_dia"),
    placeholder="Selecione uma ou mais opções"
)

classificacoes = st.sidebar.multiselect(
    "Classificação do acidente",
    options=opcoes_filtro("classificacao_acidente"),
    placeholder="Selecione uma ou mais opções"
)

tipos = st.sidebar.multiselect(
    "Tipo de acidente",
    options=opcoes_filtro("tipo_acidente"),
    placeholder="Selecione uma ou mais opções"
)


df = amostra.copy()

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
        "Nenhuma ocorrência da amostra corresponde "
        "aos filtros selecionados."
    )

    st.stop()


# ============================================================
# 10. CONTEXTO DA VISUALIZACAO
# ============================================================

if filtros_ativos:

    st.warning(
        "Visualização filtrada: os indicadores "
        "descrevem somente os registros selecionados "
        "da amostra. Não substituem as estimativas "
        "calculadas com os 383 registros originais."
    )

else:

    st.success(
        "Visualização da amostra original completa: "
        "383 ocorrências."
    )


st.caption(
    f"Registros selecionados: "
    f"{numero_br(len(df))} de "
    f"{numero_br(N_AMOSTRA)} "
    f"({numero_br(percentual(len(df), N_AMOSTRA), 2)}%)."
)


# ============================================================
# 11. INDICADORES GERAIS
# ============================================================

st.subheader("Indicadores gerais da amostra")

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
            percentual(
                acidentes_fatais,
                total_acidentes
            ),
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
# 12. ESTATISTICAS DESCRITIVAS DA AMOSTRA
# ============================================================

st.subheader("Medidas estatísticas amostrais")

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
        "Desvio-padrão amostral": (
            serie.std(ddof=1)
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
        "Desvio-padrão amostral": "{:.3f}",
        "Mínimo": "{:.0f}",
        "1º quartil": "{:.2f}",
        "3º quartil": "{:.2f}",
        "Máximo": "{:.0f}"
    }),
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Desvio-padrão amostral calculado com ddof=1, "
    "utilizando o denominador n - 1. "
    "As medidas são recalculadas quando filtros "
    "são aplicados."
)

st.divider()


# ============================================================
# 13. DISTRIBUICOES QUANTITATIVAS
# ============================================================

st.subheader("Distribuições quantitativas")

st.markdown(
    """
    Foram preservados os mesmos agrupamentos
    definidos na análise da população:

    - Pessoas envolvidas: 0 a 9 e 10 ou mais.
    - Vítimas fatais: 0 a 3 e 4 ou mais.
    - Pessoas feridas: 0 a 5 e 6 ou mais.

    As frequências são calculadas exclusivamente
    sobre os acidentes selecionados da amostra.
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
# 14. DISTRIBUICOES QUALITATIVAS
# ============================================================

st.subheader("Distribuições qualitativas")

st.markdown(
    """
    Os gráficos apresentam frequências relativas
    calculadas exclusivamente sobre a amostra.

    Para tipos de acidente, causas e unidades
    federativas, são utilizadas como referência
    as dez categorias mais frequentes da população.

    Essas categorias são ordenadas conforme suas
    frequências observadas na amostra.

    Categorias sem registros amostrais são
    preservadas quando pertencem ao conjunto
    de referência.
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
# 15. EXPLORACAO DE FREQUENCIAS
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
    "A tabela apresenta todas as categorias "
    "observadas no conjunto amostral selecionado, "
    "inclusive aquelas fora dos gráficos Top 10."
)

st.divider()


# ============================================================
# 16. CONSULTA DOS REGISTROS
# ============================================================

with st.expander("Consultar registros da amostra"):

    colunas_consulta = [
        "id",
        "data_inversa",
        "uf",
        "fase_dia",
        "tipo_acidente",
        "pessoas",
        "mortos",
        "feridos"
    ]

    st.dataframe(
        df[colunas_consulta],
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Registros pertencentes à amostra original. "
        "A consulta não realiza novos sorteios."
    )


# ============================================================
# 17. METODOLOGIA E LIMITACOES
# ============================================================

with st.expander("Metodologia e limitações"):

    st.markdown(
        """
        **Fonte dos dados**

        Polícia Rodoviária Federal — PRF 2025.

        **População operacional**

        72.529 acidentes registrados na base
        utilizada no estudo.

        **Unidade amostral**

        Uma ocorrência de acidente.

        **Método**

        Amostragem aleatória simples sem reposição.

        **Tamanho amostral**

        383 ocorrências.

        **Semente pseudoaleatória**

        `random_state=42`

        **Dimensionamento**

        - Confiança: 95%.
        - Margem de erro: 5%.
        - Proporção conservadora: 50%.

        **Reprodutibilidade**

        A amostra foi sorteada anteriormente
        no Google Colab e exportada em arquivo CSV.

        O Streamlit não realiza novos sorteios.

        **Estatísticas amostrais**

        Média, mediana, quartis, mínimo, máximo
        e desvio-padrão amostral (`ddof=1`).

        **Gráficos quantitativos**

        Foram preservados os agrupamentos
        utilizados na análise populacional.

        **Gráficos qualitativos**

        Para variáveis com muitas categorias,
        foram mantidas as categorias de referência
        da população, ordenadas conforme as
        frequências amostrais.

        **Filtros**

        Os filtros permitem explorar subconjuntos
        dos 383 acidentes.

        Os resultados filtrados não substituem
        as estimativas originais utilizadas nos
        intervalos de confiança e testes de hipóteses.

        **Limitações**

        Diferenças entre população e amostra
        são esperadas em razão da variabilidade
        do processo de amostragem.

        A presença de valores extremos pode
        influenciar as médias e os desvios-padrão.

        As frequências não representam
        automaticamente taxas de risco rodoviário.
        """
    )


# ============================================================
# 18. VALIDACAO METODOLOGICA
# ============================================================

with st.expander("Validação da amostra original"):

    st.markdown(
        """
        Esta seção apresenta a validação estrutural
        do arquivo amostral preservado.
        """
    )

    validacoes = pd.DataFrame([
        {
            "Verificação": "Tamanho da amostra",
            "Resultado": len(amostra),
            "Situação": (
                "Aprovado"
                if len(amostra) == 383
                else "Verificar"
            )
        },
        {
            "Verificação": "IDs únicos",
            "Resultado": amostra["id"].is_unique,
            "Situação": (
                "Aprovado"
                if amostra["id"].is_unique
                else "Verificar"
            )
        },
        {
            "Verificação": "IDs na população",
            "Resultado": ids_validos,
            "Situação": (
                "Aprovado"
                if ids_validos
                else "Verificar"
            )
        },
        {
            "Verificação": "Semente documentada",
            "Resultado": SEMENTE_ORIGINAL,
            "Situação": "Documentada no Colab"
        }
    ])

    st.dataframe(
        validacoes,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "A semente foi documentada no sorteio "
        "original. A leitura do CSV não permite, "
        "isoladamente, comprovar qual semente "
        "foi utilizada."
    )


# ============================================================
# 19. RODAPE
# ============================================================

st.divider()

st.caption(
    "PRF Analytics 2025 | "
    "Dashboard 2 — Análise descritiva da amostra | "
    "Probabilidade e Estatística para IA"
)
