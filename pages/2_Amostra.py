
# ============================================================
# PRF ANALYTICS 2025
# DASHBOARD 2 - ANALISE DESCRITIVA DA AMOSTRA
# Versao 1.1 - Boxplots e compatibilidade Streamlit
#
# Populacao operacional: 72.529 acidentes
# Amostra aleatoria simples: 383 acidentes
# Semente documentada do sorteio original: 42
#
# A amostra NAO e sorteada novamente.
# Todos os indicadores amostrais usam o CSV original.
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
COR_DESTAQUE = "#E58A2B"
COR_TEXTO = "#253247"
COR_GRADE = "#E9EDF2"

N_POPULACAO = 72529
N_AMOSTRA = 383
SEMENTE_ORIGINAL = 42

DIRETORIO = Path(__file__).resolve().parents[1]

ARQUIVO_AMOSTRA = DIRETORIO / "amostra_PRF_2025.csv"
ARQUIVO_POPULACAO = DIRETORIO / "datatran2025.csv"


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
    return parte / total * 100 if total else 0.0


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


def tabela_frequencia(dados, coluna):
    serie = normalizar_categoria(dados[coluna])

    resultado = (
        serie.value_counts(dropna=False)
        .rename_axis("Categoria")
        .reset_index(name="Frequência")
    )

    resultado["Percentual"] = (
        resultado["Frequência"] / len(dados) * 100
        if len(dados) else 0.0
    )

    return resultado


# ============================================================
# 3. CARREGAMENTO DOS ARQUIVOS
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
                    f"{encoding}, separador {repr(separador)}: "
                    f"{len(dados)} linhas e "
                    f"{len(dados.columns)} colunas."
                )

            except (
                pd.errors.ParserError,
                UnicodeError
            ) as erro:
                erros.append(str(erro))

    raise ValueError(
        "Formato do CSV da amostra não identificado.\n"
        + "\n".join(erros)
    )


@st.cache_data(
    show_spinner="Carregando referências populacionais..."
)
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
    st.error("A amostra não possui os 383 registros esperados.")
    st.stop()

if len(populacao) != N_POPULACAO:
    st.error(
        "A população não possui os 72.529 registros esperados."
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
    coluna for coluna in colunas_necessarias
    if (
        coluna not in amostra.columns
        or coluna not in populacao.columns
    )
]

if ausentes:
    st.error(
        "Variáveis ausentes nos arquivos: "
        + ", ".join(ausentes)
    )
    st.stop()

ids_populacao = pd.to_numeric(
    populacao["id"],
    errors="coerce"
)

ids_amostra = pd.to_numeric(
    amostra["id"],
    errors="coerce"
)

ids_unicos = bool(ids_amostra.is_unique)
ids_presentes = bool(
    ids_amostra.isin(ids_populacao).all()
)

ids_validos = bool(
    ids_amostra.notna().all()
    and ids_unicos
    and ids_presentes
)

if not ids_validos:
    st.error(
        "Falha na validação dos identificadores da amostra."
    )
    st.stop()

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

# Conferência dos resultados do Colab, sem novo sorteio.
indicadores_conferidos = bool(
    int(amostra["pessoas"].sum()) == 1051
    and int(amostra["feridos"].sum()) == 486
    and int((amostra["mortos"] > 0).sum()) == 24
    and int(
        (
            amostra["fase_dia"]
            .astype(str)
            .str.strip() == "Plena Noite"
        ).sum()
    ) == 121
)

if not indicadores_conferidos:
    st.error(
        "Os indicadores da amostra divergem dos resultados "
        "validados no Google Colab. Confira o CSV original."
    )
    st.stop()


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

def grafico_quantitativo(dados, coluna, titulo, limite):
    serie = pd.to_numeric(
        dados[coluna],
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
        tabela["Frequência"] / len(dados) * 100
        if len(dados) else 0.0
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

    maximo = float(tabela["Percentual"].max())
    fig = aplicar_estilo(fig)

    fig.update_layout(
        title=dict(text=titulo, x=0),
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
        width="stretch",
        config={"displaylogo": False}
    )


# ============================================================
# 7. GRAFICOS QUALITATIVOS
# ============================================================

def grafico_qualitativo(
    dados,
    coluna,
    titulo,
    top_n=None
):
    tabela_amostra = tabela_frequencia(dados, coluna)

    # Categorias de referencia definidas pela populacao.
    # Os percentuais continuam exclusivamente amostrais.
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
            tabela_populacao["Categoria"]
            .tolist()
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

    tabela = tabela.sort_values(
        ["Percentual", "Categoria"],
        ascending=[True, True]
    ).copy()

    tabela["Rotulo"] = tabela["Categoria"].apply(
        lambda valor: quebrar_rotulo(valor, 44)
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

    maximo = float(tabela["Percentual"].max())

    fig = aplicar_estilo(
        fig,
        altura=max(340, len(tabela) * 47 + 115)
    )

    fig.update_layout(
        title=dict(text=titulo, x=0),
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
        width="stretch",
        config={"displaylogo": False}
    )


# ============================================================
# 8. BOXPLOTS - ETAPA 13.1 DO COLAB
# ============================================================

def resumo_boxplot(dados, coluna, descricao):
    valores = pd.to_numeric(
        dados[coluna],
        errors="coerce"
    ).dropna()

    if valores.empty:
        return None

    q1 = float(valores.quantile(0.25))
    mediana = float(valores.median())
    q3 = float(valores.quantile(0.75))

    aiq = q3 - q1

    limite_inferior = q1 - 1.5 * aiq
    limite_superior = q3 + 1.5 * aiq

    regulares = valores[
        (valores >= limite_inferior)
        & (valores <= limite_superior)
    ]

    atipicos = (
        (valores < limite_inferior)
        | (valores > limite_superior)
    )

    return {
        "Variável": descricao,
        "Observações": int(len(valores)),
        "Média": float(valores.mean()),
        "Q1": q1,
        "Mediana": mediana,
        "Q3": q3,
        "AIQ": aiq,
        "Limite inferior": limite_inferior,
        "Limite superior": limite_superior,
        "Bigode inferior": float(regulares.min()),
        "Bigode superior": float(regulares.max()),
        "Mínimo": float(valores.min()),
        "Máximo": float(valores.max()),
        "Atípicos": int(atipicos.sum()),
        "Atípicos (%)": float(atipicos.mean() * 100)
    }


def grafico_boxplot(
    dados,
    coluna,
    descricao,
    exibir_atipicos
):
    valores = pd.to_numeric(
        dados[coluna],
        errors="coerce"
    ).dropna()

    if valores.empty:
        st.info("Não existem dados válidos.")
        return

    resumo = resumo_boxplot(
        dados,
        coluna,
        descricao
    )

    q1 = resumo["Q1"]
    mediana = resumo["Mediana"]
    q3 = resumo["Q3"]

    limite_inferior = resumo["Limite inferior"]
    limite_superior = resumo["Limite superior"]

    fig = go.Figure()

    # Quartis e bigodes explicitamente definidos.
    # Ocultar pontos nao estende os bigodes aos maximos.
    fig.add_trace(
        go.Box(
            x=[0.0],
            q1=[q1],
            median=[mediana],
            q3=[q3],
            lowerfence=[resumo["Bigode inferior"]],
            upperfence=[resumo["Bigode superior"]],
            name=descricao,
            boxpoints=False,
            line=dict(
                color=COR_PRINCIPAL,
                width=2
            ),
            fillcolor="rgba(22,119,200,0.25)",
            boxmean=False
        )
    )

    if exibir_atipicos:
        extremos = valores[
            (valores < limite_inferior)
            | (valores > limite_superior)
        ].to_numpy(dtype=float)

        if len(extremos):
            # Jitter exclusivamente visual.
            # A amostra NAO sofre nenhum novo sorteio.
            gerador = np.random.default_rng(2025)

            posicoes = gerador.uniform(
                -0.17,
                0.17,
                len(extremos)
            )

            fig.add_trace(
                go.Scatter(
                    x=posicoes,
                    y=extremos,
                    mode="markers",
                    marker=dict(
                        color=COR_DESTAQUE,
                        size=7,
                        opacity=0.65
                    ),
                    name="Valores potencialmente atípicos",
                    hovertemplate=(
                        "Quantidade: %{y:.0f}"
                        "<extra></extra>"
                    )
                )
            )

    fig = aplicar_estilo(
        fig,
        altura=480
    )

    fig.update_layout(
        title=dict(
            text=descricao,
            x=0
        ),
        showlegend=False,
        margin=dict(
            l=35,
            r=30,
            t=65,
            b=45
        )
    )

    fig.update_yaxes(
        title="Quantidade por acidente",
        rangemode="tozero"
    )

    fig.update_xaxes(
        title=None,
        range=[-0.40, 0.40],
        showticklabels=False
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={"displaylogo": False}
    )


# ============================================================
# 9. CABECALHO
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
    exclusivamente com os registros da amostra
    original preservada.
    """
)

st.divider()


# ============================================================
# 10. FILTROS INTERATIVOS
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
# 11. CONTEXTO DA VISUALIZACAO
# ============================================================

if filtros_ativos:
    st.warning(
        "Visualização filtrada: os indicadores descrevem "
        "somente os registros selecionados da amostra. "
        "Não substituem as estimativas inferenciais "
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
# 12. INDICADORES GERAIS
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
# 13. MEDIDAS ESTATISTICAS AMOSTRAIS
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
    width="stretch",
    hide_index=True
)

st.caption(
    "Desvio-padrão amostral calculado com ddof=1, "
    "utilizando o denominador n - 1. "
    "As medidas são recalculadas quando os filtros "
    "são aplicados."
)

st.divider()


# ============================================================
# 14. DISTRIBUICOES QUANTITATIVAS
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
# 15. BOXPLOTS - ETAPA 13.1 DO COLAB
# ============================================================

st.subheader(
    "Boxplots — dispersão e valores potencialmente atípicos"
)

st.markdown(
    """
    Os diagramas de caixa complementam os gráficos
    de frequências, mostrando **Q1, mediana, Q3,
    amplitude interquartil (AIQ) e valores extremos**.

    O critério utilizado para identificar valores
    potencialmente atípicos é **1,5 × AIQ**.

    Todos os 383 registros originais permanecem
    incluídos nos cálculos, mesmo quando um ponto
    é classificado como potencialmente atípico.
    """
)

mostrar_pontos_box = st.checkbox(
    "Exibir individualmente os pontos potencialmente atípicos",
    value=False,
    key="amostra_boxplots_exibir_atipicos",
    help=(
        "Exibe os valores fora dos limites de 1,5 × AIQ. "
        "A seleção dos registros não é alterada."
    )
)

box_col1, box_col2 = st.columns(2)

with box_col1:
    grafico_boxplot(
        df,
        "pessoas",
        "Pessoas envolvidas",
        mostrar_pontos_box
    )

with box_col2:
    grafico_boxplot(
        df,
        "feridos",
        "Pessoas feridas",
        mostrar_pontos_box
    )

resumos_box = [
    resumo_boxplot(
        df,
        "pessoas",
        "Pessoas envolvidas"
    ),
    resumo_boxplot(
        df,
        "feridos",
        "Pessoas feridas"
    )
]

resumos_box = [
    item
    for item in resumos_box
    if item is not None
]

if resumos_box:
    tabela_boxplot = pd.DataFrame(resumos_box)

    st.markdown(
        "**Medidas complementares dos boxplots**"
    )

    st.dataframe(
        tabela_boxplot.style.format({
            "Média": "{:.3f}",
            "Q1": "{:.2f}",
            "Mediana": "{:.2f}",
            "Q3": "{:.2f}",
            "AIQ": "{:.2f}",
            "Limite inferior": "{:.2f}",
            "Limite superior": "{:.2f}",
            "Bigode inferior": "{:.0f}",
            "Bigode superior": "{:.0f}",
            "Mínimo": "{:.0f}",
            "Máximo": "{:.0f}",
            "Atípicos (%)": "{:.2f}%"
        }),
        width="stretch",
        hide_index=True
    )

    st.caption(
        "Os bigodes mostram os valores observados "
        "dentro dos limites de 1,5 × AIQ. "
        "Os valores potencialmente atípicos podem "
        "ficar ocultos no gráfico, mas permanecem "
        "incluídos nas estatísticas."
    )

    if any(
        item["AIQ"] == 0
        for item in resumos_box
    ):
        st.info(
            "**Atenção ao AIQ igual a zero:** "
            "a concentração de observações em poucos "
            "valores pode resultar em Q1 = Q3, fazendo "
            "o critério classificar vários registros "
            "legítimos como potencialmente atípicos."
        )

    if not filtros_ativos:
        pessoas_box = resumos_box[0]
        feridos_box = resumos_box[1]

        validacao_box = bool(
            pessoas_box["Observações"] == 383
            and feridos_box["Observações"] == 383
            and pessoas_box["Atípicos"] == 33
            and feridos_box["Atípicos"] == 12
            and pessoas_box["Bigode superior"] == 4
            and feridos_box["Bigode superior"] == 3
            and pessoas_box["Máximo"] == 76
            and feridos_box["Máximo"] == 38
        )

        if validacao_box:
            st.caption(
                "Resultados conferidos com a Etapa 13.1 "
                "do Google Colab: 33 valores potencialmente "
                "atípicos em pessoas envolvidas e 12 em "
                "pessoas feridas; bigodes superiores "
                "iguais a 4 e 3, respectivamente."
            )
        else:
            st.warning(
                "Os boxplots diferem dos resultados "
                "de referência da Etapa 13.1. "
                "Confira a integridade dos dados."
            )

st.caption(
    "Com filtros ativos, os boxplots representam "
    "exclusivamente os registros selecionados. "
    "As estimativas inferenciais do estudo permanecem "
    "baseadas na amostra original de 383 acidentes."
)

st.divider()


# ============================================================
# 16. DISTRIBUICOES QUALITATIVAS
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
# 17. EXPLORACAO DE FREQUENCIAS
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
    width="stretch",
    hide_index=True
)

st.caption(
    "A tabela apresenta todas as categorias observadas "
    "no conjunto amostral selecionado, inclusive "
    "aquelas fora dos gráficos Top 10."
)

st.divider()


# ============================================================
# 18. CONSULTA DOS REGISTROS
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
        width="stretch",
        hide_index=True
    )

    st.caption(
        "Registros pertencentes à amostra original. "
        "A consulta não realiza novos sorteios."
    )


# ============================================================
# 19. METODOLOGIA E LIMITACOES
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

        **Semente pseudoaleatória documentada**

        `random_state=42`

        **Dimensionamento**

        - Confiança: 95%.
        - Margem de erro: 5%.
        - Proporção conservadora: 50%.
        - n₀ = 384,16; com correção para população finita, n = 382,14 → 383.
        - Conferido na calculadora da SurveyMonkey (USP/Bauru indisponível).

        **Reprodutibilidade**

        A amostra foi sorteada anteriormente
        no Google Colab e exportada em CSV.

        O Streamlit apenas lê o arquivo preservado
        e não realiza novos sorteios.

        **Estatísticas amostrais**

        Média, mediana, quartis, mínimo, máximo
        e desvio-padrão amostral (`ddof=1`).

        **Gráficos quantitativos**

        Foram preservados os agrupamentos
        utilizados na análise populacional.

        **Boxplots — Etapa 13.1**

        Os quartis foram calculados por
        interpolação linear.

        A amplitude interquartil foi definida por:

        AIQ = Q3 − Q1.

        Os limites para identificar observações
        potencialmente atípicas foram:

        - Limite inferior: Q1 − 1,5 × AIQ.
        - Limite superior: Q3 + 1,5 × AIQ.

        Os bigodes representam os valores observados
        mais extremos dentro desses limites.

        Os valores fora dos limites não foram
        excluídos dos cálculos.

        **Resultados da amostra completa**

        - Pessoas envolvidas: 33 valores
          potencialmente atípicos.
        - Pessoas feridas: 12 valores
          potencialmente atípicos.
        - Valor máximo de pessoas envolvidas: 76.
        - Valor máximo de pessoas feridas: 38.

        **Gráficos qualitativos**

        Para variáveis com muitas categorias,
        foram mantidas as categorias de referência
        da população, ordenadas conforme as
        frequências amostrais.

        **Filtros**

        Os filtros permitem explorar subconjuntos
        dos 383 acidentes.

        Resultados filtrados não substituem
        as estimativas utilizadas nos intervalos
        de confiança e testes de hipóteses.

        **Limitações**

        Diferenças entre população e amostra
        são esperadas em razão da variabilidade
        do processo de amostragem.

        A presença de valores extremos pode
        influenciar médias e desvios-padrão.

        Os boxplots são ferramentas descritivas.
        Uma observação potencialmente atípica
        não constitui, isoladamente, erro de registro.

        Frequências de acidentes não representam
        automaticamente taxas de risco rodoviário.
        """
    )


# ============================================================
# 20. VALIDACAO METODOLOGICA
# ============================================================

with st.expander("Validação da amostra original"):
    st.markdown(
        """
        Esta seção apresenta a validação estrutural
        do arquivo amostral preservado.
        """
    )

    # Texto evita mistura de inteiros e booleanos
    # na mesma coluna do PyArrow/Streamlit.
    validacoes = pd.DataFrame([
        {
            "Verificação": "Tamanho da amostra",
            "Resultado": str(len(amostra)),
            "Situação": (
                "Aprovado"
                if len(amostra) == N_AMOSTRA
                else "Verificar"
            )
        },
        {
            "Verificação": "IDs únicos",
            "Resultado": (
                "Sim" if ids_unicos else "Não"
            ),
            "Situação": (
                "Aprovado"
                if ids_unicos
                else "Verificar"
            )
        },
        {
            "Verificação": "IDs na população",
            "Resultado": (
                "Sim" if ids_presentes else "Não"
            ),
            "Situação": (
                "Aprovado"
                if ids_presentes
                else "Verificar"
            )
        },
        {
            "Verificação": "Semente documentada",
            "Resultado": str(SEMENTE_ORIGINAL),
            "Situação": "Documentada no Colab"
        },
        {
            "Verificação": "Indicadores do Colab",
            "Resultado": (
                "Conferidos"
                if indicadores_conferidos
                else "Divergentes"
            ),
            "Situação": (
                "Aprovado"
                if indicadores_conferidos
                else "Verificar"
            )
        }
    ])

    st.dataframe(
        validacoes,
        width="stretch",
        hide_index=True
    )

    st.caption(
        "A semente foi documentada no sorteio original. "
        "A leitura do CSV, isoladamente, não comprova "
        "qual semente foi utilizada."
    )


# ============================================================
# 21. RODAPE
# ============================================================

st.divider()

st.caption(
    "PRF Analytics 2025 | "
    "Dashboard 2 — Análise descritiva da amostra | "
    "Probabilidade e Estatística para IA"
)
