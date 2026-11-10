# ============================================================
# PRF ANALYTICS 2025
# DASHBOARD 1 - ANÁLISE DESCRITIVA DA POPULAÇÃO
#
# Versão 1.1 - Melhorias de apresentação
#
# Ajustes:
# - Eixos quantitativos com categorias discretas
# - Fontes e percentuais mais legíveis
# - Filtros e orientações em português
# - Informações interativas aprimoradas
# - Preservação dos cálculos estatísticos
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


# ============================================================
# 4. ESTILO PADRÃO DOS GRÁFICOS
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
        showgrid=False,
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
# 5. TABELA DE FREQUÊNCIAS QUALITATIVAS
# ============================================================

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


# ============================================================
# 6. GRÁFICOS QUALITATIVOS
# ============================================================

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
    ).copy()

    tabela["Rotulo"] = tabela["Categoria"].apply(
        lambda x: quebrar_rotulo(x, 44)
    )

    percentuais = tabela["Percentual"].tolist()
    frequencias = tabela["Frequência"].tolist()
    categorias = tabela["Categoria"].tolist()

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=percentuais,
            y=tabela["Rotulo"],
            orientation="h",
            marker=dict(
                color=COR_PRINCIPAL
            ),
            text=[
                numero_br(v, 2) + "%"
                for v in percentuais
            ],
            textposition="outside",
            textfont=dict(
                size=13,
                color=COR_TEXTO
            ),
            customdata=[
                [categoria, int(freq)]
                for categoria, freq
                in zip(categorias, frequencias)
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b>"
                "<br>Participação: %{x:.2f}%"
                "<br>Ocorrências: %{customdata[1]:,}"
                "<extra></extra>"
            ),
            cliponaxis=False
        )
    )

    maximo = max(percentuais) if percentuais else 0

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
            showgrid=False,
            type="category"
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
# 7. GRÁFICOS QUANTITATIVOS DISCRETOS
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
        str(i)
        for i in range(limite)
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

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=tabela["Categoria"],
            y=tabela["Percentual"],
            marker=dict(
                color=COR_PRINCIPAL
            ),
            text=[
                numero_br(v, 1) + "%"
                for v in tabela["Percentual"]
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
                "<br>Ocorrências: %{customdata:,}"
                "<extra></extra>"
            ),
            cliponaxis=False
        )
    )

    maximo = tabela["Percentual"].max()

    fig = aplicar_estilo(fig, altura=430)

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
# 8. VALIDAÇÃO DOS DADOS
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
        "Colunas não encontradas: "
        + ", ".join(ausentes)
    )

    st.stop()


# ============================================================
# 9. CABEÇALHO
# ============================================================

st.title("📊 Análise descritiva da população")

st.markdown(
    """
    **PRF Analytics 2025**

    Análise interativa dos acidentes registrados
    nas rodovias federais brasileiras em 2025.

    **População operacional:** 72.529 acidentes.

    Utilize os filtros para explorar características
    específicas das ocorrências.
    """
)

st.divider()


# ============================================================
# 10. FILTROS INTERATIVOS
# ============================================================

st.sidebar.title("PRF Analytics 2025")

st.sidebar.markdown("### Filtros da população")

st.sidebar.caption(
    "Os filtros podem ser combinados. "
    "Para visualizar a população completa, "
    "deixe todos sem seleção."
)


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
# 11. CONTEXTO DOS FILTROS
# ============================================================

if filtros_ativos:

    st.warning(
        "Visualização filtrada: os resultados abaixo "
        "descrevem apenas as ocorrências selecionadas. "
        "Não representam os parâmetros da população "
        "completa utilizados nos testes estatísticos."
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
# 12. INDICADORES GERAIS
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
# 13. MEDIDAS ESTATÍSTICAS
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
    "Desvio-padrão calculado com ddof=0. "
    "Com filtros, as medidas descrevem "
    "o subconjunto de registros selecionados."
)

st.divider()


# ============================================================
# 14. DISTRIBUIÇÕES QUANTITATIVAS
# ============================================================

st.subheader("Distribuições quantitativas")

st.markdown(
    """
    Os gráficos apresentam frequências relativas
    com categorias discretas, preservando os
    agrupamentos definidos na análise estatística.

    As ocorrências com valores elevados estão
    incluídas nas categorias finais, sem exclusão.
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
# 15. DISTRIBUIÇÕES QUALITATIVAS
# ============================================================

st.subheader("Distribuições qualitativas")

st.markdown(
    """
    As frequências relativas foram calculadas
    sobre o total de acidentes selecionados.

    Para variáveis com muitas categorias, os gráficos
    apresentam as dez mais frequentes no conjunto
    selecionado. As demais continuam incluídas
    nos cálculos dos percentuais.
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
# 16. TABELA DE FREQUÊNCIAS
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
    "Todas as categorias são apresentadas "
    "na tabela, inclusive aquelas que não "
    "aparecem nos gráficos Top 10."
)

st.divider()


# ============================================================
# 17. CONSULTA DOS REGISTROS
# ============================================================

with st.expander("Consultar registros selecionados"):

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
        df[colunas_consulta].head(1000),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Prévia dos primeiros 1.000 registros "
        "do conjunto atualmente selecionado."
    )


# ============================================================
# 18. METODOLOGIA
# ============================================================

with st.expander("Metodologia e limitações"):

    st.markdown(
        """
        **Fonte dos dados**

        Polícia Rodoviária Federal (PRF) —
        acidentes de trânsito de 2025.

        **População operacional**

        72.529 ocorrências registradas na base
        utilizada no estudo acadêmico.

        **Unidade de análise**

        Uma ocorrência de acidente.

        **Variáveis quantitativas**

        Pessoas envolvidas, mortes, feridos,
        feridos leves, feridos graves e veículos.

        **Estatísticas descritivas**

        Média, mediana, quartis, valores extremos
        e desvio-padrão populacional.

        **Distribuições quantitativas**

        Foram utilizados gráficos de barras,
        respeitando a natureza discreta das
        variáveis de contagem.

        Os agrupamentos finais foram mantidos
        conforme o estudo no Google Colab:

        - Pessoas: 10 ou mais;
        - Mortos: 4 ou mais;
        - Feridos: 6 ou mais.

        **Distribuições qualitativas**

        Frequências absolutas e relativas,
        utilizando gráficos de barras horizontais.

        **Interatividade**

        Os filtros possibilitam explorar
        subconjuntos da população.

        Resultados filtrados não substituem
        os parâmetros populacionais utilizados
        nos procedimentos inferenciais.

        **Limitações**

        As frequências dos registros não
        representam automaticamente taxas de
        risco rodoviário.

        Comparações entre estados e períodos
        não consideram necessariamente
        diferenças de exposição ao tráfego.

        As causas apresentadas correspondem
        às classificações registradas pela PRF.
        """
    )


# ============================================================
# 19. RODAPÉ
# ============================================================

st.divider()

st.caption(
    "PRF Analytics 2025 | "
    "Dashboard 1 — Análise descritiva da população | "
    "Probabilidade e Estatística para IA"
)
