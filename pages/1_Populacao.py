
# PRF ANALYTICS 2025
# DASHBOARD 1 - ANALISE DESCRITIVA DA POPULACAO
# Versao 1.2.3 - Boxplots com bigodes corrigidos

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


def quebrar_rotulo(texto, largura=42):
    partes = wrap(
        str(texto),
        width=largura,
        break_long_words=False,
        break_on_hyphens=False
    )
    return "<br>".join(partes) if partes else str(texto)


# ============================================================
# 3. CARREGAMENTO DOS DADOS
# ============================================================

@st.cache_data(show_spinner="Carregando dados da PRF...")
def carregar_dados():
    dados = pd.read_csv(
        ARQUIVO,
        sep=";",
        encoding="latin1",
        low_memory=False
    )

    dados.columns = dados.columns.str.strip()

    for coluna in [
        "pessoas", "mortos", "feridos",
        "feridos_leves", "feridos_graves", "veiculos"
    ]:
        if coluna in dados.columns:
            dados[coluna] = pd.to_numeric(
                dados[coluna],
                errors="coerce"
            )

    dados["data_inversa"] = pd.to_datetime(
        dados["data_inversa"],
        errors="coerce"
    )

    return dados


# ============================================================
# 4. ESTILO DOS GRAFICOS
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
            font_size=14
        ),
        margin=dict(l=20, r=65, t=65, b=75)
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
# 5. FREQUENCIAS QUALITATIVAS
# ============================================================

def tabela_frequencia(dados, coluna):
    serie = (
        dados[coluna]
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

    if len(dados):
        resultado["Percentual"] = (
            resultado["Frequência"] / len(dados) * 100
        )
    else:
        resultado["Percentual"] = 0.0

    return resultado


def grafico_qualitativo(dados, coluna, titulo, top_n=None):
    tabela = tabela_frequencia(dados, coluna)

    if top_n is not None:
        tabela = tabela.head(top_n).copy()

    tabela = tabela.sort_values(
        ["Percentual", "Categoria"],
        ascending=[True, True]
    ).copy()

    tabela["Rotulo"] = tabela["Categoria"].apply(
        lambda valor: quebrar_rotulo(valor, 44)
    )

    percentuais = tabela["Percentual"].tolist()

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=percentuais,
            y=tabela["Rotulo"],
            orientation="h",
            marker_color=COR_PRINCIPAL,
            text=[
                numero_br(valor, 2) + "%"
                for valor in percentuais
            ],
            textposition="outside",
            textfont=dict(
                size=13,
                color=COR_TEXTO
            ),
            customdata=[
                [categoria, int(freq)]
                for categoria, freq in zip(
                    tabela["Categoria"],
                    tabela["Frequência"]
                )
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
        title=dict(text=titulo, x=0),
        xaxis=dict(
            title="% dos acidentes selecionados",
            range=[0, max(1, maximo * 1.16)],
            ticksuffix="%",
            showgrid=True,
            gridcolor=COR_GRADE
        ),
        yaxis=dict(
            title=None,
            type="category",
            showgrid=False
        ),
        margin=dict(l=20, r=90, t=65, b=60)
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={"displaylogo": False}
    )


# ============================================================
# 6. DISTRIBUICOES QUANTITATIVAS
# ============================================================

def grafico_quantitativo(dados, coluna, titulo, limite):
    serie = pd.to_numeric(
        dados[coluna], errors="coerce"
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
                "<br>Ocorrências: %{customdata:,}"
                "<extra></extra>"
            ),
            cliponaxis=False
        )
    )

    maximo = tabela["Percentual"].max()
    fig = aplicar_estilo(fig)

    fig.update_layout(
        title=dict(text=titulo, x=0),
        xaxis=dict(
            title="Quantidade por acidente",
            type="category",
            categoryorder="array",
            categoryarray=categorias,
            tickangle=-35
        ),
        yaxis=dict(
            title="% dos acidentes selecionados",
            range=[0, max(1, maximo * 1.17)],
            showgrid=True,
            gridcolor=COR_GRADE
        )
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={"displaylogo": False}
    )


# ============================================================
# 7. BOXPLOTS - ETAPA 13.1
# ============================================================

def resumo_boxplot(dados, coluna, descricao):
    valores = pd.to_numeric(
        dados[coluna], errors="coerce"
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


def grafico_boxplot(dados, coluna, descricao, exibir_atipicos):
    valores = pd.to_numeric(
        dados[coluna], errors="coerce"
    ).dropna()

    if valores.empty:
        st.info("Não existem dados válidos.")
        return

    q1 = float(valores.quantile(0.25))
    mediana = float(valores.median())
    q3 = float(valores.quantile(0.75))
    aiq = q3 - q1

    li = q1 - 1.5 * aiq
    ls = q3 + 1.5 * aiq

    regulares = valores[
        (valores >= li) & (valores <= ls)
    ]

    bigode_inferior = float(regulares.min())
    bigode_superior = float(regulares.max())

    fig = go.Figure()

    # Quartis e bigodes explícitos.
    # Assim, ocultar pontos não altera os bigodes.
    fig.add_trace(
        go.Box(
            x=[0.0],
            q1=[q1],
            median=[mediana],
            q3=[q3],
            lowerfence=[bigode_inferior],
            upperfence=[bigode_superior],
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
            (valores < li) | (valores > ls)
        ].to_numpy(dtype=float)

        if len(extremos):
            # Deslocamento apenas visual.
            # Não altera a amostra ou a população.
            gerador = np.random.default_rng(2025)
            posicoes = gerador.uniform(
                -0.17, 0.17, len(extremos)
            )

            fig.add_trace(
                go.Scattergl(
                    x=posicoes,
                    y=extremos,
                    mode="markers",
                    marker=dict(
                        color=COR_DESTAQUE,
                        size=4,
                        opacity=0.45
                    ),
                    name="Valores potencialmente atípicos",
                    hovertemplate=(
                        "Quantidade: %{y:.0f}"
                        "<extra></extra>"
                    )
                )
            )

    fig = aplicar_estilo(fig, altura=480)

    fig.update_layout(
        title=dict(text=descricao, x=0),
        showlegend=False,
        margin=dict(l=35, r=30, t=65, b=45)
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
# 8. VALIDACAO DOS DADOS
# ============================================================

try:
    populacao = carregar_dados()
except Exception as erro:
    st.error("Erro ao carregar a população.")
    st.exception(erro)
    st.stop()

if len(populacao) != TOTAL_ESPERADO:
    st.error(
        "A base não corresponde aos 72.529 registros "
        "validados no Google Colab."
    )
    st.stop()

colunas_necessarias = [
    "id", "data_inversa", "uf", "fase_dia",
    "classificacao_acidente", "tipo_acidente",
    "causa_acidente", "condicao_metereologica",
    "pessoas", "mortos", "feridos",
    "feridos_leves", "feridos_graves", "veiculos"
]

ausentes = [
    coluna for coluna in colunas_necessarias
    if coluna not in populacao.columns
]

if ausentes:
    st.error(
        "Colunas não encontradas: " + ", ".join(ausentes)
    )
    st.stop()


# ============================================================
# 9. CABECALHO
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
        df["classificacao_acidente"].isin(classificacoes)
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
        "Nenhuma ocorrência corresponde aos filtros selecionados."
    )
    st.stop()


# ============================================================
# 11. CONTEXTO DOS FILTROS
# ============================================================

if filtros_ativos:
    st.warning(
        "Visualização filtrada: os resultados descrevem "
        "apenas as ocorrências selecionadas. Não substituem "
        "os parâmetros da população completa utilizados "
        "nos testes estatísticos."
    )
else:
    st.success(
        "Visualização da população completa: 72.529 ocorrências."
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
    st.metric("Acidentes", numero_br(total_acidentes))

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
# 13. MEDIDAS ESTATISTICAS
# ============================================================

st.subheader("Medidas estatísticas")

variaveis = [
    "pessoas", "mortos", "feridos",
    "feridos_leves", "feridos_graves", "veiculos"
]

linhas = []

for coluna in variaveis:
    serie = df[coluna].dropna()

    linhas.append({
        "Variável": coluna,
        "Observações válidas": len(serie),
        "Média": serie.mean(),
        "Mediana": serie.median(),
        "Desvio-padrão populacional": serie.std(ddof=0),
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
    width="stretch",
    hide_index=True
)

st.caption(
    "Desvio-padrão calculado com ddof=0. "
    "Com filtros, as medidas descrevem o subconjunto selecionado."
)

st.divider()


# ============================================================
# 14. DISTRIBUICOES QUANTITATIVAS
# ============================================================

st.subheader("Distribuições quantitativas")

st.markdown(
    """
    Os gráficos apresentam frequências relativas
    com categorias discretas.

    As ocorrências com valores elevados estão
    incluídas nas categorias finais, sem exclusão.
    """
)

q1, q2 = st.columns(2)

with q1:
    grafico_quantitativo(
        df, "pessoas", "Pessoas envolvidas", limite=10
    )

with q2:
    grafico_quantitativo(
        df, "mortos", "Vítimas fatais", limite=4
    )

grafico_quantitativo(
    df, "feridos", "Pessoas feridas", limite=6
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
    Os diagramas de caixa complementam os gráficos de
    frequências, mostrando **Q1, mediana, Q3, amplitude
    interquartil (AIQ) e valores extremos**.

    O critério utilizado é **1,5 × AIQ**.

    Todos os registros permanecem incluídos nos cálculos,
    independentemente de sua classificação como atípicos.
    """
)

mostrar_pontos_box = st.checkbox(
    "Exibir individualmente os pontos potencialmente atípicos",
    value=False,
    key="pop_boxplots_exibir_atipicos",
    help=(
        "A população possui milhares de pontos potencialmente "
        "atípicos. A exibição individual pode reduzir o desempenho."
    )
)

box_col1, box_col2 = st.columns(2)

with box_col1:
    grafico_boxplot(
        df, "pessoas", "Pessoas envolvidas",
        mostrar_pontos_box
    )

with box_col2:
    grafico_boxplot(
        df, "feridos", "Pessoas feridas",
        mostrar_pontos_box
    )

resumos_box = [
    resumo_boxplot(
        df, "pessoas", "Pessoas envolvidas"
    ),
    resumo_boxplot(
        df, "feridos", "Pessoas feridas"
    )
]

resumos_box = [
    item for item in resumos_box if item is not None
]

if resumos_box:
    tabela_boxplot = pd.DataFrame(resumos_box)

    st.markdown("**Medidas complementares dos boxplots**")

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
        "Os bigodes mostram os valores observados dentro "
        "dos limites de 1,5 × AIQ. Valores potencialmente "
        "atípicos podem ser ocultados visualmente, mas "
        "continuam incluídos nas estatísticas."
    )

    if any(item["AIQ"] == 0 for item in resumos_box):
        st.info(
            "**Atenção ao AIQ igual a zero:** quando Q1 e Q3 "
            "coincidem, o critério de 1,5 × AIQ pode "
            "classificar muitos registros legítimos "
            "como potencialmente atípicos."
        )

    if not filtros_ativos:
        observados_box = {
            "pessoas": resumos_box[0]["Atípicos"],
            "feridos": resumos_box[1]["Atípicos"]
        }

        referencia_box = {
            "pessoas": 6360,
            "feridos": 31009
        }

        if observados_box == referencia_box:
            st.caption(
                "Resultados conferidos com a Etapa 13.1 "
                "do Colab: 6.360 valores potencialmente "
                "atípicos para pessoas envolvidas e "
                "31.009 para pessoas feridas."
            )
        else:
            st.warning(
                "Os resultados diferem da conferência "
                "original da Etapa 13.1."
            )

st.caption(
    "Quando há filtros, os boxplots descrevem "
    "somente o conjunto selecionado."
)

st.divider()


# ============================================================
# 16. DISTRIBUICOES QUALITATIVAS
# ============================================================

st.subheader("Distribuições qualitativas")

st.markdown(
    """
    As frequências relativas foram calculadas
    sobre o total de acidentes selecionados.

    Variáveis com muitas categorias são exibidas
    pelos dez grupos mais frequentes.

    As demais categorias permanecem incluídas
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
# 17. TABELA DE FREQUENCIAS
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
    "Todas as categorias são apresentadas na tabela, "
    "inclusive aquelas que não aparecem nos gráficos Top 10."
)

st.divider()


# ============================================================
# 18. CONSULTA DOS REGISTROS
# ============================================================

with st.expander("Consultar registros selecionados"):
    colunas_consulta = [
        "id", "data_inversa", "uf", "fase_dia",
        "tipo_acidente", "pessoas", "mortos", "feridos"
    ]

    st.dataframe(
        df[colunas_consulta].head(1000),
        width="stretch",
        hide_index=True
    )

    st.caption(
        "Prévia dos primeiros 1.000 registros "
        "do conjunto atualmente selecionado."
    )


# ============================================================
# 19. METODOLOGIA E LIMITACOES
# ============================================================

with st.expander("Metodologia e limitações"):
    st.markdown(
        """
        **Fonte dos dados**

        Polícia Rodoviária Federal (PRF) —
        acidentes de trânsito registrados em 2025.

        **População operacional**

        72.529 ocorrências, sendo cada acidente
        uma unidade de análise.

        **Variáveis quantitativas**

        Pessoas, mortes, feridos, feridos leves,
        feridos graves e veículos.

        **Medidas descritivas**

        Média, mediana, quartis, valores extremos
        e desvio-padrão populacional.

        **Distribuições quantitativas**

        Gráficos de frequências relativas para
        variáveis discretas, incluindo agrupamentos
        finais para valores elevados.

        **Boxplots — Etapa 13.1**

        Diagramas de caixa com quartis calculados
        por interpolação linear.

        Limites para identificação de valores
        potencialmente atípicos:

        - Limite inferior: Q1 − 1,5 × AIQ.
        - Limite superior: Q3 + 1,5 × AIQ.

        Os bigodes mostram os valores observados
        mais extremos dentro desses limites.

        A identificação de valores atípicos
        é descritiva e não implica erro de registro.

        **Variável feridos**

        A população apresenta AIQ igual a zero,
        exigindo cautela na interpretação do
        percentual de valores atípicos.

        **Distribuições qualitativas**

        Frequências absolutas e relativas
        apresentadas em barras horizontais.

        **Filtros interativos**

        Permitem explorar subconjuntos da população.

        Resultados filtrados não substituem os
        parâmetros populacionais empregados
        nos procedimentos inferenciais.

        **Limitações**

        Frequências de acidentes não representam
        automaticamente taxas de risco rodoviário.

        Comparações entre estados não consideram
        necessariamente diferentes exposições
        ao tráfego.

        As causas dos acidentes correspondem
        às categorias registradas pela PRF.
        """
    )


# ============================================================
# 20. RODAPE
# ============================================================

st.divider()

st.caption(
    "PRF Analytics 2025 | "
    "Dashboard 1 — Análise descritiva da população | "
    "Probabilidade e Estatística para IA"
)
