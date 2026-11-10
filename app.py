
# ============================================================
# PRF ANALYTICS 2025
# PAGINA INICIAL - VERSAO 2.0
# Probabilidade e Estatistica para IA
#
# Preserva:
# - Carregamento robusto dos CSVs
# - Cache
# - Indicadores
# - Auditoria das bases
# - Previa de 10 registros
# - Informacoes metodologicas
#
# Adiciona:
# - Apresentacao do projeto
# - Links para os tres dashboards
# - Resumo dos resultados inferenciais
# - Conferencia dos valores amostrais
#
# Nenhum sorteio e realizado nesta pagina.
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st


# ============================================================
# 1. CONFIGURACAO
# ============================================================

st.set_page_config(
    page_title="PRF Analytics 2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

RAIZ = Path(__file__).resolve().parent

ARQUIVO_POPULACAO = RAIZ / "datatran2025.csv"
ARQUIVO_AMOSTRA = RAIZ / "amostra_PRF_2025.csv"

N_ESPERADO = 72529
AMOSTRA_ESPERADA = 383
COLUNAS_ESPERADAS = 30

URL_PRF = (
    "https://www.gov.br/prf/pt-br/"
    "acesso-a-informacao/dados-abertos/"
    "dados-abertos-da-prf"
)

URL_GITHUB = (
    "https://github.com/mendes31/"
    "prf-analytics-2025"
)


# ============================================================
# 2. FUNCOES DE FORMATACAO
# ============================================================

def numero_br(valor, casas=0):
    texto = f"{valor:,.{casas}f}"
    return (
        texto.replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )


# ============================================================
# 3. CARREGAMENTO
# ============================================================

@st.cache_data(show_spinner="Carregando população...")
def carregar_populacao():

    dados = pd.read_csv(
        ARQUIVO_POPULACAO,
        sep=";",
        encoding="latin1",
        low_memory=False
    )

    dados.columns = dados.columns.str.strip()

    return dados


@st.cache_data(show_spinner="Carregando amostra...")
def carregar_amostra():

    erros = []

    for codificacao in ["utf-8-sig", "latin1"]:

        for separador in [",", ";", "\t"]:

            try:

                dados = pd.read_csv(
                    ARQUIVO_AMOSTRA,
                    sep=separador,
                    encoding=codificacao,
                    low_memory=False
                )

                dados.columns = dados.columns.str.strip()

                if (
                    len(dados) == AMOSTRA_ESPERADA
                    and len(dados.columns) == COLUNAS_ESPERADAS
                    and "id" in dados.columns
                ):
                    return dados

                erros.append(
                    f"{codificacao}, {repr(separador)}: "
                    f"{dados.shape}"
                )

            except (
                UnicodeError,
                pd.errors.ParserError
            ) as erro:

                erros.append(str(erro))

    raise ValueError(
        "Formato do CSV da amostra não reconhecido. "
        + "; ".join(erros)
    )


# ============================================================
# 4. VALIDACAO
# ============================================================

def validar_bases(populacao, amostra):

    N = len(populacao)
    n = len(amostra)

    colunas_iguais = (
        set(populacao.columns)
        == set(amostra.columns)
    )

    possuem_id = (
        "id" in populacao.columns
        and "id" in amostra.columns
    )

    ids_validos = False
    ids_unicos_pop = False
    ids_unicos_amo = False
    ids_contidos = False
    valores_iguais = False

    variaveis = [
        "pessoas",
        "mortos",
        "feridos",
        "fase_dia"
    ]

    if possuem_id:

        ids_pop = pd.to_numeric(
            populacao["id"],
            errors="coerce"
        )

        ids_amo = pd.to_numeric(
            amostra["id"],
            errors="coerce"
        )

        ids_validos = bool(
            ids_pop.notna().all()
            and ids_amo.notna().all()
        )

        ids_unicos_pop = bool(
            ids_pop.is_unique
        )

        ids_unicos_amo = bool(
            ids_amo.is_unique
        )

        ids_contidos = bool(
            ids_amo.isin(ids_pop).all()
        )

        if (
            ids_validos
            and ids_unicos_pop
            and ids_unicos_amo
            and ids_contidos
            and all(
                coluna in populacao.columns
                and coluna in amostra.columns
                for coluna in variaveis
            )
        ):

            pop = populacao[
                ["id"] + variaveis
            ].copy()

            amo = amostra[
                ["id"] + variaveis
            ].copy()

            pop["id"] = ids_pop
            amo["id"] = ids_amo

            for coluna in [
                "pessoas",
                "mortos",
                "feridos"
            ]:

                pop[coluna] = pd.to_numeric(
                    pop[coluna],
                    errors="coerce"
                )

                amo[coluna] = pd.to_numeric(
                    amo[coluna],
                    errors="coerce"
                )

            for dados in [pop, amo]:

                dados["fase_dia"] = (
                    dados["fase_dia"]
                    .astype("string")
                    .str.strip()
                )

            conferencia = amo.merge(
                pop,
                on="id",
                how="left",
                suffixes=("_a", "_p"),
                validate="one_to_one"
            )

            verificacoes_valores = []

            for coluna in variaveis:

                iguais = (
                    conferencia[f"{coluna}_a"]
                    .eq(conferencia[f"{coluna}_p"])
                    .fillna(False)
                    .all()
                )

                verificacoes_valores.append(
                    bool(iguais)
                )

            valores_iguais = all(
                verificacoes_valores
            )

    verificacoes = [
        ("Registros da população", N == N_ESPERADO),
        ("Registros da amostra", n == AMOSTRA_ESPERADA),
        (
            "Variáveis da população",
            len(populacao.columns) == COLUNAS_ESPERADAS
        ),
        (
            "Variáveis da amostra",
            len(amostra.columns) == COLUNAS_ESPERADAS
        ),
        ("Mesmas colunas", colunas_iguais),
        ("Coluna ID presente", possuem_id),
        ("IDs válidos", ids_validos),
        ("IDs únicos na população", ids_unicos_pop),
        ("IDs únicos na amostra", ids_unicos_amo),
        ("IDs da amostra na população", ids_contidos),
        (
            "Valores das variáveis principais",
            valores_iguais
        )
    ]

    tabela = pd.DataFrame([
        {
            "Verificação": nome,
            "Situação": (
                "✅ Aprovado"
                if aprovado
                else "❌ Verificar"
            )
        }
        for nome, aprovado in verificacoes
    ])

    return tabela, all(
        aprovado
        for _, aprovado in verificacoes
    )


# ============================================================
# 5. INICIALIZACAO DOS DADOS
# ============================================================

try:

    populacao = carregar_populacao()
    amostra = carregar_amostra()

    auditoria, aprovado = validar_bases(
        populacao,
        amostra
    )

except Exception as erro:

    st.error(
        "Não foi possível carregar ou validar "
        "as bases de dados."
    )

    st.exception(erro)
    st.stop()


N = len(populacao)
n = len(amostra)

fracao = n / N * 100

if not aprovado:

    st.error(
        "Uma ou mais verificações de integridade "
        "não foram aprovadas. "
        "Revise a auditoria antes de utilizar "
        "os dashboards."
    )

    st.dataframe(
        auditoria,
        hide_index=True,
        use_container_width=True
    )

    st.stop()


# ============================================================
# 6. CABECALHO
# ============================================================

st.title("📊 PRF Analytics 2025")

st.subheader(
    "Probabilidade e Estatística para IA"
)

st.markdown(
    """
    **Análise estatística dos acidentes registrados
    nas rodovias federais brasileiras em 2025.**

    Este projeto investiga como uma amostra
    probabilística pode ser utilizada para
    estimar características de uma população
    conhecida, comparando resultados descritivos
    e inferenciais.

    **Fonte:** Polícia Rodoviária Federal (PRF)  
    **Unidade amostral:** uma ocorrência de acidente  
    **Amostragem:** aleatória simples sem reposição  
    **Semente original:** `random_state=42`
    """
)

st.divider()


# ============================================================
# 7. INDICADORES GERAIS
# ============================================================

st.header("Visão geral do estudo")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "População",
    numero_br(N)
)

c2.metric(
    "Amostra",
    numero_br(n)
)

c3.metric(
    "Fração amostral",
    numero_br(fracao, 4) + "%"
)

c4.metric(
    "Nível de confiança",
    "95%"
)

st.caption(
    "Os valores correspondem à população operacional "
    "e à amostra preservada utilizadas no estudo."
)

st.divider()


# ============================================================
# 8. ACESSO AOS DASHBOARDS
# ============================================================

st.header("Dashboards interativos")

st.markdown(
    """
    Explore as características dos acidentes,
    compare população e amostra e consulte
    os resultados dos procedimentos inferenciais.
    """
)

col1, col2, col3 = st.columns(3)

with col1:

    with st.container(border=True):

        st.subheader("📊 Dashboard 1")

        st.markdown(
            "**Análise descritiva da população**"
        )

        st.write(
            "Indicadores gerais, medidas estatísticas, "
            "distribuições quantitativas e qualitativas "
            "e filtros interativos dos 72.529 acidentes."
        )

        st.page_link(
            "pages/1_Populacao.py",
            label="Abrir análise da população",
            icon="📊",
            use_container_width=True
        )

with col2:

    with st.container(border=True):

        st.subheader("📋 Dashboard 2")

        st.markdown(
            "**Análise descritiva da amostra**"
        )

        st.write(
            "Distribuições, frequências, medidas "
            "descritivas e filtros interativos "
            "dos 383 acidentes selecionados."
        )

        st.page_link(
            "pages/2_Amostra.py",
            label="Abrir análise da amostra",
            icon="📋",
            use_container_width=True
        )

with col3:

    with st.container(border=True):

        st.subheader("📈 Dashboard 3")

        st.markdown(
            "**Inferência estatística e comparação**"
        )

        st.write(
            "Intervalos de confiança, testes de "
            "hipóteses, comparação dos parâmetros, "
            "sensibilidade e ajuste de Holm."
        )

        st.page_link(
            "pages/3_Inferencia.py",
            label="Abrir inferência estatística",
            icon="📈",
            use_container_width=True
        )

st.divider()


# ============================================================
# 9. METODOLOGIA RESUMIDA
# ============================================================

st.header("Metodologia")

m1, m2 = st.columns(2)

with m1:

    st.markdown("#### População e amostragem")

    st.write(
        "A população operacional corresponde "
        "aos 72.529 registros presentes na versão "
        "da base PRF 2025 utilizada no estudo."
    )

    st.write(
        "Foi selecionada uma amostra de "
        "383 ocorrências por amostragem aleatória "
        "simples sem reposição."
    )

    st.write(
        "O dimensionamento considerou confiança "
        "de 95%, margem de erro de referência de "
        "5% e proporção conservadora de 50%."
    )

with m2:

    st.markdown("#### Inferência estatística")

    st.write(
        "Foram utilizados dois parâmetros de "
        "proporção e dois parâmetros de média."
    )

    st.write(
        "Os intervalos de confiança foram "
        "construídos com nível de 95%, "
        "incluindo correção para população finita."
    )

    st.write(
        "Foram realizados quatro testes de "
        "hipóteses bilaterais, com significância "
        "de 5%, além de análises complementares."
    )

st.divider()


# ============================================================
# 10. RESULTADOS CONSOLIDADOS
# ============================================================

st.header("Principais resultados")

r1, r2, r3, r4 = st.columns(4)

r1.metric("Parâmetros analisados", "4")
r2.metric("ICs com cobertura", "4 de 4")
r3.metric("Rejeições individuais", "2")
r4.metric("Rejeições após Holm", "0")

st.caption(
    "Síntese dos resultados previamente calculados "
    "no Google Colab e reproduzidos no Dashboard 3."
)

st.markdown(
    """
    **Intervalos de confiança:** os quatro
    intervalos bilaterais de 95% incluíram
    seus respectivos parâmetros populacionais.

    **Testes individuais:** as hipóteses P1
    (acidentes com pelo menos uma morte) e
    M2 (média de pessoas feridas) foram rejeitadas
    ao nível de significância de 5%.

    **Comparações múltiplas:** após o ajuste de
    Holm, nenhuma das quatro hipóteses nulas
    permaneceu rejeitada.

    **Análise de sensibilidade:** a ocorrência
    extrema de ID 716289 influenciou
    especialmente o teste sobre a média
    de pessoas envolvidas.
    """
)

st.info(
    "Os procedimentos inferenciais foram realizados "
    "exclusivamente com a amostra original. "
    "Os parâmetros populacionais foram utilizados "
    "posteriormente para comparação."
)

st.divider()


# ============================================================
# 11. FONTES E LINKS
# ============================================================

st.header("Fontes e reprodutibilidade")

st.markdown(
    """
    A base utilizada foi disponibilizada pela
    Polícia Rodoviária Federal no portal de
    Dados Abertos, na modalidade de acidentes
    agrupados por ocorrência.

    Os dois arquivos empregados no estudo são:

    - `datatran2025.csv` — população operacional;
    - `amostra_PRF_2025.csv` — amostra preservada.

    O código-fonte dos dashboards está
    disponível em um repositório público.
    """
)

fonte, repositorio = st.columns(2)

with fonte:

    st.link_button(
        "Consultar fonte oficial PRF",
        URL_PRF,
        use_container_width=True
    )

with repositorio:

    st.link_button(
        "Consultar repositório GitHub",
        URL_GITHUB,
        use_container_width=True
    )

st.divider()


# ============================================================
# 12. AUDITORIA DAS BASES
# ============================================================

with st.expander(
    "🔎 Validação computacional das bases"
):

    st.write(
        "As verificações confirmam o tamanho "
        "dos conjuntos, as colunas, os IDs e "
        "a correspondência das quatro variáveis "
        "principais entre amostra e população."
    )

    st.dataframe(
        auditoria,
        use_container_width=True,
        hide_index=True
    )

    st.success(
        "As verificações de integridade foram aprovadas."
    )

    st.caption(
        "A conferência de valores considera pessoas, "
        "mortos, feridos e fase do dia. "
        "Não constitui comparação integral de "
        "todas as 30 colunas."
    )


# ============================================================
# 13. PREVIA DOS DADOS
# ============================================================

with st.expander("📋 Prévia dos registros utilizados"):

    aba_populacao, aba_amostra = st.tabs(
        ["População", "Amostra"]
    )

    with aba_populacao:

        st.markdown(
            "**Primeiros dez registros da população**"
        )

        st.dataframe(
            populacao.head(10),
            hide_index=True,
            use_container_width=True
        )

    with aba_amostra:

        st.markdown(
            "**Primeiros dez registros da amostra**"
        )

        st.dataframe(
            amostra.head(10),
            hide_index=True,
            use_container_width=True
        )


# ============================================================
# 14. INFORMACOES METODOLOGICAS DETALHADAS
# ============================================================

with st.expander(
    "📚 Informações metodológicas e limitações"
):

    st.markdown(
        """
        **Definição da população**

        A população operacional compreende
        os acidentes disponíveis na versão
        utilizada da base PRF 2025.

        **Unidade amostral**

        Uma ocorrência de acidente.

        **Procedimento de seleção**

        Amostragem aleatória simples sem reposição,
        com semente pseudoaleatória fixa de 42.

        **Tamanho amostral**

        383 ocorrências.

        **Parâmetros do dimensionamento**

        - Confiança de 95%;
        - Margem de erro de referência de 5%;
        - Proporção conservadora de 50%.

        **Intervalos de confiança**

        Método de Wald para proporções
        e distribuição t de Student para médias,
        com correção para população finita.

        **Testes de hipóteses**

        Testes Z para proporções e testes t
        para médias, com alternativas bilaterais.

        **Análise complementar**

        Diagnóstico de influência de observações
        extremas e ajuste de Holm para
        comparações múltiplas.

        **Reprodutibilidade**

        A amostra foi selecionada previamente
        no Google Colab e preservada no arquivo CSV.

        O aplicativo não realiza novos sorteios.

        **Limitações**

        As conclusões referem-se à população
        operacional estudada e não são
        automaticamente generalizáveis a
        acidentes não registrados ou outros anos.

        A assimetria e os valores extremos
        exigem cautela na interpretação
        das inferências para médias.

        Os valores das hipóteses nulas são
        referências exploratórias, não
        metas oficiais da PRF.
        """
    )


# ============================================================
# 15. RODAPE
# ============================================================

st.divider()

st.caption(
    "PRF Analytics 2025 | "
    "Probabilidade e Estatística para IA | "
    "Fonte: Polícia Rodoviária Federal | "
    "Python • Streamlit • Plotly"
)
