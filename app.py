
# ============================================================
# PRF ANALYTICS 2025
# Probabilidade e Estatística para IA
#
# Aplicação Streamlit - Versão 1.2
# Validação das bases populacional e amostral
# ============================================================

import streamlit as st
import pandas as pd

from pathlib import Path


# ============================================================
# 1. CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="PRF Analytics 2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. IDENTIFICAÇÃO DO PROJETO
# ============================================================

st.title("PRF Analytics 2025")

st.subheader("Probabilidade e Estatística para IA")

st.markdown(
    """
    Análise estatística dos acidentes registrados
    nas rodovias federais brasileiras em 2025.

    **Fonte:** Polícia Rodoviária Federal (PRF)

    **População operacional:** 72.529 ocorrências

    **Amostra:** 383 ocorrências selecionadas por
    amostragem aleatória simples (AAS) sem reposição.

    **Semente pseudoaleatória:** 42

    **Nível de confiança:** 95%
    """
)

st.divider()


# ============================================================
# 3. LOCALIZAÇÃO DOS ARQUIVOS
# ============================================================

DIRETORIO = Path(__file__).resolve().parent

ARQUIVO_POPULACAO = DIRETORIO / "datatran2025.csv"
ARQUIVO_AMOSTRA = DIRETORIO / "amostra_PRF_2025.csv"


# ============================================================
# 4. CARREGAMENTO DA POPULAÇÃO
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


# ============================================================
# 5. CARREGAMENTO DA AMOSTRA
# ============================================================

@st.cache_data(show_spinner="Carregando amostra...")
def carregar_amostra():

    erros = []

    # Testar formatos comuns de exportação CSV
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

                # Validar dimensões antes de aceitar
                if (
                    len(dados) == 383
                    and len(dados.columns) == 30
                    and "id" in dados.columns
                ):
                    return dados

                erros.append(
                    f"Separador {repr(separador)}, "
                    f"codificação {encoding}: "
                    f"{len(dados)} registros e "
                    f"{len(dados.columns)} colunas."
                )

            except (
                pd.errors.ParserError,
                UnicodeError
            ) as erro:

                erros.append(
                    f"Separador {repr(separador)}, "
                    f"codificação {encoding}: {erro}"
                )

    raise ValueError(
        "Não foi possível identificar o formato "
        "correto do CSV da amostra.\n"
        + "\n".join(erros)
    )


# ============================================================
# 6. CARREGAMENTO E TRATAMENTO DE ERROS
# ============================================================

try:

    populacao = carregar_populacao()
    amostra = carregar_amostra()

except Exception as erro:

    st.error(
        "Não foi possível carregar uma ou ambas "
        "as bases de dados."
    )

    st.exception(erro)

    st.stop()


# ============================================================
# 7. INDICADORES GERAIS
# ============================================================

st.subheader("Visão geral do estudo")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        label="População",
        value=f"{len(populacao):,}".replace(",", ".")
    )

with col2:

    st.metric(
        label="Amostra",
        value=f"{len(amostra):,}".replace(",", ".")
    )

with col3:

    fracao_amostral = (
        len(amostra) / len(populacao) * 100
    )

    st.metric(
        label="Fração amostral",
        value=f"{fracao_amostral:.4f}%".replace(".", ",")
    )

with col4:

    st.metric(
        label="Nível de confiança",
        value="95%"
    )


# ============================================================
# 8. VALIDAÇÃO DAS BASES
# ============================================================

st.divider()

st.subheader("Validação das bases de dados")

st.markdown(
    """
    Esta etapa verifica se os arquivos publicados
    correspondem à população e à amostra utilizadas
    na análise estatística do Google Colab.
    """
)

total_populacao = len(populacao)
total_amostra = len(amostra)

variaveis_populacao = len(populacao.columns)
variaveis_amostra = len(amostra.columns)

# Verificar nomes das colunas
colunas_iguais = (
    set(populacao.columns)
    == set(amostra.columns)
)

# Verificar existência do identificador
coluna_id_existe = (
    "id" in populacao.columns
    and "id" in amostra.columns
)

if coluna_id_existe:

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
        and ids_amostra.isin(ids_populacao).all()
    )

    ids_unicos = (
        ids_amostra.notna().all()
        and ids_amostra.is_unique
    )

else:

    ids_validos = False
    ids_unicos = False


# ============================================================
# 9. TABELA DE VALIDAÇÃO
# ============================================================

validacoes = [
    {
        "Verificação": "Registros da população",
        "Esperado": "72.529",
        "Encontrado": total_populacao,
        "Aprovado": total_populacao == 72529
    },
    {
        "Verificação": "Registros da amostra",
        "Esperado": "383",
        "Encontrado": total_amostra,
        "Aprovado": total_amostra == 383
    },
    {
        "Verificação": "Variáveis da população",
        "Esperado": "30",
        "Encontrado": variaveis_populacao,
        "Aprovado": variaveis_populacao == 30
    },
    {
        "Verificação": "Variáveis da amostra",
        "Esperado": "30",
        "Encontrado": variaveis_amostra,
        "Aprovado": variaveis_amostra == 30
    },
    {
        "Verificação": "Mesmas colunas",
        "Esperado": "Sim",
        "Encontrado": "Sim" if colunas_iguais else "Não",
        "Aprovado": colunas_iguais
    },
    {
        "Verificação": "IDs presentes na população",
        "Esperado": "Sim",
        "Encontrado": "Sim" if ids_validos else "Não",
        "Aprovado": ids_validos
    },
    {
        "Verificação": "IDs únicos na amostra",
        "Esperado": "Sim",
        "Encontrado": "Sim" if ids_unicos else "Não",
        "Aprovado": ids_unicos
    }
]

tabela_validacao = pd.DataFrame(validacoes)

tabela_validacao["Situação"] = (
    tabela_validacao["Aprovado"].map(
        {
            True: "✅ Aprovado",
            False: "❌ Verificar"
        }
    )
)

st.dataframe(
    tabela_validacao[
        [
            "Verificação",
            "Esperado",
            "Encontrado",
            "Situação"
        ]
    ],
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 10. RESULTADO DA VALIDAÇÃO
# ============================================================

total_verificacoes = len(tabela_validacao)

verificacoes_aprovadas = int(
    tabela_validacao["Aprovado"].sum()
)

if verificacoes_aprovadas == total_verificacoes:

    st.success(
        "Validação concluída com sucesso! "
        "As bases apresentam as dimensões esperadas, "
        "as mesmas variáveis e identificadores válidos."
    )

else:

    st.warning(
        f"Foram aprovadas {verificacoes_aprovadas} "
        f"de {total_verificacoes} verificações."
    )


# ============================================================
# 11. VISUALIZAÇÃO DOS PRIMEIROS REGISTROS
# ============================================================

st.divider()

st.subheader("Prévia dos dados")

aba_populacao, aba_amostra = st.tabs(
    ["População", "Amostra"]
)

with aba_populacao:

    st.markdown("**Primeiros registros da população**")

    st.dataframe(
        populacao.head(10),
        use_container_width=True,
        hide_index=True
    )

with aba_amostra:

    st.markdown("**Primeiros registros da amostra**")

    st.dataframe(
        amostra.head(10),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 12. INFORMAÇÕES METODOLÓGICAS
# ============================================================

st.divider()

with st.expander("Informações metodológicas"):

    st.markdown(
        """
        **População operacional**

        72.529 acidentes registrados na base
        disponibilizada pela Polícia Rodoviária Federal
        referente ao ano de 2025.

        **Unidade amostral**

        Uma ocorrência de acidente.

        **Método de amostragem**

        Amostragem aleatória simples sem reposição.

        **Tamanho amostral**

        383 ocorrências.

        **Semente pseudoaleatória**

        `random_state=42`

        **Dimensionamento**

        - Nível de confiança: 95%.
        - Margem de erro: 5%.
        - Proporção conservadora: 50%.

        **Reprodutibilidade**

        A amostra foi selecionada previamente
        no Google Colab e preservada em arquivo CSV.

        Nenhum novo sorteio é realizado no Streamlit.
        """
    )


# ============================================================
# 13. RODAPÉ
# ============================================================

st.divider()

st.caption(
    "PRF Analytics 2025 | "
    "Projeto acadêmico de Probabilidade e Estatística "
    "para IA | Fonte: Polícia Rodoviária Federal"
)
