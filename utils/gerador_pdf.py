
"""PRF Analytics 2025 — Gerador de relatório PDF consolidado.

Reproduz os indicadores e gráficos selecionados nos dashboards.
Utiliza exclusivamente a população e a amostra original preservada.
Não aplica filtros e não realiza novos sorteios.
"""

from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    PageTemplate,
    Frame,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# ============================================================
# 1. CONFIGURACOES DO PROJETO
# ============================================================

N_ESPERADO = 72529
n_ESPERADO = 383
ALFA = 0.05

AZUL = "#2474ad"
LARANJA = "#e99134"

# As seis variaveis continuam presentes na tabela estatistica.
QUANTITATIVAS = [
    "pessoas",
    "mortos",
    "feridos",
    "feridos_leves",
    "feridos_graves",
    "veiculos",
]

# Somente estas tres possuem graficos de distribuicao no app.
# O segundo elemento corresponde ao limite da faixa final.
GRAFICOS_QUANTITATIVOS = [
    ("pessoas", 10),
    ("mortos", 4),
    ("feridos", 6),
]

# Somente as seis variaveis qualitativas dos dashboards.
QUALITATIVAS = [
    "classificacao_acidente",
    "fase_dia",
    "tipo_acidente",
    "causa_acidente",
    "uf",
    "condicao_metereologica",
]

# Os dez grupos de referencia sao definidos na populacao.
QUALITATIVAS_TOP10 = {
    "tipo_acidente",
    "causa_acidente",
    "uf",
}

NOMES = {
    "pessoas": "Pessoas envolvidas",
    "mortos": "Vítimas fatais",
    "feridos": "Pessoas feridas",
    "feridos_leves": "Feridos leves",
    "feridos_graves": "Feridos graves",
    "veiculos": "Veículos",
    "uf": "Unidade federativa",
    "fase_dia": "Fase do dia",
    "classificacao_acidente": "Classificação do acidente",
    "tipo_acidente": "Tipo de acidente",
    "causa_acidente": "Causa do acidente",
    "condicao_metereologica": "Condição meteorológica",
}

HIP = {
    "P1": (
        "Acidentes com pelo menos uma morte",
        "Proporção",
        0.10,
    ),
    "P2": (
        "Acidentes em plena noite",
        "Proporção",
        0.30,
    ),
    "M1": (
        "Pessoas envolvidas por acidente",
        "Média",
        3.0,
    ),
    "M2": (
        "Pessoas feridas por acidente",
        "Média",
        1.0,
    ),
}


# ============================================================
# 2. FORMATACAO E FONTES
# ============================================================

def br(valor, casas=2):
    if pd.isna(valor):
        return "—"

    return (
        f"{float(valor):,.{casas}f}"
        .replace(",", "§")
        .replace(".", ",")
        .replace("§", ".")
    )


def _fontes():
    base = Path("/usr/share/fonts/truetype/dejavu")

    regular = base / "DejaVuSans.ttf"
    bold = base / "DejaVuSans-Bold.ttf"

    if (
        regular.exists()
        and "DejaVu" not in pdfmetrics.getRegisteredFontNames()
    ):
        pdfmetrics.registerFont(
            TTFont("DejaVu", str(regular))
        )
        pdfmetrics.registerFont(
            TTFont("DejaVu-Bold", str(bold))
        )

    if regular.exists():
        return "DejaVu", "DejaVu-Bold"

    return "Helvetica", "Helvetica-Bold"


# ============================================================
# 3. LEITURA E VALIDACAO DOS DADOS
# ============================================================

def _carregar(caminho, populacao=False):
    if populacao:
        dados = pd.read_csv(
            caminho,
            sep=";",
            encoding="latin1",
            low_memory=False,
        )

        dados.columns = dados.columns.str.strip()
        return dados

    erros = []

    for codificacao in ["utf-8-sig", "latin1"]:
        for separador in [",", ";", "\t"]:
            try:
                dados = pd.read_csv(
                    caminho,
                    sep=separador,
                    encoding=codificacao,
                    low_memory=False,
                )

                dados.columns = dados.columns.str.strip()

                if (
                    dados.shape == (n_ESPERADO, 30)
                    and "id" in dados.columns
                ):
                    return dados

            except Exception as erro:
                erros.append(str(erro))

    raise ValueError(
        "Formato da amostra original inválido. "
        + "; ".join(erros[:3])
    )


def preparar_bases(raiz):
    raiz = Path(raiz)

    pop = _carregar(
        raiz / "datatran2025.csv",
        populacao=True,
    )

    amo = _carregar(
        raiz / "amostra_PRF_2025.csv",
        populacao=False,
    )

    if pop.shape != (N_ESPERADO, 30):
        raise ValueError(
            "A população não possui 72.529 linhas e 30 colunas."
        )

    if amo.shape != (n_ESPERADO, 30):
        raise ValueError(
            "A amostra não possui 383 linhas e 30 colunas."
        )

    if set(pop.columns) != set(amo.columns):
        raise ValueError(
            "As colunas da população e amostra são diferentes."
        )

    for dados in [pop, amo]:
        dados["id"] = pd.to_numeric(
            dados["id"],
            errors="raise",
        ).astype("int64")

        for coluna in QUANTITATIVAS:
            dados[coluna] = pd.to_numeric(
                dados[coluna],
                errors="raise",
            )

        dados["fase_dia"] = (
            dados["fase_dia"]
            .astype("string")
            .str.strip()
        )

    if pop["id"].duplicated().any():
        raise ValueError(
            "Existem IDs duplicados na população."
        )

    if amo["id"].duplicated().any():
        raise ValueError(
            "Existem IDs duplicados na amostra."
        )

    if not amo["id"].isin(pop["id"]).all():
        raise ValueError(
            "A amostra contém IDs ausentes da população."
        )

    campos = [
        "pessoas",
        "mortos",
        "feridos",
        "fase_dia",
    ]

    origem = (
        pop.set_index("id")
        .loc[amo["id"], campos]
        .reset_index(drop=True)
    )

    atual = (
        amo[campos]
        .reset_index(drop=True)
    )

    for campo in campos:
        if not origem[campo].equals(atual[campo]):
            raise ValueError(
                f"Dados incompatíveis na variável {campo}."
            )

    return pop, amo


# ============================================================
# 4. INFERENCIA ESTATISTICA
# ============================================================

def valores(dados, codigo):
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

    coluna = "pessoas" if codigo == "M1" else "feridos"

    return dados[coluna].to_numpy(dtype=float)


def calcular_inferencia(pop, amo):
    N = len(pop)
    n = len(amo)

    fpc = np.sqrt((N - n) / (N - 1))

    zcrit = stats.norm.ppf(1 - ALFA / 2)
    tcrit = stats.t.ppf(1 - ALFA / 2, df=n - 1)

    resultados = {}

    for codigo, (nome, tipo, h0) in HIP.items():
        v = valores(amo, codigo)

        estimativa = float(np.mean(v))
        parametro = float(
            np.mean(valores(pop, codigo))
        )

        if tipo == "Proporção":
            se_ic = (
                np.sqrt(
                    estimativa * (1 - estimativa) / n
                ) * fpc
            )

            li = max(
                0,
                estimativa - zcrit * se_ic,
            )

            ls = min(
                1,
                estimativa + zcrit * se_ic,
            )

            se_h0 = (
                np.sqrt(h0 * (1 - h0) / n)
                * fpc
            )

            estatistica = (
                (estimativa - h0) / se_h0
            )

            valor_p = float(
                2 * stats.norm.sf(abs(estatistica))
            )

        else:
            desvio = float(
                np.std(v, ddof=1)
            )

            se = desvio / np.sqrt(n) * fpc

            li = estimativa - tcrit * se
            ls = estimativa + tcrit * se

            estatistica = (
                (estimativa - h0) / se
            )

            valor_p = float(
                2 * stats.t.sf(
                    abs(estatistica),
                    df=n - 1,
                )
            )

        resultados[codigo] = {
            "codigo": codigo,
            "nome": nome,
            "tipo": tipo,
            "h0": h0,
            "populacao": parametro,
            "amostra": estimativa,
            "li": li,
            "ls": ls,
            "estatistica": float(estatistica),
            "p": valor_p,
            "contem": (
                li <= parametro <= ls
            ),
        }

    p_originais = np.array([
        resultados[c]["p"]
        for c in HIP
    ])

    ordem = np.argsort(p_originais)

    holm = np.empty(len(p_originais))
    anterior = 0.0

    for posicao, indice in enumerate(ordem):
        anterior = max(
            anterior,
            min(
                1.0,
                (len(p_originais) - posicao)
                * p_originais[indice],
            ),
        )

        holm[indice] = anterior

    for indice, codigo in enumerate(HIP):
        resultados[codigo]["holm"] = float(
            holm[indice]
        )

    return resultados


# ============================================================
# 5. ANALISE DE SENSIBILIDADE
# ============================================================

def calc_sensibilidade(amo, resultados, N):
    saida = []

    for codigo, coluna in [
        ("M1", "pessoas"),
        ("M2", "feridos"),
    ]:
        v = amo[coluna].to_numpy(dtype=float)

        posicao = int(np.argmax(v))

        v_sem_maximo = np.delete(
            v,
            posicao,
        )

        n2 = len(v_sem_maximo)

        media = float(
            np.mean(v_sem_maximo)
        )

        desvio = float(
            np.std(v_sem_maximo, ddof=1)
        )

        fpc = np.sqrt(
            (N - n2) / (N - 1)
        )

        t = (
            (media - HIP[codigo][2])
            / (
                desvio / np.sqrt(n2) * fpc
            )
        )

        valor_p = float(
            2 * stats.t.sf(
                abs(t),
                df=n2 - 1,
            )
        )

        saida.append({
            "codigo": codigo,
            "id": int(amo.iloc[posicao]["id"]),
            "maximo": v[posicao],
            "original": resultados[codigo]["amostra"],
            "sem_maximo": media,
            "aparada": float(
                stats.trim_mean(v, 0.05)
            ),
            "p_original": resultados[codigo]["p"],
            "p_sem_maximo": valor_p,
        })

    return saida


# ============================================================
# 6. CONVERSAO DOS GRAFICOS PARA O PDF
# ============================================================

def _grafico(fig, largura=16.2):
    bio = BytesIO()

    fig.savefig(
        bio,
        format="png",
        dpi=135,
        facecolor="white",
        bbox_inches="tight",
    )

    plt.close(fig)

    bio.seek(0)

    from PIL import Image as PILImage

    imagem_original = PILImage.open(bio)

    largura_pdf = largura * cm

    altura_pdf = (
        largura_pdf
        * imagem_original.height
        / imagem_original.width
    )

    if altura_pdf > 13.5 * cm:
        altura_pdf = 13.5 * cm

        largura_pdf = (
            altura_pdf
            * imagem_original.width
            / imagem_original.height
        )

    bio.seek(0)

    return Image(
        bio,
        width=largura_pdf,
        height=altura_pdf,
    )


# ============================================================
# 7. DISTRIBUICOES QUANTITATIVAS
# ============================================================

def tabela_quantitativa(dados, coluna, limite):
    """Mesmas categorias e faixas usadas no Streamlit."""

    serie = pd.to_numeric(
        dados[coluna],
        errors="coerce",
    ).dropna()

    categorias = [
        str(i)
        for i in range(limite)
    ] + [f"{limite} ou mais"]

    agrupada = serie.apply(
        lambda valor: (
            str(int(valor))
            if valor < limite
            else f"{limite} ou mais"
        )
    )

    contagem = (
        agrupada.value_counts()
        .reindex(categorias, fill_value=0)
    )

    tabela = pd.DataFrame({
        "Categoria": categorias,
        "Frequência": contagem.values,
    })

    tabela["Percentual"] = (
        tabela["Frequência"]
        / len(dados)
        * 100
    )

    return tabela


def _grafico_quantitativo(dados, coluna, limite):
    tabela = tabela_quantitativa(
        dados,
        coluna,
        limite,
    )

    fig, ax = plt.subplots(
        figsize=(9.2, 3.6)
    )

    x = np.arange(len(tabela))

    y = tabela["Percentual"].to_numpy()

    barras = ax.bar(
        x,
        y,
        color=AZUL,
        width=0.77,
    )

    ax.set_xticks(
        x,
        tabela["Categoria"],
        rotation=32,
        ha="right",
        fontsize=8,
    )

    ax.set_ylabel(
        "% dos acidentes selecionados"
    )

    ax.set_xlabel(
        "Quantidade por acidente"
    )

    ax.set_title(
        NOMES[coluna]
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    ax.set_ylim(
        0,
        max(1, float(y.max()) * 1.20),
    )

    for barra, percentual in zip(barras, y):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height()
            + ax.get_ylim()[1] * 0.012,
            br(percentual, 1) + "%",
            ha="center",
            fontsize=7,
        )

    return _grafico(fig)


# ============================================================
# 8. DISTRIBUICOES QUALITATIVAS
# ============================================================

def tabela_qualitativa(
    dados,
    coluna,
    categorias=None,
):
    serie = (
        dados[coluna]
        .fillna("Não informado")
        .astype(str)
        .str.strip()
        .replace("", "Não informado")
    )

    contagem = serie.value_counts(
        dropna=False
    )

    if categorias is not None:
        contagem = contagem.reindex(
            categorias,
            fill_value=0,
        )

    tabela = pd.DataFrame({
        "Categoria": contagem.index.astype(str),
        "Frequência": contagem.to_numpy(),
    })

    tabela["Percentual"] = (
        tabela["Frequência"]
        / len(dados)
        * 100
    )

    return tabela


def _grafico_qualitativo(tabela, coluna):
    tabela = tabela.sort_values(
        ["Percentual", "Categoria"],
        ascending=[True, True],
    )

    rotulos = [
        str(texto)[:49] + "…"
        if len(str(texto)) > 50
        else str(texto)
        for texto in tabela["Categoria"]
    ]

    altura = max(
        3.3,
        min(5.0, len(tabela) * 0.45),
    )

    fig, ax = plt.subplots(
        figsize=(9.2, altura)
    )

    ax.barh(
        rotulos,
        tabela["Percentual"].to_numpy(),
        color=AZUL,
    )

    ax.set_xlabel(
        "% dos acidentes selecionados"
    )

    ax.set_title(
        NOMES[coluna]
    )

    ax.grid(
        axis="x",
        alpha=0.18,
    )

    fig.subplots_adjust(left=0.32)

    return _grafico(fig)


# ============================================================
# 9. BOXPLOTS
# ============================================================

def medidas_box(dados, coluna):
    valores = pd.to_numeric(
        dados[coluna],
        errors="coerce",
    ).dropna()

    q1 = float(
        valores.quantile(0.25)
    )

    mediana = float(
        valores.median()
    )

    q3 = float(
        valores.quantile(0.75)
    )

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
        "q1": q1,
        "med": mediana,
        "q3": q3,
        "aiq": aiq,
        "wlo": float(regulares.min()),
        "whi": float(regulares.max()),
        "atipicos": int(atipicos.sum()),
        "max": float(valores.max()),
    }


def _grafico_box(dados):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.2, 3.4),
    )

    for ax, coluna in zip(
        axes,
        ["pessoas", "feridos"],
    ):
        medidas = medidas_box(
            dados,
            coluna,
        )

        estatisticas = [{
            "label": NOMES[coluna],
            "q1": medidas["q1"],
            "med": medidas["med"],
            "q3": medidas["q3"],
            "whislo": medidas["wlo"],
            "whishi": medidas["whi"],
            "fliers": [],
        }]

        ax.bxp(
            estatisticas,
            showfliers=False,
            patch_artist=True,
            boxprops={
                "facecolor": AZUL,
                "alpha": 0.3,
            },
            medianprops={
                "color": LARANJA,
                "linewidth": 2,
            },
        )

        ax.set_title(
            NOMES[coluna]
        )

        ax.grid(
            axis="y",
            alpha=0.2,
        )

        margem = max(
            0.25,
            (
                medidas["whi"]
                - medidas["wlo"]
            ) * 0.25,
        )

        ax.set_ylim(
            medidas["wlo"] - margem,
            medidas["whi"] + margem,
        )

    fig.suptitle(
        "Boxplots — bigodes pelo critério "
        "1,5 × AIQ, sem pontos atípicos"
    )

    return _grafico(fig)


# ============================================================
# 10. GRAFICOS DE INFERENCIA
# ============================================================

def _grafico_comparacao(resultados, codigos):
    fig, ax = plt.subplots(
        figsize=(9.2, 3.7)
    )

    x = np.arange(len(codigos))
    largura = 0.32

    populacao = np.array([
        resultados[c]["populacao"]
        for c in codigos
    ])

    amostra = np.array([
        resultados[c]["amostra"]
        for c in codigos
    ])

    inferiores = (
        amostra
        - np.array([
            resultados[c]["li"]
            for c in codigos
        ])
    )

    superiores = (
        np.array([
            resultados[c]["ls"]
            for c in codigos
        ])
        - amostra
    )

    multiplicador = (
        100
        if codigos[0].startswith("P")
        else 1
    )

    ax.bar(
        x - largura / 2,
        populacao * multiplicador,
        largura,
        label="Parâmetro populacional",
        color=AZUL,
    )

    ax.bar(
        x + largura / 2,
        amostra * multiplicador,
        largura,
        yerr=np.vstack([
            inferiores,
            superiores,
        ]) * multiplicador,
        capsize=5,
        label="Estimativa e IC 95%",
        color=LARANJA,
        error_kw={
            "elinewidth": 1.4,
        },
    )

    ax.set_xticks(
        x,
        codigos,
    )

    ax.set_ylabel(
        "%"
        if multiplicador == 100
        else "Pessoas por acidente"
    )

    ax.set_title(
        "População x amostra com IC 95%"
    )

    ax.legend(
        fontsize=8
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    return _grafico(fig)


def _grafico_curva(resultado, codigo):
    teste_t = (
        resultado["tipo"] == "Média"
    )

    critico = (
        stats.t.ppf(0.975, 382)
        if teste_t
        else stats.norm.ppf(0.975)
    )

    x = np.linspace(
        -4.4,
        4.4,
        650,
    )

    y = (
        stats.t.pdf(x, 382)
        if teste_t
        else stats.norm.pdf(x)
    )

    fig, ax = plt.subplots(
        figsize=(9.2, 3.25)
    )

    ax.plot(
        x,
        y,
        color=AZUL,
        linewidth=2,
    )

    ax.fill_between(
        x,
        y,
        where=(
            abs(x)
            >= abs(resultado["estatistica"])
        ),
        color=LARANJA,
        alpha=0.37,
        label="Região do valor-p bilateral",
    )

    ax.axvline(
        -critico,
        color="firebrick",
        linestyle="--",
        linewidth=1.2,
    )

    ax.axvline(
        critico,
        color="firebrick",
        linestyle="--",
        linewidth=1.2,
        label="Críticos ±" + br(critico, 3),
    )

    ax.axvline(
        resultado["estatistica"],
        color="black",
        linewidth=1.5,
        label=(
            "Estatística "
            + br(resultado["estatistica"], 3)
        ),
    )

    ax.set_title(
        codigo
        + " — "
        + (
            "Distribuição t (382 gl)"
            if teste_t
            else "Distribuição normal Z"
        )
    )

    ax.legend(
        loc="upper right",
        fontsize=7,
    )

    ax.grid(
        alpha=0.15
    )

    return _grafico(fig)


def _grafico_sensibilidade(sensibilidade):
    fig, ax = plt.subplots(
        figsize=(9.2, 3.55)
    )

    x = np.arange(3)
    largura = 0.35

    for indice, (resultado, cor) in enumerate(
        zip(
            sensibilidade,
            [AZUL, LARANJA],
        )
    ):
        valores = [
            resultado["original"],
            resultado["sem_maximo"],
            resultado["aparada"],
        ]

        ax.bar(
            x + (indice - 0.5) * largura,
            valores,
            largura,
            color=cor,
            label=resultado["codigo"],
        )

    ax.set_xticks(
        x,
        [
            "Original",
            "Sem máximo",
            "Aparada 5%",
        ],
    )

    ax.set_ylabel(
        "Pessoas por acidente"
    )

    ax.set_title(
        "Análise de sensibilidade das médias"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    return _grafico(fig)


def _grafico_holm(resultados):
    codigos = list(HIP)

    x = np.arange(4)
    largura = 0.35

    fig, ax = plt.subplots(
        figsize=(9.2, 3.6)
    )

    ax.bar(
        x - largura / 2,
        [
            resultados[c]["p"]
            for c in codigos
        ],
        largura,
        label="Valor-p original",
        color=AZUL,
    )

    ax.bar(
        x + largura / 2,
        [
            resultados[c]["holm"]
            for c in codigos
        ],
        largura,
        label="Valor-p Holm",
        color=LARANJA,
    )

    ax.axhline(
        0.05,
        color="firebrick",
        linestyle="--",
        label="Alfa = 0,05",
    )

    ax.set_xticks(
        x,
        codigos,
    )

    ax.set_ylabel("Valor-p")

    ax.set_title(
        "Testes individuais e ajuste de Holm"
    )

    ax.legend(
        fontsize=8
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    return _grafico(fig)


# ============================================================
# 11. GERACAO DO PDF CONSOLIDADO
# ============================================================

def gerar_pdf(
    raiz: str | Path,
    incluir_graficos: bool = True,
    incluir_metodologia: bool = True,
) -> bytes:
    """Retorna o PDF em memória, sem modificar os arquivos."""

    pop, amo = preparar_bases(raiz)

    resultados = calcular_inferencia(
        pop,
        amo,
    )

    sensibilidade = calc_sensibilidade(
        amo,
        resultados,
        len(pop),
    )

    regular, negrito = _fontes()

    estilos = getSampleStyleSheet()

    estilos.add(
        ParagraphStyle(
            name="PRFTitle",
            fontName=negrito,
            fontSize=23,
            leading=32,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#173c59"),
            spaceAfter=20,
        )
    )

    estilos.add(
        ParagraphStyle(
            name="PRFH1",
            fontName=negrito,
            fontSize=15,
            leading=21,
            spaceBefore=14,
            spaceAfter=10,
            textColor=colors.HexColor("#173c59"),
            keepWithNext=True,
        )
    )

    estilos.add(
        ParagraphStyle(
            name="PRFH2",
            fontName=negrito,
            fontSize=11,
            leading=16,
            spaceBefore=10,
            spaceAfter=7,
            textColor=colors.HexColor("#245e88"),
            keepWithNext=True,
        )
    )

    estilos.add(
        ParagraphStyle(
            name="PRFBody",
            fontName=regular,
            fontSize=8.6,
            leading=13,
            spaceAfter=9,
        )
    )

    estilos.add(
        ParagraphStyle(
            name="PRFSmall",
            fontName=regular,
            fontSize=7.5,
            leading=11,
            spaceAfter=5,
        )
    )

    estilos.add(
        ParagraphStyle(
            name="PRFTable",
            fontName=regular,
            fontSize=7,
            leading=9,
        )
    )

    estilos.add(
        ParagraphStyle(
            name="PRFHead",
            fontName=negrito,
            fontSize=7,
            leading=9,
            textColor=colors.white,
        )
    )

    estilos.add(
        ParagraphStyle(
            name="PRFCenter",
            fontName=regular,
            fontSize=9,
            leading=13,
            alignment=TA_CENTER,
        )
    )

    memoria = BytesIO()

    pdf = BaseDocTemplate(
        memoria,
        pagesize=A4,
        rightMargin=1.65 * cm,
        leftMargin=1.65 * cm,
        topMargin=1.65 * cm,
        bottomMargin=1.65 * cm,
        title="PRF Analytics 2025 — Relatório consolidado",
        author="Projeto Probabilidade e Estatística para IA",
    )

    frame = Frame(
        pdf.leftMargin,
        pdf.bottomMargin,
        pdf.width,
        pdf.height,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )

    def rodape(canvas, documento):
        canvas.saveState()

        canvas.setFont(
            regular,
            7.2,
        )

        canvas.setFillColor(
            colors.HexColor("#64748b")
        )

        canvas.drawString(
            1.65 * cm,
            1.04 * cm,
            "PRF Analytics 2025 | População, amostra e inferência",
        )

        canvas.drawRightString(
            A4[0] - 1.65 * cm,
            1.04 * cm,
            f"Página {documento.page}",
        )

        canvas.restoreState()

    pdf.addPageTemplates(
        PageTemplate(
            id="normal",
            frames=[frame],
            onPage=rodape,
        )
    )

    elementos = []

    def texto(conteudo, estilo="PRFBody"):
        elementos.append(
            Paragraph(conteudo, estilos[estilo])
        )

    def titulo(conteudo):
        texto(conteudo, "PRFH1")

    def subtitulo(conteudo):
        texto(conteudo, "PRFH2")

    def espaco(tamanho=8):
        elementos.append(
            Spacer(1, tamanho)
        )

    def nova_pagina():
        elementos.append(PageBreak())

    def tabela(
        colunas,
        linhas,
        larguras=None,
    ):
        dados = [
            [
                Paragraph(
                    escape(str(valor)),
                    estilos["PRFHead"],
                )
                for valor in colunas
            ]
        ]

        for linha in linhas:
            dados.append([
                Paragraph(
                    escape(str(valor)),
                    estilos["PRFTable"],
                )
                for valor in linha
            ])

        objeto = Table(
            dados,
            colWidths=larguras,
            repeatRows=1,
            hAlign="LEFT",
        )

        objeto.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#245e88"),
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        colors.HexColor("#edf4f8"),
                    ],
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "LINEBELOW",
                    (0, -1),
                    (-1, -1),
                    0.4,
                    colors.HexColor("#c5d3dd"),
                ),
            ])
        )

        elementos.append(objeto)
        espaco(9)

    def imagem(objeto):
        if incluir_graficos:
            elementos.append(objeto)
            espaco(10)

    def formatar_parametro(valor, tipo):
        if tipo == "Proporção":
            return br(valor * 100, 2) + "%"

        return br(valor, 3)

    # ========================================================
    # 12. CAPA
    # ========================================================

    espaco(60)

    texto(
        "PRF ANALYTICS 2025",
        "PRFTitle",
    )

    texto(
        "Análise estatística dos acidentes "
        "em rodovias federais brasileiras",
        "PRFCenter",
    )

    espaco(28)

    texto(
        "Relatório consolidado dos quatro "
        "ambientes da aplicação",
        "PRFCenter",
    )

    espaco(25)

    tabela(
        ["BASE / RECORTE", "CARACTERÍSTICAS"],
        [
            [
                "População operacional",
                "72.529 ocorrências de acidentes",
            ],
            [
                "Amostra original",
                "383 ocorrências, AAS sem reposição",
            ],
            [
                "Variáveis",
                "30 colunas",
            ],
            [
                "Dimensionamento",
                "95% de confiança; 5% de margem de referência; p=50%",
            ],
            [
                "Estatística inferencial",
                "2 proporções, 2 médias, 4 testes, IC 95% e ajuste de Holm",
            ],
        ],
        [6 * cm, 11.7 * cm],
    )

    espaco(24)

    texto(
        "Fonte: Polícia Rodoviária Federal (PRF) "
        "— Dados Abertos | Ano de referência: 2025",
        "PRFCenter",
    )

    texto(
        "Data de geração: "
        + date.today().strftime("%d/%m/%Y"),
        "PRFCenter",
    )

    nova_pagina()

    # ========================================================
    # 13. INICIO
    # ========================================================

    titulo("Sumário")

    for item in [
        "1. Início — contextualização, base, metodologia e auditoria",
        "2. Dashboard 1 — análise descritiva da população",
        "3. Dashboard 2 — análise descritiva da amostra",
        "4. Dashboard 3 — inferência e comparação",
        "5. Análises complementares — sensibilidade e Holm",
        "6. Metodologia, limitações e referências",
    ]:
        texto(escape(item))

    texto(
        "O documento reproduz as análises e visualizações "
        "selecionadas nos dashboards, em formato estático. "
        "A interatividade permanece disponível no Streamlit.",
        "PRFSmall",
    )

    titulo("1. Início — visão geral do estudo")

    texto(
        "Este trabalho compara parâmetros calculados "
        "diretamente na população operacional de acidentes "
        "da PRF em 2025 com estimativas obtidas por "
        "amostragem aleatória simples."
    )

    tabela(
        ["Indicador", "Valor"],
        [
            [
                "Ocorrências na população",
                br(len(pop), 0),
            ],
            [
                "Ocorrências na amostra",
                br(len(amo), 0),
            ],
            [
                "Fração amostral",
                br(100 * len(amo) / len(pop), 4) + "%",
            ],
            [
                "Variáveis disponíveis",
                str(len(pop.columns)),
            ],
            [
                "Nível de confiança",
                "95%",
            ],
            [
                "Fonte",
                "Polícia Rodoviária Federal",
            ],
        ],
        [10 * cm, 7.7 * cm],
    )

    if incluir_metodologia:
        subtitulo("Amostragem e seleção")

        texto(
            "Dimensionamento de referência para proporções: "
            "N=72.529, confiança de 95%, margem de erro "
            "e=0,05 e proporção conservadora p=0,50, "
            "com correção para população finita. "
            "O resultado foi arredondado para n=383. "
            "A margem de 5% não é uma garantia universal "
            "para todas as médias estudadas."
        )

        texto(
            "A seleção original utilizou amostragem aleatória "
            "simples sem reposição, com semente pseudoaleatória "
            "fixa random_state=42. A amostra foi preservada "
            "em CSV. Nenhum novo sorteio é realizado."
        )

        subtitulo("Integridade e reprodutibilidade")

        tabela(
            ["Verificação", "Resultado"],
            [
                ["72.529 registros populacionais", "Aprovado"],
                ["383 registros amostrais", "Aprovado"],
                ["30 colunas em cada base", "Aprovado"],
                ["Mesmas colunas", "Aprovado"],
                ["IDs presentes e numéricos", "Aprovado"],
                ["IDs únicos nas duas bases", "Aprovado"],
                [
                    "Todos os IDs amostrais presentes na população",
                    "Aprovado",
                ],
                [
                    "Correspondência de pessoas, mortos, feridos e fase do dia",
                    "Aprovado",
                ],
            ],
            [13 * cm, 4.7 * cm],
        )

    subtitulo("Síntese de resultados")

    rejeicoes = sum(
        resultado["p"] < ALFA
        for resultado in resultados.values()
    )

    rejeicoes_holm = sum(
        resultado["holm"] < ALFA
        for resultado in resultados.values()
    )

    cobertura = sum(
        resultado["contem"]
        for resultado in resultados.values()
    )

    texto(
        f"{cobertura} dos 4 intervalos de confiança "
        "de 95% incluíram seus respectivos parâmetros "
        "populacionais. Houve "
        f"{rejeicoes} rejeições nos testes individuais "
        f"e {rejeicoes_holm} após o ajuste de Holm."
    )

    # ========================================================
    # 14. POPULACAO E AMOSTRA
    # ========================================================

    for dados, nome_conjunto, indice in [
        (pop, "População", 2),
        (amo, "Amostra", 3),
    ]:
        nova_pagina()

        titulo(
            f"{indice}. Dashboard {indice - 1} "
            f"— análise descritiva da {nome_conjunto.lower()}"
        )

        N = len(dados)

        envolvidos = int(dados["pessoas"].sum())
        mortos = int(dados["mortos"].sum())
        feridos = int(dados["feridos"].sum())

        veiculos = int(dados["veiculos"].sum())

        acidentes_mortes = int(
            dados["mortos"].gt(0).sum()
        )

        acidentes_feridos = int(
            dados["feridos"].gt(0).sum()
        )

        proporcao_mortes = (
            acidentes_mortes / N * 100
        )

        tabela(
            ["Indicador", nome_conjunto],
            [
                ["Acidentes", br(N, 0)],
                ["Pessoas envolvidas", br(envolvidos, 0)],
                ["Vítimas fatais", br(mortos, 0)],
                ["Pessoas feridas", br(feridos, 0)],
                ["Veículos envolvidos", br(veiculos, 0)],
                [
                    "Acidentes com mortes",
                    br(acidentes_mortes, 0),
                ],
                [
                    "Proporção com mortes",
                    br(proporcao_mortes, 2) + "%",
                ],
                [
                    "Acidentes com feridos",
                    br(acidentes_feridos, 0),
                ],
            ],
            [12.5 * cm, 5.2 * cm],
        )

        # ----------------------------------------------------
        # MEDIDAS DESCRITIVAS
        # ----------------------------------------------------

        subtitulo(
            "Medidas estatísticas das variáveis quantitativas"
        )

        medidas = []

        for coluna in QUANTITATIVAS:
            serie = dados[coluna].dropna()

            # Populacao: ddof=0.
            # Amostra: ddof=1.
            ddof = 0 if indice == 2 else 1

            medidas.append([
                NOMES[coluna],
                br(serie.mean(), 3),
                br(serie.median(), 2),
                br(serie.std(ddof=ddof), 3),
                br(serie.min(), 0),
                br(serie.max(), 0),
            ])

        tabela(
            [
                "Variável",
                "Média",
                "Mediana",
                "Desvio-padrão",
                "Mín.",
                "Máx.",
            ],
            medidas,
            [
                5.6 * cm,
                2.25 * cm,
                2.25 * cm,
                3.1 * cm,
                2.2 * cm,
                2.2 * cm,
            ],
        )

        texto(
            "Desvio-padrão calculado com "
            + (
                "ddof=0 (populacional)."
                if indice == 2
                else "ddof=1 (amostral)."
            ),
            "PRFSmall",
        )

        # ----------------------------------------------------
        # GRAFICOS QUANTITATIVOS
        # ----------------------------------------------------

        subtitulo("Distribuições quantitativas")

        texto(
            "Foram preservados os três gráficos dos dashboards: "
            "pessoas envolvidas, vítimas fatais e pessoas feridas. "
            "As faixas finais correspondem, respectivamente, "
            "a 10 ou mais, 4 ou mais e 6 ou mais."
        )

        for coluna, limite in GRAFICOS_QUANTITATIVOS:
            imagem(
                _grafico_quantitativo(
                    dados,
                    coluna,
                    limite,
                )
            )

        # ----------------------------------------------------
        # BOXPLOTS
        # ----------------------------------------------------

        subtitulo("Boxplots e valores extremos")

        linhas_box = []

        for coluna in [
            "pessoas",
            "feridos",
        ]:
            m = medidas_box(
                dados,
                coluna,
            )

            linhas_box.append([
                NOMES[coluna],
                br(m["q1"], 2),
                br(m["med"], 2),
                br(m["q3"], 2),
                br(m["aiq"], 2),
                br(m["wlo"], 2),
                br(m["whi"], 2),
                br(m["atipicos"], 0),
            ])

        tabela(
            [
                "Variável",
                "Q1",
                "Mediana",
                "Q3",
                "AIQ",
                "Bigode inf.",
                "Bigode sup.",
                "Atípicos",
            ],
            linhas_box,
            [
                3.4 * cm,
                1.6 * cm,
                2.2 * cm,
                1.5 * cm,
                1.5 * cm,
                2.5 * cm,
                2.5 * cm,
                2.5 * cm,
            ],
        )

        imagem(
            _grafico_box(dados)
        )

        texto(
            "Os boxplots não exibem individualmente os pontos "
            "atípicos para facilitar a leitura. Os bigodes "
            "são calculados pelos valores observados dentro "
            "dos limites de 1,5 × AIQ. Nenhuma ocorrência "
            "é removida dos cálculos.",
            "PRFSmall",
        )

        # ----------------------------------------------------
        # GRAFICOS QUALITATIVOS
        # ----------------------------------------------------

        subtitulo(
            "Distribuições qualitativas — frequências e proporções"
        )

        for coluna in QUALITATIVAS:
            subtitulo(
                NOMES[coluna]
            )

            # Referencia sempre definida na populacao.
            referencia = tabela_qualitativa(
                pop,
                coluna,
            )

            if coluna in QUALITATIVAS_TOP10:
                categorias = (
                    referencia["Categoria"]
                    .head(10)
                    .tolist()
                )
            else:
                categorias = (
                    referencia["Categoria"]
                    .tolist()
                )

            tabela_resultado = tabela_qualitativa(
                dados,
                coluna,
                categorias,
            )

            linhas = []

            for _, linha in tabela_resultado.iterrows():
                linhas.append([
                    linha["Categoria"],
                    br(linha["Frequência"], 0),
                    br(linha["Percentual"], 2) + "%",
                ])

            tabela(
                [
                    "Categoria",
                    "Frequência",
                    "Proporção",
                ],
                linhas,
                [
                    11.2 * cm,
                    3.1 * cm,
                    3.4 * cm,
                ],
            )

            imagem(
                _grafico_qualitativo(
                    tabela_resultado,
                    coluna,
                )
            )

            if coluna in QUALITATIVAS_TOP10:
                texto(
                    "As dez categorias foram definidas "
                    "pelas frequências na população. "
                    "Os percentuais são calculados "
                    "sobre todos os registros do conjunto.",
                    "PRFSmall",
                )

        # ----------------------------------------------------
        # PREVIA DOS REGISTROS
        # ----------------------------------------------------

        subtitulo(
            "Consulta dos registros e escopo"
        )

        texto(
            "O dashboard interativo permite aplicar filtros "
            "e consultar registros individuais. Este PDF "
            "apresenta as estatísticas não filtradas dos "
            f"{br(N, 0)} acidentes."
        )

        texto(
            "Prévia dos dez primeiros registros:",
            "PRFSmall",
        )

        previa = (
            dados[
                [
                    "id",
                    "uf",
                    "pessoas",
                    "mortos",
                    "feridos",
                    "fase_dia",
                ]
            ]
            .head(10)
        )

        tabela(
            previa.columns.tolist(),
            previa.astype(str).values.tolist(),
            [
                3 * cm,
                2 * cm,
                2.8 * cm,
                2.5 * cm,
                2.5 * cm,
                4.9 * cm,
            ],
        )

    # ========================================================
    # 15. INFERENCIA ESTATISTICA
    # ========================================================

    nova_pagina()

    titulo(
        "4. Dashboard 3 — inferência e comparação"
    )

    texto(
        "As estimativas e os testes foram calculados "
        "exclusivamente com a amostra original de "
        "383 ocorrências. Os parâmetros populacionais "
        "foram calculados separadamente para comparação."
    )

    subtitulo(
        "Comparação entre parâmetros e estimativas, com IC 95%"
    )

    linhas_ic = []

    for codigo, resultado in resultados.items():
        tipo = resultado["tipo"]

        linhas_ic.append([
            codigo,
            resultado["nome"],
            formatar_parametro(
                resultado["populacao"],
                tipo,
            ),
            formatar_parametro(
                resultado["amostra"],
                tipo,
            ),
            formatar_parametro(
                resultado["li"],
                tipo,
            )
            + " a "
            + formatar_parametro(
                resultado["ls"],
                tipo,
            ),
            (
                "Sim"
                if resultado["contem"]
                else "Não"
            ),
        ])

    tabela(
        [
            "Teste",
            "Parâmetro",
            "População",
            "Amostra",
            "IC 95%",
            "Contém",
        ],
        linhas_ic,
        [
            1.35 * cm,
            5.5 * cm,
            2.35 * cm,
            2.3 * cm,
            4.75 * cm,
            1.45 * cm,
        ],
    )

    imagem(
        _grafico_comparacao(
            resultados,
            ["P1", "P2"],
        )
    )

    imagem(
        _grafico_comparacao(
            resultados,
            ["M1", "M2"],
        )
    )

    # --------------------------------------------------------
    # TESTES DE HIPOTESES
    # --------------------------------------------------------

    subtitulo(
        "Testes bilaterais de hipóteses — α = 0,05"
    )

    linhas_testes = []

    for codigo, resultado in resultados.items():
        tipo = resultado["tipo"]

        simbolo = (
            "p"
            if tipo == "Proporção"
            else "μ"
        )

        estatistica_nome = (
            "Z"
            if tipo == "Proporção"
            else "t"
        )

        linhas_testes.append([
            codigo,
            "H0: "
            + simbolo
            + " = "
            + formatar_parametro(
                resultado["h0"],
                tipo,
            ),
            estatistica_nome
            + " = "
            + br(
                resultado["estatistica"],
                4,
            ),
            br(resultado["p"], 6),
            (
                "Rejeitar"
                if resultado["p"] < ALFA
                else "Não rejeitar"
            ),
        ])

    tabela(
        [
            "Teste",
            "Hipótese nula",
            "Estatística",
            "Valor-p",
            "Decisão H0",
        ],
        linhas_testes,
        [
            1.45 * cm,
            5.65 * cm,
            3.5 * cm,
            2.8 * cm,
            4.3 * cm,
        ],
    )

    texto(
        "Todas as hipóteses alternativas são bilaterais "
        "(≠). Os valores de referência de H0 são "
        "exploratórios e não representam metas oficiais da PRF."
    )

    subtitulo(
        "Distribuições teóricas das estatísticas de teste"
    )

    for codigo, resultado in resultados.items():
        imagem(
            _grafico_curva(
                resultado,
                codigo,
            )
        )

    subtitulo(
        "Interpretação dos resultados"
    )

    texto(
        "Os quatro intervalos de confiança de 95% "
        "contêm os parâmetros da população observada. "
        "P1 e M2 rejeitam H0 nos testes individuais; "
        "P2 e M1 não rejeitam H0. A não rejeição "
        "não demonstra igualdade com o valor de referência."
    )

    # ========================================================
    # 16. ANALISES COMPLEMENTARES
    # ========================================================

    nova_pagina()

    titulo(
        "5. Análises complementares"
    )

    subtitulo(
        "Sensibilidade das médias a observações extremas"
    )

    linhas_sensibilidade = []

    for resultado in sensibilidade:
        p_sem_maximo = (
            "<0,000001"
            if resultado["p_sem_maximo"] < 0.000001
            else br(resultado["p_sem_maximo"], 6)
        )

        linhas_sensibilidade.append([
            resultado["codigo"],
            str(resultado["id"]),
            br(resultado["maximo"], 0),
            br(resultado["original"], 4),
            br(resultado["sem_maximo"], 4),
            br(resultado["aparada"], 4),
            br(resultado["p_original"], 6),
            p_sem_maximo,
        ])

    tabela(
        [
            "Teste",
            "ID extremo",
            "Máximo",
            "Média original",
            "Sem máximo",
            "Aparada 5%",
            "p original",
            "p sem máx.",
        ],
        linhas_sensibilidade,
        [
            1.05 * cm,
            2.7 * cm,
            1.6 * cm,
            2.55 * cm,
            2.35 * cm,
            2.3 * cm,
            2.55 * cm,
            2.6 * cm,
        ],
    )

    imagem(
        _grafico_sensibilidade(
            sensibilidade
        )
    )

    texto(
        "A exclusão do máximo é um exercício de "
        "sensibilidade. Não modifica a amostra original "
        "nem substitui os resultados principais."
    )

    # --------------------------------------------------------
    # HOLM
    # --------------------------------------------------------

    subtitulo(
        "Ajuste de Holm para quatro testes"
    )

    linhas_holm = []

    for codigo, resultado in resultados.items():
        linhas_holm.append([
            codigo,
            br(resultado["p"], 6),
            br(resultado["holm"], 6),
            (
                "Rejeitar"
                if resultado["p"] < ALFA
                else "Não rejeitar"
            ),
            (
                "Rejeitar"
                if resultado["holm"] < ALFA
                else "Não rejeitar"
            ),
        ])

    tabela(
        [
            "Teste",
            "p original",
            "p Holm",
            "Decisão individual",
            "Decisão Holm",
        ],
        linhas_holm,
        [
            1.55 * cm,
            3 * cm,
            2.9 * cm,
            5.3 * cm,
            4.95 * cm,
        ],
    )

    imagem(
        _grafico_holm(resultados)
    )

    texto(
        "O procedimento de Holm controla o erro familiar "
        "associado aos quatro testes. Ao nível de 5%, "
        "nenhum teste permanece rejeitado após o ajuste."
    )

    # ========================================================
    # 17. METODOLOGIA E REFERENCIAS
    # ========================================================

    nova_pagina()

    titulo(
        "6. Metodologia, limitações e referências"
    )

    if incluir_metodologia:
        subtitulo(
            "Correção para população finita"
        )

        fpc = np.sqrt(
            (len(pop) - len(amo))
            / (len(pop) - 1)
        )

        texto(
            "FPC = raiz[(N − n)/(N − 1)] = "
            + br(fpc, 6)
            + ". Para proporções, foi utilizado o "
            "intervalo de Wald e o teste Z bilateral. "
            "Para médias, intervalos e testes t de Student "
            "com 382 graus de liberdade. "
            "A FPC foi incorporada aos erros-padrão."
        )

        texto(
            "Referências das hipóteses: "
            "P1 p0=10%; P2 p0=30%; "
            "M1 μ0=3; M2 μ0=1. "
            "Nível de significância α=0,05."
        )

        subtitulo(
            "Condições e limitações"
        )

        texto(
            "Foi examinada uma única amostra aleatória "
            "simples. A população operacional não representa "
            "necessariamente acidentes não registrados ou "
            "outros anos. As variáveis quantitativas podem "
            "apresentar assimetria e valores extremos, "
            "exigindo cautela na interpretação dos "
            "procedimentos inferenciais."
        )

        texto(
            "A cobertura observada dos intervalos nesta "
            "amostra não garante cobertura em todas as "
            "amostras possíveis. Os valores das hipóteses "
            "nulas são referências exploratórias, "
            "não metas oficiais da PRF."
        )

        subtitulo(
            "Variáveis selecionadas"
        )

        texto(
            "Tabela de medidas estatísticas: "
            + ", ".join(
                NOMES[coluna]
                for coluna in QUANTITATIVAS
            )
            + ". Gráficos quantitativos: "
            + ", ".join(
                NOMES[coluna]
                for coluna, _ in GRAFICOS_QUANTITATIVOS
            )
            + ". Gráficos qualitativos: "
            + ", ".join(
                NOMES[coluna]
                for coluna in QUALITATIVAS
            )
            + "."
        )

    subtitulo("Conclusão")

    texto(
        "A amostra aleatória simples de 383 ocorrências "
        "permitiu estimar características da população "
        "operacional de 72.529 acidentes. Os quatro "
        "intervalos de confiança incluíram os valores "
        "populacionais conhecidos, embora as estimativas "
        "pontuais não tenham sido idênticas. "
        "A análise de sensibilidade e o ajuste de Holm "
        "evidenciaram a importância de considerar "
        "variabilidade, pressupostos e comparações múltiplas."
    )

    subtitulo(
        "Referências e reprodutibilidade"
    )

    referencias = [
        (
            "Polícia Rodoviária Federal — Dados Abertos. "
            "https://www.gov.br/prf/pt-br/"
            "acesso-a-informacao/dados-abertos/"
            "dados-abertos-da-prf"
        ),
        (
            "PRF Analytics 2025 — Repositório. "
            "https://github.com/mendes31/prf-analytics-2025"
        ),
        (
            "PRF Analytics 2025 — Dashboards. "
            "https://prf-analytics-2025.streamlit.app"
        ),
        (
            "Arquivos utilizados: datatran2025.csv "
            "e amostra_PRF_2025.csv."
        ),
    ]

    for referencia in referencias:
        texto(
            escape(referencia),
            "PRFSmall",
        )

    # ========================================================
    # 18. FINALIZACAO
    # ========================================================

    pdf.build(elementos)

    return memoria.getvalue()


# ============================================================
# 19. EXECUCAO DIRETA OPCIONAL
# ============================================================

if __name__ == "__main__":
    raiz = Path(__file__).resolve().parents[1]

    destino = (
        raiz / "Relatorio_Consolidado_PRF_2025.pdf"
    )

    destino.write_bytes(
        gerar_pdf(raiz)
    )

    print(
        f"Gerado: {destino} "
        f"({destino.stat().st_size} bytes)"
    )
