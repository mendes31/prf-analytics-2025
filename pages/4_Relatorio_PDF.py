
"""Página 4 — Exportação consolidada das páginas Início, População, Amostra e Inferência."""

from pathlib import Path
import streamlit as st

from utils.gerador_pdf import gerar_pdf


st.set_page_config(
    page_title="Relatório PDF | PRF Analytics 2025",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Relatório PDF — PRF Analytics 2025")

st.markdown(
    """
    Exporte um **documento acadêmico consolidado**, construído
    diretamente com a população original e a amostra preservada.

    O relatório contém, em ordem,
    **Início → População → Amostra → Inferência estatística**,
    mais metodologia, auditoria, sensibilidade, ajuste de Holm
    e referências.
    """
)

st.info(
    "A exportação usa os dados completos sem filtros. "
    "Nenhum novo sorteio é realizado. "
    "Os gráficos do PDF são estáticos; "
    "a versão interativa continua nos dashboards."
)

with st.container(border=True):

    st.subheader("Configuração do relatório")

    incluir_graficos = st.checkbox(
        "Incluir todos os gráficos estatísticos disponíveis no relatório",
        value=True
    )

    incluir_metodologia = st.checkbox(
        "Incluir metodologia, condições de aplicação e limitações",
        value=True
    )

    st.caption(
        "O documento pode ter dezenas de páginas, "
        "pois inclui as distribuições qualitativas "
        "e quantitativas da população e da amostra."
    )

    gerar = st.button(
        "Gerar relatório consolidado",
        type="primary",
        width="stretch"
    )

if gerar:

    try:

        with st.spinner(
            "Calculando resultados e construindo o PDF..."
        ):

            pdf = gerar_pdf(
                Path(__file__).resolve().parents[1],
                incluir_graficos=incluir_graficos,
                incluir_metodologia=incluir_metodologia
            )

        st.session_state["relatorio_pdf_bytes"] = pdf

        st.session_state["relatorio_pdf_tamanho"] = len(pdf)

        st.success(
            "Relatório gerado com sucesso. "
            "Use o botão abaixo para baixar."
        )

    except Exception as exc:

        st.error(
            "Falha na geração do relatório. "
            "Os dados não foram modificados."
        )

        st.exception(exc)

if "relatorio_pdf_bytes" in st.session_state:

    st.download_button(
        "⬇️ Baixar PDF consolidado",
        st.session_state["relatorio_pdf_bytes"],
        file_name="Relatorio_Consolidado_PRF_2025.pdf",
        mime="application/pdf",
        type="primary",
        width="stretch"
    )

    st.caption(
        f"Tamanho do documento: "
        f"{st.session_state['relatorio_pdf_tamanho'] / 1024 / 1024:.2f} MB"
    )
