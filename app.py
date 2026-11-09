
import streamlit as st

st.set_page_config(
    page_title="PRF Analytics 2025",
    page_icon="📊",
    layout="wide"
)

st.title("PRF Analytics 2025")
st.subheader("Probabilidade e Estatística para IA")

st.markdown(
    """
    Análise estatística dos acidentes em rodovias
    federais brasileiras — PRF 2025.

    **Fonte:** Polícia Rodoviária Federal (PRF)

    **População:** 72.529 ocorrências

    **Amostra:** 383 ocorrências — AAS sem reposição
    """
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("População", "72.529")

with col2:
    st.metric("Amostra", "383")

with col3:
    st.metric("Nível de confiança", "95%")

st.info(
    "Aplicação inicial configurada. "
    "Os dashboards interativos serão "
    "incorporados nas próximas etapas."
)
