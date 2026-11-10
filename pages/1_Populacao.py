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

        **Boxplots complementares — Etapa 13.1**

        Diagramas de caixa das variáveis `pessoas` e `feridos`,
        com quartis calculados por interpolação linear e identificação
        de valores potencialmente atípicos pelo critério 1,5 × AIQ.
        Os dados não foram removidos; a exibição individual dos
        pontos pode ser ativada sob demanda.

        Para `feridos`, a população completa apresenta AIQ igual
        a zero; por isso o percentual de registros classificados
        como potencialmente atípicos exige interpretação cuidadosa.

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
