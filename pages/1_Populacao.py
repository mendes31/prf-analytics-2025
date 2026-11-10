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
