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
