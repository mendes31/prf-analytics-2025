
"""Relatório acadêmico do PRF Analytics 2025 — dados originais, sem novo sorteio.

A exportação usa os mesmos parâmetros de inferência do Dashboard 3. Os gráficos
são reconstruídos estaticamente a partir dos CSVs, não são capturas do Streamlit.
"""
from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
    Spacer, Table, TableStyle, PageBreak, Image, KeepTogether)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

N_ESPERADO, n_ESPERADO = 72529, 383
ALFA = 0.05
AZUL, LARANJA = '#2474ad', '#e99134'
QUANTITATIVAS = ['pessoas', 'mortos', 'feridos', 'feridos_leves', 'feridos_graves', 'veiculos']
QUALITATIVAS = ['uf', 'fase_dia', 'classificacao_acidente', 'tipo_acidente',
                'causa_acidente', 'condicao_metereologica', 'dia_semana',
                'tipo_pista', 'tracado_via', 'uso_solo', 'sentido_via']
NOMES = {'pessoas':'Pessoas envolvidas', 'mortos':'Mortos', 'feridos':'Feridos',
 'feridos_leves':'Feridos leves', 'feridos_graves':'Feridos graves',
 'veiculos':'Veículos', 'uf':'Unidade federativa', 'fase_dia':'Fase do dia',
 'classificacao_acidente':'Classificação do acidente', 'tipo_acidente':'Tipo de acidente',
 'causa_acidente':'Causa do acidente', 'condicao_metereologica':'Condição meteorológica',
 'dia_semana':'Dia da semana', 'tipo_pista':'Tipo de pista',
 'tracado_via':'Traçado da via', 'uso_solo':'Uso do solo', 'sentido_via':'Sentido da via'}
HIP = {
 'P1':('Acidentes com pelo menos uma morte','Proporção',.10),
 'P2':('Acidentes em plena noite','Proporção',.30),
 'M1':('Pessoas envolvidas por acidente','Média',3.),
 'M2':('Pessoas feridas por acidente','Média',1.),
}


def br(x, d=2):
    if pd.isna(x): return '—'
    return f'{float(x):,.{d}f}'.replace(',', '§').replace('.', ',').replace('§', '.')


def _fontes():
    base = Path('/usr/share/fonts/truetype/dejavu')
    regular, bold = base/'DejaVuSans.ttf', base/'DejaVuSans-Bold.ttf'
    if regular.exists() and 'DejaVu' not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont('DejaVu', str(regular)))
        pdfmetrics.registerFont(TTFont('DejaVu-Bold', str(bold)))
    return ('DejaVu', 'DejaVu-Bold') if regular.exists() else ('Helvetica', 'Helvetica-Bold')


def _carregar(caminho, populacao=False):
    if populacao:
        d = pd.read_csv(caminho, sep=';', encoding='latin1', low_memory=False)
    else:
        erros=[]
        for encoding in ['utf-8-sig','latin1']:
            for sep in [',',';','\t']:
                try:
                    d = pd.read_csv(caminho, sep=sep, encoding=encoding, low_memory=False)
                    if d.shape == (n_ESPERADO, 30) and 'id' in d:
                        break
                except Exception as exc:
                    erros.append(str(exc))
            else:
                continue
            break
        else:
            raise ValueError('Amostra inválida: ' + '; '.join(erros[:3]))
    d.columns=d.columns.str.strip()
    return d


def preparar_bases(raiz):
    raiz=Path(raiz)
    pop=_carregar(raiz/'datatran2025.csv', True)
    amo=_carregar(raiz/'amostra_PRF_2025.csv')
    if pop.shape!=(N_ESPERADO,30) or amo.shape!=(n_ESPERADO,30):
        raise ValueError('Dimensões diferentes de (72.529,30) e (383,30).')
    if set(pop.columns)!=set(amo.columns):
        raise ValueError('Colunas da população e da amostra são diferentes.')
    for d in (pop, amo):
        d['id']=pd.to_numeric(d['id'], errors='raise').astype('int64')
        for coluna in QUANTITATIVAS:
            d[coluna]=pd.to_numeric(d[coluna], errors='raise')
        d['fase_dia']=d['fase_dia'].astype('string').str.strip()
    if pop.id.duplicated().any() or amo.id.duplicated().any():
        raise ValueError('IDs repetidos encontrados.')
    if not amo.id.isin(pop.id).all():
        raise ValueError('A amostra contém IDs ausentes na população.')
    campos=['pessoas','mortos','feridos','fase_dia']
    origem=pop.set_index('id').loc[amo.id, campos].reset_index(drop=True)
    atual=amo[campos].reset_index(drop=True)
    for campo in campos:
        if not origem[campo].equals(atual[campo]):
            raise ValueError(f'Inconsistência entre amostra e população: {campo}')
    return pop, amo


def valores(d, codigo):
    if codigo=='P1':return d['mortos'].gt(0).astype(float).to_numpy()
    if codigo=='P2':return d['fase_dia'].eq('Plena Noite').fillna(False).astype(float).to_numpy()
    return d['pessoas' if codigo=='M1' else 'feridos'].to_numpy(dtype=float)


def calcular_inferencia(pop, amo):
    N,n=len(pop),len(amo)
    fpc=np.sqrt((N-n)/(N-1))
    zcrit=stats.norm.ppf(1-ALFA/2)
    tcrit=stats.t.ppf(1-ALFA/2, df=n-1)
    resultados={}
    for codigo,(nome,tipo,h0) in HIP.items():
        v=valores(amo,codigo)
        estimativa=float(np.mean(v)); parametro=float(np.mean(valores(pop,codigo)))
        if tipo=='Proporção':
            se_ic=np.sqrt(estimativa*(1-estimativa)/n)*fpc
            lower=max(0,estimativa-zcrit*se_ic)
            upper=min(1,estimativa+zcrit*se_ic)
            se_h0=np.sqrt(h0*(1-h0)/n)*fpc
            estatistica=(estimativa-h0)/se_h0
            valor_p=float(2*stats.norm.sf(abs(estatistica)))
        else:
            sd=float(np.std(v,ddof=1))
            se=sd/np.sqrt(n)*fpc
            lower=estimativa-tcrit*se;upper=estimativa+tcrit*se
            estatistica=(estimativa-h0)/se
            valor_p=float(2*stats.t.sf(abs(estatistica),df=n-1))
        resultados[codigo]=dict(codigo=codigo,nome=nome,tipo=tipo,h0=h0,
            populacao=parametro,amostra=estimativa,li=lower,ls=upper,
            estatistica=float(estatistica),p=valor_p,
            contem=(lower<=parametro<=upper))
    p=np.array([resultados[c]['p'] for c in HIP]); ordem=np.argsort(p)
    holm=np.empty(len(p));prev=0.
    for pos,ix in enumerate(ordem):
        prev=max(prev,min(1.,(len(p)-pos)*p[ix]));holm[ix]=prev
    for i,c in enumerate(HIP): resultados[c]['holm']=float(holm[i])
    return resultados


def calc_sensibilidade(amo, resultados, N):
    saida=[]
    for codigo,coluna in [('M1','pessoas'),('M2','feridos')]:
        v=amo[coluna].to_numpy(dtype=float)
        k=int(np.argmax(v));v2=np.delete(v,k);n2=len(v2)
        media=float(np.mean(v2));sd=float(np.std(v2,ddof=1))
        fpc=np.sqrt((N-n2)/(N-1))
        t=(media-HIP[codigo][2])/(sd/np.sqrt(n2)*fpc)
        pv=float(2*stats.t.sf(abs(t),df=n2-1))
        saida.append(dict(codigo=codigo,id=int(amo.iloc[k]['id']),maximo=v[k],
            original=resultados[codigo]['amostra'],sem_maximo=media,
            aparada=float(stats.trim_mean(v,.05)),p_original=resultados[codigo]['p'],
            p_sem_maximo=pv))
    return saida


def _grafico(fig, largura=16.2):
    bio=BytesIO()
    fig.savefig(bio,format='png',dpi=135,facecolor='white',bbox_inches='tight')
    plt.close(fig);bio.seek(0)
    from PIL import Image as PILImage
    pic=PILImage.open(bio)
    w=largura*cm; h=w*pic.height/pic.width
    if h>13.5*cm:
        h=13.5*cm;w=h*pic.width/pic.height
    bio.seek(0)
    return Image(bio,width=w,height=h)


def _grafico_quantitativo(d,col):
    s=d[col].dropna()
    cont=s.value_counts().sort_index()
    if len(cont)>13:
        values=cont[cont.index<10]
        values.loc['10+']=int(cont[cont.index>=10].sum())
    else: values=cont
    fig,ax=plt.subplots(figsize=(9.2,3.6))
    ax.bar([str(x) for x in values.index], values.values/len(s)*100, color=AZUL)
    ax.set_ylabel('Percentual dos acidentes (%)')
    ax.set_title(NOMES[col]);ax.grid(axis='y',alpha=.2)
    return _grafico(fig)


def _grafico_qualitativo(d,col,top=10):
    s=d[col].fillna('Não informado').astype(str).str.strip()
    count=s.value_counts().head(top).iloc[::-1]
    labels=[t[:49]+'…' if len(t)>50 else t for t in count.index]
    fig,ax=plt.subplots(figsize=(9.2,max(3.3,min(5.,len(count)*.45))))
    ax.barh(labels,count.values/len(d)*100,color=AZUL)
    ax.set_xlabel('Percentual dos acidentes (%)');ax.set_title(NOMES[col])
    ax.grid(axis='x',alpha=.18)
    fig.subplots_adjust(left=.32)
    return _grafico(fig)


def _grafico_box(d):
    fig,axes=plt.subplots(1,2,figsize=(9.2,3.4))
    for ax,col in zip(axes,['pessoas','feridos']):
        arr=d[col].dropna().to_numpy(dtype=float)
        ax.boxplot(arr,vert=True,showfliers=False,patch_artist=True,
            boxprops=dict(facecolor=AZUL,alpha=.4),medianprops=dict(color=LARANJA,linewidth=2))
        ax.set_title(NOMES[col]);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Boxplots — sem marcadores de atípicos (apenas para legibilidade)')
    return _grafico(fig)


def _grafico_comparacao(resultados,codigos):
    fig,ax=plt.subplots(figsize=(9.2,3.7));xx=np.arange(len(codigos));w=.32
    pop=np.array([resultados[c]['populacao'] for c in codigos]);amo=np.array([resultados[c]['amostra'] for c in codigos])
    lower=amo-np.array([resultados[c]['li'] for c in codigos]);upper=np.array([resultados[c]['ls'] for c in codigos])-amo
    mul=100 if codigos[0][0]=='P' else 1
    ax.bar(xx-w/2,pop*mul,w,label='Parâmetro populacional',color=AZUL)
    ax.bar(xx+w/2,amo*mul,w,yerr=np.vstack([lower,upper])*mul,
           capsize=5,label='Estimativa e IC 95%',color=LARANJA,error_kw={'elinewidth':1.4})
    ax.set_xticks(xx,codigos);ax.set_ylabel('%' if mul==100 else 'Pessoas por acidente')
    ax.set_title('População x amostra com IC 95%');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    return _grafico(fig)


def _grafico_curva(r, codigo):
    media=0.; sd=1.;t=(r['tipo']=='Média');crit=stats.t.ppf(.975,382) if t else stats.norm.ppf(.975)
    x=np.linspace(-4.4,4.4,650)
    y=stats.t.pdf(x,382) if t else stats.norm.pdf(x)
    fig,ax=plt.subplots(figsize=(9.2,3.25))
    ax.plot(x,y,color=AZUL,lw=2)
    ax.fill_between(x,y,where=(abs(x)>=abs(r['estatistica'])),color=LARANJA,alpha=.37,label='Região do valor-p bilateral')
    ax.axvline(-crit,color='firebrick',ls='--',lw=1.2)
    ax.axvline(crit,color='firebrick',ls='--',lw=1.2,label='Críticos ±'+br(crit,3))
    ax.axvline(r['estatistica'],color='black',lw=1.5,label='Estatística '+br(r['estatistica'],3))
    ax.set_title(codigo+' — '+('Distribuição t (382 gl)' if t else 'Distribuição normal Z'))
    ax.legend(loc='upper right',fontsize=7);ax.grid(alpha=.15)
    return _grafico(fig)


def _grafico_sensibilidade(sens):
    fig,ax=plt.subplots(figsize=(9.2,3.55))
    x=np.arange(3); w=.35
    for i,(s,color) in enumerate(zip(sens,[AZUL,LARANJA])):
        vals=[s['original'],s['sem_maximo'],s['aparada']]
        ax.bar(x+(i-.5)*w,vals,w,color=color,label=s['codigo'])
    ax.set_xticks(x,['Original','Sem máximo','Aparada 5%'])
    ax.set_ylabel('Pessoas por acidente');ax.set_title('Análise de sensibilidade das médias')
    ax.legend();ax.grid(axis='y',alpha=.2)
    return _grafico(fig)


def _grafico_holm(resultados):
    cod=list(HIP);x=np.arange(4);w=.35
    fig,ax=plt.subplots(figsize=(9.2,3.6))
    ax.bar(x-w/2,[resultados[c]['p'] for c in cod],w,label='Valor-p original',color=AZUL)
    ax.bar(x+w/2,[resultados[c]['holm'] for c in cod],w,label='Valor-p Holm',color=LARANJA)
    ax.axhline(.05,color='firebrick',ls='--',label='Alfa = 0,05')
    ax.set_xticks(x,cod);ax.set_ylabel('Valor-p');ax.legend(fontsize=8)
    ax.set_title('Testes individuais e ajuste de Holm');ax.grid(axis='y',alpha=.2)
    return _grafico(fig)


def gerar_pdf(raiz: str|Path, incluir_graficos: bool=True, incluir_metodologia: bool=True) -> bytes:
    """Retorna PDF completo em memória. Não altera, filtra ou sorteia registros."""
    pop,amo=preparar_bases(raiz)
    res=calcular_inferencia(pop,amo)
    sens=calc_sensibilidade(amo,res,len(pop))
    regular,bold=_fontes()
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='PRFTitle',fontName=bold,fontSize=23,leading=32,alignment=TA_CENTER,textColor=colors.HexColor('#173c59'),spaceAfter=20))
    styles.add(ParagraphStyle(name='PRFH1',fontName=bold,fontSize=15,leading=21,spaceBefore=14,spaceAfter=10,textColor=colors.HexColor('#173c59'),keepWithNext=True))
    styles.add(ParagraphStyle(name='PRFH2',fontName=bold,fontSize=11,leading=16,spaceBefore=10,spaceAfter=7,textColor=colors.HexColor('#245e88'),keepWithNext=True))
    styles.add(ParagraphStyle(name='PRFBody',fontName=regular,fontSize=8.6,leading=13,spaceAfter=9))
    styles.add(ParagraphStyle(name='PRFSmall',fontName=regular,fontSize=7.5,leading=11,spaceAfter=5))
    styles.add(ParagraphStyle(name='PRFTable',fontName=regular,fontSize=7,leading=9))
    styles.add(ParagraphStyle(name='PRFHead',fontName=bold,fontSize=7,leading=9,textColor=colors.white))
    styles.add(ParagraphStyle(name='PRFCenter',fontName=regular,fontSize=9,leading=13,alignment=TA_CENTER))
    buf=BytesIO()
    pdf=BaseDocTemplate(buf,pagesize=A4,rightMargin=1.65*cm,leftMargin=1.65*cm,topMargin=1.65*cm,bottomMargin=1.65*cm,
        title='PRF Analytics 2025 — Relatório consolidado',author='Projeto Probabilidade e Estatística para IA')
    frame=Frame(pdf.leftMargin,pdf.bottomMargin,pdf.width,pdf.height,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
    def rodape(canvas,doc):
        canvas.saveState()
        canvas.setFont(regular,7.2);canvas.setFillColor(colors.HexColor('#64748b'))
        canvas.drawString(1.65*cm,1.04*cm,'PRF Analytics 2025 | População, amostra e inferência')
        canvas.drawRightString(A4[0]-1.65*cm,1.04*cm,f'Página {doc.page}')
        canvas.restoreState()
    pdf.addPageTemplates(PageTemplate(id='normal',frames=[frame],onPage=rodape))
    story=[]
    def p(t,kind='PRFBody'):story.append(Paragraph(t,styles[kind]))
    def h(t):p(t,'PRFH1')
    def h2(t):p(t,'PRFH2')
    def sp(n=8):story.append(Spacer(1,n))
    def tabela(colunas, linhas, larguras=None):
        dados=[[Paragraph(escape(str(x)),styles['PRFHead']) for x in colunas]]
        for linha in linhas:
            dados.append([Paragraph(escape(str(x)),styles['PRFTable']) for x in linha])
        tab=Table(dados,colWidths=larguras,repeatRows=1,hAlign='LEFT')
        tab.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#245e88')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#edf4f8')]),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),5),
            ('TOPPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),6),
            ('RIGHTPADDING',(0,0),(-1,-1),6),
            ('LINEBELOW',(0,-1),(-1,-1),.4,colors.HexColor('#c5d3dd'))]))
        story.append(tab);sp(9)
    def imagem(obj):
        if incluir_graficos:story.append(obj);sp(10)
    def page():story.append(PageBreak())
    def por(valor,tipo):return br(valor*100,2)+'%' if tipo=='Proporção' else br(valor,3)

    sp(60)
    p('PRF ANALYTICS 2025','PRFTitle')
    p('Análise estatística dos acidentes em rodovias federais brasileiras','PRFCenter')
    sp(28)
    p('Relatório consolidado dos quatro ambientes da aplicação','PRFCenter')
    sp(25)
    tabela(['BASE / RECORTE','CARACTERÍSTICAS'],[
        ['População operacional','72.529 ocorrências de acidentes'],
        ['Amostra original','383 ocorrências, AAS sem reposição'],
        ['Variáveis','30 colunas'],
        ['Dimensionamento','95% de confiança; 5% de margem de referência; p=50%'],
        ['Estatística inferencial','2 proporções, 2 médias, 4 testes, IC 95% e ajuste de Holm'],
    ],[6*cm,11.7*cm])
    sp(24)
    p('Fonte: Polícia Rodoviária Federal (PRF) — Dados Abertos | Ano de referência: 2025','PRFCenter')
    p('Data de geração: '+date.today().strftime('%d/%m/%Y'),'PRFCenter')
    page()

    h('Sumário')
    for item in ['1. Início — contextualização, base, metodologia e auditoria',
                 '2. Dashboard 1 — análise descritiva da população',
                 '3. Dashboard 2 — análise descritiva da amostra',
                 '4. Dashboard 3 — inferência e comparação',
                 '5. Análises complementares — sensibilidade e Holm',
                 '6. Interpretação, limitações e referências']:
        p(escape(item))
    p('Nota: a estrutura reproduz o conteúdo analítico das páginas, em formato estático e acadêmico. Filtros, botões e interações permanecem disponíveis somente no Streamlit.','PRFSmall')
    h('1. Início — visão geral do estudo')
    p('Este trabalho compara parâmetros calculados diretamente em uma população operacional de acidentes da PRF de 2025 com estimativas obtidas em uma amostra aleatória simples. Cada linha da base corresponde a uma ocorrência de acidente.')
    tabela(['Indicador','Valor'],[
        ['Ocorrências na população',br(len(pop),0)],['Ocorrências na amostra',br(len(amo),0)],
        ['Fração amostral',br(100*len(amo)/len(pop),4)+'%'],['Variáveis disponíveis',str(len(pop.columns))],
        ['Nível de confiança','95%'],['Fonte','Polícia Rodoviária Federal']],[10*cm,7.7*cm])
    if incluir_metodologia:
        h2('Amostragem e seleção')
        p('Dimensionamento de referência para proporções: N=72.529, confiança de 95%, margem de erro e=0,05 e proporção conservadora p=0,50, com correção para população finita. O resultado foi arredondado para n=383. O dimensionamento de 5% não é uma garantia universal para todas as médias estudadas.')
        p('Seleção original: amostragem aleatória simples sem reposição, com semente pseudoaleatória fixa random_state=42. A amostra foi preservada em arquivo CSV; o aplicativo e este gerador não efetuam novo sorteio.')
        h2('Integridade e reprodutibilidade')
        tabela(['Verificação','Resultado'],[
            ['72.529 registros populacionais','Aprovado'],['383 registros amostrais','Aprovado'],
            ['30 colunas em cada base','Aprovado'],['Mesmas colunas','Aprovado'],
            ['IDs presentes e numéricos','Aprovado'],['IDs únicos nas duas bases','Aprovado'],
            ['Todos os IDs amostrais presentes na população','Aprovado'],
            ['Correspondência de pessoas, mortos, feridos e fase do dia','Aprovado']],[13*cm,4.7*cm])
    h2('Síntese de resultados')
    nrej=sum(res[c]['p']<ALFA for c in res)
    nholm=sum(res[c]['holm']<ALFA for c in res)
    p(f'Quatro intervalos de confiança de 95% incluíram seus parâmetros populacionais. Houve {nrej} rejeições nos testes individuais e {nholm} após o ajuste de Holm.')

    for d,titulo,idx in [(pop,'População',2),(amo,'Amostra',3)]:
        page();h(f'{idx}. Dashboard {idx-1} — análise descritiva da {titulo.lower()}')
        N=len(d);envolv=int(d.pessoas.sum());mortos=int(d.mortos.sum());feridos=int(d.feridos.sum())
        prop=float(d.mortos.gt(0).mean())
        tabela(['Indicador',titulo],[
            ['Acidentes',br(N,0)],['Pessoas envolvidas',br(envolv,0)],
            ['Vítimas fatais',br(mortos,0)],['Pessoas feridas',br(feridos,0)],
            ['Acidentes com pelo menos uma morte',br(prop*100,2)+'%']],[12.5*cm,5.2*cm])
        h2('Medidas estatísticas das variáveis quantitativas')
        metricas=[]
        for c in QUANTITATIVAS:
            s=d[c].dropna()
            metricas.append([NOMES[c],br(s.mean(),3),br(s.median(),2),br(s.std(ddof=1),3),
                br(s.min(),0),br(s.max(),0)])
        tabela(['Variável','Média','Mediana','Desvio-padrão','Mín.','Máx.'],metricas,
            [5.6*cm,2.25*cm,2.25*cm,3.1*cm,2.2*cm,2.2*cm])
        h2('Distribuições quantitativas')
        p('Gráficos de frequência percentual das quantidades por acidente. A categoria 10+ agrega valores iguais ou superiores a dez, quando necessário.')
        for c in QUANTITATIVAS:
            imagem(_grafico_quantitativo(d,c))
        h2('Boxplots e valores extremos')
        boxrows=[]
        for c in ['pessoas','feridos']:
            v=d[c].dropna();q1=float(v.quantile(.25));q3=float(v.quantile(.75));iqr=q3-q1
            out=((v<q1-1.5*iqr)|(v>q3+1.5*iqr)).sum()
            boxrows.append([NOMES[c],br(q1,2),br(v.median(),2),br(q3,2),br(iqr,2),br(out,0)])
        tabela(['Variável','Q1','Mediana','Q3','AIQ','Atípicos IQR'],boxrows,
            [5.3*cm,2.25*cm,2.45*cm,2.1*cm,2*cm,3.6*cm])
        imagem(_grafico_box(d))
        p('Os boxplots omitem apenas os pontos individuais fora dos limites para facilitar a leitura; nenhum registro é excluído dos cálculos. Quando o AIQ é zero, a regra de 1,5 × AIQ pode sinalizar grande quantidade de valores como potencialmente atípicos.','PRFSmall')
        h2('Distribuições qualitativas — frequências e proporções')
        for c in QUALITATIVAS:
            h2(NOMES[c])
            s=d[c].fillna('Não informado').astype(str).str.strip()
            counts=s.value_counts();top=counts.head(10)
            linhas=[[str(k),br(v,0),br(v/N*100,2)+'%'] for k,v in top.items()]
            tabela(['Categoria','Frequência','Proporção'],linhas,[11.2*cm,3.1*cm,3.4*cm])
            imagem(_grafico_qualitativo(d,c))
            if len(counts)>10:
                p(f'Mostradas as dez categorias mais frequentes de {len(counts)} existentes. Os percentuais usam todos os {N} registros no denominador.','PRFSmall')
        h2('Consulta dos registros e escopo')
        p(f'O painel interativo permite filtrar e consultar registros individuais. Neste PDF constam as estatísticas e distribuições não filtradas dos {br(N,0)} registros, sem replicar os controles interativos.')
        p('Prévia de dez registros (colunas selecionadas):','PRFSmall')
        preview=d[['id','uf','pessoas','mortos','feridos','fase_dia']].head(10)
        tabela(preview.columns.tolist(), preview.astype(str).values.tolist(),
            [3*cm,2*cm,2.8*cm,2.5*cm,2.5*cm,4.9*cm])

    page();h('4. Dashboard 3 — inferência e comparação')
    p('As estimativas e os testes foram calculados exclusivamente a partir da amostra original de 383 ocorrências. Os parâmetros populacionais foram calculados separadamente, para comparação posterior.')
    h2('Comparação entre parâmetros e estimativas, com IC 95%')
    linhas=[]
    for c,r in res.items():
        linhas.append([c, r['nome'],por(r['populacao'],r['tipo']),por(r['amostra'],r['tipo']),
            por(r['li'],r['tipo'])+' a '+por(r['ls'],r['tipo']), 'Sim' if r['contem'] else 'Não'])
    tabela(['Teste','Parâmetro','População','Amostra','IC 95%','Contém'],linhas,
        [1.35*cm,5.5*cm,2.35*cm,2.3*cm,4.75*cm,1.45*cm])
    imagem(_grafico_comparacao(res,['P1','P2']))
    imagem(_grafico_comparacao(res,['M1','M2']))
    h2('Testes bilaterais de hipóteses — α = 0,05')
    linhas=[]
    for c,r in res.items():
        h0=por(r['h0'],r['tipo'])
        linhas.append([c,'H0: '+('p' if r['tipo']=='Proporção' else 'μ')+' = '+h0,
            ('Z' if r['tipo']=='Proporção' else 't')+' = '+br(r['estatistica'],4),
            br(r['p'],6),'Rejeitar' if r['p']<ALFA else 'Não rejeitar'])
    tabela(['Teste','Hipótese nula','Estatística','Valor-p','Decisão H0'],linhas,
        [1.45*cm,5.65*cm,3.5*cm,2.8*cm,4.3*cm])
    p('Todas as hipóteses alternativas são bilaterais (≠). Valores de referência H0 são exploratórios; não representam metas oficiais da PRF.')
    h2('Distribuições teóricas das estatísticas de teste')
    for c,r in res.items():
        imagem(_grafico_curva(r,c))
    h2('Interpretação dos resultados')
    p('Os quatro ICs de 95% contêm os parâmetros da população observada. P1 e M2 rejeitam H0 nos testes individuais; P2 e M1 não rejeitam H0. Não rejeitar H0 não demonstra igualdade com o valor de referência.')

    page();h('5. Análises complementares')
    h2('Sensibilidade das médias a observações extremas')
    linhas=[]
    for s in sens:
        linhas.append([s['codigo'],str(s['id']),br(s['maximo'],0),br(s['original'],4),
                      br(s['sem_maximo'],4),br(s['aparada'],4),br(s['p_original'],6),('<0,000001' if s['p_sem_maximo'] < 0.000001 else br(s['p_sem_maximo'],6))])
    tabela(['Teste','ID extremo','Máximo','Média original','Sem máximo','Aparada 5%','p original','p sem máx.'],linhas,
        [1.05*cm,2.7*cm,1.6*cm,2.55*cm,2.35*cm,2.3*cm,2.55*cm,2.6*cm])
    imagem(_grafico_sensibilidade(sens))
    p('A exclusão do máximo é um exercício de sensibilidade, não uma alteração na amostra original. Os resultados principais permanecem baseados nos 383 registros.')
    h2('Ajuste de Holm para quatro testes')
    linhas=[]
    for c,r in res.items():
        linhas.append([c,br(r['p'],6),br(r['holm'],6),
            'Rejeitar' if r['p']<ALFA else 'Não rejeitar',
            'Rejeitar' if r['holm']<ALFA else 'Não rejeitar'])
    tabela(['Teste','p original','p Holm','Decisão individual','Decisão Holm'],linhas,
        [1.55*cm,3*cm,2.9*cm,5.3*cm,4.95*cm])
    imagem(_grafico_holm(res))
    p('O procedimento de Holm controla o erro familiar associado aos quatro testes simultâneos. A 5%, nenhum teste permanece rejeitado após o ajuste.')

    page();h('6. Metodologia, limitações e referências')
    if incluir_metodologia:
        h2('Correção para população finita e estatísticas utilizadas')
        fpc=np.sqrt((len(pop)-len(amo))/(len(pop)-1))
        p('FPC = raiz[(N − n)/(N − 1)] = '+br(fpc,6)+'. Para proporções foi utilizado o intervalo de Wald com aproximação normal, e teste Z bilateral calculado sob p0. Para médias, intervalos e testes t bilaterais, com desvio-padrão amostral e 382 graus de liberdade. Em todos os casos a FPC foi incorporada ao erro-padrão.')
        p('Referências de hipótese: P1 p0=10%; P2 p0=30%; M1 μ0=3; M2 μ0=1. Nível de significância α=0,05. A seleção foi por AAS sem reposição e preservação da amostra original.')
        h2('Condições e limitações')
        p('Uma única AAS é examinada. A população operacional estudada não representa necessariamente acidentes não registrados, outras rodovias ou anos. Os intervalos de Wald exigem cautela em amostras pequenas ou proporções próximas a zero ou um. A amostra apresenta valores extremos e assimetria para médias, o que recomenda interpretação cuidadosa dos procedimentos baseados em aproximações e da análise de sensibilidade.')
        p('A presença de um parâmetro populacional dentro de um IC específico não implica cobertura garantida em todas as amostras possíveis. Os quatro testes possuem referências exploratórias, não metas normativas da PRF. A regra do boxplot baseada em AIQ é apenas descritiva. Os gráficos não ajustam por exposição ao tráfego.')
        h2('Variáveis selecionadas')
        p('Quantitativas: '+', '.join(NOMES[x] for x in QUANTITATIVAS)+'. Qualitativas: '+', '.join(NOMES[x] for x in QUALITATIVAS)+'.')
    h2('Conclusão')
    p('A amostra aleatória simples de 383 ocorrências permitiu produzir estimativas para uma população operacional de 72.529 acidentes. Os quatro ICs cobriram os valores populacionais conhecidos, mas estimativas e parâmetros não foram idênticos. A análise de valores extremos e o ajuste de Holm evidenciaram a importância de considerar a variabilidade, os pressupostos e a multiplicidade dos testes na comunicação estatística.')
    h2('Referências e reprodutibilidade')
    for s in [
       'Polícia Rodoviária Federal — Dados Abertos. https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf',
       'PRF Analytics 2025 — Repositório. https://github.com/mendes31/prf-analytics-2025',
       'PRF Analytics 2025 — Dashboards. https://prf-analytics-2025.streamlit.app',
       'Arquivos: datatran2025.csv e amostra_PRF_2025.csv.']:
        p(escape(s),'PRFSmall')
    pdf.build(story)
    return buf.getvalue()


if __name__=='__main__':
    raiz=Path(__file__).resolve().parents[1]
    dest=raiz/'Relatorio_Consolidado_PRF_2025.pdf'
    dest.write_bytes(gerar_pdf(raiz))
    print(f'Gerado: {dest} ({dest.stat().st_size} bytes)')
