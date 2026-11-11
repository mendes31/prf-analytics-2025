
"""Exportação PDF fiel ao PRF Analytics 2025; não filtra nem sorteia dados."""

from __future__ import annotations
from datetime import date
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image,
    KeepTogether, PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

N_POP, N_AMO, ALFA = 72529, 383, .05
AZUL, LARANJA = '#2474ad', '#e99134'
NUMERICAS = ['pessoas', 'mortos', 'feridos', 'feridos_leves', 'feridos_graves', 'veiculos']
GRAFICOS_NUM = [('pessoas', 10), ('mortos', 4), ('feridos', 6)]
QUALITATIVAS = ['classificacao_acidente', 'fase_dia', 'tipo_acidente',
                'causa_acidente', 'uf', 'condicao_metereologica']
TOP10 = {'tipo_acidente', 'causa_acidente', 'uf'}
NOME = {'pessoas':'Pessoas envolvidas','mortos':'Vítimas fatais',
        'feridos':'Pessoas feridas','feridos_leves':'Feridos leves',
        'feridos_graves':'Feridos graves','veiculos':'Veículos',
        'classificacao_acidente':'Classificação do acidente',
        'fase_dia':'Fase do dia','tipo_acidente':'Tipo de acidente',
        'causa_acidente':'Causa do acidente','uf':'Unidade federativa',
        'condicao_metereologica':'Condição meteorológica'}
HIP = {'P1':('Acidentes com pelo menos uma morte','Proporção',.10),
       'P2':('Acidentes em plena noite','Proporção',.30),
       'M1':('Pessoas envolvidas por acidente','Média',3.),
       'M2':('Pessoas feridas por acidente','Média',1.)}


def br(valor, casas=2):
    if pd.isna(valor): return '—'
    return f'{float(valor):,.{casas}f}'.replace(',', 'X').replace('.', ',').replace('X', '.')


def fontes():
    raiz=Path('/usr/share/fonts/truetype/dejavu')
    reg, bold=raiz/'DejaVuSans.ttf', raiz/'DejaVuSans-Bold.ttf'
    if reg.exists():
        if 'DejaVu' not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont('DejaVu',str(reg)))
            pdfmetrics.registerFont(TTFont('DejaVu-Bold',str(bold)))
        return 'DejaVu', 'DejaVu-Bold'
    return 'Helvetica', 'Helvetica-Bold'


def preparar_bases(raiz):
    raiz=Path(raiz)
    pop=pd.read_csv(raiz/'datatran2025.csv',sep=';',encoding='latin1',low_memory=False)
    pop.columns=pop.columns.str.strip()
    amo=None
    for enc in ('utf-8-sig','latin1'):
        for sep in (',',';','\t'):
            try:
                x=pd.read_csv(raiz/'amostra_PRF_2025.csv',sep=sep,encoding=enc,low_memory=False)
                x.columns=x.columns.str.strip()
                if x.shape==(N_AMO,30) and 'id' in x: amo=x;break
            except (UnicodeError,pd.errors.ParserError): pass
        if amo is not None: break
    if amo is None or pop.shape!=(N_POP,30):
        raise ValueError('Dimensões ou formato dos CSVs inválidos.')
    if set(pop.columns)!=set(amo.columns):
        raise ValueError('As colunas dos arquivos são diferentes.')
    for d in (pop,amo):
        d['id']=pd.to_numeric(d['id'],errors='raise').astype('int64')
        for c in NUMERICAS: d[c]=pd.to_numeric(d[c],errors='raise')
        d['fase_dia']=d['fase_dia'].astype('string').str.strip()
    if pop.id.duplicated().any() or amo.id.duplicated().any() or not amo.id.isin(pop.id).all():
        raise ValueError('IDs duplicados ou amostra não pertencente à população.')
    campos=['pessoas','mortos','feridos','fase_dia']
    origem=pop.set_index('id').loc[amo.id,campos].reset_index(drop=True)
    atual=amo[campos].reset_index(drop=True)
    if not all(origem[c].equals(atual[c]) for c in campos):
        raise ValueError('Valores da amostra divergentes da população.')
    return pop,amo


def vetor(d,c):
    if c=='P1': return d.mortos.gt(0).to_numpy(dtype=float)
    if c=='P2': return d.fase_dia.eq('Plena Noite').fillna(False).to_numpy(dtype=float)
    return d['pessoas' if c=='M1' else 'feridos'].to_numpy(dtype=float)


def inferencia(pop,amo):
    N,n=len(pop),len(amo); fpc=np.sqrt((N-n)/(N-1)); r={}
    for c,(nome,tipo,h0) in HIP.items():
        v=vetor(amo,c); a=float(v.mean()); p=float(vetor(pop,c).mean())
        if tipo=='Proporção':
            se=np.sqrt(a*(1-a)/n)*fpc; crit=stats.norm.ppf(.975)
            ep=np.sqrt(h0*(1-h0)/n)*fpc
            t=(a-h0)/ep; pv=float(2*stats.norm.sf(abs(t)))
            li=max(0.,a-crit*se);ls=min(1.,a+crit*se)
        else:
            se=float(v.std(ddof=1)/np.sqrt(n)*fpc);crit=stats.t.ppf(.975,n-1)
            li=a-crit*se;ls=a+crit*se
            t=(a-h0)/se;pv=float(2*stats.t.sf(abs(t),n-1))
        r[c]=dict(codigo=c,nome=nome,tipo=tipo,h0=h0,pop=p,amo=a,li=li,ls=ls,
                   estat=float(t),pv=pv,cobre=li<=p<=ls,
                   ep_ic=float(se),crit=float(crit),margem=float(crit*se),
                   desvio=float(v.std(ddof=1)) if tipo=='Média' else None,
                   sucessos=int(v.sum()) if tipo=='Proporção' else None)
    p=np.array([r[c]['pv'] for c in HIP]);ordem=np.argsort(p);ajust=np.zeros(4);prev=0.
    for i,j in enumerate(ordem):
        prev=max(prev,min(1.,(4-i)*p[j]));ajust[j]=prev
    for i,c in enumerate(HIP):r[c]['holm']=float(ajust[i])
    return r


def sensibilidade(amo,r,N):
    saida=[]
    for c,col in [('M1','pessoas'),('M2','feridos')]:
        v=amo[col].to_numpy(dtype=float); pos=int(v.argmax()); v2=np.delete(v,pos)
        m=float(v2.mean());fpc=np.sqrt((N-len(v2))/(N-1))
        t=(m-HIP[c][2])/(v2.std(ddof=1)/np.sqrt(len(v2))*fpc)
        saida.append(dict(c=c,id=int(amo.iloc[pos].id),max=v[pos],orig=r[c]['amo'],
                          sem=m,apar=float(stats.trim_mean(v,.05)),
                          po=r[c]['pv'],ps=float(2*stats.t.sf(abs(t),len(v2)-1))))
    return saida


def figura(fig):
    bio=BytesIO()
    fig.savefig(bio,format='png',dpi=140,facecolor='white',bbox_inches='tight')
    plt.close(fig);bio.seek(0)
    im=PILImage.open(bio);w=16.1*cm;h=w*im.height/im.width
    if h>12.5*cm:h=12.5*cm;w=h*im.width/im.height
    bio.seek(0)
    return Image(bio,width=w,height=h)


def freq_numerica(d,col,lim):
    s=d[col].dropna()
    faixas=[str(i) for i in range(lim)]+[f'{lim} ou mais']
    cont=s.apply(lambda x: str(int(x)) if x<lim else f'{lim} ou mais').value_counts()
    return pd.DataFrame({'Categoria':faixas,'Frequência':[int(cont.get(x,0)) for x in faixas],
                         'Percentual':[100*int(cont.get(x,0))/len(d) for x in faixas]})


def plot_numerica(d,col,lim):
    df=freq_numerica(d,col,lim);fig,ax=plt.subplots(figsize=(9,3.1))
    x=np.arange(len(df));y=df.Percentual.to_numpy()
    ax.bar(x,y,color=AZUL,width=.75)
    ax.set_xticks(x,df.Categoria,rotation=28,ha='right',fontsize=8)
    ax.set_ylabel('% dos acidentes');ax.set_xlabel('Quantidade por acidente')
    ax.set_title(NOME[col],pad=12);ax.grid(axis='y',alpha=.17)
    ax.set_ylim(0,max(1,y.max()*1.18));fig.tight_layout(pad=1.5)
    return figura(fig)


def freq_categorica(d,col,ordem):
    s=d[col].fillna('Não informado').astype(str).str.strip().replace('','Não informado')
    freqs=s.value_counts().reindex(ordem,fill_value=0)
    return [(str(k),int(v),float(100*v/len(d))) for k,v in freqs.items()]


def plot_categorica(linhas,col):
    fig,ax=plt.subplots(figsize=(9,max(3.2,min(4.5,.39*len(linhas)+.6))))
    linhas=sorted(linhas,key=lambda a:a[2]);labels=[x[0][:50] for x in linhas]
    ax.barh(labels,[x[2] for x in linhas],color=AZUL)
    ax.set_xlabel('% dos acidentes');ax.set_title(NOME[col],pad=11)
    ax.grid(axis='x',alpha=.16);fig.subplots_adjust(left=.34,right=.98,top=.88,bottom=.17)
    return figura(fig)


def resumo_box(d,c):
    s=d[c].dropna();q1=float(s.quantile(.25));med=float(s.median());q3=float(s.quantile(.75))
    aiq=q3-q1;lo=q1-1.5*aiq;hi=q3+1.5*aiq
    ok=s[(s>=lo)&(s<=hi)]
    return dict(q1=q1,med=med,q3=q3,aiq=aiq,lo=float(ok.min()),hi=float(ok.max()),
                atip=int(((s<lo)|(s>hi)).sum()))


def plot_box(d):
    fig,axes=plt.subplots(1,2,figsize=(9,3.3))
    for ax,c in zip(axes,['pessoas','feridos']):
        r=resumo_box(d,c)
        ax.bxp([dict(label=NOME[c],q1=r['q1'],med=r['med'],q3=r['q3'],
                     whislo=r['lo'],whishi=r['hi'],fliers=[])],showfliers=False,
               patch_artist=True,boxprops=dict(facecolor=AZUL,alpha=.30),
               medianprops=dict(color=LARANJA,linewidth=2))
        ax.set_title(NOME[c],pad=14,fontsize=10)
        ax.set_ylim(r['lo']-max(.2,(r['hi']-r['lo'])*.25),
                    r['hi']+max(.2,(r['hi']-r['lo'])*.25))
        ax.grid(axis='y',alpha=.18)
    fig.subplots_adjust(top=.82,bottom=.17,wspace=.4)
    return figura(fig)


def plot_comparacao(r,lista):
    fig,ax=plt.subplots(figsize=(9,3.3)); x=np.arange(2);w=.32
    pop=np.array([r[c]['pop'] for c in lista]);amo=np.array([r[c]['amo'] for c in lista])
    mul=100 if lista[0].startswith('P') else 1
    erro=np.array([[r[c]['amo']-r[c]['li'] for c in lista],
                   [r[c]['ls']-r[c]['amo'] for c in lista]])*mul
    barras_pop=ax.bar(x-w/2,pop*mul,w,color=AZUL,label='População')
    barras_amo=ax.bar(x+w/2,amo*mul,w,color=LARANJA,yerr=erro,capsize=5,
                      label='Amostra (IC 95%)',error_kw={'ecolor':'#7c2d12','lw':1.5})
    for barras,interno in ((barras_pop,False),(barras_amo,True)):
        for barra in barras:
            altura=barra.get_height()
            ax.text(barra.get_x()+barra.get_width()/2,
                    altura*.55 if interno else altura+max(erro.flatten())*.08,
                    br(altura,2 if mul==100 else 3)+('%' if mul==100 else ''),
                    ha='center',va='center' if interno else 'bottom',fontsize=7,
                    color='white' if interno else '#245e88',fontweight='bold')
    ax.set_xticks(x,lista);ax.set_ylabel('%' if mul==100 else 'Pessoas/acidente')
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.16)
    ax.set_title('Comparação população × amostra',pad=12);fig.tight_layout(pad=1.4)
    return figura(fig)


def plot_ic_individual(c, r):
    """IC ampliado, com limites, estimativa e parâmetro identificados."""
    prop = r['tipo']=='Proporção'; fator=100 if prop else 1
    li, ls, est, pop = [r[k]*fator for k in ('li','ls','amo','pop')]
    faixa=max(ls-li, .05); folga=max(faixa*.13,.35 if prop else .035)
    fig, ax=plt.subplots(figsize=(9,2.5))
    ax.hlines(0,li,ls,color=AZUL,lw=4,zorder=2)
    ax.plot([li,li],[ -.06,.06],color=AZUL,lw=2)
    ax.plot([ls,ls],[ -.06,.06],color=AZUL,lw=2)
    ax.scatter([est],[0],s=85,color=AZUL,edgecolor='white',zorder=4)
    ax.scatter([pop],[0],s=95,marker='D',color=LARANJA,edgecolor='white',zorder=4)
    fmt=lambda a:br(a,2)+('%' if prop else '')
    ax.annotate('Amostra '+fmt(est),(est,0),xytext=(0,20),textcoords='offset points',
                ha='center',color=AZUL,fontsize=8,fontweight='bold')
    ax.annotate('População '+fmt(pop),(pop,0),xytext=(0,-26),textcoords='offset points',
                ha='center',color=LARANJA,fontsize=8,fontweight='bold')
    ax.annotate('LI '+fmt(li),(li,0),xytext=(0,-36),textcoords='offset points',
                ha='center',color=AZUL,fontsize=7)
    ax.annotate('LS '+fmt(ls),(ls,0),xytext=(0,-36),textcoords='offset points',
                ha='center',color=AZUL,fontsize=7)
    ax.set_xlim(max(0,li-folga),ls+folga);ax.set_ylim(-.50,.48)
    ax.set_yticks([]);ax.set_xlabel('Proporção (%)' if prop else 'Pessoas por acidente',fontsize=8)
    ax.grid(axis='x',alpha=.16)
    for s in ('left','top','right'):ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_alpha(.25)
    fig.tight_layout(pad=1.1)
    return figura(fig)


def plot_holm_ampliado(r):
    """Detalhe dos dois valores-p que mudam após Holm."""
    codigos=['P1','M2'];fig,ax=plt.subplots(figsize=(9,2.6))
    x=np.arange(2);w=.32
    barras_1=ax.bar(x-w/2,[r[c]['pv'] for c in codigos],w,color=AZUL,label='Original')
    barras_2=ax.bar(x+w/2,[r[c]['holm'] for c in codigos],w,color=LARANJA,label='Holm')
    for barras in (barras_1,barras_2):
        for barra in barras:
            ax.text(barra.get_x()+barra.get_width()/2,barra.get_height()+.0008,
                    br(barra.get_height(),6),fontsize=7,ha='center')
    ax.axhline(.05,color='firebrick',ls='--',lw=1,label='Limite 5%')
    ax.set_ylim(0,.062);ax.set_xticks(x,codigos);ax.set_ylabel('Valor-p')
    ax.legend(ncol=3,fontsize=7,loc='upper left');ax.grid(axis='y',alpha=.12)
    fig.tight_layout(pad=1.2)
    return figura(fig)


def plot_teste(c,r):
    t=(r['tipo']=='Média');crit=stats.t.ppf(.975,N_AMO-1) if t else stats.norm.ppf(.975)
    x=np.linspace(-4.5,4.5,700);y=stats.t.pdf(x,N_AMO-1) if t else stats.norm.pdf(x)
    fig,ax=plt.subplots(figsize=(9,3.15));ax.plot(x,y,color=AZUL,lw=2)
    ax.fill_between(x,y,where=abs(x)>=abs(r['estat']),alpha=.34,color=LARANJA,
                    label='Valor-p bilateral')
    for s in (-crit,crit):
        ax.axvline(s,color='firebrick',ls='--',lw=1.1)
        ax.text(s,.34,('−' if s<0 else '+')+br(abs(s),3),
                color='firebrick',fontsize=7,ha='center')
    ax.axvline(r['estat'],color='#17846A',lw=1.5,label='Observado = '+br(r['estat'],3))
    ax.set_title(c+' — '+('t de Student (382 gl)' if t else 'Normal Z'),pad=12)
    ax.legend(loc='upper right',fontsize=7);ax.grid(alpha=.13);fig.tight_layout(pad=1.5)
    return figura(fig)


def plot_sens(s):
    fig,ax=plt.subplots(figsize=(9,3.3));x=np.arange(3);w=.35
    for i,(v,cor) in enumerate(zip(s,[AZUL,LARANJA])):
        ax.bar(x+(i-.5)*w,[v['orig'],v['sem'],v['apar']],w,color=cor,label=v['c'])
    ax.set_xticks(x,['Original','Sem máximo','Aparada 5%'])
    ax.set_ylabel('Pessoas por acidente');ax.legend();ax.grid(axis='y',alpha=.16)
    fig.tight_layout(pad=1.4);return figura(fig)


def plot_holm(r):
    fig,ax=plt.subplots(figsize=(9,3.3));x=np.arange(4);w=.33;lista=list(HIP)
    ax.bar(x-w/2,[r[c]['pv'] for c in lista],w,label='Original',color=AZUL)
    ax.bar(x+w/2,[r[c]['holm'] for c in lista],w,label='Holm',color=LARANJA)
    ax.axhline(ALFA,color='firebrick',ls='--',label='α = 5%')
    ax.set_xticks(x,lista);ax.set_ylabel('Valor-p');ax.legend(fontsize=8)
    ax.grid(axis='y',alpha=.17);fig.tight_layout(pad=1.4);return figura(fig)


def gerar_pdf(raiz: str|Path,incluir_graficos:bool=True,incluir_metodologia:bool=True)->bytes:
    pop,amo=preparar_bases(raiz);r=inferencia(pop,amo);sens=sensibilidade(amo,r,len(pop))
    reg,bold=fontes();style=getSampleStyleSheet()
    for nome,fonte,tamanho,leading,cor,antes,depois,centro in [
        ('Titulo',bold,22,30,'#173c59',0,18,True),
        ('H1',bold,14,19,'#173c59',11,9,False),
        ('H2',bold,10.5,15,'#245e88',9,7,False),
        ('Corpo',reg,8.4,12.5,'#222222',0,8,False),
        ('Pequeno',reg,7.2,10.5,'#555555',0,5,False),
        ('Tabela',reg,6.8,9,'#111111',0,0,False),
        ('Cabecalho',bold,6.8,9,'#ffffff',0,0,False),
    ]:
        style.add(ParagraphStyle(name='PRF'+nome,fontName=fonte,fontSize=tamanho,
            leading=leading,textColor=colors.HexColor(cor),spaceBefore=antes,
            spaceAfter=depois,alignment=TA_CENTER if centro else 0,
            keepWithNext=nome in ('H1','H2')))
    buf=BytesIO();pdf=BaseDocTemplate(buf,pagesize=A4,leftMargin=1.6*cm,rightMargin=1.6*cm,
        topMargin=1.6*cm,bottomMargin=1.55*cm,title='PRF Analytics 2025 — Relatório consolidado')
    frame=Frame(pdf.leftMargin,pdf.bottomMargin,pdf.width,pdf.height,
        leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
    def rodape(canvas,doc):
        canvas.saveState();canvas.setFont(reg,7)
        canvas.setFillColor(colors.HexColor('#64748b'))
        canvas.drawString(1.6*cm,1.02*cm,'PRF Analytics 2025 | Relatório consolidado')
        canvas.drawRightString(A4[0]-1.6*cm,1.02*cm,f'Página {doc.page}')
        canvas.restoreState()
    pdf.addPageTemplates(PageTemplate(id='PRF',frames=[frame],onPage=rodape))
    S=[]
    def p(texto,tipo='Corpo'):S.append(Paragraph(str(texto),style['PRF'+tipo]))
    def h1(t):p(escape(t),'H1')
    def h2(t):p(escape(t),'H2')
    def esp(n=7):S.append(Spacer(1,n))
    def tabela(cols,linhas,larguras=None,indivisivel=False):
        conteudo=[[Paragraph(escape(str(x)),style['PRFCabecalho']) for x in cols]]
        conteudo += [[Paragraph(escape(str(v)),style['PRFTabela']) for v in row] for row in linhas]
        t=Table(conteudo,colWidths=larguras,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#245e88')),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#edf4f8')]),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),4.4),
            ('BOTTOMPADDING',(0,0),(-1,-1),4.4),('LEFTPADDING',(0,0),(-1,-1),5),
            ('RIGHTPADDING',(0,0),(-1,-1),5)]))
        S.append(KeepTogether([t,Spacer(1,7)]) if indivisivel else t)
        if not indivisivel:esp(7)
    def imagem(graph):
        if incluir_graficos:S.append(graph);esp(8)
    def valor(v,tipo):return br(v*100,2)+'%' if tipo=='Proporção' else br(v,3)

    # CAPA
    esp(70);p('PRF ANALYTICS 2025','Titulo');esp(16)
    p('Análise estatística dos acidentes em rodovias federais brasileiras','Titulo')
    esp(22);p('Relatório consolidado: Início, População, Amostra e Inferência')
    esp(22)
    tabela(['BASE / RECORTE','CARACTERÍSTICAS'],[
        ['População operacional','72.529 acidentes'],['Amostra original','383 ocorrências — AAS sem reposição'],
        ['Variáveis disponíveis','30'],['Dimensionamento','95% de confiança; 5% de margem; p = 50%'],
        ['Inferência','2 proporções, 2 médias, 4 testes, IC 95% e Holm']],[6*cm,11.8*cm])
    esp(22);p('Fonte: Polícia Rodoviária Federal — Dados Abertos | 2025')
    p('Gerado em '+date.today().strftime('%d/%m/%Y'));S.append(PageBreak())

    # INICIO
    h1('Sumário')
    for item in ['1. Início e metodologia','2. Dashboard 1 — População',
                 '3. Dashboard 2 — Amostra','4. Dashboard 3 — Inferência',
                 '5. Sensibilidade e Holm','6. Limitações, conclusão e referências']:
        p(escape(item))
    p('Os resultados correspondem aos conjuntos completos, sem filtros. A interatividade permanece no Streamlit.','Pequeno')
    h1('1. Início — visão geral')
    tabela(['Indicador','Valor'],[
        ['População',br(len(pop),0)],['Amostra',br(len(amo),0)],
        ['Fração amostral',br(100*len(amo)/len(pop),4)+'%'],
        ['Variáveis','30'],['Nível de confiança','95%']],[11*cm,6.8*cm])
    if incluir_metodologia:
        h2('Amostragem e seleção')
        p('O dimensionamento considerou N=72.529, confiança de 95%, margem de erro de 5% para proporções e p=50%, com correção para população finita. AAS sem reposição e semente random_state=42; a amostra original foi preservada. A margem de 5% não é garantia para todas as médias.')
        h2('Auditoria computacional')
        tabela(['Verificação','Resultado'],[
            ['72.529 registros e 30 colunas na população','Aprovado'],
            ['383 registros e 30 colunas na amostra','Aprovado'],
            ['IDs únicos e pertencentes à população','Aprovado'],
            ['Pessoas, mortos, feridos e fase do dia conferidos','Aprovado']],[13*cm,4.8*cm])
    h2('Síntese dos resultados')
    p(f"Os quatro intervalos incluem os parâmetros populacionais. Testes individuais: {sum(x['pv']<ALFA for x in r.values())} rejeições; após Holm: {sum(x['holm']<ALFA for x in r.values())} rejeições.")

    # POPULACAO E AMOSTRA
    # CondPageBreak evita paginas isoladas por quebras forcadas.
    for d,tipo,numero in [(pop,'população',2),(amo,'amostra',3)]:
        S.append(CondPageBreak(10*cm));h1(f'{numero}. Dashboard {numero-1} — análise descritiva da {tipo}')
        total=len(d);mort=d.mortos.gt(0).sum()
        tabela(['Indicador',tipo.title()],[
            ['Acidentes',br(total,0)],['Pessoas envolvidas',br(d.pessoas.sum(),0)],
            ['Vítimas fatais',br(d.mortos.sum(),0)],['Pessoas feridas',br(d.feridos.sum(),0)],
            ['Veículos envolvidos',br(d.veiculos.sum(),0)],
            ['Acidentes com mortes',br(mort,0)],['Proporção com mortes',br(100*mort/total,2)+'%'],
            ['Acidentes com feridos',br(d.feridos.gt(0).sum(),0)]],[11.5*cm,6.3*cm])
        h2('Medidas estatísticas das variáveis quantitativas')
        ddof=0 if tipo=='população' else 1
        tabela(['Variável','Média','Mediana','D.P.','Mín.','Máx.'],[
            [NOME[c],br(d[c].mean(),3),br(d[c].median(),2),br(d[c].std(ddof=ddof),3),
             br(d[c].min(),0),br(d[c].max(),0)] for c in NUMERICAS],
            [5.5*cm,2.4*cm,2.5*cm,2.8*cm,2.3*cm,2.3*cm])
        p('Desvio-padrão '+('populacional (ddof=0).' if ddof==0 else 'amostral (ddof=1).'),'Pequeno')
        h2('Distribuições quantitativas')
        p('Frequências dos gráficos existentes no aplicativo: pessoas, vítimas fatais e pessoas feridas. Faixas finais 10+, 4+ e 6+.')
        for c,lim in GRAFICOS_NUM:imagem(plot_numerica(d,c,lim))
        h2('Boxplots e valores extremos')
        tabela(['Variável','Q1','Mediana','Q3','AIQ','Big. inf.','Big. sup.','Atípicos'],[
            [NOME[c],br(z['q1'],2),br(z['med'],2),br(z['q3'],2),br(z['aiq'],2),
             br(z['lo'],2),br(z['hi'],2),br(z['atip'],0)]
            for c in ['pessoas','feridos'] for z in [resumo_box(d,c)]],
            [3.4*cm,1.6*cm,2.3*cm,1.5*cm,1.5*cm,2.5*cm,2.5*cm,2.5*cm])
        imagem(plot_box(d))
        p('Os pontos atípicos não foram desenhados individualmente para preservar a legibilidade; todos permanecem nos cálculos.','Pequeno')
        if tipo=='população':
            p('A média de pessoas (2,597) supera a mediana (2); 40,4% das ocorrências envolvem duas pessoas. A variável feridos possui AIQ igual a zero, limitando a interpretação dos atípicos pelo critério de 1,5 × AIQ.')
        else:
            p('Na amostra, a média de pessoas foi 2,744, contra 2,597 na população; 43,6% das ocorrências envolveram duas pessoas, contra 40,4% na população. As diferenças ilustram variabilidade amostral.')
        h2('Distribuições qualitativas — frequências e proporções')
        for c in QUALITATIVAS:
            # Descritivo qualitativo: preservar rigorosamente as categorias
            # existentes no CSV, sem imputacao ou harmonizacao.
            h2(NOME[c])
            base = (pop[c].fillna('Não informado').astype(str).str.strip()
                    .replace('', 'Não informado'))
            ordem = base.value_counts().index.tolist()
            if c in TOP10:
                ordem = ordem[:10]
            linhas = freq_categorica(d, c, ordem)
            tabela(['Categoria', 'Frequência', 'Proporção'], [
                [nome, br(qtd, 0), br(proporcao, 2) + '%']
                for nome, qtd, proporcao in linhas
            ], [11.2*cm, 3.1*cm, 3.5*cm])
            if c == 'classificacao_acidente' and tipo == 'população':
                p('<b>Nota metodológica — duas variáveis distintas:</b> a categoria '
                  'original <b>“Com Vítimas Fatais” tem 5.209 registros</b>, '
                  'e há <b>1 registro “Não informado”</b> (ID 652519, com 1 morte). '
                  'O indicador P1 contabiliza <b>todos os acidentes com mortos &gt; 0</b>: '
                  '<b>5.210 acidentes (7,18%)</b>, independentemente da classificação. '
                  'Nenhum registro foi reclassificado ou excluído.', 'Pequeno')
            imagem(plot_categorica(linhas, c))
            if c in TOP10:p('Dez categorias definidas pela frequência na população; percentuais calculados sobre todos os registros.','Pequeno')
        h2('Consulta dos registros e escopo')
        p('O PDF utiliza o conjunto completo sem filtros. Prévia de dez registros; consulta interativa disponível no Streamlit.','Pequeno')
        preview=d[['id','uf','pessoas','mortos','feridos','fase_dia']].head(10)
        # Evita deixar apenas os ultimos registros em uma nova pagina.
        S.append(CondPageBreak(7.5*cm))
        tabela(preview.columns.tolist(),preview.astype(str).values.tolist(),
            [3*cm,2*cm,2.8*cm,2.5*cm,2.5*cm,5*cm],indivisivel=True)

    # INFERENCIA
    S.append(CondPageBreak(9*cm));h1('4. Dashboard 3 — inferência e comparação')
    p('Somente a amostra original de 383 registros foi utilizada na inferência. Os parâmetros populacionais foram calculados separadamente.')
    h2('Estimativas e intervalos de confiança de 95%')
    tabela(['Teste','Indicador','População','Amostra','IC 95%','Contém'],[
        [c,x['nome'],valor(x['pop'],x['tipo']),valor(x['amo'],x['tipo']),
         valor(x['li'],x['tipo'])+' a '+valor(x['ls'],x['tipo']),
         'Sim' if x['cobre'] else 'Não'] for c,x in r.items()],
        [1.4*cm,5.2*cm,2.5*cm,2.3*cm,4.8*cm,1.6*cm])
    h2('Intervalos individuais — valores e memória de cálculo')
    p('Os quatro gráficos usam escalas próprias; azul: estimativa e IC amostral; laranja: parâmetro conhecido na população. Os limites foram calculados sem utilizar H0.','Pequeno')
    for c,x in r.items():
        S.append(CondPageBreak(8.5*cm))
        h2(c+' — '+x['nome'])
        imagem(plot_ic_individual(c,x))
        fator=100 if x['tipo']=='Proporção' else 1
        unidade='pontos percentuais' if fator==100 else 'pessoas por acidente'
        if fator==100:
            operacao=('p estimada = '+str(x['sucessos'])+'/'+str(len(amo))+' = '+br(x['amo']*100,4)+'%; '
                      'EP = raiz[p estimada × (1−p estimada)/n] × FPC = '+br(x['ep_ic'],8))
            formula='IC 95% = p estimada ± z(0,975) × EP; z = '+br(x['crit'],6)
        else:
            operacao=('Média = '+br(x['amo'],6)+'; desvio amostral = '+br(x['desvio'],6)+
                      '; EP = s/raiz(n) × FPC = '+br(x['ep_ic'],8))
            formula='IC 95% = média ± t(0,975;382) × EP; t = '+br(x['crit'],6)
        p(escape(operacao),'Pequeno');p(escape(formula),'Pequeno')
        tabela(['Estimativa','Margem (±)','Limite inferior','Limite superior','População'],[
            [valor(x['amo'],x['tipo']),br(x['margem']*fator,4),
             valor(x['li'],x['tipo']),valor(x['ls'],x['tipo']),valor(x['pop'],x['tipo'])]],
             [3.4*cm,3.0*cm,3.7*cm,3.7*cm,4.0*cm])
        diferenca=(x['amo']-x['pop'])*fator
        p('Interpretação: diferença amostra − população = '+br(diferenca,3)+' '+unidade+
          '. O parâmetro populacional '+('está dentro' if x['cobre'] else 'está fora')+
          ' do IC de 95%.','Pequeno')
    h2('Visualização conjunta — comparação população × amostra')
    imagem(plot_comparacao(r,['P1','P2']));imagem(plot_comparacao(r,['M1','M2']))
    p('Os quatro intervalos incluem os valores populacionais conhecidos; a coincidência nesta amostra não garante cobertura em outras seleções.')
    h2('Testes de hipóteses — bilaterais, α = 5%')
    tabela(['Teste','Hipótese H0','Estatística','Valor-p','Decisão'],[
        [c,('p' if x['tipo']=='Proporção' else 'μ')+' = '+valor(x['h0'],x['tipo']),
         ('Z' if x['tipo']=='Proporção' else 't')+' = '+br(x['estat'],4),br(x['pv'],6),
         'Rejeitar H0' if x['pv']<ALFA else 'Não rejeitar H0'] for c,x in r.items()],
        [1.7*cm,4.3*cm,3.3*cm,2.6*cm,5.9*cm])
    p('P1 e M2 foram rejeitados nos testes individuais; P2 e M1 não. Não rejeitar H0 não demonstra que o parâmetro seja igual à referência.')
    h2('Curvas das estatísticas e interpretação dos quatro testes')
    for c,x in r.items():
        S.append(CondPageBreak(7*cm))
        imagem(plot_teste(c,x))
        crit=stats.norm.ppf(.975) if x['tipo']=='Proporção' else stats.t.ppf(.975,len(amo)-1)
        ref=valor(x['h0'],x['tipo'])
        dist='Z' if x['tipo']=='Proporção' else 't'
        rel='ultrapassa' if abs(x['estat'])>crit else 'não ultrapassa'
        conclusao='Rejeitar H0' if x['pv']<ALFA else 'Não rejeitar H0'
        p(escape(c+': H0 = '+ref+'; '+dist+' observado = '+br(x['estat'],4)+
                 '; limites críticos = ±'+br(crit,4)+'; valor-p bilateral = '+br(x['pv'],6)+
                 '. A estatística '+rel+' o limite crítico. '+conclusao+
                 ' no teste individual (α = 5%).'),'Pequeno')
    p('O valor-p não é a probabilidade de H0 ser verdadeira. A decisão individual difere da decisão após Holm.','Pequeno')

    # SENSIBILIDADE E HOLM
    h1('5. Análises complementares')
    h2('Sensibilidade das médias')
    tabela(['Teste','ID extremo','Máx.','Original','Sem máx.','Aparada 5%','p orig.','p sem máx.'],[
        [x['c'],str(x['id']),br(x['max'],0),br(x['orig'],4),br(x['sem'],4),
         br(x['apar'],4),br(x['po'],6),'<0,000001' if x['ps']<1e-6 else br(x['ps'],6)]
        for x in sens],[1.5*cm,2.7*cm,1.6*cm,2.2*cm,2.3*cm,2.5*cm,2.5*cm,2.5*cm])
    imagem(plot_sens(sens))
    p('A ocorrência extrema ID 716289 tem 76 pessoas envolvidas e 38 feridas. A exclusão foi simulada apenas para diagnóstico: os resultados oficiais continuam usando os 383 acidentes.')
    for x in sens:
        reducao=100*(x['orig']-x['sem'])/x['orig']
        p(x['c']+': redução de '+br(reducao,2)+'% ('+br(x['orig']-x['sem'],4)+
          ' pessoa(s) por acidente); média original '+br(x['orig'],4)+
          ', média sem o máximo '+br(x['sem'],4)+'.','Pequeno')
    p('A média aparada de 5% elimina aproximadamente 5% dos valores em cada extremidade, apenas para descrição. Em M1, a decisão individual mudaria após a exclusão; em M2, permaneceria. Não são testes confirmatórios.','Pequeno')
    h2('Comparações múltiplas — ajuste de Holm')
    tabela(['Teste','p original','p Holm','Decisão individual','Decisão Holm'],[
        [c,br(x['pv'],6),br(x['holm'],6),
         'Rejeitar' if x['pv']<ALFA else 'Não rejeitar',
         'Rejeitar' if x['holm']<ALFA else 'Não rejeitar'] for c,x in r.items()],
        [1.5*cm,3*cm,2.9*cm,5.2*cm,5.2*cm])
    imagem(plot_holm(r))
    h2('Por que nenhuma hipótese foi rejeitada após Holm?')
    ordenados=sorted(r,key=lambda c:r[c]['pv'])
    m=len(ordenados)
    tabela(['Ordem','Teste','Valor-p','Limite sequencial','Situação'],[
        [str(i+1),c,br(r[c]['pv'],6),br(ALFA/(m-i),6),
         'Interromper; não rejeitar' if i==0 else 'Sem rejeição após parada']
        for i,c in enumerate(ordenados)],
        [1.5*cm,2*cm,3.2*cm,3.7*cm,7.4*cm])
    p('O menor valor-p foi M2 = '+br(r['M2']['pv'],6)+
      '. O primeiro limite de Holm é 0,05/4 = 0,012500. Como '+
      br(r['M2']['pv'],6)+' > 0,012500, o procedimento interrompe as rejeições.','Pequeno')
    imagem(plot_holm_ampliado(r))
    p('P1 e M2 tinham valores-p individuais inferiores a 0,05, mas os ajustados são '+
      br(r['P1']['holm'],6)+' e '+br(r['M2']['holm'],6)+
      ', ambos superiores a 0,05. Nenhuma H0 é rejeitada após Holm.','Pequeno')

    # METODOLOGIA E CONCLUSAO
    S.append(CondPageBreak(6*cm));h1('6. Metodologia, limitações e referências')
    if incluir_metodologia:
        fpc=np.sqrt((len(pop)-len(amo))/(len(pop)-1))
        h2('Correção para população finita e procedimentos')
        p('FPC = raiz[(N − n)/(N − 1)] = '+br(fpc,6)+'. Intervalos de Wald e testes Z para proporções; testes e intervalos t de Student para médias (382 graus de liberdade). Os valores das hipóteses são referências exploratórias, não metas oficiais da PRF.')
        p('IC de proporção = p estimada ± 1,959964 × raiz[p estimada × (1 − p estimada)/n] × FPC. IC de média = média ± t crítico × (desvio amostral/raiz(n)) × FPC.','Pequeno')
        p('Os ICs utilizam o erro-padrão da estimativa amostral; os testes Z para proporção usam o erro-padrão calculado sob H0. O intervalo de confiança avalia faixas plausíveis, enquanto o teste avalia uma referência específica.','Pequeno')
        h2('Limitações e condições')
        p('A amostra foi aleatória simples sem reposição. A inferência baseia-se em aproximações; assimetria e valores extremos podem afetar sua adequação. Os resultados referem-se à população operacional registrada em 2025, e não necessariamente a ocorrências não registradas ou outros anos.')
        p('A presença dos parâmetros nos quatro intervalos observados não garante cobertura em futuras amostras. Gráficos de frequência e boxplots são descritivos; não equivalem a taxas de risco ajustadas à exposição ao tráfego.')
    h2('Conclusão')
    p('A amostra de 383 acidentes produziu estimativas próximas, mas não idênticas, aos valores calculados para a população de 72.529 ocorrências. Os quatro intervalos incluíram os parâmetros reais; houve duas rejeições individuais e nenhuma após Holm. A sensibilidade a valores extremos reforça a necessidade de interpretar os resultados com cautela.')
    h2('Referências e reprodutibilidade')
    for linha in ['PRF — Dados Abertos: https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf',
                  'Repositório: https://github.com/mendes31/prf-analytics-2025',
                  'Dashboards: https://prf-analytics-2025.streamlit.app',
                  'Arquivos originais: datatran2025.csv e amostra_PRF_2025.csv']:
        p(escape(linha),'Pequeno')
    pdf.build(S)
    return buf.getvalue()


if __name__=='__main__':
    raiz=Path(__file__).resolve().parent
    destino=raiz/'Relatorio_Consolidado_PRF_2025.pdf'
    destino.write_bytes(gerar_pdf(raiz))
    print('Relatório gerado:',destino)
