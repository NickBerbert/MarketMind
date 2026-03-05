import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas


def gerar_relatorio_pdf(dados_acao, previsoes_ensemble, datas_previsao, detalhes_previsoes, confianca_geral):
    """
    Gera um relatório PDF formatado e profissional com as previsões
    """
    try:
        # Buffer para o PDF
        buffer = BytesIO()

        # Criar documento PDF
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=50,
        )

        # Container para os elementos
        elementos = []

        # Estilos
        estilos = getSampleStyleSheet()

        # Estilo customizado para título
        estilo_titulo = ParagraphStyle(
            'CustomTitle',
            parent=estilos['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#00d4ff'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )

        # Estilo para subtítulos
        estilo_subtitulo = ParagraphStyle(
            'CustomHeading',
            parent=estilos['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#00d4ff'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )

        # Estilo para texto normal
        estilo_normal = ParagraphStyle(
            'CustomBody',
            parent=estilos['Normal'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=10,
        )

        # Estilo para destaque
        estilo_destaque = ParagraphStyle(
            'CustomHighlight',
            parent=estilos['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#00d4ff'),
            fontName='Helvetica-Bold',
            spaceAfter=8,
        )

        # ====================
        # CABEÇALHO
        # ====================
        titulo = Paragraph("RELATÓRIO DE PREVISÃO", estilo_titulo)
        elementos.append(titulo)

        subtitulo = Paragraph("MarketMind AI - Análise Preditiva de Ações", estilo_normal)
        elementos.append(subtitulo)
        elementos.append(Spacer(1, 20))

        # Linha horizontal
        linha_data = [['', '', '']]
        linha_table = Table(linha_data, colWidths=[6.5*inch])
        linha_table.setStyle(TableStyle([
            ('LINEABOVE', (0,0), (-1,0), 2, colors.HexColor('#00d4ff')),
        ]))
        elementos.append(linha_table)
        elementos.append(Spacer(1, 20))

        # ====================
        # INFORMAÇÕES BÁSICAS
        # ====================
        ticker = dados_acao['ticker']
        nome = dados_acao['nome']
        preco_atual = dados_acao['preco']
        variacao_atual = dados_acao['variacao']
        agora = datetime.now().strftime('%d/%m/%Y às %H:%M')

        info_basica_data = [
            ['Ação:', f"{ticker} - {nome}"],
            ['Data do Relatório:', agora],
            ['Método:', 'Modelo GRU Temporal com Monte Carlo Dropout'],
            ['Horizonte:', '5 dias úteis'],
        ]

        info_table = Table(info_basica_data, colWidths=[2*inch, 4.5*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#1e2130')),
            ('BACKGROUND', (1, 0), (1, -1), colors.white),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.white),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        elementos.append(info_table)
        elementos.append(Spacer(1, 25))

        # ====================
        # PREVISÕES PRINCIPAIS
        # ====================
        elementos.append(Paragraph("PREVISÕES", estilo_subtitulo))

        preco_1_dia = previsoes_ensemble[0]
        preco_5_dias = previsoes_ensemble[4]
        variacao_1_dia = ((preco_1_dia - preco_atual) / preco_atual) * 100
        variacao_5_dias = ((preco_5_dias - preco_atual) / preco_atual) * 100

        # Extrair intervalos de confiança
        lower_1_dia = detalhes_previsoes.get('lower_bound', [preco_1_dia])[0]
        upper_1_dia = detalhes_previsoes.get('upper_bound', [preco_1_dia])[0]
        lower_5_dias = detalhes_previsoes.get('lower_bound', [0]*5)[4]
        upper_5_dias = detalhes_previsoes.get('upper_bound', [0]*5)[4]

        tendencia = "ALTA ↑" if preco_5_dias > preco_atual else "BAIXA ↓"
        cor_tendencia = colors.green if preco_5_dias > preco_atual else colors.red
        confianca_pct = confianca_geral * 100

        # Determinar interpretação de confiança
        if confianca_pct >= 55:
            interpretacao_conf = "Moderada"
        elif confianca_pct >= 40:
            interpretacao_conf = "Baixa"
        else:
            interpretacao_conf = "Muito Baixa"

        previsoes_data = [
            ['Métrica', 'Valor', 'Intervalo 95%'],
            ['Preço Atual', f'R$ {preco_atual:.2f}', f'{variacao_atual:+.2f}%'],
            ['Previsão 1 Dia', f'R$ {preco_1_dia:.2f}', f'R$ {lower_1_dia:.2f} - R$ {upper_1_dia:.2f}'],
            ['Previsão 5 Dias', f'R$ {preco_5_dias:.2f}', f'R$ {lower_5_dias:.2f} - R$ {upper_5_dias:.2f}'],
            ['Tendência', tendencia, '-'],
            ['Confiança do Modelo', f'{confianca_pct:.0f}% ({interpretacao_conf})', '-'],
        ]

        prev_table = Table(previsoes_data, colWidths=[2.2*inch, 2.2*inch, 2.1*inch])
        prev_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00d4ff')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))
        elementos.append(prev_table)
        elementos.append(Spacer(1, 25))

        # ====================
        # MÉTRICAS DO MODELO
        # ====================
        elementos.append(Paragraph("MÉTRICAS DO MODELO GRU", estilo_subtitulo))

        # Extrair métricas técnicas se disponíveis
        mae_val = detalhes_previsoes.get('mae_val', 0)
        volatilidade = detalhes_previsoes.get('volatilidade', 0)
        epochs_trained = detalhes_previsoes.get('epochs_trained', 0)

        metricas_data = [
            ['Métrica', 'Valor'],
            ['MAE de Validação', f'{mae_val:.4f}' if mae_val > 0 else 'N/A'],
            ['Volatilidade da Previsão', f'{volatilidade:.4f}' if volatilidade > 0 else 'N/A'],
            ['Épocas Treinadas', f'{epochs_trained}' if epochs_trained > 0 else 'N/A'],
            ['Iterações Monte Carlo', '50'],
        ]

        metricas_table = Table(metricas_data, colWidths=[3.5*inch, 3*inch])
        metricas_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e2130')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        elementos.append(metricas_table)
        elementos.append(Spacer(1, 25))

        # ====================
        # GRÁFICO DE PREVISÃO
        # ====================
        elementos.append(Paragraph("VISUALIZAÇÃO GRÁFICA", estilo_subtitulo))

        # Criar gráfico com matplotlib
        fig, ax = plt.subplots(figsize=(8, 4))

        # Histórico
        historico = dados_acao.get('historico')
        if historico is not None and not historico.empty:
            # Pegar últimos 30 dias
            hist_recente = historico.tail(30)
            ax.plot(hist_recente.index, hist_recente['Close'],
                   color='#00d4ff', linewidth=2, label='Histórico')

        # Banda de confiança se disponível
        if 'lower_bound' in detalhes_previsoes and 'upper_bound' in detalhes_previsoes:
            lower_bound = detalhes_previsoes['lower_bound']
            upper_bound = detalhes_previsoes['upper_bound']

            ax.fill_between(datas_previsao, lower_bound, upper_bound,
                           color='#ff4444', alpha=0.2, label='IC 95%')

        # Previsões (média)
        ax.plot(datas_previsao, previsoes_ensemble,
               color='#ff4444', linewidth=2, marker='o',
               markersize=6, label='Previsão GRU (Média)')

        ax.set_xlabel('Data', fontsize=10)
        ax.set_ylabel('Preço (R$)', fontsize=10)
        ax.set_title(f'{ticker} - Histórico e Previsão com Intervalos de Confiança', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        # Salvar gráfico em buffer
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()

        # Adicionar imagem ao PDF
        from reportlab.platypus import Image
        img = Image(img_buffer, width=6*inch, height=3*inch)
        elementos.append(img)
        elementos.append(Spacer(1, 25))

        # ====================
        # INTERPRETAÇÃO
        # ====================
        elementos.append(Paragraph("INTERPRETAÇÃO DOS RESULTADOS", estilo_subtitulo))

        if variacao_5_dias > 5:
            interpretacao = f"A previsão indica uma <b>forte tendência de alta</b> para os próximos 5 dias, com valorização estimada de {variacao_5_dias:.2f}%. Esta é uma sinalização positiva segundo o modelo ensemble."
        elif variacao_5_dias > 0:
            interpretacao = f"A previsão indica uma <b>tendência moderada de alta</b> para os próximos 5 dias, com valorização estimada de {variacao_5_dias:.2f}%."
        elif variacao_5_dias > -5:
            interpretacao = f"A previsão indica uma <b>tendência moderada de baixa</b> para os próximos 5 dias, com desvalorização estimada de {variacao_5_dias:.2f}%."
        else:
            interpretacao = f"A previsão indica uma <b>forte tendência de baixa</b> para os próximos 5 dias, com desvalorização estimada de {variacao_5_dias:.2f}%."

        elementos.append(Paragraph(interpretacao, estilo_normal))
        elementos.append(Spacer(1, 10))

        confianca_texto = f"O modelo GRU apresenta <b>{confianca_pct:.0f}% de confiança</b> nas previsões. "
        if confianca_pct >= 55:
            confianca_texto += "Este é um nível de confiança <b>moderado</b> para previsões de mercado. "
        elif confianca_pct >= 40:
            confianca_texto += "Este é um nível de confiança <b>baixo</b>, típico para previsões de ações. "
        else:
            confianca_texto += "Este é um nível de confiança <b>muito baixo</b>, sugere alta incerteza. "

        confianca_texto += f"O intervalo de confiança 95% mostra que há 95% de probabilidade do preço real estar dentro do range apresentado."

        elementos.append(Paragraph(confianca_texto, estilo_normal))
        elementos.append(Spacer(1, 15))

        # Adicionar explicação sobre confiança realista
        explicacao_confianca = f"""
        <b>Sobre a Confiança:</b> O modelo GRU temporal utiliza Monte Carlo Dropout para quantificar incerteza.
        Valores de confiança entre 30-65% são <b>normais e realistas</b> para previsão de ações devido à
        alta complexidade e volatilidade do mercado. Previsões de 50-55% de acurácia direcional são apenas
        <b>marginalmente melhores que sorte (50%)</b>, o que reflete a eficiência do mercado.
        """

        elementos.append(Paragraph(explicacao_confianca, estilo_normal))
        elementos.append(Spacer(1, 30))

        # ====================
        # AVISO LEGAL
        # ====================
        elementos.append(Paragraph("AVISO LEGAL E DISCLAIMER", estilo_subtitulo))

        aviso = f"""
        <b>IMPORTANTE:</b> Este relatório é gerado por um modelo GRU temporal de Machine Learning para fins
        <b>EXCLUSIVAMENTE EDUCACIONAIS E INFORMATIVOS</b>. As previsões apresentadas têm confiança <b>{interpretacao_conf.lower()}</b>
        ({confianca_pct:.0f}%), o que indica <b>alta incerteza</b>.
        <br/><br/>
        Uma confiança de {confianca_pct:.0f}% significa que o modelo tem <b>incerteza significativa</b> nas previsões.
        Modelos de previsão de ações com 50-55% de acurácia direcional são apenas <b>marginalmente melhores que sorte (50%)</b>,
        refletindo a natureza altamente eficiente e imprevisível do mercado de ações.
        <br/><br/>
        Este documento <b>NÃO constitui</b> recomendação de investimento, oferta de compra ou venda de valores mobiliários.
        Investimentos em ações envolvem riscos significativos de perda de capital.
        <br/><br/>
        Desempenho passado não é garantia de resultados futuros. Sempre consulte profissionais qualificados
        (analistas, assessores de investimentos certificados) antes de tomar qualquer decisão de investimento.
        <br/><br/>
        O MarketMind e seus desenvolvedores não se responsabilizam por perdas ou danos decorrentes do uso deste relatório.
        """

        elementos.append(Paragraph(aviso, estilo_normal))
        elementos.append(Spacer(1, 20))

        # Rodapé
        rodape_data = [['', '']]
        rodape_table = Table(rodape_data, colWidths=[6.5*inch])
        rodape_table.setStyle(TableStyle([
            ('LINEABOVE', (0,0), (-1,0), 1, colors.grey),
        ]))
        elementos.append(rodape_table)

        rodape_texto = Paragraph(
            f"<i>Gerado por MarketMind AI em {agora} | www.marketmind.com</i>",
            ParagraphStyle('Footer', parent=estilos['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
        )
        elementos.append(rodape_texto)

        # Construir PDF
        doc.build(elementos)

        # Retornar buffer
        buffer.seek(0)
        return buffer

    except Exception as e:
        st.error(f"Erro ao gerar PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None
