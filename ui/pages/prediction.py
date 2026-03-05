import streamlit as st
from datetime import datetime
from src.auth import eh_favorito, adicionar_favorito, remover_favorito
from src.ml.prediction import gerar_previsao_acao
from src.reports import gerar_relatorio_pdf
from ui.charts import criar_grafico_com_previsao


def render_prediction(dados):
    st.markdown("<div class='tela-dados'>", unsafe_allow_html=True)

    col_titulo, col_fav, col_volta, col_nova = st.columns([2, 1, 1, 1])
    with col_titulo:
        st.markdown("<h4 class='section-title'>Previsão com IA</h4>", unsafe_allow_html=True)
    with col_fav:
        is_fav = eh_favorito(dados['ticker'])
        btn_text = "★ Remover" if is_fav else "☆ Favoritar"

        if st.button(btn_text, type="secondary", use_container_width=True):
            if is_fav:
                sucesso, mensagem = remover_favorito(dados['ticker'])
            else:
                sucesso, mensagem = adicionar_favorito(dados['ticker'], dados['nome'], dados['preco'])

            if sucesso:
                st.success(mensagem)
                st.rerun()
            else:
                st.error(mensagem)
    with col_volta:
        if st.button("Voltar", type="secondary", use_container_width=True):
            st.session_state.mostrar_previsao = False
            st.rerun()
    with col_nova:
        if st.button("Nova Análise", type="secondary", use_container_width=True):
            st.session_state.mostrar_dados = False
            st.session_state.mostrar_previsao = False
            st.session_state.dados_acao = None
            st.rerun()

    # Mode selector — shown before any prediction runs
    col_m1, col_m2, col_m3 = st.columns([1, 2, 1])
    with col_m2:
        modo_selecionado = st.radio(
            "Modo de Análise",
            ["⚡ Rápida (30-60s)", "🔬 Completa (2-4 min)"],
            horizontal=True,
            key="modo_analise_radio",
        )
    modo_rapido = "Rápida" in modo_selecionado

    has_results = (
        'previsoes_ensemble' in st.session_state
        and st.session_state.get('ticker_previsao') == dados['ticker']
        and st.session_state.get('modo_usado') == modo_rapido
    )

    if not has_results:
        col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
        with col_b2:
            if st.button("Gerar Previsão", type="primary", use_container_width=True):
                with st.spinner('Gerando previsões com IA...'):
                    previsoes_ensemble, datas_previsao, detalhes_previsoes, confianca_ml, erro_ml = gerar_previsao_acao(
                        dados, modo_rapido=modo_rapido
                    )
                st.session_state.previsoes_ensemble = previsoes_ensemble
                st.session_state.datas_previsao = datas_previsao
                st.session_state.detalhes_previsoes = detalhes_previsoes
                st.session_state.confianca_ml = confianca_ml
                st.session_state.erro_ml = erro_ml
                st.session_state.ticker_previsao = dados['ticker']
                st.session_state.modo_usado = modo_rapido
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        return

    previsoes_ensemble = st.session_state.previsoes_ensemble
    datas_previsao = st.session_state.datas_previsao
    detalhes_previsoes = st.session_state.detalhes_previsoes
    confianca_ml = st.session_state.confianca_ml
    erro_ml = st.session_state.erro_ml

    if erro_ml:
        st.error(f"Erro na previsão: {erro_ml}")
        st.info("A previsão requer pelo menos 25 dias de histórico.")
    else:
        preco_atual = dados['preco']
        preco_1_dia = previsoes_ensemble[0]
        preco_5_dias = previsoes_ensemble[4]

        # Extrair bounds de confiança se disponíveis
        lower_1_dia = detalhes_previsoes.get('lower_bound', [preco_1_dia])[0]
        upper_1_dia = detalhes_previsoes.get('upper_bound', [preco_1_dia])[0]
        lower_5_dias = detalhes_previsoes.get('lower_bound', [0]*5)[4]
        upper_5_dias = detalhes_previsoes.get('upper_bound', [0]*5)[4]

        variacao_1_dia = ((preco_1_dia - preco_atual) / preco_atual) * 100
        variacao_5_dias = ((preco_5_dias - preco_atual) / preco_atual) * 100

        st.success("Previsões geradas com sucesso!")

        # Métricas de previsão com intervalos de confiança
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Próximo Dia", f"R$ {preco_1_dia:.2f}", f"{variacao_1_dia:+.2f}%")
            st.caption(f"📊 IC 95%: R$ {lower_1_dia:.2f} - R$ {upper_1_dia:.2f}")
        with col2:
            st.metric("5 Dias", f"R$ {preco_5_dias:.2f}", f"{variacao_5_dias:+.2f}%")
            st.caption(f"📊 IC 95%: R$ {lower_5_dias:.2f} - R$ {upper_5_dias:.2f}")
        with col3:
            tendencia = "Alta" if preco_5_dias > preco_atual else "Baixa"
            st.metric("Tendência", tendencia)
        with col4:
            # Confiança com interpretação
            confianca_pct = confianca_ml * 100
            if confianca_pct >= 55:
                interpretacao = "Moderada"
                emoji = "🟡"
            elif confianca_pct >= 40:
                interpretacao = "Baixa"
                emoji = "🟠"
            else:
                interpretacao = "Muito Baixa"
                emoji = "🔴"

            st.metric("Confiança", f"{confianca_pct:.0f}%")
            st.caption(f"{emoji} {interpretacao}")

        # Aviso sobre a confiança realista
        st.info("""
        ℹ️ **Sobre a Confiança**: O modelo LSTM temporal reporta confiança **realista** (30-65%).
        Valores entre 40-55% são **normais** para previsão de ações devido à alta complexidade e volatilidade do mercado.
        O intervalo de confiança 95% (IC 95%) mostra o **range de possibilidades** onde o preço real tem 95% de chance de estar.
        """)

        # Espaço antes do gráfico
        st.markdown("<br>", unsafe_allow_html=True)

        # Gráfico com previsões
        fig = criar_grafico_com_previsao(dados['historico'], dados['ticker'],
                                       previsoes_ensemble, datas_previsao, detalhes_previsoes)
        st.plotly_chart(fig, use_container_width=True)

        # Relatório PDF
        col_rel1, col_rel2, col_rel3 = st.columns([1, 2, 1])
        with col_rel2:
            if st.button("📄 Gerar Relatório PDF", type="primary", use_container_width=True):
                with st.spinner('Gerando relatório em PDF...'):
                    pdf_buffer = gerar_relatorio_pdf(dados, previsoes_ensemble, datas_previsao,
                                                     detalhes_previsoes, confianca_ml)

                if pdf_buffer:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    nome_arquivo = f"Relatorio_MarketMind_{dados['ticker']}_{timestamp}.pdf"

                    st.download_button(
                        label="💾 Download Relatório PDF",
                        data=pdf_buffer,
                        file_name=nome_arquivo,
                        mime="application/pdf",
                        use_container_width=True
                    )
                    st.success("✅ Relatório PDF gerado com sucesso!")

    st.markdown("</div>", unsafe_allow_html=True)
