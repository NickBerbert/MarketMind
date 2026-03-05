import streamlit as st
from src.auth import eh_favorito, adicionar_favorito, remover_favorito
from ui.charts import criar_grafico_com_previsao


def render_stock_detail(dados):
    st.markdown("<div class='tela-dados'>", unsafe_allow_html=True)

    col_titulo, col_fav, col_prev, col_nova = st.columns([2, 1, 1, 1])
    with col_titulo:
        st.markdown("<h4 class='section-title'>Cotação Atual</h4>", unsafe_allow_html=True)
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
    with col_prev:
        if st.button("Gerar Previsão", type="secondary", use_container_width=True):
            st.session_state.mostrar_previsao = True
            st.rerun()
    with col_nova:
        if st.button("Nova Análise", type="secondary", use_container_width=True):
            st.session_state.mostrar_dados = False
            st.session_state.dados_acao = None
            st.rerun()

    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Preço Atual", f"R$ {dados['preco']:.2f}",
                 delta=f"R$ {dados.get('variacao_valor', 0):.2f}")
    with col2:
        st.metric("Variação %", f"{dados['variacao']:.2f}%")
    with col3:
        st.metric("Máxima", f"R$ {dados['maxima']:.2f}")
    with col4:
        st.metric("Mínima", f"R$ {dados['minima']:.2f}")

    # Informações do pregão
    st.markdown("<h4 class='section-title'>Informações do Pregão</h4>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Abertura", f"R$ {dados.get('abertura', 0):.2f}")
    with col2:
        st.metric("Fechamento Anterior", f"R$ {dados.get('fechamento_anterior', 0):.2f}")
    with col3:
        volume_formatado = f"{dados.get('volume', 0):,}".replace(",", ".")
        st.metric("Volume", volume_formatado)
    with col4:
        market_cap = dados.get('market_cap', 0)
        if market_cap > 0:
            if market_cap >= 1e9:
                cap_formatado = f"R$ {market_cap/1e9:.1f}B"
            else:
                cap_formatado = f"R$ {market_cap/1e6:.1f}M"
            st.metric("Market Cap", cap_formatado)
        else:
            st.metric("Market Cap", "N/D")

    # Indicadores técnicos
    indicadores = dados.get('indicadores', {})
    if indicadores:
        st.markdown("<h4 class='section-title'>Indicadores Técnicos</h4>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            maxima_52s = indicadores.get('maxima_52s')
            st.metric("Máxima 52S", f"R$ {maxima_52s:.2f}" if maxima_52s else "N/D")
        with col2:
            minima_52s = indicadores.get('minima_52s')
            st.metric("Mínima 52S", f"R$ {minima_52s:.2f}" if minima_52s else "N/D")
        with col3:
            media_20d = indicadores.get('media_20d')
            st.metric("Média 20D", f"R$ {media_20d:.2f}" if media_20d else "N/D")
        with col4:
            volatilidade = indicadores.get('volatilidade')
            st.metric("Volatilidade", f"{volatilidade:.1f}%" if volatilidade else "N/D")

    # Comparativo 3 meses se disponível
    if dados.get('dados_3_meses'):
        st.markdown("<h4 class='section-title'>Comparativo 3 Meses</h4>", unsafe_allow_html=True)

        d3m = dados['dados_3_meses']

        col1_hist, col2_hist, col3_hist, col4_hist = st.columns(4)

        with col1_hist:
            st.metric("Preço 3M", f"R$ {d3m['preco']:.2f}")
        with col2_hist:
            volume_formatado = f"{d3m['volume']:,}".replace(",", ".")
            st.metric("Volume", volume_formatado)
        with col3_hist:
            st.metric("Data", d3m['data'])
        with col4_hist:
            if dados['preco'] > 0 and d3m['preco'] > 0:
                performance = ((dados['preco'] - d3m['preco']) / d3m['preco']) * 100
                st.metric("Performance 3M", f"{performance:.2f}%")

    # Gráfico histórico
    if dados['historico'] is not None:
        fig = criar_grafico_com_previsao(dados['historico'], dados['ticker'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("<div style='color: #ff6b6b;'>Dados históricos indisponíveis</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
