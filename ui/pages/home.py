import streamlit as st
from src.auth import carregar_favoritos, remover_favorito
from src.data import buscar_dados_acao, buscar_dados_rapidos


def render_home():
    st.markdown("<div class='screen-entrada'>", unsafe_allow_html=True)
    st.markdown("### Digite o código da ação:")

    with st.form("busca_acao"):
        ticker = st.text_input("", placeholder="Ex: PETR4, VALE3, ITUB4", label_visibility="collapsed")
        submitted = st.form_submit_button("Analisar Ação", type="primary")

    if submitted and ticker:
        dados, erro = buscar_dados_acao(ticker)
        if erro:
            st.error(f"Erro: {erro}")
        else:
            st.session_state.dados_acao = dados
            st.session_state.mostrar_dados = True
            st.session_state.mostrar_previsao = False
            st.rerun()
    elif submitted:
        st.error("Digite o código de uma ação")

    # Favoritos
    favoritos = carregar_favoritos()
    if favoritos:
        st.markdown("---")
        st.markdown("### Favoritos")

        for i in range(0, min(6, len(favoritos)), 3):
            cols = st.columns(3)

            for j, col in enumerate(cols):
                fav_index = i + j
                if fav_index < len(favoritos):
                    with col:
                        fav = favoritos[fav_index]
                        dados_rapidos = buscar_dados_rapidos(fav['ticker'])

                        with st.container():
                            btn_col1, btn_col2 = st.columns([1, 1])

                            with btn_col1:
                                if st.button("📈", key=f"analyze_{fav['ticker']}",
                                           help="Analisar", use_container_width=True):
                                    dados, erro = buscar_dados_acao(fav['ticker'])
                                    if not erro:
                                        st.session_state.dados_acao = dados
                                        st.session_state.mostrar_dados = True
                                        st.rerun()
                                    else:
                                        st.error(erro)

                            with btn_col2:
                                if st.button("❌", key=f"remove_{fav['ticker']}",
                                           help="Remover", use_container_width=True):
                                    sucesso, mensagem = remover_favorito(fav['ticker'])
                                    if sucesso:
                                        st.success(mensagem)
                                        st.rerun()
                                    else:
                                        st.error(mensagem)

                            if dados_rapidos['sucesso']:
                                st.metric(
                                    label=fav['ticker'],
                                    value=f"R$ {dados_rapidos['preco']:.2f}",
                                    delta=f"{dados_rapidos['variacao']:+.2f}%"
                                )
                            else:
                                st.metric(label=fav['ticker'], value="N/D")

    st.markdown("</div>", unsafe_allow_html=True)
