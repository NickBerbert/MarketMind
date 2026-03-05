import streamlit as st
from src.auth import get_usuario_logado, fazer_logout


def render_header_logado():
    usuario = get_usuario_logado()
    if usuario:
        col1, col2 = st.columns([4, 1])

        with col1:
            st.markdown(f"<h1 class='main-header'>MarketMind</h1>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div style='padding-top: 30px; text-align: right;'>", unsafe_allow_html=True)
            user_action = st.selectbox("", [f"Olá, {usuario}", "Logout"], index=0,
                                     key="user_dropdown", label_visibility="collapsed")
            st.markdown("</div>", unsafe_allow_html=True)

            if user_action == "Logout":
                fazer_logout()
                st.session_state.tela_atual = 'login'
                st.rerun()
