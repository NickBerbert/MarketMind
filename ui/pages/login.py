import streamlit as st
from src.auth import autenticar_usuario, fazer_login


def render_login():
    st.markdown("<div class='screen-entrada'>", unsafe_allow_html=True)

    # Header centralizado e maior
    col1, col2, col3 = st.columns([0.5, 3, 0.5])
    with col2:
        st.markdown("<h1 style='text-align: center; color: white; font-size: 3rem; font-weight: 600; font-family: Poppins, sans-serif; letter-spacing: -0.02em; margin-bottom: 1.5rem;'>MarketMind</h1>", unsafe_allow_html=True)

    # Formulário centralizado
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("### Login")

        with st.form("login_form"):
            username = st.text_input("Usuário", placeholder="Digite seu usuário")
            password = st.text_input("Senha", type="password", placeholder="Digite sua senha")

            # Botões alinhados com os inputs
            st.markdown("<br>", unsafe_allow_html=True)  # Espaço antes dos botões
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                login_button = st.form_submit_button("Entrar", type="primary", use_container_width=True)
            with col_btn2:
                if st.form_submit_button("Criar Conta", use_container_width=True):
                    st.session_state.tela_atual = 'cadastro'
                    st.rerun()

    if login_button:
        if username and password:
            sucesso, mensagem = autenticar_usuario(username, password)
            if sucesso:
                fazer_login(username)
                st.session_state.tela_atual = 'home'
                st.success(mensagem)
                st.rerun()
            else:
                st.error(mensagem)
        else:
            st.error("Preencha todos os campos")

    st.markdown("</div>", unsafe_allow_html=True)
