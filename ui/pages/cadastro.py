import streamlit as st
from src.auth import criar_usuario


def render_cadastro():
    st.markdown("<div class='screen-entrada'>", unsafe_allow_html=True)

    # Header centralizado
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center; color: #00d4ff; font-size: 2.2rem; font-weight: 600; font-family: Poppins, sans-serif; letter-spacing: -0.02em; margin-bottom: 1rem;'>MarketMind</h1>", unsafe_allow_html=True)

    # Formulário centralizado
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("### Criar Conta")

        with st.form("cadastro_form"):
            username = st.text_input("Usuário", placeholder="Nome de usuário")
            email = st.text_input("Email", placeholder="Seu email")
            password = st.text_input("Senha", type="password", placeholder="Sua senha")
            password_confirm = st.text_input("Confirmar Senha", type="password", placeholder="Confirme a senha")

            # Botões alinhados com os inputs
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                cadastro_button = st.form_submit_button("Criar Conta", type="primary")
            with col_btn2:
                if st.form_submit_button("Voltar"):
                    st.session_state.tela_atual = 'login'
                    st.rerun()

    if cadastro_button:
        if all([username, email, password, password_confirm]):
            if password != password_confirm:
                st.error("Senhas não coincidem")
            elif len(password) < 6:
                st.error("Senha deve ter pelo menos 6 caracteres")
            else:
                sucesso, mensagem = criar_usuario(username, email, password)
                if sucesso:
                    st.success(mensagem)
                    st.session_state.tela_atual = 'login'
                    st.rerun()
                else:
                    st.error(mensagem)
        else:
            st.error("Preencha todos os campos")

    st.markdown("</div>", unsafe_allow_html=True)
