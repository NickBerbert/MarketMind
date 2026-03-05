import streamlit as st
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# MUST be the first Streamlit call
st.set_page_config(page_title="MarketMind", page_icon="🧠", layout="wide")


def load_css():
    css_file = Path(__file__).parent / "assets" / "styles.css"
    if css_file.exists():
        with open(css_file, 'r', encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()

# All project imports come after set_page_config
from src.auth import get_usuario_logado, fazer_logout
from ui.components import render_header_logado
from ui.pages.login import render_login
from ui.pages.cadastro import render_cadastro
from ui.pages.home import render_home
from ui.pages.stock_detail import render_stock_detail
from ui.pages.prediction import render_prediction

# Session state initialization
if 'mostrar_dados' not in st.session_state:
    st.session_state.mostrar_dados = False
if 'dados_acao' not in st.session_state:
    st.session_state.dados_acao = None
if 'mostrar_previsao' not in st.session_state:
    st.session_state.mostrar_previsao = False
if 'tela_atual' not in st.session_state:
    st.session_state.tela_atual = 'login'

# Routing
if st.session_state.tela_atual == 'login':
    render_login()
elif st.session_state.tela_atual == 'cadastro':
    render_cadastro()
else:
    if not get_usuario_logado():
        st.session_state.tela_atual = 'login'
        st.rerun()

    render_header_logado()

    if not st.session_state.mostrar_dados:
        render_home()
    elif st.session_state.mostrar_dados and not st.session_state.mostrar_previsao:
        render_stock_detail(st.session_state.dados_acao)
    elif st.session_state.mostrar_previsao:
        render_prediction(st.session_state.dados_acao)
