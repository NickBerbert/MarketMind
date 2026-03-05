import hashlib
import streamlit as st
from src.database import DatabaseManager

db = DatabaseManager()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def criar_usuario(username, email, password):
    if db.obter_usuario(username):
        return False, "Usuário já existe"

    usuarios = db.obter_todos_usuarios()
    if any(user_data.get('email') == email for user_data in usuarios.values()):
        return False, "Email já cadastrado"

    user_id = db.criar_usuario(username, email, hash_password(password))
    return (True, "Usuário criado com sucesso") if user_id else (False, "Erro ao criar usuário")


def autenticar_usuario(username, password):
    usuario = db.obter_usuario(username)
    if not usuario:
        return False, "Usuário não encontrado"

    if usuario['password_hash'] != hash_password(password):
        return False, "Senha incorreta"

    return True, "Login realizado com sucesso"


def get_usuario_logado():
    return st.session_state.get('usuario_logado')


def fazer_login(username):
    st.session_state.usuario_logado = username


def fazer_logout():
    keys_to_remove = ['usuario_logado', 'dados_acao', 'mostrar_dados', 'mostrar_previsao']
    for key in keys_to_remove:
        st.session_state.pop(key, None)


def carregar_favoritos(username=None):
    username = username or get_usuario_logado()
    return db.obter_favoritos_usuario_por_username(username) if username else []


def adicionar_favorito(ticker, nome, preco):
    username = get_usuario_logado()
    if not username:
        return False, "Usuário não está logado"

    usuario = db.obter_usuario(username)
    if not usuario:
        return False, "Usuário não encontrado"

    favoritos = db.obter_favoritos_usuario(usuario['id'])
    if any(fav['ticker'] == ticker for fav in favoritos):
        return False, "Ação já está nos favoritos"

    try:
        db.adicionar_favorito_usuario(usuario['id'], ticker, nome, preco)
        return True, "Ação adicionada aos favoritos"
    except Exception:
        return False, "Erro ao salvar favorito"


def remover_favorito(ticker):
    username = get_usuario_logado()
    if not username:
        return False, "Usuário não está logado"

    usuario = db.obter_usuario(username)
    if not usuario:
        return False, "Usuário não encontrado"

    try:
        db.remover_favorito_usuario(usuario['id'], ticker)
        return True, "Ação removida dos favoritos"
    except Exception:
        return False, "Erro ao remover favorito"


def eh_favorito(ticker):
    username = get_usuario_logado()
    if not username:
        return False

    favoritos = db.obter_favoritos_usuario_por_username(username)
    return any(fav['ticker'] == ticker for fav in favoritos)
