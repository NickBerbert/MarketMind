import sqlite3
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path="marketmind.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Inicializa tabelas do banco de dados"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.executescript('''
                CREATE TABLE IF NOT EXISTS usuarios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    data_criacao TEXT NOT NULL
                );
                
                CREATE TABLE IF NOT EXISTS favoritos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    usuario_id INTEGER,
                    ticker TEXT NOT NULL,
                    nome TEXT NOT NULL,
                    preco REAL NOT NULL,
                    data_adicao TEXT NOT NULL,
                    FOREIGN KEY (usuario_id) REFERENCES usuarios (id)
                );
            ''')
    
    def _get_connection(self):
        """Context manager para conexões de banco"""
        return sqlite3.connect(self.db_path)
    
    def criar_usuario(self, username, email, password_hash):
        """Cria novo usuário no banco"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            data_criacao = datetime.now().strftime("%d/%m/%Y %H:%M")
            
            try:
                cursor.execute(
                    'INSERT INTO usuarios (username, email, password_hash, data_criacao) VALUES (?, ?, ?, ?)',
                    (username, email, password_hash, data_criacao)
                )
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                return None
    
    def obter_usuario(self, username):
        """Obtém usuário por username"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM usuarios WHERE username = ?', (username,))
            user = cursor.fetchone()
            
            return {
                'id': user[0],
                'username': user[1],
                'email': user[2],
                'password_hash': user[3],
                'data_criacao': user[4]
            } if user else None
    
    def obter_todos_usuarios(self):
        """Retorna dicionário com todos os usuários (compat. legado)"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT username, email, password_hash, data_criacao FROM usuarios')
            users = cursor.fetchall()
            
            return {
                user[0]: {
                    'email': user[1],
                    'password': user[2],
                    'data_criacao': user[3]
                } for user in users
            }
    
    def adicionar_favorito_usuario(self, usuario_id, ticker, nome, preco):
        """Adiciona favorito para usuário específico"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            data_adicao = datetime.now().strftime("%d/%m/%Y %H:%M")
            
            cursor.execute(
                'INSERT INTO favoritos (usuario_id, ticker, nome, preco, data_adicao) VALUES (?, ?, ?, ?, ?)',
                (usuario_id, ticker, nome, preco, data_adicao)
            )
    
    def obter_favoritos_usuario(self, usuario_id):
        """Lista favoritos do usuário por ID"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT ticker, nome, preco, data_adicao FROM favoritos WHERE usuario_id = ?',
                (usuario_id,)
            )
            favoritos = cursor.fetchall()
            
            return [
                {
                    'ticker': fav[0],
                    'nome': fav[1],
                    'preco': fav[2],
                    'data_adicao': fav[3]
                } for fav in favoritos
            ]
    
    def obter_favoritos_usuario_por_username(self, username):
        """Lista favoritos do usuário por username"""
        usuario = self.obter_usuario(username)
        return self.obter_favoritos_usuario(usuario['id']) if usuario else []
    
    def remover_favorito_usuario(self, usuario_id, ticker):
        """Remove favorito específico do usuário"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'DELETE FROM favoritos WHERE usuario_id = ? AND ticker = ?',
                (usuario_id, ticker)
            )
    
