import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path

class DatabaseManager:
    def __init__(self, db_path="marketmind.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Inicializa o banco de dados com as tabelas necessárias"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabela de usuários
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS usuarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                data_criacao TEXT NOT NULL
            )
        ''')
        
        # Tabela de favoritos
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS favoritos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                usuario_id INTEGER,
                ticker TEXT NOT NULL,
                nome TEXT NOT NULL,
                preco REAL NOT NULL,
                data_adicao TEXT NOT NULL,
                FOREIGN KEY (usuario_id) REFERENCES usuarios (id)
            )
        ''')
        
        # Tabela de favoritos globais (equivalente ao favoritos.json)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS favoritos_globais (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                nome TEXT NOT NULL,
                preco REAL NOT NULL,
                data_adicao TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_connection(self):
        """Retorna uma conexão com o banco"""
        return sqlite3.connect(self.db_path)
    
    # Métodos para usuários
    def criar_usuario(self, username, email, password_hash):
        """Cria um novo usuário"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        data_criacao = datetime.now().strftime("%d/%m/%Y %H:%M")
        
        try:
            cursor.execute('''
                INSERT INTO usuarios (username, email, password_hash, data_criacao)
                VALUES (?, ?, ?, ?)
            ''', (username, email, password_hash, data_criacao))
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            return None
        finally:
            conn.close()
    
    def obter_usuario(self, username):
        """Obtém um usuário pelo username"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM usuarios WHERE username = ?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            return {
                'id': user[0],
                'username': user[1],
                'email': user[2],
                'password_hash': user[3],
                'data_criacao': user[4]
            }
        return None
    
    def obter_todos_usuarios(self):
        """Obtém todos os usuários"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM usuarios')
        users = cursor.fetchall()
        conn.close()
        
        usuarios_dict = {}
        for user in users:
            usuarios_dict[user[1]] = {
                'email': user[2],
                'password': user[3],
                'data_criacao': user[4]
            }
        return usuarios_dict
    
    # Métodos para favoritos de usuário
    def adicionar_favorito_usuario(self, usuario_id, ticker, nome, preco):
        """Adiciona um favorito para um usuário específico"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        data_adicao = datetime.now().strftime("%d/%m/%Y %H:%M")
        
        cursor.execute('''
            INSERT INTO favoritos (usuario_id, ticker, nome, preco, data_adicao)
            VALUES (?, ?, ?, ?, ?)
        ''', (usuario_id, ticker, nome, preco, data_adicao))
        
        conn.commit()
        conn.close()
    
    def obter_favoritos_usuario(self, usuario_id):
        """Obtém os favoritos de um usuário específico"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT ticker, nome, preco, data_adicao 
            FROM favoritos 
            WHERE usuario_id = ?
        ''', (usuario_id,))
        
        favoritos = cursor.fetchall()
        conn.close()
        
        return [
            {
                'ticker': fav[0],
                'nome': fav[1],
                'preco': fav[2],
                'data_adicao': fav[3]
            }
            for fav in favoritos
        ]
    
    def obter_favoritos_usuario_por_username(self, username):
        """Obtém os favoritos de um usuário pelo username"""
        usuario = self.obter_usuario(username)
        if usuario:
            return self.obter_favoritos_usuario(usuario['id'])
        return []
    
    def remover_favorito_usuario(self, usuario_id, ticker):
        """Remove um favorito de um usuário"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM favoritos 
            WHERE usuario_id = ? AND ticker = ?
        ''', (usuario_id, ticker))
        
        conn.commit()
        conn.close()
    
    # Métodos para favoritos globais
    def adicionar_favorito_global(self, ticker, nome, preco):
        """Adiciona um favorito global"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        data_adicao = datetime.now().strftime("%d/%m/%Y %H:%M")
        
        cursor.execute('''
            INSERT INTO favoritos_globais (ticker, nome, preco, data_adicao)
            VALUES (?, ?, ?, ?)
        ''', (ticker, nome, preco, data_adicao))
        
        conn.commit()
        conn.close()
    
    def obter_favoritos_globais(self):
        """Obtém todos os favoritos globais"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT ticker, nome, preco, data_adicao FROM favoritos_globais')
        favoritos = cursor.fetchall()
        conn.close()
        
        return [
            {
                'ticker': fav[0],
                'nome': fav[1],
                'preco': fav[2],
                'data_adicao': fav[3]
            }
            for fav in favoritos
        ]
    
    def remover_favorito_global(self, ticker):
        """Remove um favorito global"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM favoritos_globais WHERE ticker = ?', (ticker,))
        
        conn.commit()
        conn.close()
    
    # Métodos de migração
    def migrar_dados_json(self):
        """Migra dados dos arquivos JSON para o banco SQLite"""
        # Migrar usuários
        usuarios_file = "usuarios.json"
        if os.path.exists(usuarios_file):
            with open(usuarios_file, 'r', encoding='utf-8') as f:
                usuarios_data = json.load(f)
            
            for username, dados in usuarios_data.items():
                self.criar_usuario(
                    username,
                    dados['email'],
                    dados['password'],
                )
        
        # Migrar favoritos globais
        favoritos_file = "favoritos.json"
        if os.path.exists(favoritos_file):
            with open(favoritos_file, 'r', encoding='utf-8') as f:
                favoritos_data = json.load(f)
            
            for favorito in favoritos_data:
                self.adicionar_favorito_global(
                    favorito['ticker'],
                    favorito['nome'],
                    favorito['preco']
                )
        
        # Migrar favoritos de usuários
        favoritos_dir = "favoritos_usuarios"
        if os.path.exists(favoritos_dir):
            for filename in os.listdir(favoritos_dir):
                if filename.endswith('_favoritos.json'):
                    username = filename.replace('_favoritos.json', '')
                    usuario = self.obter_usuario(username)
                    
                    if usuario:
                        filepath = os.path.join(favoritos_dir, filename)
                        with open(filepath, 'r', encoding='utf-8') as f:
                            favoritos_data = json.load(f)
                        
                        for favorito in favoritos_data:
                            self.adicionar_favorito_usuario(
                                usuario['id'],
                                favorito['ticker'],
                                favorito['nome'],
                                favorito['preco']
                            )