#!/usr/bin/env python3
"""
Script para migrar dados dos arquivos JSON para o banco SQLite
"""

from database import DatabaseManager
import os

def main():
    print("🔄 Iniciando migração dos dados JSON para SQLite...")
    
    # Inicializar o banco
    db = DatabaseManager()
    
    # Migrar dados
    try:
        db.migrar_dados_json()
        print("✅ Migração concluída com sucesso!")
        
        # Verificar dados migrados
        usuarios = db.obter_todos_usuarios()
        favoritos_globais = db.obter_favoritos_globais()
        
        print(f"\n📊 Dados migrados:")
        print(f"   - Usuários: {len(usuarios)}")
        print(f"   - Favoritos globais: {len(favoritos_globais)}")
        
        # Contar favoritos por usuário
        total_favoritos_usuarios = 0
        for username in usuarios.keys():
            usuario = db.obter_usuario(username)
            if usuario:
                favoritos = db.obter_favoritos_usuario(usuario['id'])
                total_favoritos_usuarios += len(favoritos)
                print(f"   - Favoritos de {username}: {len(favoritos)}")
        
        print(f"   - Total de favoritos de usuários: {total_favoritos_usuarios}")
        
    except Exception as e:
        print(f"❌ Erro na migração: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()