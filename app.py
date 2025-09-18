import streamlit as st
import pandas as pd
import numpy as np
import requests
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from database import DatabaseManager

import warnings
warnings.filterwarnings('ignore')

# Configuração da página e recursos globais
st.set_page_config(page_title="MarketMind", page_icon="🧠", layout="wide")

def load_css():
    css_file = Path(__file__).parent / "styles.css"
    if css_file.exists():
        with open(css_file, 'r', encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()
db = DatabaseManager()
API_KEY = "nUUZxG2ZdAWuSkBDhPobC2"

# Funções de autenticação e segurança
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

# Gestão de favoritos
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

# API de dados financeiros
def buscar_dados_rapidos(ticker):
    try:
        headers = {"Authorization": f"Bearer {API_KEY}"}
        response = requests.get(f"https://brapi.dev/api/quote/{ticker}", headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and data['results']:
                acao_data = data['results'][0]
                return {
                    'preco': round(acao_data.get('regularMarketPrice', 0), 2),
                    'variacao': round(acao_data.get('regularMarketChangePercent', 0), 2),
                    'sucesso': True
                }
    except Exception:
        pass
    
    return {'preco': 0, 'variacao': 0, 'sucesso': False}

def buscar_dados_acao(ticker):
    try:
        ticker = ticker.upper().strip()
        
        with st.spinner(f'Buscando dados para {ticker}...'):
            headers = {"Authorization": f"Bearer {API_KEY}"}
            
            # Dados atuais
            response = requests.get(f"https://brapi.dev/api/quote/{ticker}", headers=headers, timeout=10)
            if response.status_code != 200:
                return None, f"Erro {response.status_code}: Ticker {ticker} não encontrado"
            
            data = response.json()
            if 'results' not in data or not data['results']:
                return None, f"Ticker {ticker} não encontrado na B3"
            
            acao_data = data['results'][0]
            
            # Dados históricos
            response_hist = requests.get(f"https://brapi.dev/api/quote/{ticker}?range=3mo&interval=1d", 
                                       headers=headers, timeout=10)
            
            historico, dados_3_meses, indicadores = None, None, {}
            
            if response_hist.status_code == 200:
                data_hist = response_hist.json()
                
                if 'results' in data_hist and data_hist['results']:
                    hist_data = data_hist['results'][0].get('historicalDataPrice', [])
                    
                    if hist_data:
                        hist_list = []
                        for item in hist_data:
                            try:
                                hist_list.append({
                                    'Data': datetime.fromtimestamp(item['date']),
                                    'Close': item.get('close', 0),
                                    'Open': item.get('open', 0),
                                    'High': item.get('high', 0),
                                    'Low': item.get('low', 0),
                                    'Volume': item.get('volume', 0)
                                })
                            except (KeyError, ValueError):
                                continue
                        
                        if hist_list:
                            historico = pd.DataFrame(hist_list).set_index('Data').sort_index()
                            
                            # Dados de 3 meses atrás
                            if len(historico) > 30:
                                idx_proporcional = min(len(historico) - 1, len(historico) * 3 // 4)
                                dados_3m_atras = historico.iloc[-idx_proporcional]
                                dados_3_meses = {
                                    'data': dados_3m_atras.name.strftime('%d/%m/%Y'),
                                    'preco': round(dados_3m_atras['Close'], 2),
                                    'volume': int(dados_3m_atras['Volume'])
                                }
                            
                            # Indicadores técnicos
                            if len(historico) > 0:
                                indicadores.update({
                                    'maxima_52s': round(historico['High'].max(), 2),
                                    'minima_52s': round(historico['Low'].min(), 2),
                                    'volume_medio': int(historico['Volume'].tail(min(30, len(historico))).mean())
                                })
                                
                                if len(historico) >= 20:
                                    indicadores['media_20d'] = round(historico['Close'].tail(20).mean(), 2)
                                if len(historico) >= 50:
                                    indicadores['media_50d'] = round(historico['Close'].tail(50).mean(), 2)
                                if len(historico) >= 30:
                                    returns = historico['Close'].pct_change().tail(30)
                                    volatilidade = returns.std() * np.sqrt(252) * 100
                                    indicadores['volatilidade'] = round(volatilidade, 1)
            
            return {
                'ticker': acao_data.get('symbol', ticker),
                'nome': acao_data.get('shortName', acao_data.get('longName', 'N/A')),
                'preco': round(acao_data.get('regularMarketPrice', 0), 2),
                'variacao': round(acao_data.get('regularMarketChangePercent', 0), 2),
                'variacao_valor': round(acao_data.get('regularMarketChange', 0), 2),
                'maxima': round(acao_data.get('regularMarketDayHigh', 0), 2),
                'minima': round(acao_data.get('regularMarketDayLow', 0), 2),
                'volume': acao_data.get('regularMarketVolume', 0),
                'abertura': round(acao_data.get('regularMarketOpen', 0), 2),
                'fechamento_anterior': round(acao_data.get('regularMarketPreviousClose', 0), 2),
                'market_cap': acao_data.get('marketCap', 0),
                'historico': historico,
                'dados_3_meses': dados_3_meses,
                'indicadores': indicadores
            }, None
            
    except requests.exceptions.Timeout:
        return None, "Timeout: API demorou para responder."
    except requests.exceptions.ConnectionError:
        return None, "Erro de conexão. Verifique sua internet."
    except Exception as e:
        return None, f"Erro: {str(e)}"

# Indicadores técnicos
def calcular_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return (100 - (100 / (1 + rs))).fillna(50)

def calcular_macd(prices, fast=12, slow=26):
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    return (ema_fast - ema_slow).fillna(0)

def calcular_bollinger_bands(prices, window=20, std_dev=2):
    ma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = ma + (std * std_dev)
    lower = ma - (std * std_dev)
    return upper.fillna(prices), lower.fillna(prices)

# Preparação de dados para machine learning - VERSÃO MELHORADA
def preparar_dados_financeiros(historico_df, dias_previsao=5):
    """
    Prepara dados financeiros com features mais robustas e approach conservador
    """
    try:
        if historico_df is None or historico_df.empty:
            return None, None, None, "Dados históricos insuficientes"

        df = historico_df.copy()

        # Verificar dados mínimos necessários
        if len(df) < 60:
            return None, None, None, f"Histórico insuficiente: {len(df)} dias (mínimo: 60)"

        # Features técnicas robustas
        # 1. Médias móveis de diferentes períodos
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()

        # 2. Momentum e volatilidade
        df['RSI'] = calcular_rsi(df['Close'], period=14)
        df['MACD'] = calcular_macd(df['Close'])
        df['BB_upper'], df['BB_lower'] = calcular_bollinger_bands(df['Close'])
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['Close']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

        # 3. Retornos e volatilidade
        df['Return_1d'] = df['Close'].pct_change()
        df['Return_3d'] = df['Close'].pct_change(3)
        df['Return_5d'] = df['Close'].pct_change(5)
        df['Volatility_10d'] = df['Return_1d'].rolling(window=10).std()
        df['Volatility_20d'] = df['Return_1d'].rolling(window=20).std()

        # 4. Volume e liquidez
        df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA_10']
        df['Price_Volume'] = df['Close'] * df['Volume']

        # 5. Tendência e momentum
        df['Price_above_SMA20'] = (df['Close'] > df['SMA_20']).astype(int)
        df['SMA_trend'] = (df['SMA_5'] > df['SMA_20']).astype(int)
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_histogram'] = df['MACD'] - df['MACD_signal']

        # 6. Variáveis de preço relativas (mais estáveis)
        df['High_Low_ratio'] = (df['High'] - df['Low']) / df['Close']
        df['Open_Close_ratio'] = (df['Close'] - df['Open']) / df['Open']

        # Remover NaNs
        df = df.dropna()

        if len(df) < 40:
            return None, None, None, "Dados insuficientes após limpeza"

        # Features selecionadas (não correlacionadas)
        feature_cols = [
            'Return_1d', 'Return_3d', 'Return_5d',
            'RSI', 'MACD', 'MACD_histogram',
            'BB_width', 'BB_position',
            'Volatility_10d', 'Volatility_20d',
            'Volume_ratio', 'High_Low_ratio', 'Open_Close_ratio',
            'Price_above_SMA20', 'SMA_trend'
        ]

        # Preparar dados para modelos
        X = df[feature_cols].values

        # Target: retorno percentual dos próximos dias (mais estável que preço absoluto)
        y = []
        for i in range(len(df) - dias_previsao):
            current_price = df['Close'].iloc[i]
            future_returns = []
            for j in range(1, dias_previsao + 1):
                if i + j < len(df):
                    future_price = df['Close'].iloc[i + j]
                    ret = (future_price - current_price) / current_price
                    future_returns.append(ret)

            if len(future_returns) == dias_previsao:
                y.append(future_returns)

        if len(y) < 20:
            return None, None, None, "Dados insuficientes para criar targets"

        X_final = X[:len(y)]
        y_final = np.array(y)

        return X_final, y_final, df, None

    except Exception as e:
        return None, None, None, f"Erro ao preparar dados: {str(e)}"

# Sistema de validação temporal (Time Series Split)
def criar_splits_temporais(X, y, n_splits=3):
    """Cria splits temporais respeitando a ordem cronológica"""
    splits = []
    total_size = len(X)

    for i in range(n_splits):
        # Tamanho crescente do conjunto de treino
        train_size = int(total_size * (0.5 + i * 0.15))
        test_start = min(train_size, total_size - 10)  # Garantir pelo menos 10 amostras de teste

        train_idx = list(range(train_size))
        test_idx = list(range(test_start, min(test_start + 10, total_size)))

        if len(test_idx) >= 5:  # Mínimo de amostras para teste
            splits.append((train_idx, test_idx))

    return splits

# Novos modelos mais robustos
def treinar_modelos_financeiros(X, y):
    """
    Treina modelos especializados para dados financeiros com validação temporal
    """
    try:
        from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
        from sklearn.linear_model import Ridge, ElasticNet
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import warnings
        warnings.filterwarnings('ignore')

        modelos = {}
        scores = {}

        # Configurações dos modelos (mais conservadoras)
        model_configs = {
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'ExtraTrees': ExtraTreesRegressor(
                n_estimators=100,
                max_depth=6,
                min_samples_split=15,
                min_samples_leaf=7,
                random_state=42,
                n_jobs=-1
            ),
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        }

        # Validação temporal para cada modelo
        splits = criar_splits_temporais(X, y, n_splits=3)

        if not splits:
            # Fallback: split simples se não conseguir criar splits temporais
            split_point = int(0.8 * len(X))
            X_train, X_test = X[:split_point], X[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]

            for nome, modelo_base in model_configs.items():
                try:
                    # Treinar um modelo para cada dia de previsão
                    modelos_dias = []
                    for dia in range(y.shape[1]):
                        modelo = model_configs[nome]
                        modelo.fit(X_train, y_train[:, dia])
                        modelos_dias.append(modelo)

                    modelos[nome] = modelos_dias

                    # Calcular score médio
                    pred_test = np.array([m.predict(X_test) for m in modelos_dias]).T
                    mae = mean_absolute_error(y_test, pred_test)
                    scores[nome] = {'mae': mae, 'confidence': max(0.3, 1.0 - mae * 10)}

                except Exception as e:
                    print(f"Erro no modelo {nome}: {e}")
                    continue
        else:
            # Validação temporal robusta
            for nome, modelo_base in model_configs.items():
                try:
                    fold_scores = []
                    modelos_finais = []

                    # Treinar e validar em cada fold temporal
                    for train_idx, test_idx in splits:
                        X_train_fold = X[train_idx]
                        X_test_fold = X[test_idx]
                        y_train_fold = y[train_idx]
                        y_test_fold = y[test_idx]

                        # Treinar modelos para cada dia
                        modelos_fold = []
                        for dia in range(y.shape[1]):
                            modelo = model_configs[nome]
                            modelo.fit(X_train_fold, y_train_fold[:, dia])
                            modelos_fold.append(modelo)

                        # Validar
                        pred_fold = np.array([m.predict(X_test_fold) for m in modelos_fold]).T
                        mae_fold = mean_absolute_error(y_test_fold, pred_fold)
                        fold_scores.append(mae_fold)

                        modelos_finais = modelos_fold  # Manter o último conjunto de modelos

                    # Scores finais
                    mae_media = np.mean(fold_scores)
                    mae_std = np.std(fold_scores)
                    confidence = max(0.4, min(0.9, 1.0 - mae_media * 15))

                    modelos[nome] = modelos_finais
                    scores[nome] = {
                        'mae': mae_media,
                        'mae_std': mae_std,
                        'confidence': confidence
                    }

                except Exception as e:
                    print(f"Erro no modelo {nome}: {e}")
                    continue

        return modelos, scores, None

    except Exception as e:
        return None, None, f"Erro no treinamento: {str(e)}"

# Nova função de previsão melhorada
def fazer_previsao_financeira(modelos, scores, df_original, X_features):
    """
    Faz previsões com base em retornos percentuais e converte para preços
    """
    try:
        # Pegar as features da última observação
        last_features = X_features[-1:].reshape(1, -1)

        # Preço atual para conversão
        preco_atual = df_original['Close'].iloc[-1]

        # Previsões de cada modelo (retornos percentuais)
        previsoes_retornos = {}
        pesos_modelos = {}

        for nome_modelo, lista_modelos in modelos.items():
            try:
                # Pegar o score de confiança do modelo
                confianca = scores.get(nome_modelo, {}).get('confidence', 0.5)
                pesos_modelos[nome_modelo] = confianca

                # Fazer previsão para cada dia
                pred_dias = []
                for modelo_dia in lista_modelos:
                    pred = modelo_dia.predict(last_features)[0]
                    pred_dias.append(pred)

                previsoes_retornos[nome_modelo] = np.array(pred_dias)

            except Exception as e:
                print(f"Erro no modelo {nome_modelo}: {e}")
                continue

        if not previsoes_retornos:
            return None, None, None, None, "Nenhum modelo conseguiu fazer previsões"

        # Normalizar pesos
        total_peso = sum(pesos_modelos.values())
        if total_peso > 0:
            pesos_modelos = {k: v/total_peso for k, v in pesos_modelos.items()}
        else:
            # Pesos iguais se não tiver scores
            peso_igual = 1.0 / len(pesos_modelos)
            pesos_modelos = {k: peso_igual for k in pesos_modelos.keys()}

        # Ensemble das previsões (média ponderada)
        num_dias = len(list(previsoes_retornos.values())[0])
        previsao_retornos_ensemble = np.zeros(num_dias)

        for nome_modelo, pred_retornos in previsoes_retornos.items():
            peso = pesos_modelos[nome_modelo]
            previsao_retornos_ensemble += pred_retornos * peso

        # Converter retornos para preços absolutos
        previsao_precos = []
        preco_base = preco_atual

        for i, retorno in enumerate(previsao_retornos_ensemble):
            # Limitar retornos extremos (máximo ±20% por dia)
            retorno_limitado = np.clip(retorno, -0.20, 0.20)
            novo_preco = preco_base * (1 + retorno_limitado)
            previsao_precos.append(novo_preco)
            preco_base = novo_preco  # Usar o preço previsto como base para o próximo dia

        previsao_precos = np.array(previsao_precos)

        # Criar datas de previsão (apenas dias úteis)
        datas_previsao = []
        data_atual = df_original.index.max()
        dias_adicionados = 0

        while dias_adicionados < num_dias:
            data_atual += timedelta(days=1)
            if data_atual.weekday() < 5:  # Segunda a sexta
                datas_previsao.append(data_atual)
                dias_adicionados += 1

        # Calcular confiança geral baseada na concordância entre modelos
        if len(previsoes_retornos) > 1:
            # Calcular dispersão entre modelos
            retornos_array = np.array([pred for pred in previsoes_retornos.values()])
            dispersao = np.std(retornos_array, axis=0).mean()

            # Confiança inversamente proporcional à dispersão
            confianca_geral = max(0.5, min(0.9, 1.0 - dispersao * 20))
        else:
            # Se só tem um modelo, usar a confiança dele
            confianca_geral = list(scores.values())[0].get('confidence', 0.6)

        return previsao_precos, datas_previsao, previsoes_retornos, confianca_geral, None

    except Exception as e:
        return None, None, None, None, f"Erro na previsão: {str(e)}"

# Nova função principal para previsões
def gerar_previsao_acao(dados_acao):
    """
    Função principal melhorada para gerar previsões de ações
    """
    try:
        historico = dados_acao.get('historico')
        if historico is None or historico.empty:
            return None, None, None, None, "Dados históricos não disponíveis"

        # Preparar dados com novo sistema
        X, y, df_features, erro_prep = preparar_dados_financeiros(historico, dias_previsao=5)
        if erro_prep:
            return None, None, None, None, erro_prep

        # Treinar modelos com validação temporal
        modelos, scores, erro_treino = treinar_modelos_financeiros(X, y)
        if erro_treino:
            return None, None, None, None, erro_treino

        if not modelos:
            return None, None, None, None, "Nenhum modelo foi treinado com sucesso"

        # Fazer previsões
        previsoes, datas, detalhes, confianca, erro_pred = fazer_previsao_financeira(
            modelos, scores, df_features, X
        )

        if erro_pred:
            return None, None, None, None, erro_pred

        return previsoes, datas, detalhes, confianca, None

    except Exception as e:
        return None, None, None, None, f"Erro na previsão: {str(e)}"

# Sistema de Backtesting para Validação de Previsões
def fazer_backtesting(dados_acao, dias_retroativos=10, valor_investimento=1000):
    """
    Simula investimentos baseados nas previsões dos dias anteriores
    """
    try:
        historico = dados_acao.get('historico')
        if historico is None or historico.empty or len(historico) < 70:
            return None, "Dados históricos insuficientes para backtesting"

        resultados = []

        # Simular previsões para os últimos N dias
        for i in range(dias_retroativos, 0, -1):
            # Cortar histórico até N dias atrás
            historico_corte = historico.iloc[:-i].copy()

            if len(historico_corte) < 60:
                continue

            # Fazer previsão com dados até aquele ponto
            X, y, df_features, erro_prep = preparar_dados_financeiros(historico_corte, dias_previsao=5)
            if erro_prep:
                continue

            modelos, scores, erro_treino = treinar_modelos_financeiros(X, y)
            if erro_treino or not modelos:
                continue

            previsoes, datas, detalhes, confianca, erro_pred = fazer_previsao_financeira(
                modelos, scores, df_features, X
            )

            if erro_pred or previsoes is None:
                continue

            # Dados do dia da "decisão"
            data_decisao = historico_corte.index[-1]
            preco_compra = historico_corte['Close'].iloc[-1]

            # Resultado real após 1 dia e 5 dias
            try:
                # Encontrar os índices correspondentes no histórico completo
                idx_decisao = historico.index.get_loc(data_decisao)

                # Resultado 1 dia depois
                retorno_real_1d = None
                preco_real_1d = None
                if idx_decisao + 1 < len(historico):
                    preco_real_1d = historico['Close'].iloc[idx_decisao + 1]
                    retorno_real_1d = (preco_real_1d - preco_compra) / preco_compra

                # Resultado 5 dias depois
                retorno_real_5d = None
                preco_real_5d = None
                if idx_decisao + 5 < len(historico):
                    preco_real_5d = historico['Close'].iloc[idx_decisao + 5]
                    retorno_real_5d = (preco_real_5d - preco_compra) / preco_compra

                # Previsões
                retorno_previsto_1d = (previsoes[0] - preco_compra) / preco_compra
                retorno_previsto_5d = (previsoes[4] - preco_compra) / preco_compra

                # Simular decisão de investimento baseada na previsão
                # Investir apenas se a previsão for positiva e confiança > 60%
                deveria_investir = retorno_previsto_5d > 0.02 and confianca > 0.6  # 2% mínimo e 60% confiança

                resultado = {
                    'data_decisao': data_decisao.strftime('%d/%m/%Y'),
                    'preco_compra': preco_compra,
                    'previsao_1d': previsoes[0] if previsoes is not None else None,
                    'previsao_5d': previsoes[4] if previsoes is not None else None,
                    'retorno_previsto_1d': retorno_previsto_1d,
                    'retorno_previsto_5d': retorno_previsto_5d,
                    'preco_real_1d': preco_real_1d,
                    'preco_real_5d': preco_real_5d,
                    'retorno_real_1d': retorno_real_1d,
                    'retorno_real_5d': retorno_real_5d,
                    'confianca': confianca,
                    'deveria_investir': deveria_investir,
                    'acerto_direcao_1d': None,
                    'acerto_direcao_5d': None,
                    'lucro_1d': None,
                    'lucro_5d': None
                }

                # Calcular acertos de direção
                if retorno_real_1d is not None:
                    resultado['acerto_direcao_1d'] = (retorno_previsto_1d > 0) == (retorno_real_1d > 0)
                    if deveria_investir:
                        resultado['lucro_1d'] = valor_investimento * retorno_real_1d

                if retorno_real_5d is not None:
                    resultado['acerto_direcao_5d'] = (retorno_previsto_5d > 0) == (retorno_real_5d > 0)
                    if deveria_investir:
                        resultado['lucro_5d'] = valor_investimento * retorno_real_5d

                resultados.append(resultado)

            except Exception as e:
                print(f"Erro ao processar data {data_decisao}: {e}")
                continue

        return resultados, None

    except Exception as e:
        return None, f"Erro no backtesting: {str(e)}"

def analisar_performance_backtesting(resultados_backtesting):
    """
    Analisa os resultados do backtesting e gera métricas de performance
    """
    if not resultados_backtesting:
        return None

    # Filtrar apenas resultados válidos
    resultados_validos = [r for r in resultados_backtesting if r['retorno_real_5d'] is not None]
    investimentos_feitos = [r for r in resultados_validos if r['deveria_investir']]

    if not resultados_validos:
        return None

    # Métricas gerais
    total_simulacoes = len(resultados_validos)
    total_investimentos = len(investimentos_feitos)

    # Acurácia de direção
    acertos_1d = [r for r in resultados_validos if r['acerto_direcao_1d']]
    acertos_5d = [r for r in resultados_validos if r['acerto_direcao_5d']]

    acuracia_1d = len(acertos_1d) / total_simulacoes if total_simulacoes > 0 else 0
    acuracia_5d = len(acertos_5d) / total_simulacoes if total_simulacoes > 0 else 0

    # Performance financeira (apenas investimentos feitos)
    if investimentos_feitos:
        lucros_1d = [r['lucro_1d'] for r in investimentos_feitos if r['lucro_1d'] is not None]
        lucros_5d = [r['lucro_5d'] for r in investimentos_feitos if r['lucro_5d'] is not None]

        lucro_total_1d = sum(lucros_1d) if lucros_1d else 0
        lucro_total_5d = sum(lucros_5d) if lucros_5d else 0

        investimentos_positivos_5d = len([l for l in lucros_5d if l > 0])
        taxa_sucesso = investimentos_positivos_5d / len(lucros_5d) if lucros_5d else 0

        # Retorno médio
        retorno_medio_5d = sum([r['retorno_real_5d'] for r in investimentos_feitos]) / len(investimentos_feitos)
    else:
        lucro_total_1d = 0
        lucro_total_5d = 0
        taxa_sucesso = 0
        retorno_medio_5d = 0

    # Confiança média
    confianca_media = sum([r['confianca'] for r in resultados_validos]) / total_simulacoes

    return {
        'total_simulacoes': total_simulacoes,
        'total_investimentos': total_investimentos,
        'acuracia_direcao_1d': acuracia_1d,
        'acuracia_direcao_5d': acuracia_5d,
        'lucro_total_1d': lucro_total_1d,
        'lucro_total_5d': lucro_total_5d,
        'taxa_sucesso_investimentos': taxa_sucesso,
        'retorno_medio_5d': retorno_medio_5d,
        'confianca_media': confianca_media,
        'resultados_detalhados': resultados_validos
    }

# Geração de relatório de previsão
def gerar_relatorio_previsao(dados_acao, previsoes_ensemble, datas_previsao, detalhes_previsoes):
    try:
        # Extração de dados básicos
        ticker = dados_acao['ticker']
        nome = dados_acao['nome']
        preco_atual = dados_acao['preco']
        variacao_atual = dados_acao['variacao']
        
        # Cálculos de previsão
        preco_1_semana = previsoes_ensemble[4]
        preco_2_semanas = previsoes_ensemble[13]
        variacao_1_semana = ((preco_1_semana - preco_atual) / preco_atual) * 100
        variacao_2_semanas = ((preco_2_semanas - preco_atual) / preco_atual) * 100
        
        tendencia = "ALTA" if preco_2_semanas > preco_atual else "BAIXA"
        
        # Cálculo de confiança baseado na dispersão dos modelos
        dispersao = np.std([detalhes_previsoes[modelo][-1] for modelo in detalhes_previsoes.keys()])
        confianca = max(60, min(90, 85 - (dispersao / preco_atual * 100)))
        
        agora = datetime.now().strftime('%d/%m/%Y às %H:%M')
        
        return f"""
================================================================================
                      RELATÓRIO DE PREVISÃO - MARKETMIND AI
================================================================================

Ação: {ticker} - {nome}
Data: {agora}
Método: Ensemble de Machine Learning

================================================================================
PREVISÕES
================================================================================

Preço Atual: R$ {preco_atual:.2f} ({variacao_atual:+.2f}%)

1 Semana: R$ {preco_1_semana:.2f} ({variacao_1_semana:+.2f}%)
2 Semanas: R$ {preco_2_semanas:.2f} ({variacao_2_semanas:+.2f}%)

Tendência: {tendencia}
Confiança: {confianca:.0f}%

================================================================================
DETALHES POR MODELO
================================================================================

{chr(10).join([f'{modelo}: R$ {detalhes_previsoes[modelo][-1]:.2f}' for modelo in detalhes_previsoes.keys()])}

================================================================================
AVISO LEGAL
================================================================================

Este relatório é gerado por algoritmos de Machine Learning para fins 
EXCLUSIVAMENTE EDUCACIONAIS. NÃO constitui recomendação de investimento.

Sempre consulte profissionais qualificados antes de investir.

================================================================================
        """
        
    except Exception as e:
        return f"Erro ao gerar relatório: {str(e)}"

# Criação de gráficos
def criar_grafico_com_previsao(historico, ticker, previsoes_ensemble=None, datas_previsao=None, detalhes_previsoes=None):
    fig = go.Figure()
    
    if historico is not None and not historico.empty:
        # Linha histórica
        fig.add_trace(go.Scatter(
            x=historico.index,
            y=historico['Close'],
            mode='lines',
            name='Histórico Real',
            line=dict(color='#00d4ff', width=3),
            hovertemplate='<b>R$ %{y:.2f}</b><br>%{x}<extra></extra>'
        ))
        
        # Previsões
        if previsoes_ensemble is not None and datas_previsao is not None:
            fig.add_trace(go.Scatter(
                x=datas_previsao,
                y=previsoes_ensemble,
                mode='lines+markers',
                name='Previsão Ensemble',
                line=dict(color='#ff4444', width=3),
                marker=dict(color='#ff4444', size=8),
                hovertemplate='<b>Ensemble</b><br><b>R$ %{y:.2f}</b><br>%{x|%d/%m/%Y}<extra></extra>'
            ))
            
            # Linha conectora
            ultimo_preco = historico['Close'].iloc[-1]
            ultima_data = historico.index[-1]
            
            fig.add_trace(go.Scatter(
                x=[ultima_data, datas_previsao[0]],
                y=[ultimo_preco, previsoes_ensemble[0]],
                mode='lines',
                line=dict(color='#ff4444', width=2, dash='dash'),
                showlegend=False,
                hovertemplate='<extra></extra>'
            ))
    
    # Layout do gráfico
    fig.update_layout(
        title=f'{ticker} - Análise e Previsão',
        xaxis_title='Data',
        yaxis_title='Preço (R$)',
        height=500,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font_color='#00d4ff',
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)', color='white')
    )
    
    return fig

# Estados da sessão
if 'mostrar_dados' not in st.session_state:
    st.session_state.mostrar_dados = False
if 'dados_acao' not in st.session_state:
    st.session_state.dados_acao = None
if 'mostrar_previsao' not in st.session_state:
    st.session_state.mostrar_previsao = False
if 'tela_atual' not in st.session_state:
    st.session_state.tela_atual = 'login'

# Telas de interface
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

# Lógica principal da aplicação
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
        # Tela inicial - busca de ações
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

    elif st.session_state.mostrar_dados and not st.session_state.mostrar_previsao:
        # Tela de dados da ação
        dados = st.session_state.dados_acao
        
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

    elif st.session_state.mostrar_previsao:
        # Tela de previsão
        dados = st.session_state.dados_acao
        
        st.markdown("<div class='tela-dados'>", unsafe_allow_html=True)
        
        col_titulo, col_fav, col_volta, col_nova = st.columns([2, 1, 1, 1])
        with col_titulo:
            st.markdown("<h4 class='section-title'>Previsão com IA</h4>", unsafe_allow_html=True)
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
        with col_volta:
            if st.button("Voltar", type="secondary", use_container_width=True):
                st.session_state.mostrar_previsao = False
                st.rerun()
        with col_nova:
            if st.button("Nova Análise", type="secondary", use_container_width=True):
                st.session_state.mostrar_dados = False
                st.session_state.mostrar_previsao = False
                st.session_state.dados_acao = None
                st.rerun()

        # Gerar previsões se necessário
        if ('previsoes_ensemble' not in st.session_state or 
            st.session_state.get('ticker_previsao') != dados['ticker']):
            
            with st.spinner('Gerando previsões com IA...'):
                previsoes_ensemble, datas_previsao, detalhes_previsoes, confianca_ml, erro_ml = gerar_previsao_acao(dados)
            
            st.session_state.previsoes_ensemble = previsoes_ensemble
            st.session_state.datas_previsao = datas_previsao
            st.session_state.detalhes_previsoes = detalhes_previsoes
            st.session_state.confianca_ml = confianca_ml
            st.session_state.erro_ml = erro_ml
            st.session_state.ticker_previsao = dados['ticker']
        else:
            previsoes_ensemble = st.session_state.previsoes_ensemble
            datas_previsao = st.session_state.datas_previsao
            detalhes_previsoes = st.session_state.detalhes_previsoes
            confianca_ml = st.session_state.confianca_ml
            erro_ml = st.session_state.erro_ml
        
        if erro_ml:
            st.error(f"Erro na previsão: {erro_ml}")
            st.info("A previsão requer pelo menos 25 dias de histórico.")
        else:
            preco_atual = dados['preco']
            preco_1_dia = previsoes_ensemble[0]
            preco_5_dias = previsoes_ensemble[4]

            variacao_1_dia = ((preco_1_dia - preco_atual) / preco_atual) * 100
            variacao_5_dias = ((preco_5_dias - preco_atual) / preco_atual) * 100
            
            st.success("Previsões geradas com sucesso!")
            
            # Métricas de previsão
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Próximo Dia", f"R$ {preco_1_dia:.2f}", f"{variacao_1_dia:+.2f}%")
            with col2:
                st.metric("5 Dias", f"R$ {preco_5_dias:.2f}", f"{variacao_5_dias:+.2f}%")
            with col3:
                tendencia = "Alta" if preco_5_dias > preco_atual else "Baixa"
                st.metric("Tendência", tendencia)
            with col4:
                st.metric("Confiança", f"{confianca_ml*100:.0f}%")
            

            # Espaço antes do gráfico
            st.markdown("<br>", unsafe_allow_html=True)

            # Gráfico com previsões
            fig = criar_grafico_com_previsao(dados['historico'], dados['ticker'],
                                           previsoes_ensemble, datas_previsao, detalhes_previsoes)
            st.plotly_chart(fig, use_container_width=True)
            
            # Relatório
            col_rel1, col_rel2, col_rel3 = st.columns([1, 2, 1])
            with col_rel2:
                if st.button("📄 Gerar Relatório", type="primary", use_container_width=True):
                    relatorio = gerar_relatorio_previsao(dados, previsoes_ensemble, datas_previsao, detalhes_previsoes)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    nome_arquivo = f"Relatorio_{dados['ticker']}_{timestamp}.txt"
                    
                    st.download_button(
                        label="💾 Download Relatório",
                        data=relatorio,
                        file_name=nome_arquivo,
                        mime="text/plain",
                        use_container_width=True
                    )

        st.markdown("</div>", unsafe_allow_html=True)