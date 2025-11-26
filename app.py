import streamlit as st
import pandas as pd
import numpy as np
import requests
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
        if len(df) < 40:
            return None, None, None, f"Histórico insuficiente: {len(df)} dias (mínimo: 40)"

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

        if len(df) < 30:
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

# ============================================================================
# NOVO SISTEMA: Modelo GRU com Validação Temporal e Quantificação de Incerteza
# ============================================================================

def criar_sequencias_temporais(df, window_size=20, forecast_horizon=5):
    """
    Cria sequências temporais (sliding windows) para modelos recorrentes (GRU/LSTM)

    Args:
        df: DataFrame com features já calculadas
        window_size: Janela de lookback (dias passados para considerar)
        forecast_horizon: Dias futuros para prever

    Returns:
        X: Array 3D (samples, timesteps, features)
        y: Array 2D (samples, forecast_horizon) - retornos futuros
        feature_names: Lista de features usadas
        scaler: Scaler fitado (para inversão se necessário)
    """
    try:
        feature_cols = [
            'Return_1d', 'Return_3d', 'Return_5d',
            'RSI', 'MACD', 'MACD_histogram',
            'BB_width', 'BB_position',
            'Volatility_10d', 'Volatility_20d',
            'Volume_ratio', 'High_Low_ratio', 'Open_Close_ratio',
            'Price_above_SMA20', 'SMA_trend'
        ]

        # Normalizar features (crítico para redes neurais)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df[feature_cols].values)

        X = []
        y = []

        for i in range(window_size, len(df) - forecast_horizon):
            # Janela de features (últimos window_size dias)
            X.append(features_scaled[i-window_size:i])

            # Target: retornos dos próximos forecast_horizon dias
            current_price = df['Close'].iloc[i]
            future_returns = []

            for j in range(1, forecast_horizon + 1):
                future_price = df['Close'].iloc[i + j]
                ret = (future_price - current_price) / current_price
                future_returns.append(ret)

            y.append(future_returns)

        X = np.array(X)
        y = np.array(y)

        return X, y, feature_cols, scaler, None

    except Exception as e:
        return None, None, None, None, f"Erro ao criar sequências: {str(e)}"

def criar_modelo_gru(input_shape, forecast_horizon=5, units=50):
    """
    Cria modelo GRU com regularização forte para evitar overfitting

    Args:
        input_shape: Tuple (timesteps, features)
        forecast_horizon: Número de dias a prever
        units: Unidades na primeira camada GRU

    Returns:
        Modelo Keras compilado
    """
    model = Sequential([
        GRU(units, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),  # Dropout alto para regularização
        GRU(units // 2, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(forecast_horizon)  # Output: 5 retornos futuros
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model

def walk_forward_split(X, y, n_splits=5, test_size=10):
    """
    Walk-Forward Validation: treina com passado crescente, testa no futuro

    Exemplo com 100 samples, n_splits=5, test_size=10:
    Split 1: Train [0:50],  Test [50:60]
    Split 2: Train [0:60],  Test [60:70]
    Split 3: Train [0:70],  Test [70:80]
    Split 4: Train [0:80],  Test [80:90]
    Split 5: Train [0:90],  Test [90:100]
    """
    n_samples = len(X)
    splits = []

    # Tamanho inicial de treino (pelo menos 50% dos dados)
    initial_train_size = max(int(n_samples * 0.5), n_samples - (n_splits * test_size))

    for i in range(n_splits):
        train_end = initial_train_size + (i * test_size)
        test_start = train_end
        test_end = min(test_start + test_size, n_samples)

        if test_end > n_samples:
            break

        train_idx = list(range(0, train_end))
        test_idx = list(range(test_start, test_end))

        if len(test_idx) >= 5:  # Mínimo para teste
            splits.append((train_idx, test_idx))

    return splits

def prever_com_incerteza(model, X_input, n_iter=50):
    """
    Monte Carlo Dropout: faz múltiplas previsões com dropout ativo
    para quantificar incerteza epistêmica

    Args:
        model: Modelo Keras com Dropout
        X_input: Input para previsão
        n_iter: Número de iterações (quanto maior, mais preciso)

    Returns:
        mean_pred: Previsão média
        std_pred: Desvio padrão (incerteza)
        lower_bound: Limite inferior (95% confiança)
        upper_bound: Limite superior (95% confiança)
    """
    previsoes = []

    for _ in range(n_iter):
        # training=True mantém Dropout ativo durante inferência
        pred = model(X_input, training=True)
        previsoes.append(pred.numpy())

    previsoes = np.array(previsoes)

    # Estatísticas
    mean_pred = np.mean(previsoes, axis=0)
    std_pred = np.std(previsoes, axis=0)

    # Intervalos de confiança 95% (2.5% e 97.5% percentis)
    lower_bound = np.percentile(previsoes, 2.5, axis=0)
    upper_bound = np.percentile(previsoes, 97.5, axis=0)

    return mean_pred, std_pred, lower_bound, upper_bound

def treinar_modelo_gru_temporal(X, y, validation_split=0.2):
    """
    Treina modelo GRU com early stopping e redução de learning rate

    Args:
        X: Features (samples, timesteps, features)
        y: Targets (samples, forecast_horizon)
        validation_split: Fração para validação

    Returns:
        model: Modelo treinado
        history: Histórico de treinamento
        mae_val: MAE no conjunto de validação
    """
    # Criar modelo
    model = criar_modelo_gru(
        input_shape=(X.shape[1], X.shape[2]),
        forecast_horizon=y.shape[1],
        units=50
    )

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=0
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=0.00001,
        verbose=0
    )

    # Treinar
    history = model.fit(
        X, y,
        epochs=100,
        batch_size=16,
        validation_split=validation_split,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )

    # Avaliar no conjunto de validação
    val_split_idx = int(len(X) * (1 - validation_split))
    X_val = X[val_split_idx:]
    y_val = y[val_split_idx:]

    y_pred_val = model.predict(X_val, verbose=0)
    mae_val = mean_absolute_error(y_val.flatten(), y_pred_val.flatten())

    return model, history, mae_val

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

# Nova função principal para previsões com GRU
def gerar_previsao_acao(dados_acao):
    """
    Função principal com modelo GRU temporal e quantificação de incerteza

    Returns:
        previsoes: Array com 5 preços futuros
        datas: Datas correspondentes
        detalhes: Dict com lower_bound, upper_bound, volatilidade
        confianca: Confiança realista (0.3-0.65)
        erro: Mensagem de erro (None se sucesso)
    """
    try:
        historico = dados_acao.get('historico')
        if historico is None or historico.empty:
            return None, None, None, None, "Dados históricos não disponíveis"

        if len(historico) < 50:
            return None, None, None, None, f"Histórico insuficiente: {len(historico)} dias (mínimo: 50)"

        # Usar a função existente preparar_dados_financeiros para adicionar features
        _, _, df, erro_prep = preparar_dados_financeiros(historico, dias_previsao=5)

        if erro_prep:
            return None, None, None, None, erro_prep

        if df is None or len(df) < 30:
            return None, None, None, None, "Dados insuficientes após limpeza"

        # Criar sequências temporais para GRU
        # Ajustar window_size baseado na quantidade de dados disponível
        if len(df) < 50:
            window_size = 10  # Janela menor para poucos dados
        else:
            window_size = 15  # Janela média (mais sequências que 20)

        X, y, feature_names, scaler, erro_seq = criar_sequencias_temporais(
            df, window_size=window_size, forecast_horizon=5
        )

        if erro_seq:
            return None, None, None, None, erro_seq

        if len(X) < 20:
            return None, None, None, None, f"Sequências insuficientes: {len(X)} (mínimo: 20)"

        # Treinar modelo GRU com toda a sequência temporal
        # (no mundo real, usa todo o passado disponível)
        # Ajustar validation_split baseado no tamanho dos dados
        if len(X) < 40:
            validation_split = 0.15  # Menos validação para datasets pequenos
        else:
            validation_split = 0.2

        model, history, mae_val = treinar_modelo_gru_temporal(X, y, validation_split=validation_split)

        # Pegar última sequência para fazer previsão
        ultima_sequencia = X[-1:]  # Shape: (1, 20, 15)

        # Prever COM incerteza (Monte Carlo Dropout)
        mean_pred, std_pred, lower_bound, upper_bound = prever_com_incerteza(
            model, ultima_sequencia, n_iter=50
        )

        # Converter retornos para preços
        preco_atual = dados_acao['preco']
        previsoes = []
        lower_prices = []
        upper_prices = []

        for i in range(5):
            # Previsão média
            preco = preco_atual * (1 + mean_pred[0][i])
            previsoes.append(preco)

            # Limites do intervalo de confiança
            lower_p = preco_atual * (1 + lower_bound[0][i])
            upper_p = preco_atual * (1 + upper_bound[0][i])

            lower_prices.append(lower_p)
            upper_prices.append(upper_p)

        previsoes = np.array(previsoes)
        lower_prices = np.array(lower_prices)
        upper_prices = np.array(upper_prices)

        # Calcular CONFIANÇA REALISTA baseada na volatilidade das previsões
        volatilidade_previsao = np.mean(std_pred)

        # Confiança: 30-65% (realista para mercado financeiro)
        # Volatilidade alta = confiança baixa
        confianca = max(0.30, min(0.65, 1.0 - volatilidade_previsao * 10))

        # Ajustar confiança se MAE de validação for alto
        if mae_val > 0.03:  # MAE > 3%
            confianca = max(0.30, confianca * 0.8)

        # Gerar datas de previsão (apenas dias úteis)
        datas_previsao = []
        data_atual = df.index.max()
        dias_adicionados = 0

        while dias_adicionados < 5:
            data_atual += timedelta(days=1)
            if data_atual.weekday() < 5:  # Segunda a sexta
                datas_previsao.append(data_atual)
                dias_adicionados += 1

        # Detalhes incluem intervalos de confiança
        detalhes = {
            'lower_bound': lower_prices,
            'upper_bound': upper_prices,
            'volatilidade': volatilidade_previsao,
            'mae_val': mae_val,
            'std_pred': std_pred,
            'epochs_trained': len(history.history['loss'])
        }

        return previsoes, datas_previsao, detalhes, confianca, None

    except Exception as e:
        import traceback
        erro_detalhado = f"Erro na previsão: {str(e)}\n{traceback.format_exc()}"
        return None, None, None, None, erro_detalhado

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

# Geração de relatório de previsão em PDF
def gerar_relatorio_pdf(dados_acao, previsoes_ensemble, datas_previsao, detalhes_previsoes, confianca_geral):
    """
    Gera um relatório PDF formatado e profissional com as previsões
    """
    try:
        # Buffer para o PDF
        buffer = BytesIO()

        # Criar documento PDF
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=50,
        )

        # Container para os elementos
        elementos = []

        # Estilos
        estilos = getSampleStyleSheet()

        # Estilo customizado para título
        estilo_titulo = ParagraphStyle(
            'CustomTitle',
            parent=estilos['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#00d4ff'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )

        # Estilo para subtítulos
        estilo_subtitulo = ParagraphStyle(
            'CustomHeading',
            parent=estilos['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#00d4ff'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )

        # Estilo para texto normal
        estilo_normal = ParagraphStyle(
            'CustomBody',
            parent=estilos['Normal'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=10,
        )

        # Estilo para destaque
        estilo_destaque = ParagraphStyle(
            'CustomHighlight',
            parent=estilos['Normal'],
            fontSize=12,
            textColor=colors.HexColor('#00d4ff'),
            fontName='Helvetica-Bold',
            spaceAfter=8,
        )

        # ====================
        # CABEÇALHO
        # ====================
        titulo = Paragraph("RELATÓRIO DE PREVISÃO", estilo_titulo)
        elementos.append(titulo)

        subtitulo = Paragraph("MarketMind AI - Análise Preditiva de Ações", estilo_normal)
        elementos.append(subtitulo)
        elementos.append(Spacer(1, 20))

        # Linha horizontal
        linha_data = [['', '', '']]
        linha_table = Table(linha_data, colWidths=[6.5*inch])
        linha_table.setStyle(TableStyle([
            ('LINEABOVE', (0,0), (-1,0), 2, colors.HexColor('#00d4ff')),
        ]))
        elementos.append(linha_table)
        elementos.append(Spacer(1, 20))

        # ====================
        # INFORMAÇÕES BÁSICAS
        # ====================
        ticker = dados_acao['ticker']
        nome = dados_acao['nome']
        preco_atual = dados_acao['preco']
        variacao_atual = dados_acao['variacao']
        agora = datetime.now().strftime('%d/%m/%Y às %H:%M')

        info_basica_data = [
            ['Ação:', f"{ticker} - {nome}"],
            ['Data do Relatório:', agora],
            ['Método:', 'Modelo GRU Temporal com Monte Carlo Dropout'],
            ['Horizonte:', '5 dias úteis'],
        ]

        info_table = Table(info_basica_data, colWidths=[2*inch, 4.5*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#1e2130')),
            ('BACKGROUND', (1, 0), (1, -1), colors.white),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.white),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        elementos.append(info_table)
        elementos.append(Spacer(1, 25))

        # ====================
        # PREVISÕES PRINCIPAIS
        # ====================
        elementos.append(Paragraph("PREVISÕES", estilo_subtitulo))

        preco_1_dia = previsoes_ensemble[0]
        preco_5_dias = previsoes_ensemble[4]
        variacao_1_dia = ((preco_1_dia - preco_atual) / preco_atual) * 100
        variacao_5_dias = ((preco_5_dias - preco_atual) / preco_atual) * 100

        # Extrair intervalos de confiança
        lower_1_dia = detalhes_previsoes.get('lower_bound', [preco_1_dia])[0]
        upper_1_dia = detalhes_previsoes.get('upper_bound', [preco_1_dia])[0]
        lower_5_dias = detalhes_previsoes.get('lower_bound', [0]*5)[4]
        upper_5_dias = detalhes_previsoes.get('upper_bound', [0]*5)[4]

        tendencia = "ALTA ↑" if preco_5_dias > preco_atual else "BAIXA ↓"
        cor_tendencia = colors.green if preco_5_dias > preco_atual else colors.red
        confianca_pct = confianca_geral * 100

        # Determinar interpretação de confiança
        if confianca_pct >= 55:
            interpretacao_conf = "Moderada"
        elif confianca_pct >= 40:
            interpretacao_conf = "Baixa"
        else:
            interpretacao_conf = "Muito Baixa"

        previsoes_data = [
            ['Métrica', 'Valor', 'Intervalo 95%'],
            ['Preço Atual', f'R$ {preco_atual:.2f}', f'{variacao_atual:+.2f}%'],
            ['Previsão 1 Dia', f'R$ {preco_1_dia:.2f}', f'R$ {lower_1_dia:.2f} - R$ {upper_1_dia:.2f}'],
            ['Previsão 5 Dias', f'R$ {preco_5_dias:.2f}', f'R$ {lower_5_dias:.2f} - R$ {upper_5_dias:.2f}'],
            ['Tendência', tendencia, '-'],
            ['Confiança do Modelo', f'{confianca_pct:.0f}% ({interpretacao_conf})', '-'],
        ]

        prev_table = Table(previsoes_data, colWidths=[2.2*inch, 2.2*inch, 2.1*inch])
        prev_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00d4ff')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))
        elementos.append(prev_table)
        elementos.append(Spacer(1, 25))

        # ====================
        # MÉTRICAS DO MODELO
        # ====================
        elementos.append(Paragraph("MÉTRICAS DO MODELO GRU", estilo_subtitulo))

        # Extrair métricas técnicas se disponíveis
        mae_val = detalhes_previsoes.get('mae_val', 0)
        volatilidade = detalhes_previsoes.get('volatilidade', 0)
        epochs_trained = detalhes_previsoes.get('epochs_trained', 0)

        metricas_data = [
            ['Métrica', 'Valor'],
            ['MAE de Validação', f'{mae_val:.4f}' if mae_val > 0 else 'N/A'],
            ['Volatilidade da Previsão', f'{volatilidade:.4f}' if volatilidade > 0 else 'N/A'],
            ['Épocas Treinadas', f'{epochs_trained}' if epochs_trained > 0 else 'N/A'],
            ['Iterações Monte Carlo', '50'],
        ]

        metricas_table = Table(metricas_data, colWidths=[3.5*inch, 3*inch])
        metricas_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e2130')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        elementos.append(metricas_table)
        elementos.append(Spacer(1, 25))

        # ====================
        # GRÁFICO DE PREVISÃO
        # ====================
        elementos.append(Paragraph("VISUALIZAÇÃO GRÁFICA", estilo_subtitulo))

        # Criar gráfico com matplotlib
        fig, ax = plt.subplots(figsize=(8, 4))

        # Histórico
        historico = dados_acao.get('historico')
        if historico is not None and not historico.empty:
            # Pegar últimos 30 dias
            hist_recente = historico.tail(30)
            ax.plot(hist_recente.index, hist_recente['Close'],
                   color='#00d4ff', linewidth=2, label='Histórico')

        # Banda de confiança se disponível
        if 'lower_bound' in detalhes_previsoes and 'upper_bound' in detalhes_previsoes:
            lower_bound = detalhes_previsoes['lower_bound']
            upper_bound = detalhes_previsoes['upper_bound']

            ax.fill_between(datas_previsao, lower_bound, upper_bound,
                           color='#ff4444', alpha=0.2, label='IC 95%')

        # Previsões (média)
        ax.plot(datas_previsao, previsoes_ensemble,
               color='#ff4444', linewidth=2, marker='o',
               markersize=6, label='Previsão GRU (Média)')

        ax.set_xlabel('Data', fontsize=10)
        ax.set_ylabel('Preço (R$)', fontsize=10)
        ax.set_title(f'{ticker} - Histórico e Previsão com Intervalos de Confiança', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        # Salvar gráfico em buffer
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()

        # Adicionar imagem ao PDF
        from reportlab.platypus import Image
        img = Image(img_buffer, width=6*inch, height=3*inch)
        elementos.append(img)
        elementos.append(Spacer(1, 25))

        # ====================
        # INTERPRETAÇÃO
        # ====================
        elementos.append(Paragraph("INTERPRETAÇÃO DOS RESULTADOS", estilo_subtitulo))

        if variacao_5_dias > 5:
            interpretacao = f"A previsão indica uma <b>forte tendência de alta</b> para os próximos 5 dias, com valorização estimada de {variacao_5_dias:.2f}%. Esta é uma sinalização positiva segundo o modelo ensemble."
        elif variacao_5_dias > 0:
            interpretacao = f"A previsão indica uma <b>tendência moderada de alta</b> para os próximos 5 dias, com valorização estimada de {variacao_5_dias:.2f}%."
        elif variacao_5_dias > -5:
            interpretacao = f"A previsão indica uma <b>tendência moderada de baixa</b> para os próximos 5 dias, com desvalorização estimada de {variacao_5_dias:.2f}%."
        else:
            interpretacao = f"A previsão indica uma <b>forte tendência de baixa</b> para os próximos 5 dias, com desvalorização estimada de {variacao_5_dias:.2f}%."

        elementos.append(Paragraph(interpretacao, estilo_normal))
        elementos.append(Spacer(1, 10))

        confianca_texto = f"O modelo GRU apresenta <b>{confianca_pct:.0f}% de confiança</b> nas previsões. "
        if confianca_pct >= 55:
            confianca_texto += "Este é um nível de confiança <b>moderado</b> para previsões de mercado. "
        elif confianca_pct >= 40:
            confianca_texto += "Este é um nível de confiança <b>baixo</b>, típico para previsões de ações. "
        else:
            confianca_texto += "Este é um nível de confiança <b>muito baixo</b>, sugere alta incerteza. "

        confianca_texto += f"O intervalo de confiança 95% mostra que há 95% de probabilidade do preço real estar dentro do range apresentado."

        elementos.append(Paragraph(confianca_texto, estilo_normal))
        elementos.append(Spacer(1, 15))

        # Adicionar explicação sobre confiança realista
        explicacao_confianca = f"""
        <b>Sobre a Confiança:</b> O modelo GRU temporal utiliza Monte Carlo Dropout para quantificar incerteza.
        Valores de confiança entre 30-65% são <b>normais e realistas</b> para previsão de ações devido à
        alta complexidade e volatilidade do mercado. Previsões de 50-55% de acurácia direcional são apenas
        <b>marginalmente melhores que sorte (50%)</b>, o que reflete a eficiência do mercado.
        """
        elementos.append(Paragraph(explicacao_confianca, estilo_normal))
        elementos.append(Spacer(1, 30))

        # ====================
        # AVISO LEGAL
        # ====================
        elementos.append(Paragraph("AVISO LEGAL E DISCLAIMER", estilo_subtitulo))

        aviso = f"""
        <b>IMPORTANTE:</b> Este relatório é gerado por um modelo GRU temporal de Machine Learning para fins
        <b>EXCLUSIVAMENTE EDUCACIONAIS E INFORMATIVOS</b>. As previsões apresentadas têm confiança <b>{interpretacao_conf.lower()}</b>
        ({confianca_pct:.0f}%), o que indica <b>alta incerteza</b>.
        <br/><br/>
        Uma confiança de {confianca_pct:.0f}% significa que o modelo tem <b>incerteza significativa</b> nas previsões.
        Modelos de previsão de ações com 50-55% de acurácia direcional são apenas <b>marginalmente melhores que sorte (50%)</b>,
        refletindo a natureza altamente eficiente e imprevisível do mercado de ações.
        <br/><br/>
        Este documento <b>NÃO constitui</b> recomendação de investimento, oferta de compra ou venda de valores mobiliários.
        Investimentos em ações envolvem riscos significativos de perda de capital.
        <br/><br/>
        Desempenho passado não é garantia de resultados futuros. Sempre consulte profissionais qualificados
        (analistas, assessores de investimentos certificados) antes de tomar qualquer decisão de investimento.
        <br/><br/>
        O MarketMind e seus desenvolvedores não se responsabilizam por perdas ou danos decorrentes do uso deste relatório.
        """

        elementos.append(Paragraph(aviso, estilo_normal))
        elementos.append(Spacer(1, 20))

        # Rodapé
        rodape_data = [['', '']]
        rodape_table = Table(rodape_data, colWidths=[6.5*inch])
        rodape_table.setStyle(TableStyle([
            ('LINEABOVE', (0,0), (-1,0), 1, colors.grey),
        ]))
        elementos.append(rodape_table)

        rodape_texto = Paragraph(
            f"<i>Gerado por MarketMind AI em {agora} | www.marketmind.com</i>",
            ParagraphStyle('Footer', parent=estilos['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
        )
        elementos.append(rodape_texto)

        # Construir PDF
        doc.build(elementos)

        # Retornar buffer
        buffer.seek(0)
        return buffer

    except Exception as e:
        st.error(f"Erro ao gerar PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

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
            # Bandas de confiança (se disponíveis)
            if detalhes_previsoes and 'lower_bound' in detalhes_previsoes and 'upper_bound' in detalhes_previsoes:
                lower_bound = detalhes_previsoes['lower_bound']
                upper_bound = detalhes_previsoes['upper_bound']

                # Banda superior
                fig.add_trace(go.Scatter(
                    x=datas_previsao,
                    y=upper_bound,
                    mode='lines',
                    name='IC 95% Superior',
                    line=dict(color='rgba(255,68,68,0.3)', width=1),
                    showlegend=True,
                    hovertemplate='<b>Limite Superior</b><br><b>R$ %{y:.2f}</b><br>%{x|%d/%m/%Y}<extra></extra>'
                ))

                # Banda inferior
                fig.add_trace(go.Scatter(
                    x=datas_previsao,
                    y=lower_bound,
                    mode='lines',
                    name='IC 95% Inferior',
                    line=dict(color='rgba(255,68,68,0.3)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(255,68,68,0.2)',
                    showlegend=True,
                    hovertemplate='<b>Limite Inferior</b><br><b>R$ %{y:.2f}</b><br>%{x|%d/%m/%Y}<extra></extra>'
                ))

            # Previsão média (linha principal)
            fig.add_trace(go.Scatter(
                x=datas_previsao,
                y=previsoes_ensemble,
                mode='lines+markers',
                name='Previsão GRU (Média)',
                line=dict(color='#ff4444', width=3),
                marker=dict(color='#ff4444', size=8),
                hovertemplate='<b>GRU</b><br><b>R$ %{y:.2f}</b><br>%{x|%d/%m/%Y}<extra></extra>'
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

            # Extrair bounds de confiança se disponíveis
            lower_1_dia = detalhes_previsoes.get('lower_bound', [preco_1_dia])[0]
            upper_1_dia = detalhes_previsoes.get('upper_bound', [preco_1_dia])[0]
            lower_5_dias = detalhes_previsoes.get('lower_bound', [0]*5)[4]
            upper_5_dias = detalhes_previsoes.get('upper_bound', [0]*5)[4]

            variacao_1_dia = ((preco_1_dia - preco_atual) / preco_atual) * 100
            variacao_5_dias = ((preco_5_dias - preco_atual) / preco_atual) * 100

            st.success("Previsões geradas com sucesso!")

            # Métricas de previsão com intervalos de confiança
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Próximo Dia", f"R$ {preco_1_dia:.2f}", f"{variacao_1_dia:+.2f}%")
                st.caption(f"📊 IC 95%: R$ {lower_1_dia:.2f} - R$ {upper_1_dia:.2f}")
            with col2:
                st.metric("5 Dias", f"R$ {preco_5_dias:.2f}", f"{variacao_5_dias:+.2f}%")
                st.caption(f"📊 IC 95%: R$ {lower_5_dias:.2f} - R$ {upper_5_dias:.2f}")
            with col3:
                tendencia = "Alta" if preco_5_dias > preco_atual else "Baixa"
                st.metric("Tendência", tendencia)
            with col4:
                # Confiança com interpretação
                confianca_pct = confianca_ml * 100
                if confianca_pct >= 55:
                    interpretacao = "Moderada"
                    emoji = "🟡"
                elif confianca_pct >= 40:
                    interpretacao = "Baixa"
                    emoji = "🟠"
                else:
                    interpretacao = "Muito Baixa"
                    emoji = "🔴"

                st.metric("Confiança", f"{confianca_pct:.0f}%")
                st.caption(f"{emoji} {interpretacao}")

            # Aviso sobre a confiança realista
            st.info("""
            ℹ️ **Sobre a Confiança**: O modelo GRU temporal reporta confiança **realista** (30-65%).
            Valores entre 40-55% são **normais** para previsão de ações devido à alta complexidade e volatilidade do mercado.
            O intervalo de confiança 95% (IC 95%) mostra o **range de possibilidades** onde o preço real tem 95% de chance de estar.
            """)
            

            # Espaço antes do gráfico
            st.markdown("<br>", unsafe_allow_html=True)

            # Gráfico com previsões
            fig = criar_grafico_com_previsao(dados['historico'], dados['ticker'],
                                           previsoes_ensemble, datas_previsao, detalhes_previsoes)
            st.plotly_chart(fig, use_container_width=True)
            
            # Relatório PDF
            col_rel1, col_rel2, col_rel3 = st.columns([1, 2, 1])
            with col_rel2:
                if st.button("📄 Gerar Relatório PDF", type="primary", use_container_width=True):
                    with st.spinner('Gerando relatório em PDF...'):
                        pdf_buffer = gerar_relatorio_pdf(dados, previsoes_ensemble, datas_previsao,
                                                         detalhes_previsoes, confianca_ml)

                    if pdf_buffer:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        nome_arquivo = f"Relatorio_MarketMind_{dados['ticker']}_{timestamp}.pdf"

                        st.download_button(
                            label="💾 Download Relatório PDF",
                            data=pdf_buffer,
                            file_name=nome_arquivo,
                            mime="application/pdf",
                            use_container_width=True
                        )
                        st.success("✅ Relatório PDF gerado com sucesso!")

        st.markdown("</div>", unsafe_allow_html=True)