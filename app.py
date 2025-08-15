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

# Preparação de dados para machine learning
def preparar_dados_ensemble(historico_df, dias_previsao=14):
    try:
        if historico_df is None or historico_df.empty:
            return None, None, None, None, "Dados históricos insuficientes"
        
        df = historico_df.copy()
        
        # Features técnicas
        df['Close_MA7'] = df['Close'].rolling(window=7).mean()
        df['Close_MA21'] = df['Close'].rolling(window=21).mean()
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_MA7'] = df['Volume'].rolling(window=7).mean()
        df['Volatility'] = df['Close'].rolling(window=14).std()
        df['RSI'] = calcular_rsi(df['Close'])
        df['MACD'] = calcular_macd(df['Close'])
        df['BB_upper'], df['BB_lower'] = calcular_bollinger_bands(df['Close'])
        df['Close_lag1'] = df['Close'].shift(1)
        df['Close_lag2'] = df['Close'].shift(2)
        df['Volume_Change'] = df['Volume'].pct_change()
        
        df = df.dropna()
        
        if len(df) < 25:
            return None, None, None, None, f"Histórico insuficiente: {len(df)} dias (mínimo: 25)"
        
        feature_cols = ['Close_MA7', 'Close_MA21', 'Price_Change', 'Volume_MA7', 
                       'Volatility', 'RSI', 'MACD', 'BB_upper', 'BB_lower',
                       'Close_lag1', 'Close_lag2', 'Volume_Change']
        
        X_traditional = df[feature_cols].values
        
        # Dados para LSTM
        sequence_length = min(15, int(len(df) * 0.4))
        lstm_features = ['Close', 'Volume', 'Close_MA7', 'Close_MA21', 'Price_Change']
        lstm_data = df[lstm_features].values
        
        # Normalização
        scaler_traditional = MinMaxScaler()
        scaler_lstm = MinMaxScaler()
        X_traditional_scaled = scaler_traditional.fit_transform(X_traditional)
        lstm_data_scaled = scaler_lstm.fit_transform(lstm_data)
        
        # Preparação de sequences para treinamento
        y_multi, X_traditional_final, X_lstm_final = [], [], []
        
        for i in range(len(df) - dias_previsao):
            future_prices = df['Close'].iloc[i+1:i+1+dias_previsao].values
            if len(future_prices) == dias_previsao:
                y_multi.append(future_prices)
                X_traditional_final.append(X_traditional_scaled[i])
                
                if i >= sequence_length:
                    X_lstm_final.append(lstm_data_scaled[i-sequence_length:i])
        
        min_samples = min(len(X_traditional_final), len(X_lstm_final))
        if min_samples < 10:
            return None, None, None, None, "Dados insuficientes para treinamento"
        
        return (np.array(X_traditional_final[-min_samples:]),
                np.array(X_lstm_final[-min_samples:]),
                np.array(y_multi[-min_samples:]),
                (scaler_traditional, scaler_lstm),
                None)
        
    except Exception as e:
        return None, None, None, None, f"Erro ao preparar dados: {str(e)}"

# Treinamento de modelos ensemble
def treinar_ensemble_modelos(X_traditional, X_lstm, y_multi):
    try:
        split_idx = max(1, int(0.7 * len(X_traditional)))
        
        X_trad_train, X_trad_test = X_traditional[:split_idx], X_traditional[split_idx:]
        X_lstm_train, X_lstm_test = X_lstm[:split_idx], X_lstm[split_idx:]
        y_train, y_test = y_multi[:split_idx], y_multi[split_idx:]
        
        modelos = {}
        
        # Random Forest para cada dia
        rf_models = []
        for dia in range(14):
            rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
            rf.fit(X_trad_train, y_train[:, dia])
            rf_models.append(rf)
        modelos['RandomForest'] = rf_models
        
        # Gradient Boosting para cada dia
        gb_models = []
        for dia in range(14):
            gb = GradientBoostingRegressor(n_estimators=50, max_depth=6, random_state=42)
            gb.fit(X_trad_train, y_train[:, dia])
            gb_models.append(gb)
        modelos['GradientBoosting'] = gb_models
        
        # Linear Regression para cada dia
        lr_models = []
        for dia in range(14):
            lr = LinearRegression()
            lr.fit(X_trad_train, y_train[:, dia])
            lr_models.append(lr)
        modelos['LinearRegression'] = lr_models
        
        # LSTM
        try:
            if len(X_lstm_train) >= 5:
                lstm_model = Sequential([
                    LSTM(32, return_sequences=False, input_shape=(X_lstm.shape[1], X_lstm.shape[2])),
                    Dropout(0.3),
                    Dense(16, activation='relu'),
                    Dense(14)
                ])
                
                lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                
                epochs = min(20, max(5, len(X_lstm_train)))
                batch_size = max(1, min(4, len(X_lstm_train) // 3))
                
                with st.spinner('Treinando modelo LSTM...'):
                    lstm_model.fit(X_lstm_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
                
                modelos['LSTM'] = lstm_model
            else:
                modelos['LSTM'] = None
        except Exception:
            modelos['LSTM'] = None
        
        return modelos, {}, None
        
    except Exception as e:
        return None, None, f"Erro no treinamento: {str(e)}"

# Previsão ensemble
def fazer_previsao_ensemble(modelos, scalers, historico_df):
    try:
        scaler_traditional, scaler_lstm = scalers
        
        df = historico_df.copy()
        
        # Recalcular features
        df['Close_MA7'] = df['Close'].rolling(window=7).mean()
        df['Close_MA21'] = df['Close'].rolling(window=21).mean()
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_MA7'] = df['Volume'].rolling(window=7).mean()
        df['Volatility'] = df['Close'].rolling(window=14).std()
        df['RSI'] = calcular_rsi(df['Close'])
        df['MACD'] = calcular_macd(df['Close'])
        df['BB_upper'], df['BB_lower'] = calcular_bollinger_bands(df['Close'])
        df['Close_lag1'] = df['Close'].shift(1)
        df['Close_lag2'] = df['Close'].shift(2)
        df['Volume_Change'] = df['Volume'].pct_change()
        
        df = df.dropna()
        
        feature_cols = ['Close_MA7', 'Close_MA21', 'Price_Change', 'Volume_MA7', 
                       'Volatility', 'RSI', 'MACD', 'BB_upper', 'BB_lower',
                       'Close_lag1', 'Close_lag2', 'Volume_Change']
        
        last_features = df[feature_cols].iloc[-1:].values
        last_features_scaled = scaler_traditional.transform(last_features)
        
        # Previsões dos modelos
        previsoes = {}
        
        # Random Forest
        rf_pred = [model.predict(last_features_scaled)[0] for model in modelos['RandomForest']]
        previsoes['RandomForest'] = np.array(rf_pred)
        
        # Gradient Boosting
        gb_pred = [model.predict(last_features_scaled)[0] for model in modelos['GradientBoosting']]
        previsoes['GradientBoosting'] = np.array(gb_pred)
        
        # Linear Regression
        lr_pred = [model.predict(last_features_scaled)[0] for model in modelos['LinearRegression']]
        previsoes['LinearRegression'] = np.array(lr_pred)
        
        # LSTM
        if modelos['LSTM'] is not None:
            lstm_features = ['Close', 'Volume', 'Close_MA7', 'Close_MA21', 'Price_Change']
            sequence_length = min(15, len(df) - 1)
            last_sequence = df[lstm_features].tail(sequence_length).values
            last_sequence_scaled = scaler_lstm.transform(last_sequence)
            last_sequence_scaled = last_sequence_scaled.reshape(1, sequence_length, -1)
            
            lstm_pred = modelos['LSTM'].predict(last_sequence_scaled, verbose=0)[0]
            previsoes['LSTM'] = lstm_pred
        
        # Pesos do ensemble
        if modelos['LSTM'] is not None:
            pesos = {'RandomForest': 0.3, 'GradientBoosting': 0.3, 'LSTM': 0.25, 'LinearRegression': 0.15}
        else:
            pesos = {'RandomForest': 0.4, 'GradientBoosting': 0.4, 'LinearRegression': 0.2}
        
        # Previsão final
        previsao_final = np.zeros(14)
        for modelo, pred in previsoes.items():
            previsao_final += pred * pesos[modelo]
        
        # Datas de previsão (apenas dias úteis)
        datas_previsao = []
        data_atual = historico_df.index.max()
        dias_adicionados = 0
        
        while dias_adicionados < 14:
            data_atual += timedelta(days=1)
            if data_atual.weekday() < 5:
                datas_previsao.append(data_atual)
                dias_adicionados += 1
        
        return previsao_final, datas_previsao, previsoes, None
        
    except Exception as e:
        return None, None, None, f"Erro na previsão: {str(e)}"

def gerar_previsao_acao(dados_acao):
    try:
        historico = dados_acao.get('historico')
        if historico is None or historico.empty:
            return None, None, None, "Dados históricos não disponíveis"
        
        X_trad, X_lstm, y_multi, scalers, erro_prep = preparar_dados_ensemble(historico)
        if erro_prep:
            return None, None, None, erro_prep
        
        modelos, scores, erro_treino = treinar_ensemble_modelos(X_trad, X_lstm, y_multi)
        if erro_treino:
            return None, None, None, erro_treino
        
        previsoes, datas, detalhes, erro_pred = fazer_previsao_ensemble(modelos, scalers, historico)
        if erro_pred:
            return None, None, None, erro_pred
        
        return previsoes, datas, detalhes, None
        
    except Exception as e:
        return None, None, None, f"Erro na previsão: {str(e)}"

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
    st.markdown("<h1 class='main-header'>MarketMind</h1>", unsafe_allow_html=True)
    st.markdown("### Login")
    
    with st.form("login_form"):
        username = st.text_input("Usuário", placeholder="Digite seu usuário")
        password = st.text_input("Senha", type="password", placeholder="Digite sua senha")
        
        col1, col2 = st.columns(2)
        with col1:
            login_button = st.form_submit_button("Entrar", type="primary", use_container_width=True)
        with col2:
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
    st.markdown("<h1 class='main-header'>MarketMind</h1>", unsafe_allow_html=True)
    st.markdown("### Criar Conta")
    
    with st.form("cadastro_form"):
        username = st.text_input("Usuário", placeholder="Nome de usuário")
        email = st.text_input("Email", placeholder="Seu email")
        password = st.text_input("Senha", type="password", placeholder="Sua senha")
        password_confirm = st.text_input("Confirmar Senha", type="password", placeholder="Confirme a senha")
        
        col1, col2 = st.columns(2)
        with col1:
            cadastro_button = st.form_submit_button("Criar Conta", type="primary", use_container_width=True)
        with col2:
            if st.form_submit_button("Voltar", use_container_width=True):
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
                previsoes_ensemble, datas_previsao, detalhes_previsoes, erro_ml = gerar_previsao_acao(dados)
            
            st.session_state.previsoes_ensemble = previsoes_ensemble
            st.session_state.datas_previsao = datas_previsao
            st.session_state.detalhes_previsoes = detalhes_previsoes
            st.session_state.erro_ml = erro_ml
            st.session_state.ticker_previsao = dados['ticker']
        else:
            previsoes_ensemble = st.session_state.previsoes_ensemble
            datas_previsao = st.session_state.datas_previsao
            detalhes_previsoes = st.session_state.detalhes_previsoes
            erro_ml = st.session_state.erro_ml
        
        if erro_ml:
            st.error(f"Erro na previsão: {erro_ml}")
            st.info("A previsão requer pelo menos 25 dias de histórico.")
        else:
            preco_atual = dados['preco']
            preco_1_semana = previsoes_ensemble[4]
            preco_2_semanas = previsoes_ensemble[13]
            
            variacao_1_semana = ((preco_1_semana - preco_atual) / preco_atual) * 100
            variacao_2_semanas = ((preco_2_semanas - preco_atual) / preco_atual) * 100
            
            st.success("Previsões geradas com sucesso!")
            
            # Métricas de previsão
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("1 Semana", f"R$ {preco_1_semana:.2f}", f"{variacao_1_semana:+.2f}%")
            with col2:
                st.metric("2 Semanas", f"R$ {preco_2_semanas:.2f}", f"{variacao_2_semanas:+.2f}%")
            with col3:
                tendencia = "Alta" if preco_2_semanas > preco_atual else "Baixa"
                st.metric("Tendência", tendencia)
            with col4:
                dispersao = np.std([detalhes_previsoes[modelo][-1] for modelo in detalhes_previsoes.keys()])
                confianca = max(60, min(90, 85 - (dispersao / preco_atual * 100)))
                st.metric("Confiança", f"{confianca:.0f}%")
            
            # Detalhes por modelo
            with st.expander("📊 Detalhes por Modelo"):
                cols = st.columns(len(detalhes_previsoes))
                
                icones = {'LSTM': '🧠', 'RandomForest': '🌲', 'GradientBoosting': '🚀', 'LinearRegression': '📈'}
                nomes = {'LSTM': 'LSTM', 'RandomForest': 'Random Forest', 
                        'GradientBoosting': 'Gradient Boost', 'LinearRegression': 'Linear Reg.'}
                
                for i, modelo in enumerate(detalhes_previsoes.keys()):
                    with cols[i]:
                        pred = detalhes_previsoes[modelo][-1]
                        var = ((pred - preco_atual) / preco_atual) * 100
                        icone = icones.get(modelo, '📊')
                        nome = nomes.get(modelo, modelo)
                        st.metric(f"{icone} {nome}", f"R$ {pred:.2f}", f"{var:+.2f}%")

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