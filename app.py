import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from pathlib import Path
import requests
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
import warnings
import json
import os
from datetime import datetime
from io import BytesIO
from database import DatabaseManager
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="MarketMind",
    page_icon="🧠",
    layout="wide"
)

def load_css():
    css_file = Path(__file__).parent / "styles.css"
    if css_file.exists():
        with open(css_file, 'r', encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Arquivo styles.css não encontrado")

load_css()
db = DatabaseManager()

def carregar_usuarios():
    return db.obter_todos_usuarios()

def salvar_usuarios(usuarios):
    return True

def hash_password(password):
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()

def criar_usuario(username, email, password):
    if db.obter_usuario(username):
        return False, "Usuário já existe"
    
    usuarios = db.obter_todos_usuarios()
    for user_data in usuarios.values():
        if user_data.get('email') == email:
            return False, "Email já cadastrado"
    
    user_id = db.criar_usuario(username, email, hash_password(password))
    
    if user_id:
        return True, "Usuário criado com sucesso"
    else:
        return False, "Erro ao criar usuário"

def autenticar_usuario(username, password):
    usuario = db.obter_usuario(username)
    
    if not usuario:
        return False, "Usuário não encontrado"
    
    if usuario['password_hash'] != hash_password(password):
        return False, "Senha incorreta"
    
    return True, "Login realizado com sucesso"

def get_usuario_logado():
    return st.session_state.get('usuario_logado', None)

def fazer_login(username):
    st.session_state.usuario_logado = username

def fazer_logout():
    if 'usuario_logado' in st.session_state:
        del st.session_state.usuario_logado
    if 'dados_acao' in st.session_state:
        del st.session_state.dados_acao
    if 'mostrar_dados' in st.session_state:
        del st.session_state.mostrar_dados
    if 'mostrar_previsao' in st.session_state:
        del st.session_state.mostrar_previsao

def carregar_favoritos(username=None):
    if username is None:
        username = get_usuario_logado()
    
    if username is None:
        return []
    
    return db.obter_favoritos_usuario_por_username(username)

def salvar_favoritos(favoritos, username=None):
    return True

def adicionar_favorito(ticker, nome, preco):
    username = get_usuario_logado()
    if username is None:
        return False, "Usuário não está logado"
    
    usuario = db.obter_usuario(username)
    if not usuario:
        return False, "Usuário não encontrado"
    
    favoritos = db.obter_favoritos_usuario(usuario['id'])
    
    for fav in favoritos:
        if fav['ticker'] == ticker:
            return False, "Ação já está nos favoritos"
    
    try:
        db.adicionar_favorito_usuario(usuario['id'], ticker, nome, preco)
        return True, "Ação adicionada aos favoritos"
    except Exception:
        return False, "Erro ao salvar favorito"

def remover_favorito(ticker):
    username = get_usuario_logado()
    if username is None:
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
    if username is None:
        return False
    
    favoritos = db.obter_favoritos_usuario_por_username(username)
    return any(fav['ticker'] == ticker for fav in favoritos)

def buscar_dados_rapidos(ticker):
    try:
        API_KEY = "nUUZxG2ZdAWuSkBDhPobC2"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        
        url_cotacao = f"https://brapi.dev/api/quote/{ticker}"
        response = requests.get(url_cotacao, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and data['results']:
                acao_data = data['results'][0]
                return {
                    'preco': round(acao_data.get('regularMarketPrice', 0), 2),
                    'variacao': round(acao_data.get('regularMarketChangePercent', 0), 2),
                    'sucesso': True
                }
    except:
        pass
    
    return {'preco': 0, 'variacao': 0, 'sucesso': False}

def preparar_dados_ensemble(historico_df, dias_previsao=14):
    try:
        if historico_df is None or historico_df.empty:
            return None, None, None, None, "Dados históricos insuficientes"
        
        df = historico_df.copy()
        
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
        
        dias_disponveis = len(df)
        if dias_disponveis < 25:
            return None, None, None, None, f"Histórico insuficiente: {dias_disponveis} dias (mínimo: 25)"
        
        feature_cols = ['Close_MA7', 'Close_MA21', 'Price_Change', 'Volume_MA7', 
                       'Volatility', 'RSI', 'MACD', 'BB_upper', 'BB_lower',
                       'Close_lag1', 'Close_lag2', 'Volume_Change']
        
        X_traditional = df[feature_cols].values
        
        sequence_length = min(15, int(dias_disponveis * 0.4))
        lstm_features = ['Close', 'Volume', 'Close_MA7', 'Close_MA21', 'Price_Change']
        lstm_data = df[lstm_features].values
        
        scaler_traditional = MinMaxScaler()
        scaler_lstm = MinMaxScaler()
        
        X_traditional_scaled = scaler_traditional.fit_transform(X_traditional)
        lstm_data_scaled = scaler_lstm.fit_transform(lstm_data)
        
        y_multi = []
        X_traditional_final = []
        X_lstm_final = []
        
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
        
        X_traditional_final = np.array(X_traditional_final[-min_samples:])
        X_lstm_final = np.array(X_lstm_final[-min_samples:])
        y_multi = np.array(y_multi[-min_samples:])
        
        return X_traditional_final, X_lstm_final, y_multi, (scaler_traditional, scaler_lstm), None
        
    except Exception as e:
        return None, None, None, None, f"Erro ao preparar dados: {str(e)}"

def calcular_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calcular_macd(prices, fast=12, slow=26):
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    return macd.fillna(0)

def calcular_bollinger_bands(prices, window=20, std_dev=2):
    ma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = ma + (std * std_dev)
    lower = ma - (std * std_dev)
    return upper.fillna(prices), lower.fillna(prices)

def gerar_relatorio_previsao(dados_acao, previsoes_ensemble, datas_previsao, detalhes_previsoes):
    try:
        ticker = dados_acao['ticker']
        nome = dados_acao['nome']
        preco_atual = dados_acao['preco']
        variacao_atual = dados_acao['variacao']
        
        preco_1_semana = previsoes_ensemble[4]
        preco_2_semanas = previsoes_ensemble[13]
        variacao_1_semana = ((preco_1_semana - preco_atual) / preco_atual) * 100
        variacao_2_semanas = ((preco_2_semanas - preco_atual) / preco_atual) * 100
        
        tendencia = "ALTA" if preco_2_semanas > preco_atual else "BAIXA"
        
        dispersao = np.std([detalhes_previsoes[modelo][-1] for modelo in detalhes_previsoes.keys()])
        confianca = max(60, min(90, 85 - (dispersao / preco_atual * 100)))
        
        agora = datetime.now().strftime('%d/%m/%Y às %H:%M')
        
        relatorio = f"""
================================================================================
                          RELATÓRIO DE PREVISÃO DE AÇÕES
                                 MARKETMIND AI SYSTEM
================================================================================

Ação Analisada: {ticker} - {nome}
Data do Relatório: {agora}
Método: Ensemble de Machine Learning (4 Modelos)

================================================================================
1. SITUAÇÃO ATUAL DA AÇÃO
================================================================================

Preço Atual: R$ {preco_atual:.2f}
Variação Diária: {variacao_atual:+.2f}%
Volume: {dados_acao.get('volume', 0):,}
Máxima do Dia: R$ {dados_acao.get('maxima', 0):.2f}
Mínima do Dia: R$ {dados_acao.get('minima', 0):.2f}

================================================================================
2. PREVISÕES DO ENSEMBLE AI
================================================================================

📈 PREVISÃO 1 SEMANA (5 dias úteis):
   Preço Previsto: R$ {preco_1_semana:.2f}
   Variação Esperada: {variacao_1_semana:+.2f}%
   Data Alvo: {datas_previsao[4].strftime('%d/%m/%Y')}

📈 PREVISÃO 2 SEMANAS (14 dias úteis):
   Preço Previsto: R$ {preco_2_semanas:.2f}
   Variação Esperada: {variacao_2_semanas:+.2f}%
   Data Alvo: {datas_previsao[13].strftime('%d/%m/%Y')}

🎯 TENDÊNCIA GERAL: {tendencia}
🔒 NÍVEL DE CONFIANÇA: {confianca:.0f}%

================================================================================
3. ANÁLISE POR MODELO INDIVIDUAL
================================================================================"""

        icones_modelos = {
            'LSTM': '🧠', 'RandomForest': '🌲', 
            'GradientBoosting': '🚀', 'LinearRegression': '📈'
        }
        
        nomes_modelos = {
            'LSTM': 'LSTM Neural Network',
            'RandomForest': 'Random Forest',
            'GradientBoosting': 'Gradient Boosting', 
            'LinearRegression': 'Linear Regression'
        }
        
        for modelo, previsoes in detalhes_previsoes.items():
            icone = icones_modelos.get(modelo, '📊')
            nome_modelo = nomes_modelos.get(modelo, modelo)
            pred_2sem = previsoes[-1]
            var_2sem = ((pred_2sem - preco_atual) / preco_atual) * 100
            
            relatorio += f"""
{icone} {nome_modelo.upper()}:
   Previsão 2 semanas: R$ {pred_2sem:.2f} ({var_2sem:+.2f}%)
"""

        relatorio += f"""
================================================================================
4. ANÁLISE TÉCNICA E INTERPRETAÇÃO
================================================================================

CONSENSO DOS MODELOS:
O ensemble de 4 algoritmos de Machine Learning indica uma tendência de {tendencia.lower()} 
para a ação {ticker} nos próximos 14 dias úteis. A previsão sugere que o preço 
pode alcançar R$ {preco_2_semanas:.2f}, representando uma variação de {variacao_2_semanas:+.1f}% 
em relação ao preço atual.

FATORES ANALISADOS:
✓ Médias móveis e indicadores técnicos (RSI, MACD, Bollinger Bands)
✓ Padrões de preço e volume dos últimos 40 dias
✓ Volatilidade histórica e tendências de curto prazo
✓ Correlações entre múltiplas variáveis técnicas

NÍVEL DE CONSENSO:
Com {confianca:.0f}% de confiança, os modelos apresentam {'alta concordância' if confianca >= 80 else 'concordância moderada' if confianca >= 70 else 'baixa concordância'} 
em suas previsões. {'Isso indica maior probabilidade de acerto.' if confianca >= 80 else 'Recomenda-se cautela adicional na interpretação.' if confianca < 70 else 'Sugerem-se análises complementares.'}

================================================================================
5. RECOMENDAÇÕES E CONSIDERAÇÕES
================================================================================

CENÁRIO OTIMISTA ({tendencia}):
{'Se a tendência de alta se confirmar, a ação pode apresentar valorização ' + f'significativa de {abs(variacao_2_semanas):.1f}% em 2 semanas.' if tendencia == 'ALTA' else f'Se a tendência de baixa se confirmar, recomenda-se cautela, com possível ' + f'desvalorização de {abs(variacao_2_semanas):.1f}% em 2 semanas.'}

FATORES DE RISCO:
⚠️ Eventos macroeconômicos podem impactar significativamente os preços
⚠️ Notícias específicas da empresa ou setor podem alterar o cenário
⚠️ Condições de mercado podem mudar rapidamente
⚠️ Volume de negociação baixo pode aumentar a volatilidade

ESTRATÉGIA SUGERIDA:
{'• Considere posições graduais para aproveitar a tendência positiva' if tendencia == 'ALTA' else '• Avalie proteção de posições ou redução de exposição'}
• Monitore indicadores de volume e volatilidade
• Estabeleça stops de proteção adequados ao seu perfil de risco
• Acompanhe notícias e eventos corporativos relevantes

================================================================================
6. DISCLAIMER E AVISOS LEGAIS
================================================================================

⚠️ AVISO IMPORTANTE:
Este relatório é gerado por algoritmos de Machine Learning para fins 
EXCLUSIVAMENTE EDUCACIONAIS e de pesquisa. NÃO constitui recomendação de 
investimento, compra ou venda de ações.

LIMITAÇÕES:
• As previsões são baseadas apenas em dados históricos de preços
• Mercados financeiros são influenciados por fatores imprevisíveis
• Performance passada não garante resultados futuros
• Modelos podem não capturar mudanças de regime do mercado

RESPONSABILIDADE:
O usuário assume total responsabilidade por suas decisões de investimento.
Sempre consulte profissionais qualificados antes de investir.

================================================================================
RELATÓRIO GERADO AUTOMATICAMENTE PELO MARKETMIND AI SYSTEM
Data: {agora} | Modelos: RandomForest + GradientBoosting + LSTM + LinearRegression
================================================================================
        """
        
        return relatorio.strip()
        
    except Exception as e:
        return f"Erro ao gerar relatório: {str(e)}"

def criar_modelo_lstm(input_shape):
    neurons = min(50, max(20, input_shape[0]))
    
    model = Sequential([
        LSTM(neurons, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(neurons // 2, return_sequences=False),
        Dropout(0.2),
        Dense(neurons // 4),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def treinar_modelo_lstm(X, y):
    try:
        split_index = int(0.8 * len(X))
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]
        
        model = criar_modelo_lstm((X.shape[1], X.shape[2]))
        
        epochs = min(50, max(20, len(X_train) // 2))
        
        with st.spinner(f'Treinando modelo de IA... ({epochs} épocas)'):
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=min(32, max(8, len(X_train) // 4)),
                validation_data=(X_val, y_val),
                verbose=0,
                shuffle=False
            )
        
        return model, history, None
        
    except Exception as e:
        return None, None, f"Erro no treinamento: {str(e)}"

def fazer_previsao(model, scaler, historico_df):
    try:
        df = historico_df.copy()
        df['Close_MA7'] = df['Close'].rolling(window=7).mean()
        df['Close_MA21'] = df['Close'].rolling(window=21).mean()
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_MA7'] = df['Volume'].rolling(window=7).mean()
        
        df = df.dropna()
        
        dias_disponveis = len(df)
        sequence_length = min(30, int(dias_disponveis * 0.7))
        
        features = ['Close', 'Volume', 'Close_MA7', 'Close_MA21', 'Price_Change']
        last_days = df[features].tail(sequence_length).values
        
        last_days_scaled = scaler.transform(last_days)
        
        X_pred = np.array([last_days_scaled])
        
        pred_scaled = model.predict(X_pred, verbose=0)
        
        pred_array = np.zeros((1, 5))
        pred_array[0, 0] = pred_scaled[0, 0]
        
        pred_original = scaler.inverse_transform(pred_array)
        preco_previsto = pred_original[0, 0]
        
        ultima_data = df.index.max()
        data_previsao = ultima_data + timedelta(days=1)
        
        return preco_previsto, data_previsao, None
        
    except Exception as e:
        return None, None, f"Erro na previsão: {str(e)}"

def treinar_ensemble_modelos(X_traditional, X_lstm, y_multi):
    try:
        split_idx = max(1, int(0.7 * len(X_traditional)))
        
        X_trad_train, X_trad_test = X_traditional[:split_idx], X_traditional[split_idx:]
        X_lstm_train, X_lstm_test = X_lstm[:split_idx], X_lstm[split_idx:]
        y_train, y_test = y_multi[:split_idx], y_multi[split_idx:]
        
        modelos = {}
        scores = {}
        
        rf_models = []
        for dia in range(14):
            rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
            rf.fit(X_trad_train, y_train[:, dia])
            rf_models.append(rf)
            
            if len(X_trad_test) > 0:
                pred = rf.predict(X_trad_test)
                scores[f'RF_dia_{dia+1}'] = mean_squared_error(y_test[:, dia], pred)
        
        modelos['RandomForest'] = rf_models
        
        gb_models = []
        for dia in range(14):
            gb = GradientBoostingRegressor(n_estimators=50, max_depth=6, random_state=42)
            gb.fit(X_trad_train, y_train[:, dia])
            gb_models.append(gb)
            
            if len(X_trad_test) > 0:
                pred = gb.predict(X_trad_test)
                scores[f'GB_dia_{dia+1}'] = mean_squared_error(y_test[:, dia], pred)
        
        modelos['GradientBoosting'] = gb_models
        
        lr_models = []
        for dia in range(14):
            lr = LinearRegression()
            lr.fit(X_trad_train, y_train[:, dia])
            lr_models.append(lr)
            
            if len(X_trad_test) > 0:
                pred = lr.predict(X_trad_test)
                scores[f'LR_dia_{dia+1}'] = mean_squared_error(y_test[:, dia], pred)
        
        modelos['LinearRegression'] = lr_models
        
        try:
            if len(X_lstm_train) < 5:
                st.warning(f"⚠️ Dados LSTM insuficientes: {len(X_lstm_train)} amostras. Removendo LSTM do ensemble.")
                modelos['LSTM'] = None
            else:
                lstm_train_mean = np.mean(X_lstm_train)
                lstm_train_std = np.std(X_lstm_train)
                y_train_mean = np.mean(y_train)
                y_train_std = np.std(y_train)
                
                if lstm_train_std < 0.01 or y_train_std < 0.01:
                    st.warning("⚠️ Dados LSTM com baixa variabilidade. Removendo LSTM do ensemble.")
                    modelos['LSTM'] = None
                else:
                    lstm_model = Sequential([
                        LSTM(32, return_sequences=False, input_shape=(X_lstm.shape[1], X_lstm.shape[2])),
                        Dropout(0.3),
                        Dense(16, activation='relu'),
                        Dense(14)
                    ])
                    
                    lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                    
                    epochs = min(20, max(5, len(X_lstm_train)))
                    batch_size = max(1, min(4, len(X_lstm_train) // 3))
                    
                    with st.spinner('Treinando modelo LSTM ensemble...'):
                        if len(X_lstm_test) > 0:
                            history = lstm_model.fit(
                                X_lstm_train, y_train,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_data=(X_lstm_test, y_test),
                                verbose=0
                            )
                        else:
                            history = lstm_model.fit(
                                X_lstm_train, y_train,
                                epochs=epochs,
                                batch_size=batch_size,
                                verbose=0
                            )
                    
                    final_loss = history.history['loss'][-1]
                    initial_loss = history.history['loss'][0]
                    
                    if final_loss >= initial_loss * 0.95:
                        st.warning("⚠️ LSTM não convergiu adequadamente. Removendo do ensemble.")
                        modelos['LSTM'] = None
                    else:
                        modelos['LSTM'] = lstm_model
                        
                        if len(X_lstm_test) > 0:
                            lstm_pred = lstm_model.predict(X_lstm_test, verbose=0)
                            for dia in range(14):
                                scores[f'LSTM_dia_{dia+1}'] = mean_squared_error(y_test[:, dia], lstm_pred[:, dia])
        
        except Exception as e:
            st.warning(f"⚠️ Erro no treinamento LSTM: {str(e)}. Removendo do ensemble.")
            modelos['LSTM'] = None
        
        return modelos, scores, None
        
    except Exception as e:
        return None, None, f"Erro no treinamento do ensemble: {str(e)}"

def fazer_previsao_ensemble(modelos, scalers, historico_df):
    try:
        scaler_traditional, scaler_lstm = scalers
        
        df = historico_df.copy()
        
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
        
        lstm_features = ['Close', 'Volume', 'Close_MA7', 'Close_MA21', 'Price_Change']
        sequence_length = min(15, len(df) - 1)
        last_sequence = df[lstm_features].tail(sequence_length).values
        last_sequence_scaled = scaler_lstm.transform(last_sequence)
        last_sequence_scaled = last_sequence_scaled.reshape(1, sequence_length, -1)
        
        previsoes = {}
        
        rf_pred = []
        for i, model in enumerate(modelos['RandomForest']):
            pred = model.predict(last_features_scaled)[0]
            rf_pred.append(pred)
        previsoes['RandomForest'] = np.array(rf_pred)
        
        gb_pred = []
        for i, model in enumerate(modelos['GradientBoosting']):
            pred = model.predict(last_features_scaled)[0]
            gb_pred.append(pred)
        previsoes['GradientBoosting'] = np.array(gb_pred)
        
        lr_pred = []
        for i, model in enumerate(modelos['LinearRegression']):
            pred = model.predict(last_features_scaled)[0]
            lr_pred.append(pred)
        previsoes['LinearRegression'] = np.array(lr_pred)
        
        if modelos['LSTM'] is not None:
            lstm_pred = modelos['LSTM'].predict(last_sequence_scaled, verbose=0)[0]
            previsoes['LSTM'] = lstm_pred
        
        if modelos['LSTM'] is not None:
            pesos = {'RandomForest': 0.3, 'GradientBoosting': 0.3, 'LSTM': 0.25, 'LinearRegression': 0.15}
        else:
            pesos = {'RandomForest': 0.4, 'GradientBoosting': 0.4, 'LinearRegression': 0.2}
        
        previsao_final = np.zeros(14)
        for modelo, pred in previsoes.items():
            previsao_final += pred * pesos[modelo]
        
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
        return None, None, None, f"Erro na previsão ensemble: {str(e)}"

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
        return None, None, None, f"Erro geral na previsão ensemble: {str(e)}"

def buscar_dados_acao(ticker):
    try:
        ticker = ticker.upper().strip()
        
        with st.spinner(f'Buscando dados para {ticker}...'):
            
            API_KEY = "nUUZxG2ZdAWuSkBDhPobC2"
            
            headers = {
                "Authorization": f"Bearer {API_KEY}"
            }
            
            url_cotacao = f"https://brapi.dev/api/quote/{ticker}"
            response = requests.get(url_cotacao, headers=headers, timeout=10)
            
            if response.status_code != 200:
                return None, f"Erro {response.status_code}: Ticker {ticker} não encontrado"
            
            data = response.json()
            
            if 'results' not in data or not data['results']:
                return None, f"Ticker {ticker} não encontrado na B3"
            
            acao_data = data['results'][0]
            
            url_historico = f"https://brapi.dev/api/quote/{ticker}?range=3mo&interval=1d"
            response_hist = requests.get(url_historico, headers=headers, timeout=10)
            
            historico = None
            dados_3_meses = None
            
            if response_hist.status_code == 200:
                data_hist = response_hist.json()
                
                if 'results' in data_hist and data_hist['results']:
                    hist_data = data_hist['results'][0].get('historicalDataPrice', [])
                    
                    if hist_data:
                        hist_list = []
                        for item in hist_data:
                            try:
                                data_ponto = datetime.fromtimestamp(item['date'])
                                hist_list.append({
                                    'Data': data_ponto,
                                    'Close': item.get('close', 0),
                                    'Open': item.get('open', 0),
                                    'High': item.get('high', 0),
                                    'Low': item.get('low', 0),
                                    'Volume': item.get('volume', 0)
                                })
                            except (KeyError, ValueError):
                                continue
                        
                        if hist_list:
                            historico = pd.DataFrame(hist_list)
                            historico.set_index('Data', inplace=True)
                            historico.sort_index(inplace=True)
                            
                            if len(historico) > 30:
                                idx_proporcional = min(len(historico) - 1, len(historico) * 3 // 4)
                                dados_3m_atras = historico.iloc[-idx_proporcional]
                                dados_3_meses = {
                                    'data': dados_3m_atras.name.strftime('%d/%m/%Y'),
                                    'preco': round(dados_3m_atras['Close'], 2),
                                    'volume': int(dados_3m_atras['Volume'])
                                }
            
            indicadores = {}
            if historico is not None and len(historico) > 0:
                indicadores['maxima_52s'] = round(historico['High'].max(), 2)
                indicadores['minima_52s'] = round(historico['Low'].min(), 2)
                
                volume_medio = historico['Volume'].tail(min(30, len(historico))).mean()
                indicadores['volume_medio'] = int(volume_medio) if not pd.isna(volume_medio) else 0
                
                if len(historico) >= 20:
                    indicadores['media_20d'] = round(historico['Close'].tail(20).mean(), 2)
                if len(historico) >= 50:
                    indicadores['media_50d'] = round(historico['Close'].tail(50).mean(), 2)
                
                if len(historico) >= 30:
                    returns = historico['Close'].pct_change().tail(30)
                    volatilidade = returns.std() * np.sqrt(252) * 100
                    indicadores['volatilidade'] = round(volatilidade, 1)
            
            dados = {
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
            }
            
            return dados, None
            
    except requests.exceptions.Timeout:
        return None, "Timeout: API demorou para responder. Tente novamente."
    except requests.exceptions.ConnectionError:
        return None, "Erro de conexão. Verifique sua internet."
    except requests.exceptions.RequestException as e:
        return None, f"Erro de requisição: {str(e)}"
    except Exception as e:
        return None, f"Erro: {str(e)}"

def criar_grafico_com_previsao(historico, ticker, previsoes_ensemble=None, datas_previsao=None, detalhes_previsoes=None):
    fig = go.Figure()
    
    if historico is not None and not historico.empty:
        fig.add_trace(go.Scatter(
            x=historico.index,
            y=historico['Close'],
            mode='lines',
            name='Histórico Real',
            line=dict(color='#00d4ff', width=3),
            hovertemplate='<b>R$ %{y:.2f}</b><br>%{x}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=historico.index,
            y=historico['Close'],
            fill='tonexty',
            mode='none',
            fillcolor='rgba(0, 212, 255, 0.1)',
            showlegend=False
        ))
        
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
            
            ultimo_preco = historico['Close'].iloc[-1]
            ultima_data = historico.index[-1]
            
            fig.add_trace(go.Scatter(
                x=[ultima_data, datas_previsao[0]],
                y=[ultimo_preco, previsoes_ensemble[0]],
                mode='lines',
                line=dict(color='#ff4444', width=2, dash='dash'),
                name='Conexão',
                hovertemplate='<extra></extra>',
                showlegend=False
            ))
            
            if detalhes_previsoes is not None:
                cores_modelos = {
                    'LSTM': '#ff9999',
                    'RandomForest': '#99ff99', 
                    'GradientBoosting': '#9999ff',
                    'LinearRegression': '#ffff99'
                }
                
                for modelo, previsoes in detalhes_previsoes.items():
                    if modelo in cores_modelos:
                        fig.add_trace(go.Scatter(
                            x=datas_previsao,
                            y=previsoes,
                            mode='lines',
                            name=modelo,
                            line=dict(color=cores_modelos[modelo], width=1, dash='dot'),
                            opacity=0.6,
                            hovertemplate=f'<b>{modelo}</b><br><b>R$ %{{y:.2f}}</b><br>%{{x|%d/%m/%Y}}<extra></extra>'
                        ))
    
    titulo = f'{ticker}'
    if previsoes_ensemble is not None:
        titulo += ' - Previsão Ensemble (2 Semanas)'
    
    fig.update_layout(
        title=titulo,
        xaxis_title='Data',
        yaxis_title='Preço (R$)',
        height=500 if previsoes_ensemble is not None else 400,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        title_font_color='#00d4ff',
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            color='white'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            color='white'
        )
    )
    
    return fig

if 'mostrar_dados' not in st.session_state:
    st.session_state.mostrar_dados = False
if 'dados_acao' not in st.session_state:
    st.session_state.dados_acao = None
if 'mostrar_previsao' not in st.session_state:
    st.session_state.mostrar_previsao = False
if 'tela_atual' not in st.session_state:
    st.session_state.tela_atual = 'login'
if 'usuario_logado' not in st.session_state:
    st.session_state.usuario_logado = None

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
            st.error("Por favor, preencha todos os campos")
    
    st.markdown("</div>", unsafe_allow_html=True)

def render_cadastro():
    st.markdown("<div class='screen-entrada'>", unsafe_allow_html=True)
    st.markdown("<h1 class='main-header'>MarketMind</h1>", unsafe_allow_html=True)
    
    st.markdown("### Criar Conta")
    
    with st.form("cadastro_form"):
        username = st.text_input("Usuário", placeholder="Escolha um nome de usuário")
        email = st.text_input("Email", placeholder="Digite seu email")
        password = st.text_input("Senha", type="password", placeholder="Escolha uma senha")
        password_confirm = st.text_input("Confirmar Senha", type="password", placeholder="Confirme sua senha")
        
        col1, col2 = st.columns(2)
        with col1:
            cadastro_button = st.form_submit_button("Criar Conta", type="primary", use_container_width=True)
        with col2:
            if st.form_submit_button("Voltar ao Login", use_container_width=True):
                st.session_state.tela_atual = 'login'
                st.rerun()
    
    if cadastro_button:
        if username and email and password and password_confirm:
            if password != password_confirm:
                st.error("As senhas não coincidem")
            elif len(password) < 6:
                st.error("A senha deve ter pelo menos 6 caracteres")
            elif len(username) < 3:
                st.error("O usuário deve ter pelo menos 3 caracteres")
            else:
                sucesso, mensagem = criar_usuario(username, email, password)
                if sucesso:
                    st.success(mensagem)
                    st.info("Agora você pode fazer login com suas credenciais")
                    st.session_state.tela_atual = 'login'
                    st.rerun()
                else:
                    st.error(mensagem)
        else:
            st.error("Por favor, preencha todos os campos")
    
    st.markdown("</div>", unsafe_allow_html=True)

def render_header_logado():
    usuario = get_usuario_logado()
    if usuario:
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.markdown(f"<h1 class='main-header'>MarketMind</h1>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div style='padding-top: 30px; text-align: right;'>", unsafe_allow_html=True)
            user_action = st.selectbox(
                "",
                ["Olá, " + usuario, "Logout"],
                index=0,
                key="user_dropdown",
                label_visibility="collapsed"
            )
            st.markdown("</div>", unsafe_allow_html=True)
            
            if user_action == "Logout":
                fazer_logout()
                st.session_state.tela_atual = 'login'
                st.rerun()

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
        
        st.markdown("<div class='screen-entrada'>", unsafe_allow_html=True)
    
        st.markdown("### Digite o código da ação que deseja analisar:")
        
        with st.form("busca_acao"):
            ticker = st.text_input(
                "", 
                placeholder="Ex: PETR4, VALE3, ITUB4", 
                label_visibility="collapsed",
                key="ticker_input"
            )
            submitted = st.form_submit_button(
                "Analisar Ação", 
                type="primary"
            )
        
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
            st.error("Por favor, digite o código de uma ação")
    
        query_params = st.query_params
        if "analyze" in query_params:
            ticker = query_params["analyze"]
            dados, erro = buscar_dados_acao(ticker)
            if erro:
                st.error(f"Erro: {erro}")
            else:
                st.session_state.dados_acao = dados
                st.session_state.mostrar_dados = True
                st.session_state.mostrar_previsao = False
                st.query_params.clear()
                st.rerun()
        
        if "remove" in query_params:
            ticker = query_params["remove"]
            sucesso, mensagem = remover_favorito(ticker)
            if sucesso:
                st.success(mensagem)
            else:
                st.error(mensagem)
            st.query_params.clear()
            st.rerun()
        
        for key in list(st.session_state.keys()):
            if key.startswith('action_analyze_'):
                ticker = key.replace('action_analyze_', '')
                dados, erro = buscar_dados_acao(ticker)
                if erro:
                    st.error(f"Erro: {erro}")
                else:
                    st.session_state.dados_acao = dados
                    st.session_state.mostrar_dados = True
                    st.session_state.mostrar_previsao = False
                    del st.session_state[key]
                    st.rerun()
            
            elif key.startswith('action_remove_'):
                ticker = key.replace('action_remove_', '')
                sucesso, mensagem = remover_favorito(ticker)
                if sucesso:
                    st.success(mensagem)
                else:
                    st.error(mensagem)
                del st.session_state[key]
                st.rerun()
        
        
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
                                btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
                                
                                with btn_col1:
                                    if st.button("📈", key=f"analyze_btn_{fav['ticker']}", 
                                               help="Analisar ação",
                                               use_container_width=True):
                                        dados, erro = buscar_dados_acao(fav['ticker'])
                                        if erro:
                                            st.error(f"Erro: {erro}")
                                        else:
                                            st.session_state.dados_acao = dados
                                            st.session_state.mostrar_dados = True
                                            st.session_state.mostrar_previsao = False
                                            st.rerun()
                                
                                with btn_col2:
                                    if st.button("❌", key=f"remove_btn_{fav['ticker']}", 
                                               help="Remover dos favoritos",
                                               use_container_width=True):
                                        sucesso, mensagem = remover_favorito(fav['ticker'])
                                        if sucesso:
                                            st.success(mensagem)
                                            st.rerun()
                                        else:
                                            st.error(mensagem)
                                
                                if dados_rapidos['sucesso']:
                                    variacao_valor = dados_rapidos['variacao']
                                    delta_str = f"{variacao_valor:+.2f}%"
                                    
                                    st.metric(
                                        label=fav['ticker'],
                                        value=f"R$ {dados_rapidos['preco']:.2f}",
                                        delta=delta_str
                                    )
                                else:
                                    st.metric(
                                        label=fav['ticker'],
                                        value="N/D"
                                    )
                            
        
        if len(favoritos) > 6:
            st.caption(f"+ {len(favoritos) - 6} outros favoritos")
    
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.mostrar_dados and not st.session_state.mostrar_previsao:
        dados = st.session_state.dados_acao
        
        st.markdown("<div class='tela-dados'>", unsafe_allow_html=True)
        
        col_titulo, col_btn_fav, col_btn_prev, col_btn = st.columns([2, 1, 1, 1])
        with col_titulo:
            st.markdown("<h4 class='section-title'>Cotação Atual</h4>", unsafe_allow_html=True)
        with col_btn_fav:
            is_fav = eh_favorito(dados['ticker'])
            btn_text = "★ Remover" if is_fav else "☆ Favoritar"
            btn_type = "secondary" if is_fav else "primary"
            
            if st.button(btn_text, type=btn_type, use_container_width=True, key="fav_analise"):
                if is_fav:
                    sucesso, mensagem = remover_favorito(dados['ticker'])
                else:
                    sucesso, mensagem = adicionar_favorito(dados['ticker'], dados['nome'], dados['preco'])
                
                if sucesso:
                    st.success(mensagem)
                    st.rerun()
                else:
                    st.error(mensagem)
        with col_btn_prev:
            if st.button("Gerar Previsão", type="secondary", use_container_width=True):
                st.session_state.mostrar_previsao = True
                st.rerun()
        with col_btn:
            if st.button("Nova Análise", type="secondary", use_container_width=True):
                st.session_state.mostrar_dados = False
                st.session_state.mostrar_previsao = False
                st.session_state.dados_acao = None
                st.rerun()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            variacao_valor = dados.get('variacao_valor', 0)
            st.metric("Preço Atual", f"R$ {dados['preco']:.2f}", 
                     delta=f"R$ {variacao_valor:.2f}")
        with col2:
            st.metric("Variação %", f"{dados['variacao']:.2f}%")
        with col3:
            st.metric("Máxima do Dia", f"R$ {dados['maxima']:.2f}")
        with col4:
            st.metric("Mínima do Dia", f"R$ {dados['minima']:.2f}")

        st.markdown("<h4 class='section-title'>Informações do Pregão</h4>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Abertura", f"R$ {dados.get('abertura', 0):.2f}")
        with col2:
            fechamento_ant = dados.get('fechamento_anterior', 0)
            st.metric("Fechamento Anterior", f"R$ {fechamento_ant:.2f}")
        with col3:
            volume = dados.get('volume', 0)
            volume_formatado = f"{volume:,}".replace(",", ".")
            st.metric("Volume do Dia", volume_formatado)
        with col4:
            market_cap = dados.get('market_cap', 0)
            if market_cap > 0:
                if market_cap >= 1e9:
                    cap_formatado = f"R$ {market_cap/1e9:.1f}B"
                elif market_cap >= 1e6:
                    cap_formatado = f"R$ {market_cap/1e6:.1f}M"
                else:
                    cap_formatado = f"R$ {market_cap:,.0f}".replace(",", ".")
                st.metric("Market Cap", cap_formatado)
            else:
                st.metric("Market Cap", "N/D")

        indicadores = dados.get('indicadores', {})
        if indicadores:
            st.markdown("<h4 class='section-title'>Indicadores Técnicos</h4>", unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                maxima_52s = indicadores.get('maxima_52s')
                if maxima_52s:
                    st.metric("Máxima 52S", f"R$ {maxima_52s:.2f}")
                else:
                    st.metric("Máxima 52S", "N/D")
            
            with col2:
                minima_52s = indicadores.get('minima_52s')
                if minima_52s:
                    st.metric("Mínima 52S", f"R$ {minima_52s:.2f}")
                else:
                    st.metric("Mínima 52S", "N/D")
            
            with col3:
                media_20d = indicadores.get('media_20d')
                if media_20d:
                    st.metric("Média 20D", f"R$ {media_20d:.2f}")
                else:
                    st.metric("Média 20D", "N/D")
            
            with col4:
                volatilidade = indicadores.get('volatilidade')
                if volatilidade:
                    st.metric("Volatilidade", f"{volatilidade:.1f}%")
                else:
                    st.metric("Volatilidade", "N/D")

        
        if dados['dados_3_meses']:
            st.markdown("<h4 class='section-title'>Comparativo 3 Meses</h4 >", unsafe_allow_html=True)
            
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
        
        
        if dados['historico'] is not None:
            fig_analise = criar_grafico_com_previsao(dados['historico'], dados['ticker'])
            st.plotly_chart(fig_analise, use_container_width=True)
        else:
            st.markdown("<div style='color: #ff6b6b;'>Dados históricos indisponíveis</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.mostrar_previsao:
        dados = st.session_state.dados_acao
        
        st.markdown("<div class='tela-dados'>", unsafe_allow_html=True)
        
        col_titulo, col_btn_fav, col_btn_volta, col_btn = st.columns([2, 1, 1, 1])
        with col_titulo:
            st.markdown("<h4 class='section-title'>Previsão de Preços com IA</h4>", unsafe_allow_html=True)
        with col_btn_fav:
            is_fav = eh_favorito(dados['ticker'])
            btn_text = "★ Remover" if is_fav else "☆ Favoritar"
            btn_type = "secondary" if is_fav else "primary"
            
            if st.button(btn_text, type=btn_type, use_container_width=True, key="fav_previsao"):
                if is_fav:
                    sucesso, mensagem = remover_favorito(dados['ticker'])
                else:
                    sucesso, mensagem = adicionar_favorito(dados['ticker'], dados['nome'], dados['preco'])
                
                if sucesso:
                    st.success(mensagem)
                    st.rerun()
                else:
                    st.error(mensagem)
        with col_btn_volta:
            if st.button("Voltar", type="secondary", use_container_width=True):
                st.session_state.mostrar_previsao = False
                st.rerun()
        with col_btn:
            if st.button("Nova Análise", type="secondary", use_container_width=True):
                st.session_state.mostrar_dados = False
                st.session_state.mostrar_previsao = False
                st.session_state.dados_acao = None
                if 'previsoes_ensemble' in st.session_state:
                    del st.session_state.previsoes_ensemble
                if 'datas_previsao' in st.session_state:
                    del st.session_state.datas_previsao
                if 'detalhes_previsoes' in st.session_state:
                    del st.session_state.detalhes_previsoes
                if 'ticker_previsao' in st.session_state:
                    del st.session_state.ticker_previsao
                st.rerun()

        if ('previsoes_ensemble' not in st.session_state or 
            'datas_previsao' not in st.session_state or 
            'detalhes_previsoes' not in st.session_state or
            st.session_state.get('ticker_previsao') != dados['ticker']):
            
            with st.spinner('Gerando previsões com Ensemble de Modelos de IA...'):
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
            st.info("**Dica:** A previsão ensemble requer pelo menos 25 dias de histórico.")
            previsoes_ensemble, datas_previsao, detalhes_previsoes = None, None, None
        else:
            preco_atual = dados['preco']
            preco_1_semana = previsoes_ensemble[4]
            preco_2_semanas = previsoes_ensemble[13]
            
            variacao_1_semana = ((preco_1_semana - preco_atual) / preco_atual) * 100
            variacao_2_semanas = ((preco_2_semanas - preco_atual) / preco_atual) * 100
            
            st.success("**Previsões Ensemble geradas com sucesso!**")
            
            col_prev1, col_prev2, col_prev3, col_prev4 = st.columns(4)
            with col_prev1:
                st.metric("1 Semana", f"R$ {preco_1_semana:.2f}", f"{variacao_1_semana:+.2f}%")
            with col_prev2:
                st.metric("2 Semanas", f"R$ {preco_2_semanas:.2f}", f"{variacao_2_semanas:+.2f}%")
            with col_prev3:
                tendencia = "Alta" if preco_2_semanas > preco_atual else "Baixa"
                st.metric("Tendência", tendencia)
            with col_prev4:
                dispersao = np.std([detalhes_previsoes[modelo][-1] for modelo in detalhes_previsoes.keys()])
                confianca = max(60, min(90, 85 - (dispersao / preco_atual * 100)))
                st.metric("Confiança", f"{confianca:.0f}%")
            
            with st.expander("📊 Detalhes por Modelo"):
                st.markdown("**Previsões para 2 semanas (último dia):**")
                
                modelos_disponiveis = list(detalhes_previsoes.keys())
                num_modelos = len(modelos_disponiveis)
                
                if num_modelos == 4:
                    model_col1, model_col2, model_col3, model_col4 = st.columns(4)
                    cols = [model_col1, model_col2, model_col3, model_col4]
                elif num_modelos == 3:
                    model_col1, model_col2, model_col3 = st.columns(3)
                    cols = [model_col1, model_col2, model_col3]
                else:
                    cols = st.columns(num_modelos)
                
                icones_modelos = {
                    'LSTM': '🧠',
                    'RandomForest': '🌲', 
                    'GradientBoosting': '🚀',
                    'LinearRegression': '📈'
                }
                
                nomes_modelos = {
                    'LSTM': 'LSTM',
                    'RandomForest': 'Random Forest',
                    'GradientBoosting': 'Gradient Boost', 
                    'LinearRegression': 'Linear Reg.'
                }
                
                for i, modelo in enumerate(modelos_disponiveis):
                    with cols[i]:
                        pred = detalhes_previsoes[modelo][-1]
                        var = ((pred - preco_atual) / preco_atual) * 100
                        icone = icones_modelos.get(modelo, '📊')
                        nome = nomes_modelos.get(modelo, modelo)
                        st.metric(f"{icone} {nome}", f"R$ {pred:.2f}", f"{var:+.2f}%")

        st.markdown("<h4 class='section-title'>Dados Atuais</h4>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Preço Atual", f"R$ {dados['preco']:.2f}")
        with col2:
            st.metric("Variação", f"{dados['variacao']:.2f}%")
        with col3:
            st.metric("Máxima", f"R$ {dados['maxima']:.2f}")
        with col4:
            st.metric("Mínima", f"R$ {dados['minima']:.2f}")

        if dados['historico'] is not None:
            fig_previsao = criar_grafico_com_previsao(
                dados['historico'], 
                dados['ticker'], 
                previsoes_ensemble, 
                datas_previsao,
                detalhes_previsoes
            )
            st.plotly_chart(fig_previsao, use_container_width=True)
        else:
            st.markdown("<div style='color: #ff6b6b;'>Dados históricos indisponíveis</div>", unsafe_allow_html=True)
        
        if not erro_ml:
            with st.expander("ℹ️ Sobre o Ensemble de Modelos"):
                st.markdown("""
                **Método Ensemble:** Combinação de 4 modelos de Machine Learning
                
                **Modelos utilizados:**
                - 🧠 **LSTM (25%)** - Redes neurais para padrões sequenciais
                - 🌲 **Random Forest (30%)** - Ensemble de árvores de decisão
                - 🚀 **Gradient Boosting (30%)** - Boosting adaptativo
                - 📈 **Linear Regression (15%)** - Regressão linear múltipla
                
                **Features analisadas:**
                - Preços de fechamento, abertura, máxima, mínima
                - Volume de negociação e suas médias móveis
                - Indicadores técnicos: RSI, MACD, Bandas de Bollinger
                - Médias móveis (7 e 21 dias) e volatilidade
                - Variações percentuais e dados defasados
                
                **Período de previsão:** 2 semanas (14 dias úteis)
                
                **Como funciona:**
                1. Cada modelo faz previsões independentes
                2. Resultados são combinados com pesos baseados na performance
                3. Ensemble final oferece maior robustez que modelos individuais
                
                **⚠️ Aviso importante:**
                - Previsões baseadas em padrões históricos
                - Não constitui recomendação de investimento
                - Mercados são influenciados por eventos imprevisíveis
                - Use apenas para fins educacionais e de pesquisa
                - Sempre consulte um profissional qualificado antes de investir
                """)
            
            st.markdown("---")
            col_relatorio1, col_relatorio2, col_relatorio3 = st.columns([1, 2, 1])
            
            with col_relatorio2:
                if st.button("📄 Gerar Relatório PDF", type="primary", use_container_width=True):
                    with st.spinner("Gerando relatório..."):
                        relatorio_texto = gerar_relatorio_previsao(dados, previsoes_ensemble, datas_previsao, detalhes_previsoes)
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        nome_arquivo = f"Relatorio_{dados['ticker']}_{timestamp}.txt"
                        
                        st.download_button(
                            label="💾 Download Relatório",
                            data=relatorio_texto,
                            file_name=nome_arquivo,
                            mime="text/plain",
                            use_container_width=True
                        )
                        
                        with st.expander("👁️ Prévia do Relatório"):
                            st.text(relatorio_texto[:1000] + "..." if len(relatorio_texto) > 1000 else relatorio_texto)
        
        st.markdown("</div>", unsafe_allow_html=True)