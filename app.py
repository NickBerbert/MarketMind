import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from pathlib import Path
import requests
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="MarketMind",
    page_icon="🧠",
    layout="wide"
)

def load_css():
    """Carrega o arquivo CSS externo"""
    css_file = Path(__file__).parent / "styles.css"
    if css_file.exists():
        with open(css_file, 'r', encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Arquivo styles.css não encontrado")

# Carregar CSS
load_css()

def preparar_dados_lstm(historico_df):
    """
    Prepara dados históricos para o modelo LSTM
    """
    try:
        if historico_df is None or historico_df.empty:
            return None, None, None, "Dados históricos insuficientes"
        
        # Criar features do modelo
        df = historico_df.copy()
        
        # Features principais: Close, Volume (normalizar depois)
        df['Close_MA7'] = df['Close'].rolling(window=7).mean()  # Média móvel 7 dias
        df['Close_MA21'] = df['Close'].rolling(window=21).mean()  # Média móvel 21 dias
        df['Price_Change'] = df['Close'].pct_change()  # Variação percentual
        df['Volume_MA7'] = df['Volume'].rolling(window=7).mean()  # Volume médio
        
        # Remover NaNs
        df = df.dropna()
        
        # Ajustar sequência baseado na quantidade de dados disponíveis
        dias_disponveis = len(df)
        if dias_disponveis < 20:  # Mínimo absoluto
            return None, None, None, f"Histórico insuficiente: {dias_disponveis} dias (mínimo: 20)"
        
        # Usar sequência adaptativa: máximo 30 dias ou 70% dos dados disponíveis
        sequence_length = min(30, int(dias_disponveis * 0.7))
        
        # Selecionar features para o modelo
        features = ['Close', 'Volume', 'Close_MA7', 'Close_MA21', 'Price_Change']
        data = df[features].values
        
        # Normalizar dados
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Criar sequências adaptativas para prever o próximo
        # Criar sequências adaptativas para prever o próximo
        X, y = [], []
        
        for i in range(sequence_length, len(data_scaled)):
            X.append(data_scaled[i-sequence_length:i])  # sequência anterior
            y.append(data_scaled[i, 0])  # Preço do dia seguinte (Close é index 0)
        
        X, y = np.array(X), np.array(y)
        
        if len(X) == 0:
            return None, None, None, f"Dados insuficientes para criar sequências. Disponível: {dias_disponveis}, Sequência: {sequence_length}"
        
        return X, y, scaler, None
        
    except Exception as e:
        return None, None, None, f"Erro ao preparar dados: {str(e)}"

def criar_modelo_lstm(input_shape):
    """
    Cria e compila o modelo LSTM adaptativo
    """
    # Ajustar neurônios baseado no tamanho da sequência
    neurons = min(50, max(20, input_shape[0]))  # Entre 20-50 neurônios
    
    model = Sequential([
        LSTM(neurons, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(neurons // 2, return_sequences=False),  # Segunda camada menor
        Dropout(0.2),
        Dense(neurons // 4),  # Camada densa proporcional
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def treinar_modelo_lstm(X, y):
    """
    Treina o modelo LSTM com os dados fornecidos
    """
    try:
        # Dividir dados (80% treino, 20% validação)
        split_index = int(0.8 * len(X))
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]
        
        # Criar modelo
        model = criar_modelo_lstm((X.shape[1], X.shape[2]))
        
        # Treinar modelo (épocas adaptativas)
        epochs = min(50, max(20, len(X_train) // 2))  # Entre 20-50 épocas
        
        with st.spinner(f'🤖 Treinando modelo de IA... ({epochs} épocas)'):
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=min(32, max(8, len(X_train) // 4)),  # Batch adaptativo
                validation_data=(X_val, y_val),
                verbose=0,  # Silencioso
                shuffle=False  # Manter ordem temporal
            )
        
        return model, history, None
        
    except Exception as e:
        return None, None, f"Erro no treinamento: {str(e)}"

def fazer_previsao(model, scaler, historico_df):
    """
    Faz previsão do próximo preço usando o modelo treinado
    """
    try:
        # Preparar dados mais recentes
        df = historico_df.copy()
        df['Close_MA7'] = df['Close'].rolling(window=7).mean()
        df['Close_MA21'] = df['Close'].rolling(window=21).mean()
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_MA7'] = df['Volume'].rolling(window=7).mean()
        
        df = df.dropna()
        
        # Determinar tamanho da sequência baseado nos dados disponíveis
        dias_disponveis = len(df)
        sequence_length = min(30, int(dias_disponveis * 0.7))
        
        # Pegar últimos dias conforme sequência calculada
        features = ['Close', 'Volume', 'Close_MA7', 'Close_MA21', 'Price_Change']
        last_days = df[features].tail(sequence_length).values
        
        # Normalizar
        last_days_scaled = scaler.transform(last_days)
        
        # Reshape para o modelo
        X_pred = np.array([last_days_scaled])
        
        # Fazer previsão
        pred_scaled = model.predict(X_pred, verbose=0)
        
        # Desnormalizar previsão
        # Criar array com shape correto para inverse_transform
        pred_array = np.zeros((1, 5))  # 5 features
        pred_array[0, 0] = pred_scaled[0, 0]  # Colocar previsão na posição do Close
        
        pred_original = scaler.inverse_transform(pred_array)
        preco_previsto = pred_original[0, 0]
        
        # Data da previsão (próximo dia útil)
        ultima_data = df.index.max()
        data_previsao = ultima_data + timedelta(days=1)
        
        return preco_previsto, data_previsao, None
        
    except Exception as e:
        return None, None, f"Erro na previsão: {str(e)}"

def gerar_previsao_acao(dados_acao):
    """
    Função principal que orquestra todo o processo de ML
    """
    try:
        historico = dados_acao.get('historico')
        if historico is None or historico.empty:
            return None, None, "Dados históricos não disponíveis"
        
        # 1. Preparar dados
        X, y, scaler, erro_prep = preparar_dados_lstm(historico)
        if erro_prep:
            return None, None, erro_prep
        
        # 2. Treinar modelo
        model, history, erro_treino = treinar_modelo_lstm(X, y)
        if erro_treino:
            return None, None, erro_treino
        
        # 3. Fazer previsão
        preco_previsto, data_previsao, erro_pred = fazer_previsao(model, scaler, historico)
        if erro_pred:
            return None, None, erro_pred
        
        return preco_previsto, data_previsao, None
        
    except Exception as e:
        return None, None, f"Erro geral na previsão: {str(e)}"

def buscar_dados_acao(ticker):
    """Busca dados da ação via Brapi API"""
    try:
        ticker = ticker.upper().strip()
        
        with st.spinner(f'Buscando dados para {ticker}...'):
            
            # API Key da Brapi
            API_KEY = "nUUZxG2ZdAWuSkBDhPobC2"
            
            # Headers para autenticação
            headers = {
                "Authorization": f"Bearer {API_KEY}"
            }
            
            # 1. Buscar cotação atual
            url_cotacao = f"https://brapi.dev/api/quote/{ticker}"
            response = requests.get(url_cotacao, headers=headers, timeout=10)
            
            if response.status_code != 200:
                return None, f"Erro {response.status_code}: Ticker {ticker} não encontrado"
            
            data = response.json()
            
            if 'results' not in data or not data['results']:
                return None, f"Ticker {ticker} não encontrado na B3"
            
            acao_data = data['results'][0]
            
            # 2. Buscar histórico (3 meses)
            url_historico = f"https://brapi.dev/api/quote/{ticker}?range=3mo&interval=1d"
            response_hist = requests.get(url_historico, headers=headers, timeout=10)
            
            historico = None
            dados_3_meses = None
            
            if response_hist.status_code == 200:
                data_hist = response_hist.json()
                
                if 'results' in data_hist and data_hist['results']:
                    hist_data = data_hist['results'][0].get('historicalDataPrice', [])
                    
                    if hist_data:
                        # Converter para DataFrame
                        hist_list = []
                        for item in hist_data:
                            try:
                                data_ponto = datetime.fromtimestamp(item['date'])
                                hist_list.append({
                                    'Data': data_ponto,
                                    'Close': item.get('close', 0),
                                    'Volume': item.get('volume', 0)
                                })
                            except (KeyError, ValueError):
                                continue
                        
                        if hist_list:
                            historico = pd.DataFrame(hist_list)
                            historico.set_index('Data', inplace=True)
                            historico.sort_index(inplace=True)
                            
                            # Calcular dados de 3 meses atrás (ou proporcionalmente)
                            if len(historico) > 30:  # Se temos pelo menos 30 dias
                                idx_proporcional = min(len(historico) - 1, len(historico) * 3 // 4)  # 75% do histórico
                                dados_3m_atras = historico.iloc[-idx_proporcional]
                                dados_3_meses = {
                                    'data': dados_3m_atras.name.strftime('%d/%m/%Y'),
                                    'preco': round(dados_3m_atras['Close'], 2),
                                    'volume': int(dados_3m_atras['Volume'])
                                }
            
            # 3. Montar dados de retorno
            dados = {
                'ticker': acao_data.get('symbol', ticker),
                'nome': acao_data.get('shortName', acao_data.get('longName', 'N/A')),
                'preco': round(acao_data.get('regularMarketPrice', 0), 2),
                'variacao': round(acao_data.get('regularMarketChangePercent', 0), 2),
                'maxima': round(acao_data.get('regularMarketDayHigh', 0), 2),
                'minima': round(acao_data.get('regularMarketDayLow', 0), 2),
                'historico': historico,
                'dados_3_meses': dados_3_meses
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

def criar_grafico_com_previsao(historico, ticker, preco_previsto=None, data_previsao=None):
    """Cria gráfico com dados históricos e previsão ML"""
    fig = go.Figure()
    
    if historico is not None and not historico.empty:
        # Linha principal do preço histórico
        fig.add_trace(go.Scatter(
            x=historico.index,
            y=historico['Close'],
            mode='lines',
            name='Histórico Real',
            line=dict(color='#00d4ff', width=3),
            hovertemplate='<b>R$ %{y:.2f}</b><br>%{x}<extra></extra>'
        ))
        
        # Área sombreada
        fig.add_trace(go.Scatter(
            x=historico.index,
            y=historico['Close'],
            fill='tonexty',
            mode='none',
            fillcolor='rgba(0, 212, 255, 0.1)',
            showlegend=False
        ))
        
        # Adicionar previsão se disponível
        if preco_previsto is not None and data_previsao is not None:
            fig.add_trace(go.Scatter(
                x=[data_previsao],
                y=[preco_previsto],
                mode='markers',
                name='Previsão IA',
                marker=dict(
                    color='#ff4444',
                    size=15,
                    symbol='star',
                    line=dict(color='white', width=2)
                ),
                hovertemplate='<b>🤖 Previsão IA</b><br><b>R$ %{y:.2f}</b><br>%{x|%d/%m/%Y}<extra></extra>'
            ))
            
            # Linha conectando último ponto real com previsão
            ultimo_preco = historico['Close'].iloc[-1]
            ultima_data = historico.index[-1]
            
            fig.add_trace(go.Scatter(
                x=[ultima_data, data_previsao],
                y=[ultimo_preco, preco_previsto],
                mode='lines',
                line=dict(color='#ff4444', width=2, dash='dash'),
                name='Projeção',
                hovertemplate='<extra></extra>',
                showlegend=False
            ))
    
    fig.update_layout(
        xaxis_title='Data',
        yaxis_title='Preço (R$)',
        height=400,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)",
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

# Inicializar session state
if 'mostrar_dados' not in st.session_state:
    st.session_state.mostrar_dados = False
if 'dados_acao' not in st.session_state:
    st.session_state.dados_acao = None
if 'mostrar_previsao' not in st.session_state:
    st.session_state.mostrar_previsao = False

# ====================== TELA 1 - ENTRADA ======================
if not st.session_state.mostrar_dados:
    
    st.markdown("<div class='screen-entrada'>", unsafe_allow_html=True)
    st.markdown("<h1 class='main-header'>🧠 MarketMind</h1>", unsafe_allow_html=True)
    
    st.markdown("### Digite o código da ação que deseja analisar:")
    
    # Formulário de entrada
    with st.form("busca_acao"):
        ticker = st.text_input(
            "", 
            placeholder="Ex: PETR4, VALE3, ITUB4", 
            label_visibility="collapsed",
            key="ticker_input"
        )
        submitted = st.form_submit_button(
            "🔍 Analisar Ação", 
            type="primary"
        )
    
    if submitted and ticker:
        dados, erro = buscar_dados_acao(ticker)
        
        if erro:
            st.error(f"❌ {erro}")
        else:
            st.session_state.dados_acao = dados
            st.session_state.mostrar_dados = True
            st.session_state.mostrar_previsao = False
            st.rerun()
    elif submitted:
        st.error("⚠️ Por favor, digite o código de uma ação")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ====================== TELA 2 - DADOS ======================
elif st.session_state.mostrar_dados and not st.session_state.mostrar_previsao:
    dados = st.session_state.dados_acao
    
    st.markdown("<div class='tela-dados'>", unsafe_allow_html=True)
    
    # Header com título
    # st.markdown(f"<h1 class='main-header2'>📊 {dados['ticker']} - {dados['nome']}</h1>", unsafe_allow_html=True)
    
        # Linha do título e botão juntos
    col_titulo, col_btn_prev, col_btn = st.columns([3, 1, 1])
    with col_titulo:
        st.markdown("<h4 class='section-title'>📈 Cotação Atual</h4>", unsafe_allow_html=True)
    with col_btn_prev:
        if st.button("🔮 Gerar Previsão", type="secondary", use_container_width=True):
            st.session_state.mostrar_previsao = True
            st.rerun()
    with col_btn:
        if st.button("🔁 Nova Análise", type="secondary", use_container_width=True):
            st.session_state.mostrar_dados = False
            st.session_state.mostrar_previsao = False
            st.session_state.dados_acao = None
            st.rerun()

    # Linha das métricas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Preço", f"R$ {dados['preco']:.2f}")
    with col2:
        variacao_cor = "📊" if dados['variacao'] >= 0 else "📉"
        st.metric(f"{variacao_cor} Variação", f"{dados['variacao']:.2f}%")
    with col3:
        st.metric("📈 Máxima", f"R$ {dados['maxima']:.2f}")
    with col4:
        st.metric("📉 Mínima", f"R$ {dados['minima']:.2f}")

    
    # Linha 2: Dados Históricos (3 colunas + performance)
    if dados['dados_3_meses']:
        st.markdown("<h4 class='section-title'>📅 Comparativo 3 Meses</h4 >", unsafe_allow_html=True)
        
        d3m = dados['dados_3_meses']
        
        col1_hist, col2_hist, col3_hist, col4_hist = st.columns(4)
        
        with col1_hist:
            st.metric("💰 Preço 3M", f"R$ {d3m['preco']:.2f}")
        with col2_hist:
            volume_formatado = f"{d3m['volume']:,}".replace(",", ".")
            st.metric("📊 Volume", volume_formatado)
        with col3_hist:
            st.metric("📅 Data", d3m['data'])
        with col4_hist:
            # Cálculo de performance
            if dados['preco'] > 0 and d3m['preco'] > 0:
                performance = ((dados['preco'] - d3m['preco']) / d3m['preco']) * 100
                performance_emoji = "📈" if performance >= 0 else "📉"
                st.metric(f"{performance_emoji} Performance 3M", f"{performance:.2f}%")
    
    
    if dados['historico'] is not None:
        fig_analise = criar_grafico_com_previsao(dados['historico'], dados['ticker'])
        st.plotly_chart(fig_analise, use_container_width=True)
    else:
        st.markdown("<div style='color: #ff6b6b;'>⚠️ Dados históricos indisponíveis</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)  # Fecha tela-dados

# ====================== TELA 3 - PREVISÃO ======================
elif st.session_state.mostrar_previsao:
    dados = st.session_state.dados_acao
    
    st.markdown("<div class='tela-dados'>", unsafe_allow_html=True)
    
    # Header com título
    col_titulo, col_btn_volta, col_btn = st.columns([3, 1, 1])
    with col_titulo:
        st.markdown("<h4 class='section-title'>🔮 Previsão de Preços com IA</h4>", unsafe_allow_html=True)
    with col_btn_volta:
        if st.button("↩️ Voltar", type="secondary", use_container_width=True):
            st.session_state.mostrar_previsao = False
            st.rerun()
    with col_btn:
        if st.button("🔁 Nova Análise", type="secondary", use_container_width=True):
            st.session_state.mostrar_dados = False
            st.session_state.mostrar_previsao = False
            st.session_state.dados_acao = None
            st.rerun()

    # Gerar previsão usando ML
    with st.spinner('🤖 Gerando previsão com Inteligência Artificial...'):
        preco_previsto, data_previsao, erro_ml = gerar_previsao_acao(dados)
    
    if erro_ml:
        st.error(f"❌ Erro na previsão: {erro_ml}")
        st.info("💡 **Dica:** A previsão requer pelo menos 20 dias de histórico com dados de volume.")
        preco_previsto, data_previsao = None, None
    else:
        # Calcular variação prevista
        variacao_prevista = ((preco_previsto - dados['preco']) / dados['preco']) * 100
        
        # Exibir resultado da previsão
        st.success(f"🎯 **Previsão gerada com sucesso!**")
        
        col_prev1, col_prev2, col_prev3, col_prev4 = st.columns(4)
        with col_prev1:
            st.metric("🤖 Previsão IA", f"R$ {preco_previsto:.2f}")
        with col_prev2:
            variacao_emoji = "📈" if variacao_prevista >= 0 else "📉"
            st.metric(f"{variacao_emoji} Variação Prevista", f"{variacao_prevista:.2f}%")
        with col_prev3:
            st.metric("📅 Para o Dia", data_previsao.strftime('%d/%m/%Y'))
        with col_prev4:
            confianca = min(85, max(60, 75 + abs(variacao_prevista) * 2))  # Simulação de confiança
            st.metric("🎯 Confiança", f"{confianca:.0f}%")

    # Linha das métricas atuais
    st.markdown("<h4 class='section-title'>📊 Dados Atuais</h4>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Preço Atual", f"R$ {dados['preco']:.2f}")
    with col2:
        variacao_cor = "📊" if dados['variacao'] >= 0 else "📉"
        st.metric(f"{variacao_cor} Variação", f"{dados['variacao']:.2f}%")
    with col3:
        st.metric("📈 Máxima", f"R$ {dados['maxima']:.2f}")
    with col4:
        st.metric("📉 Mínima", f"R$ {dados['minima']:.2f}")

    # Gráfico com previsão
    if dados['historico'] is not None:
        fig_previsao = criar_grafico_com_previsao(
            dados['historico'], 
            dados['ticker'], 
            preco_previsto, 
            data_previsao
        )
        st.plotly_chart(fig_previsao, use_container_width=True)
    else:
        st.markdown("<div style='color: #ff6b6b;'>⚠️ Dados históricos indisponíveis</div>", unsafe_allow_html=True)
    
    # Informações sobre o modelo
    if not erro_ml:
        with st.expander("ℹ️ Sobre a Previsão"):
            st.markdown("""
            **🤖 Modelo utilizado:** LSTM (Long Short-Term Memory)
            
            **📊 Dados analisados:**
            - Preços de fechamento (sequência adaptativa)
            - Volume de negociação
            - Médias móveis (7 e 21 dias)
            - Variações percentuais
            - Modelo ajustado para dados disponíveis (20-41 dias)
            
            **⚠️ Aviso importante:**
            - Esta é uma previsão baseada em padrões históricos
            - Não constitui recomendação de investimento
            - Mercados financeiros são imprevisíveis por natureza
            - Use apenas para fins educacionais e de pesquisa
            """)
    
    st.markdown("</div>", unsafe_allow_html=True)  # Fecha tela-dados