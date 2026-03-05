import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.indicators import calcular_rsi, calcular_macd, calcular_bollinger_bands


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
