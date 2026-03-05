import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from src.indicators import (
    calcular_rsi, calcular_macd, calcular_bollinger_bands,
    calcular_atr, calcular_stochastic, calcular_obv,
    calcular_williams_r, calcular_roc
)


def _construir_features_brutas(df):
    """
    Stage 1: Build ~35 raw technical features.
    Returns augmented DataFrame with all feature columns.
    """
    d = df.copy()

    close = d['Close']
    high = d['High']
    low = d['Low']
    volume = d['Volume']

    # --- Trend / Moving Averages ---
    d['SMA_5'] = close.rolling(5).mean()
    d['SMA_10'] = close.rolling(10).mean()
    d['SMA_20'] = close.rolling(20).mean()
    d['SMA_50'] = close.rolling(50).mean()
    d['EMA_12'] = close.ewm(span=12).mean()
    d['EMA_26'] = close.ewm(span=26).mean()

    # Relative price position vs MAs (normalised — avoids scale issues)
    d['Price_vs_SMA20'] = (close - d['SMA_20']) / d['SMA_20']
    d['Price_vs_SMA50'] = (close - d['SMA_50']) / d['SMA_50']
    d['SMA5_vs_SMA20'] = (d['SMA_5'] - d['SMA_20']) / d['SMA_20']

    # --- Momentum ---
    d['RSI_14'] = calcular_rsi(close, period=14)
    d['RSI_7'] = calcular_rsi(close, period=7)
    d['MACD'] = calcular_macd(close)
    d['MACD_signal'] = d['MACD'].ewm(span=9).mean()
    d['MACD_hist'] = d['MACD'] - d['MACD_signal']
    d['Stochastic_K'] = calcular_stochastic(high, low, close, period=14)
    d['Williams_R'] = calcular_williams_r(high, low, close, period=14)
    d['ROC_10'] = calcular_roc(close, period=10)
    d['ROC_5'] = calcular_roc(close, period=5)

    # --- Volatility ---
    bb_upper, bb_lower = calcular_bollinger_bands(close, window=20)
    d['BB_upper'] = bb_upper
    d['BB_lower'] = bb_lower
    d['BB_width'] = (bb_upper - bb_lower) / close
    d['BB_position'] = (close - bb_lower) / (bb_upper - bb_lower).where(
        (bb_upper - bb_lower) != 0, 1
    )
    d['ATR_14'] = calcular_atr(high, low, close, period=14)
    d['ATR_ratio'] = d['ATR_14'] / close  # normalised ATR

    d['Return_1d'] = close.pct_change()
    d['Return_3d'] = close.pct_change(3)
    d['Return_5d'] = close.pct_change(5)
    d['Return_10d'] = close.pct_change(10)
    d['Volatility_10d'] = d['Return_1d'].rolling(10).std()
    d['Volatility_20d'] = d['Return_1d'].rolling(20).std()

    # --- Volume ---
    d['Volume_SMA_10'] = volume.rolling(10).mean()
    d['Volume_ratio'] = volume / d['Volume_SMA_10'].where(d['Volume_SMA_10'] != 0, 1)
    d['OBV'] = calcular_obv(close, volume)
    d['OBV_change'] = d['OBV'].pct_change(5).fillna(0)

    # VWAP distance (intraday proxy using daily OHLC)
    typical_price = (high + low + close) / 3
    vwap_proxy = (typical_price * volume).rolling(10).sum() / volume.rolling(10).sum()
    d['VWAP_distance'] = (close - vwap_proxy) / vwap_proxy.where(vwap_proxy != 0, 1)

    # --- Price structure ---
    d['High_Low_ratio'] = (high - low) / close
    d['Open_Close_ratio'] = (close - d['Open']) / d['Open']
    d['Price_above_SMA20'] = (close > d['SMA_20']).astype(int)
    d['SMA_trend'] = (d['SMA_5'] > d['SMA_20']).astype(int)

    return d


def _filtrar_correlacao(X_df, threshold=0.90):
    """
    Stage 2: Remove highly correlated features (|corr| > threshold).
    Keeps the first of each correlated pair.
    """
    corr_matrix = X_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return X_df.drop(columns=to_drop)


def _selecionar_por_importancia(X, y, feature_names, top_n=22):
    """
    Stage 3: Use RandomForest feature importance to keep top_n features.
    y is 1D (next-day return) for the importance calculation only.
    """
    rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)
    importances = rf.feature_importances_
    ranked = np.argsort(importances)[::-1]
    selected = ranked[:top_n]
    selected_sorted = sorted(selected)  # preserve chronological column order
    return [feature_names[i] for i in selected_sorted]


def preparar_dados_financeiros(historico_df, dias_previsao=5):
    """
    Full 3-stage feature pipeline for tree-based models (flat 2D input).
    Returns X (n_samples, n_features), y (n_samples, dias_previsao), df, error.
    """
    try:
        if historico_df is None or historico_df.empty:
            return None, None, None, "Dados históricos insuficientes"

        if len(historico_df) < 60:
            return None, None, None, f"Histórico insuficiente: {len(historico_df)} dias (mínimo: 60)"

        # Stage 1
        df = _construir_features_brutas(historico_df)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        if len(df) < 40:
            return None, None, None, "Dados insuficientes após limpeza de features"

        # Build targets first (needed for stage 3 importance)
        y = []
        for i in range(len(df) - dias_previsao):
            current_price = df['Close'].iloc[i]
            future_returns = []
            for j in range(1, dias_previsao + 1):
                future_price = df['Close'].iloc[i + j]
                future_returns.append((future_price - current_price) / current_price)
            if len(future_returns) == dias_previsao:
                y.append(future_returns)

        if len(y) < 20:
            return None, None, None, "Dados insuficientes para criar targets"

        y = np.array(y)

        # Stage 2: correlation filter
        feature_cols_raw = [c for c in df.columns if c not in
                            ['Open', 'High', 'Low', 'Close', 'Volume',
                             'BB_upper', 'BB_lower', 'SMA_5', 'SMA_10',
                             'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                             'Volume_SMA_10', 'OBV', 'ATR_14',
                             'MACD_signal']]

        X_df = df[feature_cols_raw].iloc[:len(y)]
        X_df_filtered = _filtrar_correlacao(X_df, threshold=0.90)

        # Stage 3: RF importance selection
        selected_features = _selecionar_por_importancia(
            X_df_filtered.values, y[:, 0],
            list(X_df_filtered.columns), top_n=min(22, len(X_df_filtered.columns))
        )

        X_final = X_df_filtered[selected_features].values

        print(f"[MarketMind] Features selecionadas ({len(selected_features)}): {selected_features}")

        return X_final, y, df, None

    except Exception as e:
        return None, None, None, f"Erro ao preparar dados: {str(e)}"


def criar_sequencias_temporais(df, window_size=60, forecast_horizon=5, feature_cols=None):
    """
    Creates sliding-window 3D sequences for LSTM input.
    Runs all 3 pipeline stages internally and returns:
      X (samples, timesteps, features), y (samples, forecast_horizon),
      feature_names, scaler, error
    """
    try:
        if df is None or df.empty:
            return None, None, None, None, "DataFrame vazio"

        # Stage 1
        df_aug = _construir_features_brutas(df)
        df_aug = df_aug.replace([np.inf, -np.inf], np.nan).dropna()

        if len(df_aug) < window_size + forecast_horizon + 10:
            return None, None, None, None, (
                f"Histórico insuficiente para janela de {window_size} dias"
            )

        # Candidate features (exclude raw OHLCV and intermediate cols)
        exclude = {'Open', 'High', 'Low', 'Close', 'Volume',
                   'BB_upper', 'BB_lower', 'SMA_5', 'SMA_10',
                   'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                   'Volume_SMA_10', 'OBV', 'ATR_14', 'MACD_signal'}
        candidate_cols = [c for c in df_aug.columns if c not in exclude]

        # Build targets for importance step
        n = len(df_aug)
        y_importance = []
        for i in range(n - forecast_horizon):
            cp = df_aug['Close'].iloc[i]
            fp = df_aug['Close'].iloc[i + 1]
            y_importance.append((fp - cp) / cp)

        X_imp = df_aug[candidate_cols].iloc[:len(y_importance)]
        X_imp_filtered = _filtrar_correlacao(X_imp, threshold=0.90)

        selected_features = _selecionar_por_importancia(
            X_imp_filtered.values, np.array(y_importance),
            list(X_imp_filtered.columns), top_n=min(22, len(X_imp_filtered.columns))
        )

        print(f"[MarketMind] LSTM features ({len(selected_features)}): {selected_features}")

        # Normalise
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df_aug[selected_features].values)

        X, y = [], []
        for i in range(window_size, n - forecast_horizon):
            X.append(features_scaled[i - window_size:i])
            current_price = df_aug['Close'].iloc[i]
            future_returns = [
                (df_aug['Close'].iloc[i + j] - current_price) / current_price
                for j in range(1, forecast_horizon + 1)
            ]
            y.append(future_returns)

        X = np.array(X)
        y = np.array(y)

        return X, y, selected_features, scaler, None

    except Exception as e:
        return None, None, None, None, f"Erro ao criar sequências: {str(e)}"
