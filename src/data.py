import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st


def _normalizar_ticker_br(ticker):
    """Adiciona sufixo .SA para tickers da B3 se necessário."""
    ticker = ticker.upper().strip()
    if not ticker.endswith('.SA'):
        ticker = ticker + '.SA'
    return ticker


def buscar_dados_rapidos(ticker):
    """Busca preço e variação atual para cards de favoritos (rápido)."""
    try:
        ticker_yf = _normalizar_ticker_br(ticker)
        fi = yf.Ticker(ticker_yf).fast_info
        preco = round(float(fi.last_price), 2)
        prev = float(fi.previous_close) if fi.previous_close else preco
        variacao = round(((preco - prev) / prev) * 100, 2) if prev else 0.0
        return {'preco': preco, 'variacao': variacao, 'sucesso': True}
    except Exception:
        return {'preco': 0, 'variacao': 0, 'sucesso': False}


def buscar_dados_acao(ticker):
    """
    Busca dados completos de uma ação da B3 via Yahoo Finance.
    Retorna 5 anos de histórico diário ajustado por splits/dividendos.
    """
    try:
        ticker_original = ticker.upper().strip()
        ticker_yf = _normalizar_ticker_br(ticker_original)

        with st.spinner(f'Buscando dados para {ticker_original}...'):
            yf_ticker = yf.Ticker(ticker_yf)

            # Histórico: 5 anos, dados diários, ajustados automaticamente
            historico = yf.download(
                ticker_yf,
                period='5y',
                interval='1d',
                progress=False,
                auto_adjust=True
            )

            if historico is None or historico.empty:
                return None, f"Ticker {ticker_original} não encontrado na B3"

            # yfinance >= 0.2.x pode retornar MultiIndex em alguns casos
            if isinstance(historico.columns, pd.MultiIndex):
                historico.columns = historico.columns.droplevel(1)

            historico.index = pd.to_datetime(historico.index)
            historico = historico.sort_index()

            # Manter apenas colunas OHLCV
            historico = historico[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

            # Remover linhas com Close zero ou NaN (feriados, etc.)
            historico = historico[historico['Close'] > 0].dropna(subset=['Close'])

            if len(historico) < 30:
                return None, f"Histórico insuficiente para {ticker_original}"

            # Preços atuais via fast_info
            fi = yf_ticker.fast_info
            preco = round(float(fi.last_price), 2)
            prev_close = float(fi.previous_close) if fi.previous_close else preco
            variacao_valor = round(preco - prev_close, 2)
            variacao_pct = round((variacao_valor / prev_close) * 100, 2) if prev_close else 0.0

            day_high = round(float(fi.day_high), 2) if fi.day_high else round(float(historico['High'].iloc[-1]), 2)
            day_low = round(float(fi.day_low), 2) if fi.day_low else round(float(historico['Low'].iloc[-1]), 2)
            day_open = round(float(fi.open), 2) if fi.open else round(float(historico['Open'].iloc[-1]), 2)
            volume_hoje = int(historico['Volume'].iloc[-1])

            # Nome da empresa e market cap (chamada separada, tolera falha)
            nome = ticker_original
            market_cap = 0
            try:
                info = yf_ticker.info
                nome = info.get('shortName') or info.get('longName') or ticker_original
                market_cap = info.get('marketCap', 0) or 0
            except Exception:
                pass

            # Indicadores técnicos básicos para exibição na tela de detalhes
            indicadores = {}
            if len(historico) >= 20:
                indicadores['maxima_52s'] = round(float(historico['High'].tail(252).max()), 2)
                indicadores['minima_52s'] = round(float(historico['Low'].tail(252).min()), 2)
                indicadores['volume_medio'] = int(historico['Volume'].tail(30).mean())
                indicadores['media_20d'] = round(float(historico['Close'].tail(20).mean()), 2)
            if len(historico) >= 50:
                indicadores['media_50d'] = round(float(historico['Close'].tail(50).mean()), 2)
            if len(historico) >= 30:
                returns = historico['Close'].pct_change().tail(30)
                indicadores['volatilidade'] = round(float(returns.std() * np.sqrt(252) * 100), 1)

            # Comparativo 3 meses (~63 dias úteis)
            dados_3_meses = None
            if len(historico) > 63:
                d3m = historico.iloc[-63]
                dados_3_meses = {
                    'data': d3m.name.strftime('%d/%m/%Y'),
                    'preco': round(float(d3m['Close']), 2),
                    'volume': int(d3m['Volume'])
                }

            return {
                'ticker': ticker_original,
                'nome': nome,
                'preco': preco,
                'variacao': variacao_pct,
                'variacao_valor': variacao_valor,
                'maxima': day_high,
                'minima': day_low,
                'volume': volume_hoje,
                'abertura': day_open,
                'fechamento_anterior': round(prev_close, 2),
                'market_cap': market_cap,
                'historico': historico,
                'dados_3_meses': dados_3_meses,
                'indicadores': indicadores
            }, None

    except Exception as e:
        return None, f"Erro ao buscar {ticker}: {str(e)}"
