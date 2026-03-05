import requests
import pandas as pd
import numpy as np
from datetime import datetime
import streamlit as st

API_KEY = "nUUZxG2ZdAWuSkBDhPobC2"


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
