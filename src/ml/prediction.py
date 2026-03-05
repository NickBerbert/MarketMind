import numpy as np
from datetime import timedelta

from src.ml.features import criar_sequencias_temporais
from src.ml.models import treinar_lstm_rapido, treinar_lstm_completo
from src.ml.metrics import (
    verificar_overfitting, calcular_confianca, logar_metricas
)


def prever_com_incerteza(model, X_input, n_iter=50):
    """
    Monte Carlo Dropout: runs n_iter forward passes with dropout active
    to quantify epistemic uncertainty.

    Returns:
        mean_pred, std_pred, lower_bound, upper_bound  (all numpy arrays)
    """
    previsoes = []
    for _ in range(n_iter):
        pred = model(X_input, training=True)
        previsoes.append(pred.numpy())

    previsoes = np.array(previsoes)
    mean_pred = np.mean(previsoes, axis=0)
    std_pred = np.std(previsoes, axis=0)
    lower_bound = np.percentile(previsoes, 2.5, axis=0)
    upper_bound = np.percentile(previsoes, 97.5, axis=0)

    return mean_pred, std_pred, lower_bound, upper_bound


def _proximas_datas_uteis(data_base, n=5):
    datas = []
    d = data_base
    while len(datas) < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            datas.append(d)
    return datas


def gerar_previsao_acao(dados_acao, modo_rapido=True):
    """
    Main prediction entry point. Trains LSTM and returns 5-day price forecast.

    Args:
        dados_acao: dict from buscar_dados_acao — must contain 'historico' and 'preco'
        modo_rapido: True → quick 80/20 split (~30-60s)
                     False → 5-fold walk-forward validation (~2-4 min)

    Returns:
        previsoes:      np.array of 5 forecast prices
        datas_previsao: list of 5 business-day dates
        detalhes:       dict with lower_bound, upper_bound, mae_val,
                        directional_accuracy, gap_ratio, gap_status, modo
        confianca:      float [0.30, 0.65]
        erro:           None or error string
    """
    try:
        historico = dados_acao.get('historico')
        ticker = dados_acao.get('ticker', '???')

        if historico is None or historico.empty:
            return None, None, None, None, "Dados históricos não disponíveis"

        if len(historico) < 80:
            return None, None, None, None, (
                f"Histórico insuficiente: {len(historico)} dias (mínimo: 80)"
            )

        # Build sequences with 60-day window
        X, y, feature_names, scaler, erro_seq = criar_sequencias_temporais(
            historico, window_size=60, forecast_horizon=5
        )

        if erro_seq:
            return None, None, None, None, erro_seq

        if X is None or len(X) < 25:
            return None, None, None, None, (
                f"Sequências insuficientes: {len(X) if X is not None else 0} (mínimo: 25)"
            )

        # Train
        if modo_rapido:
            model, metricas_treino, metricas_val, erro_treino = treinar_lstm_rapido(X, y)
        else:
            model, metricas_treino, metricas_val, erro_treino = treinar_lstm_completo(X, y)

        if erro_treino:
            return None, None, None, None, erro_treino

        # Metrics & confidence
        gap_ratio, gap_status = verificar_overfitting(metricas_treino['mae'], metricas_val['mae'])
        confianca = calcular_confianca(metricas_val['directional_accuracy'])

        logar_metricas(ticker, modo_rapido, metricas_treino, metricas_val,
                       gap_ratio, gap_status, confianca)

        # Monte Carlo prediction on last sequence
        ultima_seq = X[-1:]  # shape (1, 60, n_features)
        mean_pred, std_pred, lower_bound, upper_bound = prever_com_incerteza(
            model, ultima_seq, n_iter=50
        )

        # Convert returns → prices
        preco_atual = float(dados_acao['preco'])
        previsoes = []
        lower_prices = []
        upper_prices = []

        for i in range(5):
            ret = float(mean_pred[0][i])
            ret = np.clip(ret, -0.20, 0.20)

            previsoes.append(preco_atual * (1 + ret))
            lower_prices.append(preco_atual * (1 + float(lower_bound[0][i])))
            upper_prices.append(preco_atual * (1 + float(upper_bound[0][i])))

        previsoes = np.array(previsoes)
        lower_prices = np.array(lower_prices)
        upper_prices = np.array(upper_prices)

        datas_previsao = _proximas_datas_uteis(historico.index.max(), n=5)

        detalhes = {
            'lower_bound': lower_prices,
            'upper_bound': upper_prices,
            'volatilidade': float(np.mean(std_pred)),
            'mae_val': metricas_val['mae'],
            'directional_accuracy': metricas_val['directional_accuracy'],
            'gap_ratio': gap_ratio,
            'gap_status': gap_status,
            'modo': 'rapido' if modo_rapido else 'completo',
        }

        return previsoes, datas_previsao, detalhes, confianca, None

    except Exception as e:
        import traceback
        return None, None, None, None, f"Erro na previsão: {str(e)}\n{traceback.format_exc()}"
