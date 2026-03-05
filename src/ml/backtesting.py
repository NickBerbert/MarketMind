import numpy as np
from src.ml.features import criar_sequencias_temporais
from src.ml.models import treinar_lstm_rapido
from src.ml.metrics import calcular_confianca, verificar_overfitting


def fazer_backtesting(dados_acao, dias_retroativos=10, valor_investimento=1000):
    """
    Simulates investments based on LSTM predictions from past cut-off points.
    Always uses quick mode (single 80/20 split) to keep runtime manageable.
    """
    try:
        historico = dados_acao.get('historico')
        if historico is None or historico.empty or len(historico) < 120:
            return None, "Dados históricos insuficientes para backtesting (mínimo: 120 dias)"

        resultados = []

        for i in range(dias_retroativos, 0, -1):
            historico_corte = historico.iloc[:-i].copy()

            if len(historico_corte) < 100:
                continue

            X, y, feature_names, scaler, erro_seq = criar_sequencias_temporais(
                historico_corte, window_size=60, forecast_horizon=5
            )

            if erro_seq or X is None or len(X) < 25:
                continue

            model, metricas_treino, metricas_val, erro_treino = treinar_lstm_rapido(X, y)

            if erro_treino or model is None:
                continue

            # Predict on last sequence (most recent cut-off point)
            from src.ml.prediction import prever_com_incerteza
            mean_pred, _, _, _ = prever_com_incerteza(model, X[-1:], n_iter=20)

            gap_ratio, _ = verificar_overfitting(metricas_treino['mae'], metricas_val['mae'])
            confianca = calcular_confianca(metricas_val['directional_accuracy'])

            data_decisao = historico_corte.index[-1]
            preco_compra = float(historico_corte['Close'].iloc[-1])

            previsoes_retornos = mean_pred[0]  # shape (5,)
            previsoes_precos = np.array([
                preco_compra * (1 + np.clip(float(r), -0.20, 0.20))
                for r in previsoes_retornos
            ])

            retorno_previsto_1d = float(previsoes_retornos[0])
            retorno_previsto_5d = float(previsoes_retornos[4])
            deveria_investir = retorno_previsto_5d > 0.02 and confianca > 0.50

            try:
                idx_decisao = historico.index.get_loc(data_decisao)

                preco_real_1d = retorno_real_1d = None
                if idx_decisao + 1 < len(historico):
                    preco_real_1d = float(historico['Close'].iloc[idx_decisao + 1])
                    retorno_real_1d = (preco_real_1d - preco_compra) / preco_compra

                preco_real_5d = retorno_real_5d = None
                if idx_decisao + 5 < len(historico):
                    preco_real_5d = float(historico['Close'].iloc[idx_decisao + 5])
                    retorno_real_5d = (preco_real_5d - preco_compra) / preco_compra

                resultado = {
                    'data_decisao': data_decisao.strftime('%d/%m/%Y'),
                    'preco_compra': preco_compra,
                    'previsao_1d': float(previsoes_precos[0]),
                    'previsao_5d': float(previsoes_precos[4]),
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
                    'lucro_5d': None,
                }

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
                print(f"[MarketMind] Backtesting — erro em {data_decisao}: {e}")
                continue

        return resultados, None

    except Exception as e:
        return None, f"Erro no backtesting: {str(e)}"


def analisar_performance_backtesting(resultados_backtesting):
    """Computes summary performance metrics from backtesting results."""
    if not resultados_backtesting:
        return None

    resultados_validos = [r for r in resultados_backtesting if r['retorno_real_5d'] is not None]
    investimentos_feitos = [r for r in resultados_validos if r['deveria_investir']]

    if not resultados_validos:
        return None

    total_simulacoes = len(resultados_validos)
    total_investimentos = len(investimentos_feitos)

    acertos_1d = [r for r in resultados_validos if r['acerto_direcao_1d']]
    acertos_5d = [r for r in resultados_validos if r['acerto_direcao_5d']]
    acuracia_1d = len(acertos_1d) / total_simulacoes if total_simulacoes else 0
    acuracia_5d = len(acertos_5d) / total_simulacoes if total_simulacoes else 0

    lucro_total_1d = lucro_total_5d = taxa_sucesso = retorno_medio_5d = 0
    if investimentos_feitos:
        lucros_1d = [r['lucro_1d'] for r in investimentos_feitos if r['lucro_1d'] is not None]
        lucros_5d = [r['lucro_5d'] for r in investimentos_feitos if r['lucro_5d'] is not None]
        lucro_total_1d = sum(lucros_1d) if lucros_1d else 0
        lucro_total_5d = sum(lucros_5d) if lucros_5d else 0
        taxa_sucesso = len([l for l in lucros_5d if l > 0]) / len(lucros_5d) if lucros_5d else 0
        retorno_medio_5d = (
            sum(r['retorno_real_5d'] for r in investimentos_feitos) / len(investimentos_feitos)
        )

    confianca_media = sum(r['confianca'] for r in resultados_validos) / total_simulacoes

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
        'resultados_detalhados': resultados_validos,
    }
