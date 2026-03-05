import numpy as np
from src.ml.features import preparar_dados_financeiros
from src.ml.models import treinar_modelos_financeiros
from src.ml.prediction import fazer_previsao_financeira


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
