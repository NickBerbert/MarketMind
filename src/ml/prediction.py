import numpy as np
from datetime import timedelta
from src.ml.features import preparar_dados_financeiros, criar_sequencias_temporais
from src.ml.models import treinar_modelo_gru_temporal, treinar_modelos_financeiros


def prever_com_incerteza(model, X_input, n_iter=50):
    """
    Monte Carlo Dropout: faz múltiplas previsões com dropout ativo
    para quantificar incerteza epistêmica

    Args:
        model: Modelo Keras com Dropout
        X_input: Input para previsão
        n_iter: Número de iterações (quanto maior, mais preciso)

    Returns:
        mean_pred: Previsão média
        std_pred: Desvio padrão (incerteza)
        lower_bound: Limite inferior (95% confiança)
        upper_bound: Limite superior (95% confiança)
    """
    previsoes = []

    for _ in range(n_iter):
        # training=True mantém Dropout ativo durante inferência
        pred = model(X_input, training=True)
        previsoes.append(pred.numpy())

    previsoes = np.array(previsoes)

    # Estatísticas
    mean_pred = np.mean(previsoes, axis=0)
    std_pred = np.std(previsoes, axis=0)

    # Intervalos de confiança 95% (2.5% e 97.5% percentis)
    lower_bound = np.percentile(previsoes, 2.5, axis=0)
    upper_bound = np.percentile(previsoes, 97.5, axis=0)

    return mean_pred, std_pred, lower_bound, upper_bound


def fazer_previsao_financeira(modelos, scores, df_original, X_features):
    """
    Faz previsões com base em retornos percentuais e converte para preços
    """
    try:
        # Pegar as features da última observação
        last_features = X_features[-1:].reshape(1, -1)

        # Preço atual para conversão
        preco_atual = df_original['Close'].iloc[-1]

        # Previsões de cada modelo (retornos percentuais)
        previsoes_retornos = {}
        pesos_modelos = {}

        for nome_modelo, lista_modelos in modelos.items():
            try:
                # Pegar o score de confiança do modelo
                confianca = scores.get(nome_modelo, {}).get('confidence', 0.5)
                pesos_modelos[nome_modelo] = confianca

                # Fazer previsão para cada dia
                pred_dias = []
                for modelo_dia in lista_modelos:
                    pred = modelo_dia.predict(last_features)[0]
                    pred_dias.append(pred)

                previsoes_retornos[nome_modelo] = np.array(pred_dias)

            except Exception as e:
                print(f"Erro no modelo {nome_modelo}: {e}")
                continue

        if not previsoes_retornos:
            return None, None, None, None, "Nenhum modelo conseguiu fazer previsões"

        # Normalizar pesos
        total_peso = sum(pesos_modelos.values())
        if total_peso > 0:
            pesos_modelos = {k: v/total_peso for k, v in pesos_modelos.items()}
        else:
            # Pesos iguais se não tiver scores
            peso_igual = 1.0 / len(pesos_modelos)
            pesos_modelos = {k: peso_igual for k in pesos_modelos.keys()}

        # Ensemble das previsões (média ponderada)
        num_dias = len(list(previsoes_retornos.values())[0])
        previsao_retornos_ensemble = np.zeros(num_dias)

        for nome_modelo, pred_retornos in previsoes_retornos.items():
            peso = pesos_modelos[nome_modelo]
            previsao_retornos_ensemble += pred_retornos * peso

        # Converter retornos para preços absolutos
        previsao_precos = []
        preco_base = preco_atual

        for i, retorno in enumerate(previsao_retornos_ensemble):
            # Limitar retornos extremos (máximo ±20% por dia)
            retorno_limitado = np.clip(retorno, -0.20, 0.20)
            novo_preco = preco_base * (1 + retorno_limitado)
            previsao_precos.append(novo_preco)
            preco_base = novo_preco  # Usar o preço previsto como base para o próximo dia

        previsao_precos = np.array(previsao_precos)

        # Criar datas de previsão (apenas dias úteis)
        datas_previsao = []
        data_atual = df_original.index.max()
        dias_adicionados = 0

        while dias_adicionados < num_dias:
            data_atual += timedelta(days=1)
            if data_atual.weekday() < 5:  # Segunda a sexta
                datas_previsao.append(data_atual)
                dias_adicionados += 1

        # Calcular confiança geral baseada na concordância entre modelos
        if len(previsoes_retornos) > 1:
            # Calcular dispersão entre modelos
            retornos_array = np.array([pred for pred in previsoes_retornos.values()])
            dispersao = np.std(retornos_array, axis=0).mean()

            # Confiança inversamente proporcional à dispersão
            confianca_geral = max(0.5, min(0.9, 1.0 - dispersao * 20))
        else:
            # Se só tem um modelo, usar a confiança dele
            confianca_geral = list(scores.values())[0].get('confidence', 0.6)

        return previsao_precos, datas_previsao, previsoes_retornos, confianca_geral, None

    except Exception as e:
        return None, None, None, None, f"Erro na previsão: {str(e)}"


def gerar_previsao_acao(dados_acao):
    """
    Função principal com modelo GRU temporal e quantificação de incerteza

    Returns:
        previsoes: Array com 5 preços futuros
        datas: Datas correspondentes
        detalhes: Dict com lower_bound, upper_bound, volatilidade
        confianca: Confiança realista (0.3-0.65)
        erro: Mensagem de erro (None se sucesso)
    """
    try:
        historico = dados_acao.get('historico')
        if historico is None or historico.empty:
            return None, None, None, None, "Dados históricos não disponíveis"

        if len(historico) < 50:
            return None, None, None, None, f"Histórico insuficiente: {len(historico)} dias (mínimo: 50)"

        # Usar a função existente preparar_dados_financeiros para adicionar features
        _, _, df, erro_prep = preparar_dados_financeiros(historico, dias_previsao=5)

        if erro_prep:
            return None, None, None, None, erro_prep

        if df is None or len(df) < 30:
            return None, None, None, None, "Dados insuficientes após limpeza"

        # Criar sequências temporais para GRU
        # Ajustar window_size baseado na quantidade de dados disponível
        if len(df) < 50:
            window_size = 10  # Janela menor para poucos dados
        else:
            window_size = 15  # Janela média (mais sequências que 20)

        X, y, feature_names, scaler, erro_seq = criar_sequencias_temporais(
            df, window_size=window_size, forecast_horizon=5
        )

        if erro_seq:
            return None, None, None, None, erro_seq

        if len(X) < 20:
            return None, None, None, None, f"Sequências insuficientes: {len(X)} (mínimo: 20)"

        # Treinar modelo GRU com toda a sequência temporal
        # (no mundo real, usa todo o passado disponível)
        # Ajustar validation_split baseado no tamanho dos dados
        if len(X) < 40:
            validation_split = 0.15  # Menos validação para datasets pequenos
        else:
            validation_split = 0.2

        model, history, mae_val = treinar_modelo_gru_temporal(X, y, validation_split=validation_split)

        # Pegar última sequência para fazer previsão
        ultima_sequencia = X[-1:]  # Shape: (1, 20, 15)

        # Prever COM incerteza (Monte Carlo Dropout)
        mean_pred, std_pred, lower_bound, upper_bound = prever_com_incerteza(
            model, ultima_sequencia, n_iter=50
        )

        # Converter retornos para preços
        preco_atual = dados_acao['preco']
        previsoes = []
        lower_prices = []
        upper_prices = []

        for i in range(5):
            # Previsão média
            preco = preco_atual * (1 + mean_pred[0][i])
            previsoes.append(preco)

            # Limites do intervalo de confiança
            lower_p = preco_atual * (1 + lower_bound[0][i])
            upper_p = preco_atual * (1 + upper_bound[0][i])

            lower_prices.append(lower_p)
            upper_prices.append(upper_p)

        previsoes = np.array(previsoes)
        lower_prices = np.array(lower_prices)
        upper_prices = np.array(upper_prices)

        # Calcular CONFIANÇA REALISTA baseada na volatilidade das previsões
        volatilidade_previsao = np.mean(std_pred)

        # Confiança: 30-65% (realista para mercado financeiro)
        # Volatilidade alta = confiança baixa
        confianca = max(0.30, min(0.65, 1.0 - volatilidade_previsao * 10))

        # Ajustar confiança se MAE de validação for alto
        if mae_val > 0.03:  # MAE > 3%
            confianca = max(0.30, confianca * 0.8)

        # Gerar datas de previsão (apenas dias úteis)
        datas_previsao = []
        data_atual = df.index.max()
        dias_adicionados = 0

        while dias_adicionados < 5:
            data_atual += timedelta(days=1)
            if data_atual.weekday() < 5:  # Segunda a sexta
                datas_previsao.append(data_atual)
                dias_adicionados += 1

        # Detalhes incluem intervalos de confiança
        detalhes = {
            'lower_bound': lower_prices,
            'upper_bound': upper_prices,
            'volatilidade': volatilidade_previsao,
            'mae_val': mae_val,
            'std_pred': std_pred,
            'epochs_trained': len(history.history['loss'])
        }

        return previsoes, datas_previsao, detalhes, confianca, None

    except Exception as e:
        import traceback
        erro_detalhado = f"Erro na previsão: {str(e)}\n{traceback.format_exc()}"
        return None, None, None, None, erro_detalhado
