import numpy as np


def calcular_metricas(y_true, y_pred):
    """
    Computes MAE, MAPE, RMSE, and directional accuracy.

    Args:
        y_true: array-like of actual returns
        y_pred: array-like of predicted returns

    Returns:
        dict with keys: mae, mape, rmse, directional_accuracy
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) \
        if nonzero.any() else 0.0

    direction_match = np.sign(y_true) == np.sign(y_pred)
    directional_accuracy = float(np.mean(direction_match))

    return {
        'mae': mae,
        'mape': mape,
        'rmse': rmse,
        'directional_accuracy': directional_accuracy,
    }


def verificar_overfitting(train_mae, val_mae):
    """
    Gap ratio = val_mae / train_mae.
    > 2.0  → strong overfitting
    1.5-2.0 → moderate
    < 1.5  → acceptable
    """
    if train_mae == 0:
        return None, 'indefinido'
    gap = val_mae / train_mae
    if gap > 2.0:
        status = 'forte'
    elif gap > 1.5:
        status = 'moderado'
    else:
        status = 'aceitavel'
    return round(gap, 2), status


def calcular_confianca(directional_accuracy):
    """
    Maps directional accuracy → honest confidence score.
    50% accuracy → 30% confidence (random baseline)
    60% accuracy → 65% confidence (ceiling)
    Formula: max(0.30, min(0.65, 0.30 + (da - 0.50) * 3.5))
    """
    return round(max(0.30, min(0.65, 0.30 + (directional_accuracy - 0.50) * 3.5)), 4)


def logar_metricas(ticker, modo, metricas_treino, metricas_val, gap_ratio, gap_status, confianca):
    """Prints a structured metrics summary to the terminal."""
    prefix = "[MarketMind]"
    print(f"{prefix} ====== Métricas — {ticker} | Modo: {'Rápido' if modo else 'Completo'} ======")
    print(f"{prefix}  Treino  — MAE: {metricas_treino['mae']:.5f} | "
          f"RMSE: {metricas_treino['rmse']:.5f} | "
          f"Dir.Acc: {metricas_treino['directional_accuracy']:.1%}")
    print(f"{prefix}  Validação — MAE: {metricas_val['mae']:.5f} | "
          f"RMSE: {metricas_val['rmse']:.5f} | "
          f"Dir.Acc: {metricas_val['directional_accuracy']:.1%}")
    if gap_ratio is not None:
        print(f"{prefix}  Overfitting gap ratio: {gap_ratio} ({gap_status})")
    print(f"{prefix}  Confiança calculada: {confianca:.1%}")
    print(f"{prefix} {'=' * 55}")
