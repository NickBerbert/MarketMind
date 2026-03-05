import numpy as np
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.losses import Huber  # type: ignore


def criar_modelo_lstm(input_shape, forecast_horizon=5):
    """
    LSTM with BatchNorm and Dropout for financial return prediction.

    Architecture:
      LSTM(64, return_sequences=True) + BatchNorm + Dropout(0.2)
      LSTM(32, return_sequences=False) + Dropout(0.2)
      Dense(16, relu) + Dropout(0.1)
      Dense(forecast_horizon)

    Huber loss (delta=0.02) is robust to outlier returns.
    Adam with clipnorm=1.0 prevents gradient explosions.
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.1),
        Dense(forecast_horizon),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
        loss=Huber(delta=0.02),
        metrics=['mae'],
    )
    return model


def walk_forward_split(X, y, n_splits=5, test_size=10):
    """
    Walk-Forward Validation: growing training window, fixed test windows.

    Example (100 samples, n_splits=5, test_size=10):
      Split 1: Train [0:50],  Test [50:60]
      Split 2: Train [0:60],  Test [60:70]
      ...
    """
    n_samples = len(X)
    initial_train_size = max(int(n_samples * 0.5), n_samples - n_splits * test_size)
    splits = []

    for i in range(n_splits):
        train_end = initial_train_size + i * test_size
        test_end = min(train_end + test_size, n_samples)
        if test_end > n_samples:
            break
        test_idx = list(range(train_end, test_end))
        if len(test_idx) >= 5:
            splits.append((list(range(train_end)), test_idx))

    return splits


def _callbacks():
    return [
        EarlyStopping(monitor='val_loss', patience=15,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=8, min_lr=1e-5, verbose=0),
    ]


def treinar_lstm_rapido(X, y):
    """
    Quick mode: single 80/20 temporal split, ~30-60 s.

    Returns:
        model: trained Keras model
        metricas_treino: dict from calcular_metricas on train set
        metricas_val: dict from calcular_metricas on val set
        error: None or error string
    """
    try:
        from src.ml.metrics import calcular_metricas

        split = int(len(X) * 0.80)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        if len(X_train) < 20 or len(X_val) < 5:
            return None, None, None, "Dados insuficientes para treino rápido"

        model = criar_modelo_lstm(
            input_shape=(X.shape[1], X.shape[2]),
            forecast_horizon=y.shape[1],
        )

        model.fit(
            X_train, y_train,
            epochs=80,
            batch_size=16,
            validation_data=(X_val, y_val),
            callbacks=_callbacks(),
            verbose=0,
        )

        y_pred_train = model.predict(X_train, verbose=0)
        y_pred_val = model.predict(X_val, verbose=0)

        metricas_treino = calcular_metricas(y_train, y_pred_train)
        metricas_val = calcular_metricas(y_val, y_pred_val)

        return model, metricas_treino, metricas_val, None

    except Exception as e:
        return None, None, None, f"Erro no treino rápido: {str(e)}"


def treinar_lstm_completo(X, y, n_splits=5):
    """
    Full mode: 5-fold walk-forward validation + final model on all data, ~2-4 min.

    Returns:
        model: final model trained on full dataset
        metricas_treino: dict averaged across folds (train)
        metricas_val: dict averaged across folds (validation)
        error: None or error string
    """
    try:
        from src.ml.metrics import calcular_metricas

        splits = walk_forward_split(X, y, n_splits=n_splits, test_size=max(10, len(X) // 20))

        if not splits:
            # Fallback to quick mode if not enough data for walk-forward
            return treinar_lstm_rapido(X, y)

        fold_train_metrics = []
        fold_val_metrics = []

        for fold_i, (train_idx, test_idx) in enumerate(splits):
            X_tr = X[train_idx]
            y_tr = y[train_idx]
            X_te = X[test_idx]
            y_te = y[test_idx]

            fold_model = criar_modelo_lstm(
                input_shape=(X.shape[1], X.shape[2]),
                forecast_horizon=y.shape[1],
            )
            fold_model.fit(
                X_tr, y_tr,
                epochs=100,
                batch_size=16,
                validation_data=(X_te, y_te),
                callbacks=_callbacks(),
                verbose=0,
            )

            fold_train_metrics.append(calcular_metricas(y_tr, fold_model.predict(X_tr, verbose=0)))
            fold_val_metrics.append(calcular_metricas(y_te, fold_model.predict(X_te, verbose=0)))

            print(f"[MarketMind] Fold {fold_i + 1}/{len(splits)} — "
                  f"val_mae: {fold_val_metrics[-1]['mae']:.5f} | "
                  f"dir_acc: {fold_val_metrics[-1]['directional_accuracy']:.1%}")

        # Aggregate fold metrics
        def _mean_metrics(metrics_list):
            keys = metrics_list[0].keys()
            return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}

        metricas_treino = _mean_metrics(fold_train_metrics)
        metricas_val = _mean_metrics(fold_val_metrics)

        # Final model trained on all data
        final_model = criar_modelo_lstm(
            input_shape=(X.shape[1], X.shape[2]),
            forecast_horizon=y.shape[1],
        )
        final_model.fit(
            X, y,
            epochs=100,
            batch_size=16,
            validation_split=0.10,
            callbacks=_callbacks(),
            verbose=0,
        )

        return final_model, metricas_treino, metricas_val, None

    except Exception as e:
        return None, None, None, f"Erro no treino completo: {str(e)}"
