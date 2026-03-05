import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import warnings


def criar_modelo_gru(input_shape, forecast_horizon=5, units=50):
    """
    Cria modelo GRU com regularização forte para evitar overfitting

    Args:
        input_shape: Tuple (timesteps, features)
        forecast_horizon: Número de dias a prever
        units: Unidades na primeira camada GRU

    Returns:
        Modelo Keras compilado
    """
    model = Sequential([
        GRU(units, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),  # Dropout alto para regularização
        GRU(units // 2, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(forecast_horizon)  # Output: 5 retornos futuros
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    return model


def walk_forward_split(X, y, n_splits=5, test_size=10):
    """
    Walk-Forward Validation: treina com passado crescente, testa no futuro

    Exemplo com 100 samples, n_splits=5, test_size=10:
    Split 1: Train [0:50],  Test [50:60]
    Split 2: Train [0:60],  Test [60:70]
    Split 3: Train [0:70],  Test [70:80]
    Split 4: Train [0:80],  Test [80:90]
    Split 5: Train [0:90],  Test [90:100]
    """
    n_samples = len(X)
    splits = []

    # Tamanho inicial de treino (pelo menos 50% dos dados)
    initial_train_size = max(int(n_samples * 0.5), n_samples - (n_splits * test_size))

    for i in range(n_splits):
        train_end = initial_train_size + (i * test_size)
        test_start = train_end
        test_end = min(test_start + test_size, n_samples)

        if test_end > n_samples:
            break

        train_idx = list(range(0, train_end))
        test_idx = list(range(test_start, test_end))

        if len(test_idx) >= 5:  # Mínimo para teste
            splits.append((train_idx, test_idx))

    return splits


def criar_splits_temporais(X, y, n_splits=3):
    """Cria splits temporais respeitando a ordem cronológica"""
    splits = []
    total_size = len(X)

    for i in range(n_splits):
        # Tamanho crescente do conjunto de treino
        train_size = int(total_size * (0.5 + i * 0.15))
        test_start = min(train_size, total_size - 10)  # Garantir pelo menos 10 amostras de teste

        train_idx = list(range(train_size))
        test_idx = list(range(test_start, min(test_start + 10, total_size)))

        if len(test_idx) >= 5:  # Mínimo de amostras para teste
            splits.append((train_idx, test_idx))

    return splits


def treinar_modelo_gru_temporal(X, y, validation_split=0.2):
    """
    Treina modelo GRU com early stopping e redução de learning rate

    Args:
        X: Features (samples, timesteps, features)
        y: Targets (samples, forecast_horizon)
        validation_split: Fração para validação

    Returns:
        model: Modelo treinado
        history: Histórico de treinamento
        mae_val: MAE no conjunto de validação
    """
    # Criar modelo
    model = criar_modelo_gru(
        input_shape=(X.shape[1], X.shape[2]),
        forecast_horizon=y.shape[1],
        units=50
    )

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=0
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=0.00001,
        verbose=0
    )

    # Treinar
    history = model.fit(
        X, y,
        epochs=100,
        batch_size=16,
        validation_split=validation_split,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )

    # Avaliar no conjunto de validação
    val_split_idx = int(len(X) * (1 - validation_split))
    X_val = X[val_split_idx:]
    y_val = y[val_split_idx:]

    y_pred_val = model.predict(X_val, verbose=0)
    mae_val = mean_absolute_error(y_val.flatten(), y_pred_val.flatten())

    return model, history, mae_val


def treinar_modelos_financeiros(X, y):
    """
    Treina modelos especializados para dados financeiros com validação temporal
    """
    try:
        from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
        from sklearn.linear_model import Ridge, ElasticNet
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import warnings
        warnings.filterwarnings('ignore')

        modelos = {}
        scores = {}

        # Configurações dos modelos (mais conservadoras)
        model_configs = {
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'ExtraTrees': ExtraTreesRegressor(
                n_estimators=100,
                max_depth=6,
                min_samples_split=15,
                min_samples_leaf=7,
                random_state=42,
                n_jobs=-1
            ),
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        }

        # Validação temporal para cada modelo
        splits = criar_splits_temporais(X, y, n_splits=3)

        if not splits:
            # Fallback: split simples se não conseguir criar splits temporais
            split_point = int(0.8 * len(X))
            X_train, X_test = X[:split_point], X[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]

            for nome, modelo_base in model_configs.items():
                try:
                    # Treinar um modelo para cada dia de previsão
                    modelos_dias = []
                    for dia in range(y.shape[1]):
                        modelo = model_configs[nome]
                        modelo.fit(X_train, y_train[:, dia])
                        modelos_dias.append(modelo)

                    modelos[nome] = modelos_dias

                    # Calcular score médio
                    pred_test = np.array([m.predict(X_test) for m in modelos_dias]).T
                    mae = mean_absolute_error(y_test, pred_test)
                    scores[nome] = {'mae': mae, 'confidence': max(0.3, 1.0 - mae * 10)}

                except Exception as e:
                    print(f"Erro no modelo {nome}: {e}")
                    continue
        else:
            # Validação temporal robusta
            for nome, modelo_base in model_configs.items():
                try:
                    fold_scores = []
                    modelos_finais = []

                    # Treinar e validar em cada fold temporal
                    for train_idx, test_idx in splits:
                        X_train_fold = X[train_idx]
                        X_test_fold = X[test_idx]
                        y_train_fold = y[train_idx]
                        y_test_fold = y[test_idx]

                        # Treinar modelos para cada dia
                        modelos_fold = []
                        for dia in range(y.shape[1]):
                            modelo = model_configs[nome]
                            modelo.fit(X_train_fold, y_train_fold[:, dia])
                            modelos_fold.append(modelo)

                        # Validar
                        pred_fold = np.array([m.predict(X_test_fold) for m in modelos_fold]).T
                        mae_fold = mean_absolute_error(y_test_fold, pred_fold)
                        fold_scores.append(mae_fold)

                        modelos_finais = modelos_fold  # Manter o último conjunto de modelos

                    # Scores finais
                    mae_media = np.mean(fold_scores)
                    mae_std = np.std(fold_scores)
                    confidence = max(0.4, min(0.9, 1.0 - mae_media * 15))

                    modelos[nome] = modelos_finais
                    scores[nome] = {
                        'mae': mae_media,
                        'mae_std': mae_std,
                        'confidence': confidence
                    }

                except Exception as e:
                    print(f"Erro no modelo {nome}: {e}")
                    continue

        return modelos, scores, None

    except Exception as e:
        return None, None, f"Erro no treinamento: {str(e)}"
