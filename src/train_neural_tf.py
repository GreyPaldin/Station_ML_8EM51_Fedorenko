import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import datetime
import yaml

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# ============================
# ЗАГРУЗКА ПАРАМЕТРОВ
# ============================
with open('params.yaml', 'r', encoding='utf-8') as f:
    params = yaml.safe_load(f)
    nn_params = params.get('neural_networks_tf', {})

# ============================
# ЗАГРУЗКА ДАННЫХ
# ============================
print("📥 Загрузка данных...")
X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
X_val = pd.read_csv('data/processed/X_val.csv')
y_val = pd.read_csv('data/processed/y_val.csv').squeeze()
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

print("⚖️ Нормализация...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_s = scaler_X.fit_transform(X_train)
X_val_s = scaler_X.transform(X_val)
X_test_s = scaler_X.transform(X_test)
y_train_s = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_val_s = scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()
y_test_s = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

os.makedirs('models/neural_tf', exist_ok=True)
os.makedirs('metrics/neural_tf', exist_ok=True)
os.makedirs('reports/neural_tf/learning_curves', exist_ok=True)

# ============================
# ПОСТРОЕНИЕ МОДЕЛИ
# ============================
def build_model(input_dim, layers, activation='relu', dropout_rate=0.0, l2_reg=0.0):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    for units in layers:
        if l2_reg > 0:
            model.add(Dense(units, activation=activation, kernel_regularizer=l2(l2_reg)))
        else:
            model.add(Dense(units, activation=activation))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))
    return model

# ============================
# ГРАФИКИ
# ============================
def plot_learning_curves(history, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history.history['loss']) + 1)
    axes[0].plot(epochs, history.history['loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history.history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_title(f'Loss - {model_name}')
    axes[0].set_xlabel('Эпохи')
    axes[0].set_ylabel('MSE')
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(epochs, history.history['mae'], 'b-', label='Train MAE')
    axes[1].plot(epochs, history.history['val_mae'], 'r-', label='Val MAE')
    axes[1].set_title(f'MAE - {model_name}')
    axes[1].set_xlabel('Эпохи')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(f'reports/neural_tf/learning_curves/{model_name}_learning_curves.png', dpi=150)
    plt.close()

# ============================
# ОБУЧЕНИЕ
# ============================
results = {}
for model_name, cfg in nn_params.items():
    if not cfg.get('enabled', True):
        continue
    print(f"\n{'='*60}")
    print(f"🧠 Обучение: {model_name}")
    print(f"{'='*60}")

    layers = cfg['layers']
    epochs = cfg['epochs']
    batch_size = cfg['batch_size']
    lr = cfg['learning_rate']
    activation = cfg.get('activation', 'relu')
    dropout_rate = cfg.get('dropout_rate', 0.0)
    l2_reg = cfg.get('l2_reg', 0.0)
    patience = cfg.get('patience', 50)

    model = build_model(X_train_s.shape[1], layers, activation, dropout_rate, l2_reg)
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse', metrics=['mae'])

    log_dir = f"reports/neural_tf/tensorboard/{model_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    callbacks = [
        TensorBoard(log_dir=log_dir, histogram_freq=1),
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience//2, min_lr=1e-7)
    ]

    history = model.fit(
        X_train_s, y_train_s,
        validation_data=(X_val_s, y_val_s),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    model.save(f'models/neural_tf/{model_name}.keras')

    y_pred_train = scaler_y.inverse_transform(model.predict(X_train_s, verbose=0)).ravel()
    y_pred_val = scaler_y.inverse_transform(model.predict(X_val_s, verbose=0)).ravel()
    y_pred_test = scaler_y.inverse_transform(model.predict(X_test_s, verbose=0)).ravel()

    def metrics(y_true, y_pred):
        return {
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred))
        }

    res = {
        'model_name': model_name,
        'params': cfg,
        'train': metrics(y_train, y_pred_train),
        'val': metrics(y_val, y_pred_val),
        'test': metrics(y_test, y_pred_test),
        'epochs_trained': len(history.history['loss']),
        'tensorboard_logdir': log_dir
    }

    plot_learning_curves(history, model_name)
    with open(f'metrics/neural_tf/{model_name}_metrics.json', 'w') as f:
        json.dump(res, f, indent=2)

    results[model_name] = res
    print(f"✅ {model_name}: R² = {res['test']['r2']:.4f}, RMSE = {res['test']['rmse']:.4f}")

# ============================
# СВОДКА
# ============================
summary = {
    'models': list(results.keys()),
    'best_r2': max(results.keys(), key=lambda x: results[x]['test']['r2']),
    'best_rmse': min(results.keys(), key=lambda x: results[x]['test']['rmse']),
    'results': {name: {
        'test_r2': res['test']['r2'],
        'test_rmse': res['test']['rmse'],
        'test_mae': res['test']['mae'],
        'epochs': res['epochs_trained']
    } for name, res in results.items()}
}
with open('metrics/neural_tf/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("📊 СВОДКА ПО МОДЕЛЯМ")
print("="*60)
for name, res in summary['results'].items():
    print(f"{name.upper()}: R² = {res['test_r2']:.4f}, RMSE = {res['test_rmse']:.4f}, MAE = {res['test_mae']:.4f}, эпох = {res['epochs']}")
print(f"\n🏆 Лучшая по R²: {summary['best_r2']}")
print(f"🏆 Лучшая по RMSE: {summary['best_rmse']}")
print("\n📊 TensorBoard: tensorboard --logdir=reports/neural_tf/tensorboard")
print("\n✅ Обучение завершено!")