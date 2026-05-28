import pandas as pd
import numpy as np
import os
import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ========== ЗАГРУЗКА ДАННЫХ ==========
print("Загрузка данных...")
X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
X_val = pd.read_csv('data/processed/X_val.csv')
y_val = pd.read_csv('data/processed/y_val.csv').squeeze()
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

# Нормализация
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_s = scaler_X.fit_transform(X_train)
X_val_s = scaler_X.transform(X_val)
X_test_s = scaler_X.transform(X_test)
y_train_s = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_val_s = scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()

# ========== ПАРАМЕТРЫ МОДЕЛЕЙ ==========
models_config = {
    'nn_small': {
        'layers': [64, 32],
        'batch_size': 64,
        'learning_rate': 0.001,
        'epochs': 200
    },
    'nn_medium': {
        'layers': [128, 64, 32],
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 300
    },
    'nn_deep': {
        'layers': [256, 128, 64, 32],
        'batch_size': 16,
        'learning_rate': 0.0005,
        'epochs': 500
    }
}

# ========== ОБУЧЕНИЕ ДЛЯ TENSORBOARD ==========
for name, cfg in models_config.items():
    print(f"\n{'='*60}")
    print(f"🔥 TensorBoard обучение: {name}")
    print(f"{'='*60}")
    
    # Создание модели
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(X_train_s.shape[1],)))
    
    for units in cfg['layers']:
        model.add(tf.keras.layers.Dense(units, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))  # регуляризация
    
    model.add(tf.keras.layers.Dense(1))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg['learning_rate']),
        loss='mse',
        metrics=['mae']
    )
    
    # TensorBoard callback
    log_dir = f"reports/neural/tensorboard/{name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    
    # Обучение
    history = model.fit(
        X_train_s, y_train_s,
        validation_data=(X_val_s, y_val_s),
        epochs=cfg['epochs'],
        batch_size=cfg['batch_size'],
        callbacks=[tensorboard_callback, early_stop],
        verbose=1
    )
    
    # Сохранение модели
    os.makedirs('models/neural_tf', exist_ok=True)
    model.save(f'models/neural_tf/{name}.keras')
    
    # Сохранение истории обучения (кривые)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{name} - Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title(f'{name} - MAE')
    plt.legend()
    
    plt.tight_layout()
    os.makedirs('reports/neural/tf_curves', exist_ok=True)
    plt.savefig(f'reports/neural/tf_curves/{name}_history.png', dpi=150)
    plt.close()
    
    print(f"{name} обучена")
    print(f"   TensorBoard: tensorboard --logdir={log_dir}")
    print(f"   График: reports/neural/tf_curves/{name}_history.png")

print("\nВсе модели для TensorBoard обучены!")
print("\nЗапусти TensorBoard:")
print("   tensorboard --logdir=reports/neural/tensorboard")