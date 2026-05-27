import pandas as pd
import numpy as np
import json
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    max_error, median_absolute_error
)
from sklearn.preprocessing import StandardScaler
import yaml
from datetime import datetime
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import sys
import threading

# ========== НАСТРОЙКА ПАРАЛЛЕЛЬНОСТИ ==========
N_JOBS = multiprocessing.cpu_count()
print(f"⚡ Доступно ядер CPU: {N_JOBS}")
print(f"⚡ Будем использовать: {N_JOBS} потоков")

# ========== КЛАСС ДЛЯ УПРАВЛЕНИЯ ПРОГРЕСС-БАРАМИ ==========
class MultiModelProgress:
    """Класс для управления прогресс-барами нескольких моделей"""
    
    def __init__(self, total_models):
        self.total_models = total_models
        self.model_bars = {}
        self.main_bar = tqdm(
            total=total_models,
            desc="📊 ВСЕ МОДЕЛИ",
            position=0,
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
        self.lock = threading.Lock()
        
    def add_model(self, model_name, total_stages=5):
        """Добавить прогресс-бар для конкретной модели"""
        with self.lock:
            self.model_bars[model_name] = tqdm(
                total=100,
                desc=f"   📌 {model_name}",
                position=len(self.model_bars) + 1,
                leave=False,
                bar_format="   {l_bar}{bar}| {percentage:3.0f}% [{elapsed}]"
            )
            return self.model_bars[model_name]
    
    def update_training_progress(self, model_name, current_iter, loss, max_iter, eta=""):
        """Обновляет прогресс обучения в реальном времени"""
        with self.lock:
            if model_name in self.model_bars:
                percent = (current_iter / max_iter) * 100
                self.model_bars[model_name].set_postfix({
                    'iter': f"{current_iter}/{max_iter}",
                    'loss': f"{loss:.4f}",
                    'eta': eta
                })
                self.model_bars[model_name].n = percent
                self.model_bars[model_name].refresh()
    
    def update_model(self, model_name, stage_name=""):
        """Обновить прогресс модели на этапах до/после обучения"""
        with self.lock:
            if model_name in self.model_bars:
                bar = self.model_bars[model_name]
                bar.set_postfix({"stage": stage_name})
                bar.update(1)
                bar.refresh()
    
    def finish_model(self, model_name):
        """Завершить модель"""
        with self.lock:
            if model_name in self.model_bars:
                self.model_bars[model_name].n = 100
                self.model_bars[model_name].set_postfix({"stage": "готово"})
                self.model_bars[model_name].refresh()
                time.sleep(0.5)
                self.model_bars[model_name].close()
                del self.model_bars[model_name]
            self.main_bar.update(1)
    
    def close(self):
        """Закрыть все бары"""
        with self.lock:
            for bar in self.model_bars.values():
                bar.close()
            self.main_bar.close()

# ========== ЗАГРУЗКА ПАРАМЕТРОВ ==========
print("📋 Загрузка параметров...")
with open('params.yaml', 'r', encoding='utf-8') as f:
    params = yaml.safe_load(f)
    nn_params = params['neural_networks']

# ========== ЗАГРУЗКА И НОРМАЛИЗАЦИЯ ДАННЫХ ==========
print("📥 Загрузка данных...")
X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
X_val = pd.read_csv('data/processed/X_val.csv')
y_val = pd.read_csv('data/processed/y_val.csv').squeeze()
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

print(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

print("⚖️ Нормализация данных...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

# Создаём директории
os.makedirs('models/neural', exist_ok=True)
os.makedirs('metrics/neural', exist_ok=True)
os.makedirs('reports/neural/learning_curves', exist_ok=True)
os.makedirs('reports/neural/weights', exist_ok=True)

# ========== КОНВЕРТЕР ДЛЯ JSON ==========
def convert_to_serializable(obj):
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

# ========== ВСЕ МЕТРИКИ ==========
def calculate_all_metrics(y_true, y_pred):
    metrics = {}
    metrics['mse'] = float(mean_squared_error(y_true, y_pred))
    metrics['rmse'] = float(np.sqrt(metrics['mse']))
    metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
    metrics['median_ae'] = float(median_absolute_error(y_true, y_pred))
    metrics['max_error'] = float(max_error(y_true, y_pred))
    metrics['r2'] = float(r2_score(y_true, y_pred))
    
    residuals = y_true - y_pred
    metrics['residuals_mean'] = float(np.mean(residuals))
    metrics['residuals_std'] = float(np.std(residuals))
    metrics['residuals_skew'] = float(pd.Series(residuals).skew())
    
    return metrics

# ========== КРИВАЯ ОБУЧЕНИЯ ==========
def plot_learning_curve(model, model_name, X, y, X_val, y_val):
    train_sizes = np.linspace(0.1, 1.0, 6) * len(X)
    train_scores = []
    val_scores = []
    
    for size in train_sizes:
        size = int(size)
        X_subset = X[:size]
        y_subset = y[:size]
        
        model_copy = MLPRegressor(**model.get_params())
        model_copy.max_iter = 100
        model_copy.fit(X_subset, y_subset)
        
        train_pred = model_copy.predict(X_subset)
        val_pred = model_copy.predict(X_val)
        
        train_scores.append(mean_squared_error(y_subset, train_pred))
        val_scores.append(mean_squared_error(y_val, val_pred))
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores, 'o-', label='Train MSE', color='blue', linewidth=2)
    plt.plot(train_sizes, val_scores, 'o-', label='Validation MSE', color='red', linewidth=2)
    plt.xlabel('Размер обучающей выборки')
    plt.ylabel('MSE')
    plt.title(f'Кривая обучения - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'reports/neural/learning_curves/{model_name}_learning.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'train_sizes': train_sizes.tolist(),
        'train_mse': train_scores,
        'val_mse': val_scores
    }

# ========== КРИВАЯ ПОТЕРЬ ==========
def plot_loss_curve(model, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(model.loss_curve_, label='Train loss', color='blue', linewidth=2)
    plt.xlabel('Итерации')
    plt.ylabel('Loss')
    plt.title(f'Кривая потерь - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'reports/neural/learning_curves/{model_name}_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return [float(x) for x in model.loss_curve_]

# ========== ГИСТОГРАММЫ ВЕСОВ ==========
def plot_weight_histograms(model, model_name):
    n_layers = len(model.coefs_)
    fig, axes = plt.subplots(2, n_layers, figsize=(5*n_layers, 8))
    
    if n_layers == 1:
        axes = axes.reshape(2, 1)
    
    weight_stats = {}
    
    for i, (coef, intercept) in enumerate(zip(model.coefs_, model.intercepts_)):
        axes[0, i].hist(coef.flatten(), bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0, i].set_title(f'Слой {i+1} - Веса')
        axes[0, i].axvline(x=0, color='red', linestyle='--', linewidth=1)
        axes[0, i].grid(True, alpha=0.3)
        
        axes[1, i].hist(intercept, bins=30, alpha=0.7, color='coral', edgecolor='black')
        axes[1, i].set_title(f'Слой {i+1} - Смещения')
        axes[1, i].axvline(x=0, color='red', linestyle='--', linewidth=1)
        axes[1, i].grid(True, alpha=0.3)
        
        weight_stats[f'layer_{i+1}'] = {
            'weights_mean': float(np.mean(coef)),
            'weights_std': float(np.std(coef)),
            'bias_mean': float(np.mean(intercept))
        }
    
    plt.suptitle(f'Распределение весов - {model_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'reports/neural/weights/{model_name}_histograms.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return weight_stats

# ========== ОБУЧЕНИЕ ОДНОЙ МОДЕЛИ ==========
def train_single_model(model_config, model_name, progress_bar_manager):
    """Обучает одну модель с периодическим логированием прогресса"""
    
    try:
        # Создаём прогресс-бар для этой модели
        max_iter = model_config.get('max_iter', 200)
        model_bar = progress_bar_manager.add_model(model_name, total_stages=6)
        
        def update_progress(stage_name):
            progress_bar_manager.update_model(model_name, stage_name)
        
        update_progress("инициализация")
        params = {k: v for k, v in model_config.items() if k != 'enabled'}
        
        # Настройка использования всех ядер
        os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
        os.environ["MKL_NUM_THREADS"] = str(N_JOBS)
        
        update_progress("обучение")
        start_time = time.time()
        
        # Используем warm_start для отслеживания прогресса
        if max_iter > 50:
            # Разбиваем обучение на чанки
            chunk_size = 10
            n_iter_total = max_iter
            
            # Создаём модель с warm_start
            chunk_params = params.copy()
            chunk_params['warm_start'] = True
            chunk_params['max_iter'] = chunk_size
            
            chunk_model = MLPRegressor(**chunk_params)
            
            for chunk in range(0, n_iter_total, chunk_size):
                # Обучаем очередной чанк
                if chunk == 0:
                    chunk_model.fit(X_train_scaled, y_train_scaled)
                else:
                    chunk_model.fit(X_train_scaled, y_train_scaled)
                
                # Получаем текущий loss
                current_loss = chunk_model.loss_curve_[-1] if hasattr(chunk_model, 'loss_curve_') and chunk_model.loss_curve_ else 0
                
                # Обновляем прогресс
                current_iter = min(chunk + chunk_size, n_iter_total)
                
                # ETA
                elapsed = time.time() - start_time
                if current_iter > 0:
                    eta = (elapsed / current_iter) * (n_iter_total - current_iter)
                    eta_str = f"{eta:.0f} сек"
                else:
                    eta_str = "?"
                
                progress_bar_manager.update_training_progress(
                    model_name, current_iter, current_loss, n_iter_total, eta_str
                )
            
            model = chunk_model
            train_time = time.time() - start_time
            
        else:
            # Стандартное обучение
            model = MLPRegressor(**params)
            model.fit(X_train_scaled, y_train_scaled)
            train_time = time.time() - start_time
            
            # Финальное обновление
            progress_bar_manager.update_training_progress(
                model_name, max_iter, 0, max_iter, "0 сек"
            )
        
        update_progress("предсказание")
        y_pred_train_scaled = model.predict(X_train_scaled)
        y_pred_val_scaled = model.predict(X_val_scaled)
        y_pred_test_scaled = model.predict(X_test_scaled)
        
        y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).ravel()
        y_pred_val = scaler_y.inverse_transform(y_pred_val_scaled.reshape(-1, 1)).ravel()
        y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()
        
        update_progress("метрики")
        metrics = {
            'model_name': model_name,
            'params': params,
            'train': calculate_all_metrics(y_train, y_pred_train),
            'val': calculate_all_metrics(y_val, y_pred_val),
            'test': calculate_all_metrics(y_test, y_pred_test),
            'n_iter': int(model.n_iter_),
            'training_time_sec': train_time,
            'n_features': int(X_train.shape[1]),
            'cpu_cores_used': N_JOBS,
            'n_samples': {
                'train': int(len(y_train)),
                'val': int(len(y_val)),
                'test': int(len(y_test))
            }
        }
        
        update_progress("кривые")
        metrics['learning_curve'] = plot_learning_curve(
            model, model_name, X_train_scaled, y_train_scaled, 
            X_val_scaled, y_val_scaled
        )
        
        if hasattr(model, 'loss_curve_'):
            metrics['loss_curve'] = [float(x) for x in model.loss_curve_]
            plot_loss_curve(model, model_name)
        
        metrics['weight_stats'] = plot_weight_histograms(model, model_name)
        
        update_progress("сохранение")
        model_path = f'models/neural/{model_name}.pkl'
        joblib.dump(model, model_path)
        
        scaler_path = f'models/neural/{model_name}_scaler.pkl'
        joblib.dump({'X': scaler_X, 'y': scaler_y}, scaler_path)
        
        metrics = convert_to_serializable(metrics)
        metrics_path = f'metrics/neural/{model_name}_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        progress_bar_manager.finish_model(model_name)
        
        return model_name, metrics, True
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        progress_bar_manager.finish_model(model_name)
        return model_name, str(e), False

# ========== ОСНОВНОЙ ЗАПУСК ==========
if __name__ == "__main__":
    print("\n" + "="*80)
    print(f"🧠 ОБУЧЕНИЕ НЕЙРОСЕТЕЙ")
    print("="*80)
    print(f"⚡ Режим: Параллельное обучение на {N_JOBS} ядрах")
    print("="*80)
    
    # Собираем активные модели
    active_models = [(name, cfg) for name, cfg in nn_params.items() if cfg.get('enabled', True)]
    
    if not active_models:
        print("❌ Нет активных моделей для обучения")
        sys.exit(0)
    
    print(f"📌 Найдено моделей: {len(active_models)}")
    
    # Создаём менеджер прогресс-баров
    progress_manager = MultiModelProgress(len(active_models))
    
    # Запускаем параллельное обучение
    results = {}
    errors = []
    
    with ThreadPoolExecutor(max_workers=min(len(active_models), N_JOBS)) as executor:
        futures = {
            executor.submit(
                train_single_model, 
                model_config, 
                model_name,
                progress_manager
            ): model_name
            for model_name, model_config in active_models
        }
        
        for future in as_completed(futures):
            model_name, result, success = future.result()
            if success:
                results[model_name] = result
            else:
                errors.append(f"{model_name}: {result}")
    
    progress_manager.close()
    
    # ========== СВОДКА ==========
    if results:
        print("\n" + "="*80)
        print("📊 СВОДКА ПО МОДЕЛЯМ")
        print("="*80)
        
        summary = {
            'models_trained': list(results.keys()),
            'total_time_sec': sum(r['training_time_sec'] for r in results.values()),
            'cpu_cores_used': N_JOBS,
            'best_by_r2': max(results.keys(), key=lambda x: results[x]['test']['r2']),
            'best_by_rmse': min(results.keys(), key=lambda x: results[x]['test']['rmse']),
            'results': {
                name: {
                    'test_r2': res['test']['r2'],
                    'test_rmse': res['test']['rmse'],
                    'test_mae': res['test']['mae'],
                    'n_iter': res['n_iter'],
                    'time_sec': res['training_time_sec']
                } for name, res in results.items()
            }
        }
        
        with open('metrics/neural/summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n🏆 Лучшая по R²:   {summary['best_by_r2']}")
        print(f"🏆 Лучшая по RMSE: {summary['best_by_rmse']}")
        print(f"⏱️  Общее время: {summary['total_time_sec']:.1f} сек")
        print(f"⚡ Использовано ядер: {summary['cpu_cores_used']}")
        print("-"*80)
        
        for name, res in summary['results'].items():
            print(f"\n{name.upper()}:")
            print(f"  R²:   {res['test_r2']:.4f}")
            print(f"  RMSE: {res['test_rmse']:.4f}")
            print(f"  MAE:  {res['test_mae']:.4f}")
            print(f"  Время: {res['time_sec']:.1f} сек")
        
        if errors:
            print("\n⚠️ Ошибки:")
            for err in errors:
                print(f"  {err}")
    
    print("\n" + "="*80)
    print("✅ ОБУЧЕНИЕ НЕЙРОСЕТЕЙ ЗАВЕРШЕНО!")
    print("="*80)