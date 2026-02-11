import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json

def analyze_correlations(dataset_path, target_col, save_dir='reports'):
    """
    Анализ корреляций с целевой переменной и сохранение результатов
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Загрузка данных
    df = pd.read_csv(dataset_path)
    print(f"📊 Анализ: {os.path.basename(dataset_path)}")
    print(f"   Строк: {len(df):,}, Колонок: {len(df.columns)}")
    
    # Числовые колонки
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_col not in numeric_cols:
        print(f"❌ Целевая колонка '{target_col}' не найдена")
        return None
    
    # Удаляем строки с пропусками
    df_numeric = df[numeric_cols].dropna()
    
    # Корреляционная матрица
    corr_matrix = df_numeric.corr()
    
    # 1. ТЕПЛОВАЯ КАРТА
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='RdBu_r', center=0, square=True, 
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(f'Корреляции', fontsize=16)
    plt.tight_layout()
    
    heatmap_path = os.path.join(save_dir, 'correlation_heatmap.png')
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"💾 Тепловая карта: {heatmap_path}")
    
    # 2. КОРРЕЛЯЦИИ С ЦЕЛЕВОЙ ПЕРЕМЕННОЙ
    target_corr = corr_matrix[target_col].drop(target_col).sort_values(ascending=False)
    
    # Столбчатая диаграмма
    plt.figure(figsize=(12, 6))
    colors = ['green' if x > 0 else 'red' for x in target_corr.values]
    bars = plt.bar(range(len(target_corr)), target_corr.values, color=colors)
    plt.axhline(y=0, color='black', linewidth=0.5)
    plt.xticks(range(len(target_corr)), target_corr.index, rotation=45, ha='right')
    plt.ylabel('Коэффициент корреляции')
    plt.title(f'Корреляция с {target_col}')
    plt.grid(axis='y', alpha=0.3)
    
    # Добавляем значения
    for bar, val in zip(bars, target_corr.values):
        plt.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + (0.02 if val > 0 else -0.08),
                f'{val:.2f}', ha='center', va='bottom' if val > 0 else 'top',
                fontsize=8)
    
    plt.tight_layout()
    bar_path = os.path.join(save_dir, f'correlation_with_{target_col}.png')
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"💾 Диаграмма: {bar_path}")
    
    # 3. СОХРАНЕНИЕ МЕТРИК
    metrics = {
        'dataset': os.path.basename(dataset_path),
        'target_column': target_col,
        'total_rows': len(df),
        'numeric_columns': len(numeric_cols),
        'rows_used_for_correlation': len(df_numeric),
        'correlations': target_corr.to_dict(),
        'top_positive': target_corr.head(5).to_dict(),
        'top_negative': target_corr.tail(5).to_dict(),
        'multicollinearity': {}
    }
    
    # Анализ мультиколлинеарности
    threshold = 0.8
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                high_corr.append({
                    'col1': col1, 
                    'col2': col2, 
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    if high_corr:
        metrics['multicollinearity'] = {
            'threshold': threshold,
            'pairs': high_corr[:10]
        }
    
    # Сохраняем JSON
    metrics_path = os.path.join(save_dir, 'correlation_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"💾 Метрики: {metrics_path}")
    
    # 4. КРАТКИЙ ВЫВОД
    print(f"\n📌 ТОП ПОЛОЖИТЕЛЬНЫХ КОРРЕЛЯЦИЙ С {target_col}:")
    for col, val in target_corr.head(33).items():
        print(f"   {col}: {val:.3f}")
    
    print(f"\n📌 ТОП ОТРИЦАТЕЛЬНЫХ КОРРЕЛЯЦИЙ С {target_col}:")
    for col, val in target_corr.tail(33).items():
        print(f"   {col}: {val:.3f}")
    
    if high_corr:
        print(f"\n⚠️ МУЛЬТИКОЛЛИНЕАРНОСТЬ (> {threshold}): {len(high_corr)} пар")
    
    return df, corr_matrix, metrics

# ==================== ЗАПУСК ====================
if __name__ == "__main__":
    # Только измени это:
    DATASET_PATH = "data/processed/working_copy.csv"
    TARGET_COLUMN = "MeanTemp"  # или MaxTemp, MinTemp, Precipitation
    
    if os.path.exists(DATASET_PATH):
        analyze_correlations(DATASET_PATH, TARGET_COLUMN)
    else:
        print(f"❌ Файл не найден: {DATASET_PATH}")