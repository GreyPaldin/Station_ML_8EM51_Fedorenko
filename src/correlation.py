import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def correlation_heatmap_analysis(dataset_path, save_dir='reports', target_col=None):
    """
    Построение тепловой карты корреляций и анализ взаимосвязей
    
    Args:
        dataset_path: путь к CSV файлу с данными
        save_dir: директория для сохранения графиков
        target_col: целевая переменная для анализа корреляций (опционально)
    """
    
    print(f"\n{'='*80}")
    print("🔥 ПОСТРОЕНИЕ ТЕПЛОВОЙ КАРТЫ КОРРЕЛЯЦИЙ")
    print(f"{'='*80}")
    
    # Проверяем существование файла
    if not os.path.exists(dataset_path):
        print(f"❌ Файл не найден: {dataset_path}")
        print(f"   Текущая рабочая директория: {os.getcwd()}")
        return None
    
    print(f"📁 Загружаю датасет: {os.path.basename(dataset_path)}")
    print(f"📂 Путь: {dataset_path}")
    
    try:
        # Загрузка данных
        df = pd.read_csv(dataset_path)
        print(f"✅ Загружено: {len(df):,} строк, {len(df.columns)} колонок")
        
        # Создаем директорию для сохранения
        os.makedirs(save_dir, exist_ok=True)
        
        # ==================== ПРЕДВАРИТЕЛЬНЫЙ АНАЛИЗ ====================
        
        print(f"\n📊 ПРЕДВАРИТЕЛЬНЫЙ АНАЛИЗ:")
        
        # 1. Типы данных
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"   • Числовых колонок: {len(numeric_cols)}")
        print(f"   • Категориальных колонок: {len(categorical_cols)}")
        
        if len(numeric_cols) < 2:
            print("❌ Недостаточно числовых колонок для анализа корреляций")
            return df
        
        # 2. Пропуски в числовых колонках
        print(f"\n🚨 ПРОПУСКИ В ЧИСЛОВЫХ КОЛОНКАХ:")
        for col in numeric_cols[:10]:  # первые 10
            null_count = df[col].isnull().sum()
            if null_count > 0:
                percent = (null_count / len(df)) * 100
                print(f"   • {col}: {null_count:,} пропусков ({percent:.1f}%)")
        
        # 3. Автопоиск целевой переменной
        if target_col is None:
            possible_targets = ['MaxTemp', 'MinTemp', 'MeanTemp', 'Temperature', 'temp', 
                               'PRCP', 'Precipitation', 'SNF', 'Snowfall']
            for col in possible_targets:
                if col in df.columns and col in numeric_cols:
                    target_col = col
                    print(f"\n🎯 Автоопределена целевая переменная: {target_col}")
                    break
        
        # ==================== КОРРЕЛЯЦИОННЫЙ АНАЛИЗ ====================
        
        print(f"\n📈 РАСЧЕТ КОРРЕЛЯЦИОННОЙ МАТРИЦЫ...")
        
        # Убираем пропуски для корреляции
        df_numeric = df[numeric_cols].dropna()
        
        if len(df_numeric) < len(df) * 0.5:  # если удалили больше половины
            print(f"⚠️  После удаления пропусков осталось: {len(df_numeric):,} строк")
            print(f"   Рассмотри заполнение пропусков вместо удаления")
        
        # Рассчитываем корреляционную матрицу
        corr_matrix = df_numeric.corr()
        
        print(f"✅ Рассчитана матрица {corr_matrix.shape[0]}x{corr_matrix.shape[1]}")
        
        # ==================== ТЕПЛОВАЯ КАРТА 1: ПОЛНАЯ ====================
        
        print(f"\n🎨 СОЗДАЮ ТЕПЛОВУЮ КАРТУ...")
        
        plt.figure(figsize=(16, 14))
        
        # Маска для верхнего треугольника
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Тепловая карта
        heatmap = sns.heatmap(corr_matrix,
                             mask=mask,
                             annot=True,
                             fmt='.2f',
                             cmap='RdBu_r',  # Красно-синяя палитра
                             center=0,
                             square=True,
                             linewidths=0.5,
                             cbar_kws={"shrink": 0.8, "label": "Коэффициент корреляции"},
                             annot_kws={"size": 8})
        
        plt.title(f'Тепловая карта корреляций\n{os.path.basename(dataset_path)}', 
                 fontsize=16, pad=20, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(fontsize=9)
        
        # Сохраняем
        heatmap_path = os.path.join(save_dir, 'correlation_heatmap_full.png')
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        print(f"💾 Сохранено: {heatmap_path}")
        plt.show()
        
        # ==================== ТЕПЛОВАЯ КАРТА 2: ТОЛЬКО СИЛЬНЫЕ КОРРЕЛЯЦИИ ====================
        
        if target_col and target_col in corr_matrix.columns:
            print(f"\n🔥 АНАЛИЗ КОРРЕЛЯЦИЙ С ЦЕЛЕВОЙ ПЕРЕМЕННОЙ '{target_col}':")
            
            # Корреляции с целевой переменной
            target_correlations = corr_matrix[target_col].sort_values(ascending=False)
            
            # Визуализация топ-15 корреляций
            plt.figure(figsize=(12, 8))
            
            # Топ-10 положительных и топ-5 отрицательных
            top_positive = target_correlations[1:11]  # исключаем саму целевую
            top_negative = target_correlations[-5:]
            
            top_corr = pd.concat([top_positive, top_negative])
            
            # График
            colors = ['green' if x > 0 else 'red' for x in top_corr.values]
            bars = plt.barh(range(len(top_corr)), top_corr.values, color=colors)
            
            plt.yticks(range(len(top_corr)), top_corr.index)
            plt.xlabel('Коэффициент корреляции')
            plt.title(f'Топ корреляций с {target_col}', fontsize=14, pad=15)
            plt.grid(axis='x', alpha=0.3)
            
            # Добавляем значения на бары
            for bar, value in zip(bars, top_corr.values):
                width = bar.get_width()
                plt.text(width if width > 0 else width - 0.02, 
                        bar.get_y() + bar.get_height()/2,
                        f'{value:.3f}', 
                        va='center',
                        fontweight='bold',
                        color='white' if abs(width) > 0.3 else 'black')
            
            target_heatmap_path = os.path.join(save_dir, f'correlation_with_{target_col}.png')
            plt.tight_layout()
            plt.savefig(target_heatmap_path, dpi=300, bbox_inches='tight')
            print(f"💾 Сохранено: {target_heatmap_path}")
            plt.show()
            
            # Текстовая информация
            print(f"\n📋 ТОП-10 ПОЛОЖИТЕЛЬНЫХ КОРРЕЛЯЦИЙ:")
            for i, (col, corr) in enumerate(top_positive.items(), 1):
                stars = "***" if abs(corr) > 0.7 else "**" if abs(corr) > 0.5 else "*"
                print(f"   {i:2}. {col:<25} {corr:7.3f} {stars}")
            
            print(f"\n📋 ТОП-5 ОТРИЦАТЕЛЬНЫХ КОРРЕЛЯЦИЙ:")
            for i, (col, corr) in enumerate(top_negative.items(), 1):
                stars = "***" if abs(corr) > 0.7 else "**" if abs(corr) > 0.5 else "*"
                print(f"   {i:2}. {col:<25} {corr:7.3f} {stars}")
        
        # ==================== АНАЛИЗ МУЛЬТИКОЛЛИНЕАРНОСТИ ====================
        
        print(f"\n⚠️  АНАЛИЗ МУЛЬТИКОЛЛИНЕАРНОСТИ (сильно коррелирующие пары):")
        
        high_corr_pairs = []
        threshold = 0.8  # порог для сильной корреляции
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = abs(corr_matrix.iloc[i, j])
                if corr_value > threshold:
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    high_corr_pairs.append((col1, col2, corr_matrix.iloc[i, j]))
        
        if high_corr_pairs:
            print(f"   Найдено {len(high_corr_pairs)} пар с корреляцией > {threshold}:")
            for col1, col2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:10]:
                print(f"   • {col1:<20} ↔ {col2:<20}: {corr:.3f}")
                
                # Рекомендация
                if abs(corr) > 0.9:
                    print(f"     🚨 КРИТИЧЕСКАЯ КОЛЛИНЕАРНОСТЬ! Удали одну из колонок")
                elif abs(corr) > 0.8:
                    print(f"     ⚠️  Высокая коллинеарность. Рассмотри удаление")
        else:
            print(f"   ✓ Нет сильной мультиколлинеарности")
        
        # ==================== СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ====================
        
        # Сохраняем корреляционную матрицу в CSV
        corr_csv_path = os.path.join(save_dir, 'correlation_matrix.csv')
        corr_matrix.to_csv(corr_csv_path)
        print(f"\n📄 Корреляционная матрица сохранена: {corr_csv_path}")
        
        # Создаем текстовый отчет
        report_path = os.path.join(save_dir, 'correlation_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"ОТЧЕТ О КОРРЕЛЯЦИОННОМ АНАЛИЗЕ\n")
            f.write(f"="*60 + "\n")
            f.write(f"Датасет: {os.path.basename(dataset_path)}\n")
            f.write(f"Время анализа: {pd.Timestamp.now()}\n")
            f.write(f"Всего строк: {len(df):,}\n")
            f.write(f"Числовых колонок: {len(numeric_cols)}\n")
            f.write(f"Целевая переменная: {target_col if target_col else 'не определена'}\n\n")
            
            if target_col and target_col in corr_matrix.columns:
                f.write(f"КОРРЕЛЯЦИИ С {target_col.upper()}:\n")
                f.write("-"*40 + "\n")
                for col, corr in target_correlations.items():
                    if col != target_col:
                        f.write(f"{col:<25}: {corr:7.3f}\n")
            
            if high_corr_pairs:
                f.write(f"\nМУЛЬТИКОЛЛИНЕАРНОСТЬ (корреляция > {threshold}):\n")
                f.write("-"*40 + "\n")
                for col1, col2, corr in high_corr_pairs[:20]:
                    f.write(f"{col1:<20} ↔ {col2:<20}: {corr:.3f}\n")
        
        print(f"📄 Текстовый отчет сохранен: {report_path}")
        
        return df, corr_matrix
        
    except Exception as e:
        print(f"❌ Ошибка при анализе: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# ==================== БЫСТРЫЙ ЗАПУСК ====================

def quick_correlation(dataset_path, target_col=None):
    """Быстрый запуск с указанным путем"""
    return correlation_heatmap_analysis(dataset_path, target_col=target_col)

# ==================== ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ ====================

if __name__ == "__main__":
    """
    Примеры использования:
    
    1. Прямой вызов с указанием пути:
       correlation_heatmap_analysis("data/processed/my_data.csv")
    
    2. С указанием целевой переменной:
       correlation_heatmap_analysis("data.csv", target_col="MaxTemp")
    
    3. Быстрый вызов:
       quick_correlation("data.csv")
    """
    
    # Автоматический запуск с твоим путем
    YOUR_DATASET_PATH = "data/processed/working_copy.csv"  # <-- ИЗМЕНИ НА СВОЙ ПУТЬ!
    
    if os.path.exists(YOUR_DATASET_PATH):
        print(f"🚀 Запускаю анализ для: {YOUR_DATASET_PATH}")
        df, corr_matrix = correlation_heatmap_analysis(
            dataset_path=YOUR_DATASET_PATH,
            target_col="MeanTemp"  # Укажи свою целевую переменную
        )
    else:
        print(f"⚠️  Файл не найден: {YOUR_DATASET_PATH}")
        print("Создайте файл или измените путь в переменной YOUR_DATASET_PATH")