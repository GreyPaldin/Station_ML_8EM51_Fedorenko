import pandas as pd
import os

YOUR_DATASET_PATH = "data/processed/working_copy.csv"

def analyze_snowfall_column(dataset_path):
    """Анализ колонки Snowfall с детальной информацией"""
    
    print("="*60)
    print("❄️ АНАЛИЗ КОЛОНКИ SNOWFALL")
    print("="*60)
    
    # Проверяем существование файла
    if not os.path.exists(dataset_path):
        print(f"❌ Файл не найден: {dataset_path}")
        print(f"   Текущая рабочая директория: {os.getcwd()}")
        return None
    
    # Загружаем данные
    print(f"📁 Загружаю файл: {os.path.basename(dataset_path)}")
    df = pd.read_csv(dataset_path)
    print(f"✅ Загружено: {len(df):,} строк, {len(df.columns)} колонок")
    
    # Проверяем наличие колонки Snowfall
    if 'Snowfall' not in df.columns:
        print(f"\n❌ Колонка 'Snowfall' не найдена!")
        print(f"📋 Доступные колонки:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2}. {col}")
        return df
    
    # Получаем данные колонки
    snowfall_col = df['Snowfall']
    
    print(f"\n📊 ОСНОВНАЯ ИНФОРМАЦИЯ:")
    print(f"   • Тип данных: {snowfall_col.dtype}")
    print(f"   • Уникальных значений: {snowfall_col.nunique():,}")
    print(f"   • Пропусков (NaN): {snowfall_col.isna().sum():,}")
    
    if snowfall_col.notna().sum() > 0:
        percent_null = (snowfall_col.isna().sum() / len(df)) * 100
        print(f"     ({percent_null:.1f}% от всех строк)")
    
    # Базовые статистики
    print(f"\n📈 БАЗОВАЯ СТАТИСТИКА:")
    if snowfall_col.dtype in ['float64', 'int64']:
        print(f"   • Минимум: {snowfall_col.min():.2f}")
        print(f"   • Максимум: {snowfall_col.max():.2f}")
        print(f"   • Среднее: {snowfall_col.mean():.2f}")
        print(f"   • Медиана: {snowfall_col.median():.2f}")
        print(f"   • Стандартное отклонение: {snowfall_col.std():.2f}")
    else:
        print(f"   ⚠️  Колонка не числовая, статистика недоступна")
    
    # Анализ уникальных значений
    print(f"\n🔍 АНАЛИЗ УНИКАЛЬНЫХ ЗНАЧЕНИЙ:")
    
    # Топ-20 самых частых значений
    value_counts = snowfall_col.value_counts(dropna=False)
    print(f"   Топ-20 самых частых значений:")
    for i, (value, count) in enumerate(value_counts.head(20).items(), 1):
        percent = (count / len(df)) * 100
        print(f"     {i:2}. {str(value)[:30]:<30} : {count:>8,} ({percent:5.1f}%)")
    
    # Группировка по диапазонам (если числовая)
    if snowfall_col.dtype in ['float64', 'int64']:
        print(f"\n📊 РАСПРЕДЕЛЕНИЕ ПО ДИАПАЗОНАМ:")
        
        # Создаем диапазоны
        snowfall_not_null = snowfall_col.dropna()
        if len(snowfall_not_null) > 0:
            bins = [0, 0.1, 1, 5, 10, 20, 50, 100, float('inf')]
            labels = ['0 (нет)', '0.1-1', '1-5', '5-10', '10-20', '20-50', '50-100', '>100']
            
            # Группируем
            try:
                ranges = pd.cut(snowfall_not_null, bins=bins, labels=labels, right=False)
                range_counts = ranges.value_counts().sort_index()
                
                for label, count in range_counts.items():
                    percent = (count / len(snowfall_not_null)) * 100
                    print(f"     • {label:<10} : {count:>8,} ({percent:5.1f}%)")
            except:
                print(f"     ⚠️  Не удалось создать диапазоны")
    
    # Вывод примеров
    print(f"\n👀 ПРИМЕРЫ ЗНАЧЕНИЙ:")
    print(f"   Первые 10 строк:")
    for i in range(min(10, len(df))):
        print(f"     Строка {i}: {snowfall_col.iloc[i]}")
    
    # Информация о типе данных
    print(f"\n💡 ИНФОРМАЦИЯ О ТИПЕ ДАННЫХ:")
    print(f"   pandas dtype: {snowfall_col.dtype}")
    print(f"   Python type первого значения: {type(snowfall_col.iloc[0])}")
    
    # Проверка возможности преобразования
    if snowfall_col.dtype == 'object':
        print(f"\n🔄 ПРОВЕРКА ВОЗМОЖНОСТИ ПРЕОБРАЗОВАНИЯ:")
        
        # Пробуем преобразовать в числовой тип
        numeric_converted = pd.to_numeric(snowfall_col, errors='coerce')
        successful = numeric_converted.notna().sum()
        conversion_rate = (successful / len(df)) * 100
        
        print(f"   • Можно преобразовать в float: {successful:,}/{len(df):,} ({conversion_rate:.1f}%)")
        
        if successful > 0:
            print(f"   • Min после преобразования: {numeric_converted.min():.2f}")
            print(f"   • Max после преобразования: {numeric_converted.max():.2f}")
        
        # Показать проблемные значения
        if successful < len(df):
            problematic = df[snowfall_col.notna() & numeric_converted.isna()]['Snowfall'].unique()[:10]
            print(f"   • Проблемные значения (первые 10): {list(problematic)}")
    
    # Сохранение отчета
    print(f"\n💾 СОХРАНЕНИЕ ОТЧЕТА...")
    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)
    
    report_path = os.path.join(report_dir, "snowfall_analysis.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("ОТЧЕТ ПО КОЛОНКЕ SNOWFALL\n")
        f.write("="*50 + "\n")
        f.write(f"Датасет: {os.path.basename(dataset_path)}\n")
        f.write(f"Время анализа: {pd.Timestamp.now()}\n")
        f.write(f"Всего строк: {len(df):,}\n\n")
        
        f.write(f"ТИП ДАННЫХ: {snowfall_col.dtype}\n")
        f.write(f"УНИКАЛЬНЫХ ЗНАЧЕНИЙ: {snowfall_col.nunique():,}\n")
        f.write(f"ПРОПУСКОВ: {snowfall_col.isna().sum():,}\n\n")
        
        if snowfall_col.dtype in ['float64', 'int64']:
            f.write(f"СТАТИСТИКА:\n")
            f.write(f"  Min: {snowfall_col.min():.2f}\n")
            f.write(f"  Max: {snowfall_col.max():.2f}\n")
            f.write(f"  Mean: {snowfall_col.mean():.2f}\n")
            f.write(f"  Median: {snowfall_col.median():.2f}\n\n")
        
        f.write(f"ТОП-15 ЗНАЧЕНИЙ:\n")
        for value, count in value_counts.head(15).items():
            percent = (count / len(df)) * 100
            f.write(f"  {str(value)[:30]:<30} : {count:>8,} ({percent:5.1f}%)\n")
    
    print(f"✅ Отчет сохранен: {report_path}")
    
    return df

# Быстрая версия для вывода только основной информации
def quick_snowfall_info(dataset_path):
    """Быстрый вывод информации о Snowfall"""
    
    if not os.path.exists(dataset_path):
        print(f"❌ Файл не найден: {dataset_path}")
        return
    
    df = pd.read_csv(dataset_path)
    
    if 'Snowfall' not in df.columns:
        print(f"❌ Колонка 'Snowfall' не найдена!")
        return
    
    snowfall = df['Snowfall']
    
    print("\n❄️ SNOWFALL - БЫСТРЫЙ АНАЛИЗ:")
    print("-"*40)
    print(f"Тип: {snowfall.dtype}")
    print(f"Уникальных: {snowfall.nunique():,}")
    print(f"Пропусков: {snowfall.isna().sum():,}")
    
    if snowfall.notna().sum() > 0:
        print(f"Min: {snowfall.min():.2f}")
        print(f"Max: {snowfall.max():.2f}")
        print(f"Mean: {snowfall.mean():.2f}")
    
    print(f"\nПримеры значений:")
    for i in range(min(5, len(df))):
        print(f"  Строка {i}: {snowfall.iloc[i]}")

# ==================== ЗАПУСК ====================

if __name__ == "__main__":
    # Полный анализ
    print("🚀 ЗАПУСК ПОЛНОГО АНАЛИЗА SNOWFALL")
    df = analyze_snowfall_column(YOUR_DATASET_PATH)
    
    # Быстрый анализ (раскомментировать если нужно)
    # quick_snowfall_info(YOUR_DATASET_PATH)
    
    # Дополнительно: создание гистограммы если нужно
    if df is not None and 'Snowfall' in df.columns:
        import matplotlib.pyplot as plt
        
        snowfall_col = df['Snowfall']
        if snowfall_col.dtype in ['float64', 'int64']:
            print(f"\n📊 СОЗДАНИЕ ГИСТОГРАММЫ...")
            
            plt.figure(figsize=(12, 6))
            
            # Гистограмма
            plt.subplot(1, 2, 1)
            snowfall_not_null = snowfall_col.dropna()
            plt.hist(snowfall_not_null, bins=50, alpha=0.7, color='blue', edgecolor='black')
            plt.title('Распределение Snowfall')
            plt.xlabel('Snowfall')
            plt.ylabel('Частота')
            plt.grid(True, alpha=0.3)
            
            # Box plot
            plt.subplot(1, 2, 2)
            plt.boxplot(snowfall_not_null, vert=False)
            plt.title('Box Plot Snowfall')
            plt.xlabel('Snowfall')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Сохраняем график
            plot_path = "reports/snowfall_distribution.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✅ График сохранен: {plot_path}")
            plt.show()

