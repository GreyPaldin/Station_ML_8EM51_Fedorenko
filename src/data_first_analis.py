import pandas as pd
import numpy as np
import os

def analyze_csv(file_path):
    """Анализ одного CSV файла"""
    print(f"\n{'='*60}")
    print(f"АНАЛИЗ ФАЙЛА: {os.path.basename(file_path)}")
    print('='*60)
    
    try:
        # Загрузка данных
        df = pd.read_csv(file_path)
        
        # Основная информация
        print(f"1. Размер данных: {df.shape[0]} строк, {df.shape[1]} столбцов")
        print(f"2. Информация о типах данных:")
        print(df.dtypes.to_string())
        
        # Пропущенные значения
        null_sum = df.isnull().sum()
        null_percent = (null_sum / len(df)) * 100
        
        print(f"\n3. Пропущенные значения (NULL/NaN):")
        for col in df.columns:
            if null_sum[col] > 0:
                print(f"   - {col}: {null_sum[col]} пропусков ({null_percent[col]:.2f}%)")
        
        total_null = null_sum.sum()
        rows_with_null = df.isnull().any(axis=1).sum()
        print(f"\n   Всего пропусков в файле: {total_null}")
        print(f"   Строк с хотя бы одним пропуском: {rows_with_null} ({rows_with_null/len(df)*100:.2f}% от всех строк)")
        
        # Дубликаты - УЛУЧШЕННАЯ ВЕРСИЯ
        duplicates_mask = df.duplicated(keep=False)  # keep=False помечает ВСЕ дубликаты
        duplicates_count = df.duplicated().sum()  # только дополнительные копии
        
        print(f"\n4. Анализ дубликатов:")
        print(f"   Полных дубликатов строк: {duplicates_count}")
        
        if duplicates_count > 0:
            # Находим группы дубликатов
            duplicate_groups = df[duplicates_mask]
            
            # Группируем по самим строкам
            duplicate_indices = {}
            for idx, row in duplicate_groups.iterrows():
                # Создаем кортеж из значений строки для использования как ключ
                row_tuple = tuple(row.values)
                if row_tuple not in duplicate_indices:
                    duplicate_indices[row_tuple] = []
                duplicate_indices[row_tuple].append(idx)
            
            # Выводим только группы с дубликатами (больше 1 строки)
            duplicate_groups_filtered = {k: v for k, v in duplicate_indices.items() if len(v) > 1}
            
            print(f"\n   Группы дублирующихся строк:")
            for i, (row_values, indices) in enumerate(list(duplicate_groups_filtered.items())[:10], 1):  # первые 10 групп
                print(f"   Группа {i}:")
                print(f"     Строки: {sorted(indices)}")
                print(f"     Количество копий: {len(indices)}")
                
                # Пример значений первой строки в группе
                sample_row = df.iloc[indices[0]]
                print(f"     Пример значений (первые 5 полей):")
                for col, val in list(sample_row.items())[:5]:
                    print(f"       {col}: {val}")
                print()
            
            if len(duplicate_groups_filtered) > 10:
                print(f"   ... и еще {len(duplicate_groups_filtered) - 10} групп дубликатов")
            
            # Создаем список ВСЕХ строк с дубликатами
            all_duplicate_indices = sorted(df[duplicates_mask].index.tolist())
            print(f"\n   Всего строк с дубликатами: {len(all_duplicate_indices)}")
            print(f"   Индексы всех строк с дубликатами:")
            
            # Разбиваем на блоки для удобства чтения
            chunk_size = 20
            for i in range(0, len(all_duplicate_indices), chunk_size):
                chunk = all_duplicate_indices[i:i+chunk_size]
                indices_str = ', '.join(map(str, chunk))
                print(f"     [{indices_str}]")
        
        # Статистики для числовых колонок
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"\n5. Числовые колонки ({len(numeric_cols)}):")
            print(", ".join(numeric_cols))
            
            print(f"\n   Основные статистики числовых данных:")
            stats = df[numeric_cols].describe().T
            print(stats[['mean', 'std', 'min', '25%', '50%', '75%', 'max']].to_string())
        
        # Категориальные колонки
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            print(f"\n6. Категориальные колонки ({len(categorical_cols)}):")
            for col in categorical_cols:
                unique_vals = df[col].nunique()
                print(f"   - {col}: {unique_vals} уникальных значений")
                if unique_vals <= 10:
                    print(f"     Примеры: {df[col].unique()[:5]}")
        
        
        return df, duplicates_mask if 'duplicates_mask' in locals() else None
        
    except Exception as e:
        print(f"ОШИБКА при анализе файла: {e}")
        return None, None

def find_duplicate_details(df):
    """Найти детальную информацию о дубликатах"""
    if df is None:
        return {}
    
    # Находим все дубликаты
    duplicates_mask = df.duplicated(keep=False)
    
    if not duplicates_mask.any():
        return {}
    
    # Создаем словарь для хранения информации
    duplicate_info = {
        'total_duplicate_rows': duplicates_mask.sum(),
        'unique_duplicate_groups': 0,
        'groups': []
    }
    
    # Группируем дубликаты
    duplicate_rows = df[duplicates_mask]
    
    # Создаем хэш для каждой строки для группировки
    rows_dict = {}
    for idx, row in duplicate_rows.iterrows():
        # Используем строковое представление для хэширования
        row_str = str(tuple(row.fillna('').values))
        if row_str not in rows_dict:
            rows_dict[row_str] = []
        rows_dict[row_str].append(idx)
    
    # Сохраняем только группы с более чем 1 строкой
    for row_str, indices in rows_dict.items():
        if len(indices) > 1:
            duplicate_info['unique_duplicate_groups'] += 1
            duplicate_info['groups'].append({
                'indices': sorted(indices),
                'count': len(indices),
                'sample_data': df.iloc[indices[0]].to_dict()
            })
    
    return duplicate_info

def compare_files(df1, df2, name1, name2, dup_info1=None, dup_info2=None):
    """Сравнение двух DataFrame"""
    print(f"\n{'='*60}")
    print("СРАВНЕНИЕ ДВУХ ФАЙЛОВ")
    print('='*60)
    
    print(f"1. Размеры:")
    print(f"   {name1}: {df1.shape[0]} строк, {df1.shape[1]} столбцов")
    print(f"   {name2}: {df2.shape[0]} строк, {df2.shape[1]} столбцов")
    
    # Общие колонки
    common_cols = set(df1.columns) & set(df2.columns)
    unique_to_1 = set(df1.columns) - set(df2.columns)
    unique_to_2 = set(df2.columns) - set(df1.columns)
    
    print(f"\n2. Анализ колонок:")
    print(f"   Общих колонок: {len(common_cols)}")
    print(f"   Уникальных в {name1}: {len(unique_to_1)}")
    print(f"   Уникальных в {name2}: {len(unique_to_2)}")
    
    if unique_to_1:
        print(f"\n   Колонки только в {name1}:")
        for col in sorted(unique_to_1):
            print(f"   - {col}")
    
    if unique_to_2:
        print(f"\n   Колонки только в {name2}:")
        for col in sorted(unique_to_2):
            print(f"   - {col}")
    
    # Сравнение дубликатов
    if dup_info1 or dup_info2:
        print(f"\n3. Сравнение дубликатов:")
        
        dup1_count = dup_info1.get('total_duplicate_rows', 0) if dup_info1 else 0
        dup2_count = dup_info2.get('total_duplicate_rows', 0) if dup_info2 else 0
        
        print(f"   Дубликатов в {name1}: {dup1_count} строк")
        print(f"   Дубликатов в {name2}: {dup2_count} строк")
        
        if dup_info1 and dup_info1.get('groups'):
            print(f"\n   Группы дубликатов в {name1}: {dup_info1['unique_duplicate_groups']}")
        if dup_info2 and dup_info2.get('groups'):
            print(f"   Группы дубликатов в {name2}: {dup_info2['unique_duplicate_groups']}")

# Основная часть
if __name__ == "__main__":
    # Укажите пути к вашим файлам
    file1 = "data/raw/Summary_of_Weather.csv"
    file2 = "data/raw/Weather_Station_Locations.csv"
    
    print("НАЧАЛО АНАЛИЗА CSV ФАЙЛОВ")
    print('='*60)
    
    # Анализ каждого файла
    df1, dup_mask1 = analyze_csv(file1)
    df2, dup_mask2 = analyze_csv(file2)
    
    # Детальная информация о дубликатах
    dup_info1 = find_duplicate_details(df1) if df1 is not None else None
    dup_info2 = find_duplicate_details(df2) if df2 is not None else None
    
    # Сравнение если оба файла загружены успешно
    if df1 is not None and df2 is not None:
        compare_files(df1, df2, os.path.basename(file1), os.path.basename(file2), dup_info1, dup_info2)
    
    print(f"\n{'='*60}")
    print("АНАЛИЗ ЗАВЕРШЕН")
    print('='*60)
    