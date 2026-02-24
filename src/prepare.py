import pandas as pd
import json
import shutil
import os

# ==================== КОНФИГУРАЦИЯ ====================
SOURCE_DATASET = "data/raw/Summary_of_Weather.csv"
STATIONS_DATASET = "data/raw/Weather_Station_Locations.csv"
WORKING_COPY_PATH = "data/processed/working_copy.csv"
COLUMNS_TO_DELETE = ["WindGustSpd","DR","SPD",
"MAX", "MIN", "MEA","SND","FT","FB","FTI","ITH","PGT",
"TSHDSBRSGF","SD3","RHX","RHN","RVG","WTE","PoorWeather",
"LAT","LON","MinTemp","NAME","STATE/COUNTRY ID","Date","PRCP","SNF","DA","YR"] #"MinTemp","MaxTemp"
WEATHER_STATION_ID = "STA"
STATIONS_STATION_ID = "WBAN"
PROBLEM_VALUE = 9999
PROBLEM_COLUMN = "ELEV"
STATION_ID_IN_STATIONS = "WBAN"
# ======================================================

def create_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return pd.read_csv(dst), dst

def merge_with_stations(weather_df, stations_df, weather_id_col, stations_id_col):
    stations_df_renamed = stations_df.rename(columns={stations_id_col: weather_id_col})
    return pd.merge(weather_df, stations_df_renamed, on=weather_id_col, how='left', suffixes=('', '_station'))

def find_problem_stations(stations_df, problem_col, problem_val, station_id_col):
    if problem_col not in stations_df.columns:
        print(f"Колонка {problem_col} не найдена в stations_df")
        print(f"Доступные колонки: {list(stations_df.columns)}")
        return []
    problem_stations = stations_df[stations_df[problem_col] == problem_val]
    return problem_stations[station_id_col].tolist()

def clean_problem_stations(stations_df, problem_col, problem_val):
    """Удалить проблемные станции из stations_df перед объединением"""
    if problem_col not in stations_df.columns:
        return stations_df
    
    initial_count = len(stations_df)
    stations_clean = stations_df[stations_df[problem_col] != problem_val].copy()
    removed = initial_count - len(stations_clean)
    
    if removed > 0:
        print(f"Удалено из stations_df: {removed} станций с {problem_col}={problem_val}")
    
    return stations_clean

def delete_columns(df, columns):
    existing = [c for c in columns if c in df.columns]
    return df.drop(columns=existing) if existing else df

def delete_duplicates(df):
    initial = len(df)
    df_clean = df.drop_duplicates()
    removed = initial - len(df_clean)
    if removed > 0:
        print(f"Удалено дубликатов: {removed}")
    return df_clean

def convert_snowfall_to_float_simple(df):
    """
    Простое преобразование Snowfall в float.
    Не-цифры становятся NaN.
    """
    if 'Snowfall' in df.columns:
        print(f"🔄 Snowfall: {df['Snowfall'].dtype} -> float")
        df['Snowfall'] = pd.to_numeric(df['Snowfall'], errors='coerce')
        nulls_added = df['Snowfall'].isna().sum() - df['Snowfall'].isna().sum()
        if nulls_added > 0:
            print(f"   Добавлено NaN: {nulls_added} строк")
    return df

def delete_null_rows(df):
    """Удаляет строки с Null и показывает статистику по колонкам"""
    initial_rows = len(df)
    
    # Считаем Null по колонкам
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0].sort_values(ascending=False)
    
    # Показываем статистику
    if len(null_counts) > 0:
        print("Null по колонкам:")
        for col, count in null_counts.items():
            percent = (count / initial_rows) * 100
            print(f"  {col}: {count} строк ({percent:.1f}%)")
    else:
        print("Null колонок нет")
    
    # Удаляем строки с любым Null
    df_cleaned = df.dropna()
    removed = initial_rows - len(df_cleaned)
    
    if removed > 0:
        print(f"Удалено строк с Null: {removed} ({removed/initial_rows*100:.1f}%)")
        print(f"Осталось: {len(df_cleaned)} строк")
    else:
        print("Null строк нет")
    
    return df_cleaned

def print_analysis(df, label=""):
    """Вывод аналитики в консоль"""
    if label:
        print(f"\n{label}")
    print(f"Строк: {len(df):,}")
    print(f"Колонок: {len(df.columns)}")
    print(f"Дупликатов: {df.duplicated().sum():,}")
    

def delite_T():
    df['Precip'] = df['Precip'].replace('T', '0')

# ==================== ИСПОЛНЕНИЕ ====================
if __name__ == "__main__":
    print("="*50)
    print("НАЧАЛО ОБРАБОТКИ")
    print("="*50)
    
    # 1. Копия основного датасета
    df, copy_path = create_copy(SOURCE_DATASET, WORKING_COPY_PATH)
    print_analysis(df, "1. Копия основного датасета:")
    
    # 2. Обработка станций
    if os.path.exists(STATIONS_DATASET):
        stations_df = pd.read_csv(STATIONS_DATASET)
        print(f"\n2. Данные станций загружены: {stations_df.shape}")
        
        # 3. НАЙТИ проблемные станции
        problem_ids = find_problem_stations(stations_df, PROBLEM_COLUMN, PROBLEM_VALUE, STATION_ID_IN_STATIONS)
        print(f"   Проблемных станций: {len(problem_ids)}")
        
        # 4. УДАЛИТЬ проблемные станции ИЗ stations_df
        stations_df_clean = clean_problem_stations(stations_df, PROBLEM_COLUMN, PROBLEM_VALUE)
        print(f"   Станции после очистки: {stations_df_clean.shape}")
        
        # 5. Объединить с ОЧИЩЕННЫМИ станциями
        df = merge_with_stations(df, stations_df_clean, WEATHER_STATION_ID, STATIONS_STATION_ID)
        print_analysis(df, "\n5. После объединения с очищенными станциями:")
    
    # 6. Удалить колонки
    if COLUMNS_TO_DELETE:
        initial_cols = len(df.columns)
        df = delete_columns(df, COLUMNS_TO_DELETE)
        print(f"\n6. Удалено колонок: {initial_cols - len(df.columns)}")
        print(f"   Осталось колонок: {len(df.columns)}")
    
    # 7. Дубликаты
    df = delete_duplicates(df)
    print_analysis(df, "\n7. После удаления дубликатов:")
   
    # 7.5. Преобразовать Snowfall в float
    print("\n7.5. Преобразование Snowfall в float:")
    df = convert_snowfall_to_float_simple(df)
    
    # 8. Null строки
    print("\n8. Удаление Null строк:")
    df = delete_null_rows(df)
    
    # 8.1 Удаление T в Precip
    delite_T()
    
    # 9. Сохранить
    df.to_csv(copy_path, index=False)
    print(f"\n9. Сохранено: {os.path.basename(copy_path)}")
    print(f"   Финальный размер: {df.shape}")
    
    # 9.5. Сохраняем метрики для DVC
    metrics = {
        'original_rows': int(pd.read_csv(SOURCE_DATASET).shape[0]),
        'final_rows': len(df),
        'rows_removed': int(pd.read_csv(SOURCE_DATASET).shape[0] - len(df)),
        'original_columns': int(pd.read_csv(SOURCE_DATASET).shape[1]),
        'final_columns': len(df.columns),
        'columns_removed': len(COLUMNS_TO_DELETE),
        'stations_removed': len(find_problem_stations(pd.read_csv(STATIONS_DATASET), PROBLEM_COLUMN, PROBLEM_VALUE, STATION_ID_IN_STATIONS)) if os.path.exists(STATIONS_DATASET) else 0,
        'duplicates_removed': int(pd.read_csv(SOURCE_DATASET).shape[0] - len(df))  # приблизительно
    }
    
    os.makedirs('metrics', exist_ok=True)
    with open('metrics/prepare_metrics.json', 'w') as f:
        import json
        json.dump(metrics, f, indent=2)
    
    # 10. Итоговый анализ
    print("\n" + "="*50)
    print("ИТОГОВЫЙ АНАЛИЗ")
    print("="*50)
    print(f"Строк: {len(df):,}")
    print(f"Колонок: {len(df.columns)}")
    print(f"Колонки: {list(df.columns)}")