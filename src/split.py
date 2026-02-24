import pandas as pd
import numpy as np
import yaml
import json
import os

# ========== ЗАГРУЗКА ПАРАМЕТРОВ ==========
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)['split']

# ========== ЗАГРУЗКА ДАННЫХ ==========
df = pd.read_csv('data/processed/working_copy.csv')
print(f"Всего строк: {len(df)}")

# ========== ЦЕЛЕВАЯ ПЕРЕМЕННАЯ ==========
target_col = params['target_column']
if target_col not in df.columns:
    raise ValueError(f"Колонка {target_col} не найдена")

# ========== РАЗДЕЛЕНИЕ ПО МЕСЯЦАМ (ВАРИАНТ 3) ==========
if params.get('group_by_month', True):
    print("Разделение сбалансированно по месяцам")
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    for month in sorted(df['MO'].unique()):
        month_data = df[df['MO'] == month]
        n = len(month_data)
        
        train_end = int(n * params['train_ratio'])
        val_end = train_end + int(n * params['val_ratio'])
        
        train_indices.extend(month_data.iloc[:train_end].index)
        val_indices.extend(month_data.iloc[train_end:val_end].index)
        test_indices.extend(month_data.iloc[val_end:].index)
    
    train_df = df.loc[train_indices]
    val_df = df.loc[val_indices]
    test_df = df.loc[test_indices]
    
else:
    # Простое хронологическое разделение
    print("Хронологическое разделение")
    n = len(df)
    train_end = int(n * params['train_ratio'])
    val_end = train_end + int(n * params['val_ratio'])
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

# ========== РАЗДЕЛЕНИЕ НА X И Y ==========
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]

X_val = val_df.drop(columns=[target_col])
y_val = val_df[target_col]

X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

# ========== СОХРАНЕНИЕ ==========
os.makedirs('data/processed', exist_ok=True)

X_train.to_csv('data/processed/X_train.csv', index=False)
X_val.to_csv('data/processed/X_val.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)

y_train.to_csv('data/processed/y_train.csv', index=False)
y_val.to_csv('data/processed/y_val.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

print(f"\nTrain: {len(X_train)} ({len(X_train)/len(df):.1%})")
print(f"Val:   {len(X_val)} ({len(X_val)/len(df):.1%})")
print(f"Test:  {len(X_test)} ({len(X_test)/len(df):.1%})")

# ========== МЕТРИКИ РАЗДЕЛЕНИЯ ==========
metrics = {
    'total_rows': int(len(df)),
    'train_rows': int(len(X_train)),
    'val_rows': int(len(X_val)),
    'test_rows': int(len(X_test)),
    'train_ratio': float(params['train_ratio']),
    'val_ratio': float(params['val_ratio']),
    'test_ratio': float(params['test_ratio']),
    'strategy': 'monthly_balanced' if params.get('group_by_month') else 'chronological',
    'months': [int(m) for m in sorted(df['MO'].unique())]
}

os.makedirs('metrics', exist_ok=True)
with open('metrics/split_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("\n✅ Разделение завершено")