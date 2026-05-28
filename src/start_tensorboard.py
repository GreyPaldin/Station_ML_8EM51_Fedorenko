import os
import subprocess
import json

# Загружаем последние логи
with open('metrics/neural/summary_pytorch.json', 'r') as f:
    summary = json.load(f)

print("TensorBoard доступен по адресам:")
for log_dir in summary['tensorboard_dirs']:
    print(f"\n  tensorboard --logdir={log_dir}")
    print(f"  Затем открой в браузере: http://localhost:6006")