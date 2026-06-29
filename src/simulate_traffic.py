import os
import time
import requests
import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(BASE_DIR, "data", "production_stream.csv")
API_URL = "http://127.0.0.1:8000/predict"


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Отправляю {len(df)} запросов в сервис, эмулируя 10 дней трафика...\n")

    sent = 0
    failed = 0

    for day, group in df.groupby("day"):
        for _, row in group.iterrows():
            payload = {"text": row["text"], "request_id": str(row["id"])}
            try:
                resp = requests.post(API_URL, json=payload, timeout=5)
                resp.raise_for_status()
                sent += 1
            except requests.RequestException as e:
                failed += 1
                print(f"  [день {day}] ошибка запроса id={row['id']}: {e}")
        print(f"День {day}: отправлено {len(group)} запросов")

    print(f"\nГотово. Успешно: {sent}, ошибок: {failed}")
    print("Теперь можно запустить src/monitor.py для анализа качества по дням.")


if __name__ == "__main__":
    main()
