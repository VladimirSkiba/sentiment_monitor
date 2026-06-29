
import os
import sqlite3
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DB_PATH = os.path.join(BASE_DIR, "logs", "predictions.db")
STREAM_PATH = os.path.join(BASE_DIR, "data", "production_stream.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

ACCURACY_THRESHOLD = 0.80
CONFIDENCE_THRESHOLD = 0.75


def load_joined_data() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    logs_df = pd.read_sql_query(
        "SELECT request_id, predicted_label, confidence, latency_ms, created_at FROM predictions",
        conn,
    )
    conn.close()

    logs_df["request_id"] = logs_df["request_id"].astype(int)
    stream_df = pd.read_csv(STREAM_PATH)[["id", "day", "true_label"]]
    stream_df = stream_df.rename(columns={"id": "request_id"})

    joined = logs_df.merge(stream_df, on="request_id", how="inner")
    return joined


def compute_daily_metrics(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for day, group in joined.groupby("day"):
        acc = accuracy_score(group["true_label"], group["predicted_label"])
        f1 = f1_score(group["true_label"], group["predicted_label"])
        avg_conf = group["confidence"].mean()
        avg_latency = group["latency_ms"].mean()
        rows.append({
            "day": int(day),
            "n_samples": len(group),
            "accuracy": round(acc, 4),
            "f1": round(f1, 4),
            "avg_confidence": round(avg_conf, 4),
            "avg_latency_ms": round(avg_latency, 4),
        })
    return pd.DataFrame(rows).sort_values("day")


def detect_alerts(daily: pd.DataFrame) -> list[str]:
    alerts = []
    for _, row in daily.iterrows():
        if row["accuracy"] < ACCURACY_THRESHOLD:
            alerts.append(
                f"[ALERT] День {int(row['day'])}: accuracy={row['accuracy']} "
                f"< порога {ACCURACY_THRESHOLD}. Возможен дрифт данных, нужна проверка / переобучение."
            )
        if row["avg_confidence"] < CONFIDENCE_THRESHOLD:
            alerts.append(
                f"[ALERT] День {int(row['day'])}: avg_confidence={row['avg_confidence']} "
                f"< порога {CONFIDENCE_THRESHOLD}. Модель чаще 'не уверена' — возможны новые паттерны в данных."
            )
    return alerts


def plot_metrics(daily: pd.DataFrame, out_path: str):
    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(daily["day"], daily["accuracy"], marker="o", label="Accuracy", color="tab:blue")
    ax1.plot(daily["day"], daily["avg_confidence"], marker="s", label="Avg confidence", color="tab:orange")
    ax1.axhline(ACCURACY_THRESHOLD, color="tab:blue", linestyle="--", alpha=0.5, label="Accuracy threshold")
    ax1.axhline(CONFIDENCE_THRESHOLD, color="tab:orange", linestyle="--", alpha=0.5, label="Confidence threshold")
    ax1.axvline(5.5, color="gray", linestyle=":", alpha=0.7)
    ax1.text(5.6, 0.4, "начало дрифта\n(день 6+)", fontsize=9, color="gray")

    ax1.set_xlabel("День продакшена")
    ax1.set_ylabel("Значение метрики")
    ax1.set_title("Мониторинг качества модели по дням")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="lower left")
    ax1.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"График сохранён: {out_path}")


def main():
    joined = load_joined_data()
    daily = compute_daily_metrics(joined)

    print("Метрики качества по дням:\n")
    print(daily.to_string(index=False))

    alerts = detect_alerts(daily)
    print("\n--- Алерты ---")
    if alerts:
        for a in alerts:
            print(a)
    else:
        print("Алертов нет, качество в норме.")

    # сохраняем отчёт
    report = {
        "thresholds": {
            "accuracy": ACCURACY_THRESHOLD,
            "avg_confidence": CONFIDENCE_THRESHOLD,
        },
        "daily_metrics": daily.to_dict(orient="records"),
        "alerts": alerts,
    }
    report_path = os.path.join(REPORTS_DIR, "monitoring_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nJSON-отчёт сохранён: {report_path}")

    plot_path = os.path.join(REPORTS_DIR, "quality_over_time.png")
    plot_metrics(daily, plot_path)


if __name__ == "__main__":
    main()
