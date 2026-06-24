import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)



def main():
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))    
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test.csv')) 
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(train_df["text"], train_df["label"])
    preds = pipeline.predict(test_df['text'])

    metrics = {
        "accuracy": round(accuracy_score(test_df["label"], preds), 4),
        "precision": round(precision_score(test_df["label"], preds), 4),
        "recall": round(recall_score(test_df["label"], preds), 4),
        "f1": round(f1_score(test_df["label"], preds), 4),
        "train_size": len(train_df),
        "test_size": len(test_df),
    }
    
    model_path = os.path.join(MODELS_DIR, "sentiment_model.joblib")
    joblib.dump(pipeline, model_path)

    metrics_path = os.path.join(MODELS_DIR, "baseline_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Метрики на holdout-выборке (test.csv):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nМодель сохранена: {model_path}")
    print(f"Базовые метрики сохранены: {metrics_path}")


if __name__ == "__main__":
    main()