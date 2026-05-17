import os
import re
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)


DATA_DIR = "data"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_sms_dataset():
    """
    SMS Spam veri setini Kaggle spam.csv veya UCI SMSSpamCollection.txt formatından okur.
    """

    spam_csv_path = os.path.join(DATA_DIR, "spam.csv")
    uci_txt_path = os.path.join(DATA_DIR, "SMSSpamCollection.txt")

    if os.path.exists(spam_csv_path):
        print("spam.csv dosyası bulundu. Kaggle formatı okunuyor...")

        df = pd.read_csv(spam_csv_path, encoding="latin-1")

        if "v1" in df.columns and "v2" in df.columns:
            df = df[["v1", "v2"]]
            df.columns = ["label", "message"]
        else:
            raise ValueError("spam.csv içinde v1 ve v2 sütunları bulunamadı.")

    elif os.path.exists(uci_txt_path):
        print("SMSSpamCollection.txt dosyası bulundu. UCI formatı okunuyor...")

        df = pd.read_csv(
            uci_txt_path,
            sep="\t",
            header=None,
            names=["label", "message"],
            encoding="utf-8"
        )

    else:
        raise FileNotFoundError(
            "data klasöründe spam.csv veya SMSSpamCollection.txt bulunamadı."
        )

    return df


def clean_text(text):
    """
    SMS metnini temel seviyede temizler.
    Bu fonksiyon TF-IDF içinde preprocessor olarak kullanılacaktır.
    """

    text = str(text)
    text = text.lower()

    # URL temizleme
    text = re.sub(r"http\S+|www\S+", " ", text)

    # E-posta temizleme
    text = re.sub(r"\S+@\S+", " ", text)

    # Sayı temizleme
    text = re.sub(r"\d+", " ", text)

    # Noktalama ve özel karakterleri temizleme
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Fazla boşlukları temizleme
    text = re.sub(r"\s+", " ", text).strip()

    return text


def prepare_data(df):
    """
    Eksik veri kontrolü, label encoding ve train-test split işlemlerini yapar.
    """

    # Eksik label varsa çıkar
    df = df.dropna(subset=["label"])

    # Eksik mesajları boş metin ile doldur
    df["message"] = df["message"].fillna("")

    # Boş mesajları çıkar
    df = df[df["message"].str.strip() != ""]

    # Label encoding: ham = 0, spam = 1
    df["label_encoded"] = df["label"].map({
        "ham": 0,
        "spam": 1
    })

    # Bilinmeyen etiket varsa çıkar
    df = df.dropna(subset=["label_encoded"])
    df["label_encoded"] = df["label_encoded"].astype(int)

    X = df["message"]
    y = df["label_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, df


def build_svm_pipeline():
    """
    TF-IDF + Feature Selection + SVM modelinden oluşan Pipeline kurar.
    CalibratedClassifierCV kullanıldığı için predict_proba alınabilir.
    Dashboard tarafında spam olasılığı göstermek için bu önemlidir.
    """

    svm_model = LinearSVC(
        C=1.0,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    calibrated_svm = CalibratedClassifierCV(
        estimator=svm_model,
        cv=3
    )

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                preprocessor=clean_text,
                stop_words="english",
                ngram_range=(1, 2),
                max_features=5000
            )
        ),
        (
            "feature_selection",
            SelectKBest(
                score_func=chi2,
                k=3000
            )
        ),
        (
            "model",
            calibrated_svm
        )
    ])

    return pipeline


def evaluate_model(model, X_test, y_test):
    """
    Model performans metriklerini hesaplar.
    """

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_proba)
    }

    return y_pred, y_proba, metrics


def save_metrics(metrics):
    """
    Metrikleri txt dosyasına kaydeder.
    """

    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    metrics_path = os.path.join(OUTPUTS_DIR, "svm_metrics.txt")

    with open(metrics_path, "w", encoding="utf-8") as file:
        file.write("SVM Model Performans Metrikleri\n")
        file.write("=" * 40 + "\n\n")

        for metric_name, metric_value in metrics.items():
            file.write(f"{metric_name}: {metric_value:.4f}\n")

    print(f"Metrikler kaydedildi: {metrics_path}")


def save_classification_report(y_test, y_pred):
    """
    Classification report çıktısını CSV olarak kaydeder.
    """

    report = classification_report(
        y_test,
        y_pred,
        target_names=["ham", "spam"],
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    report_path = os.path.join(OUTPUTS_DIR, "svm_classification_report.csv")
    report_df.to_csv(report_path, encoding="utf-8")

    print(f"Classification report kaydedildi: {report_path}")


def plot_confusion_matrix(y_test, y_pred):
    """
    Confusion matrix heatmap grafiğini oluşturur ve kaydeder.
    """

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["ham", "spam"],
        yticklabels=["ham", "spam"]
    )

    plt.title("SVM Confusion Matrix")
    plt.xlabel("Tahmin Edilen Sınıf")
    plt.ylabel("Gerçek Sınıf")
    plt.tight_layout()

    cm_path = os.path.join(OUTPUTS_DIR, "svm_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    print(f"Confusion matrix kaydedildi: {cm_path}")


def plot_roc_curve(y_test, y_proba, roc_auc):
    """
    ROC eğrisini oluşturur ve kaydeder.
    """

    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"SVM ROC Curve AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

    plt.title("SVM ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()

    roc_path = os.path.join(OUTPUTS_DIR, "svm_roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()

    print(f"ROC eğrisi kaydedildi: {roc_path}")


def save_model(model):
    """
    Eğitilmiş Pipeline modelini models klasörüne kaydeder.
    """

    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, "svm_pipeline.pkl")
    joblib.dump(model, model_path)

    print(f"Model kaydedildi: {model_path}")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("Veri seti yükleniyor...")
    df = load_sms_dataset()

    print("\nVeri hazırlanıyor...")
    X_train, X_test, y_train, y_test, df = prepare_data(df)

    print("\nVeri boyutları:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    print("\nPipeline kuruluyor...")
    svm_pipeline = build_svm_pipeline()

    print("\nSVM modeli eğitiliyor...")
    svm_pipeline.fit(X_train, y_train)

    print("\nModel değerlendiriliyor...")
    y_pred, y_proba, metrics = evaluate_model(
        svm_pipeline,
        X_test,
        y_test
    )

    print("\nSVM Model Performans Metrikleri:")
    print("-" * 40)

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    save_metrics(metrics)
    save_classification_report(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba, metrics["ROC_AUC"])
    save_model(svm_pipeline)

    print("\nİşlem tamamlandı.")
    print("SVM Pipeline başarıyla eğitildi ve çıktılar kaydedildi.")


if __name__ == "__main__":
    main()