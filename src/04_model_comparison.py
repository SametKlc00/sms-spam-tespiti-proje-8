import os
import re
import joblib
import warnings
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


warnings.filterwarnings("ignore")


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
    SMS mesajı için temel metin temizleme işlemleri.
    """

    text = str(text)
    text = text.lower()

    # URL temizleme
    text = re.sub(r"http\S+|www\S+", " ", text)

    # E-posta temizleme
    text = re.sub(r"\S+@\S+", " ", text)

    # Sayı temizleme
    text = re.sub(r"\d+", " ", text)

    # Noktalama ve özel karakter temizleme
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Fazla boşluk temizleme
    text = re.sub(r"\s+", " ", text).strip()

    return text


def prepare_data(df):
    """
    Eksik veri kontrolü, label encoding ve train-test split işlemlerini yapar.
    """

    df = df.dropna(subset=["label"])
    df["message"] = df["message"].fillna("")
    df = df[df["message"].str.strip() != ""]

    df["label_encoded"] = df["label"].map({
        "ham": 0,
        "spam": 1
    })

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


def build_pipelines():
    """
    k-NN, SVM ve ANN modelleri için ayrı Pipeline yapıları oluşturur.
    """

    svm_base = LinearSVC(
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    svm_model = CalibratedClassifierCV(
        estimator=svm_base,
        cv=3
    )

    svm_pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                preprocessor=clean_text,
                stop_words="english"
            )
        ),
        (
            "feature_selection",
            SelectKBest(
                score_func=chi2
            )
        ),
        (
            "model",
            svm_model
        )
    ])

    knn_pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                preprocessor=clean_text,
                stop_words="english"
            )
        ),
        (
            "feature_selection",
            SelectKBest(
                score_func=chi2
            )
        ),
        (
            "model",
            KNeighborsClassifier(
                metric="cosine",
                algorithm="brute"
            )
        )
    ])

    ann_pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                preprocessor=clean_text,
                stop_words="english"
            )
        ),
        (
            "svd",
            TruncatedSVD(
                random_state=RANDOM_STATE
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            MLPClassifier(
                random_state=RANDOM_STATE,
                max_iter=300,
                early_stopping=True
            )
        )
    ])

    pipelines = {
        "SVM": svm_pipeline,
        "kNN": knn_pipeline,
        "ANN": ann_pipeline
    }

    return pipelines


def build_param_grids():
    """
    Her model için GridSearchCV hiperparametre arama alanlarını tanımlar.
    Aralıklar çok büyük tutulmadı; böylece eğitim süresi kontrol altında kalır.
    """

    svm_param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__max_features": [3000, 5000],
        "feature_selection__k": [1500, 3000],
        "model__estimator__C": [0.5, 1.0, 2.0]
    }

    knn_param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__max_features": [3000, 5000],
        "feature_selection__k": [1500, 3000],
        "model__n_neighbors": [3, 5, 7]
    }

    ann_param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__max_features": [3000, 5000],
        "svd__n_components": [100, 200],
        "model__hidden_layer_sizes": [(50,), (100,)]
    }

    param_grids = {
        "SVM": svm_param_grid,
        "kNN": knn_param_grid,
        "ANN": ann_param_grid
    }

    return param_grids


def get_prediction_scores(model, X_test):
    """
    ROC-AUC için modelden olasılık veya skor üretir.
    """

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]

    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)

    raise ValueError("Model ROC-AUC için skor üretemiyor.")


def evaluate_model(model, X_test, y_test):
    """
    Eğitilmiş modeli test verisi üzerinde değerlendirir.
    """

    y_pred = model.predict(X_test)
    y_score = get_prediction_scores(model, X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_score)
    }

    return y_pred, y_score, metrics


def save_model_comparison_plot(results_df):
    """
    Model karşılaştırma grafiğini kaydeder.
    """

    metric_columns = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]

    plot_df = results_df.set_index("Model")[metric_columns]

    ax = plot_df.plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.title("Model Karşılaştırma Sonuçları")
    plt.xlabel("Model")
    plt.ylabel("Skor")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.tight_layout()

    plot_path = os.path.join(OUTPUTS_DIR, "model_comparison_bar.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Model karşılaştırma grafiği kaydedildi: {plot_path}")


def save_confusion_matrices(confusion_results):
    """
    Her model için confusion matrix görseli oluşturur.
    """

    for model_name, cm in confusion_results.items():
        plt.figure(figsize=(6, 5))

        plt.imshow(cm)
        plt.title(f"{model_name} Confusion Matrix")
        plt.xlabel("Tahmin Edilen Sınıf")
        plt.ylabel("Gerçek Sınıf")

        plt.xticks([0, 1], ["ham", "spam"])
        plt.yticks([0, 1], ["ham", "spam"])

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center"
                )

        plt.colorbar()
        plt.tight_layout()

        file_name = f"{model_name.lower()}_comparison_confusion_matrix.png"
        file_path = os.path.join(OUTPUTS_DIR, file_name)

        plt.savefig(file_path, dpi=300)
        plt.close()

        print(f"{model_name} confusion matrix kaydedildi: {file_path}")


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

    pipelines = build_pipelines()
    param_grids = build_param_grids()

    all_results = []
    confusion_results = {}
    best_models = {}

    for model_name in pipelines:
        print("\n" + "=" * 60)
        print(f"{model_name} modeli için GridSearchCV başlıyor...")
        print("=" * 60)

        grid_search = GridSearchCV(
            estimator=pipelines[model_name],
            param_grid=param_grids[model_name],
            scoring="f1",
            cv=3,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_models[model_name] = best_model

        print(f"\n{model_name} en iyi parametreler:")
        print(grid_search.best_params_)

        y_pred, y_score, metrics = evaluate_model(
            best_model,
            X_test,
            y_test
        )

        cm = confusion_matrix(y_test, y_pred)
        confusion_results[model_name] = cm

        result_row = {
            "Model": model_name,
            "Best_Params": str(grid_search.best_params_),
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1": metrics["F1"],
            "ROC_AUC": metrics["ROC_AUC"]
        }

        all_results.append(result_row)

        report = classification_report(
            y_test,
            y_pred,
            target_names=["ham", "spam"],
            output_dict=True
        )

        report_df = pd.DataFrame(report).transpose()
        report_path = os.path.join(
            OUTPUTS_DIR,
            f"{model_name.lower()}_classification_report.csv"
        )
        report_df.to_csv(report_path, encoding="utf-8")

        model_path = os.path.join(
            MODELS_DIR,
            f"{model_name.lower()}_best_pipeline.pkl"
        )
        joblib.dump(best_model, model_path)

        print(f"{model_name} classification report kaydedildi: {report_path}")
        print(f"{model_name} en iyi model kaydedildi: {model_path}")

        print(f"\n{model_name} Test Sonuçları:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

    results_df = pd.DataFrame(all_results)

    comparison_path = os.path.join(OUTPUTS_DIR, "model_comparison_results.csv")
    results_df.to_csv(comparison_path, index=False, encoding="utf-8")

    print("\n" + "=" * 60)
    print("Model karşılaştırma sonuçları:")
    print("=" * 60)
    print(results_df)

    save_model_comparison_plot(results_df)
    save_confusion_matrices(confusion_results)

    # En iyi modeli F1 skoruna göre seç
    best_row = results_df.sort_values(by="F1", ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    best_model = best_models[best_model_name]

    final_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_model, final_model_path)

    best_summary_path = os.path.join(OUTPUTS_DIR, "best_model_summary.txt")

    with open(best_summary_path, "w", encoding="utf-8") as file:
        file.write("En İyi Model Özeti\n")
        file.write("=" * 40 + "\n\n")
        file.write(f"En iyi model: {best_model_name}\n")
        file.write(f"Accuracy: {best_row['Accuracy']:.4f}\n")
        file.write(f"Precision: {best_row['Precision']:.4f}\n")
        file.write(f"Recall: {best_row['Recall']:.4f}\n")
        file.write(f"F1: {best_row['F1']:.4f}\n")
        file.write(f"ROC_AUC: {best_row['ROC_AUC']:.4f}\n")
        file.write("\nEn iyi parametreler:\n")
        file.write(str(best_row["Best_Params"]))

    print("\nEn iyi model seçildi:")
    print(f"Model: {best_model_name}")
    print(f"F1 skoru: {best_row['F1']:.4f}")

    print(f"\nFinal model kaydedildi: {final_model_path}")
    print(f"En iyi model özeti kaydedildi: {best_summary_path}")

    print("\nİşlem tamamlandı.")


if __name__ == "__main__":
    main()