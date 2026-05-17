import os
import re
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from pathlib import Path


# =========================================================
# Sayfa ayarları
# =========================================================

st.set_page_config(
    page_title="SMS Spam Tespiti Dashboard",
    page_icon="📩",
    layout="wide"
)


# =========================================================
# Proje yolları
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"


# =========================================================
# Pickle model yükleme için temizleme fonksiyonu
# Eğitim dosyalarında da aynı clean_text fonksiyonu kullanılmıştı.
# =========================================================

def clean_text(text):
    """
    SMS metnini temel seviyede temizler.
    Model pickle yüklenirken de bu fonksiyon gerekli olabilir.
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


# Joblib pickle, clean_text fonksiyonunu __main__ altında ararsa hata vermesin diye:
sys.modules["__main__"].clean_text = clean_text


# =========================================================
# Yardımcı fonksiyonlar
# =========================================================

@st.cache_resource
def load_model(model_path):
    """
    Eğitilmiş modeli yükler.
    """
    return joblib.load(model_path)


@st.cache_data
def load_comparison_results():
    """
    Model karşılaştırma sonuçlarını yükler.
    """
    comparison_path = OUTPUTS_DIR / "model_comparison_results.csv"

    if comparison_path.exists():
        return pd.read_csv(comparison_path)

    return None


def get_available_models():
    """
    models klasöründeki mevcut modelleri dashboard'a ekler.
    """

    model_files = {
        "En İyi Model": MODELS_DIR / "best_model.pkl",
        "SVM": MODELS_DIR / "svm_best_pipeline.pkl",
        "k-NN": MODELS_DIR / "knn_best_pipeline.pkl",
        "ANN": MODELS_DIR / "ann_best_pipeline.pkl",
        "SVM İlk Pipeline": MODELS_DIR / "svm_pipeline.pkl",
    }

    available_models = {}

    for name, path in model_files.items():
        if path.exists():
            available_models[name] = path

    return available_models


def predict_spam_probability(model, message):
    """
    Modelin spam olasılığını döndürür.
    predict_proba varsa direkt kullanır.
    Yoksa decision_function skorunu sigmoid ile yaklaşık olasılığa çevirir.
    """

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba([message])[0][1]
        return probability

    if hasattr(model, "decision_function"):
        score = model.decision_function([message])[0]
        probability = 1 / (1 + np.exp(-score))
        return probability

    prediction = model.predict([message])[0]
    return float(prediction)


def show_metric_cards(results_df):
    """
    En iyi modeli ve metriklerini kart olarak gösterir.
    """

    if results_df is None or results_df.empty:
        st.warning("Model karşılaştırma sonuçları bulunamadı.")
        return

    best_row = results_df.sort_values(by="F1", ascending=False).iloc[0]

    st.subheader("🏆 En İyi Model Özeti")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Model", best_row["Model"])
    col2.metric("Accuracy", f"{best_row['Accuracy']:.4f}")
    col3.metric("Precision", f"{best_row['Precision']:.4f}")
    col4.metric("Recall", f"{best_row['Recall']:.4f}")
    col5.metric("F1", f"{best_row['F1']:.4f}")

    st.metric("ROC-AUC", f"{best_row['ROC_AUC']:.4f}")


def show_image_if_exists(title, image_path):
    """
    Dosya varsa görseli gösterir.
    """

    if image_path.exists():
        st.subheader(title)
        st.image(str(image_path), use_container_width=True)


# =========================================================
# Sidebar
# =========================================================

st.sidebar.title("📌 Menü")

page = st.sidebar.radio(
    "Sayfa seçiniz:",
    [
        "Ana Sayfa",
        "SMS Tahmini",
        "Model Karşılaştırma",
        "Grafikler",
        "Proje Bilgisi"
    ]
)

available_models = get_available_models()

if not available_models:
    st.error(
        "models klasöründe model dosyası bulunamadı. "
        "Önce src/04_model_comparison.py dosyasını çalıştırmalısın."
    )
    st.stop()


# =========================================================
# Ana Sayfa
# =========================================================

if page == "Ana Sayfa":
    st.title("📩 SMS Spam Tespiti Dashboard")

    st.markdown(
        """
        Bu dashboard, girilen SMS mesajının **spam** veya **normal/ham** olup olmadığını tahmin etmek için hazırlanmıştır.

        Projede SMS Spam Collection veri seti kullanılmıştır. Metinler TF-IDF yöntemiyle sayısal özelliklere dönüştürülmüş,
        ardından k-NN, SVM ve ANN modelleri karşılaştırılmıştır.

        Dashboard içerisinde:

        - SMS mesajı girilerek anlık tahmin alınabilir.
        - Spam olasılığı görüntülenebilir.
        - Modellerin performans metrikleri incelenebilir.
        - Confusion matrix ve ROC grafikleri görüntülenebilir.
        """
    )

    st.info(
        "Bu proje danışmalı makine öğrenmesi kapsamında ikili metin sınıflandırma problemidir."
    )

    results_df = load_comparison_results()
    show_metric_cards(results_df)

    show_image_if_exists(
        "📊 Model Karşılaştırma Grafiği",
        OUTPUTS_DIR / "model_comparison_bar.png"
    )


# =========================================================
# SMS Tahmini
# =========================================================

elif page == "SMS Tahmini":
    st.title("🔍 SMS Spam Tahmini")

    selected_model_name = st.selectbox(
        "Kullanılacak modeli seçiniz:",
        list(available_models.keys())
    )

    selected_model_path = available_models[selected_model_name]

    try:
        model = load_model(selected_model_path)
        st.success(f"{selected_model_name} başarıyla yüklendi.")
    except Exception as e:
        st.error("Model yüklenirken hata oluştu.")
        st.exception(e)
        st.stop()

    st.markdown("Aşağıdaki alana bir SMS mesajı giriniz:")

    user_message = st.text_area(
        "SMS mesajı:",
        height=150,
        placeholder="Örnek: Congratulations! You have won a free prize. Click now!"
    )

    col1, col2 = st.columns([1, 3])

    with col1:
        predict_button = st.button("Tahmin Et", type="primary")

    if predict_button:
        if user_message.strip() == "":
            st.warning("Lütfen bir SMS mesajı giriniz.")
        else:
            prediction = model.predict([user_message])[0]
            spam_probability = predict_spam_probability(model, user_message)

            label = "SPAM" if prediction == 1 else "HAM / NORMAL"

            st.subheader("Tahmin Sonucu")

            if prediction == 1:
                st.error(f"🚨 Tahmin: {label}")
            else:
                st.success(f"✅ Tahmin: {label}")

            st.metric(
                "Spam Olasılığı",
                f"%{spam_probability * 100:.2f}"
            )

            st.progress(float(spam_probability))

            st.markdown("### Temizlenmiş Metin")
            st.code(clean_text(user_message))

            st.markdown(
                """
                **Yorum:**  
                Spam olasılığı yüksekse model mesajın reklam, ödül, kampanya veya şüpheli bağlantı içerme ihtimalini yüksek görmüştür.
                """
            )

    st.divider()

    st.markdown("### Deneme Mesajları")

    sample_messages = [
        "Congratulations! You have won a free ticket. Call now to claim your prize.",
        "Hi, are we still meeting at 6 pm today?",
        "URGENT! Your account has been selected for a cash reward.",
        "Can you send me the homework file when you are free?"
    ]

    for msg in sample_messages:
        st.code(msg)


# =========================================================
# Model Karşılaştırma
# =========================================================

elif page == "Model Karşılaştırma":
    st.title("📊 Model Karşılaştırma Sonuçları")

    results_df = load_comparison_results()

    if results_df is None:
        st.warning(
            "outputs/model_comparison_results.csv bulunamadı. "
            "Önce src/04_model_comparison.py dosyasını çalıştırmalısın."
        )
    else:
        st.subheader("Model Performans Tablosu")

        display_df = results_df.copy()

        metric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]

        for col in metric_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(4)

        st.dataframe(display_df, use_container_width=True)

        show_metric_cards(results_df)

        st.subheader("Metriklerin Anlamı")

        st.markdown(
            """
            - **Accuracy:** Tüm mesajlar içinde doğru sınıflandırılanların oranıdır.
            - **Precision:** Spam denilen mesajların gerçekten spam olma oranıdır.
            - **Recall:** Gerçek spam mesajların ne kadarının yakalandığını gösterir.
            - **F1:** Precision ve recall değerlerinin dengeli ortalamasıdır.
            - **ROC-AUC:** Modelin sınıfları ayırma başarısını gösterir.
            """
        )

        show_image_if_exists(
            "📈 Model Karşılaştırma Bar Grafiği",
            OUTPUTS_DIR / "model_comparison_bar.png"
        )


# =========================================================
# Grafikler
# =========================================================

elif page == "Grafikler":
    st.title("📉 Grafikler ve Görseller")

    st.markdown(
        """
        Bu bölümde modellerin confusion matrix, ROC ve karşılaştırma grafikleri görüntülenir.
        """
    )

    show_image_if_exists(
        "SVM Confusion Matrix",
        OUTPUTS_DIR / "svm_comparison_confusion_matrix.png"
    )

    show_image_if_exists(
        "k-NN Confusion Matrix",
        OUTPUTS_DIR / "knn_comparison_confusion_matrix.png"
    )

    show_image_if_exists(
        "ANN Confusion Matrix",
        OUTPUTS_DIR / "ann_comparison_confusion_matrix.png"
    )

    show_image_if_exists(
        "SVM ROC Curve",
        OUTPUTS_DIR / "svm_roc_curve.png"
    )

    show_image_if_exists(
        "Model Karşılaştırma Grafiği",
        OUTPUTS_DIR / "model_comparison_bar.png"
    )


# =========================================================
# Proje Bilgisi
# =========================================================

elif page == "Proje Bilgisi":
    st.title("ℹ️ Proje Bilgisi")

    st.markdown(
        """
        ## Proje 8: SMS Spam Tespiti

        **Problem tipi:** İkili sınıflandırma  
        **Amaç:** SMS mesajının spam veya normal olduğunu tahmin etmek.

        ## Kullanılan Yaklaşım

        Bu projede ham SMS metinleri doğrudan modele verilmemiştir. Öncelikle metinler temizlenmiş,
        ardından TF-IDF yöntemi ile sayısal özelliklere dönüştürülmüştür.

        ## Kullanılan Modeller

        - k-NN
        - SVM
        - ANN / MLPClassifier

        ## Ön İşleme Adımları

        - Eksik veri kontrolü
        - Boş mesaj kontrolü
        - Küçük harfe dönüştürme
        - URL temizleme
        - E-posta temizleme
        - Sayı ve noktalama temizleme
        - Label encoding
        - TF-IDF feature extraction
        - Feature selection / extraction

        ## Değerlendirme Metrikleri

        - Accuracy
        - Precision
        - Recall
        - F1-score
        - ROC-AUC
        - Confusion matrix
        - ROC curve

        ## Sınırlılıklar

        Veri seti İngilizce SMS mesajlarından oluştuğu için modelin Türkçe SMS mesajlarında aynı başarıyı göstermesi garanti değildir.
        Ayrıca gerçek dünyadaki spam mesaj türleri zamanla değişebileceği için modelin yeni verilerle güncellenmesi gerekir.
        """
    )

    summary_path = OUTPUTS_DIR / "best_model_summary.txt"

    if summary_path.exists():
        st.subheader("En İyi Model Özeti")

        with open(summary_path, "r", encoding="utf-8") as file:
            st.text(file.read())