# Proje 8 - SMS Spam Tespiti

Bu proje, Danışmalı Makine Öğrenmesi final projesi kapsamında hazırlanmıştır. Amaç, bir SMS mesajının **spam** mı yoksa **normal/ham** mı olduğunu makine öğrenmesi yöntemleriyle tahmin etmektir.

## Öğrenci Bilgisi

**Ad Soyad:** Samet Kılıç  
**Proje Konusu:** SMS Spam Tespiti  
**Problem Tipi:** İkili Metin Sınıflandırma  
**Kullanılan Modeller:** SVM, k-NN, ANN / MLPClassifier  

---

## Proje Özeti

Bu çalışmada SMS Spam Collection veri seti kullanılmıştır. Veri setindeki SMS mesajları önce temel metin ön işleme adımlarından geçirilmiş, ardından TF-IDF yöntemiyle sayısal özelliklere dönüştürülmüştür.

Modelleme aşamasında üç farklı danışmalı makine öğrenmesi modeli karşılaştırılmıştır:

- k-NN
- SVM
- ANN / MLPClassifier

Tüm modelleme süreci veri sızıntısını azaltmak ve yeniden üretilebilirliği sağlamak amacıyla `scikit-learn Pipeline` yapısı içinde gerçekleştirilmiştir. Hiperparametre araması için `GridSearchCV` kullanılmıştır.

Projenin uygulama tarafında ise `Streamlit` ile bir dashboard hazırlanmıştır. Kullanıcı dashboard üzerinden SMS mesajı girerek mesajın spam olasılığını görebilir.

---

## Kullanılan Teknolojiler

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

---

## Proje Klasör Yapısı

```text
Proje_8_SMS_Spam_Tespiti_Samet_Kilic/
│
├── data/
│   └── SMSSpamCollection.txt
│
├── src/
│   ├── 01_data_check.py
│   ├── 02_preprocessing_split.py
│   ├── 03_train_svm_pipeline.py
│   └── 04_model_comparison.py
│
├── dashboard/
│   └── app.py
│
├── models/
│   ├── best_model.pkl
│   ├── svm_best_pipeline.pkl
│   ├── knn_best_pipeline.pkl
│   └── ann_best_pipeline.pkl
│
├── outputs/
│   ├── model_comparison_results.csv
│   ├── model_comparison_bar.png
│   ├── svm_roc_curve.png
│   ├── combined_roc_curve.png
│   └── confusion matrix görselleri
│
├── report/
│   ├── Proje_8_SMS_Spam_Tespiti_Samet_Kilic_Final_Rapor.pdf
│   └── Proje_8_SMS_Spam_Tespiti_Samet_Kilic_Final_Rapor.docx
│
├── requirements.txt
└── README.md
