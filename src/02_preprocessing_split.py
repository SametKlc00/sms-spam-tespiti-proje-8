import os
import re
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DATA_DIR = "data"
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
    """

    # Metni stringe çevir
    text = str(text)

    # Küçük harfe çevir
    text = text.lower()

    # URL benzeri ifadeleri temizle
    text = re.sub(r"http\S+|www\S+", " ", text)

    # E-posta adreslerini temizle
    text = re.sub(r"\S+@\S+", " ", text)

    # Sayıları temizle
    text = re.sub(r"\d+", " ", text)

    # Noktalama ve özel karakterleri temizle
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Fazla boşlukları tek boşluğa indir
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    df = load_sms_dataset()

    print("\nOrijinal veri boyutu:")
    print(df.shape)

    print("\nİlk 5 kayıt:")
    print(df.head())

    # Eksik veri kontrolü
    print("\nEksik veri sayısı:")
    print(df.isnull().sum())

    # Eksik label varsa çıkar
    df = df.dropna(subset=["label"])

    # Eksik mesaj varsa boş string ile doldur
    df["message"] = df["message"].fillna("")

    # Boş mesajları kontrol et
    empty_message_count = (df["message"].str.strip() == "").sum()
    print("\nBoş mesaj sayısı:")
    print(empty_message_count)

    # Temizleme işlemi
    df["clean_message"] = df["message"].apply(clean_text)

    # Temizleme sonrası boş kalan mesajları çıkar
    df = df[df["clean_message"].str.strip() != ""]

    print("\nTemizleme sonrası veri boyutu:")
    print(df.shape)

    # Label encoding
    label_encoder = LabelEncoder()
    df["label_encoded"] = label_encoder.fit_transform(df["label"])

    print("\nLabel encoding sonucu:")
    for original_label, encoded_label in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)):
        print(f"{original_label} -> {encoded_label}")

    print("\nSınıf dağılımı:")
    print(df["label"].value_counts())

    print("\nSınıf dağılımı yüzdesel:")
    print(df["label"].value_counts(normalize=True) * 100)

    # X ve y ayırma
    X = df["clean_message"]
    y = df["label_encoded"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nEğitim ve test veri boyutları:")
    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test :", y_test.shape)

    print("\nEğitim seti sınıf dağılımı:")
    print(y_train.value_counts())

    print("\nTest seti sınıf dağılımı:")
    print(y_test.value_counts())

    print("\nTemizlenmiş örnek mesajlar:")
    sample_df = pd.DataFrame({
        "original_message": df["message"].head(5),
        "clean_message": df["clean_message"].head(5),
        "label": df["label"].head(5),
        "label_encoded": df["label_encoded"].head(5)
    })

    print(sample_df)

    # İşlenmiş veriyi outputs klasörüne kaydet
    os.makedirs("outputs", exist_ok=True)

    processed_path = os.path.join("outputs", "processed_sms_data.csv")
    df.to_csv(processed_path, index=False, encoding="utf-8")

    print(f"\nİşlenmiş veri kaydedildi: {processed_path}")


if __name__ == "__main__":
    main()