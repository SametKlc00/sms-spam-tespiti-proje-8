import os
import pandas as pd


DATA_DIR = "data"


def load_sms_dataset():
    """
    SMS Spam veri setini farklı dosya formatlarına göre okur.
    Kaggle spam.csv veya UCI SMSSpamCollection.txt formatını destekler.
    """

    spam_csv_path = os.path.join(DATA_DIR, "spam.csv")
    uci_txt_path = os.path.join(DATA_DIR, "SMSSpamCollection.txt")

    if os.path.exists(spam_csv_path):
        print("spam.csv dosyası bulundu. Kaggle formatı okunuyor...")

        df = pd.read_csv(spam_csv_path, encoding="latin-1")

        # Kaggle veri setinde genelde v1 ve v2 sütunları olur.
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


def main():
    df = load_sms_dataset()

    print("\nİlk 5 kayıt:")
    print(df.head())

    print("\nVeri seti boyutu:")
    print(df.shape)

    print("\nSütunlar:")
    print(df.columns.tolist())

    print("\nEksik veri sayısı:")
    print(df.isnull().sum())

    print("\nSınıf dağılımı:")
    print(df["label"].value_counts())

    print("\nSınıf dağılımı yüzdesel:")
    print(df["label"].value_counts(normalize=True) * 100)


if __name__ == "__main__":
    main()