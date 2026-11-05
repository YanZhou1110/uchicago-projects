import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score


def add_features(df):
    df = df.copy()

    df["Title"] = df["Name"].str.extract(r",\s*([^.]*)\.", expand=False)
    df["Title"] = df["Title"].fillna("Unknown")

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    df["Deck"] = df["Cabin"].astype(str).str[0]
    df.loc[df["Cabin"].isna(), "Deck"] = "Unknown"

    t = df["Ticket"].astype(str).str.upper()
    df["TicketPrefix"] = t.str.replace(r"[^A-Z]+", "", regex=True)
    df.loc[df["TicketPrefix"] == "", "TicketPrefix"] = "NONE"

    df = df.drop(columns=["Name", "Ticket", "Cabin"])
    return df


def main():
    df = pd.read_csv("train.csv")

    y = df["Survived"]
    X = df.drop(columns=["Survived"])

    X = add_features(X)

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols)
        ]
    )

    model = LogisticRegression(max_iter=2000)

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_val)

    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds, zero_division=0)
    f1 = f1_score(y_val, preds, zero_division=0)

    print("Validation accuracy:", round(acc, 4))
    print("Validation precision:", round(prec, 4))
    print("Validation F1:", round(f1, 4))

    joblib.dump(pipe, "titanic_model.joblib")
    print("Saved titanic_model.joblib")


if __name__ == "__main__":
    main()
