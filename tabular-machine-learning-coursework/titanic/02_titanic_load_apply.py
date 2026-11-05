import pandas as pd
import joblib


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
    model = joblib.load("titanic_model.joblib")

    holdout = pd.read_csv("holdout_test.csv")
    X_hold = holdout.drop(columns=["Survived"], errors="ignore")

    X_hold = add_features(X_hold)

    preds = model.predict(X_hold)

    out = pd.DataFrame({"Survived": preds})

    netid = "yanzhou"
    filename = f"Titanic Results from {netid}.csv"
    out.to_csv(filename, index=False)

    print("Saved:", filename)
    print(out.head())


if __name__ == "__main__":
    main()
