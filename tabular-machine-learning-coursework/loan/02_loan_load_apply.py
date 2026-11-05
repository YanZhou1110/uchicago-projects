import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


def _clean_str(s):
    return (
        s.astype(str)
        .str.replace("\u00a0", " ", regex=False)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "NULL": np.nan, "null": np.nan})
    )


def _to_num(series):
    s = _clean_str(series)
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace("$", "", regex=False)
    s = s.str.replace("%", "", regex=False)
    s = s.str.replace("(", "-", regex=False).str.replace(")", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def clean_df(df):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == "object":
            s = _clean_str(df[c])
            sample = s.dropna().head(80)
            if len(sample) == 0:
                df[c] = s
                continue
            num_like = sample.str.match(r"^[-+]?[\d,]*\.?\d+%?$").mean()
            money_like = sample.str.contains(r"[\$,]").mean()
            if (num_like > 0.6) or (money_like > 0.6):
                df[c] = _to_num(df[c])
            else:
                df[c] = s
    return df


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = self._to_2d_array(X)
        n_rows, n_cols = X.shape
        self.freq_maps_ = []
        for j in range(n_cols):
            col = pd.Series(X[:, j]).fillna("__MISSING__").astype(str)
            vc = col.value_counts(dropna=False)
            freq_map = (vc / n_rows).to_dict()
            self.freq_maps_.append(freq_map)
        return self

    def transform(self, X):
        X = self._to_2d_array(X)
        n_rows, n_cols = X.shape
        out = np.zeros((n_rows, n_cols), dtype=float)
        for j in range(n_cols):
            freq_map = self.freq_maps_[j]
            col = pd.Series(X[:, j]).fillna("__MISSING__").astype(str)
            out[:, j] = col.map(freq_map).fillna(0.0).to_numpy(dtype=float)
        return out

    @staticmethod
    def _to_2d_array(X):
        if isinstance(X, pd.DataFrame):
            return X.to_numpy()
        if isinstance(X, pd.Series):
            return X.to_frame().to_numpy()
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X


def main():
    model = joblib.load("loan_model.joblib")

    hold = pd.read_csv("Holdout for Testing.csv", low_memory=False)
    hold = clean_df(hold)

    X_hold = hold.iloc[:, 1:].copy()

    preds = model.predict(X_hold)

    out = pd.DataFrame({hold.columns[0]: preds})

    name = "yanzhou"
    out_name = f"Results from {name}.csv"
    out.to_csv(out_name, index=False)

    print("Saved:", out_name)
    print(out.head())


if __name__ == "__main__":
    main()
