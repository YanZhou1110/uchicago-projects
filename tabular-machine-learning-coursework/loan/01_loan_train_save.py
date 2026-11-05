import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor


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
    # Frequency-encode every column passed in (works for both DataFrame and ndarray)
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


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    df = pd.read_csv("Data for Cleaning & Modeling.csv", low_memory=False)
    df = clean_df(df)

    target_col = df.columns[0]   # should be X1 = interest rate
    y = _to_num(df[target_col])
    X = df.drop(columns=[target_col])

    mask = y.notna()
    y = y.loc[mask]
    X = X.loc[mask]

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    high_card_cols = []
    low_card_cols = []
    for c in cat_cols:
        if X[c].nunique(dropna=True) > 30:
            high_card_cols.append(c)
        else:
            low_card_cols.append(c)

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]), num_cols),

            ("lowcat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), low_card_cols),

            ("highcat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("freq", FrequencyEncoder())
            ]), high_card_cols),
        ],
        remainder="drop"
    )

    models = [
        ("ridge", Ridge(alpha=3.0)),
        ("hgb", HistGradientBoostingRegressor(random_state=42)),
        ("rf", RandomForestRegressor(
            random_state=42,
            n_estimators=300,
            n_jobs=-1,
            max_features="sqrt",
            min_samples_leaf=2
        )),
    ]

    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    best_name, best_pipe, best_rmse = None, None, float("inf")

    for name, model in models:
        pipe = Pipeline([("pre", pre), ("model", model)])
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_va)
        score = rmse(y_va, pred)
        print(name, "RMSE:", round(score, 4))
        if score < best_rmse:
            best_name, best_pipe, best_rmse = name, pipe, score

    best_pipe.fit(X, y)
    joblib.dump(best_pipe, "loan_model.joblib")
    print("Best model:", best_name)
    print("Best validation RMSE:", round(best_rmse, 4))
    print("Saved loan_model.joblib")


if __name__ == "__main__":
    main()
