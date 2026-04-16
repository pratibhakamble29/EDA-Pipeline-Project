# ============================================================
#  DATA CLEANING & EXPLORATORY DATA ANALYSIS (EDA) PIPELINE
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os

warnings.filterwarnings("ignore")

# ── GLOBAL SETTINGS ─────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")

OUTPUT_DIR = "eda_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. DATA GENERATOR
# ============================================================
def generate_messy_dataset(n=500, seed=42):
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "CustomerID": range(1, n + 1),
        "Age": rng.integers(18, 70, n).astype(float),
        "Income": rng.normal(55000, 15000, n),
        "Score": rng.uniform(1, 10, n),
        "Region": rng.choice(["North", "South", "East", "West", None], n),
        "Gender": rng.choice(["Male", "Female", "male", "female", "M", "F", None], n),
        "Purchased": rng.choice(["Yes", "No", "yes", "no", "1", "0", None], n),
        "JoinDate": pd.date_range("2022-01-01", periods=n).astype(str)
    })

    # Missing values
    for col in ["Age", "Income", "Score"]:
        df.loc[rng.choice(n, int(n * 0.05)), col] = np.nan

    # Duplicates
    df = pd.concat([df, df.sample(20)], ignore_index=True)

    return df


# ============================================================
# 2. CLEANING STEPS
# ============================================================

def step1_overview(df):
    print("Shape:", df.shape)
    print(df.dtypes)
    print(df.head())


def step2_missing(df):
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def step3_duplicates(df):
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.drop("CustomerID", axis=1)
    return df


def step4_types(df):
    # Gender clean
    gender_map = {"male": "Male", "m": "Male", "female": "Female", "f": "Female"}
    df["Gender"] = df["Gender"].str.lower().map(gender_map)
    df["Gender"] = df["Gender"].fillna("Unknown")

    # Purchased clean
    purchase_map = {"yes": "Yes", "1": "Yes", "no": "No", "0": "No"}
    df["Purchased"] = df["Purchased"].str.lower().map(purchase_map)
    df["Purchased"] = df["Purchased"].fillna("No")
    # Date
    df["JoinDate"] = pd.to_datetime(df["JoinDate"], errors="coerce")
    df = df.dropna(subset=["JoinDate"])

    return df


def step5_outliers(df):
    for col in df.select_dtypes(include=np.number).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[col] = df[col].clip(lower, upper)

    return df


def step6_features(df):
    df["AgeGroup"] = pd.cut(df["Age"], bins=[0,25,35,50,100],
                           labels=["18-25","26-35","36-50","50+"])

    df["IncomeGroup"] = pd.cut(df["Income"], bins=[-np.inf,35000,60000,np.inf],
                              labels=["Low","Medium","High"])

    df["JoinYear"] = df["JoinDate"].dt.year
    df["JoinMonth"] = df["JoinDate"].dt.month

    return df


# ============================================================
# 3. EDA
# ============================================================

def eda_plots(df):

    # Distribution
    cols = ["Age", "Income", "Score"]

    for col in cols:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True)
        sns.histplot(df["Age"], kde=True, bins=15)
        plt.title(f"Distribution of {col}")
        plt.savefig(f"{OUTPUT_DIR}/{col}_dist.png")
        plt.show()

    ts = df.groupby("JoinMonth").size()
    ts.plot(kind="line", marker="o")
    plt.title("Monthly Joins")
    plt.savefig(f"{OUTPUT_DIR}/timeseries.png")
    plt.show()
    # Correlation
    corr = df[["Age", "Income", "Score"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.savefig(f"{OUTPUT_DIR}/corr.png")
    plt.show()

    # Countplot
    sns.countplot(x="Purchased", hue="Gender", data=df)
    plt.savefig(f"{OUTPUT_DIR}/purchase.png")
    plt.show()

    # Pairplot (sample for performance)
    sample_df = df.sample(min(200, len(df)))
    sns.pairplot(sample_df, hue="Purchased")
    plt.savefig(f"{OUTPUT_DIR}/pair.png")
    plt.show()


# ============================================================
# 4. MAIN
# ============================================================

def run_pipeline():
    df = generate_messy_dataset()

    step1_overview(df)
    df = step2_missing(df)
    df = step3_duplicates(df)
    df = step4_types(df)
    df = step5_outliers(df)
    df = step6_features(df)

    eda_plots(df)

    df.to_csv(f"{OUTPUT_DIR}/cleaned.csv", index=False)
    print("✅ Done!")

    return df


if __name__ == "__main__":
    run_pipeline()
