"""Central configuration: paths, feature lists, split fractions and the business cost model."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "creditcard.csv"
SPLIT_DIR = DATA_DIR / "splits"
ARTIFACT_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

MODEL_PATH = ARTIFACT_DIR / "fraud_pipeline.joblib"
BASELINE_LR_PATH = ARTIFACT_DIR / "baseline_logreg.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
REFERENCE_PROFILE_PATH = ARTIFACT_DIR / "reference_profile.json"

# Dataset: ULB / Worldline "Credit Card Fraud Detection" (Kaggle mlg-ulb/creditcardfraud)
DATA_URL = (
    "https://raw.githubusercontent.com/nsethi31/"
    "Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
)
DATA_SHA256 = "33a178be65174943a07d1736acf5898f6c8e1fa80950dbbe598c17eda968ec59"

TARGET = "Class"
TIME_COL = "Time"
AMOUNT_COL = "Amount"
PCA_COLS = [f"V{i}" for i in range(1, 29)]
RAW_INPUT_COLS = [TIME_COL] + PCA_COLS + [AMOUNT_COL]  # what a client sends

# Temporal split (fractions of the time-ordered stream)
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15  # test = remaining 0.15

RANDOM_STATE = 42

# ---- Business cost model (EUR) -------------------------------------------------
# False negative: the fraudulent amount is charged back to the issuer + admin fee.
# False positive: analyst review / customer friction (call, SMS, declined card).
FN_FIXED_COST = 15.0
FP_COST = 5.0

# Serving guard-rails
MAX_MISSING_FEATURES = 10   # more nulls than this -> payload rejected (422)
MAX_BATCH_SIZE = 1000
