"""The feature contract and inference pipeline of the 148-sample T50 model."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


FEATURES = ['SiO2', 'Al2O3', 'FeO', 'MgO', 'CaO', 'TiO2']
TARGET = 'T_Liquidus_50'
ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / 'models' / 'trained_model_148.joblib'


def load_model(path):
    bundle = joblib.load(path)
    scaler, model = bundle['scaler'], bundle['model']
    if (list(bundle['feature_names']) != FEATURES
            or not isinstance(scaler, StandardScaler)
            or not isinstance(model, XGBRegressor)
            or list(scaler.feature_names_in_) != FEATURES
            or model.n_features_in_ != len(FEATURES)):
        raise ValueError('Model type or six input features do not match the T50 workflow')
    return bundle


def predict(bundle, values):
    frame = pd.DataFrame(values, columns=FEATURES)
    scaled = bundle['scaler'].transform(frame)
    return bundle['model'].predict(scaled)
