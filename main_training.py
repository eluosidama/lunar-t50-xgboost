"""Reproduce the 118/30 split and the six-oxide XGBoost fit."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from t50 import FEATURES, ROOT, TARGET, predict


def main():
    parser = argparse.ArgumentParser(description='Reproduce T50 training and testing with 148 oxide compositions')
    parser.add_argument('--data', type=Path, default=ROOT / 'data' / 't50_148.csv')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/training'))
    args = parser.parse_args()
    table = pd.read_csv(args.data, float_precision='round_trip').loc[:, [*FEATURES, TARGET]]
    if len(table) != 148 or not np.isfinite(table.to_numpy(dtype=float)).all():
        parser.error('Training table must contain 148 complete finite rows of six oxide features and T_Liquidus_50')
    train, test = train_test_split(table, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[FEATURES])
    model = XGBRegressor(n_estimators=300, random_state=42, verbosity=0, tree_method='hist')
    model.fit(x_train, train[TARGET])
    bundle = {'model': model, 'scaler': scaler, 'feature_names': FEATURES}
    predicted = predict(bundle, test[FEATURES])
    measured = test[TARGET]
    metrics = {
        'n_train': len(train), 'n_test': len(test),
        'RMSE_C': root_mean_squared_error(measured, predicted),
        'MAE_C': mean_absolute_error(measured, predicted),
        'R2': r2_score(measured, predicted),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.output_dir / 'trained_model_148.joblib')
    result = test[FEATURES].assign(y_true=measured, y_pred=predicted)
    result.to_csv(args.output_dir / 'predictions_table.csv', index=False)
    summary = json.dumps(metrics, indent=2)
    (args.output_dir / 'metrics.json').write_text(summary + '\n', encoding='utf-8')
    print(summary)


if __name__ == '__main__':
    main()
