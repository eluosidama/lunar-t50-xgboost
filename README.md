# Lunar T50 XGBoost

Predict temperature at 50% liquid fraction, T50 (degrees C), from six oxide concentrations (wt%). The supplied bundle includes the original XGBoost model and training-set scaler for the 148-sample dataset.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`. Dependency versions are specified in `requirements.txt`.

## Reproduce training

```bash
python main_training.py --output-dir outputs/training
```

`data/t50_148.csv` contains 148 rows, six oxide features and target `T_Liquidus_50`, preserving the supplied row order. Random seed 42 produces 118 training and 30 test samples. StandardScaler is fitted only on training features; an XGBoost regressor with 300 trees and `tree_method="hist"` predicts temperature directly. Outputs are `trained_model_148.joblib`, `predictions_table.csv` and `metrics.json`.

The previous documentation reported RMSE 3.1743 degrees C, MAE 2.4337 degrees C and R2 0.97238 on the 30 held-out FactSage samples. These values have not been revalidated in this packaging step. They do not quantify errors propagated from oxide inversion or full lunar mapping.

## Predict from CNN output

```bash
python main_predicting.py --oxide-dir ../lunar-oxide-cnn/outputs/predictions/pred_blocks_YYYYMMDD_HHMMSS --output-dir outputs/t50
```

Replace the timestamped directory with an actual CNN prediction directory. Files must follow `<oxide>_blockNN_...tif`. Each complete group of six produces a corresponding T50 GeoTIFF.

Alternatively, supply a six-band VRT/GeoTIFF in this exact order: **SiO2, Al2O3, FeO, MgO, CaO, TiO2**.

```bash
python main_predicting.py --model models/trained_model_148.joblib --input-raster oxides.vrt --output-raster outputs/T50.tif --block-size 512
```

Input values are wt% (45 means 45 wt%). Individual oxide rasters must have identical dimensions, CRS and pixel alignment. Pixels with NoData, non-finite or negative inputs, or all-zero compositions, produce -9999. Individual zero concentrations are allowed. Processing uses 512 by 512 pixel windows by default and preserves the input grid. Output is single-band Float32. No demo raster is supplied.

## Files and model provenance

`main_training.py` reproduces training; `main_predicting.py` handles raster input; `t50.py` defines feature order, model loading and scaled prediction. The original model bundle is retained unchanged at `models/trained_model_148.joblib`.

Original bundle SHA-256: `65da787b384967864000273acf491672e8c43cbf5064e74cfcf5ee5b8fcad24e`.

No license was supplied for this workflow. The authors should add their selected code and data licenses before public release.

## Manuscript and contact

**Regional variations in characteristic melting temperatures of lunar regolith revealed by Chang’e-6-constrained oxide mapping**

Publication status: Manuscript under submission.

Corresponding author: Feng Li, Beihang University.  
Email: [lifeng98@buaa.edu.cn](mailto:lifeng98@buaa.edu.cn).
