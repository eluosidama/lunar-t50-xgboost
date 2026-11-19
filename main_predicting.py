"""Apply the saved T50 regressor to a six-band raster or lunar-oxide-cnn tiles."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import re
from uuid import uuid4

import numpy as np
import rasterio
from rasterio.windows import Window

from t50 import DEFAULT_MODEL, FEATURES, TARGET, load_model, predict


NODATA = -9999.0
BLOCK = re.compile(r'(?P<oxide>[^_]+)_(?P<area>block\d{2}_[+-]\d+to[+-]\d+)\.tif$')


def inputs_by_block(directory):
    groups = {}
    for path in sorted(directory.glob('*.tif')):
        parsed = BLOCK.fullmatch(path.name)
        if parsed and parsed['oxide'] in FEATURES:
            groups.setdefault(parsed['area'], {})[parsed['oxide']] = path
    if not groups:
        raise ValueError(f'{directory} contains no upstream <oxide>_blockNN_...tif files')
    result = {}
    for area, paths in sorted(groups.items()):
        missing = [name for name in FEATURES if name not in paths]
        if missing:
            raise ValueError(f'{area} is missing oxide rasters: {", ".join(missing)}')
        result[area] = [paths[name] for name in FEATURES]
    return result


def predict_raster(paths, output, bundle, size):
    if output.resolve() in {path.resolve() for path in paths}:
        raise ValueError('Output and input raster paths must differ')
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(path)) for path in paths]
        first = sources[0]
        if len(sources) == 1:
            if first.count != len(FEATURES):
                raise ValueError(f'{paths[0]} must contain six bands; found {first.count}')
            for expected, label in zip(FEATURES, first.descriptions):
                if label in FEATURES and label != expected:
                    raise ValueError(f'{paths[0]} band description does not match feature order: expected {expected}, found {label}')
        else:
            for path, src in zip(paths, sources):
                if (src.count != 1 or src.shape != first.shape or src.transform != first.transform
                        or src.crs != first.crs):
                    raise ValueError(f'{path} must be a single-band raster with matching dimensions, CRS and pixel alignment')
        if first.crs is None:
            raise ValueError(f'{paths[0]} has no raster CRS')

        profile = dict(driver='GTiff', width=first.width, height=first.height, count=1,
                       crs=first.crs, transform=first.transform, dtype='float32', nodata=NODATA,
                       tiled=True, blockxsize=256, blockysize=256, compress='deflate',
                       BIGTIFF='IF_SAFER')
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(f'.{output.name}.{uuid4().hex}.part')
        try:
            with rasterio.open(partial, 'w', **profile) as dst:
                dst.set_band_description(1, TARGET)
                dst.set_band_unit(1, 'degC')
                for y in range(0, first.height, size):
                    for x in range(0, first.width, size):
                        w, h = min(size, first.width - x), min(size, first.height - y)
                        window = Window(x, y, w, h)
                        if len(sources) == 1:
                            tile = first.read(window=window, masked=True)
                        else:
                            tile = np.ma.stack([src.read(1, window=window, masked=True) for src in sources])
                        values = tile.data
                        valid = (~np.ma.getmaskarray(tile).any(axis=0)
                                 & np.isfinite(values).all(axis=0)
                                 & (values >= 0).all(axis=0)
                                 & (values > 0).any(axis=0))
                        temperatures = np.full((h, w), NODATA, dtype=np.float32)
                        if valid.any():
                            temperatures[valid] = predict(bundle, values[:, valid].T)
                        dst.write(temperatures, 1, window=window)
            partial.replace(output)
        finally:
            partial.unlink(missing_ok=True)
    print(output)


def main():
    parser = argparse.ArgumentParser(description='Predict T50 (degrees C) per pixel from six oxide concentrations (wt%)')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input-raster', '--input_raster', type=Path, help='Six-band VRT/TIF stacked in the documented feature order')
    source.add_argument('--oxide-dir', type=Path, help='One upstream pred_blocks_* directory')
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument('--output-raster', '--output_raster', type=Path, help='Output GeoTIFF for six-band input')
    output.add_argument('--output-dir', type=Path, help='Output directory for block input')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--block-size', '--block_size', type=int, default=512)
    args = parser.parse_args()
    if args.block_size < 1:
        parser.error('--block-size must be positive')
    if (args.input_raster is None) != (args.output_raster is None):
        parser.error('--input-raster requires --output-raster; --oxide-dir requires --output-dir')
    bundle = load_model(args.model)
    if args.input_raster:
        predict_raster([args.input_raster], args.output_raster, bundle, args.block_size)
    else:
        for area, paths in inputs_by_block(args.oxide_dir).items():
            predict_raster(paths, args.output_dir / f'T50_{area}.tif', bundle, args.block_size)


if __name__ == '__main__':
    main()
