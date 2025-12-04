import sys
import os
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models.neuralnetwork import NeuralNetwork
from app.function.layer import Dense
from app.function.activations import ReLU, Softmax, Linear
from app.function.regularization import BatchNormalization, Dropout
from app.function.check_loss import CategoricalCrossentropy, MeanSquaredError
from app.function.metrics import calculate_accuracy
from app.data.generate_dataset import create_data

# Normalisasi min-max untuk array numpy
def _minmax_scale_np(arr, minv=None, maxv=None):
    arr = np.array(arr, dtype=float)
    if minv is None:
        minv = arr.min(axis=0)
    if maxv is None:
        maxv = arr.max(axis=0)
    denom = np.where((maxv - minv) == 0.0, 1.0, (maxv - minv))
    scaled = (arr - minv) / denom
    return scaled, minv, maxv

# Diskretisasi target untuk klasifikasi
def discretize_target(y, n_bins=5):
    percentiles = np.percentile(y, np.linspace(0, 100, n_bins + 1))
    labels = np.digitize(y, percentiles[1:-1])
    return labels, percentiles

# Membuat urutan data time series yang di-flattenkan
def create_sequences_flattened(df, features, target, seq_len):
    """Create flattened sliding windows from the entire dataframe."""
    # Sort by time if available
    df_sorted = df.sort_values('time') if 'time' in df.columns else df
    
    feat_arr = df_sorted[features].values
    target_arr = df_sorted[target].values
    
    Xf = []
    Yf = []
    for i in range(len(feat_arr) - seq_len):
        win = feat_arr[i:i + seq_len]
        Xf.append(win.flatten())
        Yf.append(target_arr[i + seq_len])
    
    Xf = np.array(Xf, dtype=float)
    Yf = np.array(Yf, dtype=float)
    return Xf, Yf

# Melatih dan mengevaluasi model
def train_and_eval_model(X, Y, is_regression=False, n_classes=5, epochs=100, batch_size=64, lr=0.005):
    # Membangun model berdasarkan arsitektur default sederhana yang mirip dengan aplikasi lainnya
    input_size = X.shape[1]
    model = NeuralNetwork()
    model.add(Dense(input_size, 128, learning_rate=lr))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.1))
    model.add(Dense(128, 64, learning_rate=lr))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.1))
    if not is_regression:
        # Memastikan Y dalam bentuk label integer untuk klasifikasi
        Y_arr = np.array(Y)
        if Y_arr.ndim > 1:
            Y_flat = Y_arr.flatten()
        else:
            Y_flat = Y_arr
        # Jika float atau banyak nilai unik -> diskretisasi menjadi n_classes
        if Y_flat.dtype.kind in 'fc' or len(np.unique(Y_flat)) > n_classes:
            Y_classes, _ = discretize_target(Y_flat, n_bins=n_classes)
        else:
            Y_classes = Y_flat.astype(int)
        model.add(Dense(64, n_classes, learning_rate=lr))
        model.add(Softmax())
        model.set_loss(CategoricalCrossentropy(regularization_l2=1e-4))
        model.train(X, Y_classes, epochs=epochs, batch_size=batch_size)
        preds = model.predict_proba(X)
        acc = calculate_accuracy(Y_classes, preds)
        return model, acc, None
    else:
        model.add(Dense(64, 1, learning_rate=lr))
        model.add(Linear())
        model.set_loss(MeanSquaredError())
        model.train(X, Y.reshape(-1, 1), epochs=epochs, batch_size=batch_size)
        preds = model.predict_proba(X).flatten()
        mae = np.mean(np.abs(preds - Y.flatten()))
        return model, mae, None


def main(dataset_path='app/data/soil_moisture_level.csv', use_timeseries=False, generate=False, seq_length=15, n_classes=5, epochs=100, batch_size=64, lr=0.005, regression=False):
    print('Starting pipeline...')
    if generate:
        print('Generating synthetic dataset...')
        X, Y = create_data(samples=1000)
        is_reg = regression
        model, metric, _ = train_and_eval_model(X, Y, is_reg, n_classes=n_classes, epochs=epochs, batch_size=batch_size, lr=lr)
        print('Done. Metric:', metric)
        return
    elif os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
    else:
        # Use inline create_data wrapper for simple synthetic tabular data
        X, Y = create_data(samples=1000)
        is_reg = regression
        model, metric, _ = train_and_eval_model(X, Y, is_reg, n_classes=n_classes, epochs=epochs, batch_size=batch_size, lr=lr)
        print('Done. Metric:', metric)
        return

    # Normalize columns
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    # map ttime -> time if present
    if 'ttime' in df.columns and 'time' not in df.columns:
        df.rename(columns={'ttime': 'time'}, inplace=True)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])

    # Choose features for sensor dataset (if available) otherwise fallback to numeric columns
    if 'pm1' in df.columns and 'soil_moisture' in df.columns:
        feature_cols = [c for c in ['pm1', 'pm2', 'pm3', 'ammonia', 'luminosity', 'temperature', 'humidity', 'pressure'] if c in df.columns]
        target_col = 'soil_moisture'
    else:
        numeric_feats = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_feats) >= 2:
            feature_cols = numeric_feats[:-1]
            target_col = numeric_feats[-1]
        else:
            raise RuntimeError('No suitable feature/target columns found in dataset')

    # Create sequences from the entire dataset
    X_seq, Y_seq = create_sequences_flattened(train_df_scaled, feature_cols, target_col, seq_length)
    if X_seq.size == 0:
        raise RuntimeError('No sequences generated')

    if not regression:
        Y_classes, edges = discretize_target(Y_seq, n_bins=n_classes)
        model, metric, _ = train_and_eval_model(X_seq, Y_classes, is_regression=False, n_classes=n_classes, epochs=epochs, batch_size=batch_size, lr=lr)
        print('Train accuracy:', metric)
    else:
        model, metric, _ = train_and_eval_model(X_seq, Y_seq, is_regression=True, n_classes=n_classes, epochs=epochs, batch_size=batch_size, lr=lr)
        print('Train MAE:', metric)

    # No holdout evaluation required; completed training on full dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='app/data/soil_moisture_level.csv')
    parser.add_argument('--seq_length', type=int, default=15)
    parser.add_argument('--n_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--regression', action='store_true')
    parser.add_argument('--generate', action='store_true')
    args = parser.parse_args()

    main(dataset_path=args.dataset, use_timeseries=True, generate=args.generate, seq_length=args.seq_length, n_classes=args.n_classes, epochs=args.epochs, batch_size=args.batch_size, lr=args.learning_rate, regression=args.regression)
