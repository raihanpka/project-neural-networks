import numpy as np

def calculate_accuracy(y_true, y_pred):
    """Menghitung akurasi prediksi

    Fungsi ini menghitung akurasi dari prediksi yang dihasilkan oleh model
    dengan menggunakan metode argmax. Akurasi dirinci sebagai jumlah
    prediksi yang benar dibagi dengan jumlah semua prediksi.

    Parameters
    ----------
    y_true : array-like
        Array yang berisi label benar (ground truth)
    y_pred : array-like
        Array yang berisi prediksi yang dihasilkan oleh model

    Returns
    -------
    float
        Nilai akurasi prediksi
    """
    predictions = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
    return np.mean(predictions == y_true)

def calculate_r2_score(y_true, y_pred):
    """Menghitung R² (coefficient of determination)
    
    R² = 1 - (SS_res / SS_tot)
    Nilai berkisar dari -inf hingga 1.0, dimana 1.0 adalah perfect prediction
    
    Parameters
    ----------
    y_true : array-like
        Array yang berisi nilai benar (ground truth)
    y_pred : array-like
        Array yang berisi prediksi yang dihasilkan oleh model
    
    Returns
    -------
    float
        Nilai R² score
    """
    ss_res = np.sum((y_true - y_pred) ** 2)  # residual sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # total sum of squares
    if ss_tot == 0:
        return 0.0
    r2 = 1 - (ss_res / ss_tot)
    return r2

def calculate_mae(y_true, y_pred):
    """Menghitung Mean Absolute Error (MAE)
    
    Parameters
    ----------
    y_true : array-like
        Array yang berisi nilai benar (ground truth)
    y_pred : array-like
        Array yang berisi prediksi yang dihasilkan oleh model
    
    Returns
    -------
    float
        Nilai MAE
    """
    return np.mean(np.abs(y_true - y_pred))

def calculate_mse(y_true, y_pred):
    """Menghitung Mean Squared Error (MSE)
    
    Parameters
    ----------
    y_true : array-like
        Array yang berisi nilai benar (ground truth)
    y_pred : array-like
        Array yang berisi prediksi yang dihasilkan oleh model
    
    Returns
    -------
    float
        Nilai MSE
    """
    return np.mean((y_true - y_pred) ** 2)

def calculate_rmse(y_true, y_pred):
    """Menghitung Root Mean Squared Error (RMSE)
    
    Parameters
    ----------
    y_true : array-like
        Array yang berisi nilai benar (ground truth)
    y_pred : array-like
        Array yang berisi prediksi yang dihasilkan oleh model
    
    Returns
    -------
    float
        Nilai RMSE
    """
    return np.sqrt(calculate_mse(y_true, y_pred))

def calculate_mape(y_true, y_pred, epsilon=1e-8):
    """Menghitung Mean Absolute Percentage Error (MAPE)
    
    Parameters
    ----------
    y_true : array-like
        Array yang berisi nilai benar (ground truth)
    y_pred : array-like
        Array yang berisi prediksi yang dihasilkan oleh model
    epsilon : float, optional
        Nilai kecil untuk menghindari division by zero, default 1e-8
    
    Returns
    -------
    float
        Nilai MAPE dalam persentase (0-100)
    """
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def confusion_matrix(y_true, y_pred, num_classes):
    """Menghitung matriks konfusi

    Fungsi ini menghitung matriks konfusi yang menunjukkan jumlah kejadian
    benar positif (TP), kejadian salah positif (FP), dan kejadian salah
    negatif (FN).

    Parameters
    ----------
    y_true : array-like
        Array yang berisi label benar (ground truth)
    y_pred : array-like
        Array yang berisi prediksi yang dihasilkan oleh model
    num_classes : int
        Jumlah kelas yang ada dalam dataset

    Returns
    -------
    numpy.ndarray
        Matriks konfusi
    """
    predictions = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
    cm = np.zeros((num_classes, num_classes))
    for true, pred in zip(y_true, predictions):
        cm[true][pred] += 1
    return cm

def precision_recall_f1(y_true, y_pred, num_classes):
    """Menghitung precision, recall, dan F1 score

    Fungsi ini menghitung precision, recall, dan F1 score untuk setiap kelas
    yang ada dalam dataset.

    Parameters
    ----------
    y_true : array-like
        Array yang berisi label benar (ground truth)
    y_pred : array-like
        Array yang berisi prediksi yang dihasilkan oleh model
    num_classes : int
        Jumlah kelas yang ada dalam dataset

    Returns
    -------
    tuple
        Tuple berisi array precision, recall, dan f1 score
    """
    cm = confusion_matrix(y_true, y_pred, num_classes)
    
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
    
    return precision, recall, f1


def cross_validate(model_builder, X, y, n_splits=5, epochs=100, batch_size=32, 
                   regression=True, shuffle=True, random_state=None, verbose=True):
    """Perform K-Fold Cross Validation untuk Neural Network
    
    Parameters
    ----------
    model_builder : callable
        Fungsi yang mengembalikan model neural network baru
    X : array-like
        Data fitur
    y : array-like
        Data target
    n_splits : int, default=5
        Jumlah folds untuk cross validation
    epochs : int, default=100
        Jumlah epochs untuk training
    batch_size : int, default=32
        Ukuran batch untuk training
    regression : bool, default=True
        True untuk regression task, False untuk classification
    shuffle : bool, default=True
        Apakah data diacak sebelum dibagi menjadi folds
    random_state : int, optional
        Random seed untuk reproducibility
    verbose : bool, default=True
        Print progress selama cross validation
    
    Returns
    -------
    dict
        Dictionary berisi hasil cross validation:
        - 'fold_scores': list skor untuk setiap fold
        - 'mean_score': rata-rata skor
        - 'std_score': standard deviasi skor
        - 'metric': nama metric yang digunakan ('R2' atau 'Accuracy')
    """
    fold_scores = []
    fold_r2_scores = [] if regression else []
    fold_mae_scores = [] if regression else []
    
    # Generate fold indices
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        np.random.shuffle(indices)
    
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[:n_samples % n_splits] += 1
    
    current = 0
    fold_splits = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        fold_splits.append((train_idx, val_idx))
        current = stop
    
    # Perform cross validation
    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        if verbose:
            print(f"\n{'='*50}")
            print(f"Fold {fold_idx + 1}/{n_splits}")
            print(f"{'='*50}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Build fresh model for this fold
        model = model_builder()
        
        # Train model
        if verbose:
            print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")
        
        model.train(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                   verbose=False)
        
        # Evaluate
        val_pred = model.predict_proba(X_val)
        
        if regression:
            # For regression: calculate R² and MAE
            r2 = calculate_r2_score(y_val.flatten(), val_pred.flatten())
            mae = calculate_mae(y_val.flatten(), val_pred.flatten())
            score = r2
            fold_r2_scores.append(r2)
            fold_mae_scores.append(mae)
            
            if verbose:
                print(f"Fold {fold_idx + 1} - R²: {r2:.4f}, MAE: {mae:.4f}")
        else:
            # For classification: calculate accuracy
            score = calculate_accuracy(y_val, val_pred)
            
            if verbose:
                print(f"Fold {fold_idx + 1} - Accuracy: {score:.4f}")
        
        fold_scores.append(score)
    
    # Calculate statistics
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    
    results = {
        'fold_scores': fold_scores,
        'mean_score': mean_score,
        'std_score': std_score,
        'metric': 'R²' if regression else 'Accuracy',
        'n_splits': n_splits
    }
    
    if regression:
        results['fold_r2_scores'] = fold_r2_scores
        results['fold_mae_scores'] = fold_mae_scores
        results['mean_r2'] = np.mean(fold_r2_scores)
        results['std_r2'] = np.std(fold_r2_scores)
        results['mean_mae'] = np.mean(fold_mae_scores)
        results['std_mae'] = np.std(fold_mae_scores)
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"Cross Validation Results ({n_splits}-Fold)")
        print(f"{'='*50}")
        print(f"Mean {results['metric']}: {mean_score:.4f} (+/- {std_score:.4f})")
        if regression:
            print(f"Mean MAE: {results['mean_mae']:.4f} (+/- {results['std_mae']:.4f})")
        print(f"Fold scores: {[f'{s:.4f}' for s in fold_scores]}")
    
    return results