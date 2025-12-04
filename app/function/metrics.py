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