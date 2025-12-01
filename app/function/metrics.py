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