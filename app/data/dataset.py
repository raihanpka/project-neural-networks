import numpy as np
import matplotlib.pyplot as plt

def create_data(samples=100, classes=3, plot=True):
    """
    Membuat dataset sythetic dengan kelompokan yang sangat terpisah untuk pengklasifikasian yang lebih mudah.
    
    Argumen:
        samples: Jumlah data point yang akan dibuat
        classes: Jumlah kelas yang akan dibuat
        plot: Apakah data akan ditampilkan visual
        
    Mengembalikan:
        X: Data fitur, shape (samples, 2)
        y: Label, shape (samples,)
    """
    # Atur seed acak untuk reprodukbilitas
    np.random.seed(0)
    
    # Membuat data sythetic dengan kelompokan yang sangat terpisah untuk pengklasifikasian yang lebih mudah
    X = np.zeros((samples, 2))
    y = np.zeros(samples, dtype=np.int32)
    
    # Membuat kelompokan sangat terpisah untuk setiap kelas
    jumlah_point_per_kelas = samples // classes
    pusat = [
        [-5, -5],  # Kelas 0
        [0, 5],    # Kelas 1
        [5, 0]     # Kelas 2
    ]
    
    for nomor_kelas in range(classes):
        # Dapatkan pusat untuk kelas ini
        pusat_x, pusat_y = pusat[nomor_kelas]
        
        # Membuat kluster yang tergolong sangat dekat sekitar pusat
        ix = range(jumlah_point_per_kelas * nomor_kelas, jumlah_point_per_kelas * (nomor_kelas + 1))
        X[ix] = np.random.randn(jumlah_point_per_kelas, 2) * 0.3 + np.array([pusat_x, pusat_y])
        y[ix] = nomor_kelas
    
    # Tampilkan data jika diminta
    if plot:
        plt.figure(figsize=(10, 8))
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.8)
        plt.colorbar(label='Kelas')
        plt.title('Dataset Sintetis')
        plt.xlabel('Fitur 1')
        plt.ylabel('Fitur 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('scatter_plot.png')  # Simpan plot
        print("Scatter Plot")
    return X, y 