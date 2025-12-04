import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_data(samples=100, plot=True, save_csv=False, path='app/data/soil_moisture_synthetic.csv'):
    """Generate synthetic soil moisture data untuk satu lokasi.

    X: array(N, features=8) [pm1, pm2, pm3, ammonia, luminosity, temperature, humidity, pressure]
    y: array(N,) soil_moisture (0-100)
    """
    np.random.seed(0)
    
    # Suhu: Normal sekitar 25C, dibatasi antara 5 dan 45
    temperature = np.random.normal(loc=25.0, scale=6.0, size=samples)
    temperature = np.clip(temperature, 5.0, 45.0).round(1)

    # Kelembaban: Normal sekitar 60%, dibatasi 5-100
    humidity = np.random.normal(loc=60.0, scale=15.0, size=samples)
    humidity = np.clip(humidity, 5.0, 100.0).round(1)

    # Hujan: Sebagian besar nilai kecil dengan kejadian besar sesekali
    rainfall = np.random.exponential(scale=3.0, size=samples)
    dry_mask = np.random.rand(samples) < 0.35
    rainfall[dry_mask] = 0.0
    rainfall = np.round(rainfall, 1)

    # Tutup awan: 0-100 uniform
    cloud_cover = np.random.uniform(0, 100, samples).round(1)

    # Sensor columns
    pm1 = np.clip(10 + (100 - humidity) * 0.25 + np.random.normal(0.0, 5.0, samples), 0, None).round(1)
    pm2 = np.clip(pm1 * (0.9 + np.random.normal(0.0, 0.05, samples)), 0, None).round(1)
    pm3 = np.clip(pm1 * (0.8 + np.random.normal(0.0, 0.06, samples)), 0, None).round(1)
    ammonia = np.clip(np.random.normal(0.5, 0.2, samples), 0, None).round(2)
    luminosity = np.clip((1 - (cloud_cover / 100.0)) * 1000 + np.random.normal(0.0, 50.0, samples), 0, None).round(1)
    pressure = np.clip(np.random.normal(1013.0, 8.0, samples), 950.0, 1050.0).round(1)

    # Hitung soil moisture
    t_min, t_max = 5.0, 45.0
    h_min, h_max = 5.0, 100.0
    r_min, r_max = 0.0, max(rainfall.max(), 1.0)
    c_min, c_max = 0.0, 100.0

    t_norm = (temperature - t_min) / (t_max - t_min)
    h_norm = (humidity - h_min) / (h_max - h_min)
    r_norm = (rainfall - r_min) / (r_max - r_min) if r_max != r_min else np.zeros_like(rainfall)
    c_norm = (cloud_cover - c_min) / (c_max - c_min)

    soil_norm = 0.12 + (0.55 * r_norm) + (0.20 * h_norm) - (0.15 * t_norm) + (0.05 * c_norm)
    soil_norm += np.random.normal(0.0, 0.03, size=samples)
    soil_norm = np.clip(soil_norm, 0.0, 1.0)
    soil_moisture = np.round(soil_norm * 100, 1)

    df = pd.DataFrame({
        'pm1': pm1,
        'pm2': pm2,
        'pm3': pm3,
        'ammonia': ammonia,
        'luminosity': luminosity,
        'pressure': pressure,
        'temperature': temperature,
        'humidity': humidity,
        'rainfall': rainfall,
        'cloud_cover': cloud_cover,
        'soil_moisture': soil_moisture
    })
    
    if save_csv:
        df.to_csv(path, index=False)
    
    sensor_features = ['pm1', 'pm2', 'pm3', 'ammonia', 'luminosity', 'temperature', 'humidity', 'pressure']
    X = df[sensor_features].values.astype(float)
    y = df['soil_moisture'].values.astype(float)
    return X, y


if __name__ == '__main__':
    X, y = create_data(samples=1000, save_csv=True)
    print(f"Generated {len(X)} samples with {X.shape[1]} features")