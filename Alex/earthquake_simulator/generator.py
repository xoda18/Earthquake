"""
Независимый генератор землетрясений (в сырых отсчётах ±596)
Не использует Dataset_Central, создаёт данные с нуля
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class IndependentEarthquakeGenerator:
    """Генератор данных в масштабе сырых отсчётов (±596)"""

    def __init__(self, sampling_rate=250, duration_sec=60):
        self.sampling_rate = sampling_rate
        self.duration_sec = duration_sec
        self.num_samples = sampling_rate * duration_sec

    def generate_earthquake_data(self, magnitude=4.5, p_wave_start_sec=10,
                                 earthquake_duration_sec=15, noise_level=0.005):
        """
        Генерирует землетрясение в единицах g (±2.0 g) — соответствует MPU6500.

        Args:
            magnitude: Магнитуда землетрясения (2-7)
            p_wave_start_sec: Когда начинается P-волна (сек)
            earthquake_duration_sec: Длительность S-волны (сек)
            noise_level: Уровень фонового шума (g); MPU6500 в покое ≈ 0.003–0.005 g
        """
        time = np.arange(self.num_samples) / self.sampling_rate
        accel_x = np.random.normal(0, noise_level, self.num_samples)
        accel_y = np.random.normal(0, noise_level, self.num_samples)
        accel_z = np.random.normal(0, noise_level, self.num_samples)

        # P-волна (быстрая, слабая)
        p_start_idx = int(p_wave_start_sec * self.sampling_rate)
        p_duration_idx = int(3 * self.sampling_rate)  # 3 секунды
        p_amplitude = magnitude * 0.008  # Амплитуда в g

        for i in range(p_start_idx, min(p_start_idx + p_duration_idx, self.num_samples)):
            t_local = (i - p_start_idx) / self.sampling_rate
            envelope = np.exp(-t_local / 1.5)
            phase = 2 * np.pi * 6 * t_local  # 6 Гц P-волна

            accel_x[i] += p_amplitude * envelope * np.sin(phase)
            accel_y[i] += p_amplitude * envelope * np.sin(phase + np.pi/4)

        # S-волна (мощная, низкочастотная)
        s_start_idx = int((p_wave_start_sec + 3) * self.sampling_rate)
        s_duration_idx = int(earthquake_duration_sec * self.sampling_rate)
        s_amplitude = magnitude * 0.015  # в g

        for i in range(s_start_idx, min(s_start_idx + s_duration_idx, self.num_samples)):
            t_local = (i - s_start_idx) / self.sampling_rate
            envelope = max(0, 1 - t_local / (earthquake_duration_sec + 2))

            phase1 = 2 * np.pi * 2 * t_local  # 2 Гц
            phase2 = 2 * np.pi * 4 * t_local  # 4 Гц
            wave = 0.6 * np.sin(phase1) + 0.4 * np.sin(phase2)

            accel_x[i] += s_amplitude * envelope * wave
            accel_y[i] += s_amplitude * envelope * wave * np.cos(np.pi/3)
            accel_z[i] += s_amplitude * envelope * wave * 0.4

        # Поверхностные волны (долгоживущие)
        surf_start_idx = int((p_wave_start_sec + 8) * self.sampling_rate)
        surf_duration_idx = int((earthquake_duration_sec + 15) * self.sampling_rate)
        surf_amplitude = magnitude * 0.010  # в g

        for i in range(surf_start_idx, min(surf_start_idx + surf_duration_idx, self.num_samples)):
            t_local = (i - surf_start_idx) / self.sampling_rate
            envelope = max(0, 1 - t_local / (earthquake_duration_sec + 15))
            phase = 2 * np.pi * 0.8 * t_local

            accel_x[i] += surf_amplitude * envelope * np.sin(phase)
            accel_y[i] += surf_amplitude * envelope * np.cos(phase)

        # Обрезать до ±2.0 g (физический предел MPU6500 в режиме ±2g)
        accel_x = np.clip(accel_x, -2.0, 2.0)
        accel_y = np.clip(accel_y, -2.0, 2.0)
        accel_z = np.clip(accel_z, -2.0, 2.0)

        return np.column_stack([accel_x, accel_y, accel_z])

    def generate_noise_data(self, noise_level=0.005):
        """Генерирует только фоновый шум (без землетрясения)"""
        accel_x = np.random.normal(0, noise_level, self.num_samples)
        accel_y = np.random.normal(0, noise_level, self.num_samples)
        accel_z = np.random.normal(0, noise_level, self.num_samples)

        return np.column_stack([accel_x, accel_y, accel_z])

    def save_to_csv(self, data, filename, label):
        """Сохраняет данные в CSV"""
        start_time = datetime.now()
        timestamps = [start_time + timedelta(seconds=t) for t in np.arange(len(data)) / self.sampling_rate]

        df = pd.DataFrame({
            'timestamp': timestamps,
            'x': data[:, 0].astype(int),
            'y': data[:, 1].astype(int),
            'z': data[:, 2].astype(int),
            'label': label
        })

        df.to_csv(filename, index=False)
        print(f"✅ {filename}: {len(df)} отсчётов, label={label}")
        return df


def main():
    print("🌍 НЕЗАВИСИМЫЙ ГЕНЕРАТОР ЗЕМЛЕТРЯСЕНИЙ\n")
    print("=" * 80)

    gen = IndependentEarthquakeGenerator(sampling_rate=250, duration_sec=60)

    # Генерировать разные сценарии
    print("📊 Генерирую данные...\n")

    # 1. Сильное землетрясение (mag 5.5)
    data_strong = gen.generate_earthquake_data(magnitude=5.5, p_wave_start_sec=10,
                                               earthquake_duration_sec=15)
    df_strong = gen.save_to_csv(data_strong, 'earthquake_strong.csv', label=1)

    # 2. Слабое землетрясение (mag 3.0)
    data_weak = gen.generate_earthquake_data(magnitude=3.0, p_wave_start_sec=10,
                                            earthquake_duration_sec=10)
    df_weak = gen.save_to_csv(data_weak, 'earthquake_weak.csv', label=1)

    # 3. Умеренное землетрясение (mag 4.0)
    data_moderate = gen.generate_earthquake_data(magnitude=4.0, p_wave_start_sec=10,
                                                earthquake_duration_sec=12)
    df_moderate = gen.save_to_csv(data_moderate, 'earthquake_moderate.csv', label=1)

    # 4. Фоновый шум (без землетрясения)
    data_noise = gen.generate_noise_data(noise_level=50)
    df_noise = gen.save_to_csv(data_noise, 'earthquake_noise.csv', label=0)

    print("\n" + "=" * 80)
    print("📈 СТАТИСТИКА")
    print("=" * 80)

    for name, df in [('Сильное (5.5)', df_strong), ('Слабое (3.0)', df_weak),
                     ('Умеренное (4.0)', df_moderate), ('Шум', df_noise)]:
        print(f"\n{name}:")
        print(f"  X: [{df['x'].min():6d}, {df['x'].max():6d}]")
        print(f"  Y: [{df['y'].min():6d}, {df['y'].max():6d}]")
        print(f"  Z: [{df['z'].min():6d}, {df['z'].max():6d}]")

    print("\n🎉 Готово!")


if __name__ == '__main__':
    main()
