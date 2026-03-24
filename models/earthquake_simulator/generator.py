"""
Independent earthquake generator (g-unit scale ±2.0)
Does not use Dataset_Central, creates data from scratch
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class IndependentEarthquakeGenerator:
    """Generate synthetic earthquake data in g-units (±2.0 g)"""

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

        # P-wave (fast, weak)
        p_start_idx = int(p_wave_start_sec * self.sampling_rate)
        p_duration_idx = int(3 * self.sampling_rate)  # 3 seconds
        p_amplitude = magnitude * 0.008  # Amplitude in g

        for i in range(p_start_idx, min(p_start_idx + p_duration_idx, self.num_samples)):
            t_local = (i - p_start_idx) / self.sampling_rate
            envelope = np.exp(-t_local / 1.5)
            phase = 2 * np.pi * 6 * t_local  # 6 Hz P-wave

            accel_x[i] += p_amplitude * envelope * np.sin(phase)
            accel_y[i] += p_amplitude * envelope * np.sin(phase + np.pi/4)

        # S-wave (strong, low-frequency)
        s_start_idx = int((p_wave_start_sec + 3) * self.sampling_rate)
        s_duration_idx = int(earthquake_duration_sec * self.sampling_rate)
        s_amplitude = magnitude * 0.015  # in g

        for i in range(s_start_idx, min(s_start_idx + s_duration_idx, self.num_samples)):
            t_local = (i - s_start_idx) / self.sampling_rate
            envelope = max(0, 1 - t_local / (earthquake_duration_sec + 2))

            phase1 = 2 * np.pi * 2 * t_local  # 2 Hz
            phase2 = 2 * np.pi * 4 * t_local  # 4 Hz
            wave = 0.6 * np.sin(phase1) + 0.4 * np.sin(phase2)

            accel_x[i] += s_amplitude * envelope * wave
            accel_y[i] += s_amplitude * envelope * wave * np.cos(np.pi/3)
            accel_z[i] += s_amplitude * envelope * wave * 0.4

        # Surface waves (long-lived)
        surf_start_idx = int((p_wave_start_sec + 8) * self.sampling_rate)
        surf_duration_idx = int((earthquake_duration_sec + 15) * self.sampling_rate)
        surf_amplitude = magnitude * 0.010  # in g

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

    def generate_knock_data(self, n_knocks=None, knock_amplitude=None, noise_level=0.005):
        """
        Generate table knock / impact data (NOT earthquake).
        Short sharp impulses that decay in <0.5 sec — high frequency, no sustained shaking.
        """
        if n_knocks is None:
            n_knocks = np.random.randint(1, 6)
        if knock_amplitude is None:
            knock_amplitude = np.random.uniform(0.05, 0.8)  # g

        accel_x = np.random.normal(0, noise_level, self.num_samples)
        accel_y = np.random.normal(0, noise_level, self.num_samples)
        accel_z = np.random.normal(0, noise_level, self.num_samples)

        for _ in range(n_knocks):
            # Random time for each knock
            knock_sec = np.random.uniform(2, self.duration_sec - 2)
            knock_idx = int(knock_sec * self.sampling_rate)

            # Knock: sharp spike + fast exponential decay (0.05–0.3 sec)
            decay_time = np.random.uniform(0.05, 0.3)
            decay_samples = int(decay_time * self.sampling_rate)
            freq = np.random.uniform(15, 40)  # Hz — table resonance

            amp = knock_amplitude * np.random.uniform(0.5, 1.0)

            for j in range(min(decay_samples, self.num_samples - knock_idx)):
                t_local = j / self.sampling_rate
                envelope = np.exp(-t_local / (decay_time * 0.3))
                wave = np.sin(2 * np.pi * freq * t_local)

                idx = knock_idx + j
                # Knocks mainly hit Z axis (vertical)
                accel_z[idx] += amp * envelope * wave
                accel_x[idx] += amp * 0.3 * envelope * wave * np.random.uniform(0.5, 1.5)
                accel_y[idx] += amp * 0.3 * envelope * wave * np.random.uniform(0.5, 1.5)

        accel_x = np.clip(accel_x, -2.0, 2.0)
        accel_y = np.clip(accel_y, -2.0, 2.0)
        accel_z = np.clip(accel_z, -2.0, 2.0)

        return np.column_stack([accel_x, accel_y, accel_z])

    def generate_noise_data(self, noise_level=0.005):
        """Generate background noise only (no earthquake)"""
        accel_x = np.random.normal(0, noise_level, self.num_samples)
        accel_y = np.random.normal(0, noise_level, self.num_samples)
        accel_z = np.random.normal(0, noise_level, self.num_samples)

        return np.column_stack([accel_x, accel_y, accel_z])

    def save_to_csv(self, data, filename, label):
        """Save data to CSV"""
        start_time = datetime.now()
        timestamps = [start_time + timedelta(seconds=t) for t in np.arange(len(data)) / self.sampling_rate]

        df = pd.DataFrame({
            'timestamp': timestamps,
            'x': data[:, 0],
            'y': data[:, 1],
            'z': data[:, 2],
            'label': label
        })

        df.to_csv(filename, index=False)
        print(f"✅ {filename}: {len(df)} samples, label={label}")
        return df


def main():
    print("🌍 INDEPENDENT EARTHQUAKE GENERATOR\n")
    print("=" * 80)

    gen = IndependentEarthquakeGenerator(sampling_rate=250, duration_sec=60)

    # Generate different scenarios
    print("📊 Generating data...\n")

    # 1. Strong earthquake (mag 5.5)
    data_strong = gen.generate_earthquake_data(magnitude=5.5, p_wave_start_sec=10,
                                               earthquake_duration_sec=15)
    df_strong = gen.save_to_csv(data_strong, 'earthquake_strong.csv', label=1)

    # 2. Weak earthquake (mag 3.0)
    data_weak = gen.generate_earthquake_data(magnitude=3.0, p_wave_start_sec=10,
                                            earthquake_duration_sec=10)
    df_weak = gen.save_to_csv(data_weak, 'earthquake_weak.csv', label=1)

    # 3. Moderate earthquake (mag 4.0)
    data_moderate = gen.generate_earthquake_data(magnitude=4.0, p_wave_start_sec=10,
                                                earthquake_duration_sec=12)
    df_moderate = gen.save_to_csv(data_moderate, 'earthquake_moderate.csv', label=1)

    # 4. Background noise (no earthquake)
    data_noise = gen.generate_noise_data(noise_level=0.005)
    df_noise = gen.save_to_csv(data_noise, 'earthquake_noise.csv', label=0)

    print("\n" + "=" * 80)
    print("📈 STATISTICS")
    print("=" * 80)

    for name, df in [('Strong (5.5)', df_strong), ('Weak (3.0)', df_weak),
                     ('Moderate (4.0)', df_moderate), ('Noise', df_noise)]:
        print(f"\n{name}:")
        print(f"  X: [{df['x'].min():6.4f}, {df['x'].max():6.4f}]")
        print(f"  Y: [{df['y'].min():6.4f}, {df['y'].max():6.4f}]")
        print(f"  Z: [{df['z'].min():6.4f}, {df['z'].max():6.4f}]")

    print("\n🎉 Done!")


if __name__ == '__main__':
    main()
