#%%
import os
import scipy.io
import mne
import numpy as np
import matplotlib.pyplot as plt

# Đường dẫn tới thư mục chứa các file .mat đã giải nén
data_root = 'eeg_data'

# Lấy danh sách tất cả các file .mat trong thư mục
mat_files = [f for f in os.listdir(data_root) if f.endswith('.mat')]

# Duyệt qua từng file .mat
for mat_file in mat_files:
    mat_file_path = os.path.join(data_root, mat_file)
    
    # Đọc file .mat
    mat = scipy.io.loadmat(mat_file_path)

    # Trích xuất thông tin từ trường 'o'
    try:
        # Truy xuất sample frequency (sampFreq)
        samp_freq = mat['o']['sampFreq'][0, 0][0][0]

        # Truy xuất dữ liệu (data)
        data = mat['o']['data'][0, 0]

        # In ra thông tin kiểm tra
        print(f"File: {mat_file}")
        print(f"Sample Frequency: {samp_freq}")
        print(f"Data Shape: {data.shape}")
        print(f"First 5 data points:\n{data[:5]}")
    
    except KeyError as e:
        print(f"Missing key in file {mat_file}: {e}")

#%%
import numpy as np

channel_indices = np.array(range(3, 17))
channel_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
channel_map = dict(zip(channel_names, channel_indices))
#%%
import pandas as pd

df = pd.DataFrame.from_dict(data)
#%%
# Lấy số lượng mẫu từ dữ liệu
num_samples = data.shape[0]  # Số dòng trong ma trận `data`

# Tần số lấy mẫu (Hz)
samp_freq = 128  # Theo mô tả, tần số lấy mẫu là 128 Hz

# Tính thời gian thí nghiệm (giây)
experiment_duration = num_samples / samp_freq

# Hiển thị kết quả
print(f"Thời gian của thí nghiệm là {experiment_duration:.2f} giây.")
print(f"Tương đương {experiment_duration / 60:.2f} phút.")

#%%
i_ch = 0
time = np.arange(1000) / samp_freq
channel = data[:1000, channel_indices[i_ch]]
plt.plot(time, channel)
plt.xlabel('time (s)')
plt.ylabel(f'EEG channel {channel_names[i_ch]}')
#%%
# Giả sử `data` là dữ liệu EEG đã tải
time = np.arange(1000) / samp_freq  # Tạo trục thời gian với 1000 mẫu
plt.figure(figsize=(15, 10))  # Tăng kích thước đồ thị

# Vẽ tất cả các kênh
for i_ch, ch_name in enumerate(channel_names):
    channel = data[:1000, channel_indices[i_ch]]
    plt.plot(time, channel, label=f'EEG ch {ch_name}')

# Thêm nhãn và chú thích
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Amplitude (μV)', fontsize=14)
plt.title('EEG Signals from All Channels', fontsize=16)
plt.legend(loc='upper right', fontsize=12)
plt.grid(True)

# Hiển thị đồ thị
plt.show()
#%%
# Tạo trục thời gian cho toàn bộ dữ liệu
time = np.arange(data.shape[0]) / samp_freq  # Trục thời gian với toàn bộ mẫu dữ liệu

plt.figure(figsize=(15, 10))  # Tăng kích thước đồ thị

# Vẽ tất cả các kênh
for i_ch, ch_name in enumerate(channel_names):
    channel = data[:, channel_indices[i_ch]]  # Lấy toàn bộ dữ liệu của từng kênh
    plt.plot(time, channel, label=f'EEG ch {ch_name}')

# Thêm nhãn và chú thích
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Amplitude (μV)', fontsize=14)
plt.title('EEG Signals from All Channels (Full Dataset)', fontsize=16)
plt.legend(loc='upper right', fontsize=12)
plt.grid(True)

# Hiển thị đồ thị
plt.show()

#%%
states = {
 'focused': data[:samp_freq * 10 * 60, :],
  'unfocused': data[samp_freq * 10 * 60:samp_freq * 20 * 60, :],
  'drowsy': data[samp_freq * 30 * 60:, :],
}
#%%
bands = {'alpha': (8, 13), 'delta': (0.5, 4), 'beta': (13, 30), 'gamma': (30, np.inf)}
#%%
# Định nghĩa các dải tần số (bands)
bands = {
    'delta': (0.5, 4),   # Delta: 0.5–4 Hz
    'theta': (4, 8),     # Theta: 4–8 Hz
    'alpha': (8, 13),    # Alpha: 8–13 Hz
    'beta': (13, 30),    # Beta: 13–30 Hz
    'gamma': (30, 50)    # Gamma: 30–50 Hz
}

#%%
def get_data(filename):
    mat = scipy.io.loadmat(os.path.join(data_root, filename))
    data = mat['o']['data'][0, 0]
    FS = mat['o']['sampFreq'][0][0][0][0]

    states = {
     'focused': data[:FS * 10 * 60, :],
      'unfocused': data[FS * 10 * 60:FS * 20 * 60, :],
      'drowsy': data[FS * 30 * 60:, :],
    }
    return states
#%%
def get_powers(channel, FS=128):
    # Loại bỏ NaN hoặc kiểm tra mảng trống
    if channel.size == 0 or np.all(np.isnan(channel)):
        return {band: 0 for band in bands.keys()}  # Gửi trả 0 nếu không có dữ liệu hợp lệ

    # Tiến hành tính toán
    channel = channel - channel.mean()  # Trung bình hóa
    freq, psd = signal.periodogram(channel, fs=FS, nfft=256)

    powers = {}
    for band_name, band_limits in bands.items():
        low, high = band_limits
        band_psd = psd[(freq >= low) & (freq < high)]
        if band_psd.size > 0:
            powers[band_name] = band_psd.mean()
        else:
            powers[band_name] = 0  # Nếu không có giá trị, gán về 0
    return powers

#%%
from scipy import signal

rows_list = []
for subject_idx in range(1, 35):
    states = get_data(f"eeg_record{subject_idx}.mat")
    for ch_name, ch_idx in channel_map.items():
        for state, eeg in states.items():
            powers = get_powers(eeg[:, ch_idx])
            powers['state'] = state
            powers['channel'] = ch_name
            powers['subject'] = f"subject_{subject_idx}"
            rows_list.append(powers)
#%%
df = pd.DataFrame(rows_list)
#%%
df.head()
#%%
df.info()
#%%
df.describe()
#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))

# Vẽ các histogram cho các đặc trưng số (alpha, delta, beta, gamma)
plt.subplot(2, 2, 1)
sns.histplot(df['alpha'], kde=True, bins=30)
plt.title('Distribution of Alpha')

plt.subplot(2, 2, 2)
sns.histplot(df['delta'], kde=True, bins=30)
plt.title('Distribution of Delta')

plt.subplot(2, 2, 3)
sns.histplot(df['beta'], kde=True, bins=30)
plt.title('Distribution of Beta')

plt.subplot(2, 2, 4)
sns.histplot(df['gamma'], kde=True, bins=30)
plt.title('Distribution of Gamma')

# Hiển thị các đồ thị
plt.tight_layout()
plt.show()

#%%
import pandas as pd

def remove_outliers_iqr(df, columns):
    """
    Loại bỏ outliers cho các cột chỉ định trong DataFrame bằng phương pháp IQR (Interquartile Range).
    
    Args:
    - df: DataFrame chứa dữ liệu
    - columns: Danh sách các tên cột cần kiểm tra và loại bỏ outliers
    
    Returns:
    - DataFrame đã loại bỏ outliers
    """
    for col in columns:
        Q1 = df[col].quantile(0.25)  # Tính Q1
        Q3 = df[col].quantile(0.75)  # Tính Q3
        IQR = Q3 - Q1  # Tính khoảng IQR
        
        lower_bound = Q1 - 1.5 * IQR  # Giới hạn dưới
        upper_bound = Q3 + 1.5 * IQR  # Giới hạn trên
        
        # Lọc bỏ các giá trị ngoài giới hạn
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df

columns_to_check = ['alpha', 'delta', 'beta', 'gamma']
df_cleaned = remove_outliers_iqr(df, columns_to_check)

# In DataFrame đã loại bỏ outliers
df_cleaned.head()

#%%
plt.figure(figsize=(12, 10))

# Vẽ các histogram cho các đặc trưng số (alpha, delta, beta, gamma)
plt.subplot(2, 2, 1)
sns.histplot(df_cleaned['alpha'], kde=True, bins=30)
plt.title('Distribution of Alpha')

plt.subplot(2, 2, 2)
sns.histplot(df_cleaned['delta'], kde=True, bins=30)
plt.title('Distribution of Delta')

plt.subplot(2, 2, 3)
sns.histplot(df_cleaned['beta'], kde=True, bins=30)
plt.title('Distribution of Beta')

plt.subplot(2, 2, 4)
sns.histplot(df_cleaned['gamma'], kde=True, bins=30)
plt.title('Distribution of Gamma')

# Hiển thị các đồ thị
plt.tight_layout()
plt.show()

#%%

# Giả sử df là DataFrame chứa các cột 'alpha', 'delta', 'beta', 'gamma', 'channel'
# Lấy danh sách các kênh
channels = df_cleaned['channel'].unique()

# Tạo figure
plt.figure(figsize=(15, 10))

# Duyệt qua từng đặc trưng
features = ['alpha', 'delta', 'beta', 'gamma']
for feature in features:
    feature_values = []
    
    # Tính trung bình giá trị của đặc trưng theo từng kênh
    for ch in channels:
        mean_value = df_cleaned[df_cleaned['channel'] == ch][feature].mean()
        feature_values.append(mean_value)

    # Vẽ đường biểu diễn giá trị trung bình đặc trưng của từng kênh
    plt.plot(channels, feature_values, label=f'{feature.title()} Power')

# Thêm nhãn, tiêu đề, và chú thích
plt.xlabel('EEG Channels', fontsize=14)
plt.ylabel('Mean Power', fontsize=14)
plt.title('Mean Power of EEG Bands across Channels', fontsize=16)
plt.legend(loc='upper right', fontsize=12)
plt.grid(True)
plt.xticks(rotation=45)  # Xoay nhãn kênh nếu cần
plt.tight_layout()

# Hiển thị đồ thị
plt.show()

#%%
import numpy as np

# Giá trị giả định
channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

# Tạo dữ liệu ngẫu nhiên (mỗi band có giá trị tại các kênh)
data = {
    band: np.random.rand(len(channels)) for band in bands
}

#%% md
# - wavelet transform: shannon entrophy, wavelet energy, mean, variance, kurtosis, skewness
# - Wavelet transform là một công cụ mạnh mẽ để phân tích tín hiệu EEG trong miền thời gian - tần số. Sau khi thực hiện wavelet transform, bạn có thể trích xuất các đặc điểm như Shannon entropy, wavelet energy, mean, variance, kurtosis, và skewness từ các coefficients của wavelet
#%%
import pywt
import numpy as np

def wavelet_transform(signal, wavelet='db4', level=4):
    # wavelet='db4' là Daubechies 4, phổ biến trong phân tích EEG
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return coeffs

#%%
def shannon_entropy(coeff):
    prob = np.abs(coeff) / np.sum(np.abs(coeff))  # Tính xác suất
    entropy = -np.sum(prob * np.log2(prob + 1e-12))  # Tính entropy
    return entropy

#%%
def wavelet_energy(coeff):
    return np.sum(np.square(coeff))

#%%
def compute_mean(coeff):
    return np.mean(coeff)

#%%
from scipy.stats import kurtosis

def compute_kurtosis(coeff):
    return kurtosis(coeff)
    
#%%
from scipy.stats import skew

def compute_skewness(coeff):
    return skew(coeff)

#%%
def extract_wavelet_features(signal, wavelet='db4', level=4):
    coeffs = wavelet_transform(signal, wavelet, level)
    
    features = {}
    for i, coeff in enumerate(coeffs):
        features[f'level_{i+1}_entropy'] = shannon_entropy(coeff)
        features[f'level_{i+1}_energy'] = wavelet_energy(coeff)
        features[f'level_{i+1}_mean'] = compute_mean(coeff)
        features[f'level_{i+1}_variance'] = compute_variance(coeff)
        features[f'level_{i+1}_kurtosis'] = compute_kurtosis(coeff)
        features[f'level_{i+1}_skewness'] = compute_skewness(coeff)
    
    return features

#%%
from scipy import signal
import pandas as pd
import numpy as np

# Định nghĩa các dải tần số
bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50)
}

# Hàm tính năng lượng các băng tần
def get_powers(channel, FS=128):
    if channel.size == 0 or np.all(np.isnan(channel)):
        return {band: 0 for band in bands.keys()}  # Trả về 0 nếu dữ liệu không hợp lệ

    channel = channel - channel.mean()  # Trung bình hóa tín hiệu
    freq, psd = signal.periodogram(channel, fs=FS, nfft=256)

    powers = {}
    for band_name, band_limits in bands.items():
        low, high = band_limits
        band_psd = psd[(freq >= low) & (freq < high)]
        if band_psd.size > 0:
            powers[band_name] = band_psd.mean()
        else:
            powers[band_name] = 0  # Nếu không có giá trị, gán về 0
    return powers

# Giả sử get_data và channel_map đã được định nghĩa
rows_list = []
for subject_idx in range(1, 35):
    states = get_data(f"eeg_record{subject_idx}.mat")
    for ch_name, ch_idx in channel_map.items():
        for state, eeg in states.items():
            powers = get_powers(eeg[:, ch_idx])
            powers['state'] = state
            powers['channel'] = ch_name
            powers['subject'] = f"subject_{subject_idx}"
            powers['raw_signal'] = eeg[:, ch_idx].tolist()  # Lưu tín hiệu thô vào DataFrame
            rows_list.append(powers)

# Tạo DataFrame
df = pd.DataFrame(rows_list)

#%%
df
#%%
print(df['raw_signal'].head())
print(df['raw_signal'].apply(len).describe())  # Xem độ dài tín hiệu
#%%
# Giữ lại các tín hiệu có độ dài > 0
df = df[df['raw_signal'].apply(lambda x: len(x) > 0)]
#%%
import pywt
import numpy as np
from scipy.stats import kurtosis, skew

def extract_wavelet_features(signal, wavelet='db4', level=4):
    if len(signal) == 0:  # Bỏ qua tín hiệu rỗng
        return {f'wavelet_level_{i}_{stat}': np.nan 
                for i in range(level + 1) 
                for stat in ['energy', 'mean', 'var', 'kurt', 'skew']}
    
    try:
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        features = {}
        for i, coeff in enumerate(coeffs):
            features[f'wavelet_level_{i}_energy'] = np.sum(coeff ** 2)  # Năng lượng
            features[f'wavelet_level_{i}_mean'] = np.mean(coeff)       # Giá trị trung bình
            features[f'wavelet_level_{i}_var'] = np.var(coeff)         # Phương sai
            features[f'wavelet_level_{i}_kurt'] = kurtosis(coeff)      # Kurtosis
            features[f'wavelet_level_{i}_skew'] = skew(coeff)          # Skewness
        return features
    except Exception as e:
        print(f"Error processing signal: {e}")
        return {f'wavelet_level_{i}_{stat}': np.nan 
                for i in range(level + 1) 
                for stat in ['energy', 'mean', 'var', 'kurt', 'skew']}

#%%
