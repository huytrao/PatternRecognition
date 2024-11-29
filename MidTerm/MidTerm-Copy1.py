#%%
import os
import scipy.io
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from scipy.fftpack import fft

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
    
    except KeyError as e:
        print(f"Missing key in file {mat_file}: {e}")

#%%
channel_indices = np.array(range(3, 17))
channel_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
channel_map = dict(zip(channel_names, channel_indices))
#%%
df = pd.DataFrame.from_dict(data)
#%%
states = {
 'focused': data[:samp_freq * 10 * 60, :],
  'unfocused': data[samp_freq * 10 * 60:samp_freq * 20 * 60, :],
  'drowsy': data[samp_freq * 30 * 60:, :],
}
#%%
# Lấy số lượng mẫu từ dữ liệu
num_samples = data.shape[0]  # Số dòng trong ma trận `data`

# Tần số lấy mẫu (Hz)
FS = 128  # Theo mô tả, tần số lấy mẫu là 128 Hz

# Tính thời gian thí nghiệm (giây)
experiment_duration = num_samples / samp_freq

# Hiển thị kết quả
print(f"Thời gian của thí nghiệm là {experiment_duration:.2f} giây.")
print(f"Tương đương {experiment_duration / 60:.2f} phút.")

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

# Hàm lấy dữ liệu từ tệp .mat
def get_data(filename):
    mat = scipy.io.loadmat(filename)
    data = mat['o']['data'][0, 0]  # Dữ liệu EEG
    FS = mat['o']['sampFreq'][0][0][0][0]  # Tần số mẫu (Hz)
    
    states = {
        'focused': data[:FS * 10 * 60, :],  # 10 phút đầu
        'unfocused': data[FS * 10 * 60:FS * 20 * 60, :],  # 10 phút tiếp theo
        'drowsy': data[FS * 30 * 60:, :],  # Sau 30 phút
    }
    
    return states

#%%
from scipy.signal import butter, filtfilt

def bandpass_filter(eeg_signal, lowcut, highcut, FS, order=5):
    nyquist = 0.5 * FS
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, eeg_signal)
    return filtered_signal

#%%
def remove_mean(eeg_signal):
    return eeg_signal - np.mean(eeg_signal)

#%%
def normalize_signal(eeg_signal):
    return (eeg_signal - np.min(eeg_signal)) / (np.max(eeg_signal) - np.min(eeg_signal))

#%%
import numpy as np

def create_segments(signal, FS, window_size):
    """
    Cắt tín hiệu thành các đoạn nhỏ theo độ dài cửa sổ (window_size).

    Parameters:
    - signal: Tín hiệu EEG đã được xử lý.
    - FS: Tần số mẫu (sampling frequency).
    - window_size: Độ dài cửa sổ (tính bằng giây).

    Returns:
    - segments: Danh sách các đoạn tín hiệu.
    """
    segment_length = int(FS * window_size)  # Số mẫu trong một cửa sổ
    segments = []

    # Cắt tín hiệu thành các đoạn có độ dài bằng với segment_length
    for start in range(0, len(signal), segment_length):
        end = start + segment_length
        segment = signal[start:end]
        
        # Nếu đoạn tín hiệu có đủ kích thước, thêm vào danh sách
        if len(segment) == segment_length:
            segments.append(segment)
    
    return segments

#%%
def segment_signal(eeg_signal, window_size, FS):
    window_samples = window_size * FS
    segments = []
    for i in range(0, len(eeg_signal), window_samples):
        segment = eeg_signal[i:i+window_samples]
        if len(segment) == window_samples:
            segments.append(segment)
    return np.array(segments)

#%%
def preprocess_eeg(eeg_signal, FS, lowcut=0.5, highcut=40, window_size=2):
    # Kiểm tra chiều dài tín hiệu trước khi lọc
    if len(eeg_signal) < 35:  # Kiểm tra nếu tín hiệu quá ngắn
        print(f"Tín hiệu quá ngắn, bỏ qua: {len(eeg_signal)} mẫu")
        return []  # Trả về một danh sách rỗng để bỏ qua tín hiệu này
    
    # Lọc tín hiệu
    filtered_signal = bandpass_filter(eeg_signal, lowcut, highcut, FS)
    
    # Loại bỏ trung bình tín hiệu
    cleaned_signal = remove_mean(filtered_signal)
    
    # Cắt tín hiệu thành các đoạn nhỏ
    segments = create_segments(cleaned_signal, FS, window_size)
    
    return segments

#%%
def get_band_powers(eeg_signal, FS):
    N = len(eeg_signal)  # Số mẫu tín hiệu
    fft_result = fft(eeg_signal)  # Fourier Transform
    positive_frequencies = np.fft.fftfreq(N, d=1/FS)[:N//2]  # Tần số dương
    fft_magnitude = np.abs(fft_result[:N//2])  # Biên độ phổ
    
    band_powers = {}
    for band, (low, high) in bands.items():
        band_indices = np.where((positive_frequencies >= low) & (positive_frequencies < high))
        band_powers[band] = np.sum(fft_magnitude[band_indices])  # Năng lượng trong dải tần
        
    return band_powers

#%%
import os
import scipy.io

data_root = "eeg_data"  # Thay bằng đường dẫn chính xác đến thư mục chứa tệp .mat

# Lấy danh sách tất cả các tệp .mat trong thư mục
mat_files = [f for f in os.listdir(data_root) if f.endswith('.mat')]

rows_list = []

for mat_file in mat_files:
    states = get_data(os.path.join(data_root, mat_file))  # Đảm bảo cung cấp đường dẫn đầy đủ
    for ch_name, ch_idx in channel_map.items():
        for state, eeg in states.items():
            eeg_signal = eeg[:, ch_idx]  # Lấy tín hiệu của kênh ch_idx
            
            # Tiến hành preprocessing và tính toán band powers
            segments = preprocess_eeg(eeg_signal, FS)
            
            for segment in segments:
                band_powers = get_band_powers(segment, FS)
                band_powers['state'] = state
                band_powers['channel'] = ch_name
                band_powers['subject'] = mat_file  # Lưu tên tệp .mat vào cột subject
                rows_list.append(band_powers)

# Chuyển đổi kết quả thành DataFrame
df = pd.DataFrame(rows_list)
df.head()

#%%
df.info()
#%%
import os
import scipy.io
import pywt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import kurtosis, skew

# Đường dẫn tới thư mục chứa dữ liệu .mat
data_root = "eeg_data"  # Thay bằng đường dẫn chính xác đến thư mục chứa tệp .mat

# Hàm wavelet transform
def wavelet_transform(signal, wavelet='db4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return coeffs

# Hàm tính Shannon entropy
def shannon_entropy(coeff):
    prob = np.abs(coeff) / np.sum(np.abs(coeff))  # Tính xác suất
    entropy = -np.sum(prob * np.log2(prob + 1e-12))  # Tính entropy
    return entropy

# Hàm tính wavelet energy
def wavelet_energy(coeff):
    return np.sum(np.square(coeff))

# Hàm tính mean
def compute_mean(coeff):
    return np.mean(coeff)

# Hàm tính variance
def compute_variance(coeff):
    return np.var(coeff)

# Hàm tính kurtosis
def compute_kurtosis(coeff):
    return kurtosis(coeff)

# Hàm tính skewness
def compute_skewness(coeff):
    return skew(coeff)

# Hàm trích xuất các đặc trưng wavelet
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

# Định nghĩa các băng tần
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

# Hàm đọc dữ liệu EEG từ các tệp .mat
def get_data(mat_file_path):
    data = scipy.io.loadmat(mat_file_path)  # Đọc tệp .mat
    # Bạn cần xác định cách lấy thông tin từ tệp .mat cụ thể cho dữ liệu EEG của mình
    # Giả sử data là một dictionary chứa thông tin cần thiết
    return data

# Đọc tất cả các tệp .mat từ thư mục data_root
mat_files = [f for f in os.listdir(data_root) if f.endswith('.mat')]

rows_list = []

# Lặp qua tất cả các tệp .mat và các kênh
for mat_file in mat_files:
    mat_file_path = os.path.join(data_root, mat_file)  # Lấy đường dẫn đầy đủ tới tệp .mat
    states = get_data(mat_file_path)
    
    for ch_name, ch_idx in channel_map.items():
        for state, eeg in states.items():
            # Lấy đặc trưng năng lượng các băng tần
            powers = get_powers(eeg[:, ch_idx])

            # Trích xuất các đặc trưng wavelet
            wavelet_features = extract_wavelet_features(eeg[:, ch_idx])

            # Kết hợp các đặc trưng vào một dictionary
            features = powers.copy()  # Sao chép các đặc trưng băng tần
            features.update(wavelet_features)  # Thêm các đặc trưng wavelet

            # Thêm thông tin bổ sung
            features['state'] = state
            features['channel'] = ch_name
            features['subject'] = mat_file
            features['raw_signal'] = eeg[:, ch_idx].tolist()  # Lưu tín hiệu thô vào DataFrame

            # Thêm vào danh sách rows_list
            rows_list.append(features)

# Tạo DataFrame
df = pd.DataFrame(rows_list)
df.head()

#%%
