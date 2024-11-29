#%%
##%% 
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, ArrayType, StringType, DoubleType
from pyspark.sql.functions import pandas_udf, PandasUDFType, col
import os
import scipy.io
import numpy as np

# Tạo SparkSession
spark = SparkSession.builder \
    .appName("EEG Data Processing with Fourier Transform") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

#%%
##%%
# Định nghĩa schema cho Spark DataFrame
schema = StructType([
    StructField("subject", StringType(), True),
    StructField("state", StringType(), True),
    StructField("channel", StringType(), True),
    StructField("raw_signal", ArrayType(DoubleType()), True)
])

# Hàm đọc file `.mat`
def read_mat_file(file_path):
    mat = scipy.io.loadmat(file_path)
    data = mat['o']['data'][0, 0]
    samp_freq = mat['o']['sampFreq'][0, 0][0][0]
    states = {
        'focused': data[:samp_freq * 10 * 60, :],
        'unfocused': data[samp_freq * 10 * 60:samp_freq * 20 * 60, :],
        'drowsy': data[samp_freq * 30 * 60:, :]
    }
    return states

# Tạo dữ liệu từ các file
data_root = 'eeg_data'
rows = []
channel_map = {
    'AF3': 3, 'F7': 4, 'F3': 5, 'FC5': 6, 'T7': 7, 'P7': 8,
    'O1': 9, 'O2': 10, 'P8': 11, 'T8': 12, 'FC6': 13, 'F4': 14, 'F8': 15, 'AF4': 16
}


for mat_file in os.listdir(data_root):
    if mat_file.endswith('.mat'):
        states = read_mat_file(os.path.join(data_root, mat_file))
        subject_id = mat_file.split('.')[0]
        for state, eeg_data in states.items():
            for channel_name, channel_idx in channel_map.items():
                signal = eeg_data[:, channel_idx]
                rows.append((subject_id, state, channel_name, signal.tolist()))

# Chuyển dữ liệu thành Spark DataFrame
df = spark.createDataFrame(rows, schema=schema)
df.show(5)

#%%
spark = SparkSession.builder \
    .appName("Fourier Features Extraction") \
    .config("spark.sql.execution.arrow.enabled", "true") \
    .getOrCreate()
#%%
from pyspark.sql import Row
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import StructType, StructField, DoubleType

@pandas_udf(StructType([
    StructField("fft_mean", DoubleType(), True),
    StructField("fft_variance", DoubleType(), True),
    StructField("fft_kurtosis", DoubleType(), True),
    StructField("fft_skewness", DoubleType(), True),
    StructField("fft_max_frequency", DoubleType(), True),
    StructField("fft_power_spectrum", DoubleType(), True)
]), PandasUDFType.SCALAR)
def extract_fourier_features_udf(signals):
    results = []
    sampling_rate = 128
    for signal in signals:
        if len(signal) == 0:
            results.append([None] * 6)
            continue
        N = len(signal)
        freq = fftfreq(N, d=1/sampling_rate)
        fft_values = np.abs(fft(signal))

        features = [
            np.mean(fft_values),
            np.var(fft_values),
            kurtosis(fft_values),
            skew(fft_values),
            freq[np.argmax(fft_values)],
            np.sum(fft_values**2) / N
        ]
        results.append(features)
    return pd.DataFrame(results, columns=["fft_mean", "fft_variance", "fft_kurtosis", "fft_skewness", "fft_max_frequency", "fft_power_spectrum"])

df = df.withColumn("fourier_features", extract_fourier_features_udf(col("raw_signal")))
df.select("subject", "state", "channel", "fourier_features").show(5, truncate=False)

#%%
# UDF tính các đặc trưng wavelet
@pandas_udf("map<string,double>", PandasUDFType.SCALAR)
def extract_wavelet_features_udf(signals):
    results = []
    for signal in signals:
        if len(signal) == 0:
            results.append({})
            continue
        coeffs = pywt.wavedec(signal, 'db4', level=4)
        features = {}
        for i, coeff in enumerate(coeffs):
            features[f'level_{i}_energy'] = np.sum(coeff**2)
            features[f'level_{i}_mean'] = np.mean(coeff)
            features[f'level_{i}_kurt'] = kurtosis(coeff)
            features[f'level_{i}_skew'] = skew(coeff)
        results.append(features)
    return results

# Thêm cột wavelet features
df = df.withColumn("wavelet_features", extract_wavelet_features_udf(col("raw_signal")))
df.select("subject", "state", "channel", "wavelet_features").show(5, truncate=False)

#%%
from pyspark.sql.functions import expr

def remove_outliers(df, col_name):
    q1 = df.approxQuantile(col_name, [0.25], 0.01)[0]
    q3 = df.approxQuantile(col_name, [0.75], 0.01)[0]
    iqr = q3 - q1
    lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return df.filter((col(col_name) >= lower_bound) & (col(col_name) <= upper_bound))

# Áp dụng loại bỏ outliers
columns_to_check = ['band_powers', 'wavelet_features']
for col_name in columns_to_check:
    df = remove_outliers(df, col_name)

df.show(5)

#%%
import matplotlib.pyplot as plt
import pandas as pd

# Chuyển đổi về Pandas DataFrame
pandas_df = df.toPandas()

# Vẽ histogram
pandas_df['band_powers'].apply(pd.Series).hist(figsize=(10, 8))
plt.show()
