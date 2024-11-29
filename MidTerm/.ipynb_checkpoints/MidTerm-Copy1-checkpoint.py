from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StructType, StructField, StringType, FloatType, ArrayType
from pyspark.ml.feature import VectorAssembler
import pyspark.ml.feature as feature
import numpy as np
import scipy.io
import pywt
from scipy.stats import kurtosis, skew
from scipy import signal

def create_spark_session():
    return SparkSession.builder \
        .appName("EEG Data Processing") \
        .getOrCreate()

def load_mat_files(spark, data_root):
    def extract_eeg_data(filename):
        mat = scipy.io.loadmat(f"{data_root}/{filename}")
        data = mat['o']['data'][0, 0]
        fs = mat['o']['sampFreq'][0][0][0][0]
        
        states = {
            'focused': data[:fs * 10 * 60, :],
            'unfocused': data[fs * 10 * 60:fs * 20 * 60, :],
            'drowsy': data[fs * 30 * 60:, :]
        }
        return states

    # Prepare schema for DataFrame
    schema = StructType([
        StructField("subject", StringType(), False),
        StructField("channel", StringType(), False),
        StructField("state", StringType(), False),
        StructField("raw_signal", ArrayType(FloatType()), False)
    ])

    # Process all .mat files
    rows = []
    for subject_idx in range(1, 35):
        filename = f"eeg_record{subject_idx}.mat"
        states = extract_eeg_data(filename)
        
        channel_map = {
            'AF3': 3, 'F7': 4, 'F3': 5, 'FC5': 6, 'T7': 7, 
            'P7': 8, 'O1': 9, 'O2': 10, 'P8': 11, 'T8': 12, 
            'FC6': 13, 'F4': 14, 'F8': 15, 'AF4': 16
        }
        
        for ch_name, ch_idx in channel_map.items():
            for state, eeg in states.items():
                rows.append({
                    "subject": f"subject_{subject_idx}",
                    "channel": ch_name,
                    "state": state,
                    "raw_signal": eeg[:, ch_idx].tolist()
                })
    
    return spark.createDataFrame(rows, schema)

def compute_band_powers(signal_array, fs=128):
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    
    if len(signal_array) == 0:
        return {band: 0 for band in bands.keys()}
    
    signal = np.array(signal_array) - np.mean(signal_array)
    freq, psd = signal.periodogram(signal, fs=fs, nfft=256)
    
    powers = {}
    for band_name, (low, high) in bands.items():
        band_psd = psd[(freq >= low) & (freq < high)]
        powers[band_name] = band_psd.mean() if band_psd.size > 0 else 0
    
    return powers

def extract_wavelet_features(signal_array, wavelet='db4', level=4):
    if len(signal_array) == 0:
        return {f'wavelet_level_{i}_{stat}': 0.0 
                for i in range(level + 1) 
                for stat in ['energy', 'mean', 'var', 'kurt', 'skew']}
    
    try:
        coeffs = pywt.wavedec(signal_array, wavelet, level=level)
        features = {}
        for i, coeff in enumerate(coeffs):
            features[f'wavelet_level_{i}_energy'] = float(np.sum(coeff ** 2))
            features[f'wavelet_level_{i}_mean'] = float(np.mean(coeff))
            features[f'wavelet_level_{i}_var'] = float(np.var(coeff))
            features[f'wavelet_level_{i}_kurt'] = float(kurtosis(coeff))
            features[f'wavelet_level_{i}_skew'] = float(skew(coeff))
        return features
    except Exception:
        return {f'wavelet_level_{i}_{stat}': 0.0 
                for i in range(level + 1) 
                for stat in ['energy', 'mean', 'var', 'kurt', 'skew']}

def main():
    spark = create_spark_session()
    
    # Load EEG data
    eeg_df = load_mat_files(spark, 'eeg_data')
    
    # Register UDFs
    band_power_udf = udf(compute_band_powers)
    wavelet_features_udf = udf(extract_wavelet_features)
    
    # Apply feature extraction
    processed_df = eeg_df.withColumn(
        "band_powers", 
        band_power_udf(col("raw_signal"))
    ).withColumn(
        "wavelet_features", 
        wavelet_features_udf(col("raw_signal"))
    )
    
    # Flatten the features for further processing
    feature_cols = [
        f"band_powers.{band}" for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']
    ] + [
        f"wavelet_features.wavelet_level_{level}_{stat}" 
        for level in range(5) 
        for stat in ['energy', 'mean', 'var', 'kurt', 'skew']
    ]
    
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    final_df = assembler.transform(processed_df)
    
    # Show results
    final_df.show()
    
    spark.stop()

if __name__ == "__main__":
    main()