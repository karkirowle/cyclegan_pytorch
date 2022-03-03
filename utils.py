import matplotlib.pyplot as plt
import librosa
import numpy as np
import os
import pyworld
from scipy.interpolate import interp1d
from sklearn import preprocessing
import pycwt as wavelet


def get_continuos_f0(f0):
    """CONVERT F0 TO CONTINUOUS F0

    Args:
        f0 (ndarray): original f0 sequence with the shape (T)

    Return:
        (ndarray): continuous f0 with the shape (T)
    """
    # get uv information as binary
    uv = np.float32(f0 != 0)

    # get start and end of f0
    if (f0 == 0).all():
        print("all of the f0 values are 0.")
        return uv, f0
    start_f0 = f0[f0 != 0][0]
    end_f0 = f0[f0 != 0][-1]

    # padding start and end of f0 sequence
    start_idx = np.where(f0 == start_f0)[0][0]
    end_idx = np.where(f0 == end_f0)[0][-1]
    f0[:start_idx] = start_f0
    f0[end_idx:] = end_f0

    # get non-zero frame index
    nz_frames = np.where(f0 != 0)[0]

    # perform linear interpolation
    f = interp1d(nz_frames, f0[nz_frames])
    cont_f0 = f(np.arange(0, f0.shape[0]))

    return uv, cont_f0


def get_continous_log_f0(f0, frame_period=5.0):
    uv, continous_f0 = get_continuos_f0(f0)
    # cont_f0_lpf = low_pass_filter(cont_f0_lpf, int(1.0 / (frame_period * 0.001)), cutoff=20)
    return uv, np.log(continous_f0)


def get_log_f0_cwt(log_f0):
    wavelet_ = wavelet.MexicanHat()
    dt = 0.005
    dj = 1
    s0 = dt*2
    J = 9
    # C_delta = 3.541
    w, sj, freqs, coi, fft, fftfreqs = wavelet.cwt(
        np.squeeze(log_f0), dt, dj, s0, J, wavelet_)
    w = np.real(w).T
   # Wavelet_le = np.real(Wavelet_le).T   # (T, D=10)
  # 0lf0_le_cwt = np.concatenate((Wavelet_lf0, Wavelet_le), -1)
  #  iwave = wavelet.icwt(np.squeeze(lf0), scales, dt, dj, mother) * std
    return w, sj


def inverse_cwt(W, sj):
    log_f0_rec = np.zeros([W.shape[0], len(sj)])
    for i in range(0, len(sj)):
        log_f0_rec[:, i] = W[:, i]*((i+1+2.5)**(-2.5))
    log_f0_rec_sum = np.sum(log_f0_rec, axis=1)
    log_f0_rec_sum = preprocessing.scale(log_f0_rec_sum)
    return log_f0_rec_sum


def norm_scale(W):
    W_norm = np.zeros((W.shape[0], W.shape[1]))
    mean = np.zeros((1, W.shape[1]))  # [1,10]
    std = np.zeros((1, W.shape[1]))
    for scale in range(W.shape[1]):
        mean[:, scale] = W[:, scale].mean()
        std[:, scale] = W[:, scale].std()
        W_norm[:, scale] = (
            W[:, scale]-mean[:, scale])/std[:, scale]

    return W_norm, mean, std


def compute_log_f0_cwt_norm(f0, mean, std):
    uv, f0_log = get_continous_log_f0(f0)
    f0_log_norm = (f0_log - mean) / std

    f0_log_cwt, scales = get_log_f0_cwt(f0_log_norm)  # [560,10]
    f0_log_cwt_norm, mean_scale, std_scale = norm_scale(
        f0_log_cwt)  # [560,10],[1,10],[1,10]

    # f0_log_cwt_norm.append(f0_log_cwt_norm)

    return f0_log_cwt_norm, uv, scales, mean_scale, std_scale


def get_log_f0_cwt_norm(f0s, mean, std):

    uvs = list()
    cont_lf0_lpfs = list()
    cont_lf0_lpf_norms = list()
    Wavelet_lf0s = list()
    Wavelet_lf0s_norm = list()
    scaless = list()

    means = list()
    stds = list()
    for f0 in f0s:

        uv, cont_lf0_lpf = get_continous_log_f0(f0)
        cont_lf0_lpf_norm = (cont_lf0_lpf - mean) / std

        Wavelet_lf0, scales = get_log_f0_cwt(cont_lf0_lpf_norm)  # [560,10]
        Wavelet_lf0_norm, mean_scale, std_scale = norm_scale(
            Wavelet_lf0)  # [560,10],[1,10],[1,10]

        Wavelet_lf0s_norm.append(Wavelet_lf0_norm)
        uvs.append(uv)
        cont_lf0_lpfs.append(cont_lf0_lpf)
        cont_lf0_lpf_norms.append(cont_lf0_lpf_norm)
        Wavelet_lf0s.append(Wavelet_lf0)
        scaless.append(scales)
        means.append(mean_scale)
        stds.append(std_scale)

    return Wavelet_lf0s_norm, scaless, means, stds


def denormalize(w_norm, mean, std):
    w = np.zeros(
        (w_norm.shape[0], w_norm.shape[1]))
    for scale in range(w_norm.shape[1]):
        w[:, scale] = w_norm[:, scale]*std[:, scale]+mean[:, scale]
    return w


def load_wavs(wav_dir, sr):
    wavs = list()
    for file in os.listdir(wav_dir):
        file_path = os.path.join(wav_dir, file)
        wav, _ = librosa.load(file_path, sr=sr, mono=True)
        # wav = wav.astype(np.float64)
        wavs.append(wav)

    return wavs


def world_decompose(wav, fs, frame_period=5.0):
    # Decompose speech signal into f0, spectral envelope and aperiodicity using WORLD
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(
        wav, fs, frame_period=frame_period, f0_floor=71.0, f0_ceil=800.0)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap = pyworld.d4c(wav, f0, timeaxis, fs)

    return f0, timeaxis, sp, ap


def world_encode_spectral_envelop(sp, fs, dim=24):
    # Get Mel-cepstral coefficients (MCEPs)

    # sp = sp.astype(np.float64)
    coded_sp = pyworld.code_spectral_envelope(sp, fs, dim)

    return coded_sp


def world_decode_spectral_envelop(coded_sp, fs):
    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    # coded_sp = coded_sp.astype(np.float32)
    # coded_sp = np.ascontiguousarray(coded_sp)
    decoded_sp = pyworld.decode_spectral_envelope(coded_sp, fs, fftlen)

    return decoded_sp


def world_encode_data(wavs, fs, frame_period=5.0, coded_dim=24):
    f0s = list()
    timeaxes = list()
    sps = list()
    aps = list()
    coded_sps = list()

    for wav in wavs:
        f0, timeaxis, sp, ap = world_decompose(
            wav=wav, fs=fs, frame_period=frame_period)
        coded_sp = world_encode_spectral_envelop(
            sp=sp, fs=fs, dim=coded_dim)
        f0s.append(f0)
        timeaxes.append(timeaxis)
        sps.append(sp)
        aps.append(ap)
        coded_sps.append(coded_sp)

    return f0s, timeaxes, sps, aps, coded_sps


def transpose_in_list(lst):
    transposed_lst = list()
    for array in lst:
        transposed_lst.append(array.T)
    return transposed_lst


def world_decode_data(coded_sps, fs):
    decoded_sps = list()

    for coded_sp in coded_sps:
        decoded_sp = world_decode_spectral_envelop(coded_sp, fs)
        decoded_sps.append(decoded_sp)

    return decoded_sps


def world_speech_synthesis(f0, decoded_sp, ap, fs, frame_period):
    # decoded_sp = decoded_sp.astype(np.float64)
    wav = pyworld.synthesize(f0, decoded_sp, ap, fs, frame_period)
    # Librosa could not save wav if not doing so
    wav = wav.astype(np.float32)

    return wav


def world_synthesis_data(f0s, decoded_sps, aps, fs, frame_period):
    wavs = list()

    for f0, decoded_sp, ap in zip(f0s, decoded_sps, aps):
        wav = world_speech_synthesis(f0, decoded_sp, ap, fs, frame_period)
        wavs.append(wav)

    return wavs


def coded_sps_normalization_fit_transoform(coded_sps):
    coded_sps_concatenated = np.concatenate(coded_sps, axis=1)
    coded_sps_mean = np.mean(coded_sps_concatenated, axis=1, keepdims=True)
    coded_sps_std = np.std(coded_sps_concatenated, axis=1, keepdims=True)

    coded_sps_normalized = list()
    for coded_sp in coded_sps:
        coded_sps_normalized.append(
            (coded_sp - coded_sps_mean) / coded_sps_std)

    return coded_sps_normalized, coded_sps_mean, coded_sps_std


def coded_sps_normalization_transoform(coded_sps, coded_sps_mean, coded_sps_std):
    coded_sps_normalized = list()
    for coded_sp in coded_sps:
        coded_sps_normalized.append(
            (coded_sp - coded_sps_mean) / coded_sps_std)

    return coded_sps_normalized


def coded_sps_normalization_inverse_transoform(normalized_coded_sps, coded_sps_mean, coded_sps_std):
    coded_sps = list()
    for normalized_coded_sp in normalized_coded_sps:
        coded_sps.append(normalized_coded_sp * coded_sps_std + coded_sps_mean)

    return coded_sps


def coded_sp_padding(coded_sp, multiple=4):
    num_features = coded_sp.shape[0]
    num_frames = coded_sp.shape[1]
    num_frames_padded = int(np.ceil(num_frames / multiple)) * multiple
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    coded_sp_padded = np.pad(
        coded_sp, ((0, 0), (num_pad_left, num_pad_right)), 'constant', constant_values=0)

    return coded_sp_padded


def wav_padding(wav, sr, frame_period, multiple=4):
    assert wav.ndim == 1
    num_frames = len(wav)
    num_frames_padded = int(
        (np.ceil((np.floor(num_frames / (sr * frame_period / 1000)) + 1) / multiple + 1) * multiple - 1) * (
            sr * frame_period / 1000))
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    wav_padded = np.pad(wav, (num_pad_left, num_pad_right),
                        'constant', constant_values=0)

    return wav_padded


def wp_padding(wp, multiple=4):
    num_frames = len(wp)
    num_frames_padded = int(np.ceil(num_frames / multiple)) * multiple
    num_frames_diff = num_frames_padded - num_frames
    return np.pad(wp, ((num_frames_diff // 2, num_frames_diff - (num_frames_diff // 2)), (0, 0)), 'edge')


def logf0_statistics(f0s):
    log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
    log_f0s_mean = log_f0s_concatenated.mean()
    log_f0s_std = log_f0s_concatenated.std()

    return log_f0s_mean, log_f0s_std


def pitch_conversion_with_logf0(logf0, mean_log_src, std_log_src, mean_log_target, std_log_target):
    """Because we saved the masked values we need this more complicated version with logical indexing"""

    f0_converted = np.zeros_like(logf0)
    f0_converted[logf0._mask == False] = np.exp((logf0[logf0._mask == False] -
                                                 mean_log_src) / std_log_src * std_log_target + mean_log_target)
    return f0_converted


def pitch_conversion(f0, mean_log_src, std_log_src, mean_log_target, std_log_target):
    # Logarithm Gaussian normalization for Pitch Conversions
    f0_converted = np.exp((np.ma.log(f0) - mean_log_src) /
                          std_log_src * std_log_target + mean_log_target)

    return f0_converted


def wavs_to_specs(wavs, n_fft=1024, hop_length=None):
    stfts = list()
    for wav in wavs:
        stft = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)
        stfts.append(stft)

    return stfts


def wavs_to_mfccs(wavs, sr, n_fft=1024, hop_length=None, n_mels=128, n_mfcc=24):
    mfccs = list()
    for wav in wavs:
        mfcc = librosa.feature.mfcc(
            y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc)
        mfccs.append(mfcc)

    return mfccs


def mfccs_normalization(mfccs):
    mfccs_concatenated = np.concatenate(mfccs, axis=1)
    mfccs_mean = np.mean(mfccs_concatenated, axis=1, keepdims=True)
    mfccs_std = np.std(mfccs_concatenated, axis=1, keepdims=True)

    mfccs_normalized = list()
    for mfcc in mfccs:
        mfccs_normalized.append((mfcc - mfccs_mean) / mfccs_std)

    return mfccs_normalized, mfccs_mean, mfccs_std


def sample_train_data(dataset_A, dataset_B, n_frames=128):
    num_samples = min(len(dataset_A), len(dataset_B))
    train_data_A_idx = np.arange(len(dataset_A))
    train_data_B_idx = np.arange(len(dataset_B))
    np.random.shuffle(train_data_A_idx)
    np.random.shuffle(train_data_B_idx)
    train_data_A_idx_subset = train_data_A_idx[:num_samples]
    train_data_B_idx_subset = train_data_B_idx[:num_samples]

    train_data_A = list()
    train_data_B = list()

    for idx_A, idx_B in zip(train_data_A_idx_subset, train_data_B_idx_subset):
        data_A = dataset_A[idx_A]
        frames_A_total = data_A.shape[1]
        assert frames_A_total >= n_frames
        start_A = np.random.randint(frames_A_total - n_frames + 1)
        end_A = start_A + n_frames
        train_data_A.append(data_A[:, start_A:end_A])

        data_B = dataset_B[idx_B]
        frames_B_total = data_B.shape[1]
        assert frames_B_total >= n_frames
        start_B = np.random.randint(frames_B_total - n_frames + 1)
        end_B = start_B + n_frames
        train_data_B.append(data_B[:, start_B:end_B])

    train_data_A = np.array(train_data_A)
    train_data_B = np.array(train_data_B)
