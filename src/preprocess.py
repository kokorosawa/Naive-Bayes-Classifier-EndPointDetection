import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import os
import librosa
from tqdm import tqdm

frameSize = 1024
overlap = 192
hop = frameSize - overlap

def preprocess(audio_path):
    rate, data = wavfile.read(audio_path)
    wav_name = audio_path.split('/')[-1].split('.')[0]
    start,end = int(wav_name.split('_')[1]) , int(wav_name.split('_')[2])

    data = data / (np.max(np.abs(data)))
    data = data - np.mean(data)
    data = enframe(data, frameSize, overlap)
    label = np.int8(np.zeros(data.shape[0]))
    start = sample2frame(start, frameSize, overlap)
    end = sample2frame(end, frameSize, overlap)
    label[start:end] = 1

    # vol_data = frame2volume(data)
    # vol_data = vol_data.reshape(vol_data.shape[0], 1)
    # zcr = zero_crossing_rate(data)
    # zcr = zcr.reshape(zcr.shape[0], 1)

    # mfcc = extract_mfcc(data, rate, frameSize, hop)
    # mfcc = mfcc.reshape(mfcc.shape[0], -1)

    all_data = np.hstack((vol_data, zcr, mfcc))
    return data, label

def frame2volume(data):
    volume = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        volume[i] = np.mean(np.abs(data[i]))
    return volume

def zero_crossing_rate(data):
    num_frames = data.shape[0]
    zcr = np.zeros(num_frames)
    for i in range(num_frames):
        zcr[i] = 0.5 * np.mean(np.abs(np.diff(np.sign(data[i]))))
    return zcr

def sample2frame(samples, frame_len = frameSize, frame_shift = overlap):
    frames = np.array(samples / frame_shift)
    return int(frames)

def frame2sample(frames, frame_len = frameSize, frame_shift = overlap):
    samples = np.array(frames * frame_shift)
    return samples

def enframe(signal, frame_len, frame_shift):
    num_samples = len(signal)
    num_frames = int(np.floor((num_samples - frame_len) / frame_shift) + 1)
    frames = np.zeros((num_frames, frame_len))
    for i in range(num_frames):
        frames[i] = signal[i * frame_shift: i * frame_shift + frame_len]
    return frames

def extract_mfcc(data, sr, frame_len, frame_shift, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc, hop_length=frame_shift, n_fft=frame_len)
    return mfcc

def read_wavefile(audio_path, split_ratio=0.2):
    person_dir = os.listdir(audio_path)
    num_train = int(len(person_dir) * (1 - split_ratio))
    train_person_dir = person_dir[:num_train]
    test_person_dir = person_dir[num_train:]

    train_person_dir = [os.path.join(audio_path, i) for i in train_person_dir]
    test_person_dir = [os.path.join(audio_path, i) for i in test_person_dir]

    return train_person_dir, test_person_dir

def read_person_dir(person_dir):
    x = []
    y = []

    for wav_path in person_dir:
        wav_files = os.listdir(wav_path)
        for wav_file in wav_files:
            wav_file_path = os.path.join(wav_path, wav_file)
            data, lebel = preprocess(wav_file_path)
            x.extend(data)
            y.extend(lebel)

    x = np.array(x)
    y = np.array(y)
    return x, y

    
if __name__ == "__main__":
    audio_path = 'waveFile_2008'
    read_wavefile(audio_path)   