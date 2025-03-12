import os
import numpy as np
from preprocess import frame2sample, preprocess, read_wavefile, read_person_dir
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle

def eval_wavefile(dir_list):
    x = []
    y = []
    score_list = []
    model:GaussianNB = pickle.load(open('model/model.pkl', 'rb'))
    for person in dir_list:
        wav_files = os.listdir(person)
        for wav_file in wav_files:
            wav_file_path = os.path.join(person, wav_file)
            data, lebel = preprocess(wav_file_path)
            pred_y = model.predict(data)
            score_list.append(score(pred_y, lebel))
    
    x = np.array(x)
    y = np.array(y)

    recogRate = sum(score_list) / len(score_list)
    print(f"Recognition Rate: {recogRate * 100}%")
    return x, y

def score(pred_y, y):
    try:
        pred_start = frame2sample(np.where(pred_y == 1)[0][0]).reshape(-1)[0]
    except:
        pred_start = 0
    try:
        pred_end = frame2sample(np.where(pred_y == 1)[0][-1]).reshape(-1)[0]
    except:
        pred_end = len(pred_y)

    label_start = frame2sample(np.where(y == 1)[0][0]).reshape(-1)[0]
    label_end = frame2sample(np.where(y == 1)[0][-1]).reshape(-1)[0]

    fs = 16000  # assuming 16kHz sampling rate
    time_diff = 0.0625  # 200ms threshold

    score = np.mean(np.abs(np.array([pred_start, pred_end]) - np.array([label_start, label_end])) < fs * time_diff)
    return score

def dev_model(test_x):
    model:GaussianNB = pickle.load(open('model/model.pkl', 'rb'))
    score_list = []
    for wav_path in test_x:
        wav_files = os.listdir(wav_path)
        for wav_file in wav_files:
            wav_file_path = os.path.join(wav_path, wav_file)
            data, label = preprocess(wav_file_path)
            pred_y = model.predict(data)

            pred_start = frame2sample(np.where(pred_y == 1)[0][0]).reshape(-1)[0]
            pred_end = frame2sample(np.where(pred_y == 1)[0][-1]).reshape(-1)[0]
            label_start = frame2sample(np.where(label == 1)[0][0]).reshape(-1)[0]
            label_end = frame2sample(np.where(label == 1)[0][-1]).reshape(-1)[0]


if __name__ == '__main__':
    eval_wavefile('waveFile_2008')
