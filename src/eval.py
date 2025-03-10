from preprocess import read_wavefile
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle

def eval(audio_path):
    x, y = read_wavefile(audio_path)
    model : GaussianNB = pickle.load(open('model/model.pkl', 'rb'))
    pred_y = model.predict(x)
    accuracy = accuracy_score(y, pred_y)
    print('Accuracy: ', accuracy*100, '%')
    return accuracy

if __name__ == '__main__':
    eval('/Users/wunianze/course/FCU-SLP/hw2/wavfile')
