import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from loguru import logger
import pickle
from eval import dev_model, eval_wavefile
from preprocess import read_wavefile,read_person_dir

logger.add('logs/main.log')

logger.info('loading data...')
train, test = read_wavefile('waveFile_2008', split_ratio=0.2)

train_x, train_y = read_person_dir(train)
test_x, test_y = read_person_dir(test)

model = GaussianNB()
logger.info('training model...')
model.fit(train_x, train_y)
pickle.dump(model, open('model/model.pkl', 'wb'))

pred_y = model.predict(test_x)
logger.info('evaluating model...')
accuracy = accuracy_score(test_y, pred_y)
eval_wavefile(test)

print('Accuracy: ', accuracy*100, '%')
dir = 'waveFile_2008'
eval_list = [os.path.join(dir,i) for i in os.listdir(dir)]
eval_wavefile(eval_list)