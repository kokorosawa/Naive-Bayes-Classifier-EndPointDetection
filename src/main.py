from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from loguru import logger
import pickle
from preprocess import read_wavefile

logger.add('logs/main.log')

logger.info('loading data...')
x, y = read_wavefile('wavefiles-all')

logger.info('splitting data...')
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

model = GaussianNB()
logger.info('training model...')
model.fit(train_x, train_y)
pickle.dump(model, open('model/model.pkl', 'wb'))

pred_y = model.predict(test_x)
logger.info('evaluating model...')
accuracy = accuracy_score(test_y, pred_y)

print('Accuracy: ', accuracy*100, '%')