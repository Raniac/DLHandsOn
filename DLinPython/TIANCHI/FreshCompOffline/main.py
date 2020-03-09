import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import joblib

def test():
    data = pd.read_csv('data/features/features_behavior_count_balanced.csv')
    features = data[['num_views', 'num_favorites', 'num_add2carts']]
    label = data['buyOrNot']
    print(features)
    print(label)

    features = preprocessing.scale(features)

    train_X, test_X, train_y, test_y = train_test_split(features, label, random_state=0)

    model = SVC(C=1, kernel='linear')

    model.fit(train_X, train_y)
    predictions = model.predict(test_X)

    tn, fp, fn, tp = confusion_matrix(test_y, predictions).ravel()
    print('The confusion matrix is:', (tn, fp, fn, tp))
    cnf_accuracy = (tn + tp) / (tn + fp + fn + tp)
    print('The accuracy is: %.2f' % cnf_accuracy)
    cnf_sensitivity = tp / (tp + fn)
    print('The sensitivity is: %.2f' % cnf_sensitivity)
    cnf_specificity = tn / (tn + fp)
    print('The specificity is: %.2f' % cnf_specificity)

    joblib.dump(model, 'models/test.model')

def test_predict():
    data = pd.read_csv('data/features/features_behavior_count_balanced.csv')
    features = data[['num_views', 'num_favorites', 'num_add2carts']]
    label = data['buyOrNot']
    print(list(label)[:10])
    features = preprocessing.scale(features)

    model = joblib.load('models/test.model')

    predictions = model.predict(features)
    print(list(predictions)[:10])

def train_behavior_prediction(data_path, model):
    pass

def predict_behavior():
    pass

def train_item_prediction():
    pass

def predict_item():
    pass

if __name__ == "__main__":
    # test()
    test_predict()