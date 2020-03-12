import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import joblib

def test_train():
    data = pd.read_csv('data/features/features_behavior_count_balanced.csv')
    features = data[['num_views', 'num_favorites', 'num_add2carts']]
    label = data['buyOrNot']
    print(features)
    print(label)

    features = preprocessing.scale(features)

    train_X, test_X, train_y, test_y = train_test_split(features, label, random_state=0)

    model = SVC(C=1, kernel='linear', probability=True)

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

    predictions = model.predict_proba(features)
    print(list(predictions)[:10])
    predictions_df = pd.DataFrame({
        'user_id': data['user_id'],
        'item_category': data['item_category'],
        'predictions': predictions[:, 1] > 0.5,
        'predict_prob': predictions[:, 1]
    })
    predictions_df.to_csv('data/test_predictions.csv', index=False)

def train_behavior_prediction(data_path, model_name, model_path):
    """
    :type data_path: str
    :type model_name: str
    :type model_path: str
    """
    
    data = pd.read_csv(data_path)
    features = data[['num_views', 'num_favorites', 'num_add2carts']]
    label = data['buyOrNot']
    print(features)
    print(label)

    features = preprocessing.scale(features)

    train_X, test_X, train_y, test_y = train_test_split(features, label, random_state=0)

    model = SVC(C=1, kernel='linear', probability=True)

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

    joblib.dump(model, model_path)

def predict_behavior(data_path, model_path, predict_path):
    """
    :type data_path: str
    :type model_path: str
    :type predict_path: str
    """
    
    data = pd.read_csv(data_path)
    features = data[['num_views', 'num_favorites', 'num_add2carts']]
    label = data['buyOrNot']
    features = preprocessing.scale(features)

    model = joblib.load(model_path)

    predictions = model.predict_proba(features)
    # TODO output predictions to a csv file
    predictions_df = pd.DataFrame({
        'user_id': data['user_id'],
        'item_category': data['item_category'],
        'predictions': predictions[:, 1] > 0.5,
        'predict_prob': predictions[:, 1]
    })
    predictions_df.to_csv(predict_path, index=False)

def item_rank(data_path):
    """
    :type data_path: str
    """
    data = pd.read_csv(data_path)
    # TODO rank items according to their ratings
    # DataFrame.apply: https://blog.csdn.net/Evan_Blog/article/details/82787984
    data = data.groupby(['item_category'], sort=False).apply(lambda x: x.sort_values('rating', ascending=False)).reset_index(drop=True)
    # data = data.sort_values(['item_category', 'rating'], ascending=False)
    print(data)

def predict_item(data_path):
    """
    :type data_path: str
    """
    # TODO predict item using the ranking of its category
    pass

if __name__ == "__main__":
    # test_train()
    test_predict()
    # item_rank('data/features/features_item_rating.csv')