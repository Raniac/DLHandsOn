import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s')

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
        'prediction': predictions[:, 1] > 0.5,
        'predict_proba': predictions[:, 1]
    })
    predictions_df.to_csv('data/test_predictions.csv', index=False)

def train_behavior_prediction(data_paths, model_name, model_path):
    """
    :type data_path: str
    :type model_name: str
    :type model_path: str
    """
    logging.info('Training begins.')

    model = SVC(C=1, kernel='linear', probability=True)
    
    for i in range(len(data_paths)):
        logging.info('Training on data {0:02d}'.format(i+1))
        data = pd.read_csv(data_paths[i])
        features = data[['num_views', 'num_favorites', 'num_add2carts']]
        label = data['buyOrNot']
        logging.info('Data loaded.')

        features = preprocessing.scale(features)

        train_X, test_X, train_y, test_y = train_test_split(features, label, random_state=0)

        model.fit(train_X, train_y)
        predictions = model.predict(test_X)

        tn, fp, fn, tp = confusion_matrix(test_y, predictions).ravel()
        logging.info('The confusion matrix is: %d TN, %d FP, %d FN, %d TP' % (tn, fp, fn, tp))
        cnf_accuracy = (tn + tp) / (tn + fp + fn + tp)
        logging.info('The accuracy is: %.2f' % cnf_accuracy)
        cnf_sensitivity = tp / (tp + fn)
        logging.info('The sensitivity is: %.2f' % cnf_sensitivity)
        cnf_specificity = tn / (tn + fp)
        logging.info('The specificity is: %.2f' % cnf_specificity)

    joblib.dump(model, model_path)
    logging.info('Done. Model saved in %s' % model_path)

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
    predictions_df = pd.DataFrame({
        'user_id': data['user_id'],
        'item_category': data['item_category'],
        'prediction': predictions[:, 1] > 0.5,
        'predict_proba': predictions[:, 1]
    })
    predictions_df.to_csv(predict_path, index=False)

def item_rank(data_path, rank_path):
    """
    :type data_path: str
    """
    data = pd.read_csv(data_path)
    # TODO rank items according to their ratings
    # DataFrame.apply: https://blog.csdn.net/Evan_Blog/article/details/82787984
    # data = data.groupby(['item_category'], sort=False).apply(lambda x: x.sort_values('rating', ascending=False)).reset_index(drop=True)
    data = data.groupby(['item_category'], sort=False).apply(lambda x: x[x.rating == x.rating.max()]).reset_index(drop=True)
    data.to_csv('data/features/features_item_rank.csv', index=False)

def predict_item(predict_path, rank_path):
    """
    :type data_path: str
    """
    # TODO predict item using the ranking of its category
    predictions = pd.read_csv(predict_path)
    predictions = predictions[predictions['prediction'] == True].groupby(['user_id'], sort=False).apply(lambda x: x[x.predict_proba == x.predict_proba.max()]).reset_index(drop=True)
    print(predictions)
    ranking = pd.read_csv(rank_path)
    data = predictions[['user_id', 'item_category']].join(ranking.set_index('item_category'), on='item_category', how='inner')
    print(data[['user_id', 'item_id']])

if __name__ == "__main__":
    # test_train()
    # test_predict()
    # item_rank('data/features/features_item_rating.csv')
    predict_item('data/test_predictions.csv', 'data/features/test/features_item_rank.csv')

    # data_paths = []
    # for i in range(1, 18):
    #     data_paths.append('data/features/behavior_counts_all/features_behavior_counts_all_{0:03d}_balanced.csv'.format(i))
    #     print(data_paths)
    # train_behavior_prediction(data_paths, '', 'models/exp_200315.model')