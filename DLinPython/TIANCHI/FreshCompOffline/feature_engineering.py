import pandas as pd
import random

def extract_behavior_counts():
    users = pd.read_csv('data/tianchi_fresh_comp_train_user.csv')
    users_group = users.drop('user_geohash', axis=1).groupby(['user_id', 'item_category'])
    features_df = pd.DataFrame(columns=['user_id', 'buyOrNot', 'num_views', 'num_favorites', 'num_add2carts'])

    for user in users_group:
        behavior_counts = user[1]['behavior_type'].value_counts()
        num_views = behavior_counts.get(1, 0)
        num_favorites = behavior_counts.get(2, 0)
        num_add2carts = behavior_counts.get(3, 0)
        buyOrNot = 1 if behavior_counts.get(4, 0) else 0

        new_row = [{'user_id': user[0][0], 'buyOrNot': buyOrNot, 'num_views': num_views, 'num_favorites': num_favorites, 'num_add2carts': num_add2carts}]
        features_df = features_df.append(new_row, ignore_index=True)

    features_df.to_csv('data/features/features_behavior_count.csv')

def data_balancing():
    data_file = 'data/features/features_test.csv'
    data_df = pd.read_csv(data_file)
    pos_samples = data_df.loc[data_df['buyOrNot'] == 1]
    neg_samples = data_df.loc[data_df['buyOrNot'] == 0]
    num_pos = pos_samples.shape[0]
    print('Number of Positive Samples: %d' % num_pos)
    num_neg = neg_samples.shape[0]
    print('Number of Negative Samples: %d' % num_neg)

    if num_pos < num_neg:
        # do data balancing
        neg_samples_balanced = neg_samples.sample(n=num_pos)
        data_balanced = pd.concat([pos_samples, neg_samples_balanced], axis=0)
        data_balanced.to_csv('data/features/features_test_balanced.csv')
    
    return

if __name__ == "__main__":
    # extract_behavior_counts()
    data_balancing()
