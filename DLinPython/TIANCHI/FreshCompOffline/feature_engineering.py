import pandas as pd
import random
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s')

def extract_behavior_counts():
    logging.info('Extraction begins.')
    users = pd.read_csv('data/tianchi_fresh_comp_train_user.csv')
    users_group = users.drop('user_geohash', axis=1).groupby(['user_id', 'item_category'])
    features_df = pd.DataFrame(columns=['user_id', 'item_category', 'buyOrNot', 'num_views', 'num_favorites', 'num_add2carts'])
    logging.info('Data loaded.')

    count = 0
    for user in users_group:
        behavior_counts = user[1]['behavior_type'].value_counts()
        num_views = behavior_counts.get(1, 0)
        num_favorites = behavior_counts.get(2, 0)
        num_add2carts = behavior_counts.get(3, 0)
        buyOrNot = 1 if behavior_counts.get(4, 0) else 0

        new_row = [{'user_id': user[0][0], 'item_category': user[0][1], 'buyOrNot': buyOrNot, 'num_views': num_views, 'num_favorites': num_favorites, 'num_add2carts': num_add2carts}]
        features_df = features_df.append(new_row, ignore_index=True)
        count += 1

        if count % 100 == 0:
            logging.info('Extracted %d features.' % count)
            if count % 1e5 == 0:
                features_df.to_csv('data/features/features_behavior_counts_all_' + str(count) + '.csv', index=False)
                features_df = pd.DataFrame(columns=['user_id', 'item_category', 'buyOrNot', 'num_views', 'num_favorites', 'num_add2carts'])                

    logging.info('All done.')

def data_balancing():
    data_file = 'data/features/features_behavior_count.csv'
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
        data_balanced.to_csv('data/features/features_behavior_count_balanced.csv', index=False)
    
    return

def extract_item_rating():
    logging.info('Extraction begins.')
    users = pd.read_csv('data/tianchi_fresh_comp_train_user.csv')
    items_group = users.drop('user_geohash', axis=1).groupby(['item_category', 'item_id'])
    features_df = pd.DataFrame(columns=['item_category', 'item_id', 'rating'])
    logging.info('Data loaded.')

    count = 0
    for item in items_group:
        behavior_counts = item[1]['behavior_type'].value_counts()
        num_views = behavior_counts.get(1, 0)
        num_favorites = behavior_counts.get(2, 0)
        num_add2carts = behavior_counts.get(3, 0)
        num_buys = behavior_counts.get(4, 0)
        rating = num_views * 0.1 + num_add2carts + num_favorites + num_buys * 5

        new_row = [{'item_category': item[0][0], 'item_id': item[0][1], 'rating': rating}]
        features_df = features_df.append(new_row, ignore_index=True)
        count += 1

        if count % 100 == 0:
            logging.info('Extracted %d features.' % count)
            if count == 100:
                break

    features_df.to_csv('data/features/features_item_rating.csv', index=False)

if __name__ == "__main__":
    extract_behavior_counts()
    # extract_item_rating()
    # data_balancing()
