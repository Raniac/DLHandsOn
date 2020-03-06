import pandas as pd

def extract_behavior_counts():
    users = pd.read_csv('data/tianchi_fresh_comp_train_user.csv').iloc[:1000]
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

    features_df.to_csv('data/features/features_test.csv')

if __name__ == "__main__":
    extract_behavior_counts()
