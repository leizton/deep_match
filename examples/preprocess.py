import random
import numpy as np
from keras.preprocessing.sequence import pad_sequences


def gen_data_set(data, negsample=5):
    negsample = max(1, negsample)
    data.sort_values('timestamp', inplace=True)
    item_ids = data['movie_id'].unique()
    train_set, test_set = [], []
    for user_id, user_history in data.groupby('user_id'):
        pos_list = user_history['movie_id'].tolist()
        rating_list = user_history['rating'].tolist()
        if len(pos_list) < 30:
            continue
        candidate_set = list(set(item_ids) - set(pos_list))
        neg_list = np.random.choice(candidate_set, size=len(pos_list)*negsample, replace=True)
        #
        test_set_start_index = max(1, int(len(pos_list) * 0.9))
        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            # 第1个是正样本, 跟着negsample个负样本
            sample_list = [pos_list[i]]
            sample_list += [neg_list[item_idx] for item_idx in np.random.choice(neg_list, negsample)]
            sample = (user_id, hist[::-1], sample_list, 0, len(hist[::-1]), rating_list[i])
            #
            if i < test_set_start_index:
                train_set += [sample]
            else:
                test_set += [sample]
    random.shuffle(train_set)
    random.shuffle(test_set)
    print('train_size=%s, test_size=%s' % (len(train_set), len(test_set)))
    return train_set,test_set


def gen_model_input(train_set, user_profile, seq_max_len):
    train_uid = np.array([line[0] for line in train_set])
    train_seq = [line[1] for line in train_set]
    train_itemid = np.array([line[2] for line in train_set])
    train_label = np.array([line[3] for line in train_set])
    train_hist_len = np.array([line[4] for line in train_set])
    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    train_model_input = {
        'user_id': train_uid, 'movie_id': train_itemid,
        'hist_movie_id': train_seq_pad, 'hist_len': train_hist_len
    }
    for key in ['gender', 'age', 'occupation', 'zip']:
        train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values
    return train_model_input, train_label

def gen_model_input_sdm(train_set, user_profile, seq_short_len, seq_prefer_len):
    train_uid = np.array([line[0] for line in train_set])
    short_train_seq = [line[1] for line in train_set]
    prefer_train_seq = [line[2] for line in train_set]
    train_iid = np.array([line[3] for line in train_set])
    train_label = np.array([line[4] for line in train_set])
    train_short_len = np.array([line[5] for line in train_set])
    train_prefer_len = np.array([line[6] for line in train_set])
    short_train_seq_genres = np.array([line[8] for line in train_set])
    prefer_train_seq_genres = np.array([line[9] for line in train_set])
    train_short_item_pad = pad_sequences(short_train_seq, maxlen=seq_short_len, padding='post', truncating='post', value=0)
    train_prefer_item_pad = pad_sequences(prefer_train_seq, maxlen=seq_prefer_len, padding='post', truncating='post', value=0)
    train_short_genres_pad = pad_sequences(short_train_seq_genres, maxlen=seq_short_len, padding='post', truncating='post', value=0)
    train_prefer_genres_pad = pad_sequences(prefer_train_seq_genres, maxlen=seq_prefer_len, padding='post', truncating='post', value=0)
    train_model_input = {
        'user_id': train_uid, 'movie_id': train_iid, 'short_movie_id': train_short_item_pad,
        'prefer_movie_id': train_prefer_item_pad, 'prefer_sess_length': train_prefer_len,
        'short_sess_length': train_short_len, 'short_genres': train_short_genres_pad,
        'prefer_genres': train_prefer_genres_pad
    }
    for key in ['gender', 'age', 'occupation', 'zip']:
        train_model_input[key] = user_profile.loc[train_model_input['user_id']][key].values
    return train_model_input, train_label
