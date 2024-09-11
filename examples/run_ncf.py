import sys
sys.path.append("../")

import pandas as pd
from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat
from preprocess import gen_data_set, gen_model_input
from sklearn.preprocessing import LabelEncoder
from deepmatch_torch.models import FM, NCF, DSSM

if __name__ == "__main__":

    data = pd.read_csvdata = pd.read_csv("./movielens_sample.txt")
    sparse_features = ["movie_id", "user_id",
                       "gender", "age", "occupation", "zip", ]
    SEQ_LEN = 50
    negsample = 3

    # 1.Label Encoding for sparse features,and process sequence features with `gen_date_set` and `gen_model_input`

    features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip']
    feature_max_idx = {}
    for feature in features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1

    user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates('user_id')

    item_profile = data[["movie_id"]].drop_duplicates('movie_id')

    user_profile.set_index("user_id", inplace=True)

    user_item_list = data.groupby("user_id")['movie_id'].apply(list)
    
    train_set, test_set = gen_data_set(data, negsample)

    train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN)
    test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN)
    
    # 2.count #unique features for each sparse field and generate feature config for sequence feature
    embedding_dim = 8

    # user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], embedding_dim),
    #                         SparseFeat("gender", feature_max_idx['gender'], embedding_dim),
    #                         SparseFeat("age", feature_max_idx['age'], embedding_dim),
    #                         SparseFeat("occupation", feature_max_idx['occupation'], embedding_dim),
    #                         SparseFeat("zip", feature_max_idx['zip'], embedding_dim),
    #                         VarLenSparseFeat(SparseFeat('hist_movie_id', feature_max_idx['movie_id'], embedding_dim,
    #                                                     embedding_name="movie_id"), SEQ_LEN, 'mean', 'hist_len'),
    #                         ]
    user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], embedding_dim)
                            ]

    item_feature_columns = [SparseFeat('movie_id', feature_max_idx['movie_id'], embedding_dim)]

    # 3.Define Model and train

    model = NCF(
        user_feature_columns, 
        item_feature_columns,  
        user_gmf_embedding_dim=20,
        item_gmf_embedding_dim=20, 
        user_mlp_embedding_dim=32, 
        item_mlp_embedding_dim=32,
        dnn_hidden_units=[128, 64, 32],
        optimizer='Adam',
        config={
            'device': 'cpu'
            }
    )  

    model.fit(train_model_input, train_label, 
                        max_epochs=10, batch_size=128 )
    # model.fit(train_model_input, train_label,  # train_label,
                        # batch_size=256*3, epochs=1, verbose=1, validation_split=0.0, )
    # print(model.item_embedding.shape)
    # print(model.user_embedding.shape)
    # # 4. Generate user features for testing and full item features for retrieval
    # test_user_model_input = test_model_input
    # all_item_model_input = {"movie_id": item_profile['movie_id'].values}

    # # user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    # # item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)

    # user_embs = model.predict(test_user_model_input, batch_size=2 ** 12)
    # item_embs = model.predict(all_item_model_input, batch_size=2 ** 12)

    # print(user_embs.shape)
    # print(item_embs.shape)



# ncf

# exit(1)

# if __name__ == "__main__":
#     data = pd.read_csvdata = pd.read_csv("./examples/movielens_sample.txt")
#     sparse_features = ["movie_id", "user_id",
#                        "gender", "age", "occupation", "zip", ]
#     SEQ_LEN = 50
#     negsample = 3

#     # 1.Label Encoding for sparse features,and process sequence features with `gen_date_set` and `gen_model_input`

#     features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip']
#     feature_max_idx = {}
#     for feature in features:
#         lbe = LabelEncoder()
#         data[feature] = lbe.fit_transform(data[feature]) + 1
#         feature_max_idx[feature] = data[feature].max() + 1

#     user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates('user_id')

#     item_profile = data[["movie_id"]].drop_duplicates('movie_id')

#     user_profile.set_index("user_id", inplace=True)

#     user_item_list = data.groupby("user_id")['movie_id'].apply(list)

#     train_set, test_set = gen_data_set(data, negsample)

#     train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN)
#     test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN)

#     # 2.count #unique features for each sparse field and generate feature config for sequence feature

#     user_feature_columns = {"user_id": feature_max_idx['user_id'], 'gender': feature_max_idx['gender'],
#                             "age": feature_max_idx['age'],
#                             "occupation": feature_max_idx["occupation"], "zip": feature_max_idx["zip"]}

#     item_feature_columns = {"movie_id": feature_max_idx['movie_id']}

#     # 3.Define Model,train,predict and evaluate
#     model = NCF(user_feature_columns, item_feature_columns, user_gmf_embedding_dim=20,
#                 item_gmf_embedding_dim=20, user_mlp_embedding_dim=32, item_mlp_embedding_dim=32,
#                 dnn_hidden_units=[128, 64, 32], )
#     model.summary()
#     model.compile("adam", "binary_crossentropy",
#                   metrics=['binary_crossentropy'], )

#     history = model.fit(train_model_input, train_label,
#                         batch_size=64, epochs=20, verbose=2, validation_split=0.2, )
#     pred_ans = model.predict(test_model_input, batch_size=64)