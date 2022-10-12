import numpy as np
import pandas as pd
import pickle


#詐欺しかしないユーザを除去
def remove_fraud_only_user(train_df):
    train_id = train_df['cc_num'].unique()
    for id_val in train_id:
        tmp_df = train_df[train_df['cc_num'] == id_val]
        fraud_sum = sum(tmp_df['is_fraud'])
        if(fraud_sum== tmp_df.shape[0]):
            train_df = train_df[train_df['cc_num'] != id_val]
    train_id = train_df['cc_num'].unique()
    return train_df, train_id


#訓練データにないユーザーを削除
def remove_test_only_user(test_df, train_id):
    test_id = test_df['cc_num'].unique()
    for id_val in test_id:
        if(id_val not in train_id):
            test_df = test_df[test_df['cc_num'] != id_val]
    test_id = test_df['cc_num'].unique()
    return test_df, test_id


def label_encoding(df, columns=["month", "day", "hour"]):
    out_df = df.copy()

    #カラム追加と初期化
    for i in range(len(columns)):
        out_df.insert(len(out_df.columns), columns[i], 0)

    #数値化
    for c in columns:
        dummy_columns = [d for d in df.columns if c in d]
        undummied = out_df[c]
        for col in dummy_columns:
            undummied[df[col]==1] = int(df[col].name.strip(c+"_"))

    return out_df


#idごとに時間によってsort
def sort_by_time(df, df_id):
    sorted_df = pd.DataFrame(index=[], columns=df.columns)
    sorted_df_list = []
    for i, cc_num in enumerate(df_id):
        sorted_df_list.append(df[df["cc_num"] == cc_num].sort_values(['month', 'day', "hour"]))
        # if i % 10 == 0:
        #     print(i,"番目/",len(df_id))
    sorted_df = pd.concat(sorted_df_list)
    return sorted_df


def preprocess(train, test):

    #ID選定&ID取得
    train, train_id = remove_fraud_only_user(train)
    test, test_id = remove_test_only_user(test, train_id)

    print("id selected & id got!")

    #クレカデータではdayofweekがlabel encodingに邪魔であり、予測に影響しないため排除
    train = train.drop(['dayofweek_0',
       'dayofweek_1', 'dayofweek_2', 'dayofweek_3', 'dayofweek_4',
       'dayofweek_5', 'dayofweek_6'], axis=1)
    test = test.drop(['dayofweek_0',
       'dayofweek_1', 'dayofweek_2', 'dayofweek_3', 'dayofweek_4',
       'dayofweek_5', 'dayofweek_6'], axis=1)

    #データ数が爆発するのでonehotの代わりにlabel encoding
    train = label_encoding(train)
    test = label_encoding(test)

    print("label encoded!")

    train = sort_by_time(train, train_id).loc[:,['cc_num', 'amt', 'merch_lat',
       'merch_long', 'age', 'distance', 'month', "day", "hour",'is_fraud']]
    test = sort_by_time(test, test_id).loc[:,['cc_num', 'amt', 'merch_lat',
       'merch_long', 'age', 'distance', 'month', "day", "hour",'is_fraud']]

    print("sorted!")

    return train, test


#1つのID(cc_num)ごとに、直接入力アプローチのデータと時系列データを作成
def prepare_dataset(df, cc_num, ts_clips, X_d_pre, y_d_pre, window_size, stride):
    data = df[df.cc_num==cc_num].reset_index(drop=True).values

    for i in range(0, data.shape[0], stride):
        app_data = data[i:i+window_size]
        if(app_data.shape[0]<window_size):
            break
        ts_clips.append(app_data)
        X_d_pre.append(data[i+window_size-1,1:-1])
        y_d_pre.append(data[i+window_size-1,-1])

    return ts_clips, X_d_pre, y_d_pre


def labeling(ts_clips, X_d_pre, y_d_pre, spike_area):
    X_ts = np.empty([len(ts_clips), ts_clips[0].shape[0], ts_clips[0].shape[1]-1], dtype=object)
    y_ts = np.zeros([len(ts_clips)])
    X_d = np.empty([len(X_d_pre), X_d_pre[0].shape[0]])
    y_d = np.zeros([len(X_d_pre)])

    i = 0
    for j, clip in enumerate(ts_clips):
        if(sum(clip[:,-1]) == 0):
            # print(ts_clips)
            X_ts[i, :] = clip[:, :-1]
            y_ts[i] = 0
            X_d[i] = X_d_pre[j]
            y_d[i] = y_d_pre[j]
            i += 1
        elif sum(clip[0:-spike_area, -1])==0 and sum(clip[-spike_area:,-1]) > 0:
            X_ts[i, :] = clip[:, :-1]
            y_ts[i] = 1
            X_d[i] = X_d_pre[j]
            y_d[i] = y_d_pre[j]
            i += 1

    X_ts = X_ts[:, :, 1:]

    return X_ts[:i], y_ts[:i], X_d[:i], y_d[:i]


def create_dataset(df, window_size=15, stride=1, spike_area=1):
    cc_num_list = list(df['cc_num'].unique())
    ts_clips = []
    X_d_pre, y_d_pre = [], []

    for cc_num in cc_num_list:
        ts_clips, X_d_pre, y_d_pre = prepare_dataset(df, cc_num, ts_clips, X_d_pre, y_d_pre, window_size=window_size, stride=stride)

    #時系列データのラベル付け
    #また時系列データに合わせた直接入力データの再ラベル付け
    X_ts, y_ts, X_d, y_d = labeling(ts_clips, X_d_pre, y_d_pre, spike_area)

    return X_ts, y_ts, X_d, y_d


if __name__ == "__main__":

    base_path = "../dataset/Dataset_fin(2)/"
    train_df = pd.read_csv(base_path+"cv_fraudTrain_oh.csv")
    test_df = pd.read_csv(base_path+"cv_fraudTest_oh.csv")

    print("data loaded!")

    train_df, test_df = preprocess(train_df.copy(),test_df.copy())

    print("preprocessed!")

    X_train_ts, y_train_ts, X_train, y_train = create_dataset(train_df, window_size=15, stride=1, spike_area=1)
    print("train data created!")
    X_test_ts, y_test_ts, X_test, y_test = create_dataset(test_df, window_size=15, stride=1, spike_area=1)
    print("test data created!")

with open('../dataset/fraud_ts.pkl', 'wb') as f:
    pickle.dump([X_train_ts, y_train_ts, X_test_ts, y_test_ts] , f)
with open('../dataset/fraud_d.pkl', 'wb') as f:
    pickle.dump([X_train, y_train, X_test, y_test] , f)

