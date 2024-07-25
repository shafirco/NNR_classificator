import time
import json
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from typing import List
from scipy.spatial.distance import pdist,squareform
from collections import Counter
from sklearn.preprocessing import StandardScaler

#gets a point x with n feafutres and data_trn - an array of point with n features and calculates which points in
#data_trn that their distance from x is r or less.
# returns an array of the indexes
def get_indexes_of_features_in_radius(x,data_trn,r):
    distances = np.linalg.norm(data_trn - x, axis=1)
    indexes= np.where(distances<=r)
    if not np.any(indexes):
        min = np.min(distances)
        indexes= np.where(distances==min)

    return indexes

# gets two data frames - one for test and one for train, and a float for radius.
# the function predicts for each row in the train data set its classes.
# returns a list of classes.
def get_predictions(data_tst,data_trn ,r):
    indexes= [get_indexes_of_features_in_radius(x,data_trn.drop(data_trn.columns[-1],axis=1).to_numpy(),r) for x in data_tst.to_numpy()]
    classes = [np.take(data_trn.to_numpy(),index,axis=0)[:,:,-1] for index in indexes]
    classes=[Counter(class_list.flat).most_common()[0][0] for class_list in classes]

    return classes

# a function to calculate the range that the optimal radius for NNR model should be.
# returns the start and end of that range.
def get_radius_range(x_train_scaled,y_train):
    pairwise_distances= squareform(pdist(x_train_scaled,metric='euclidean'))
    start=pairwise_distances.mean()-len(np.unique(y_train))
    end=pairwise_distances.mean()
    return start,end

# a function to find the optimal radius for NNR model.
# returns that radius
def find_optimal_radius(data_vld):
    optimal_radius = 1.5
    rows = data_vld.shape[0]
    train = data_vld[:int(rows * 4 / 5)]
    tst = data_vld[int(rows * 4 / 5):]
    labels = tst['class'].values
    tst.drop(tst.columns[-1], axis=1, inplace=True)
    scaler = StandardScaler()
    predictor_scaled_tst = pd.DataFrame(scaler.fit_transform(tst))
    predictor_scaled_trn = pd.DataFrame(scaler.fit_transform(train.iloc[:, :-1]))
    predictor_scaled_trn["target"] = train.iloc[:, -1:]

    start,end=get_radius_range(predictor_scaled_trn.iloc[:, :-1],labels)


    best_accuracy_score=0
    for radius in np.arange(start,end,0.01) :
        predicted = get_predictions(predictor_scaled_tst,predictor_scaled_trn,radius)
        curr_accuracy_score=accuracy_score(labels, predicted)

        if curr_accuracy_score>best_accuracy_score:
            best_accuracy_score=curr_accuracy_score
            optimal_radius=radius
    return optimal_radius


# a function that classify the data tst using the train set and valid set.
def classify_with_NNR(data_trn, data_vld, data_tst) -> List:
    # todo: implement this function
    print(f'starting classification with {data_trn}, {data_vld}, and {data_tst}')

    predictions = list()  # todo: return a list of your predictions for test instances

    df_tst=pd.read_csv(data_tst)
    df_trn = pd.read_csv(data_trn)
    df_tst.drop(df_tst.columns[-1],axis=1, inplace=True)
    scaler = StandardScaler()
    predictor_scaled_tst = pd.DataFrame(scaler.fit_transform(df_tst))
    predictor_scaled_trn = pd.DataFrame(scaler.fit_transform(df_trn.iloc[:, :-1]))
    predictor_scaled_trn["target"]=df_trn.iloc[:,-1:]

    df_vld = pd.read_csv(data_vld)
    r=find_optimal_radius(df_vld)
    predictions = get_predictions(predictor_scaled_tst,predictor_scaled_trn,r)

    return predictions


if __name__ == '__main__':
    start = time.time()

    with open('config.json', 'r', encoding='utf8') as json_file:
        config = json.load(json_file)

    predicted = classify_with_NNR(config['data_file_train'],
                                  config['data_file_validation'],
                                  config['data_file_test'])

    df = pd.read_csv(config['data_file_test'])
    labels = df['class'].values

    if not predicted:  # empty prediction, should not happen in your implementation
        predicted = list(range(len(labels)))

    assert(len(labels) == len(predicted))  # make sure you predict label for all test instances
    print(f'test set classification accuracy: {accuracy_score(labels, predicted)}')

    print(f'total time: {round(time.time()-start, 0)} sec')
