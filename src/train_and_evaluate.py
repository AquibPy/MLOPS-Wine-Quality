from ast import arg
import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score,recall_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from get_data import read_params
import argparse
import joblib
import json
import pickle

def eval_metrics(actual,pred):
    accuracy = accuracy_score(actual,pred)
    precision = precision_score(actual,pred,average='weighted',labels=np.unique(pred))
    recall = recall_score(actual,pred,average='weighted')
    return accuracy, precision, recall

def train_and_evaluate(config_path):
    config = read_params(config_path)
    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]
    random_state = config['base']['random_state']
    model_dir = config['model_dir']
    
    alpha = config['estimators']['SGDClassifier']['params']['alpha']
    l1_ratio = config['estimators']['SGDClassifier']['params']['l1_ratio']

    target = [config['base']['target_col']]

    train = pd.read_csv(train_data_path,sep=",")
    test = pd.read_csv(test_data_path,sep=",")

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target,axis=1)
    test_x = test.drop(target,axis=1)

    clf = SGDClassifier(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    clf.fit(train_x,train_y.values.ravel())
    predicted_qualities = clf.predict(test_x)

    (accuracy, precision, recall) = eval_metrics(test_y, predicted_qualities)
    print("#"*30)
    print("SGDClassifier model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  Accuracy: %s" % accuracy)
    print("  Precision: %s" % precision)
    print("  Recall: %s" % recall)
    print("#"*30)

    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]
    with open(scores_file, "w") as f:
        scores = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall
        }
        json.dump(scores,f,indent = 4)
    
    with open(params_file,"w") as f:
        params = {
            "alpha": alpha,
            "l1_ratio": l1_ratio
        }
        json.dump(params,f,indent=4)
    
    os.makedirs(model_dir,exist_ok=True)
    model_path = os.path.join(model_dir,"model.joblib")
    joblib.dump(clf,model_path)


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config",default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(parsed_args.config)