import pandas as pd
import numpy as np
import torch
from joblib import Parallel, delayed
from opart_functions import gen_data_dict, get_acc_rate, get_err_df, add_row_to_csv

from BIC import BIC
from linear import linear
from MLP import cv_learn, mlp

# PATHs (edit these paths depending on dataset)
dataset = 'genome'

# training data
features_fold1_path = 'training_data/' + dataset + '/seq_features.csv'
features_fold2_path = 'training_data/' + dataset + '/seq_features.csv'  
target_fold1_path = 'training_data/' + dataset + '/target_fold1.csv'
target_fold2_path = 'training_data/' + dataset + '/target_fold2.csv'

# sequences and labels
seqs_path   = 'raw_data/' + dataset + '/signals.csv'
labels_path = 'raw_data/' + dataset + '/labels.csv'

# err for each log_lambda
err_fold1_path = 'training_data/' + dataset + '/errors_fold1.csv'
err_fold2_path = 'training_data/' + dataset + '/errors_fold2.csv'

# writing accuracy rate path
acc_rate_path = 'acc_rate/' + dataset + '.csv'

# path to write df to csv
output_df_path = 'record_dataframe/' + dataset + '/'



def record(method_name, df_fold1, df_fold2, output_df_path, acc_rate_path):
    # save df into csv
    df_fold1.to_csv(output_df_path + method_name + '.fold1.csv', index=False)
    df_fold2.to_csv(output_df_path + method_name + '.fold2.csv', index=False)
    
    # add a row in acc_rate csv
    acc_1 = get_acc_rate(df_fold1)
    acc_2 = get_acc_rate(df_fold2)
    add_row_to_csv(acc_rate_path, ["method", "fold1.test", "fold2.test"], [method_name, acc_1, acc_2])



def gen_BIC_lldas(features_df):
    feature = features_df['length'].to_numpy()
    lldas = BIC(feature)
    return lldas



def gen_target_from_df(target_df):
    targets_low  = torch.Tensor(target_df.iloc[:, 1:2].to_numpy())
    targets_high = torch.Tensor(target_df.iloc[:, 2:3].to_numpy())
    targets = torch.cat((targets_low, targets_high), dim=1)
    return targets



def gen_linear_lldas(features_df, target_df):
    # features
    feature = features_df['length'].to_numpy()
    feature = np.log10(np.log(feature)).reshape(-1,1)
    feature = torch.Tensor(feature)

    # target
    targets = gen_target_from_df(target_df)

    # learn lldas
    lldas = linear(feature, targets)
    return lldas



def gen_mlp_lldas(features_df, target_df, hidden_layers, hidden_size, batch_size, max_ite):
    # features
    chosen_feature = ['std_deviation', 'length', 'sum_diff', 'range_value']
    X = features_df.iloc[:, 1:][chosen_feature].to_numpy()
    X0 = np.log(X[:, 0]).reshape(-1, 1)
    X1 = np.log(np.log(X[:, 1])).reshape(-1, 1)
    X2 = np.log(np.log(X[:, 2])).reshape(-1, 1)
    X3 = np.log(X[:, 3]).reshape(-1, 1)

    X = np.concatenate([X0, X1, X2, X3], axis=1)
    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)
    X = (X-mean)/std_dev
    X = torch.Tensor(X)
    
    # targets
    targets = gen_target_from_df(target_df)

    # learn best number of iterations
    n_ite = cv_learn(2, X, targets, hidden_layers, hidden_size, batch_size, max_ite)

    # train model to get lldas
    lldas = mlp(X, targets, hidden_layers, hidden_size, batch_size, n_ite+1)
    return lldas



if __name__ == "__main__":
    # generate sequence and label dictionary
    seqs_dict   = gen_data_dict(seqs_path)
    labels_dict = gen_data_dict(labels_path)

    # getting dataframe of error count for each log_lambda
    err_fold1_df = pd.read_csv(err_fold1_path)
    err_fold2_df = pd.read_csv(err_fold2_path)

    # features
    features_df_fold1 = pd.read_csv(features_fold1_path)
    features_df_fold2 = pd.read_csv(features_fold2_path)

    # targets
    target_df_fold1 = pd.read_csv(target_fold1_path)
    target_df_fold2 = pd.read_csv(target_fold2_path)


    # getting results
    methods = ['BIC.1', 'linear.1', 'linear.4', 'mlp.4.1.8', 'mlp.4.2.16', 'mlp.4.3.32', 'mlp.4.4.64']
    for method in methods:
        if(method == 'BIC.1'):
            lldas_train_fold1 = delayed(gen_BIC_lldas)(features_df_fold1)
            lldas_train_fold2 = delayed(gen_BIC_lldas)(features_df_fold2)
        elif(method == 'linear.1'):
            lldas_train_fold1 = delayed(gen_linear_lldas)(features_df_fold1, target_df_fold1)
            lldas_train_fold2 = delayed(gen_linear_lldas)(features_df_fold2, target_df_fold2)
        elif(method == 'linear.4'):
            lldas_train_fold1 = delayed(gen_mlp_lldas)(features_df_fold1, target_df_fold1, 0, 0, 1000, 10000)
            lldas_train_fold2 = delayed(gen_mlp_lldas)(features_df_fold2, target_df_fold2, 0, 0, 1000, 10000)
        elif(method == 'mlp.4.1.8'):
            lldas_train_fold1 = delayed(gen_mlp_lldas)(features_df_fold1, target_df_fold1, 1, 8, 1000, 10000)
            lldas_train_fold2 = delayed(gen_mlp_lldas)(features_df_fold2, target_df_fold2, 1, 8, 1000, 10000)
        elif(method == 'mlp.4.2.16'):
            lldas_train_fold1 = delayed(gen_mlp_lldas)(features_df_fold1, target_df_fold1, 2, 16, 1000, 10000)
            lldas_train_fold2 = delayed(gen_mlp_lldas)(features_df_fold2, target_df_fold2, 2, 16, 1000, 10000)
        elif(method == 'mlp.4.3.32'):
            lldas_train_fold1 = delayed(gen_mlp_lldas)(features_df_fold1, target_df_fold1, 3, 32, 1000, 10000)
            lldas_train_fold2 = delayed(gen_mlp_lldas)(features_df_fold2, target_df_fold2, 3, 32, 1000, 10000)
        elif(method == 'mlp.4.4.64'):
            lldas_train_fold1 = delayed(gen_mlp_lldas)(features_df_fold1, target_df_fold1, 4, 64, 1000, 10000)
            lldas_train_fold2 = delayed(gen_mlp_lldas)(features_df_fold2, target_df_fold2, 4, 64, 1000, 10000)
        
        lldas_train_fold1, lldas_train_fold2 = Parallel(n_jobs=2)([lldas_train_fold1, lldas_train_fold2])
        df_fold1 = get_err_df(lldas_train_fold1, 1, seqs_dict, labels_dict, err_fold1_df)
        df_fold2 = get_err_df(lldas_train_fold2, 2, seqs_dict, labels_dict, err_fold2_df)
        record(method, df_fold1, df_fold2, output_df_path, acc_rate_path)