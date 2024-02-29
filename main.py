import pandas as pd
import numpy as np
import torch
from opart_functions import gen_data_dict, get_acc_rate, get_err_df, add_row_to_csv
from BIC import BIC
from linear import linear


# PATHs (edit these paths depending on dataset)
# training data
features_fold1_path = 'training_data/genome/seq_features.csv'
features_fold2_path = 'training_data/genome/seq_features.csv'  
target_fold1_path = 'training_data/genome/target_fold1.csv'
target_fold2_path = 'training_data/genome/target_fold2.csv'

# sequences and labels
seqs_path   = 'raw_data/genome/signals.csv'
labels_path = 'raw_data/genome/labels.csv'

# err for each log_lambda
err_fold1_path = 'training_data/genome/errors_fold1.csv'
err_fold2_path = 'training_data/genome/errors_fold2.csv'

# writing accuracy rate path
acc_rate_path = 'acc_rate/genome.csv'

# path to write df to csv
output_df_path = 'record_dataframe/genome/'



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



def gen_linear_lldas(features_df, target_df):
    # features
    feature = features_df['length'].to_numpy()
    feature = np.log10(np.log(feature)).reshape(-1,1)
    feature = torch.FloatTensor(feature)

    # target
    targets_low  = torch.FloatTensor(target_df.iloc[:, 1:2].to_numpy())
    targets_high = torch.FloatTensor(target_df.iloc[:, 2:3].to_numpy())
    target = torch.cat((targets_low, targets_high), dim=1)

    # learn lldas
    lldas = linear(feature, target)
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
    methods = ['BIC', 'linear']
    for method in methods:
        if(method == 'BIC'):
            lldas_train_fold1 = gen_BIC_lldas(features_df_fold1)
            lldas_train_fold2 = gen_BIC_lldas(features_df_fold1)
        elif(method == 'linear'):
            lldas_train_fold1 = gen_linear_lldas(features_df_fold1, target_df_fold1)
            lldas_train_fold2 = gen_linear_lldas(features_df_fold2, target_df_fold2)

        df_fold1 = get_err_df(lldas_train_fold2, 1, seqs_dict, labels_dict, err_fold1_df)
        df_fold2 = get_err_df(lldas_train_fold1, 2, seqs_dict, labels_dict, err_fold2_df)
        record(method, df_fold1, df_fold2, output_df_path, acc_rate_path)