# packages
import numpy as np
import pandas as pd
import csv
import os
import torch
import torch.nn as nn



# generate a dictionary of data grouped by sequenceID
def gen_data_dict(file_path):
    df = pd.read_csv(file_path)
    _dict = tuple(df.groupby('sequenceID'))
    return _dict



# get data from sequence i
def get_data(i, fold, seqs_dict, labels_dict):
    # sequence
    sequence = seqs_dict[i][1]['logratio'].to_numpy()
    sequence = np.append([0], sequence)

    # labels dataframe
    lab_df = labels_dict[i][1]

    # get label sets
    lab_df = lab_df[lab_df['fold'] == fold]

    pos_lab_df = lab_df[lab_df['changes'] == 1]
    neg_lab_df = lab_df[lab_df['changes'] == 0]

    neg_start, neg_end = neg_lab_df['start'].to_numpy(), neg_lab_df['end'].to_numpy()
    pos_start, pos_end = pos_lab_df['start'].to_numpy(), pos_lab_df['end'].to_numpy()

    return sequence, neg_start, neg_end, pos_start, pos_end



# add row to csv
def add_row_to_csv(path, head, row):
    file_exists = os.path.exists(path)              # Check if the file exists
    is_row_exist = False                            # default False for is_row_exist
    with open(path, 'a', newline='') as csvfile:    # Open the CSV file in append mode
        writer = csv.writer(csvfile)
        if not file_exists:                         # If the file doesn't exist, write the header
            writer.writerow(head)
        with open(path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for existing_row in reader:             # Iterate over each row
                if existing_row[0] == row[0]:             # Check if the row already exists
                    is_row_exist = True
        if(not is_row_exist):
            writer.writerow(row)                    # Write the row



# get accuracy rate from df
def get_acc_rate(df):
    total_label = df['total_labels'].sum()
    err = df['err'].sum()
    rate = 100 * (total_label - err)/total_label
    return rate



# get err df from trained log_lamdas for both folds
def get_err_df(lldas_train, test_fold, seqs_dict, labels_dict, err_df):
    header = ['sequenceID', 'llambda', 'total_labels', 'err']
    rows = []
    for i in range(len(seqs_dict)):
        # generate data
        _, neg_start, _, pos_start, _ = get_data(i, test_fold, seqs_dict, labels_dict)

        # get total labels
        total_labels = len(neg_start) + len(pos_start)

        # get err
        err = err_df.iloc[i][str(lldas_train[i])]

        # add row into df
        row = [seqs_dict[i][0], lldas_train[i],  total_labels, err]
        rows.append(row)

    df = pd.DataFrame(rows, columns=header)
    return df



# tuning lldas
def tune_lldas(lldas):
    lldas = np.round(lldas*2)/2
    lldas[lldas > 5.0] = 5.0
    lldas[lldas < -5.0] = -5.0
    lldas[np.isclose(lldas, -0.0)] = 0.0
    lldas[np.isnan(lldas)] = 0.0
    return lldas



# Hinged Square Loss
class SquaredHingeLoss(nn.Module):
    def __init__(self, margin=1):
        super(SquaredHingeLoss, self).__init__()
        self.margin = margin

    def forward(self, predicted, y):
        low, high = y[:, 0], y[:, 1]
        loss_low = torch.relu(low - predicted + self.margin)
        loss_high = torch.relu(predicted - high + self.margin)
        loss = loss_low + loss_high
        return torch.mean(torch.square(loss))
