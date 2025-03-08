import argparse
from exp.exp_main import Exp_Main
import os
import torch
import numpy as np
import random


fix_seed = 421
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


def get_args():
    parser = argparse.ArgumentParser(description='A benchmark for SOH estimation')
    parser.add_argument('--random_seed', type=int, default=2023)
    
    # Folders
    parser.add_argument('--data_folder',type=str,default='data')
    parser.add_argument('--result_folder',type=str,default='result')


    #Dataset parameters
    parser.add_argument('--dataset', type=str, default='XJTU',choices=['XJTU','MIT']) # Type of Dataset. Default is XJTU
    parser.add_argument('--input_type',type=str,default='charge',choices=['charge','partial_charge','handcraft_features']) # Type of input data. Default is charge.
    parser.add_argument('--battery_batch', type=int, default=1,choices=[1,2,3,4,5,6,7,8,9]) # Type of battery batch. Default is 1. XJTU has 6 batches and MIT has 9 batches.
    parser.add_argument('--test_battery_id',type=int,default=1,help='test battery id, 1-8 for XJTU (1-15 for batch-2), 1-5 for MIT')
    parser.add_argument('--features',type=int,default=3,help='Number of features in the dataset')
    parser.add_argument("--post_features", type=int, default=3, help='Number of features after pre-processing.')
    parser.add_argument('--seq_len',type=int,default=128,help='Window size for the handcraft_features dataset')

    # Normalization Parameters
    parser.add_argument('--normalized_type',type=str,default='minmax',choices=['minmax','standard'])
    parser.add_argument('--minmax_range',type=tuple,default=(-1,1),choices=[(0,1),(-1,1)])
    

    # Model
    parser.add_argument('--model',type=str,default='GRU') # Type of Model. Default is CNN

    # Model Parameters
    # MLP
    parser.add_argument('--mlp_hidden_size',type=int,default=128)
    
    #LSTM and GRU
    parser.add_argument('--lstm_hidden_size',type=int,default=128)
    parser.add_argument('--lstm_num_layers',type=int,default=2)

    # CNN
    parser.add_argument('--cnn_hidden_size',type=int,default=128)
    parser.add_argument('--cnn_kernel_size',type=int,default=3)
    parser.add_argument('--cnn_stride',type=int,default=1)
    parser.add_argument('--cnn_padding',type=str,default="same")
    parser.add_argument('--cnn_output_channel',type=int,default=16)
    
    
    
    # Predictor Parameters
    parser.add_argument('--pred_hidden_size',type=int,default=64)


    # Training Parameters
    parser.add_argument('--batch_size',type=int,default=128)
    parser.add_argument('--lr',type=float,default=2e-3)
    parser.add_argument('--weight_decay', default=5e-4)
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--patience',default=5)
    parser.add_argument('--device',default='cpu') #defaults to cuda if available
    parser.add_argument('--loss', default="mse")


    # Experiment Parameters
    parser.add_argument('--experiment_num',default=1,type=int,help='The number of times you want to repeat the same experiment')

    args = parser.parse_args()
    return args


if __name__ == '__main__':


    
    try:
        args = get_args()
    except Exception as e:
        raise KeyError(f"{e}")
        args = None

    print("{}_model__{}_dataset__{}_features__{}_batch__{}_battery_test_id__{}_batch_size__{}_epochs__{}_lr__{}_device".format(
        args.model,
        args.dataset,
        args.input_type,
        args.battery_batch,
        args.test_battery_id,
        args.batch_size,
        args.epochs,
        args.lr,
        args.device))
    
    exp = Exp_Main(args)

    exp.Train()

    exp.Test()
    

