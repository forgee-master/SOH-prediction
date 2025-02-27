import argparse
from exp.exp_main import Exp_Main
import os
import torch


def get_args():
    parser = argparse.ArgumentParser(description='A benchmark for SOH estimation')
    parser.add_argument('--random_seed', type=int, default=2023)
    
    # Type of dataset: XJTU or MIT
    parser.add_argument('--dataset', type=str, default='XJTU',choices=['XJTU','MIT']) # Type of Dataset. Default is XJTU
    parser.add_argument('--input_type',type=str,default='charge',choices=['charge','partial_charge','handcraft_features']) # Type of input data. Default is charge.
    parser.add_argument('--batch', type=int, default=1,choices=[1,2,3,4,5,6,7,8,9]) # Type of batch. Default is 1. XJTU has 6 batches and MIT has 9 batches.
    parser.add_argument('--test_battery_id',type=int,default=1,help='test battery id, 1-8 for XJTU (1-15 for batch-2), 1-5 for MIT')

    # Normalization Parameters
    parser.add_argument('--normalized_type',type=str,default='minmax',choices=['minmax','standard'])
    parser.add_argument('--minmax_range',type=tuple,default=(-1,1),choices=[(0,1),(-1,1)])
    

    # Model
    parser.add_argument('--model',type=str,default='CNN',choices=['CNN','LSTM','GRU','MLP','Attention'])

    # Training Parameters
    parser.add_argument('--batch_size',type=int,default=128)
    parser.add_argument('--lr',type=float,default=2e-3)
    parser.add_argument('--weight_decay', default=5e-4)
    parser.add_argument('--n_epoch',type=int,default=100)
    parser.add_argument('--early_stop',default=30)
    parser.add_argument('--device',default='cpu') #defaults to cuda if available
    parser.add_argument('--save_folder',default='results')

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
        args.batch,
        args.test_battery_id,
        args.batch_size,
        args.n_epoch,
        args.lr,
        args.device))
    
    exp = Exp_Main(args)

    #exp.Train()
    

