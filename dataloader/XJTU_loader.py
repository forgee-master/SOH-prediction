from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset,DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import matplotlib.pyplot as plt
from utils.Scaler import Scaler

class XJTUDdataset:
    """
    A dataset class for handling the XJTU battery dataset.
    It loads, preprocesses, and encapsulates battery charge/discharge data into PyTorch DataLoaders.
    """

    def __init__(self, args):
        """
        Initialize the dataset loader with the provided arguments.
        
        Args:
            args: Argument parser object containing various configurations such as:
                - data_folder: Path to the dataset folder.
                - normalized_type: Type of normalization ('minmax' or 'standard').
                - minmax_range: Range for MinMax scaling.
                - battery_batch: Batch identifier for the dataset.
                - batch_size: Size of data batches for training/testing.
        """
        self.args = args
        self.root = f'{args.data_folder}/XJTU'
        self.max_capacity = 2.0  # Maximum battery capacity (assumed constant)
        self.normalized_type = args.normalized_type
        self.minmax_range = args.minmax_range
        self.batch = args.battery_batch
        self.batch_size = args.batch_size

    def _parser_mat_data(self, battery_i_mat):
        """
        Parse .mat file data and extract features and labels.
        
        Args:
            battery_i_mat: Battery cycle data in .mat format (shape: (1, num_cycles)).
        
        Returns:
            data (np.array): Processed feature matrix (num_cycles, num_features, sequence_length).
            soh (np.array): State of Health (SOH) labels normalized by max_capacity.
        """
        data = []
        label = []

        for i in range(battery_i_mat.shape[1]):
            cycle_i_data = battery_i_mat[0, i]
            
            # Extract features (current, voltage, temperature)
            current = cycle_i_data['current_A']  # Shape: (389, 1, 128)
            voltage = cycle_i_data['voltage_V']  # Shape: (389, 1, 128)
            temperature = cycle_i_data['temperature_C']  # Shape: (389, 1, 128)
            
            # Extract battery capacity (SOH label)
            capacity = cycle_i_data['capacity'][0] # Shape: (389, 1)
            label.append(capacity)
            
            # Concatenate features along the first axis
            cycle_i = np.concatenate([current, voltage, temperature], axis=0) # Shape: (389, 3, 128)
            data.append(cycle_i)
        
        data = np.array(data, dtype=np.float32)  # Convert to NumPy array
        label = np.array(label, dtype=np.float32)  # Convert labels to NumPy array
        
        soh = label / self.max_capacity  # Normalize SOH labels
    
        return data, soh

    def _encapsulation(self, train_x, train_y, test_x, test_y):
        """
        Encapsulates NumPy arrays into PyTorch DataLoader objects.
        
        Args:
            train_x, train_y: Training data and labels.
            test_x, test_y: Testing data and labels.
        
        Returns:
            train_loader, valid_loader, test_loader: PyTorch DataLoaders for training, validation, and testing.
        """
        train_x = torch.from_numpy(train_x)
        test_x = torch.from_numpy(test_x)
        train_y = torch.unsqueeze(torch.from_numpy(train_y), -1)
        test_y = torch.unsqueeze(torch.from_numpy(test_y), -1)

        # Split training data into train and validation sets
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2)

        print(f"\nTrain Data:\tX={train_x.shape},\ty={train_y.shape}")
        print(f"Valid Data:\tX={valid_x.shape},\ty={valid_y.shape}")
        print(f"Test Data:\tX={test_x.shape},\ty={test_y.shape}")

        train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(TensorDataset(valid_x, valid_y), batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=self.batch_size, shuffle=False)
        
        return train_loader, valid_loader, test_loader

    def _get_raw_data(self, path, test_battery_id):
        """
        Load and preprocess raw battery data.
        
        Args:
            path: Path to the .mat data file.
            test_battery_id: ID of the battery used for testing.
        
        Returns:
            train_loader, valid_loader, test_loader: Preprocessed PyTorch DataLoaders.
        """
        mat = loadmat(path)
        battery = mat['battery']
        battery_ids = list(range(1, battery.shape[1] + 1))
        
        if test_battery_id not in battery_ids:
            raise IndexError(f'"test_battery" must be in {battery_ids}, but got {test_battery_id}.')

        # Extract test battery data
        test_battery = battery[0, test_battery_id - 1][0]
        test_x, test_y = self._parser_mat_data(test_battery)
        print(f"Test battery id {test_battery_id}:\tdata={test_x.shape}\tlabel={test_y.shape}")

        # Extract training battery data
        train_x, train_y = [], []
        for id in battery_ids:
            if id == test_battery_id:
                continue
            train_battery = battery[0, id - 1][0]
            x, y = self._parser_mat_data(train_battery)
            print(f"Train battery id {id}:\tdata={x.shape}\tlabel={y.shape}")
            train_x.append(x)
            train_y.append(y)
        
        train_x = np.concatenate(train_x, axis=0)
        train_y = np.concatenate(train_y, axis=0)
        
        # Normalize data
        scaler = MinMaxScaler(feature_range=self.minmax_range) if self.normalized_type.lower() == "minmax" else StandardScaler()
        for i in range(train_x.shape[1]):
            train_x[:, i, :] = scaler.fit_transform(train_x[:, i, :])
            test_x[:, i, :] = scaler.transform(test_x[:, i, :])
        
        return self._encapsulation(train_x, train_y, test_x, test_y)

    def get_charge_data(self, test_battery_id=1):
        """
        Load full charge cycle data.
        
        Args:
            test_battery_id: ID of the battery to be used for testing.
        
        Returns:
            Dictionary containing train, validation, and test DataLoaders.
        """
        file_name = f'batch-{self.batch}.mat'
        self.charge_path = os.path.join(self.root, 'charge', file_name)
        return self._get_raw_data(self.charge_path, test_battery_id)

    def get_partial_data(self, test_battery_id=1):
        """
        Load partial charge cycle data.
        
        Args:
            test_battery_id: ID of the battery to be used for testing.
        
        Returns:
            Dictionary containing train, validation, and test DataLoaders.
        """
        file_name = f'batch-{self.batch}_3.7-4.1.mat' if self.batch != 6 else f'batch-{self.batch}_3.9-4.19.mat'
        self.partial_path = os.path.join(self.root, 'partial_charge', file_name)
        return self._get_raw_data(self.partial_path, test_battery_id)

    def _parser_xlsx(self,df_i):
        '''
        features dataframe
        :param df_i: shape:(N,C+1)
        :return:
        '''
        N = df_i.shape[0]
        x = np.array(df_i.iloc[:, :-1],dtype=np.float32)
        label = np.array(df_i['label'],dtype=np.float32).reshape(-1, 1)

        scaler = Scaler(x)
        if self.normalized_type == 'standard':
            data = scaler.standerd()
        else:
            data = scaler.minmax(feature_range=self.minmax_range)
        soh = label / self.max_capacity

        return data, soh

    def get_features(self,test_battery_id=1):
        print('----------- load features -------------')
        file_name = f'batch-{self.batch}_features.xlsx'
        self.features_path = os.path.join(self.root, 'handcraft_features', file_name)
        df = pd.read_excel(self.features_path,sheet_name=None)
        sheet_names = list(df.keys())
        battery_ids = list(range(1, len(sheet_names)+1))

        if test_battery_id not in battery_ids:
            raise IndexError(f'"test_battery" must be in the {battery_ids}, but got {test_battery_id}. ')
        test_battery_df = pd.read_excel(self.features_path,sheet_name=test_battery_id-1,header=0)
        test_x,test_y = self._parser_xlsx(test_battery_df)
        print(f'test battery id: {test_battery_id}, test data shape: {test_x.shape}, {test_y.shape}')

        train_x, train_y = [], []
        for id in battery_ids:
            if id == test_battery_id:
                continue
            sheet_name = sheet_names[id-1]
            df_i = df[sheet_name]
            x, y = self._parser_xlsx(df_i)
            print(f'train battery id: {id}, {x.shape}, {y.shape}')
            train_x.append(x)
            train_y.append(y)
        train_x = np.concatenate(train_x,axis=0)
        train_y = np.concatenate(train_y,axis=0)
        print('train data shape: ', train_x.shape, train_y.shape)

        train_loader, valid_loader, test_loader = self._encapsulation(train_x, train_y, test_x, test_y)
        data_dict = {'train': train_loader,
                     'test': test_loader,
                     'valid': valid_loader}
        print('---------------  finished !  ----------------')
        return data_dict
