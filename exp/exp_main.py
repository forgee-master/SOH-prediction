import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader.XJTU_loader import XJTUDdataset
from dataloader.MIT_loader import MITDdataset
#from nets.Model import SOHModel
from layers.preprocessing import PreProcessingNet
from nets import CNN, LSTM, GRU, MLP, Attention, DCTLowRank
from utils.metrics import metric
import matplotlib.pyplot as plt

import os
import uuid
import csv

class Exp_Main:

    def __init__(self, args, **kwargs):
        self.args = args
        self.model = self._load_model()

        self.train_loader, self.valid_loader, self.test_loader = self._load_data(args)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=3,
                gamma=0.9,
            )

        self.criterion = self._select_criterion()

        self.run_id = uuid.uuid4().hex[:4] # Short unique ID
        self.results_dir = self._make_dir(self.run_id)
    
        self.best_model_path = f"checkpoints/best_model_{self.args.model}_{self.run_id}.pth"

    
    def _make_dir(self, id):
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")

        if not os.path.exists("results"):
            os.makedirs("results")

        result_dir = f"results/{self.args.model}_{id}"
        os.makedirs(result_dir)

        return result_dir

    def _load_data(self, args, **kwargs):

        ## Add your dataloaders for new datasets here.
        datasets = {
            "XJTU": XJTUDdataset,
            "MIT": MITDdataset
            }
        
        try:
            loader = datasets[args.dataset](args)

            if args.input_type == 'charge':
                return loader.get_charge_data(test_battery_id=args.test_battery_id)
            elif args.input_type == 'partial_charge':
                return loader.get_partial_data(test_battery_id=args.test_battery_id)
            else:
                return loader.get_features(test_battery_id=args.test_battery_id)
        except Exception as e:
            raise ValueError(f"Data Loading failed.\n{e}")

        return data_loader   
        
    def _select_criterion(self):
        
        if self.args.loss == "mae":
            criterion = nn.L1Loss()
        elif self.args.loss == "mse":
            criterion = nn.MSELoss()
        elif self.args.loss == "smooth":
            criterion = nn.SmoothL1Loss()
        else:
            criterion = nn.MSELoss()
        return criterion
    

    def _load_model(self):

        ## Add your new models here.
        model_dict = {
            "CNN": CNN,
            "LSTM": LSTM,
            "GRU": GRU,
            "MLP": MLP,
            "Attention": Attention,
            "DCTLora": DCTLowRank
            }
        try:
            model = nn.Sequential(
                #PreProcessingNet(self.args).float(),
                model_dict[self.args.model].Model(self.args).float()
            )
        except Exception as e:
            raise ValueError(f"Model Initialization failed.\n{e}")
        
        return model

    

    def Train(self):
        self.model.to(self.args.device)
        self.model.train()

        best_val_loss = float("inf")
        

        self.train_loss = []
        self.valid_loss = []

        patience = int(self.args.patience)  # Number of epochs to wait before stopping
        patience_counter = 0

        for epoch in range(self.args.epochs):
            epoch_train_loss = 0.0

            for inputs,labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
            
            epoch_train_loss /= len(self.train_loader)
            self.train_loss.append(epoch_train_loss)
        
            # Validation Phase
            #self.model.eval()
            epoch_val_loss = 0.0

            with torch.no_grad():
                for inputs,labels in self.valid_loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    epoch_val_loss += loss.item()
                
            epoch_val_loss /= len(self.valid_loader)
            self.valid_loss.append(epoch_val_loss)

            # Save the best model
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                torch.save(self.model.state_dict(), self.best_model_path)
                patience_counter = 0  # Reset counter when improvement is seen
                model_saved = "TRUE"
            else:
                patience_counter += 1  # Increment counter when no improvement
                model_saved = "FALSE"
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs with best validation loss {best_val_loss:.4f}")
                break
            
            self.scheduler.step()
            statement = f"Epoch {epoch+1}/{self.args.epochs}, Train Loss: {epoch_train_loss:.5f}, Val Loss: {epoch_val_loss:.5f}, Model Saved: {model_saved}, lr: {self.optimizer.param_groups[0]['lr']}, patience: {patience_counter}/{patience}"
            print(statement)
            

    def Test(self):
        self.model.load_state_dict(torch.load(self.best_model_path, map_location= self.args.device))
        self.model.eval()

        
        pred = []
        true = []


        with torch.no_grad():
            for inputs,labels in self.test_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                
                pred.append(outputs.detach().clone().cpu().numpy())
                true.append(labels.detach().clone().cpu().numpy())

        preds = np.concatenate(pred, axis=0)
        trues = np.concatenate(true, axis=0)

        mae, mse, rmse, mape = metric(preds, trues)

        file_path = "results/result.csv"
        file_exists = os.path.exists(file_path)
        
        with open(file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            
            # Write header if file does not exist
            if not file_exists:
                writer.writerow(["Model", "Run_ID", "Dataset", "Input_Type", "Batch", "Test_Battery_ID", "Batch_Size", "Epochs", "LR", "Device", "MAE", "MSE", "RMSE", "MAPE"])
            
            # Write data
            writer.writerow([
                self.args.model,
                self.run_id,
                self.args.dataset,
                self.args.input_type,
                self.args.battery_batch,
                self.args.test_battery_id,
                self.args.batch_size,
                self.args.epochs,
                self.args.lr,
                self.args.device,
                mae * 1e3,
                mse * 1e3,
                rmse * 1e3,
                mape * 1e3
            ])

        #self._save_results(preds, trues)
        self._plot_results(preds, trues)
        print(f"\nTest Loss MAE:{mae * 1e3:.3f} MSE:{mse * 1e3:.3f} MAPE:{mape * 1e3:.3f}")


    def _save_results(self, preds, trues):
        """Save predictions and ground truth values as .npy files."""
        np.save(f"{self.results_dir}/predictions_{self.run_id}.npy", preds)
        np.save(f"{self.results_dir}/ground_truth_{self.run_id}.npy", trues)


    def _plot_results(self, preds, trues):
        """Plot predictions vs. ground truth and save the figure."""
        
        preds = preds.reshape(-1,1)
        trues = trues.reshape(-1,1)

        plt.figure(figsize=(10, 5))
        plt.plot(trues, label="Ground Truth", linestyle="dashed")
        plt.plot(preds, label="Predictions", alpha=0.7)
        plt.xlabel("Time Step")
        plt.ylabel("SOH")
        plt.title("Predictions vs Ground Truth")
        plt.legend()
        plt.grid()
        
        plot_path = f"{self.results_dir}/plot.png"
        plt.savefig(plot_path)
        #plt.show()
        