import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader.XJTU_loader import XJTUDdataset
from dataloader.MIT_loader import MITDdataset
#from nets.Model import SOHModel
from nets.preprocessing import PreProcessing
from nets import CNN, LSTM, GRU, MLP, Attention

class Exp_Main:

    def __init__(self, args, **kwargs):
        self.args = args
        self.data_loader = self._load_data(args)
        self.model = self._load_model()
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )



        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                [30,70],
                gamma=0.5,
            )

        self.criterion = self._select_criterion()

    def _load_data(self, args, **kwargs):

        ## Add your dataloaders for new datasets here.
        datasets = {
            "XJTU": XJTUDdataset,
            "MIT": MITDdataset
            }
        
        try:
            loader = datasets[args.dataset](args)

            if args.input_type == 'charge':
                data_loader = loader.get_charge_data(test_battery_id=args.test_battery_id)
            elif args.input_type == 'partial_charge':
                data_loader = loader.get_partial_data(test_battery_id=args.test_battery_id)
            else:
                data_loader = loader.get_features(test_battery_id=args.test_battery_id)
        except Exception as e:
            raise ValueError(f"Data Loading failed.\n{e}")

        return data_loader   
        
    def _select_criterion(self):
        
        # if self.args.loss == "mae":
        #     criterion = nn.L1Loss()
        # elif self.args.loss == "mse":
        #     criterion = nn.MSELoss()
        # elif self.args.loss == "smooth":
        #     criterion = nn.SmoothL1Loss()
        # else:
        criterion = nn.MSELoss()
        return criterion
    

    def _load_model(self):

        ## Add your new models here.
        model_dict = {
            "CNN": CNN,
            "LSTM": LSTM,
            "GRU": GRU,
            "MLP": MLP,
            "Attention": Attention
            }
        try:
            model = nn.Sequential(
                PreProcessing(self.args.input_type).float(),
                model_dict[self.args.model].Model(self.args).float()
            )
        except Exception as e:
            raise ValueError(f"Model Initialization failed.\n{e}")
        
        return model

    

    def Train(self):
        self.model.to(self.args.device)
        self.model.train()

        train_loader = self.data_loader["train"]
        val_loader = self.data_loader["valid"]

        best_val_loss = float("inf")
        best_model_path = f"checkpoints/best_model_{self.args.model}.pth"

        self.train_loss = []
        self.valid_loss = []

        patience = self.args.patience  # Number of epochs to wait before stopping
        patience_counter = 0

        for epoch in range(self.args.epochs):
            epoch_train_loss = 0.0

            for inputs,labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item()
            
            epoch_train_loss /= len(train_loader)
            self.train_loss.append(epoch_train_loss)
        
            # Validation Phase
            self.model.eval()
            epoch_val_loss = 0.0

            with torch.no_grad():
                for inputs,labels in val_loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    epoch_val_loss += loss.item()
                
            epoch_val_loss /= len(val_loader)
            self.valid_loss.append(epoch_val_loss)

            # Save the best model
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                torch.save(self.model.state_dict(), best_model_path)
                #print(f"Epoch {epoch+1}: New best model saved with validation loss {epoch_val_loss:.4f}")
                patience_counter = 0  # Reset counter when improvement is seen
            else:
                patience_counter += 1  # Increment counter when no improvement
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs with best validation loss {best_val_loss:.4f}")
                break
            
            self.scheduler.step()
            print(f"Epoch {epoch+1}/{self.args.epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    def Test(self):
        self.model.to(self.args.device)
        self.model.eval()

        test_loader = self.data_loader["test"]
        total_loss = 0.0

        with torch.no_grad():
            for inputs,labels in test_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

        avg_test_loss = total_loss / len(test_loader)
        print(f"Test Loss: {avg_test_loss:.4f}")


    def _save_results(self):
        pass


    def _plot_results(self):
        pass