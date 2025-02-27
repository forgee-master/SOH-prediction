import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader.XJTU_loader import XJTUDdataset
from dataloader.MIT_loader import MITDdataset
from nets.Model import SOHModel

class Exp_Main:

    def __init__(self, args, **kwargs):
        self.args = args
        self.data_loader = self._load_data(args)
        self.model = SOHModel(args)
        
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


    def _load_data(self, args, **kwargs):
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
            raise ValueError(f"{e}")

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
    


    

    def Train(self):
        pass