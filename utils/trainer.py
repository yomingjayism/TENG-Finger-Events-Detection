'''
Created on Thu Dec 12 15:48:47 2024

@author: Kendrick
'''
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class TrainerBase():
    def __init__(
        self,
        trainset:Dataset=None,
        validset:Dataset=None,
        optimizer:torch.optim.Optimizer=None,
        model:nn.Module=None,
        device:torch.DeviceObjType=torch.device("cpu")
        ) -> None:
        
        self.trainset = trainset
        self.validset = validset
        self.optimizer = optimizer
        self.model = model
        self.device = device
        
    def train(self):
        raise NotImplementedError
        
    def _do_train_step(self, inputs):
        raise NotImplementedError
    
    def _do_valid_step(self, inputs):
        raise NotImplementedError
    
class Trainer(TrainerBase):
    def __init__(
        self,
        batch_size:int,
        num_epoch:int,
        trainset:Dataset=None,
        validset:Dataset=None,
        optimizer:torch.optim.Optimizer=None,
        model:nn.Module=None,
        device:torch.DeviceObjType=torch.device("cpu")
    ):
        super().__init__(
            trainset,
            validset,
            optimizer,
            model,
            device
        )
        
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.window_size= self.trainset.window_size
        
        self.trainloader = DataLoader(self.trainset, batch_size, True, num_workers=4, drop_last=True)
        if validset is not None:
            self.validloader = DataLoader(self.validset, batch_size, True, num_workers=4, drop_last=True)
            
    def train(self):
        best_val_loss = 1e13
        for i_epoch in range(self.num_epoch):
            avg_train_loss = 0
            trainloader = tqdm(self.trainloader)
            for i, data in enumerate(trainloader):
                signal = data[0].permute(0, 2, 1).to(self.device).float()
                label = data[1].view(self.batch_size, self.window_size, -1).to(self.device).float().permute(0, 2, 1)
                
                inputs = {"signal": signal, "label": label}
                loss = self._do_train_step(inputs)
                avg_train_loss += loss
                trainloader.set_postfix(
                    {"train_loss": round(loss, 4)})
            
            avg_train_loss = round(avg_train_loss / i, 4)
            avg_valid_loss = 0
            validloader = tqdm(self.validloader)
            for i, data in enumerate(validloader):
                signal = data[0].permute(0, 2, 1).to(self.device).float()
                label = data[1].view(self.batch_size, self.window_size, -1).to(self.device).float().permute(0, 2, 1)
                
                inputs = {"signal": signal, "label": label}
                loss = self._do_valid_step(inputs)
                avg_valid_loss += loss
            
                validloader.set_postfix(
                    {"valid_loss": round(loss, 4)})
            
            avg_valid_loss = round(avg_valid_loss / i, 4)
            if avg_valid_loss <= best_val_loss:
                best_val_loss = avg_valid_loss
                model_config = self.model.config
                model_weight = self.model.state_dict()
                ckpt = {"config":model_config, "checkpoints": model_weight}
                torch.save(ckpt, f"checkpoints/{model_config.model_name}.pt")
                
            print(f"Epoch:{i_epoch+1} || train loss:{avg_train_loss} || valid loss:{avg_valid_loss}")

    def _do_train_step(self, inputs):
        outputs = self.model(inputs["signal"], inputs["label"])
        loss = outputs["loss"]
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    @torch.no_grad()
    def _do_valid_step(self, inputs):
        outputs = self.model(inputs["signal"], inputs["label"])
        loss = outputs["loss"]
        return loss.item()
        
        