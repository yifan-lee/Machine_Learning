import torch
import torch.nn as nn
import torch.nn.functional as F # activation function ReLU
from torch.optim import SGD # stochastic gradient descent

import lightning as L
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

### Parameters

input_doses = torch.linspace(start=0, end=1, steps=11)

inputs = torch.tensor([0.,0.5,1.])
labels = torch.tensor([0.,1.,0.])
dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset)

### Basic Lightning NN

class BasicLightning(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.w00 = nn.Parameter(torch.tensor(1.7),requires_grad=False) # don't optims it
        self.b00 = nn.Parameter(torch.tensor(-0.85),requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8),requires_grad=False)
        self.w10 = nn.Parameter(torch.tensor(12.6),requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.),requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7),requires_grad=False)
        self.final_bias = nn.Parameter(torch.tensor(-16.),requires_grad=False)


    def forward(self, input):
        input_to_top_relu = input*self.w00+self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scale_top_relu_output = top_relu_output*self.w01
        
        input_to_bottom_relu = input*self.w10+self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scale_bottom_relu_output = bottom_relu_output*self.w11
        
        input_to_final_relu = scale_top_relu_output+scale_bottom_relu_output+self.final_bias
        output = F.relu(input_to_final_relu)
        return output
    
model = BasicLightning()
output_values = model(input_doses)
sns.set_theme(style="whitegrid")
sns.lineplot(x=input_doses, y=output_values, color='green',linewidth=2.5)
plt.ylabel('Effectuvebess')
plt.xlabel('Dose')

### Trained Lightning NN

class BasicLightningTrain(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.w00 = nn.Parameter(torch.tensor(1.7),requires_grad=False) # don't optims it
        self.b00 = nn.Parameter(torch.tensor(-0.85),requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8),requires_grad=False)
        self.w10 = nn.Parameter(torch.tensor(12.6),requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.),requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7),requires_grad=False)
        self.final_bias = nn.Parameter(torch.tensor(0.),requires_grad=True)
        
        self.learning_rate = 0.01


    def forward(self, input):
        input_to_top_relu = input*self.w00+self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scale_top_relu_output = top_relu_output*self.w01
        
        input_to_bottom_relu = input*self.w10+self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scale_bottom_relu_output = bottom_relu_output*self.w11
        
        input_to_final_relu = scale_top_relu_output+scale_bottom_relu_output+self.final_bias
        output = F.relu(input_to_final_relu)
        return output
    
    
        
model = BasicLightningTrain()
output_values = model(input_doses)
sns.set_theme(style="whitegrid")
sns.lineplot(x=input_doses, y=output_values.detach(), color='green',linewidth=2.5)
plt.ylabel('Effectuvebess')
plt.xlabel('Dose')

class BasicLightningTrain_Opt(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.w00 = nn.Parameter(torch.tensor(1.7),requires_grad=False) # don't optims it
        self.b00 = nn.Parameter(torch.tensor(-0.85),requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8),requires_grad=False)
        self.w10 = nn.Parameter(torch.tensor(12.6),requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.),requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7),requires_grad=False)
        self.final_bias = nn.Parameter(torch.tensor(0.),requires_grad=True)
        
        self.learning_rate = 0.01


    def forward(self, input):
        input_to_top_relu = input*self.w00+self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scale_top_relu_output = top_relu_output*self.w01
        
        input_to_bottom_relu = input*self.w10+self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scale_bottom_relu_output = bottom_relu_output*self.w11
        
        input_to_final_relu = scale_top_relu_output+scale_bottom_relu_output+self.final_bias
        output = F.relu(input_to_final_relu)
        return output
        
    def configure_optimizers(self):
        return(SGD(self.parameters(), lr=self.learning_rate))
    
    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss = (output_i-label_i)**2
        return loss    
    
model = BasicLightningTrain_Opt()
trainer = L.Trainer(max_epochs=2000)

tuner = L.pytorch.tuner.Tuner(trainer)
lr_find_results = tuner.lr_find(model, 
                                       train_dataloaders = dataloader,
                                       min_lr = 0.001,
                                       max_lr = 1.,
                                       early_stop_threshold = None)
new_lr = lr_find_results.suggestion()
print(f"lr_find suggests {new_lr:.5f} for the learning rate.")

model.learning_rate = new_lr
trainer.fit(model, train_dataloaders = dataloader)
print(model.final_bias.data)

output_values = model(input_doses)
sns.set_theme(style="whitegrid")
sns.lineplot(x=input_doses, y=output_values.detach(), color='green',linewidth=2.5)
plt.ylabel('Effectuvebess')
plt.xlabel('Dose')

### GPU speedup

model = BasicLightningTrain_Opt()

### difference
trainer = L.Trainer(max_epochs=2000,accelerator='auto', devices='auto')

tuner = L.pytorch.tuner.Tuner(trainer)
lr_find_results = tuner.lr_find(model, 
                                       train_dataloaders = dataloader,
                                       min_lr = 0.001,
                                       max_lr = 1.,
                                       early_stop_threshold = None)
new_lr = lr_find_results.suggestion()
print(f"lr_find suggests {new_lr:.5f} for the learning rate.")


model.learning_rate = new_lr
trainer.fit(model, train_dataloaders = dataloader)
print(model.final_bias.data)