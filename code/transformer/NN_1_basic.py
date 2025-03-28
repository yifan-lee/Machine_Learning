import torch
import torch.nn as nn
import torch.nn.functional as F # activation function ReLU
from torch.optim import SGD # stochastic gradient descent

import matplotlib.pyplot as plt
import seaborn as sns


### Parameters

input_doses = torch.linspace(start=0, end=1, steps=11)
inputs = torch.tensor([0.,0.5,1.])
labels = torch.tensor([0.,1.,0.])

### Basic NN Model

class BasicNN(nn.Module):
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
    


model = BasicNN()
output_values = model(input_doses)
sns.set_theme(style="whitegrid")
sns.lineplot(x=input_doses, y=output_values, color='green',linewidth=2.5)
plt.ylabel('Effectuvebess')
plt.xlabel('Dose')
plt.title('Basic NN model')
plt.show()



### Trained NN model

class BasicNN_train(nn.Module):
    def __init__(self):
        super().__init__()
        self.w00 = nn.Parameter(torch.tensor(1.7),requires_grad=False) # don't optims it
        self.b00 = nn.Parameter(torch.tensor(-0.85),requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8),requires_grad=False)
        self.w10 = nn.Parameter(torch.tensor(12.6),requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.),requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7),requires_grad=False)
        self.final_bias = nn.Parameter(torch.tensor(0.),requires_grad=True)


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
    
model = BasicNN_train()
output_values = model(input_doses)
sns.set_theme(style="whitegrid")
sns.lineplot(x=input_doses, y=output_values.detach(), color='green',linewidth=2.5)
plt.ylabel('Effectuvebess')
plt.xlabel('Dose')
plt.title('Untrained NN model')
plt.show()

optimizer = SGD(model.parameters(), lr=0.1)
print(f"Final bias, before optimization: {model.final_bias.data}\n")
for epoch in range(100):
    total_loss = 0
    for iteration in range(len(inputs)):
        input_i = inputs[iteration]
        label_i = labels[iteration]
        output_i = model(input_i)
        loss = (output_i-label_i)**2
        loss.backward() # backward pass
        total_loss += float(loss)
    if total_loss < 0.0001:
        print(f"Num steps: {epoch}")
        break
    
    optimizer.step() # update parameters
    optimizer.zero_grad() # zero the gradients
    print(f"Step: {epoch} Final bias: {model.final_bias.data}\n")
print(f"Total loss: {total_loss}\n")
print(f"Final bias, after optimization: {model.final_bias.data}\n")

output_values = model(input_doses)
sns.set_theme(style="whitegrid")
sns.lineplot(x=input_doses, y=output_values.detach(), color='green',linewidth=2.5)
plt.ylabel('Effectuvebess')
plt.xlabel('Dose')
plt.title('Trained NN model')
plt.show()