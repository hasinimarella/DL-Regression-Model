# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression problems involve predicting a continuous output variable based on input features. Traditional linear regression models often struggle with complex patterns in data. Neural networks, specifically feedforward neural networks, can capture these complex relationships by using multiple layers of neurons and activation functions. In this experiment, a neural network model is introduced with a single linear layer that learns the parameters weight and bias using gradient descent.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: Generate Dataset

Create input values  from 1 to 50 and add random noise to introduce variations in output values .

### STEP 2: Initialize the Neural Network Model

Define a simple linear regression model using torch.nn.Linear() and initialize weights and bias values randomly.

### STEP 3: Define Loss Function and Optimizer

Use Mean Squared Error (MSE) as the loss function and optimize using Stochastic Gradient Descent (SGD) with a learning rate of 0.001.

### STEP 4: Train the Model

Run the training process for 100 epochs, compute loss, update weights and bias using backpropagation.

### STEP 5: Plot the Loss Curve

Track the loss function values across epochs to visualize convergence.

### STEP 6: Visualize the Best-Fit Line

Plot the original dataset along with the learned linear model.

### STEP 7: Make Predictions

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:MARELLA HASINI

### Register Number:212223240083

```
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
X = torch.linspace(1,70,70).reshape(-1,1)
torch.manual_seed(71)
e = torch.randint(-8,9,(70,1),dtype=torch.float)
y = 2*X + 1 + e
print(y.shape)
plt.scatter(X.numpy(), y.numpy(),color='red')  # Scatter plot of data points
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Data for Linear Regression')
plt.show()
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
torch.manual_seed(59)
model = nn.Linear(1, 1)
print('Weight:', model.weight.item())
print('Bias:  ', model.bias.item())
loss_function = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
epochs = 50
losses = []

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_function(y_pred, y)
    losses.append(loss.item())

    loss.backward()
    optimizer.step()


    print(f'epoch: {epoch:2}  loss: {loss.item():10.8f}  '
          f'weight: {model.weight.item():10.8f}  '
          f'bias: {model.bias.item():10.8f}')
plt.plot(range(epochs), losses)
plt.ylabel('Loss')
plt.xlabel('epoch');
plt.show()

x1 = torch.tensor([X.min().item(), X.max().item()])


w1, b1 = model.weight.item(), model.bias.item()


y1 = x1 * w1 + b1

print(f'Final Weight: {w1:.8f}, Final Bias: {b1:.8f}')
print(f'X range: {x1.numpy()}')
print(f'Predicted Y values: {y1.numpy()}')

plt.scatter(X.numpy(), y.numpy(), label="Original Data")
plt.plot(x1.numpy(), y1.numpy(), 'r', label="Best-Fit Line")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Trained Model: Best-Fit Line')
plt.legend()
plt.show()
```

### Dataset Information
<img width="697" height="505" alt="image" src="https://github.com/user-attachments/assets/d0678b4b-6701-4d1a-94fd-f8c17bcdfe53" />


### OUTPUT
## Training Loss Vs Iteration Plot

<img width="576" height="587" alt="image" src="https://github.com/user-attachments/assets/7be08282-e126-4721-bd57-ff844285b65b" />

## Best Fit line plot

<img width="742" height="577" alt="image" src="https://github.com/user-attachments/assets/08336734-f560-4660-8b90-377bb00a0513" />

<img width="785" height="535" alt="Screenshot 2025-08-26 134409" src="https://github.com/user-attachments/assets/232563b1-aeef-45db-bdbc-804619c56464" />

### New Sample Data Prediction

<img width="493" height="80" alt="image" src="https://github.com/user-attachments/assets/5574d320-19ec-447c-a585-821e3065aa0c" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
