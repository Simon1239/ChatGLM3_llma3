#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import swanlab
swanlab.login("DDSBZOY03klTgFwA6u0X9")

num_epochs = 20
lr = 1e-4
batch_size = 8
num_classes = 2
device = 'cuda'
swanlab.init(
    experiment_name='ResNet50', 
    description='Train ResNet50 for Cats and Dogs classification.',
    config = {
        "model":"resnet50",
        "optim":"Adam",
        "lr":lr,
        "batch_size":batch_size,
        "num_epochs":num_epochs,
        "num_classes":num_classes,
        "device":device
    }
)

import get_data
import torch
import torchvision
from torchvision.models import ResNet50_Weights

train_dataset = get_data.DatasetLoader(get_data.ms_train_dataset)
train_loader = (torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True))

model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, num_classes)
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for iter, (inputs, labels) in enumerate(train_loader):
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}'.format(
        iter + 1, num_epochs, iter + 1, len(train_loader), loss.item()
    ))
    swanlab.log({"train_loss": loss.item()})