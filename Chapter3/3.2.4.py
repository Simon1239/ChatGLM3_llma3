#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : 3.2.4.py

import torch
import swanlab
import get_data
import torchvision
from torchvision.models import ResNet50_Weights

def train(model, device, train_dataloader, optimizer, criterion, epoch, num_epochs):
    model.to(device)
    for iter, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(
            'Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, iter + 1, len(train_dataloader), loss.item())
        )
        swanlab.log({"train_loss": loss.item()})



def test(model, device, test_dataloader, epoch):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print("Accuracy: {:.2f}%".format(accuracy))
    swanlab.log({"test_acc": 100 * correct / total})

if __name__=="__main__":
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

    train_dataset = get_data.DatasetLoader(get_data.ms_train_dataset)
    train_loader = (torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True))
    model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    train(model, 'cuda', train_loader, torch.optim.Adam(model.parameters(), lr=1e-4), torch.nn.CrossEntropyLoss(), 0, 400)

