import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import types
import os
import numpy as np
import torchvision

from torch.utils.data import DataLoader
from torchvision.datasets import  CIFAR10, MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import transforms
from tensorboardX import SummaryWriter
from model import AlexNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def load_dataset(batch_size):
    data_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, ))])
    trainset = torchvision.datasets.MNIST("data/mnist", True, data_transform, download=True)
    testset = torchvision.datasets.MNIST("data/mnist", False, data_transform, download=False)

    train_loader = DataLoader(trainset,
                              batch_size,
                              shuffle=True,
                              num_workers=2)
    test_loader = DataLoader(testset,
                             batch_size,
                             shuffle=False,
                             num_workers=2)

    return train_loader, test_loader



# 学习率衰减
def adjust_learning_rate(optimizer, shrink_factor):
    print("DECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr']))
    


# 训练模型
def train(model, train_loader, optimizer, criterion, epoch, scheduler=None, wm_loader=None):
    model.train()
    loss_meter = 0  # 记录一代的所有loss
    acc_meter = 0  # 记录一代的所有acc
    runcount = 0
    iter_wm_loader = iter(wm_loader) if wm_loader is not None else None
    
    for i, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        
        if iter_wm_loader is not None:
            try:
                wm_data, wm_target = next(iter_wm_loader)
            except StopIteration:
                iter_wm_loader = iter(wm_loader)
                wm_data, wm_target = next(iter_wm_loader)
            wm_data = wm_data.to(device)
            wm_target = wm_target.to(device)
            data = torch.cat([data, wm_data], dim=0)
            target = torch.cat([target, wm_target], dim=0)
        
        optimizer.zero_grad()
        pred = model(data)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()
        if scheduler: scheduler.step()
        pred = pred.max(1, keepdim=True)[1] # 找到概率最大的下标
        
        loss_meter += loss.item()
        acc_meter += pred.eq(target.view_as(pred)).sum().item()
        runcount += data.size(0)
        
        print(f'Epoch {epoch} [{i}/{len(train_loader)}]  Loss: {loss_meter/(i+1):.4f}  Acc: {acc_meter/runcount:.4f}', end='\r')
        
    loss_meter /= (i+1)
    acc_meter /= runcount
    print()
    
    return {'loss': loss_meter,
            'acc': acc_meter}



# 测试模型
def test(model, test_loader, criterion):
    model.eval()
    loss_meter = 0
    acc_meter = 0
    runcount = 0
    
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)
            pred = model(data)
            loss = criterion(pred, target)
            pred = pred.max(1, keepdim=True)[1] # 找到概率最大的下标

            loss_meter += loss.item()
            acc_meter += pred.eq(target.view_as(pred)).sum().item()
            runcount += data.size(0)
        
    loss_meter /= (i+1)
    acc_meter /= runcount
    
    print(f'Test result:   Loss: {loss_meter:.4f}   Acc: {acc_meter:.4f}')
    print()
    
    return {'loss': loss_meter,
            'acc': acc_meter}



outf = f'logs'
while os.path.exists(outf): outf += '_'
os.mkdir(outf)
batch_size = 128
learning_rate = 0.01
epochs = 100
start_epoch = 1
check_point = ''

train_loader, test_loader = load_dataset(batch_size)
model = AlexNet().to(device)
if check_point:
    model.load_state_dict(torch.load(check_point))

optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
criterion = nn.CrossEntropyLoss()

writer = SummaryWriter(outf+'/exp')
best_acc = float('-inf')
best_since_last = 0
for ep in range(start_epoch, epochs + 1):
    if best_since_last == 20: break
    elif best_since_last % 8 == 0 and best_since_last != 0: 
        adjust_learning_rate(optimizer, 0.5)
        
    train_metrics = train(model, train_loader, optimizer, criterion, ep)
    test_metrics = test(model, test_loader, criterion)
#     print(train_metrics)
#     print(test_metrics)
    writer.add_scalar('train-acc', train_metrics['acc'], global_step=ep)
    writer.add_scalar('train-loss', train_metrics['loss'], global_step=ep)
    writer.add_scalar('test-acc', test_metrics['acc'], global_step=ep)
    writer.add_scalar('test-loss', test_metrics['loss'], global_step=ep)

    if best_acc < test_metrics['acc']:
        best_since_last = 0
        print(f'Found best at epoch {ep}\n')
        best_acc = test_metrics['acc']
        torch.save(model.cpu().state_dict(), outf+'/best.pth')
    else: best_since_last += 1

    torch.save(model.cpu().state_dict(), outf+'/last.pth')
    model.to(device)