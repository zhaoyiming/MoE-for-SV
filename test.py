from __future__ import print_function
import os
import torch
import torch.nn.functional as F
import pandas as pd

from models.resnet56_moe_debug import  resnet56, L1_loss
from config.config_moe_origin import Config as Config_origin
from config.config_moe_wide_origin import Config as Config_wide_origin
from config.config_moe_wide_01 import Config as Config_wide_01
from models.resnet56 import resnet_origin, resnet_wide_origin


import torchvision
import torch
import numpy as np
import time
from torch.nn import DataParallel
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# from thop import profile
from fvcore.nn import parameter_count_table, flop_count_table, FlopCountAnalysis

def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    res=np.sum([p.numel() for p in model.parameters()]).item()
    return res/1024/1024



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def get_data():
    train_dataset = datasets.CIFAR10(root=opt.data_path,
    train=True, 
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(), 
                                ]),
    download=False)

    test_dataset = datasets.CIFAR10(root=opt.data_path,
    train=False,
    transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    download=False)

  
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.train_batch_size, shuffle=True, drop_last=True, num_workers=opt.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, drop_last=True, num_workers=opt.num_workers, pin_memory=True)
    return train_loader, test_loader

def val_origin(test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    end = time.time()

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device)


            output= model(data_input)
            loss = criterion(output, label) 

            
            prec1 = accuracy(output.float().data, label)[0]
            losses.update(loss.item(), data[0].size(0))
            top1.update(prec1.item(), data[0].size(0))
            


            batch_time.update(time.time() - end)
            end = time.time()

            if i % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(test_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}\t'
          'Time ({batch_time.avg:.3f})\t'
          .format(top1=top1, batch_time=batch_time))
    return top1.avg



   
def val(test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    end = time.time()

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device)


            output, _, _ = model(data_input)
            

      
            # loss_basenet = opt.para_miu * criterion(output_basenet, label)
            # loss_moe = opt.para_lambda * l1_loss(e)
            loss = criterion(output, label) 


            
            prec1 = accuracy(output.data, label)[0]
            losses.update(loss.data.item(), data[0].size(0))
            top1.update(prec1.item(), data[0].size(0))
            


            batch_time.update(time.time() - end)
            end = time.time()

            if i % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
         
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(test_loader), batch_time=batch_time, loss=losses, 
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}\t'
          'Time ({batch_time.avg:.3f})\t'
          .format(top1=top1, batch_time=batch_time))
    return top1.avg

    



if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    global best_prec1
    best_prec1 = 0
    opt = Config_wide_01()
    
    device = torch.device("cuda:0" if opt.USE_CUDA else "cpu")

    _, test_loader=get_data()

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    if opt.backbone == 'resnet110':
            model = resnet110(num_classes=opt.num_classes, w_base=opt.w_base, embedding_size=opt.embedding_size).to(device)
    elif opt.backbone == 'resnet56':
            model = resnet56(num_classes=opt.num_classes, w_base=opt.w_base, embedding_size=opt.embedding_size).to(device)


    criterion = torch.nn.CrossEntropyLoss().to(device)
    l1_loss = L1_loss().to(device)
    # model = DataParallel(model)
    # model.module.set_freeze()
   
    # model=resnet_wide_origin().to(device)
    # model=resnet_origin().to(device)
    # input1=torch.rand(1, 3, 32, 32).to(device)
    # input2=[torch.rand(1, opt.embedding_size).to(device) for i in range(36)]

    # from thop import profile
    # flops, params = profile(model, inputs=(input1,))
    # print("flops: ", flops/1024/1024/1024, ", params: ", params/1024/1024)

    # flops1, params1 = profile(model1, inputs=(input1,))
    # print("flops: ", flops1/1024/1024/1024, ", params: ", params1/1024/1024)


    # model=resnet_origin().to(device)



    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)
    
  
    
    print("loading best checkpoint...")


    checkpoint = torch.load(opt.path_checkpoint)  # 加载断点
    model.load_state_dict(checkpoint['state_dict'])  # 加载模型可学习参数
    optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
    scheduler.load_state_dict(checkpoint['scheduler'])


    # flops_model = flop_count_table(FlopCountAnalysis(model, input1))
    # # params_model = parameter_count_table(model)
    # # print(params_model)
    # # print(params_count(model))
    # print(flops_model)
  

    # flops_model1 = flop_count_table(FlopCountAnalysis(model1, input1))
    # print(flops_model1)
    # # print(params_count(model1))

    # exit()



    start = time.time()
    val(test_loader, model, criterion)
    # val_origin(test_loader, model, criterion)
    print(time.time() - start)
    