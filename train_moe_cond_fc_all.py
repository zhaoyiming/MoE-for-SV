from __future__ import print_function
import os
import torch
import torch.nn.functional as F


from config.config_moe_cond_fc_all import Config
from models.resnet56_cond_fc_all import resnet_cond
from models.resnet56 import resnet56

import torchvision
import torch
import numpy as np
import time
from torch.nn import DataParallel
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# from fvcore.nn import parameter_count_table, flop_count_table, FlopCountAnalysis

def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    res=np.sum([p.numel() for p in model.parameters()]).item()
    return res/1024/1024


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


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

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

   
    model.train()
    end = time.time()

    for i, data in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        data_input, label = data
        data_input = data_input.to(device)
        label = label.to(device).long()


        output = model(data_input)
        loss = criterion(output, label) 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec1 = accuracy(output.float().data, label)[0]
        losses.update(loss.float().item(), data[0].size(0))

        top1.update(prec1.item(), data[0].size(0))
        


        top1.update(prec1.item(), data[0].size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
        
   
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
            label = label.to(device).long()


            output= model(data_input)
            

      

            loss = criterion(output, label)

            
            prec1 = accuracy(output.float().data, label)[0]
            losses.update(loss.float().item(), data[0].size(0))
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

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))
    return top1.avg

    



if __name__ == '__main__':
    # torch.backends.cudnn.benchmark = True

    global best_prec1
    best_prec1 = 0
    opt = Config()
    device = torch.device("cuda:0" if opt.USE_CUDA else "cpu")

    train_loader, test_loader=get_data()

    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    model=resnet_cond()
    criterion = torch.nn.CrossEntropyLoss().to(device)
    model.to(device)
    # model = DataParallel(model)

    input1=torch.rand(1, 3, 32, 32).to(device)
    flops_model = flop_count_table(FlopCountAnalysis(model, input1))
    print(flops_model)
  
    exit()



    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                        lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)


  
    start_epoch=0
    if opt.RESUME:
        print("loading best checkpoint...")
      
        checkpoint = torch.load(opt.path_checkpoint)  # 加载断点

        model.load_state_dict(checkpoint['state_dict'])  # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch
        scheduler.load_state_dict(checkpoint['scheduler'])
 


    start = time.time()

    for epoch in range(start_epoch, opt.max_epoch):
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch)
        scheduler.step()
        prec1 = val(test_loader, model, criterion)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        
        if epoch > 0 and epoch % opt.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(opt.save_dir, 'checkpoint.th'))
          
 

        if is_best :
            save_checkpoint({
            'epoch': epoch + 1,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(opt.save_dir, 'model.th'))


            with open(opt.save_dir+"/"+opt.backbone+"_"+opt.dataset+"_best", "a")as f:
                f.write("Epoch: "+str(epoch)+"\t Prec@1 "+ str(best_prec1)+"\n")
                f.close()
