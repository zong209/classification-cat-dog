# -*- coding=utf8 -*-
'''
@Filename  : main.py
@Author    : Gaozong
@Date      : 2020-02-04
@Contact   : zong209@163.com
@Describe  : Entry function
'''
import os
import torch
import numpy as np
import time
import torch.optim as optim
from torch import nn
from dataset import AnimalDataset
from torchvision import transforms

CUDA_AVALIABLE = torch.cuda.is_available()


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


def adjust_learning_rate(optimizer, epoch, lr, lr_steps, weight_decay):
    """Sets the learning rate to the initial LR decayed by 10 every step"""
    decay = 0.1**(sum(epoch >= np.array(lr_steps)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']
        param_group['weight_decay'] = decay * param_group['weight_decay']
    return optimizer


def train(pertrained=False, resume_file=None):
    if pertrained:
        from model import alexnet
        net = alexnet(pretrained=True, num_classes=NUMBER_CLASSES)
    else:
        from model import AlexNet
        net = AlexNet(num_classes=NUMBER_CLASSES)
    valid_precision = 0
    policies = net.parameters()

    optimizer = optim.SGD(policies,
                          lr=LR,
                          momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY)

    train_log = open(
        "logs/train_logs_{}.log".format(
            time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())), "w")
    valid_log = open(
        "logs/valid_logs_{}.log".format(
            time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())), "w")
    train_log.write("{}\t{}\t{}\n".format("epoch", "losses ", "correct"))
    valid_log.write("{}\t{}\t{}\n".format("epoch", "losses ", "correct"))
    # 恢复训练
    if resume_file:
        if os.path.isfile(resume_file):
            print(("=> loading checkpoint '{}'".format(resume_file)))
            checkpoint = torch.load(resume_file)
            start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['model_state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})".format(
                resume_file, checkpoint['epoch'])))
    else:
        start_epoch = 0
        print(("=> no checkpoint found at '{}'".format(resume_file)))

    # valid_precision = valid(net)
    for epoch in range(start_epoch, EPOCHES):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        correct = AverageMeter()
        end = time.time()

        optimizer = adjust_learning_rate(optimizer, epoch, LR, LR_steps,
                                         WEIGHT_DECAY)

        for i_batch, sample_batched in enumerate(train_dataloader):
            # measure data loading time
            data_time.update(time.time() - end)
            inputs, labels = sample_batched
            if CUDA_AVALIABLE:
                outputs = net.forward(inputs.cuda())
                labels = labels.long().flatten().cuda()
            else:
                outputs = net.forward(inputs)
                labels = labels.long().flatten()

            outputs = outputs.reshape([-1, NUMBER_CLASSES])
            loss = criterion(outputs, labels)
            # 更新统计数据
            losses.update(loss.item(), inputs.size(0))
            _, predicted = torch.max(outputs.data, 1)
            # 计算准确率
            correct.update(
                (predicted == labels.long()).sum().item() / len(labels),
                inputs.size(0))

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i_batch % 10 == 0:
                print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                           epoch,
                           i_batch,
                           len(train_dataloader),
                           batch_time=batch_time,
                           data_time=data_time,
                           loss=losses,
                           top1=correct,
                           lr=optimizer.param_groups[-1]['lr'])))

        train_log.write("{:5d}\t{:.5f}\t{:.5f}\n".format(
            epoch, losses.avg, correct.avg))
        train_log.flush()

        if epoch % 1 == 0:
            valid_precision = valid(net, epoch, valid_log)
        # 保存网络
        if (epoch > 0 and epoch % 10 == 0) or epoch == EPOCHES - 1:
            save_path = os.path.join(
                "models",
                "{:d}_{}_{:d}_{:d}_{:.5f}.pt".format(int(time.time()),
                                                     "alexnet", epoch,
                                                     BATCHSIZE,
                                                     valid_precision))
            print("[INFO] Save weights to " + save_path)
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dir': optimizer.state_dict,
                    'loss': loss
                }, save_path)

    train_log.close()
    valid_log.close()


def valid(net, epoch=None, valid_log=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    correct = AverageMeter()
    net.eval()

    end = time.time()
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(valid_dataloader):

            inputs, labels = sample_batched
            if CUDA_AVALIABLE:
                outputs = net.forward(inputs.cuda())
                labels = labels.long().flatten().cuda()
            else:
                outputs = net.forward(inputs)
                labels = labels.long().flatten()

            outputs = outputs.reshape([-1, NUMBER_CLASSES])
            loss = criterion(outputs, labels)
            # 更新统计数据
            losses.update(loss.item(), inputs.size(0))
            _, predicted = torch.max(outputs.data, 1)
            # 计算准确率
            correct.update(
                (predicted == labels.long()).sum().item() / len(labels),
                inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i_batch % 10 == 0 or i_batch == len(valid_dataloader) - 1:
                print(('Test: [{0}/{1}]\t'
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                           i_batch + 1,
                           len(valid_dataloader),
                           batch_time=batch_time,
                           loss=losses,
                           top1=correct)))
    if valid_log:
        valid_log.write("{:5d}\t{:.5f}\t{:.5f}\n".format(
            epoch, losses.avg, correct.avg))
        valid_log.flush()
    return correct.avg


if __name__ == "__main__":
    TRAIN_DIR = "data/train"
    VALID_DIR = "data/val"
    NUMBER_CLASSES = 2
    BATCHSIZE = 64
    EPOCHES = 200
    LR = 0.01
    LR_steps = [60, 120]
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.005
    criterion = nn.CrossEntropyLoss()

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        # resize随机长宽比裁剪原始图片，最后将图片resize到设定好的size- 输出的分辨率 scale- 随机crop的大小区间
        transforms.RandomHorizontalFlip(),
        # 依据概率p对PIL图片进行水平翻转 0.5
        # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1] 注意事项：归一化至[0-1]是直接除以255
        transforms.ToTensor(),
        # 对数据按通道进行标准化，即先减均值，再除以标准差，注意是 hwc
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])
    ])
    valid_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])
    ])

    train_datasets = AnimalDataset(TRAIN_DIR,
                                   "jpg",
                                   transform=train_transforms)
    valid_datasets = AnimalDataset(VALID_DIR,
                                   "jpg",
                                   transform=valid_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_datasets,
                                                   batch_size=BATCHSIZE,
                                                   shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_datasets,
                                                   batch_size=BATCHSIZE,
                                                   shuffle=True)

    train(pertrained=False,
          resume_file="models/1580961759_alexnet_10_64_0.51562.pt")
