# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 13:08:06 2020

@author: tjt
https://www.kaggle.com/c/dogs-vs-cats/data
利用torchvision.datasets中的ImageFolder类直接进行封装，
这个类直接将每一个文件夹下的图片与类别对应，返回结果为迭代器。
然后将这个迭代器传入dataloader，按每个batch提取数据
https://www.bilibili.com/video/av68084880/
https://www.jianshu.com/p/f804862a960a
https://hackmd.io/@lido2370/S1aX6e1nN?type=view
https://www.learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/
"""
'''
1 想看Net2的cnn过程中特征图（conv1 pool1 conv2 pool2 conv3 conv4 conv5 pool3 dropout1 fc1 dropout2 fc2 fc3
2 各种做图  plot train_losses test_losses 
3 inference的最好方法,不是用batch=1，目的一次仅识别指定的1个图片。现在预测一次时间比较长
4 提高精度
Train Epoch: 7 [1120/2491 (45%)]        Loss: 0.723704
Train Epoch: 7 [1280/2491 (51%)]        Loss: 0.626161
Train Epoch: 7 [1440/2491 (58%)]        Loss: 0.615381
Train Epoch: 7 [1600/2491 (64%)]        Loss: 0.679356
Train Epoch: 7 [1760/2491 (71%)]        Loss: 0.623481
Train Epoch: 7 [1920/2491 (77%)]        Loss: 0.662350
Train Epoch: 7 [2080/2491 (83%)]        Loss: 0.667878
Train Epoch: 7 [2240/2491 (90%)]        Loss: 0.689613
Train Epoch: 7 [2400/2491 (96%)]        Loss: 0.671595

Test set: Average loss: 0.6973, Accuracy: 193/384 (50%)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from torch.autograd import Variable
import numpy as np
from torchvision import models
import matplotlib.pyplot as plt
from PIL import Image

batch_size=16
train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),#resize随机长宽比裁剪原始图片，最后将图片resize到设定好的size- 输出的分辨率 scale- 随机crop的大小区间
        transforms.RandomHorizontalFlip(),#依据概率p对PIL图片进行水平翻转 0.5
        #将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1] 注意事项：归一化至[0-1]是直接除以255
        transforms.ToTensor(),
        #对数据按通道进行标准化，即先减均值，再除以标准差，注意是 hwc
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.255])
        ])
val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.255])

        ])
test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.255])

        ])
train_dir = "C:/coding/data1/train"
train_datasets=datasets.ImageFolder(train_dir,transform=train_transforms)
train_dataloader = torch.utils.data.DataLoader(train_datasets,batch_size=batch_size,shuffle=True)

val_dir = "C:/coding/data1/val"
val_datasets=datasets.ImageFolder(val_dir,transform=val_transforms)
val_dataloader = torch.utils.data.DataLoader(val_datasets,batch_size=batch_size,shuffle=True)


#https://blog.csdn.net/u012348774/article/details/90047539
class AlexNet(nn.Module):
    def __init__(self,num_classes=2):
        super(AlexNet,self).__init__()
        #self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=11,stride=4,padding=2)
        self.conv1 = nn.Conv2d(3,64,11,4,2)
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        #self.conv2 = nn.Conv2d(in_clannels=64,out_channels=192,kernel_size=5,padding=2)
        self.conv2 = nn.Conv2d(64,192,kernel_size=5,padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv3 = nn.Conv2d(192,384,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(384,256,kernel_size=3,padding=1)
        self.conv5 = nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3,stride=2)
        #torch.nn.Dropout对所有元素中每个元素按照概率0.5更改为零
        #torch.nn.Dropout2d是对每个通道按照概率0.5置为0
        self.dropout1 = nn.Dropout()
        self.fc1 = nn.Linear(in_features=256*6*6,out_features=4096)
        self.dropout2 = nn.Dropout()
        self.fc2 = nn.Linear(in_features=4096,out_features=4096)
        self.fc3 = nn.Linear(in_features=4096,out_features=num_classes)
    #https://pytorch.apachecn.org/docs/0.4/16.html  
    '''
    inplace为True，将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出。
    例子见：https://blog.csdn.net/tmk_01/article/details/80679991
    '''
    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x),inplace=True))
        x = self.pool2(F.relu(self.conv2(x),inplace=True))
        x = F.relu(self.conv3(x),inplace=True)
        x = F.relu(self.conv4(x),inplace=True)
        x = self.pool3(F.relu(self.conv5(x),inplace=True))
        x = x.view(x.size(0),256*6*6)
        x = F.relu(self.fc1(self.dropout1(x)),inplace=True)
        x = F.relu(self.fc2(self.dropout1(x)),inplace=True)
        x = self.fc3(x)
        return x
    
model = AlexNet()
#if torch.cuda.is_available():
#    model.cuda()
#momentum (float, 可选) – 动量因子（默认：0）参数将会使用lr的学习率，并且0.5的momentum将会被用于所 有的参数。
#PyTorch学习之6种优化方法介绍 https://zhuanlan.zhihu.com/p/62585696
optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def train(net,epoch):
    net.train()
    for batch_idx,(data,target) in enumerate(train_dataloader):
#        if torch.cuda.is_available():
#            data,target = data.cuda(),target.cuda()
        data,target = Variable(data),Variable(target)
        #optimizer.zero_grad()意思是把梯度置零，也就是把loss关于weight的导数变成0
        '''
        根据pytorch中的backward()函数的计算，当网络参量进行反馈时，梯度是被积累的而不是被替换掉；
        但是在每一个batch时毫无疑问并不需要将两个batch的梯度混合起来累积，因此这里就需要每个batch设置一遍zero_grad 了
        关于这一点可以参考：https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/3
        关于backward()的计算可以参考：https://discuss.pytorch.org/t/how-to-use-the-backward-functions-for-multiple-losses/1826
        '''
        optimizer.zero_grad()
        output = net(data)
        loss = F.cross_entropy(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
#            print('train epoch:{}[{}/{}({:.0f}%)]\tLoss:{:.6f}'.format(
#                    epoch,batch_idx*len(data),len(train_dataloader.dataset),
#                    100.* batch_idx/len(train_dataloader),loss.data[0]))
# IndexError: invalid index of a 0-dim tensor. Use tensor.item() to convert a 0-dim tensor to a Python number loss.data[0]--loss.item()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_dataloader.dataset),
                100. * batch_idx / len(train_dataloader), loss.item()))
def test(net):
    net.eval()
    test_loss = 0
    correct = 0
    for data,target in val_dataloader:
#        if torch.cuda.is_available():
#            data,target = data.cuda(),target.cuda()
        #因而volatile=True的节点不会求导，即使requires_grad=True，也不会进行反向传播，对于不需要反向传播的情景(inference，测试推断)，该参数可以实现一定速度的提升
        #https://xmfbit.github.io/2018/04/27/pytorch-040-migration-guide/
        #print(torch.__version__)
#        data,target = Variable(data,volatile=True),Variable(target)
#        https://blog.csdn.net/weixin_41797117/article/details/80237179
        with torch.no_grad():
            data,target = Variable(data),Variable(target)
            output = net(data)
            test_loss +=F.cross_entropy(output,target,size_average=False).item()
            #torch.max(input, dim, keepdim=False, out=None) -> (Tensor, LongTensor)
            #按维度dim 返回最大值 https://blog.csdn.net/Z_lbj/article/details/79766690
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(val_dataloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_dataloader.dataset),
        100. * correct / len(val_dataloader.dataset)))
for epoch in range(1,8):
    train(model,epoch)
    test(model)
#https://pytorch.org/docs/stable/torch.html?highlight=torch%20load#torch.load            
torch.save(model,'model.pkl')
Net2 = torch.load('C:/coding/model.pkl')
Net2 = torch.load('C:/coding/model.pkl',map_location='cpu') 
#按官网的推荐保持状态字典
#torch.save(model.state_dict(), 'model_state_dick.pkl')
#Net2 = AlexNet()
#Net2.load_state_dict(torch.load('model_state_dick.pkl'))
#微调参数
params = [{'params':md.parameters()} for md in Net2.children() if md in [Net2.fc1,Net2.fc2,Net2.fc3]]
optimizer = optim.SGD(params,lr=0.001,momentum=0.5)
for epoch in range(1,5):
    train(Net2,epoch)
    test(Net2)

#def predict_imgage(img):
#    
#    image_tensor = test_transforms(img)
#'''
#torch.unsqueeze()这个函数主要是对数据维度进行扩充。给指定位置加上维数为一的维度，
#比如原本有个三行的数据（3），在0的位置加了一维就变成一行三列（1,3）。
#a.squeeze(N) 就是在a中指定位置N加上一个维数为1的维度。
#还有一种形式就是b=torch.squeeze(a，N) a就是在a中指定位置N加上一个维数为1的维度
#'''

    
img = Image.open('C:/coding/data1/test/123.jpg')   
img_t = test_transforms(img)#torch.Size([3, 224, 224])
batch_t = torch.unsqueeze(img_t,0)#torch.Size([1, 3, 224, 224])
Net2.eval()
predict_imgage = Net2(batch_t)
print(predict_imgage.shape)
_, index = torch.max(predict_imgage, 1)
pred = predict_imgage.data.max(1)[1]
print('pred=',pred)
percentage = torch.nn.functional.softmax(predict_imgage, dim=1)[0] * 100
print(percentage)
_, indices = torch.sort(predict_imgage, descending=True)