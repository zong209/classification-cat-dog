目录

    ├── Readme.md
    ├── catdogclass.py 原始文件 by TZH
    ├── data 数据文件夹
    │ ├── test 测试数据
    │ ├── train 训练数据
    │ └── val 验证数据
    ├── dataset.py 数据类
    ├── main.py 训练入口文件
    ├── model.py 模型文件
    ├── models 模型存储路径
    │ └── pretrained 预训练模型
    ├── predict.py 预测函数
    └── requirements.txt 依赖

## 训练模型

### step1: 准备数据

数据如下放置: 以类别名命名对应数据文件夹

    ├── test
    │   ├── XXX.jpg
    │   └── XXX.jpg
    ├── train
    │   ├── cat
    │   │   └── XXX.jpg ...
    │   └── dog
    │   │   └── XXX.jpg ...
    └── val
        ├── cat
        │   └── XXX.jpg ...
        └── dog
            └── XXX.jpg ...

### step2: 准备预训练模型

    wget https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth -o models/pretrained/alexnet-owt-4df8aa71.pth

### step3: 模型训练

修改 `main.py` 中 最后一行为：

    train()

然后执行： `python main.py`

## 测试模型

### 测试单张图片

    python predict.py -i imagepath -m modle_file path

### 获取`feature map`

1. 获取指定层的`feature map`

   python predict.py -i imagepath -m modle_file path -f index of layer
   example:
   python predict.py -i data/test/123.jpg -m models/1580889991_alexnet_10_16.pt -f 1

   | features index | layer                       |
   | -------------- | --------------------------- |
   | 0              | nn.Conv2d(3, 64,11, 4, =2)  |
   | 1              | nn.ReLU                     |
   | 2              | nn.MaxPool2d(=3, 2)         |
   | 3              | nn.Conv2d(64, 192, =5, =2)  |
   | 4              | nn.ReLU                     |
   | 5              | nn.MaxPool2d(=3, 2)         |
   | 6              | nn.Conv2d(192, 384, =3, =1) |
   | 7              | nn.ReLU                     |
   | 8              | nn.Conv2d(384, 256, =3, =1) |
   | 9              | nn.ReLU                     |
   | 10             | nn.Conv2d(256, 256, =3, =1) |
   | 11             | nn.ReLU                     |
   | 12             | nn.MaxPool2d(=3, 2)         |

2. 获取所有层`feature map`

   python predict.py -i imagepath -m modle_file path -f all
   example:
   python predict.py -i data/test/123.jpg -m models/1580889991_alexnet_10_16.pt -f all
