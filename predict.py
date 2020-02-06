# -*- coding=utf8 -*-
'''
@Filename  : predict.py
@Author    : Gaozong
@Date      : 2020-02-05
@Contact   : zong209@163.com
@Describe  : Predict cat or Dog
'''
import time
import torch
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import argparse
from torchvision import transforms
from PIL import Image
from model import AlexNet

test_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.255])
])


def get_labels(class_data_dir="data/train"):
    import os
    return os.listdir(class_data_dir)


def predict(image_path):
    start_time = time.time()
    image = Image.open(image_path)
    inputs = test_transforms(image)
    outputs = net(inputs.reshape((1, 3, 224, 224)))
    out_index = torch.argmax(outputs, 1).tolist()[0]
    outputs = F.softmax(outputs, 1).tolist()[0]
    str_result = "{}:{:.4f}, spend:{:.4f}s".format(labels[out_index],
                                                   outputs[out_index],
                                                   time.time() - start_time)
    plt.imshow(image)
    plt.text(10, 10, str_result, color="red")
    plt.show()
    print(str_result)


class FeatureVisualization():
    def __init__(self, net, img_path):
        self.net = net
        self.img_path = img_path
        self.pretrained_model = net.features

    def process_image(self):
        image = Image.open(image_path)
        inputs = test_transforms(image)
        self.origin_img = image
        return inputs.reshape((1, 3, 224, 224))

    def get_feature(self, selected_layer=None):
        features_list = list()
        # input = Variable(torch.randn(1, 3, 224, 224))
        input = self.process_image()
        # print(input.shape)
        x = input
        for index, layer in enumerate(self.pretrained_model):
            x = layer(x)
            if (selected_layer is not None and index == selected_layer):
                return x
            features_list.append(x)
        return features_list

    def get_single_feature(self, selected_layer=None):
        features = self.get_feature(selected_layer).data.numpy()
        # print(features.shape)

        feature = features[:, 0, :, :]
        # print(feature.shape)

        feature = feature.reshape((feature.shape[1], feature.shape[2]))
        # print(feature.shape)

        return self.feature_to_array(feature)

    def get_all_feature(self):
        features = self.get_feature()
        feature_list = list()
        for feature_map in features:
            feature = feature_map.data.numpy()
            feature = feature[:, 0, :, :]
            # print(feature.shape)
            feature = feature.reshape((feature.shape[1], feature.shape[2]))
            print(feature.shape)
            feature_array = self.feature_to_array(feature)
            feature_list.append(feature_array)
        return feature_list

    def feature_to_array(self, feature):
        # use sigmod to [0,1]
        feature = 1.0 / (1 + np.exp(-1 * feature))

        # to [0,255]
        feature = np.round(feature * 255)
        # print(feature[0])
        return feature


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Predict cat or dog in image")
    parser.add_argument("--image_path",
                        "-i",
                        required=True,
                        help="image path need to detect",
                        default="data/test/456.jpg")
    parser.add_argument("--model_file",
                        "-m",
                        required=True,
                        help="model file path")
    parser.add_argument("--features",
                        '-f',
                        help="show features type",
                        default=None)

    args = parser.parse_args()

    image_path = args.image_path
    pt_file = args.model_file
    # image_path = "data/test/456.jpg"
    # pt_file = "models/1580889991_alexnet_10_16.pt"
    labels = get_labels()
    net = AlexNet(num_classes=2)
    print("=> Loading model ...")
    checkpoint = torch.load(pt_file)
    net.load_state_dict(checkpoint['model_state_dict'])
    print("=> Weights loaded")

    if not args.features:
        predict(image_path)
    elif args.features == "all":
        features_map = FeatureVisualization(net, image_path)
        # # show all features
        features = features_map.get_all_feature()
        plt.figure()
        feature_columns = math.ceil((len(features) + 1) / 3)
        plt.subplot(3, feature_columns, 1)
        plt.imshow(features_map.origin_img)
        for index, image in enumerate(features):
            plt.subplot(3, feature_columns, index + 2)
            plt.imshow(image)
        plt.show()
    elif int(args.features):
        features_map = FeatureVisualization(net, image_path)
        # show single feature
        conv1_features = features_map.get_single_feature(int(args.features))
        plt.imshow(Image.fromarray(conv1_features))
        plt.show()
    else:
        print("[Error] Unrecognize value of params 'features'")
