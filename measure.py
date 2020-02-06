# -*- coding=utf8 -*-
'''
@Filename  : measure.py
@Author    : Gaozong
@Date      : 2020-02-06
@Contact   : zong209@163.com
@Describe  : Plot train-loss & valid-loss
'''
import argparse
import numpy as np
import matplotlib.pyplot as plt


def get_loss_data(file_path):
    label = list()
    data = list()
    file = open(file_path, 'r')
    for index, line in enumerate(file.readlines()):
        if index == 0:
            label = line.split()
        else:
            data.append([float(x) for x in line.split()])
    x_data = np.array(data)[:, 0]
    y_data_1 = np.array(data)[:, 1]
    y_data_2 = np.array(data)[:, 2]

    return label, x_data, y_data_1, y_data_2


def plt_loss(train_loss_file):
    label, x, y1, y2 = get_loss_data(train_loss_file)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y1, '-', label=label[1])

    ax2 = ax.twinx()
    ax2.plot(x, y2, '-r', label=label[2])

    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid()

    ax.set_xlabel(label[0])
    ax.set_ylabel(label[1])
    ax2.set_ylabel(label[2])

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", '-l', help="loss log file path")

    args = parser.parse_args()
    log_path = args.logs
    # plt_loss("logs/train_logs_2020-02-05-21:53:33.log")
    plt_loss(log_path)
