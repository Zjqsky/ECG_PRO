#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import monte_carlo

from read_data import read_a_txt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    pass

def get_probability(data, w, c0, c1, m=2):
    size = data.shape[0]
    Y = data * sigmoid(w - 0.5) / 100

    d0_2 = np.sum(((Y - c0) ** 2), axis=1).reshape(size, 1)
    d1_2 = np.sum(((Y - c1) ** 2), axis=1).reshape(size, 1)

    p_i0 = 1 / (1 + np.power((d0_2 / d1_2), 1 / (m - 1)))
    p_i1 = 1 / (1 + np.power((d1_2 / d0_2), 1 / (m - 1)))
    p = np.append(p_i0, p_i1, axis=1)
    return p

def show_encoder_spike_output(signalFile, spikeTrainFile):
    spikefile = open(spikeTrainFile)
    sigValues = []
    spiValues = []

    sigValues = read_a_txt(signalFile)
    spiValues = list(map(int, spikefile.read()))
    spikefile.close()

    x = range(0,2000)
    plt.subplot(211)
    plt.plot(x, sigValues[0:2000])
    plt.subplot(212)
    plt.plot(x, spiValues[0:2000])
    plt.show()
    pass

def show_neuron_spike_num(dataFile):
    data = pd.read_csv(dataFile, header = None)
    sns.heatmap(data)
    plt.show()
    pass

def show_roc_plot(dataFile, WCoC1File):
    print('getting spikes...')
    spikes = pd.read_csv(dataFile, header=None).values.T
    W, C0, C1 = pd.read_csv(WCoC1File, header=None).values[:, 1:]

    print('getting probability...')
    pro = get_probability(spikes, W, C0, C1)
    heartRate = np.sum(np.argmax(pro, axis = 1))
    trueCenter = 0
    if heartRate < (600 - heartRate):
        trueCenter = 1
    pro = pro[:, trueCenter]
    print('probability:',pro)

    print('getting label and AP,AN...')
    label = np.zeros([600])
    labelIndexes = pd.read_csv('caijing.f.txt',header=None).values[:, 0] * 4
    labelIndexes = labelIndexes.reshape(labelIndexes.shape[0])
    labelIndexes = labelIndexes[labelIndexes < 60000]
    for labelIndex in labelIndexes:
        label[labelIndex // 100] = 1
    AP = np.sum(label)
    AN = 600 - AP
    print('label:', label)
    print('AP:', AP, '\tAN:', AN)

    print('getting tpr and fpr...')
    tpr = np.zeros(101)
    fpr = np.zeros(101)
    for i in range(0, 101, 1):
        threshold = i / 100
        PP = np.sum(pro > threshold)
        TP = np.sum(label[np.where(pro > threshold)])
        FP = PP - TP
        tpr[i] = TP / AP
        fpr[i] = FP / AN
    print('tpr:', tpr)
    print('fpr:', fpr)
    plt.plot(fpr, tpr)
    plt.show()
    pass

if __name__ == '__main__':
    # show_encoder_spike_output('caijing.a.txt', 'caijing.or.txt')
    # show_neuron_spike_num('spike.txt')
    show_roc_plot('spike.txt','WCoC120180909094249.csv')
    pass
