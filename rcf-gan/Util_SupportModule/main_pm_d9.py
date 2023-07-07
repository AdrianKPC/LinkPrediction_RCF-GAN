#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import matplotlib.pyplot as plt
from cfgan_model_d9 import Model
#import time
#import os
#import matplotlib.pyplot as plt
#import numpy as np
#import scipy.sparse as sp
#import torch
#import torch.optim as optim
#import torch.nn as nn
#import torch.nn.init as init
#from torch.optim import lr_scheduler
#from loss_function import CFLossFunc
#from network_module import SampleNet, AdvNet, InnerProductDecoder
#from utility import prepare_dir, get_input_white_noise, get_graph_target, record_as_img, avg_record, \
#    tensorboard_img_writer, mask_test_edges, preprocess_graph, get_roc_score, normalise_adj
#from progress.bar import Bar as Bar
#from torch.utils.tensorboard import SummaryWriter
# state for using GPU-1
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np

# training epoch
#epoch = 50

epoch = 50

load = False

# hyperparameters for target image data
target_source = 'pubmed'  # cora citeseer pubmed nell.0.1
#target_source = 'AMiner'  # cora citeseer pubmed
target_batch_size = 64
feature_dim = 128
#input, ie noise dimension
target_size = 0  # 0 means using all data samples

ae_loss_reg = np.linspace(0.6, 1, 300)

# hyperparameters for generating nets (Note: activations are only valid for dcgan)
generator_training_param = {'input_noise_dim': feature_dim,
                            'input_noise_var': 0.3, 'net_type': 'gnn', 'lr': 2e-4, 'inner_ite_per_batch': 1,
                            'activations': [('relu', None), ('sigmoid', None)], 'net_dim': [128],
                            'weight_decay': 0, 'lr_step_size_decay': 0, 'lr_decay_gamma': 0.5}

#changes net_dim from 512 to 128

# Hyperparameters for adversarial nets (Note: activations_a are only valid for dcgan)
adversarial_training_param = {'input_t_dim': feature_dim, 'input_t_batchsize': target_batch_size,
                              'input_t_var': 1, 'net_type': 'gnn', 'lr': 2e-4, 'inner_ite_per_batch': 1,
                              'activations_a': [('lrelu', 0.2), ('tanh', None)], 'net_dim': [128],
                              'weight_decay': 0, 'lr_step_size_decay': 0, 'lr_decay_gamma': 0.5,
                              'adv_t_sigma_num': target_batch_size,
                              'activations_t': [('lrelu', 0.2), ('tanh', None)]}
#changes net_dim from 512 to 128

# hyperparameters for CFLossFunc
loss_alpha = 1  # amplitude
loss_beta =  1 # phase
# threshold is useless when normalization is None
loss_type = None
loss_threshold = 1

# training
mark = 'link_prediction'
if adversarial_training_param['adv_t_sigma_num'] > 0:
    model_label = 'gl_' + target_source + '_t_net_' + generator_training_param['net_type'] + '_' + mark
else:
    model_label = 'gl_' + target_source + '_t_normal_' + generator_training_param['net_type'] + '_' + mark



rcfmodel = Model(model_label,
              target_source, target_size, target_batch_size,
              adversarial_training_param, generator_training_param,
              loss_type, loss_alpha, loss_beta, loss_threshold, ae_loss_reg,
              epoch)

rcfmodel.train(load)
#training_accuracy = rcfmodel.roc_score
#
#plt.subplots(1, figsize=(10,10))
##        plt.plot(epoch, roc_score, '--', color="#111111",  label="ROC score")
#plt.plot(epoch, training_accuracy, color="#111111", label="AP score")
#
##    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
##    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")
#
#plt.title("Cora Training")
#plt.xlabel("Epoch"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
#plt.tight_layout()
#plt.show()

#plt.subplots(1, figsize=(10,10))
#plt.plot(epoch, roc_score, '--', color="#111111",  label="ROC score")
#plt.plot(epoch, ap_score, color="#111111", label="AP score")
#
##    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
##    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")
#
#plt.title("Cora Training")
#plt.xlabel("Epoch"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
#plt.tight_layout()
#plt.show()
