import torch.nn.functional as F
import torch
from torch import nn
from collections import OrderedDict
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
#device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device='cuda'
from convknrm_model import CONVKNRM
from data_reader3 import DataAndQuery
import os
import numpy as np
import torch.nn.functional as F
import pandas as pd
import subprocess
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import random
import argparse
from collections import defaultdict
from utils import *

parser = argparse.ArgumentParser(description='Conv-KNRM-pointwise', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--emsize', type=int, default=50)
parser.add_argument('--device', type=int, default=0)

parser.add_argument('--nbins', type=int, default=5)

args = parser.parse_args()
print(torch.cuda.current_device())
torch.cuda.set_device(args.device)
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(args.device))
print(torch.cuda.is_available())
args.device='cuda:'+str(args.device)

#args.device='cpu'

out_str = str(args)
print(out_str)


def kernal_mus(n_kernels):
    """
    get the mu for each gaussian kernel. Mu is the middle of each bin
    :param n_kernels: number of kernels (including exact match). first one is exact match
    :return: l_mu, a list of mu.
    """
    l_mu = [1]
    if n_kernels == 1:
        return l_mu

    bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
    l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
    for i in range(1, n_kernels - 1):
        l_mu.append(l_mu[i] - bin_size)
    return l_mu


def kernel_sigmas(n_kernels):
    """
    get sigmas for each gaussian kernel.
    :param n_kernels: number of kernels (including exactmath.)
    :param lamb:
    :param use_exact:
    :return: l_sigma, a list of simga
    """
    bin_size = 2.0 / (n_kernels - 1)
    l_sigma = [0.001]  # for exact match. small variance -> exact match
    if n_kernels == 1:
        return l_sigma

    l_sigma += [0.1] * (n_kernels - 1)
    return l_sigma

args.mu = kernal_mus(args.nbins)
args.sigma = kernel_sigmas(args.nbins)


def evaluation_metrics(outputs,labels):
    avg = defaultdict(float)  # average
    tp = defaultdict(int)  # true positives
    tpfn = defaultdict(int)  # true positives + false negatives
    tpfp = defaultdict(int)  # true positives + false positives
    for index, y1 in enumerate(outputs):  # actual value, prediction
        y0=labels[index]
        tp[y0] += (y0 == y1)
        tpfn[y0] += 1
        tpfp[y1] += 1
    print()
    for y in sorted(tpfn.keys()):
        pr = (tp[y] / tpfp[y]) if tpfp[y] else 0
        rc = (tp[y] / tpfn[y]) if tpfn[y] else 0
        avg["macro_pr"] += pr
        avg["macro_rc"] += rc
        print("label = %s" % y)
        print("precision = %f (%d/%d)" % (pr, tp[y], tpfp[y]))
        print("recall = %f (%d/%d)" % (rc, tp[y], tpfn[y]))
        print("f1 = %f\n" % f1(pr, rc))
    avg["macro_pr"] /= len(tpfn)
    avg["macro_rc"] /= len(tpfn)
    avg["micro_f1"] = sum(tp.values()) / sum(tpfn.values())
    print("macro precision = %f" % avg["macro_pr"])
    print("macro recall = %f" % avg["macro_rc"])
    print("macro f1 = %f" % f1(avg["macro_pr"], avg["macro_rc"]))
    print("micro f1 = %f" % avg["micro_f1"])


def load_checkpoint(model, optimizer, losslogger, filename,testing_accuracy,start_epoch):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        losslogger = checkpoint['losslogger']
        testing_accuracy = checkpoint['testing_accuracy']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, losslogger,testing_accuracy

def load_checkpoint_for_eval(model, filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model


def test_output(test_iter, model):

    #model = load_checkpoint_for_eval(model, save_path)
    # move the model to GPU if has one
    model=model.to(args.device)

    # need this for dropout
    model.eval()

    epoch_loss = 0
    num_batches = len(test_iter)
    all_outputs = []
    all_labels=[]
    for batch_desc,batch_query, labels in test_iter:
        batch_desc, batch_query, labels = batch_desc.to(args.device), batch_query.to(args.device), labels.to(args.device)
        batch_query = torch.squeeze(batch_query,1)
        batch_query = torch.squeeze(batch_query, 1)
        batch_desc = torch.squeeze(batch_desc,1)
        batch_desc = torch.squeeze(batch_desc,1)

        outputs = model(batch_query, batch_desc).to(args.device)
        outputs=m(outputs)

        #loss = loss_function(outputs, labels.float())
        loss = loss_(outputs, labels.float())
        epoch_loss += loss.item()

        all_outputs += outputs.tolist()
        all_labels += labels.tolist()

    losslogger = epoch_loss / num_batches

    # Accuracy
    all_outputs = torch.Tensor(all_outputs)
    all_labels = torch.Tensor(all_labels)
    output = (all_outputs > 0.5).float()
    output = output.type(torch.int64)

    all_labels = all_labels.type(torch.int64)
    correct = (output == all_labels).float().sum()
    accuracy=correct / all_outputs.shape[0]
    print("Loss: {:.3f}, Test Accuracy: {:.3f}".format(losslogger,accuracy))
    evaluation_metrics(output.tolist(), all_labels.tolist())

    return all_outputs,losslogger,all_labels,accuracy

loss_function=nn.MSELoss()
batch_size=32

train_dataset = DataAndQuery(use_same_collection=True)
print(len(train_dataset))
train_iter = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

args.index_to_word=train_dataset.index_to_word
args.wv=train_dataset.wv

model = CONVKNRM(args).to(args.device)

optimizer = torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=1e-8)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-8)
losslogger=np.inf
best_accuracy=0
save_path = './stage2_model.pt'
start_epoch=0
NUM_EPOCH=100

model, optimizer, start_epoch, losslogger,best_accuracy=load_checkpoint(model, optimizer, losslogger, save_path,best_accuracy,start_epoch)


test_dataset = DataAndQuery(train_dataset.wv, train_dataset.word_to_index,
                                train_dataset.index_to_word,False,train_dataset.train_set,train_dataset.labels_set,use_same_collection=True,nb_all=train_dataset.nb_all,
                            nb_train=train_dataset.nb_train)
print(len(test_dataset))
test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

m = nn.Sigmoid() # initialize sigmoid layer
loss_ = nn.BCELoss() # initialize loss function

for epoch in range(start_epoch, NUM_EPOCH + start_epoch):
    model.train()
    epoch_loss = 0
    num_batches = len(train_iter)
    all_outputs = []
    all_labels=[]

    for batch_desc,batch_query, labels in train_iter:
        batch_desc, batch_query, labels = batch_desc.to(
            args.device), batch_query.to(args.device), labels.to(
            args.device)

        batch_query = torch.squeeze(batch_query, 1)
        batch_query = torch.squeeze(batch_query, 1)
        batch_desc = torch.squeeze(batch_desc, 1)
        batch_desc = torch.squeeze(batch_desc, 1)


        outputs = model(batch_query, batch_desc).to(args.device)
        outputs=m(outputs)
        all_outputs += outputs.tolist()
        all_labels += labels.tolist()


        loss = loss_(outputs, labels.float())

        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losslogger = epoch_loss / num_batches

    # Accuracy
    all_outputs=torch.Tensor(all_outputs)
    all_labels = torch.Tensor(all_labels)
    output = ( all_outputs> 0.5).float()
    output = output.type(torch.int64)

    all_labels=all_labels.type(torch.int64)
    correct = (output == all_labels).float().sum()
    print("Epoch {}/{}, Loss: {:.3f}, Train Accuracy: {:.3f}".format(epoch + 1, NUM_EPOCH, losslogger, correct / all_outputs.shape[0]))

    outputs_test, testing_loss, _,testing_accuracy = test_output(test_iter, model)


    # if testing_accuracy>best_accuracy:
    #     best_accuracy=testing_accuracy
    #     state = {'epoch': epoch+1, 'state_dict': model.state_dict(),
    #                  'optimizer': optimizer.state_dict(), 'losslogger': losslogger,'testing_accuracy':best_accuracy, }
    #     torch.save(state, save_path)

best_accuracy=testing_accuracy
state = {'epoch': epoch+1, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'losslogger': losslogger,'testing_accuracy':best_accuracy, }
torch.save(state, save_path)
