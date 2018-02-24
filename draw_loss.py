#!/usr/bin/env python 
########################################
#draw loss curve after training.
import numpy as np
import visdom
import os  
import sys  
import matplotlib.pyplot as plt  
import math  
import re  
import pylab  
from pylab import figure, show, legend  
from mpl_toolkits.axes_grid1 import host_subplot  
        
###############################################################
#read from log file and then draw the loss curve.
'''
eg下为log基本的格式:
I0530 08:54:19.183091 10143 solver.cpp:229] Iteration 22000, loss = 0.173712
I0530 08:54:19.183137 10143 solver.cpp:245]     Train net output #0: rpn_cls_loss = 0.101713 (* 1 = 0.101713 loss)
I0530 08:54:19.183145 10143 solver.cpp:245]     Train net output #1: rpn_loss_bbox = 0.071999 (* 1 = 0.071999 loss)
I0530 08:54:19.183148 10143 sgd_solver.cpp:106] Iteration 22000, lr = 0.001
change according to your log file's format.
'''

# read the log file  
fp = open('log.txt', 'r')   
train_iterations = []  
train_loss = []  
test_iterations = []  
#test_accuracy = []  
  
for ln in fp:  
  # get train_iterations and train_loss  
  if '] Iteration ' in ln and 'loss = ' in ln:  
    arr = re.findall(r'ion \b\d+\b,',ln)  
    train_iterations.append(int(arr[0].strip(',')[4:]))  
    train_loss.append(float(ln.strip().split(' = ')[-1]))  
      
fp.close()  
  
host = host_subplot(111)  
plt.subplots_adjust(right=0.8) # ajust the right boundary of the plot window  
#par1 = host.twinx()  
# set labels  
host.set_xlabel("iterations")  
host.set_ylabel("loss")  
#par1.set_ylabel("validation accuracy")  
  
# plot curves  
p1, = host.plot(train_iterations, train_loss, label="train loss")  
#p2, = par1.plot(test_iterations, test_accuracy, label="validation accuracy")  
  
# set location of the legend,   
# 1->rightup corner, 2->leftup corner, 3->leftdown corner  
# 4->rightdown corner, 5->rightmid ...  
host.legend(loc=1)  
  
# set label color  
host.axis["left"].label.set_color(p1.get_color())  
#par1.axis["right"].label.set_color(p2.get_color())  
# set the range of x axis of host and y axis of par1  
host.set_xlim([-1500,60000])  
host.set_ylim([0., 1.6])  

plt.draw()  
plt.show()  
