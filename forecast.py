#from __future__ import division
import numpy as np
import random
import xgboost as xgb
import cv2
import os
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error


#----------------讀data-----------------------------------------------------
#把所有資料先讀入至data_raw
#從data_raw中取6:00~18:00的資料寫成data_ripe 

data_raw = np.zeros((5075,4),dtype = np.float32)
data_ripe = np.zeros((1392,4), dtype = np.float32)

with open('.\pv_data.csv', 'r') as file:
    rows = file.readlines()
    i = 0
    for row in rows:
        data_raw[i,0] = row.split(',')[1]
        data_raw[i,1] = row.split(',')[2]
        data_raw[i,2] = round(float(row.split(',')[3]), 3)
        data_raw[i,3] = round(float(row.split(',')[4]), 3)
        i += 1
print(data_raw.shape)

i = 0
j = 0
k = 12
p = 0
o = 0

for i in range(29):
    for j in range(48):
        data_ripe[o, :] = data_raw[k + p, :]
        k += 3
        o += 1
    p += 31   
print(data_ripe.shape)
'''
#-----------隨機決定trainning data------------------------------
#把data_ripe裡的資料全部打散
#再用隨機函數(ran)決定拿多少筆資料當trainning data和testing data

ran = np.arange(1392)
random.shuffle(ran)
num = random.randint(800, 1150)
print(num)

x_train = np.zeros((num,3),dtype = np.float32)
y_train = np.zeros((num,),dtype = np.float32)
x_test = np.zeros((len(ran)-num,3), dtype = np.float32)
y_test = np.zeros((len(ran)-num,), dtype = np.float32)

i = 0
j = 0
for i in range(len(ran)):
    for j in range(len(data_ripe[0,:])):
        if i < num :
            if j != 3:
                x_train[i,j] = data_ripe[ran[i],j]
            if j == 3 :
                y_train[i,] = data_ripe[ran[i],j]
        else:
            if j != 3:
                x_test[i-num,j] = data_ripe[ran[i],j]
            if j == 3 :
                y_test[i-num,] = data_ripe[ran[i],j]

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)
'''

#--------------------------------------------------
#最後兩天當testing data
x_train = np.zeros((1296,3),dtype = np.float32)
y_train = np.zeros((1296,),dtype = np.float32)
x_test = np.zeros((96,3), dtype = np.float32)
y_test = np.zeros((96,), dtype = np.float32)

x_train = data_ripe[0:1296,0:3]
y_train = data_ripe[0:1296, 3]
x_test = data_ripe[1296:, 0:3]
y_test = data_ripe[1296:, 3]

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)


#--------------------------------------------------
xgb_train = xgb.XGBRegressor()

#xgb_train = xgb.XGBRegressor(n_estimators = 100, learning_rate = 0.01, gamma = 0, subsample = 0.7, colsample_bytree = 1, max_depth = 5)
xgb_train.fit(x_train, y_train)
#pre = xgb_train.predict(x_test)
pre = xgb_train.predict([385, 38, 29.091])
#print('Score : ', explained_variance_score(y_test, pre))
#print('MAE : ', mean_absolute_error(y_test, pre))
print(pre)

'''
plt.plot(pre, 'r', y_test, 'b')
plt.title('Plot Number : '+ str(98))
plt.show()
'''
