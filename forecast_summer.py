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
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
import keras
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor


#----------------讀data-----------------------------------------------------
#2015-11 雲林
#把所有資料先讀入至data_raw
#從data_raw中取6:00~18:00的資料寫成data_ripe 

data_raw = np.zeros((45422,7),dtype = np.float32)
data_ripe = np.zeros((3024,7), dtype = np.float32)

with open('.\summer.csv', 'r') as file:
    rows = file.readlines()
    i = 0
    for row in rows:
        '''
        data_raw[i,0] = round(float(row.split(',')[7]), 3) #照度
        data_raw[i,1] = round(float(row.split(',')[6]), 3) #模組溫度
        data_raw[i,2] = round(float(row.split(',')[5]), 3) #環境溫度
        data_raw[i,3] = round(float(row.split(',')[8]), 3) #發電量
        '''
        data_raw[i,0] = round(float(row.split(',')[2]), 3)
        data_raw[i,1] = round(float(row.split(',')[3]), 3)
        data_raw[i,2] = round(float(row.split(',')[4]), 3)
        data_raw[i,3] = round(float(row.split(',')[5]), 3)
        data_raw[i,4] = round(float(row.split(',')[6]), 3)
        data_raw[i,5] = round(float(row.split(',')[7]), 3)
        data_raw[i,6] = round(float(row.split(',')[8]), 3)
        
        i += 1
print(data_raw.shape)




i = 0
j = 0
k = 0
o = 0

data_ripe[0,:] = data_raw[0,:]

for i in range(63): #幾天
    for j in range(48): #一天幾筆資料
        data_ripe[o, :] = data_raw[k, :]
        k += 15
        o += 1
    k +=1
    
print(data_ripe.shape)


'''
#-----------隨機決定trainning data------------------------------
#把data_ripe裡的資料全部打散
#再用隨機函數(ran)決定拿多少筆資料當trainning data和testing data

ran = np.arange(3024)
random.shuffle(ran)
num = random.randint(2116, 2570)
print(num)

x_train = np.zeros((num,6),dtype = np.float32)
y_train = np.zeros((num,),dtype = np.float32)
x_test = np.zeros((len(ran)-num,6), dtype = np.float32)
y_test = np.zeros((len(ran)-num,), dtype = np.float32)

i = 0
j = 0
for i in range(len(ran)):
    for j in range(len(data_ripe[0,:])):
        if i < num :
            if j != 6:
                x_train[i,j] = data_ripe[ran[i],j]
            if j == 6 :
                y_train[i,] = data_ripe[ran[i],j]
        else:
            if j != 6:
                x_test[i-num,j] = data_ripe[ran[i],j]
            if j == 6 :
                y_test[i-num,] = data_ripe[ran[i],j]

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)
'''

#--------------------------------------------------
#最後兩天當testing data
x_train = np.zeros((2928,6),dtype = np.float32)
y_train = np.zeros((2928,),dtype = np.float32)
x_test = np.zeros((96,6), dtype = np.float32)
y_test = np.zeros((96,), dtype = np.float32)

x_train = data_ripe[:2928,0:6]
y_train = data_ripe[:2928, 6]
x_test = data_ripe[2928:, 0:6]
y_test = data_ripe[2928:, 6]

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)


'''
#--------------------------------------------------
#XGB
#xgb_train = xgb.XGBRegressor()

#xgb_train = xgb.XGBRegressor(n_estimators = 500, learning_rate = 0.05, gamma = 0, subsample = 0.7, colsample_bytree = 1, max_depth = 6)
xgb_train = xgb.XGBRegressor(n_estimators = 400, learning_rate = 0.01, max_depth = 20, objective= 'reg:linear')
xgb_train.fit(x_train, y_train)
#pre = xgb_train.predict(x_test)
pre = xgb_train.predict(x_test)
print('Score : ', explained_variance_score(y_test, pre))
print('MAE : ', mean_absolute_error(y_test, pre))

plt.plot(pre, 'r', y_test, 'b')
plt.show()
'''

#--------------------------------------------------
#RandomForest
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42, max_depth = 5)
rf.fit(x_train, y_train)

pre = rf.predict(x_test)
error = abs(pre-y_test)
print('Score : ', explained_variance_score(y_test, pre))
print("MAE : ",round(np.mean(error), 2))



plt.plot(pre, 'r', y_test, 'b')
plt.show()



#------------------------------------------
#特徵重要性
feature_list = list('風' '大' '濕' '環' '模' '照')
importance = list(rf.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importance)]
feature_importances = sorted(feature_importances, key = lambda x:x[1], reverse = True)
[print('Variable : {:20} Importance : {}'.format(*pair)) for pair in feature_importances]


'''
#--------------------------------------
#AdaBoost
regr_1 = DecisionTreeRegressor(max_depth = 4)
regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 4), n_estimators = 500, random_state = np.random.RandomState(1))
regr_2.fit(x_train, y_train)
pre = regr_2.predict(x_test)
error = abs(pre - y_test)
print('Score : ', explained_variance_score(y_test, pre))
print('MSE : ', round(np.mean(error), 2))

plt.plot(pre, 'r', y_test, 'b')
plt.show()
'''
'''
#------------Neural Network--------------------------------------------------------



model = Sequential()
model.add(Dense(input_dim = 3,units = 3, kernel_initializer='normal', activation = 'relu'))
model.add(Dense(units = 3, kernel_initializer='normal', activation = 'relu'))

#model.add(Dropout(0.7))
#model.add(Dense(units = 689, activation = 'relu'))
#model.add(Dropout(0.7))
#for i in range(10):
    #model.add(Dense(units = 200, activation = 'relu'))
model.add(Dense(units = 1,  kernel_initializer='normal'))

model.compile(loss='mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = 100, epochs = 30)

result = model.evaluate(x_train, y_train)
print ('\nTrain Acc: ', result[1])

result = model.evaluate(x_test, y_test)
print ('\nTest Acc: ', result[1])
'''


'''
#------------------------------------------------
#plot
plt.plot(pre, 'r', y_test, 'b')
plt.title('Plot Number : '+ str(98))
plt.show()
'''


'''
#------------------------------------------------
#Predict 2015-10 雲林

predict_raw = np.zeros((5424,4),dtype = np.float32)
predict_ripe = np.zeros((1488,4), dtype = np.float32)

with open('.\pv_predict.csv', 'r') as file:
    rows = file.readlines()
    i = 0
    for row in rows:
        predict_raw[i,0] = round(float(row.split(',')[1]), 3)
        predict_raw[i,1] = round(float(row.split(',')[2]), 3)
        predict_raw[i,2] = round(float(row.split(',')[3]), 3)
        predict_raw[i,3] = round(float(row.split(',')[4]), 3)
        i += 1
print(predict_raw.shape)

i = 0
j = 0
k = 12
p = 0
o = 0

for i in range(31):
    for j in range(48):
        predict_ripe[o, :] = predict_raw[k + p, :]
        k += 3
        o += 1
    p += 31   
print(predict_ripe.shape)

x_predict = predict_ripe[:,0:3]
y_predict = predict_ripe[:, 3]


#-----------------------------------
#XGB
pre = xgb_train.predict(x_predict)


#-----------------------------------
#RandomForest
pre = rf.predict(x_predict)

#-----------------------------------
#RandomForest
pre = regr_2.predict(x_predict)


plt.plot(pre, 'r', y_predict, 'b')
plt.show()
'''

'''
#------------------------------------------------
#Predict 2015-12 雲林

predict_raw = np.zeros((5073,4),dtype = np.float32)
predict_ripe = np.zeros((1392,4), dtype = np.float32)

with open('.\pv_predict2.csv', 'r') as file:
    rows = file.readlines()
    i = 0
    for row in rows:
        predict_raw[i,0] = round(float(row.split(',')[1]), 3)
        predict_raw[i,1] = round(float(row.split(',')[2]), 3)
        predict_raw[i,2] = round(float(row.split(',')[3]), 3)
        predict_raw[i,3] = round(float(row.split(',')[4]), 3)
        i += 1
print(predict_raw.shape)

i = 0
j = 0
k = 12
p = 0
o = 0

for i in range(29):
    for j in range(48):
        predict_ripe[o, :] = predict_raw[k + p, :]
        k += 3
        o += 1
    p += 31   
print(predict_ripe.shape)

x_predict = predict_ripe[:,0:3]
y_predict = predict_ripe[:, 3]


#-----------------------------------
#XGB
pre = xgb_train.predict(x_predict)


#-----------------------------------
#RandomForest
pre = rf.predict(x_predict)


#-----------------------------------
#Adaboost
pre = regr_2.predict(x_predict)


plt.plot(pre, 'r', y_predict, 'b')
plt.show()
'''