# output : out/nowcasting.csv
# model  : models/nowcasting_lstm.h5
# checkpoint: models/nowcasting.hdf5
import os
import time
import glob
import pandas as pd
import numpy as np

## Immport Libraries
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras import models, layers 
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn import preprocessing

import matplotlib.pyplot as plt

# 氣象資料集路徑
dataPath = 'datasets/weather/observatory/' 

# 選定一個氣象站資料做預測分析 
## select Observatory ID  # 台灣氣象觀測站名
#ObservatoryID = '466880' # BanQiao     板橋
#ObservatoryID = '466900' # TamSui      淡水
#ObservatoryID = '466910' # AnBu        鞍部
ObservatoryID  = '466920' # Taipei      台北市 
#ObservatoryID = '466930' # ZhuZiHu     竹子湖
#ObservatoryID = '466940' # KeeLung     基隆
#ObservatoryID = '466950' # PengJiaYu   彭佳嶼
#ObservatoryID = '466990' # HuaLien     花蓮
#ObservatoryID = '467060' # Su-Ao       蘇澳
#ObservatoryID = '467080' # YiLan       宜蘭
#ObservatoryID = '467110' # KinMen      金門
#ObservatoryID = '467300' # DongJiDao   東吉島
#ObservatoryID = '467350' # PengHu      澎湖
#ObservatoryID = '467410' # Tainan      台南市
#ObservatoryID = '467420' # YongKang    永康
#ObservatoryID = '467440' # KaoSiung    高雄
#ObservatoryID = '467480' # ChiaYi      嘉義市
#ObservatoryID = '467490' # Taichung    台中市
#ObservatoryID = '467530' # Alishan     阿里山
#ObservatoryID = '467540' # DaWu        大武
#ObservatoryID = '467550' # Yushan      玉山
#ObservatoryID = '467571' # HsinChu     新竹
#ObservatoryID = '467590' # HengChun    恆春
#ObservatoryID = '467610' # ChengGong   成功
#ObservatoryID = '467620' # LanYu       蘭嶼
#ObservatoryID = '467650' # SunMoonLake 日月潭
#ObservatoryID = '467660' # TaiTung     台東
#ObservatoryID = '467770' # WuQi        梧棲
#ObservatoryID = '467990' # MaTsu       馬祖

# Observatory Dataset : 2020-01-18 ~ 02/26
dateList = os.listdir(dataPath)
dateList.sort()

# .csv file list of one observatory
fList=[]
for date in dateList:
    f = dataPath + date +'/'+ObservatoryID+'-'+date+'.csv'
    fList.append(f)
	
# create the Dataframe Array for the observatory data of all stations
df = df0 = pd.DataFrame()
for file in fList:
    print(file)
    df0 = pd.read_csv(file)
    df0 = df0.drop(columns=['ObsTime', 'SunShine', 'GloblRad', 'Visb', 'UVI', 'Cloud Amount'])    
    for i in range(len(df0)): # hour no. = 1~24
        for key in df0.columns:
            if df0.loc[i][key]=='T':
                df0.at[i,key]=0.1
            if df0.loc[i][key]=='V': 
                df0.at[i,key]=0
            if df0.loc[i][key]=='/':
                df0.at[i,key]=0
        df = df.append(df0.loc[i], ignore_index=True)   

# convert dataframe to numpy-array
dataX = df.to_numpy()
dataY = df.loc[:,'Temperature'].to_numpy()
print('dataX:',dataX.shape)
print('dataY:',dataY.shape)

# observing points are 24*14 hours
historyPoints = 48
batch_size = historyPoints

## Build Model
m_shape = (historyPoints, dataX.shape[1])
m_input = layers.Input(shape=m_shape,  name='m_input')
units = historyPoints

# LSTM
m  = layers.LSTM(units, name='m_lstm_0')(m_input)

t0 = layers.Dense(units, activation="sigmoid")(m)
t0 = layers.Dense(1, activation="linear", name='temp0')(t0)

t1 = layers.Dense(units, activation="sigmoid")(m)
t1 = layers.Dense(1, activation="linear", name='temp1')(t1)

t2 = layers.Dense(units, activation="sigmoid")(m)
t2 = layers.Dense(1, activation="linear", name='temp2')(t2)

t3 = layers.Dense(units, activation="sigmoid")(m)
t3 = layers.Dense(1, activation="linear", name='temp3')(t3)

t4 = layers.Dense(units, activation="sigmoid")(m)
t4 = layers.Dense(1, activation="linear", name='temp4')(t4)

t5 = layers.Dense(units, activation="sigmoid")(m)
t5 = layers.Dense(1, activation="linear", name='temp5')(t5)

model = models.Model(inputs=m_input, outputs=[t0, t1, t2, t3, t4, t5])

#if os.path.exists('models/nowcasting_lstm.h5'):
#	model = models.load_model('models/nowcasting_lstm.h5')

model.summary()

# Compile Model
model.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])

checkpoint = ModelCheckpoint("models/nowcasting.hdf5", monitor='loss', verbose=0, save_best_only=True, mode='auto', save_freq=1)

# Plot real vs predicted 
def plot_nowcasting(observatoryId, title, period, real, pred):
	fig = plt.figure()
	plt.gcf().set_size_inches(21, 7, forward=True)
	start = 0
	end = -1
	plt.plot(real[start:end], label='real')
	plt.plot(pred[start:end], label='pred')
	plt.title(observatoryId+' '+title+' Nowcasting ['+period+']')
	plt.legend(['Real', 'Pred'])
	plt.show()  
	fig.savefig(observatoryId+title+'.png')

dfLoss = pd.DataFrame(columns = ['loss','real0','pred0','diff0','real1','pred1','diff1','real2','pred2','diff2','real3','pred3','diff3','real4','pred4','diff4','real5','pred5','diff5'])

## Prepare Dataset
dataPeriod = '2020-01-18 ~ 02-07' 
#dataPeriod = '2020-01-18 ~ 02-14' 
#dataPeriod = '2020-01-18 ~ 02-21' 
#dataPeriod = '2020-01-18 ~ 02-28' 
trainDays  = 21

incHour    = 0

start, end = 0+incHour, 24*trainDays+incHour
# training dataset
trainX  = np.array([dataX[i:i+ historyPoints] for i in range(start, end)])
trainY0 = np.array([dataY[i+ historyPoints+0] for i in range(start, end)]) 
trainY1 = np.array([dataY[i+ historyPoints+1] for i in range(start, end)]) 
trainY2 = np.array([dataY[i+ historyPoints+2] for i in range(start, end)])
trainY3 = np.array([dataY[i+ historyPoints+3] for i in range(start, end)])
trainY4 = np.array([dataY[i+ historyPoints+4] for i in range(start, end)])
trainY5 = np.array([dataY[i+ historyPoints+5] for i in range(start, end)])
print('trainX: ',trainX.shape)
trainY0 = trainY0.reshape(trainY0.shape[0], 1)
trainY1 = trainY1.reshape(trainY1.shape[0], 1)
trainY2 = trainY2.reshape(trainY2.shape[0], 1)
trainY3 = trainY3.reshape(trainY3.shape[0], 1)
trainY4 = trainY4.reshape(trainY4.shape[0], 1)
trainY5 = trainY5.reshape(trainY5.shape[0], 1)
print('trainY: ',trainY0.shape)
print(trainY1.shape)
print(trainY2.shape)
print(trainY3.shape)
print(trainY4.shape)
print(trainY5.shape)

# testing dataset 
testX = np.array(dataX[end : end+historyPoints]) # 24 hours data
testY = np.array(dataY[end+historyPoints: end+historyPoints+6]) # 6 hours predicted
testX = testX.reshape(1, testX.shape[0], testX.shape[1])
print(testX.shape)
testY = testY.reshape(testY.shape[0], 1)
print(testY.shape)

# normalise X 
trainX = trainX.astype('float') / 1024.0
testX  = testX.astype('float') / 1024.0

## Train Model
if os.path.exists('models/nowcasting_lstm.h5'):
	num_epochs = 500 # incremental training
else:
	num_epochs = 8000 # new training

# target loss <0.1 (for temperature)
start_time = time.time()
history = model.fit(trainX, [trainY0, trainY1, trainY2, trainY3, trainY4, trainY5], batch_size=batch_size, epochs=num_epochs, verbose=2 #)
         ,callbacks=[checkpoint])
print("--- model trained : %s minutes ---" % str((time.time() - start_time)/60)) 

## Save Model
models.save_model('models/nowcasting_lstm.h5')

## Evaluate Model
# *show predicton using training dataset*
loss = min(history.history['loss'])
model.load_weights("models/nowcasting.hdf5") # restore callbacks checkpoint

(predY0, predY1, predY2, predY3, predY4, predY5) = model.predict(trainX)
plot_nowcasting(ObservatoryID, '_Train_'+str(trainDays)+'days_+'+str(incHour)+'hour', dataPeriod, trainY0, predY0)

## Test Model
(preY0, predY1, predY2, predY3, predY4, predY5) = model.predict(testX)
real = testY
pred = [predY0[0], predY1[0], predY2[0], predY3[0], predY4[0], predY5[0]]
ll = []
ll.append(loss)
for i in range(6):
	ll.append(real[i][0])
	ll.append(pred[i][0])
	ll.append(pred[i][0]-real[i][0])
print(ll)
dfLoss.loc[len(dfLoss)] = ll
dfLoss.to_csv('out/nowcasting.csv',index=False)

# To fresh training the model
if os.path.exists('models/nowcasting.hdf5'):
	os.remove('models/nowcasting.hdf5')
