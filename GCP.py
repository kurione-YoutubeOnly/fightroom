#モジュールのインポート
import numpy as np
import pandas as pd
import sys
from matplotlib import pylab as plt
from matplotlib.pylab import rcParams
import keras
from keras.models import Sequential
from keras.layers import Dense,Activation,LSTM
from keras.optimizers import Adam
import tensorflow as tf
rcParams['figure.figsize'] = 8, 6

#警告非表示
import os
import logging
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)

#それぞれの手に対した確率の算出
def single_probability(hand):
    #ファイルの読み込み
    data = pd.read_csv('010101GCP.csv')
    data.head()

    #ケラス用に型変換
    input_data = data[hand].values.astype(float)
    look_back_time = 50

    #入力/教師データ分割
    data, target = [], []
    for i in range(len(input_data)-look_back_time):
        data.append(input_data[i:i + look_back_time])
        target.append(input_data[i + look_back_time])
    x0 = np.array(data).reshape(len(data),look_back_time,1)
    y0 = np.array(target).reshape(len(data),1)

    #学習/検証データ分割
    train_size = int(len(x0)*0.9)
    x0_train = x0[:train_size]
    x0_test = x0[train_size:]
    y0_train = y0[:train_size]
    y0_test = y0[train_size:]
    np.set_printoptions(threshold=np.inf)

    #モデル構築
    lstm_model = Sequential()
    lstm_model.add(LSTM(50,batch_input_shape=(None,look_back_time,1)))
    lstm_model.add(Dense(1))
    
    #モデルコンパイル
    lstm_model.compile(loss='mean_squared_error',optimizer=Adam() , metrics = ['accuracy'])
    
    #学習
    lstm_model.fit(x0_train,y0_train,
        epochs=150,
        validation_data=(x0_test,y0_test),
        verbose=0,
        batch_size=20)
    
    y0_pred_train = lstm_model.predict(x0_train)
    y0_pred_test = lstm_model.predict(x0_test)
    plt.plot(x0[:,0,0],color='blue',)
    plt.plot(y0_pred_train,color='red',)
    plt.plot(range(len(x0_train),len(x0_test)+len(x0_train)),y0_pred_test,color='green',)
    plt.show()
    
    #予測
    future_pred = x0[:,0,0].copy()
    X0_future_pred = future_pred[-1*look_back_time:]
    y0_future_pred = lstm_model.predict(X0_future_pred.reshape(1,look_back_time,1))

    #確率を出力
    return y0_future_pred

#人工知能以外の全体制御
def controler():
#新データ入力
    new = input('前回: ')
    if new == 'f':
        sys.exit()

    #ファイルにデータ入力
    if new == '0' or new == '1' or new == '2':
        file = open('012GCP.csv', 'a')
        file.write(new + '\n')
        file.close()
    else:
        print('0(グー)か1(チョキ)か2(パー)の値を入力してください')
        controler()
    origin_data = pd.read_csv('012GCP.csv').values.tolist()

    #ファイルからpandasデータフレームの作成
    rock_data = []
    scissors_data = []
    paper_data = []
    for number in origin_data:
        if number[0] == 0:
            rock_data.append(1)
            scissors_data.append(0)
            paper_data.append(0)
        if number[0] == 1:
            rock_data.append(0)
            scissors_data.append(1)
            paper_data.append(0)
        if number[0] == 2:
            rock_data.append(0)
            scissors_data.append(0)
            paper_data.append(1)
    data_making = pd.DataFrame({
        'rock' : rock_data,
        'scissors' : scissors_data,
        'paper' : paper_data})

    #データフレームからファイルの作成
    data_making.to_csv('010101GCP.csv',index=False)

    #グーチョキパーデータ入力
    rock = single_probability('rock')
    scissors = single_probability('scissors')
    paper = single_probability('paper')
    rock_probability = rock / (rock + scissors + paper) * 100
    scissors_probability = scissors / (rock + scissors + paper)* 100
    paper_probability = paper / (rock + scissors + paper)* 100
    print('次の手\n' + 'グー:' + str(rock_probability[0][0]) + '%\n' + 'チョキ:' + str(scissors_probability[0][0]) + '%\n' + 'パー:'  + str(paper_probability[0][0]) + '%')

    #次の周回へ
    controler()

#最初の起動
controler()
