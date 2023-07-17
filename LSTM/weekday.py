import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import SGD
from keras.metrics import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler

# 데이터 로드
hp_data = pd.read_csv('C:/Users/User/Desktop/water/data/weekday_data.csv', usecols=['time', 'JM1_p'], parse_dates=['time'], index_col='time')

data = hp_data[:]
test = hp_data['2020-1':'2020-1']

# train, test 데이터
def timeseries_train_test(hp_data, time_steps, for_periods):
    train_data = data.values
    test_data = test.values
    train_data_len = len(train_data)
    test_data_len = len(test_data)

    print("train data shape : " , train_data.shape)
    print("test data shape : " , test_data.shape)
    x_train = []
    y_train = []
    #print("hp_data : ", hp_data2) #hp_data :  [[0.95652174], [0.91304348],  [0.95652174]

    # train
    for i in range(time_steps, train_data_len - for_periods + 1):
        #print("i: ", i, "i~time_steps : ", i+time_steps) # 0 ~ 0+60
        x_train.append(train_data[i - time_steps:i, 0])
        y_train.append(train_data[i:i + for_periods, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # 3차원 변경 (샘플수, 시간 단계) -> (샘플수, 시간 단계, 특성수)

    inputs = pd.concat((hp_data['JM1_p'][:], hp_data['JM1_p']['2020-01':'2020-01']), axis=0).values
    inputs = inputs[len(inputs) - len(test_data) - time_steps:]
    inputs = inputs.reshape(-1, 1)

    x_test = []
    for i in range(time_steps, test_data_len + time_steps - for_periods):
        x_test.append(inputs[i - time_steps:i, 0])
        
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], time_steps, 1))

    return x_train, y_train, x_test

def normalize(hp_data, time_steps, for_periods):
    train_data = data.values
    test_data = test.values
    train_data_len = len(train_data)
    test_data_len = len(test_data)

    sc = MinMaxScaler(feature_range=(0, 1))
    ts_train_scaled = sc.fit_transform(train_data)

    x_train = []
    y_train = []

    for i in range(time_steps, train_data_len - for_periods + 1):
        #print("i: ", i, "i~time_steps : ", i+time_steps) # 0 ~ 0+60
        x_train.append(ts_train_scaled[i - time_steps:i, 0])
        y_train.append(ts_train_scaled[i:i + for_periods, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    inputs = pd.concat((hp_data['JM1_p'][:], hp_data['JM1_p']['2020-01':'2020-01']), axis=0).values
    inputs = inputs[len(inputs) - len(test_data) - time_steps:]
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)

    x_test = []

    for i in range(time_steps, test_data_len + time_steps - for_periods):
        x_test.append(inputs[i - time_steps:i, 0])
        
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], time_steps, 1))

    return x_train, y_train, x_test, sc

# lstm_model
def lstm_arch(x_train, y_train, x_test, for_periods):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1), activation='tanh')) #tanh???
    lstm_model.add(LSTM(units=50, activation='tanh'))
    lstm_model.add(Dense(units=for_periods)) #DENSE : 출력 뉴런의 수 지정 후 생성.
    # 순환 신경망 / 입력, 출력 연결

    #GRIDSEARCH()
    lstm_model.compile(optimizer=SGD(learning_rate=0.01, decay=1e-7, momentum=0.9, nesterov=False), loss='mean_squared_error')

    lstm_model.fit(x_train, y_train, epochs=700, batch_size=128, verbose=0, shuffle=False)  #batch_size=???

    lstm_prediction = lstm_model.predict(x_test)
    # lstm_prediction = sc.inverse_transform(lstm_prediction)

    return lstm_model, lstm_prediction

# LOAD TRAINED MODEL 
def predictions_plot(preds):
    predictions_plot = pd.DataFrame(columns=['JM1_p', 'result'])
    # predictions_plot['JM1_p'] = hp_data.loc[:len(preds), january_data.columns.get_loc('JM1_p')]
    predictions_plot['JM1_p'] = hp_data.loc['2020-01':'2020-01', 'JM1_p'][0:len(preds)]
    predictions_plot['result'] = preds[:, 0]

    mse = MeanSquaredError()
    mse.update_state(np.array(predictions_plot['JM1_p']), np.array(predictions_plot['result']))

    return (mse.result().numpy(), predictions_plot.plot(figsize=(40, 10)))
    
#1주일 -> 1주일 (24 * 5)
time_steps = 120
for_periods = 120

# 정규화 train, test
x_train, y_train, x_test, sc = normalize(hp_data, time_steps, for_periods)

# x_train과 y_train 출력
print("n_x_train shape:", x_train.shape)
print("n_y_train shape:", y_train.shape)
# print("n_x_train:", x_train)
# print("n_y_train:", y_train)

# lstm_model
lstm_model, lstm_prediction = lstm_arch(x_train, y_train, x_test, for_periods)
lstm_prediction = sc.inverse_transform(lstm_prediction)
#SAVE THE TRAINED MODEL H5
lstm_model.save("C:/Users/User/Desktop/water/data/save/700_128_model.h5")

print("n_predictions length:", len(lstm_prediction))

# 평균 제곱 오차. 예측 결과 시각화
result, plot = predictions_plot(lstm_prediction)
print("Mean Squared Error:", result)
print("RMSE : ", np.sqrt(result))

plt.savefig('C:/Users/User/Desktop/water/data/save/700_128_Jan.jpg')
