import numpy as np
import pandas as pd 
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import random

#결측치(Missing Data)
def Missing(data):
    #결측치 확인
    missing_values = data.isnull().sum()

    #이전값 대체. forward fill
    method_fill = 'ffill'

    # 결측치가 있는 열에 대해 이전 값으로 대체 또는 스플라인 보간 실행
    for col in missing_values[missing_values > 0].index:
        # 결측치가 10초 이상 지속되면 스플라인 보간, 그렇지않으면 이전 값으로 대체
        if missing_values[col] >= 10:
            data[col] = data[col].interpolate(method='spline', order=3)
        else:
            data[col] = data[col].fillna(method=method_fill)

    return data

#이상치(Anomalous Data)
def Anomalous(data, percent, m, ws):
    #이상치 검출 및 처리(범위 : 전 ws초간 데이터 i-ws~i-1)
    window_size = ws #윈도우 사이즈(초)

    for col in data.columns[1:]:
        for i in range(len(data[col])):
            d = []
            #이전 10개의 데이터
            if i <= window_size:
                for j in range(window_size):
                    d.append(data.loc[j, col])
            else:
                for j in range(window_size):
                    d.append(data.loc[i-j-1, col])
            
            d_sorted = np.sort(d) #정렬
        
            median = np.median(d_sorted) #중앙값
            q1 = np.percentile(d, 25) #제1사분위수
            q3 = np.percentile(d, 75) #제3사분위수
            IQR = q3 - q1 #사분위범위
            UIF = q3 + (IQR * 1.5) #정상치 범위(최대)
            LIF = q1 - (IQR * 1.5) #정상치 범위(최소)

            raw = float(data.loc[i, col])
            if (raw > UIF) or (raw < LIF): #UIF와 LIF 범위를 벗어나는 경우
                if (median * (1 + percent/100) < raw) or (median * (1 - percent/100) > raw): 
                #중앙값과의 차이가 percent를 벗어나는 경우
                    #이상치를 평균으로 설정
                    if m == 1:
                        data.loc[i, col] = sum(d) / len(d)
                        print(sum(d) / len(d))
                    else:
                        if i == 0: #0번째 데이터 : 중앙값으로 설정
                            data.loc[i, col] = median
                        else: #이전 값으로 설정
                            data.loc[i, col] = data.loc[i-1, col]

    return data


def Graph(data, num):
    for col in data.columns[1:]:
        x = data[data.columns[0]]
        y = data[col]

        plt.plot(x, y)
        plt.xlabel('Time')
        plt.ylabel(col)

        if(num == 1):
            plt.title(col + ' before')
            plt.savefig('C:/Users/user/Desktop/water/result/' + col + '_수정전2.jpg', format='jpg')
        elif(num == 2):
            plt.title(col + ' after')
            plt.savefig('C:/Users/user/Desktop/water/result/' + col + '_수정후2.jpg', format='jpg')

        #plt.show()


if __name__ == "__main__":
    #데이터 읽어오기
    path = input("데이터 경로(csv) : ")

    data = pd.read_csv(path)

    #데이터 수정전 그래프
    Graph(data, 1)

    #결축치 확인 및 처리
    # if data.isnull().sum() != 0 :
    print("결측치 개수 (수정전)\n",data.isnull().sum())
    data = Missing(data)

    # 윈도우 사이즈 입력
    window_size = int(input("Enter Window size (sec) : "))

    percent = float(input("이상치로 판단할 범위(%) Enter minimum number : "))
    m = int(input("이상치를 평균 값으로 수정할 것인가? (수정o:1, 수정x:0) : "))

    #이상치
    data = Anomalous(data, percent, m, window_size)

    #데이터 수정 후 그래프
    Graph(data, 2)

    #파일 생성
    data.to_csv("C:/Users/user/Desktop/water/result/result4.csv")

