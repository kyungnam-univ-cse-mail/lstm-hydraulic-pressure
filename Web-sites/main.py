
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# 웹사이트 제목 
st.write("## LSTM을 이용한 IoT센서 기반 상수관망 수압 이상탐지 연구")
st.write('\n')

# 메뉴 만들기
menu = ['csv 업로드','그래프 현황'] 

# 메뉴를 선택할 수 있는 사이드바 생성
choice = st.sidebar.selectbox('메뉴', menu)

if choice == menu[0]:   # csv 업로드를 선택하였을때
    csv_file = st.file_uploader('CSV 파일 업로드', type=['csv'])
    st.write('\n')
    if csv_file is not None:
        df = pd.read_csv(csv_file, encoding ='cp949')
        df.columns = ['time','JM1','JM2','JM3','JM4','JM5','JM6','SD Intake','SD Discharge','N IN','N OUT','NJS1 IN','NJS1 OUT','NJS2 IN','NJS2 OUT']
        df_new = df[['time','JM1']]
        
        st.set_option('deprecation.showPyplotGlobalUse', False) # 오류 피하기 위해 넣어줌
        
        fig = df_new.plot(x = 'time', y = 'JM1').get_legend().remove() # get_legend().remove() 범례 삭제
        st.pyplot(fig, clear_figure=False, use_container_width=True)
        
      
        
        
        # st.line_chart(data=pd.DataFrame(df_new), x='time', y='JM1')

       

#df = pd.read_csv(r'python\5.정읍수도_압력.csv',encoding ='cp949')
#df.head()