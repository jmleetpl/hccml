import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost

# 모델 및 scaler 로드 (학습한 모델과 scaler 사용)
model = pd.read_pickle('best_xgb_model.pkl')
scaler = pd.read_pickle('scaler.pkl')

# 웹페이지 제목
st.title('간암 재발 예측 계산기')

# 사용자 입력 폼
st.header('환자 정보 입력')

def user_input_features():
    Ht = st.number_input('키(cm)', 100, 200, 170)
    BWt = st.number_input('체중(kg)', 30, 150, 65)
    age = st.number_input('나이', 18, 90, 50)
    Nutrophil = st.number_input('호중구(%)', 0.0, 100.0, 50.0)
    Lymphocyete = st.number_input('림프구(%)', 0.0, 100.0, 25.0)
    MeldPELDScore = st.number_input('MELD/PELD Score', 0.0, 50.0, 15.0)
    AFP = st.number_input('AFP', 0, 10000, 100)
    PIVKAII = st.number_input('PIVKA-II', 0, 10000, 50)
    age = st.number_input('나이', 0, 100, 50)
    # 추가 필요한 입력 항목을 여기 계속 작성...

    data = {
        'Ht': Ht,
        'BWt': BWt,
        'age': age,
        'Nutrophil': Nutrophil,
        'Lymphocyete': Lymphocyete,
        'MeldPELDScore': MeldPELDScore,
        #... 여기에 실제로 사용된 컬럼을 다 추가해야 합니다.
        # 예시 목적이므로 실제 데이터셋과 동일하게 맞추어 작성 필요
        }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# 입력 데이터를 표준화
input_scaled = scaler.transform(input_df)

# 예측 수행
if st.button('예측하기'):
    prediction_proba = model.predict_proba(input_scaled)[:, 1][0]
    prediction = model.predict(input_scaled)[0]

    st.subheader('예측 결과')
    if prediction == 1:
        st.error(f'⚠️ 간암 재발 가능성이 높습니다. (위험도: {prediction_proba*100:.2f}%)')
    else:
        st.success(f'간암 재발 가능성이 낮습니다. (확률: {prediction_proba:.2%})')

import joblib
joblib.dump(best_model, 'best_xgb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

