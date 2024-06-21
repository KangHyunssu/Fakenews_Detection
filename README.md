# Fakenews_Detection
# FakeBuster: 딥러닝 기반 가짜 뉴스 탐지 시스템

## 서비스 소개

FakeBuster는 딥러닝 기법을 활용하여 가짜 뉴스를 탐지하는 시스템입니다. 현대 사회는 정보의 홍수 속에서 살아가며, 소셜 미디어와 온라인 커뮤니티의 발달로 인해 가짜 뉴스가 빠르게 확산되고 있습니다. FakeBuster는 LSTM 및 CNN 모델을 사용하여 텍스트 데이터를 분석하고, 가짜 뉴스를 정확하게 식별함으로써 신뢰할 수 있는 정보 환경을 구축하는 데 기여합니다.


## 기술 스택

- **프로그래밍 언어:** Python
- **딥러닝 프레임워크:** TensorFlow, Keras
- **데이터 처리:** Pandas, NumPy
- **자연어 처리:** NLTK
- **모델 최적화:** RandomizedSearchCV
- **시각화:** Matplotlib, Seaborn
- **배포:** Google Colab, Google Drive

## 핵심 기능

### 1. 데이터 전처리
- **텍스트 정제:** HTML 태그, 특수 문자 및 숫자 제거.
- **정규화:** 소문자 변환 및 표제어 추출.
- **토큰화:** 텍스트를 의미 있는 단위로 분할.
- **패딩:** 입력 길이를 모델에 맞게 조정.

### 2. 모델 개발
- **LSTM 모델:** 텍스트 데이터의 순차적 특성과 문맥 정보를 학습.
- **CNN 모델:** 텍스트 데이터의 지역적 특징과 패턴을 인식.
- **메타 모델:** LSTM과 CNN 모델의 장점을 결합하여 정확도 향상.

### 3. 학습 및 평가
- **하이퍼파라미터 튜닝:** RandomizedSearchCV를 사용하여 최적의 모델 성능 도출.
- **교차 검증:** 모델의 성능을 평가하고 과적합 방지.
- **성능 지표:** 정확도, 정밀도, 재현율 및 F1 스코어를 사용한 평가.

### 4. 실시간 가짜 뉴스 탐지
- **예측:** 학습된 모델을 사용하여 새로운 기사 분석 및 가짜 뉴스 가능성 예측.
- **시각화:** 예측 결과와 모델 성능을 시각적으로 제공.

### 5. 설명 가능성
- **특징 분석:** 모델이 예측에 사용한 주요 특징과 패턴 제공.
- **결과 해석:** 탐지 결과에 대한 자세한 설명 제공으로 사용자 신뢰성 향상.

## 시작하기

### 사전 준비
- Python 3.6 이상
- Google Colab 계정
- 머신러닝 및 자연어 처리에 대한 기본 지식

### 설치

1. 저장소 클론:
    ```bash
    git clone https://github.com/yourusername/FakeBuster.git
    cd FakeBuster
    ```

2. 필요한 라이브러리 설치:
    ```bash
    pip install -r requirements.txt
    ```

3. NLTK 데이터 다운로드:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

### 사용법

1. **데이터 전처리:**
    ```python
    from preprocess import preprocess_text
    data = preprocess_text(raw_data)
    ```

2. **모델 학습:**
    ```python
    from models import create_lstm_model, create_cnn_model
    lstm_model = create_lstm_model()
    cnn_model = create_cnn_model()
    ```

3. **모델 평가:**
    ```python
    from evaluate import evaluate_model
    evaluate_model(lstm_model, test_data)
    evaluate_model(cnn_model, test_data)
    ```

4. **새로운 기사 예측:**
    ```python
    from predict import predict_article
    predictions = predict_article(new_articles, lstm_model, cnn_model)
    ```

### 결과

FakeBuster 시스템은 가짜 뉴스 탐지에서 높은 정확도를 달성했으며, 다양한 데이터셋에서 강력한 성능 지표를 보였습니다. LSTM과 CNN 모델의 결합을 통해 텍스트 데이터의 문맥 이해와 패턴 인식이 동시에 이루어졌습니다.

<img width="1204" alt="image" src="https://github.com/KangHyunssu/Fakenews_Detection/assets/128908098/198faaa0-f894-44a6-a0a9-a8cc137cbca7">

