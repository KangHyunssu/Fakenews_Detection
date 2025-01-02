# 📰 FakeBuster: 딥러닝 기반 가짜 뉴스 탐지 시스템 🚀

**FakeBuster**는 딥러닝 기법을 활용하여 가짜 뉴스를 탐지하는 시스템입니다.  
현대 사회에서 **가짜 뉴스**는 정보의 신뢰성을 저해하고 사회적 혼란을 일으키고 있습니다.  
FakeBuster는 **LSTM** 및 **CNN 모델**을 활용하여 텍스트 데이터를 분석하고,  
정확한 **가짜 뉴스 탐지**를 통해 신뢰할 수 있는 정보 환경 구축을 목표로 합니다. 🧠✨  

---

## 📚 서비스 소개

### 🎯 **FakeBuster의 핵심**
- **문맥 이해**: LSTM 모델로 텍스트의 시간적 순서와 문맥 파악.  
- **패턴 인식**: CNN 모델로 텍스트의 중요한 특징을 추출.  
- **높은 정확도**: 두 모델을 결합한 메타 모델로 가짜 뉴스 탐지 성능 극대화.  

---

## 🛠️ 기술 스택

| 구성 요소            | 사용 기술                                         |
|----------------------|--------------------------------------------------|
| **프로그래밍 언어**   | Python                                          |
| **딥러닝 프레임워크** | TensorFlow, Keras                               |
| **데이터 처리**       | Pandas, NumPy                                   |
| **자연어 처리**       | NLTK                                           |
| **모델 최적화**       | RandomizedSearchCV                              |
| **시각화**           | Matplotlib, Seaborn                             |
| **배포 환경**         | Google Colab, Google Drive                      |

---

## 🌟 주요 기능

### 🧹 **1. 데이터 전처리**
- HTML 태그, 특수 문자 제거 🧽  
- 소문자 변환 및 표제어 추출 🔡  
- 텍스트를 의미 있는 단위로 분할 (토큰화) ✂️  
- 모델 입력 길이에 맞춘 패딩 처리 🧩  

### 🧠 **2. 모델 개발**
- **LSTM 모델**: 문맥과 순차적 특징 학습.  
- **CNN 모델**: 텍스트 데이터의 패턴과 지역적 특징 인식.  
- **메타 모델**: LSTM과 CNN 결합으로 성능 극대화.  

### 🎯 **3. 학습 및 평가**
- RandomizedSearchCV로 하이퍼파라미터 최적화 🎛️  
- 교차 검증으로 모델의 일반화 능력 확인 📈  
- 정확도, 정밀도, 재현율, F1 스코어로 평가 🏅  

### 🔍 **4. 실시간 가짜 뉴스 탐지**
- 새로운 기사를 모델에 입력하여 **가짜 뉴스 가능성 예측**.  
- 결과를 시각화하여 직관적으로 제공 📊.  

### 💡 **5. 설명 가능성 (Explainability)**
- 모델이 예측에 사용한 주요 특징과 패턴 제공.  
- 사용자 신뢰성 향상을 위한 결과 해석.  

---

## 🎯 FakeBuster 사용법

### **1. 사전 준비**
- Python 3.6 이상 설치  
- Google Colab 계정 생성  
- 머신러닝 및 자연어 처리 기초 지식 권장  

### **2. 설치**
1. **저장소 클론**:
    ```bash
    git clone https://github.com/yourusername/FakeBuster.git
    cd FakeBuster
    ```

2. **필수 라이브러리 설치**:
    ```bash
    pip install -r requirements.txt
    ```

3. **NLTK 데이터 다운로드**:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

### **3. 실행**
1. 데이터 전처리:
    ```python
    from preprocess import preprocess_text
    data = preprocess_text(raw_data)
    ```

2. 모델 학습:
    ```python
    from models import create_lstm_model, create_cnn_model
    lstm_model = create_lstm_model()
    cnn_model = create_cnn_model()
    ```

3. 모델 평가:
    ```python
    from evaluate import evaluate_model
    evaluate_model(lstm_model, test_data)
    evaluate_model(cnn_model, test_data)
    ```

4. 실시간 예측:
    ```python
    from predict import predict_article
    predictions = predict_article(new_articles, lstm_model, cnn_model)
    ```

---

## 📊 결과 예시

FakeBuster는 다양한 데이터셋에서 **높은 정확도**를 달성했으며,  
LSTM과 CNN 모델 결합으로 **문맥 이해**와 **패턴 인식** 성능을 강화했습니다.  

![결과 시각화](https://github.com/KangHyunssu/Fakenews_Detection/assets/128908098/198faaa0-f894-44a6-a0a9-a8cc137cbca7)

---

## 📁 프로젝트 구성

- `preprocess.py`: 텍스트 데이터 전처리 모듈  
- `models.py`: LSTM 및 CNN 모델 생성 코드  
- `evaluate.py`: 학습된 모델의 성능 평가  
- `predict.py`: 새로운 데이터 예측 모듈  
- `requirements.txt`: 의존성 파일  

---

## 📚 참고 자료

- TensorFlow 및 Keras 공식 문서  
- NLTK 자연어 처리 라이브러리  
- Matplotlib 및 Seaborn 시각화 문서  

---

