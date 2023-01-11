[랩업리포트](https://www.notion.so/NLP04-ODQA-Wrap-up-Report-Public-06cff06e99c0431393ada242757420eb)

## ODQA

![Untitled](https://user-images.githubusercontent.com/42907231/211742201-ae51bdaa-b5fd-44c5-b5cc-522fe2251342.png)

## 프로젝트 개요

**Question Answering (QA)은 다양한 종류의 질문에 대해 대답하는 인공지능**을 만드는 연구 분야이다. 다양한 QA 시스템 중, **Open-Domain Question Answering (ODQA) 은 주어지는 지문이 따로 존재하지 않고 사전에 구축되어있는 Knowledge resource 에서 질문에 대답할 수 있는 문서를 찾는** 과정이 추가되기 때문에 더 어려운 문제이다. **첫 단계는 질문에 관련된 문서를 찾아주는 "retriever"** 단계이고, **다음으로는 관련된 문서를 읽고 적절한 답변을 찾거나 만들어주는 "reader"** 단계이다. 두 가지 단계를 각각 구성하고 그것들을 적절히 통합하게 되면, 어려운 질문을 던져도 답변을 해주는 ODQA 시스템이 된다.

## 팀 구성 및 역할

| 김해원 | 김혜빈 | 박준형 | 양봉석 | 이예령 |
| --- | --- | --- | --- | --- |
| 실험 세팅 (WandB) <br> ElasticSearch 구현 및 실험 | 데이터 EDA <br> 데이터 전처리 | 코드리뷰 <br> DPR 구현 및 실험 | PM <br> Reader model 실험 | 실험 세팅 (huggingface) <br> 외부 데이터 탐색 및 활용 |

## 실험 내역

- preprocessing
- Retriever
    - TF-IDF
    - Dense Passage Retrieval
    - ElasticSearch
- Reader
    - 추가 데이터셋 (Korquad v1.0)
    - PLM
    - doc_stride
    - Query shape
- Postprocessing
    - Probability 합산

## 훈련, 평가, 추론

### Train & Evaluation

- 학습을 위해서 `do_train` 인자를 넣어주어야 합니다. `do_eval` 인자를 같이 넣어주면 학습과 평가를 동시에 진행합니다.

```python
python train.py \
--output_dir ./models/train_dataset \
--do_train \
--do_eval \
--overwrite_output_dir True \
--model_name_or_path "klue/bert-base" \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--num_train_epochs 3 \
--weight_decay 0.01 \
--warmup_ratio 0.1 \
--learning_rate 3e-5 \
--eval_steps 500 \
--logging_steps 500 \
--evaluation_strategy "steps" \
--project_name "test" \
--push_to_hub \
--push_to_hub_organization='nlp04' \
--push_to_hub_model_id="model_name"
```

### Inference

- 예측값을 출력하고자 한다면 `do_predict` 인자를 넣어주어야 합니다.

```python
python inference.py \
--output_dir "./outputs/test_dataset/" \
--dataset_name "../data/test_dataset/" \
--model_name_or_path "./models/train_dataset/" \
--do_predict \
--overwrite_output_dir True \
--use_faiss False \
--eval_retrieval True \
--retrieval_type dense \
--retrieval_class SparseRetrieval \
--retrieval_model_name_or_path "klue/bert-base" \
```

## 자체 평가 의견

### 잘한 점들

- 기존에 하던 task들과 다르게 크게 두가지 sub task로 나뉘어 서로 영향을 미치는 system적인 구조를 가지고 있다 보니 task를 정확하게 이해하는 것조차 어려웠음에도 불구하고, 할 수 있는 부분에서 팀원 모두 최선을 다했다.
- 난이도가 높은 task였음에도 불구하고 다양한 실험을 통해 성능 향상을 시도하였다.
- 학습한 Reader 모델을 HuggingFace Hub을 통해 팀원들과 공유함으로써 실험 시간을 줄이고 원활한 협업에 도움이 될 수 있었다.

### 시도 했으나 잘 되지 않았던 것들

- Query 길이를 증가시켰을 때 Reader 모델의 성능 향상이 이뤄지지 않았다. named entity뿐만 아니라 명사 전체를 추출하여 이어붙였으면 성능 향상이 가능하지 않았을까 하는 생각이 든다.
- Elasticsearch뿐만 아니라 BM25 임베딩을 직접 구현하고자 하였으나 실패하였다.
- 각 데이터 내에서 특수기호 전처리를 시도했지만 오히려 성능이 떨어져서 전처리를 하지 않았다.
- DPR이 TF-IDF에 비해서 실질적으로 성능향상에 기여하지 못하였다. 그리고 구현하는 과정에서 Faiss를 적용하는 부분이 잘 되지 않았다.

### 아쉬웠던 점들

- 대회 중간에 쉬는 날이 있었고, task를 이해하느라 실험을 늦게 시작해서 다른 대회보다 실험을 많이 못해본 것이 아쉽다.
- Retriever 성능 향상, Reader 성능 향상을 별도로 진행했었다면, 성능 개선을 정확히 어떤 부분에서 해야 하는지를 명확하게 알 수 있었을 것 같다.
- 모델의 일반화 성능에 대한 고려가 부족하였다. public score에서는 EM이 70.00으로 1등을 기록했으나, 최종 private 점수는 63.89로, 다른 조에 비해 많이 떨어져 등수 하락으로 이어졌다.

### 프로젝트를 통해 배운 점 또는 시사점

- ODQA라는 새로운 자연어 처리 분야에 대해서 배우고, 탐구할 수 있었다.
- 두 개의 모델을 연결짓는 파이프라인 구조에 대해 이해할 수 있었다.
