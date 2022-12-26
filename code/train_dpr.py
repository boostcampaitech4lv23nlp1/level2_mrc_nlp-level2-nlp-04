import logging
import os
import sys
from typing import NoReturn
import wandb

from arguments import DataTrainingArguments, ModelArguments, WandbArguments
from datasets import DatasetDict, load_from_disk, load_metric
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from retrievals.dense import DenseRetrieval


def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    # parser = HfArgumentParser(
    #     (ModelArguments, DataTrainingArguments, TrainingArguments, WandbArguments)
    # )
    # model_args, data_args, training_args, wandb_args = parser.parse_args_into_dataclasses()

    # [참고] argument를 manual하게 수정하고 싶은 경우에 아래와 같은 방식을 사용할 수 있습니다
    # training_args.per_device_train_batch_size = 4
    # print(training_args.per_device_train_batch_size)

    # print(f"model is from {model_args.model_name_or_path}")
    # print(f"data is from {data_args.dataset_name}")
    

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(42)
    wandb_args =WandbArguments()
    wandb_args.entity_name = "sajo-tuna"
    wandb_args.project_name = "DPRTest"
    
    args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01
    )

    model_name_or_path = "klue/bert-base"

    retriever = DenseRetrieval(
        args=args,
        model_name_or_path = model_name_or_path,
        num_neg=2,
        
    )  
    
    retriever.train(wandb_args)
    retriever.get_embedding()
    
    query = "미국의 대통령은?"
    retriever.retrieve(query_or_dataset=query, topk=5)

if __name__ == "__main__":
    main()
