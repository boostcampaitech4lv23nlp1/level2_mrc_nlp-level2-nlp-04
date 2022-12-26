from arguments import RetrievalArguments, WandbArguments
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from retrievals.dense import DenseRetrieval


def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    parser = HfArgumentParser(
        (RetrievalArguments, WandbArguments)
    )
    retrieval_args, wandb_args = parser.parse_args_into_dataclasses()

    # 설정하지 않은 나머지 인자들은 TrainingArguments의 기본 인자를 사용하기 위해서 이렇게 선언해줍니다.
    retrieval_training_args = TrainingArguments(
        output_dir=retrieval_args.retrieval_output_dir,
        learning_rate=retrieval_args.retrieval_learning_rate,
        per_device_train_batch_size=retrieval_args.retrieval_per_device_train_batch_size,
        per_device_eval_batch_size=retrieval_args.retrieval_per_device_eval_batch_size,
        gradient_accumulation_steps=retrieval_args.retrieval_gradient_accumulation_steps,
        num_train_epochs=retrieval_args.retrieval_num_train_epochs,
        weight_decay=retrieval_args.retrieval_weight_decay,
        warmup_steps=retrieval_args.retrieval_warmup_steps
    )

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(retrieval_training_args.seed)

    retriever = DenseRetrieval(
        training_args=retrieval_training_args,
        retrieval_args=retrieval_args
    )

    retriever.train(wandb_args)
    retriever.get_embedding()

    # test query
    query = "미국의 대통령은?"
    retriever.retrieve(query_or_dataset=query, topk=5)


if __name__ == "__main__":
    main()
