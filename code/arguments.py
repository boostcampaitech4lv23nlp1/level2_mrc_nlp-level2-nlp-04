from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="klue/bert-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="../data/train_dataset",
        metadata={"help": "The name of the dataset to use."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    eval_retrieval: bool = field(
        default=True,
        metadata={
            "help": "Whether to run passage retrieval using sparse embedding."},
    )
    retrieval_class: str = field(
        default="SparseRetrieval",
        metadata={
            "help": "Retrieval ????????? ??????"
        },
    )
    num_clusters: int = field(
        default=64, metadata={"help": "Define how many clusters to use for faiss."}
    )
    top_k_retrieval: int = field(
        default=10,
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
    )
    use_faiss: bool = field(
        default=False, metadata={"help": "Whether to build with faiss"}
    )
    index_name: str = field(
        default="wiki-base", metadata={"help": "Elasticsearch Index Settings Name"}
    )

    add_ner: bool = field(
        default=False, metadata={"help": "Whether put the NER result after the query"}
    )
    query_filter: bool = field(
        default=False, metadata={"help": "Whether to pre-process the query"}
    )


@dataclass
class WandbArguments:
    """
    Arguments for wandb setting
    """
    project_name: str = field(
        default="test",
    )
    entity_name: Optional[str] = field(
        default="sajo-tuna",
    )


@dataclass
class RetrievalArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    retrieval_model_name_or_path: str = field(
        default="klue/bert-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )

    retrieval_dataset_name: Optional[str] = field(
        default="../data/train_dataset",
        metadata={"help": "The name of the dataset to use."},
    )

    num_neg: int = field(
        default=2
    )
    retrieval_type: str = field(
        default=None,
        metadata={
            "help": "retrieval type"
        },
    )
    # TrainingArgument??? ???????????? ???????????????
    # --------------------------------------
    retrieval_output_dir: Optional[str] = field(
        default="dense_retrieval",
    )

    retrieval_learning_rate: float = field(
        default=2e-5
    )

    retrieval_per_device_train_batch_size: int = field(
        default=8
    )

    retrieval_per_device_eval_batch_size: int = field(
        default=8
    )

    retrieval_num_train_epochs: int = field(
        default=3
    )

    retrieval_weight_decay: float = field(
        default=0.01
    )

    retrieval_warmup_steps: float = field(
        default=0
    )
    # --------------------------------------
