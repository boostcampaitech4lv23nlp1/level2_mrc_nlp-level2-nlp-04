import os
import json
import pickle
import time
import wandb
import faiss
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

from transformers import (
    AutoTokenizer,
    BertModel,
    BertPreTrainedModel,
    AdamW,
    get_linear_schedule_with_warmup
)
from datasets import Dataset, load_from_disk

from tqdm import trange
from tqdm.auto import tqdm
from typing import List, Optional, Tuple, Union
from contextlib import contextmanager

# TODO: wandb logging and huggingface hub porting


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class BertEncoder(BertPreTrainedModel):
    def __init__(self,
                 config
                 ):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None
                ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs[1]
        return pooled_output


class DenseRetrieval:
    def __init__(
        self,
        training_args,
        retrieval_args,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> None:
        """
        Arguments:
            args:
                TrainingArguments 형태의 argument입니다.
            retrieval_args:
                RetrievalArguments 형태의 argument입니다.
                model_name_or_path, dataset_name을 사용합니다

                model_name_or_path: pre_trained model 경로입니다
                dataset_name: mrc의 train 데이터 셋이 있는 경로입니다. 모델 훈련시 필요합니다
            data_path:
                데이터가 보관되어 있는 경로입니다.
            context_path:
                Passage들이 묶여있는 파일명입니다.
            data_path/context_path가 존재해야합니다.
        Summary:
            파일을 불러오고 Encoder 선언 및 in batch negative를 수행합니다.
        """

        self.data_path = data_path
        self.model_name_or_path = retrieval_args.retrieval_model_name_or_path
        self.dataset_name = retrieval_args.retrieval_dataset_name
        self.training_args = training_args

        with open(os.path.join(self.data_path, context_path), "r", encoding="utf-8") as f:
            self.wiki = json.load(f)
        self.contexts = list(dict.fromkeys(
            [v["text"] for v in self.wiki.values()]))  # set 은 매번 순서가 바뀌므로

        self.dataset = load_from_disk(self.dataset_name)

        self.train_dataset = self.dataset["train"]  # train인지 확인!
        self.validation_dataset = self.dataset["validation"]

        self.origin_dataset = load_from_disk("../data/train_dataset")
        self.origin_validation_dataset = self.origin_dataset["validation"]
        # 훈련시 필요한 contexts
        self.validation_contexts = np.array(
            list([example for example in self.validation_dataset["context"]]))

        # p_embedding의 저장 경로 지정
        pickle_name = f"dense_embedding.bin"
        self.emd_path = os.path.join(self.data_path, pickle_name)

        # hugging face encoder
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

        # gpu 메모리를 효율적으로 사용하기 위해 train 또는 retriever 단계에서 선언합니다
        self.p_encoder = None
        self.q_encoder = None

        self.train_tensor = self.prepare_in_batch_negative(
            dataset=self.train_dataset)
        self.validation_tensor = self.prepare_in_batch_negative(
            dataset=self.validation_dataset)

        self.p_embedding = None  # get_embedding()로 생성합니다
        self.indexer = None  # build_faiss()로 생성합니다.

    def prepare_in_batch_negative(self,
                                  dataset,
                                  ):
        """
        Arguments:
            dataset : each_dataset
        Summary:
            Tensor 형태로 만듭니다.
        """

        tokenizer = self.tokenizer

        q_seqs = tokenizer(
            dataset["question"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        p_seqs = tokenizer(
            dataset["context"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        Tensor_dataset = TensorDataset(
            p_seqs["input_ids"], p_seqs["attention_mask"], p_seqs["token_type_ids"],
            q_seqs["input_ids"], q_seqs["attention_mask"], q_seqs["token_type_ids"]
        )

        return Tensor_dataset

    def train(self, wandb_args):
        """
        Summary:
            실제 train이 수행되는 함수입니다.
            klue/bert-base 기준으로 작성되었습니다.
        """

        accumulation_steps = 16
        args = self.training_args
        origin_loss = 1000000  # acc 계산

        wandb.init(project=wandb_args.project_name,
                   entity=wandb_args.entity_name)

        # 기존의 p_embbeding이 있다면 삭제합니다
        if os.path.isfile(self.emd_path):
            os.remove(self.emd_path)

        # 훈련시킬 사전학습된 encoder load
        self.p_encoder = BertEncoder.from_pretrained(
            self.model_name_or_path).to(args.device)
        self.q_encoder = BertEncoder.from_pretrained(
            self.model_name_or_path).to(args.device)

        train_batch_size = args.per_device_train_batch_size
        validation_batch_size = args.per_device_eval_batch_size

        # Dataloader
        train_sampler = RandomSampler(self.train_tensor)
        train_dataloader = DataLoader(
            self.train_tensor, sampler=train_sampler, batch_size=train_batch_size, drop_last=True)
        validation_dataloader = DataLoader(
            self.validation_tensor, batch_size=validation_batch_size, drop_last=True)

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.p_encoder.named_parameters() if not any(
                nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in self.p_encoder.named_parameters() if any(
                nd in n for nd in no_decay)], "weight_decay": 0.0},
            {"params": [p for n, p in self.q_encoder.named_parameters() if not any(
                nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
            {"params": [p for n, p in self.q_encoder.named_parameters() if any(
                nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon
        )

        t_total = len(
            train_dataloader) // self.training_args.gradient_accumulation_steps * self.training_args.num_train_epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=t_total
        )

        # Start training!
        global_step = 0

        # gradient 초기화
        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        for _ in range(int(args.num_train_epochs)):
            with tqdm(train_dataloader, unit="batch") as tepoch:
                for step, batch in enumerate(tepoch):

                    self.p_encoder.train()
                    self.q_encoder.train()
                    if torch.cuda.is_available():
                        batch = tuple(t.cuda() for t in batch)

                    # Bert
                    p_inputs = {
                        "input_ids": batch[0].to(args.device),
                        "attention_mask": batch[1].to(args.device),
                        "token_type_ids": batch[2].to(args.device)
                    }

                    q_inputs = {
                        "input_ids": batch[3].to(args.device),
                        "attention_mask": batch[4].to(args.device),
                        "token_type_ids": batch[5].to(args.device)
                    }

                    del batch
                    torch.cuda.empty_cache()

                    p_outputs = self.p_encoder(**p_inputs)
                    q_outputs = self.q_encoder(**q_inputs)

                    # Calculate similarity score & loss
                    # (batch_size, emb_dim) x (emb_dim, batch_size) = (batch_size, batch_size)
                    sim_scores = torch.matmul(
                        q_outputs, torch.transpose(p_outputs, 0, 1))

                    # target: position of positive samples = diagonal element
                    targets = torch.arange(
                        0, args.per_device_train_batch_size).long()
                    if torch.cuda.is_available():
                        targets = targets.to('cuda')

                    sim_scores = F.log_softmax(sim_scores, dim=1)
                    loss = F.nll_loss(sim_scores, targets)
                    loss = loss / accumulation_steps

                    tepoch.set_postfix(loss=f" {str(loss.item())}")
                    wandb.log({"train loss": loss})

                    loss.backward()

                    if (step+1) % accumulation_steps == 0:
                        optimizer.step()
                        scheduler.step()

                        self.q_encoder.zero_grad()
                        self.p_encoder.zero_grad()

                    global_step += 1

                    del p_inputs, q_inputs
                    torch.cuda.empty_cache()

            with tqdm(validation_dataloader, unit="batch") as vepoch:

                losses = []
                for batch in vepoch:
                    with torch.no_grad():
                        self.p_encoder.eval()
                        self.q_encoder.eval()

                        if torch.cuda.is_available():
                            batch = tuple(t.cuda() for t in batch)

                        # Bert
                        p_inputs = {
                            "input_ids": batch[0].to(args.device),
                            "attention_mask": batch[1].to(args.device),
                            "token_type_ids": batch[2].to(args.device)
                        }

                        q_inputs = {
                            "input_ids": batch[3].to(args.device),
                            "attention_mask": batch[4].to(args.device),
                            "token_type_ids": batch[5].to(args.device)
                        }

                        del batch
                        torch.cuda.empty_cache()

                        p_outputs = self.p_encoder(**p_inputs)
                        q_outputs = self.q_encoder(**q_inputs)

                        # Calculate similarity score & loss
                        # (batch_size, emb_dim) x (emb_dim, batch_size) = (batch_size, batch_size)
                        sim_scores = torch.matmul(
                            q_outputs, torch.transpose(p_outputs, 0, 1))

                        # target: position of positive samples = diagonal element
                        targets = torch.arange(
                            0, args.per_device_train_batch_size).long()
                        if torch.cuda.is_available():
                            targets = targets.to('cuda')

                        sim_scores = F.log_softmax(sim_scores, dim=1)

                        loss = F.nll_loss(sim_scores, targets)
                        losses.append(loss.item())

                        del p_inputs, q_inputs
                        torch.cuda.empty_cache()

                validation_log_dict = dict()
                validation_log_dict["validation loss"] = np.array(
                    losses).mean()
                wandb.log(validation_log_dict)

        self.p_encoder.save_pretrained(args.output_dir+"/p_encoder")
        self.q_encoder.save_pretrained(args.output_dir+"/q_encoder")
        wandb.finish()

    def get_embedding(self):
        """
        Summary:
            Passage Embedding을 만들고
            Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        args = self.training_args

        # 만약 p_embedding Pickle 파일이 없다면 이를 저장합니다.
        # train 함수가 한번이라도 실행되어 p_encoder, q_encoder가 존재하는 상태로 가정합니다
        if os.path.isfile(self.emd_path):
            with open(self.emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            print("Embedding pickle load")
        else:
            print("Build passage embedding")

            self.p_encoder = BertEncoder.from_pretrained(
                args.output_dir+"/p_encoder").to(args.device)  # 파일이 없다면 p_encoder를 load 합니다

            with torch.no_grad():
                self.p_encoder.eval()
                p_embs = []
                with tqdm(self.contexts) as texts:
                    for p in texts:
                        p = self.tokenizer(
                            p, padding="max_length", truncation=True, return_tensors="pt").to(args.device)
                        p_emb = self.p_encoder(**p).to("cpu").numpy()
                        p_embs.append(p_emb)

            self.p_embedding = torch.Tensor(
                p_embs).squeeze()  # (num_passage, emb_dim)

            with open(self.emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            print("Embedding pickle saved.")

        self.q_encoder = BertEncoder.from_pretrained(
            args.output_dir+"/q_encoder").to(args.device)  # q_encoder를 불러옵니다

        # accuracy를 계산합니다
        
        queries = self.origin_validation_dataset["question"]
        ground_truth = self.origin_validation_dataset["context"]

        with torch.no_grad():
            self.q_encoder.eval()
            q_seqs_val = self.tokenizer(queries, padding="max_length", truncation=True, return_tensors="pt").to(
                args.device)  # (num_query, emb_dim)
            q_emb = self.q_encoder(**q_seqs_val).to("cpu")

            # (num_passage, emb_dim)
            dot_prod_scores = torch.matmul(
                q_emb, torch.transpose(self.p_embedding, 0, 1))
            rank = torch.argsort(dot_prod_scores, dim=1,
                                 descending=True).squeeze()
            topks = [10, 20, 50, 100]  # 10,20,50,100을 기준으로 accuracy를 계산합니다
            # query 및 ground truth를 받아와야함
            for topk in topks:
                score = 0
                for idx, query in enumerate(queries):
                    r = rank[idx]
                    r_ = r[:topk+1]
                    passages = [self.contexts[i] for i in r_]
                    if ground_truth[idx] in passages:
                        score += 1

                accuracy = score / len(queries)
                print(f"top k-{topk} accuracy : {accuracy}")

    def get_relevant_doc(self, query, k=1, args=None):
        """
        Args
        ----
        query (str)
            문자열로 주어진 질문입니다.

        k (int)
            상위 몇 개의 유사한 passage를 뽑을 것인지 결정합니다.

        args
            Configuration을 필요한 경우 넣어줍니다.
            만약 None이 들어오면 self.training_args를 쓰도록 짜면 좋을 것 같습니다.

        Do
        --
        1. query를 받아서 embedding을 하고
        2. 전체 passage와의 유사도를 구한 후
        3. 상위 k개의 문서 index를 반환합니다.
        """

        args = self.training_args

        with torch.no_grad():
            self.q_encoder.eval()

            q_seqs_val = self.tokenizer(
                [query], padding="max_length", truncation=True, return_tensors="pt").to(args.device)
            # (num_query, emb_dim)
            q_emb = self.q_encoder(**q_seqs_val).to("cpu")

        dot_prod_scores = torch.matmul(
            q_emb, torch.transpose(self.p_embedding, 0, 1))

        sorted_result = torch.argsort(
            dot_prod_scores, dim=1, descending=True).squeeze()
        doc_score = dot_prod_scores.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        args = self.training_args

        with torch.no_grad():
            self.q_encoder.eval()

            q_seqs_val = self.tokenizer(
                queries, padding="max_length", truncation=True, return_tensors="pt").to(args.device)
            # (num_query, emb_dim)

            q_emb = self.q_encoder(**q_seqs_val).to("cpu")

        dot_prod_scores = torch.matmul(
            q_emb, torch.transpose(self.p_embedding, 0, 1))

        if not isinstance(dot_prod_scores, np.ndarray):
            dot_prod_scores = dot_prod_scores.to("cpu").numpy()

        doc_scores = []
        doc_indices = []
        for i in range(dot_prod_scores.shape[0]):
            sorted_result = np.argsort(dot_prod_scores[i, :])[::-1]
            doc_scores.append(
                dot_prod_scores[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])

        return doc_scores, doc_indices

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """
        args = self.training_args

        assert self.p_embedding is not None, "get_embedding() 메소드를 먼저 수행해줘야합니다."

        self.q_encoder = BertEncoder.from_pretrained(
            args.output_dir+"/q_encoder").to(args.device)  # q_encoder를 불러옵니다

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(
                query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])
                print(doc_indices[i])
            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Dense retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas
