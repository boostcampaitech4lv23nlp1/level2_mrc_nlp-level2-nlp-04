import os
import json
import pickle

import time
import numpy as np
import pandas as pd
from tqdm import trange
from tqdm.auto import tqdm
from pprint import pprint

import torch

from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    BertModel, BertPreTrainedModel, RobertaModel, RobertaPreTrainedModel,
    AdamW, get_linear_schedule_with_warmup,
)

from typing import List, NoReturn, Optional, Tuple, Union
from contextlib import contextmanager
from datasets import Dataset, load_from_disk


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


class RobertaEncoder(RobertaPreTrainedModel):

    def __init__(self,
                 config
                 ):
        super(RobertaEncoder, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.init_weights()

    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None
                ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs[1]
        return pooled_output


class DenseRetrieval:
    def __init__(
        self,
        args,
        model_name_or_path,
        num_neg,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
        train_path="train_dataset"
    ) -> NoReturn:
        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """

        self.data_path = data_path
        with open(os.path.join(self.data_path, context_path), "r", encoding="utf-8") as f:
            self.wiki = json.load(f)
        self.contexts = list(dict.fromkeys(
            [v["text"] for v in self.wiki.values()]))  # set 은 매번 순서가 바뀌므로

        self.args = args
        self.dataset = load_from_disk(os.path.join(self.data_path, train_path))

        self.train_dataset = self.dataset["train"]
        self.validation_dataset = self.dataset["validation"]

        # 훈련시 필요한 contexts
        self.train_contexts = np.array(
            list(set([example for example in self.train_dataset["context"]])))
        self.validation_contexts = np.array(
            list(set([example for example in self.validation_dataset["context"]])))

        print(f"Lengths of Train unique contexts : {len(self.train_contexts)}")
        print(
            f"Lengths of Validation unique contexts : {len(self.validation_contexts)}")

        self.num_neg = num_neg

        # hugging face encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.p_encoder = RobertaEncoder.from_pretrained(
            model_name_or_path).to(args.device)
        self.q_encoder = RobertaEncoder.from_pretrained(
            model_name_or_path).to(args.device)

        self.train_tensor = self.prepare_in_batch_negative(
            dataset=self.train_dataset, contexts=self.train_contexts)
        self.validation_tensor = self.prepare_in_batch_negative(
            dataset=self.validation_dataset, contexts=self.validation_contexts)

        self.p_embedding = None  # get_embedding()로 생성합니다

    def prepare_in_batch_negative(self,
                                  dataset,
                                  contexts
                                  ):
        """
        Args
        dataset : each_dataset
        contexts : each_total_contexts
        ----
        dataset
            in-batch negative를 추가
        """

        # num neg, tokenizer, args 정의
        num_neg = self.num_neg
        tokenizer = self.tokenizer
        args = self.args

        p_with_neg = []

        for c in dataset["context"]:
            while True:
                neg_idxs = np.random.randint(len(contexts), size=num_neg)

                if not c in contexts[neg_idxs]:
                    p_neg = contexts[neg_idxs]

                    p_with_neg.append(c)
                    p_with_neg.extend(p_neg)
                    break

        q_seqs = tokenizer(
            dataset["question"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        p_seqs = tokenizer(
            p_with_neg,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        max_len = p_seqs["input_ids"].size(-1)  # hidden_size
        # negative sampling + positive
        p_seqs["input_ids"] = p_seqs["input_ids"].view(
            -1, num_neg + 1, max_len)
        p_seqs["attention_mask"] = p_seqs["attention_mask"].view(
            -1, num_neg + 1, max_len)
        # p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(-1, num_neg + 1, max_len)

        Tensor_dataset = TensorDataset(
            # p_seqs["token_type_ids"],
            p_seqs["input_ids"], p_seqs["attention_mask"],
            # q_seqs["token_type_ids"]
            q_seqs["input_ids"], q_seqs["attention_mask"],
        )

        return Tensor_dataset

    def train(self):
        """
        do_train.
        encoder들과 dataloader가 속성으로 저장되어있는 점에 유의해주세요.
        """

        args = self.args
        num_neg = self.num_neg

        train_batch_size = args.per_device_train_batch_size
        validation_batch_size = args.per_device_eval_batch_size

        # Dataloader
        train_dataloader = DataLoader(
            self.train_tensor, batch_size=train_batch_size)
        validation_dataloader = DataLoader(
            self.validation_tensor, batch_size=validation_batch_size)

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
            train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

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

        train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
        for _ in train_iterator:
            # epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            with tqdm(train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:

                    self.p_encoder.train()
                    self.q_encoder.train()

                    # positive example은 전부 첫 번째에 위치하므로
                    targets = torch.zeros(train_batch_size).long()
                    targets = targets.to(args.device)

                    # Bert
                    # p_inputs = {
                    #    "input_ids": batch[0].view(train_batch_size * (num_neg + 1), -1).to(args.device),
                    #    "attention_mask": batch[1].view(train_batch_size * (num_neg + 1), -1).to(args.device),
                    #    "token_type_ids": batch[2].view(train_batch_size * (num_neg + 1), -1).to(args.device)
                    # }

                    # q_inputs = {
                    #    "input_ids": batch[3].to(args.device),
                    #    "attention_mask": batch[4].to(args.device),
                    #    "token_type_ids": batch[5].to(args.device)
                    # }

                    # Roberta
                    p_inputs = {
                        "input_ids": batch[0].view(train_batch_size * (num_neg + 1), -1).to(args.device),
                        "attention_mask": batch[1].view(train_batch_size * (num_neg + 1), -1).to(args.device),
                    }

                    q_inputs = {
                        "input_ids": batch[2].to(args.device),
                        "attention_mask": batch[3].to(args.device),
                    }

                    del batch

                    torch.cuda.empty_cache()
                    # (batch_size * (num_neg + 1), emb_dim)
                    p_outputs = self.p_encoder(**p_inputs)
                    # (batch_size, emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)

                    # Calculate similarity score & loss
                    p_outputs = p_outputs.view(
                        train_batch_size, num_neg + 1, -1)
                    q_outputs = q_outputs.view(train_batch_size, 1, -1)

                    # (batch_size, num_neg + 1)
                    sim_scores = torch.bmm(
                        q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()
                    sim_scores = sim_scores.view(train_batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f" {str(loss.item())}")
                    # wandb.log({"train loss":loss})

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.q_encoder.zero_grad()
                    self.p_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()
                    del p_inputs, q_inputs

            with tqdm(validation_dataloader, unit="batch") as vepoch:

                losses = []
                for batch in vepoch:
                    with torch.no_grad():
                        self.p_encoder.eval()
                        self.q_encoder.eval()

                        # positive example은 전부 첫 번째에 위치하므로
                        targets = torch.zeros(validation_batch_size).long()
                        targets = targets.to(args.device)

                        # Bert
                       # p_inputs = {
                       #     "input_ids": batch[0].view(validation_batch_size * (num_neg + 1), -1).to(args.device),
                       #     "attention_mask": batch[1].view(validation_batch_size * (num_neg + 1), -1).to(args.device),
                       #     "token_type_ids": batch[2].view(validation_batch_size * (num_neg + 1), -1).to(args.device)
                       # }

                       # q_inputs = {
                       #     "input_ids": batch[3].to(args.device),
                       #     "attention_mask": batch[4].to(args.device),
                       #     "token_type_ids": batch[5].to(args.device)
                       # }

                        # Roberta
                        p_inputs = {
                            "input_ids": batch[0].view(validation_batch_size * (num_neg + 1), -1).to(args.device),
                            "attention_mask": batch[1].view(validation_batch_size * (num_neg + 1), -1).to(args.device)
                        }

                        q_inputs = {
                            "input_ids": batch[2].to(args.device),
                            "attention_mask": batch[3].to(args.device)
                        }

                        del batch

                        torch.cuda.empty_cache()
                        # (batch_size * (num_neg + 1), emb_dim)
                        p_outputs = self.p_encoder(**p_inputs)
                        # (batch_size, emb_dim)
                        q_outputs = self.q_encoder(**q_inputs)

                        # Calculate similarity score & loss
                        p_outputs = p_outputs.view(
                            validation_batch_size, num_neg + 1, -1)
                        q_outputs = q_outputs.view(
                            validation_batch_size, 1, -1)

                        # (batch_size, num_neg + 1)
                        sim_scores = torch.bmm(
                            q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()
                        sim_scores = sim_scores.view(validation_batch_size, -1)
                        sim_scores = F.log_softmax(sim_scores, dim=1)

                        loss = F.nll_loss(sim_scores, targets)
                        losses.append(loss.item())
                print("validation loss= ", np.array(losses).mean())

    def get_embedding(self):
        """
        Summary:
            Passage Embedding을 만들고
            Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        args = self.args

        # Pickle을 저장합니다.
        pickle_name = f"dense_embedding.bin"
        emd_path = os.path.join(self.data_path, pickle_name)
        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            print("Embedding pickle load")
        else:
            print("Build passage embedding")

            self.train()

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

            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            print("Embedding pickle saved.")

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
            만약 None이 들어오면 self.args를 쓰도록 짜면 좋을 것 같습니다.

        Do
        --
        1. query를 받아서 embedding을 하고
        2. 전체 passage와의 유사도를 구한 후
        3. 상위 k개의 문서 index를 반환합니다.
        """

        args = self.args

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

        args = self.args

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

        assert self.p_embedding is not None, "get_embedding() 메소드를 먼저 수행해줘야합니다."

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
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
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
