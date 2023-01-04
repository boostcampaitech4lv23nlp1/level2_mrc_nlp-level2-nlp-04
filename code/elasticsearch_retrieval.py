import time
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm
from elasticsearch_setting import *
from utils_qa import add_ner_func
from arguments import DataTrainingArguments


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

class ElasticSearchRetrieval:
    def __init__(
        self, index_name : str = "wiki-base"
    ) -> NoReturn:
        self.es = elasticsearch_setting()
        self.index_name = index_name
        # 인덱스 저장 확인 => 저장되어 있지 않을시 elasticsearch_setting.py로 저장 후 실행
        if self.es.indices.exists(index=self.index_name):
            print("Index Exists")
        else:
            print("Need to save index")

    # 쿼리 결과 확인
    def query_search(
        self, query : str, topk : int
    ): 
        # query 문장에 NER을 이어붙일지의 여부
        if DataTrainingArguments.add_ner:
            query = add_ner_func(query)
            
        body = {
            "query": {
                "bool": {
                    "must": [{"match": {"document_text": query}}]
                }
            }
        }
        result = self.es.search(index=self.index_name, body=body, size=topk)
        return result 

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

        if isinstance(query_or_dataset, str):
            print("[Search query]\n", query_or_dataset, "\n")
            doc_scores, doc_indices, docs = self.get_relevant_doc(query_or_dataset, topk)

            for i in range(topk):
                print(f"[Top-{i+1}] Doc ID: {doc_indices[i]}\t Score: {doc_scores[i]:.3f}") # topk 문서 id, score 출력
                print(docs[i]['_source']['document_text']) # topk 문서 내용 출력

            return (doc_scores, [docs[i]['_source']['document_text'] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices, docs = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval with ElasticSearch")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [docs[idx][i]['_source']['document_text'] for i in range(topk)]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    #tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(
        self, query: str, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        doc_scores, doc_indices = [], []
        result = self.query_search(query, k)
        docs = result['hits']['hits']

        for hit in result['hits']['hits']:
            doc_scores.append(hit['_score'])
            doc_indices.append(hit['_id'])     
        
        return doc_scores, doc_indices, docs

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        doc_scores, doc_indices, docs = [], [], []

        for query in queries:
            result = self.query_search(query, k)
            doc_score, doc_indice = [], []
            for hit in result['hits']['hits']:
                doc_score.append(hit['_score'])
                doc_indice.append(hit['_id'])
            doc_scores.append(doc_score)
            doc_indices.append(doc_indice)
            docs.append(result['hits']['hits'])

        return doc_scores, doc_indices, docs