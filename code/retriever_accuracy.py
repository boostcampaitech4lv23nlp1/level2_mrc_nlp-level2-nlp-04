
############### 설명 ###############
# retriever의 topk(k=1, 10, 20, 50, 100) accuracy를 계산하여 csv파일로 반환한다. 
# sparse retriever를 테스트할 때 기존의 sparse_embedding과 tf-idf 파일을 지우고 tokenizer에 맞춰서 다시 생성한다. 

############### 실행시킬 때 주의사항 ###############
# 테스트할 retriever를 인자로 전달합니다. 
# --dataset_path        : 테스트하고자 하는 데이터셋. default='../data/train_dataset'
# --retrieval_type      : ['sparse', 'elastic']
# --sparse_tokenizer    : sparse tokenizer 종류 ['bert-multi', 'koelectra', 'xlm-roberta-large', 'split']
# --index_name          : elasticsearch 인덱스 이름

############### 예시 명령어 ###############
# sparse    : python retriever_accuracy.py --retrieval_type sparse --sparse_tokenizer koelectra
# elastic   : python retriever_accuracy.py --retrieval_type elastic --index_name wiki-base


from retrievals import SparseRetrieval
from elasticsearch_retrieval import ElasticSearchRetrieval
from transformers import AutoTokenizer
import pandas as pd
import os
from datasets import load_from_disk
import argparse


def split_fn(str):
    return list(str.split())

retrieval_tokenizer_fn = {
    'bert-multi': AutoTokenizer.from_pretrained("bert-base-multilingual-cased").tokenize,
    'koelectra': AutoTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator').tokenize,
    'xlm-roberta-large': AutoTokenizer.from_pretrained('xlm-roberta-large').tokenize,
    'split': split_fn,
#    'mecab_morphs': Mecab().morphs,
#    'mecab_nouns': Mecab().nouns,
}


TOP_K = [1, 10, 20, 50, 100]

def get_accuracy_sparse(df):
    cnt_total = len(df)
    cnt_correct = 0


    for i in range(cnt_total):
        topk_contexts = df['context'][i]
        org_context = df['original_context'][i]

        if org_context in topk_contexts:
            cnt_correct += 1


    return round(cnt_correct/cnt_total*100, 2)


def sparse_accuracy(data_path, tokenizer):
    
    datasets = load_from_disk(data_path)
    
    result_dict = {
        'tokenizer': [],
        'split': [],
        'top1': [],
        'top10': [],
        'top20': [],
        'top50': [],
        'top100': []
    }
    
    emb_file = '../data/sparse_embedding.bin'
    tfidf_file = '../data/tfidv.bin'
    if os.path.isfile(emb_file):
        os.remove(emb_file)
    if os.path.isfile(tfidf_file):
        os.remove(tfidf_file)

    retriever = SparseRetrieval(retrieval_tokenizer_fn[tokenizer], "../data/", "wikipedia_documents.json")
    retriever.get_embedding()
    
    for split in ['train', 'validation']:

        result_dict['tokenizer'].append(tokenizer)
        result_dict['split'].append(split)

        for topk in TOP_K:

            df = retriever.retrieve(datasets[split], topk=topk)

            accuracy = get_accuracy_sparse(df)

            result_dict[f'top{topk}'].append(accuracy)


    result_df = pd.DataFrame(result_dict)
    result_df.to_csv('./outputs/topk_acc_sparse.csv')


def get_accuracy_elastic(df, dataset):
    cnt_total = len(df)
    cnt_correct = 0


    for i in range(cnt_total):
        topk_contexts = df['context'][i]
        org_context = dataset[i]['context']

        if org_context in topk_contexts:
            cnt_correct += 1


    return round(cnt_correct/cnt_total*100, 2)


def elastic_accuracy(data_path, index_name):
    datasets = load_from_disk(data_path)
    retriever = ElasticSearchRetrieval(index_name)
    
    result_dict = {
        'index': [],
        'split': [],
        'top1': [],
        'top10': [],
        'top20': [],
        'top50': [],
        'top100': [],
    }

    for split in ['train', 'validation']:
        result_dict['split'].append(split)
        result_dict['index'].append(index_name)
        for topk in TOP_K:
            df = retriever.retrieve(datasets[split], topk=topk)
            print(df.columns)
            acc = get_accuracy_elastic(df, datasets[split])
            result_dict['top'+str(topk)].append(acc)

    result_df = pd.DataFrame(result_dict)

    result_df.to_csv('./outputs/topk_acc_elastic.csv')





def main(args):
    if args.retrieval_type == 'sparse':
        sparse_accuracy(args.dataset_path, args.sparse_tokenizer)
    elif args.retrieval_type == 'elastic':
        elastic_accuracy(args.dataset_path, args.index_name)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_path", default='../data/train_dataset', help='테스트하고자 하는 데이터셋.')
    parser.add_argument("--retrieval_type", default='sparse', type=str, help="['sparse', 'elastic']")
    parser.add_argument("--sparse_tokenizer", default='koelectra', type=str, help="sparse tokenizer 종류 ['bert-multi', 'koelectra', 'xlm-roberta-large', 'split']")
    parser.add_argument("--index_name", default="wiki-base", type=str, help="Set index name")
    args = parser.parse_args()
    
    if args.retrieval_type not in ['sparse', 'elastic']:
        print('인자를 확인해주세요')
    else:
        main(args)