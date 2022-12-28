from tqdm import tqdm
import argparse
import json
import warnings

from elasticsearch import Elasticsearch

warnings.filterwarnings('ignore')


def elasticsearch_setting():
    # Elasticsearch server setting
    es = Elasticsearch('http://localhost:9200', max_retries=20, retry_on_timeout=True) # Elastichsearch 접속
    if es.ping(): # 정상 접속시 True 반환
        print("Elasticsearch Connected")
    else:
        print("Elasticsearch connection failed")
    return es
    
# 인덱스 생성
def index_setting(es, index_name, index_settings_path):
    if es.indices.exists(index=index_name):
        print(f"Delete existing index : {index_name}")  
        es.indices.delete(index=index_name)

    with open(index_settings_path, "r") as f:
        index_settings = json.load(f)
        
    es.indices.create(index=index_name, body=index_settings)
    print(f"Index {index_name} Setting Completed")     
    
# 위키 데이터 로드
def data_load(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        wiki = json.load(f)  

    wiki_contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))  
    wiki_articles = [{"document_text": wiki_contexts[i]} for i in range(len(wiki_contexts))] # 인덱스 셋팅에 맞게 형식 맞춤
    return wiki_articles
                
# 문서 삽입
def doc_setting(es, index_name, wiki_articles):
    for doc_id, doc in enumerate(tqdm(wiki_articles)):
        es.index(index=index_name, id=doc_id, document=doc)

    n_records = es.count(index=index_name)['count']
    print(f'Succesfully loaded {n_records} docs into {index_name}') # 인덱스에 문서 삽입 완료
    

def main(args):
    es = elasticsearch_setting()
    index_setting(es, args.index_name, args.index_settings_path)
    wiki_articles = data_load(args.data_path)
    doc_setting(es, args.index_name, wiki_articles)
    print("Elasticsearch setting is done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--index_name", default="wiki-base", type=str, help="Set index name")
    parser.add_argument("--index_settings_path", default="index_settings.json", type=str, help="Index Settings json File Path")
    parser.add_argument("--data_path", default="./data/wikipedia_documents.json", type=str, help="Set wiki data path")
    args = parser.parse_args()
    main(args)