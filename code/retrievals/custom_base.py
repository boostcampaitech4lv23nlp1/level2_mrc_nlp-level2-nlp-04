
class CustomRetrieval:
    """
    Summary:
        Abstract class for custom retrieval

        * init에서는 tokenizer_fn, data_path, context_path를 받아와야 함
        * get_embedding, retrieve 함수가 포함되어 있어야 함
            (faiss=True일 때는 build_faiss, retrieve_faiss함수가 구현되어야 함)
        * 새로운 retrieval 함수 파일 추가 후 __init__.py에 
            from .custom_base import CustomRetrieval
            을 추가해 주어야 함
        (위 세개는 inference.py에서의 호환성 때문임)

    """

    def __init__(
        self,
        tokenize_fn,
        data_path,
        context_path,):
        return 0

    def get_embedding(self):
        return 0
    
    def retrieve(self):
        return 0