from datasets import load_dataset
import json
import os

data_files = {
    'train': 'KorQuAD_v1.0_train.json',
    'validation': 'KorQuAD_v1.0_dev.json'
}

schema = {
    "__index_level_0__": 0,
    'answers': {
        'answer_start': 0,
        'text': '답'
    },
    'context': '글',
    'document_id': 0,
    'id': '28-9-9',
    'question': '질문',
    'title': '제목',
}

dataset = load_dataset("json", data_files=data_files)
print(dataset)
# print(dataset['train']['data'][0][0]['paragraphs'])
# dataset.save_to_disk("test")
# st_json = json.dumps(student_data)


def new_schema(dataset, split):
    idx_cnt = 0

    for data_list in dataset['data']:
        for doc_dict in data_list:

            paras = doc_dict['paragraphs']
            title = doc_dict['title']

            for para_dict in paras:
                qass = para_dict['qas']
                context = para_dict['context']

                for qas_dict in qass:
                    question = qas_dict['question']
                    id = qas_dict['id']
                    anss = qas_dict['answers']

                    ans_list = []
                    ans_start_list = []

                    for ans_dict in anss:
                        ans_start = ans_dict['answer_start']
                        ans_text = ans_dict['text']

                        ans_list.append(ans_text)
                        ans_start_list.append(ans_start)

                    new_sample = {
                        "__index_level_0__": idx_cnt,
                        'answers': {
                            'answer_start': ans_start_list,
                            'text': ans_list
                        },
                        'context': context,
                        'document_id': int(id.split('-')[0]),
                        'id': 'korquad-'+id,
                        'question': question,
                        'title': title,
                    }
                    idx_cnt += 1

                    with open(split+'-korquadv1.json', 'a', encoding='utf-8') as f:
                        json.dump(new_sample, f)



if 'train-korquadv1.json' not in os.listdir():
    new_schema(dataset['train'], 'train')
if 'valid-korquadv1.json' not in os.listdir():
    new_schema(dataset['validation'], 'valid')

new_data_files = {
    'train': 'train-korquadv1.json',
    'validation': 'valid-korquadv1.json'
}

new_dataset = load_dataset("json", data_files=new_data_files)

print(new_dataset)
print(new_dataset['train'][1])

new_dataset.save_to_disk("korquadv1")