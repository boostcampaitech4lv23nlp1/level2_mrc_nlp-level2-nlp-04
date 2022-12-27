from datasets import concatenate_datasets
from datasets import load_from_disk
from datasets import DatasetDict

original_data = load_from_disk('../data/train_dataset')
korquad_data = load_from_disk('./korquadv1')

new_data = original_data.copy()
new_data['train'] = concatenate_datasets([original_data['train'], korquad_data['train']])

new_data = DatasetDict(new_data)

new_data.save_to_disk('org_and_korquad')