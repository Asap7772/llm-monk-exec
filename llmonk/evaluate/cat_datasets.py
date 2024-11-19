import datasets
import os
os.environ['HF_TOKEN'] = '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'

all_dataset_names = [
    'Asap7772/code_contests_llamabase_mc_intermediate-part1-of-4',
    'Asap7772/code_contests_llamabase_mc_intermediate-part2-of-4',
    'Asap7772/code_contests_llamabase_mc_intermediate-part3-of-4',
    'Asap7772/code_contests_llamabase_mc_intermediate-part4-of-4',
]

all_ds = [datasets.load_dataset(name) for name in all_dataset_names]
all_ds = datasets.concatenate_datasets([x['train'] for x in all_ds])
all_ds.push_to_hub('Asap7772/code_contests_llamabase_mc_intermediate')