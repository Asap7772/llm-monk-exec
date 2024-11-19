from absl import app, flags
import datasets
import pandas as pd
from llmonk.evaluate.grade_problems_nodocker import grade_problems

FLAGS = flags.FLAGS
# flags.DEFINE_string("dataset", "Asap7772/code_contests", "Directory to load data from")
flags.DEFINE_string("dataset", "Asap7772/code_contests_llamabase_passk-part1-of-1", "Directory to load data from")
flags.DEFINE_integer("num_workers", 512, "Number of workers to use for grading")
flags.DEFINE_string("save_dir", "results", "Directory to save results in")
# flags.DEFINE_string("split", "valid", "Split to evaluate on")
flags.DEFINE_string("split", "train", "Split to evaluate on")
# flags.DEFINE_string('solution_col', 'solutions', 'Column name for solutions')
flags.DEFINE_string('solution_col', 'responses', 'Column name for solutions')
flags.DEFINE_integer('max_solutions', 256, 'Maximum number of solutions to evaluate')
flags.DEFINE_float('per_testcases', -1.0, 'Percentage of testcases to evaluate')

def load_data_from_dataset(ds, solution_col="responses", max_solutions=16, per_testcases=-1):
    keys = ['test_cases', 'solutions', 'name', 'timeout']
    all_prompts = sorted(list(set(ds['prompt'])))
    prompt_to_idx = {prompt: i for i, prompt in enumerate(all_prompts)}
    
    def map_fn(examples):
        return_dict = {k:[] for k in keys}
        for i in range(len(examples['prompt'])):
            curr_prompt = examples['prompt'][i]
            which_prompt = prompt_to_idx[curr_prompt]
            name = f"problem{which_prompt}"
            return_dict['name'].append(name)
            if per_testcases > 0:
                num_testcases = len(examples['test_cases'][i])
                num_testcases_to_keep = int(num_testcases * per_testcases)
                test_cases = examples['test_cases'][i][:num_testcases_to_keep]
                return_dict['test_cases'].append(test_cases)
            else:
                return_dict['test_cases'].append(examples['test_cases'][i])
            return_dict['solutions'].append(examples[solution_col][i][:max_solutions])
            return_dict['timeout'].append(examples['timeout'][i])
        return return_dict
    all_cols = list(ds.column_names)
    rm_cols = [col for col in all_cols if col not in keys]
    ds_mapped = ds.map(map_fn, batched=True, num_proc=8, remove_columns=rm_cols)
    # convert to dict of lists
    df = ds_mapped.data.to_pandas()
    return df.to_dict(orient='records')


def main(_):
    ds = datasets.load_dataset(FLAGS.dataset, split=FLAGS.split)
    solutions_data = load_data_from_dataset(ds, solution_col=FLAGS.solution_col, max_solutions=FLAGS.max_solutions, per_testcases=FLAGS.per_testcases)
    grade_problems(solutions_data)

    df = pd.DataFrame(solutions_data)
    ds = datasets.Dataset.from_pandas(df)
    if FLAGS.per_testcases > 0:
        output_name = f"{FLAGS.dataset}_graded_{FLAGS.per_testcases}"
    else:
        output_name = f"{FLAGS.dataset}_graded"
    
    ds.push_to_hub(output_name)
    
if __name__ == "__main__":
    app.run(main)
