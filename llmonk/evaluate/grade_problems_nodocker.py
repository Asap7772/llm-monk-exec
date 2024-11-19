from tqdm import tqdm
import concurrent.futures
import re
import os
import tempfile
import math
import subprocess
from llmonk.evaluate.code_contests_utils.compare_results import outputs_match
from llmonk.evaluate.code_contests_utils.schema import ExecuteCodeRequest, ExecuteCodeResult

MAX_CONCURRENT_REQUESTS = os.cpu_count() * 2
# semaphore = threading.Semaphore(value=MAX_CONCURRENT_REQUESTS)
NUM_RETRIES = 1
RETRY_BACKOFF = 3


def execute_with_input(
    code_file_name: str, input_str: str, timeout: float, memory_limit_bytes: int
):

    try:
        with tempfile.NamedTemporaryFile(mode="w+", delete=True) as temp_input_file:
            temp_input_file.write(input_str)
            temp_input_file.flush()  # Ensure the data is written to disk

            # Rewind the file so the subprocess reads it from the beginning
            temp_input_file.seek(0)

            memory_limit_kb = math.ceil(memory_limit_bytes / 1024)
            cmd = f"ulimit -v {memory_limit_kb} && python {code_file_name}"

            try:
                result = subprocess.run(
                    ["bash", "-c", cmd],
                    capture_output=True,
                    stdin=temp_input_file.fileno(),
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                print(f"{cmd} {temp_input_file.name} timed out. Timeout was {timeout}")
                return None
            
            return result.stdout.decode()
    except Exception as e:
        print(f"Failed to execute with error {e}")
        return None

def execute_python_code(request: ExecuteCodeRequest):
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=True) as temp:
            
            with open(temp.name, "w") as f:
                f.write(request.code)

            for input_str, expected_output in request.input_expected_output_pairs:
                # Intentionally serial - most programs fail.
                actual_output = execute_with_input(
                    code_file_name=temp.name,
                    input_str=input_str,
                    timeout=request.timeout,
                    memory_limit_bytes=request.memory_limit_bytes,
                )

                if actual_output is None or not outputs_match(expected_output, actual_output):
                    return ExecuteCodeResult(correct=False)

            return ExecuteCodeResult(correct=True)
    except Exception as e:
        print(f"Failed to execute code: \n {e}")
        return ExecuteCodeResult(correct=False)

def is_valid_python(snippet):
    try:
        compile(snippet, "<string>", "exec")
        return True
    except SyntaxError:
        return False
           
def extract_first_code(output_string: str):
    trimmed = output_string.strip()

    # Extracting the first occurrence of content between backticks
    code_match = re.search(r"```(.*?)```", trimmed, re.DOTALL)

    if code_match:
        # Strip leading and trailing whitespace from the extracted code
        code = code_match.group(1).strip()

        # sometimes the block of code is ```python ... ``` instead of ``` ... ```
        # in this case strip the python out

        if code.startswith("python"):
            code = code[len("python") :].strip()

        return code

    if is_valid_python(trimmed):
        return trimmed

    return None

def solution_is_correct(
    code: str | None,
    problem: dict,
):
    if code is None:
        return False

    assert len(problem["test_cases"]["input"]) == len(problem["test_cases"]["output"])

    input_expected_output_pairs = list(
        zip(problem["test_cases"]["input"], problem["test_cases"]["output"])
    )
    is_correct = False
    code = extract_first_code(code)
    if code is not None:
        is_correct = execute_python_code(
            ExecuteCodeRequest(
                code=code,
                input_expected_output_pairs=input_expected_output_pairs,
                timeout=problem["timeout"] + 10,  # buffer for 10
                memory_limit_bytes=2_000_000_000_000,  # double max limit
        )).correct
    return is_correct

def grade_problems(solutions_data: dict):
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_CONCURRENT_REQUESTS
    ) as executor:
        future_to_index = {}
        futures = []
        for idx, solution_data in tqdm(enumerate(solutions_data), total=len(solutions_data), desc="Queued for execution: "):
            for code in solution_data["solutions"]:
                future = executor.submit(
                            solution_is_correct,
                            code=code,
                            problem=solution_data,
                        )
                future_to_index[future] = idx

        is_corrects_list = [[]] * len(solutions_data) 
        for future in tqdm(concurrent.futures.as_completed(future_to_index), total=len(future_to_index), desc="Running tests on problem"):
            idx = future_to_index[future]
            is_corrects_list[idx].append(future.result(timeout=10))
        
        for idx, is_corrects in enumerate(is_corrects_list):
            solutions_data[idx]["is_corrects"] = is_corrects