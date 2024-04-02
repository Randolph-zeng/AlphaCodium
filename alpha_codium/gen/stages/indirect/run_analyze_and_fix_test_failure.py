import ast
import difflib
import functools
import logging
import yaml
from alpha_codium.llm.ai_invoker import send_inference
from alpha_codium.log import get_logger
from alpha_codium.settings.config_loader import get_settings

logger = get_logger(__name__)


async def run_analyze_and_fix_test_failure(self, problem, iter_num):
    counter_retry = 0
    while True:
        try:
            # format the unit tests and the error strings
            curr_iter_info = problem[f'iteration_info_{iter_num}']
            problem['code_recent_solution'] = curr_iter_info['solution']
            unit_tests_and_errors = '' 
            for failure_dict in curr_iter_info['failed_tests']:
                unit_tests_and_errors += f"Unit-Test Input: {failure_dict['inputs']}\n"
                unit_tests_and_errors += f"Unit-Test Expected Output: {failure_dict['expected_outputs']}\n"
                # unit_tests_and_errors += f"Solution Output: {failure_dict['solution_outputs']}\n"
                unit_tests_and_errors += f"Error Message: {failure_dict['error_str']}\n"
                unit_tests_and_errors += f"Trace Message: {failure_dict['trace_str']}\n\n"
            problem['unit_tests_and_errors'] = unit_tests_and_errors
            f = functools.partial(self._run, problem=problem, prompt=choose_prompt())
            response_analyze_failure, _ = await send_inference(f)

            response_analyze_failure = response_analyze_failure.rstrip("'` \n") # remove trailing spaces and newlines from yaml response
            if response_analyze_failure.startswith("```yaml"):
                response_analyze_failure = response_analyze_failure[8:]
            response_analyze_failure_yaml = yaml.safe_load(response_analyze_failure)
            problem['response_analyze_failure'] = response_analyze_failure
            code_solution = response_analyze_failure_yaml['fixed_code'].rstrip("'` \n")

            # some cleaning
            if code_solution.startswith("```python"):
                code_solution = code_solution[10:]
            elif code_solution.startswith("python"):
                code_solution = code_solution[6:]
            try:
                ast.parse(code_solution)
            except:
                code_solution_fallback = '\n'.join(code_solution.splitlines()[:-1]).rstrip("'` \n")
                try:
                    ast.parse(code_solution_fallback)
                    code_solution = code_solution_fallback
                except:
                    logger.error(f"Invalid code:\n{code_solution}")
                    return problem
            problem[f'code_solution_iter_{iter_num+1}'] = code_solution
            return problem
        except Exception as e:
            logging.error(f"'analyze_and_fix_test_failure' stage, counter_retry {counter_retry}, Error: {e}")
            counter_retry += 1
            if counter_retry > 2:
                raise e

def choose_prompt():
    return "code_contests_prompt_analyze_and_fix"
