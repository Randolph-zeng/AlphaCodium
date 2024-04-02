import functools
import logging

from alpha_codium.gen.stages.indirect.run_validate_ai_test import run_validate_ai_tests
from alpha_codium.gen.utils import load_yaml
from alpha_codium.settings.config_loader import get_settings
from alpha_codium.llm.ai_invoker import send_inference
from alpha_codium.log import get_logger
from alpha_codium.gen.stages.run_tests import run_tests
import random
logger = get_logger(__name__)


async def run_generate_ai_tests(self, problem):
    counter_retry = 0
    while True:
        try:
            logger.info("--generate ai tests stage--")
            # ZZ: select a ground truth solution 
            problem['sampled_ground_truth_solution'] = random.choice(problem['ground_truth_python_solutions'])   

            # get settings
            validate_ai_tests = get_settings().get('generate_ai_tests.validate_ai_tests', False)
            problem['number_of_ai_tests'] = get_settings().get("generate_ai_tests.number_of_ai_tests", 8)
            problem['use_ground_truth_solution'] = get_settings().get('generate_ai_tests.use_ground_truth_solution')

            # get prompt
            f = functools.partial(self._run, problem=problem, prompt="code_contests_prompts_generate_ai_tests")

            # inference
            response_problem_tests, _ = await send_inference(f)
            problem['generated_tests'] = load_yaml(response_problem_tests,
                                                    keys_fix_yaml=["input:", "output:", "explanation:"])['tests']

            if validate_ai_tests:            
                logging.info(f"evaluating AI generated tests.")
                # ZZ: iterate through all the ai generated tests and make sure gt solution passed the test
                filtered_ai_tests = []
                for ai_test in problem['generated_tests']:
                    test_passed, sol_output, error_str, trace_str, tests_timeout \
                        = run_tests(problem['name'], problem['sampled_ground_truth_solution'], [ai_test['input']], [ai_test['output']])
                    if test_passed:
                        filtered_ai_tests.append(ai_test)
                if len(filtered_ai_tests) == 0:
                    raise Exception("None of the generated AI test passed the ground truth solution")
                problem['generated_tests'] = filtered_ai_tests
            return problem
        except Exception as e:
            logging.error(f"'generate ai tests' stage, counter_retry {counter_retry}, Error: {e}")
            counter_retry += 1
            if counter_retry > 2:
                raise e
