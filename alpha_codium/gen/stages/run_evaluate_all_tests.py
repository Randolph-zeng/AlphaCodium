import copy
import logging

from alpha_codium.gen.stages.indirect.run_analyze_and_fix_test_failure import run_analyze_and_fix_test_failure
from alpha_codium.settings.config_loader import get_settings
from alpha_codium.gen.stages.run_tests import run_tests
from alpha_codium.log import get_logger

logger = get_logger(__name__)


async def run_evaluate_all_tests(self, problem):
    counter_retry = 0
    while True:
        try:
            logger.info("--iterate on ALL tests stage--")
            # configurations
            problem['use_self_reflection_public'] = get_settings().get('all_tests.use_self_reflection', False)
            max_allowed_fixes = get_settings().get("all_tests.max_allowed_calls", 6)
            for fix_iter in range(max_allowed_fixes):
                # loop through all the tests and run fix against error messages
                # TODO add ai tests here
                test_inputs_all = problem['public_tests']['input'] + [obj['input'] for obj in problem['generated_tests']]
                test_outputs_all = problem['public_tests']['output'] + [obj['output'] for obj in problem['generated_tests']]
                test_explanations_all = problem['tests_explanations'] + problem['generated_tests']
                all_passed_public = True
                passed_tests, failed_tests = [], []
                for test_inputs, test_outputs, test_explanation in zip(test_inputs_all, test_outputs_all,
                                                                        test_explanations_all):
                    if not isinstance(test_inputs, list):
                        test_inputs = [test_inputs]
                        test_outputs = [test_outputs]
                    # run the code on the test
                    passed_specific_test, sol_output, error_str, trace_str, tests_timeout \
                        = run_tests(problem['name'], problem[f'code_solution_iter_{fix_iter}'], 
                                    test_inputs, test_outputs)
                    if passed_specific_test:
                        passed_tests.append({
                            "inputs": test_inputs,
                            "outputs": test_outputs
                        })
                    else:
                        all_passed_public = False
                        failed_tests.append({
                            "inputs": test_inputs,
                            "expected_outputs": test_outputs,
                            "solution_outputs": sol_output,
                            "error_str": error_str,
                            "trace_str": trace_str,
                            "tests_timeout": tests_timeout,
                            "test_explanation": test_explanation
                        })
                curr_iter_info = {
                    'solution': problem[f'code_solution_iter_{fix_iter}'],
                    'passed_tests': passed_tests,
                    'failed_tests': failed_tests
                }
                problem[f'iteration_info_{fix_iter}'] = curr_iter_info
                problem['code_final_solution'] = problem[f'code_solution_iter_{fix_iter}']
                if all_passed_public:
                    logger.info(f"==================")
                    logger.info(f"Passed all public tests")
                    logger.info(f"==================")
                    break
                else:
                    logger.info(f"==================")
                    logger.info(f"Failed to pass all public tests")
                    logger.info(f"==================")
                    if fix_iter < max_allowed_fixes -1:
                        problem = await run_analyze_and_fix_test_failure(self, problem, fix_iter)

            return problem
        except Exception as e:
            logging.error(f"'All tests' stage, counter_retry {counter_retry}, Error: {e}")
            counter_retry += 1
            if counter_retry > 2:
                raise e
