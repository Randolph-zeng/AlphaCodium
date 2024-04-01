import asyncio, json
import logging
import os
from jinja2 import Environment, StrictUndefined

from alpha_codium.code_contests.data.provider import CodeContestDataProvider
from alpha_codium.gen.stages.run_baseline import run_baseline
from alpha_codium.gen.stages.run_choose_best_solution import run_choose_best_solution
from alpha_codium.gen.stages.run_evaluate_all_tests import run_evaluate_all_tests
from alpha_codium.gen.stages.run_generate_ai_test import run_generate_ai_tests
from alpha_codium.gen.stages.run_generate_possible_solutions import run_generate_possible_solutions
from alpha_codium.gen.stages.run_self_reflect import run_self_reflect
from alpha_codium.gen.stages.run_initial_code_generation import run_initial_code_generation
from alpha_codium.gen.stages.utils import set_configurations
from alpha_codium.gen.utils import evaluate_solution_on_subset
from alpha_codium.llm.ai_handler import AiHandler
from alpha_codium.log import get_logger
from alpha_codium.settings.config_loader import get_settings


class CodeContestsCompetitor:
    def __init__(self):
        self.prompt = {}
        for set in get_settings():
            if 'code_contests_prompt' in set.lower():
                self.prompt[set.lower()] = get_settings()[set]
        self.ai_handler = AiHandler()

    def render(self, problem_json, prompt: str):
        environment = Environment(undefined=StrictUndefined)
        environment.globals["zip"] = zip
        environment.globals["enumerate"] = enumerate
        sys_prompt = environment.from_string(self.prompt[prompt].system).render(problem_json)
        usr_prompt = environment.from_string(self.prompt[prompt].user).render(problem_json)
        if hasattr(self.prompt[prompt], 'temperature'):
            temperature = self.prompt[prompt].temperature
        else:
            temperature = 0.2
        if hasattr(self.prompt[prompt], 'frequency_penalty'):
            frequency_penalty = self.prompt[prompt].frequency_penalty
        else:
            frequency_penalty = None
        if hasattr(self.prompt[prompt], 'top_p'):
            top_p = self.prompt[prompt].top_p
        else:
            top_p = 0.9
        return sys_prompt, usr_prompt, temperature, frequency_penalty, top_p

    async def _run(self, model, problem, prompt:str = "code_contests_prompt_reflect"):
        system_prompt, user_prompt, temperature, frequency_penalty, top_p = self.render(problem, prompt)

        if frequency_penalty == None:
            frequency_penalty = get_settings().get("config.frequency_penalty")
        try:
            response, finish_reason = await self.ai_handler.chat_completion(
                model=model, system=system_prompt, user=user_prompt, top_p=top_p,
                temperature=temperature, frequency_penalty=frequency_penalty,
            )
        except Exception as e:
            logging.error("Unknown error during calling of ai handler: ", e)
            raise e
        
        return response, finish_reason

    async def run(self, problem, iteration=0, logger_ext=None):
        if logger_ext:
            logger = logger_ext
        else:
            logger = get_logger(__name__)
        logger.info(f"Running code contests competitor, model {get_settings().config['model']}")

        try:
            if get_settings().get("solve.use_baseline", False):
                problem['code_recent_solution'] = await run_baseline(self, problem)
            else:
                # configurations
                problem = set_configurations(problem, iteration)

                # self-reflect
                problem = await run_self_reflect(self, problem)

                # generate solutions
                problem = await run_generate_possible_solutions(self, problem)

                # choose best solution
                problem = await run_choose_best_solution(self, problem)

                # generate ai tests
                problem = await run_generate_ai_tests(self, problem)

                # initial code generation
                problem = await run_initial_solve(self, problem)

                # evaluate on all tests available
                problem = await run_evaluate_all_tests(self, problem)

            return problem
        except Exception as e:
            logging.error(f"Error: {e}")
            return ""

    def solve_problem_in_dataset(self, loop, example, iteration=0, logger_ext=None):
        problem = {k: example.get(k) for k in ["name", "description", 'public_tests']}
        # filter out ground truth python solutions to be used in AI tests generation  
        filtered_sols = []
        for lang, sol in zip(example['solutions']['language'], example['solutions']['solution']):
            if (lang.lower() == 'python3'):
                filtered_sols.append(sol)
        problem['ground_truth_python_solutions'] = filtered_sols 
        revised_problem = loop.run_until_complete(self.run(problem=problem, iteration=iteration, logger_ext=logger_ext))
        return revised_problem['code_recent_solution'], revised_problem


def solve_problem(dataset_name,
                  split_name="valid",
                  problem_name="",
                  problem_number=0):

    # load dataset
    logger = get_logger(__name__)
    data_provider = CodeContestDataProvider(dataset_location=dataset_name)
    if problem_number and problem_name:
        logger.info(f"problem_number and problem_name are both specified, using problem_name")
    if not problem_name and problem_number:
        problem_name = data_provider.dataset[split_name][int(problem_number)]['name']
        logger.info(f"problem_name: {problem_name}")

    # find problem
    problem = data_provider.find_problem(ds=data_provider.dataset, problem_name=problem_name, split_name=split_name)
    logger.info(f"problem['name']: {problem['name']}")

    # # check if problem is valid (at least one of the provided solutions actually passes the generated tests)
    # if not problem.get('is_valid_problem', True):
    #     logger.info(f"problem['is_valid_problem'] == False, skipping")
    #     return None, None

    # evaluate prev solutions
    evaluate_prev_solutions = get_settings().get("dataset.evaluate_prev_solutions", False)
    if evaluate_prev_solutions:
        try:
            if not problem['solutions']['solution']:
                logger.info("No public solutions for this problem")
            found_solution = False
            for index_published, sol_published in enumerate(problem['solutions']['solution']):
                if 'python' not in problem['solutions']['language'][index_published].lower():
                    found_solution = True
                    continue
                logger.info(f"evaluating public solution {index_published} on private tests...")
                test_results, test_passed_private, test_failed_private, test_timeout_private \
                    = evaluate_solution_on_subset('private_tests', problem, sol_published, silent=True)
                logger.info(f"evaluating public solution {index_published} on generated tests...")
                test_results, test_passed_generate, test_failed_generate, test_timeout_generate = (
                    evaluate_solution_on_subset('generated_tests', problem, sol_published, silent=True))

                if (test_failed_private == test_failed_generate == test_timeout_private == test_timeout_generate == 0) \
                        and test_passed_private + test_passed_generate > 0:
                    logger.info(f"sol_published index {index_published} passed all tests:\n{sol_published}")
                    found_solution = True
                    break

            if not found_solution:
                logger.info(f"None of the public solutions passed all tests")
        except Exception as e:
            logger.error(f"Error evaluating public solutions: {e}")
            pass


    return solve_my_problem(problem)


def solve_my_problem(problem):

    base_path = os.getcwd()
    logger = get_logger(__name__)

    solver = CodeContestsCompetitor()
    os.chdir(base_path)
    # ZZ:create an overall loop without calling async.run everytime
    loop = asyncio.get_event_loop()
    solution, revised_problem = solver.solve_problem_in_dataset(loop, problem)
    logger.info(f"testing solution on private tests with prediction:\n{solution}")

    logger.info(f"evaluating solution on public tests...")
    sol_eval_timeout = get_settings().get('config.sol_eval_timeout')
    test_results, test_passed_public, test_failed_public, test_timeout_public = evaluate_solution_on_subset('public_tests',
                                                                                                       problem,
                                                                                                       solution,
                                                                                                       silent=True,
                                                                                                       timeout=sol_eval_timeout)


    logger.info(f"evaluating solution on private tests...")
    test_results, test_passed_private, test_failed_private, test_timeout_private = evaluate_solution_on_subset('private_tests',
                                                                                                       problem,
                                                                                                       solution,
                                                                                                       silent=True,
                                                                                                       timeout=sol_eval_timeout)

    logger.info(f"evaluating solution on generated tests...")
    test_results, test_passed_generate, test_failed_generate, test_timeout_generate = evaluate_solution_on_subset(
        'generated_tests', problem, solution, silent=True, timeout=sol_eval_timeout)

    logger.info(f"\ntest_passed_generate: {test_passed_generate}, test_passed_private: {test_passed_private}, test_passed_public: {test_passed_public}"
                f"\ntest_failed_generate: {test_failed_generate}, test_failed_private: {test_failed_private}, test_failed_public: {test_failed_public}"
                f"\ntest_timeout_generate: {test_timeout_generate}, test_timeout_private: {test_timeout_private}, test_timeout_public: {test_timeout_public}")
    with open(f'/home/appuser/AlphaCodium/code_contest_data/generated_problem_data/{revised_problem["name"]}.json', 'w') as file:
        json.dump(revised_problem, file, indent=2, ensure_ascii=False)
    
    return solution, test_results
