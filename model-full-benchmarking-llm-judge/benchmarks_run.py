#!/usr/bin/env python3
"""
Advanced Unified LLM Benchmarking Tool

This script combines:
1. Latency benchmarking (from latency-benchmarking-tool.ipynb)
2. LLM-as-judge evaluation
3. Cost analysis

It produces a comprehensive PDF report with visualizations and recommendations.
"""

import argparse
import boto3
import concurrent.futures
import datetime
import json
import logging
import pprint
import pytz
import random
import re
import time
from botocore.config import Config
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from visualize_resultsv2 import *

# Initialize logging
def setup_logging():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        filename=f"logs/advanced-benchmark-{timestamp}.log",
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return timestamp

# Create a function to get a new boto3 client
def get_bedrock_client(region):
    config = Config(
        retries = dict(
            max_attempts = 5
        )
    )
    return boto3.client(
        service_name='bedrock-runtime',
        region_name=region,
        config=config
    )

def get_timestamp():
    dt = datetime.fromtimestamp(time.time(), tz=pytz.utc)
    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')


# Judge model configuration
DEFAULT_JUDGE_MODEL = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
DEFAULT_JUDGE_REGION = "us-west-2"


#----------------------------------------
# PERFORMANCE BENCHMARK FUNCTIONS
#----------------------------------------
def get_body(prompt, max_tokens, task_types, task_criteria, temperature, top_p):
    """Prepare the request body with system instructions specific to task types"""
    # Create system prompt based on task types
    system_prompt = ""
    # Construct message for different model types
    body = [
        {
            'role': 'user',
            'content': [
                {
                    'text': f"{system_prompt}\n##USER:{prompt}"
                },
            ]
        },
    ]

    inferenceConfig = {
        'maxTokens': max_tokens,
        'temperature': temperature,
        'topP': top_p
    }

    return body, inferenceConfig

def post_iteration(scenario_config):
    """Sleep between API calls to avoid throttling"""
    logging.info(f'Sleeping for {scenario_config["sleep_between_invocations"]} seconds.')
    time.sleep(scenario_config["sleep_between_invocations"])


#----------------------------------------
# LLM-AS-JUDGE EVALUATION
#----------------------------------------

def get_judge_body(prompt, model_response, golden_answer, task_types, task_criteria):
    """Prepare the request body for LLM-as-judge evaluation"""
    # Construct judge prompt
    judge_prompt = f"""You are an expert evaluator of AI assistant responses. Your task is to determine if a model's response successfully completes the requested task.

TASK TYPE: {task_types}

EVALUATION CRITERIA:
- {task_criteria}\n

General criteria:
- Correctness: Information must be factually accurate
- Completeness: All parts of the task must be addressed
- Relevance: Response must be on-topic and address the prompt
- Format: Response should follow any formatting requirements

ORIGINAL PROMPT:
{prompt}

MODEL RESPONSE:
{model_response}

GOLDEN ANSWER (Reference):
{golden_answer}

INSTRUCTIONS:
1. Carefully compare the model response to the golden answer
2. Determine if the response successfully meets all requested tasks
3. Provide your judgment as "PASS" or "FAIL" do not start with nothing else
4. If the judgment is a "FAIL" include the reason choose from ['Correctness', 'Completeness', 'Relevance', 'Format'] nothing else
5. If the judgment is a "PASS" say "Model output meets golden answer criteria" nothing else
6. Even if the model response differs from the golden answer, it can PASS if it correctly fulfills the required tasks

JUDGMENT:
"""
    
    # Construct message body
    body = [
        {
            'role': 'user',
            'content': [
                {
                'text': judge_prompt
                },
            ]
        },
    ]
    
    inference_config = {
        'maxTokens': 500,
        'temperature': 0.1,  # Low temperature for consistent judgments
        'topP': 0.9
    }
    
    return body, inference_config

def evaluate_with_llm_judge(bedrock, judge_model_id, judge_region, prompt, model_response, golden_answer, task_types, task_criteria):
    """Use an LLM to judge the quality of the model response"""
    
    # Get a client for the judge model
    judge_client = get_bedrock_client(judge_region)
    
    # Prepare judge request
    body, inference_config = get_judge_body(prompt, model_response, golden_answer, task_types, task_criteria)

    try:
        # Make the API call to the judge model
        response = judge_client.converse(
            messages=body,
            modelId=judge_model_id,
            inferenceConfig=inference_config
        )

        # Extract the judge's response
        judge_response = response['output']['message']['content'][0]['text']

        # Parse the judgment (PASS/FAIL)
        judgment = "FAIL"  # Default to fail
        if "PASS" in judge_response.upper():
            judgment = "PASS"

        # Extract the explanation (everything after PASS/FAIL)
        explanation = judge_response.strip()
        if "PASS" in judge_response.upper():
            explanation = re.sub(r'^PASS\s*[:\-]?\s*', '', explanation, flags=re.IGNORECASE)
        elif "FAIL" in judge_response.upper():
            # explanation = re.sub(r'^FAIL\s*[:\-]?\s*', '', explanation, flags=re.IGNORECASE)
            target_reasons = ['CORRECTNESS', 'COMPLETENESS', 'RELEVANCE', 'FORMAT']
            found_words = []
            words = re.findall(r'\b\w+\b', judge_response.upper())
            for word in words:
                if word in target_reasons:
                    found_words.append(word)
            found_words = list(set(found_words))
            explanation = ','.join(found_words)


        return {
            "judgment": judgment,
            "explanation": explanation,
            "full_response": judge_response
        }

    except Exception as e:
        logging.error(f"Error in LLM judge evaluation: {e}")
        return {
            "judgment": "ERROR",
            "explanation": f"Error in judge evaluation: {str(e)}",
            "full_response": ""
        }

def benchmark(bedrock, prompt, task_types, task_criteria, golden_answer, latency_inference_profile,
              max_tokens, model_id="", input_token_cost=0, output_token_cost=0,
              temperature=1, top_p=1, use_llm_judge=False,
              judge_model_id=DEFAULT_JUDGE_MODEL, judge_region=DEFAULT_JUDGE_REGION,
              stream=True, sleep_on_throttling=5):
    """
    Run full benchmark with both latency and performance measurement in a single LLM call
    """
    api_call_status = 'Success'
    full_error_message = 'Success'
    dt = datetime.fromtimestamp(time.time(), tz=pytz.utc)
    job_timestamp_iso = dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    performance_metrics = {}
    judge_result = {}
    cost = 0.0
    time_to_first_byte = None
    time_to_last_byte = None
    model_output_tokens = None
    model_input_tokens = None
    model_response = ""

    # Prepare the request body
    body, inference_config = get_body(prompt, max_tokens, task_types, task_criteria, temperature, top_p)

    try:
        # Make a single API call that captures both latency and response data
        start = time.time()
        response = bedrock.converse_stream(
            messages=body,
            modelId=model_id,
            inferenceConfig=inference_config,
            performanceConfig={
                'latency': latency_inference_profile
            }
        )

        first_byte = None
        event_stream = response.get('stream')
        for event in event_stream:
            if 'contentBlockDelta' in event:
                chunk = event['contentBlockDelta']
                if 'delta' in chunk:
                    if 'text' in chunk['delta']:
                        text_chunk = chunk['delta']['text']
                        model_response += text_chunk
                        # Capture time to first byte on first content chunk
                        if not first_byte:
                            first_byte = time.time()
            elif 'messageStop' in event:
                stop_reason = event['messageStop'].get('stopReason', 'Unknown')
            elif 'metadata' in event:
                metadata = event['metadata']
                if 'usage' in metadata:
                    model_output_tokens = metadata['usage'].get('outputTokens', None)
                    model_input_tokens = metadata['usage'].get('inputTokens', None)

                    # Calculate cost
                    if model_input_tokens is not None and model_output_tokens is not None:
                        cost = (model_input_tokens * input_token_cost) + (model_output_tokens * output_token_cost)

        # Calculate duration metrics
        last_byte = time.time()
        if first_byte:
            time_to_first_byte = round(first_byte - start, 2)
            time_to_last_byte = round(last_byte - start, 2)

        # Evaluate with LLM judge if requested
        if use_llm_judge and model_response:
            judge_result = evaluate_with_llm_judge(
                bedrock, judge_model_id, judge_region,
                prompt, model_response, golden_answer, task_types, task_criteria
            )

            # Add judge result to performance metrics
            performance_metrics['judge_success'] = judge_result['judgment'] == 'PASS'
            performance_metrics['judge_explanation'] = judge_result.get('explanation', '')

    except ClientError as err:
        full_error_message = err
        api_call_status = err.response['Error']['Code']
        print(f"Got Error: {api_call_status}")
        print(f"Full Error Message: {full_error_message}")

    return (time_to_first_byte, time_to_last_byte, job_timestamp_iso,
            api_call_status, full_error_message, model_output_tokens, model_input_tokens,
            model_response, performance_metrics, judge_result, cost)


def execute_benchmark(client, scenarios, scenario_config, num_parallel_calls=4, early_break=False, logging_lock=None):
    """Execute benchmark scenarios in parallel"""
    pp = pprint.PrettyPrinter(indent=2)
    all_invocations = []
    
    def process_scenario(scenario):
        local_client = get_bedrock_client(scenario.get('region', 'us-east-1'))
        local_invocations = []
        prompt = scenario.get('prompt', '')
        
        for invocation_id in range(scenario_config["invocations_per_scenario"]):
            try:
                (time_to_first_byte, time_to_last_byte, job_timestamp_iso, api_call_status,
                 full_error_message, model_output_tokens, model_input_tokens,
                 model_response, performance_metrics, judge_result, cost) = benchmark(
                    local_client,
                    prompt,
                    scenario.get('task_types', []),
                    scenario.get('task_criteria', []),
                    scenario.get('golden_answer', ''),
                    latency_inference_profile=scenario.get('latency_inference_profile', 'standard'),
                    max_tokens=scenario.get('configured_output_tokens_for_request', 500),
                    model_id=scenario.get('model_id', ''),
                    input_token_cost=scenario.get('input_token_cost', 0),
                    output_token_cost=scenario.get('output_token_cost', 0),
                    temperature=scenario_config.get('TEMPERATURE', 1),
                    top_p=scenario_config.get('TOP_P', 1),
                    # top_k=scenario_config.get('TOP_K', 250),
                    use_llm_judge=scenario_config.get('use_llm_judge', False),
                    judge_model_id=scenario_config.get('judge_model_id', DEFAULT_JUDGE_MODEL),
                    judge_region=scenario_config.get('judge_region', DEFAULT_JUDGE_REGION),
                    stream=True,
                    sleep_on_throttling=scenario_config.get('sleep_between_invocations', 5)
                )

                invocation = {
                    'time_to_first_byte': time_to_first_byte,
                    'time_to_last_byte': time_to_last_byte,
                    'job_timestamp_iso': job_timestamp_iso,
                    'configured_output_tokens_for_request': scenario.get('configured_output_tokens_for_request', 100),
                    'model_input_tokens': model_input_tokens,
                    'model_output_tokens': model_output_tokens,
                    'model': scenario.get('model_id', ''),
                    'region': scenario.get('region', ''),
                    'invocation_id': invocation_id,
                    'api_call_status': api_call_status,
                    'full_error_message': full_error_message,
                    'TEMPERATURE': scenario_config.get('TEMPERATURE', 1),
                    'TOP_P': scenario_config.get('TOP_P', 1),
                    # 'TOP_K': scenario_config.get('TOP_K', 250),
                    'EXPERIMENT_NAME': scenario_config.get('EXPERIMENT_NAME', 'Unnamed Experiment'),
                    'task_types': scenario.get('task_types', 'none'),
                    # 'detected_tasks': ','.join(detected_tasks),
                    # 'task_success': task_success,
                    'judge_success': performance_metrics.get('judge_success', None),
                    # 'combined_success': combined_success,
                    'success_rate': performance_metrics.get('success_rate', 0),
                    'model_response': model_response,
                    'golden_answer': scenario.get('golden_answer', ''),
                    'inference_profile': scenario.get('latency_inference_profile', 'standard'),
                    'input_token_cost': scenario.get('input_token_cost', 0),
                    'output_token_cost': scenario.get('output_token_cost', 0),
                    'response_cost': cost,
                    'judge_explanation': performance_metrics.get('judge_explanation', '')
                }
                local_invocations.append(invocation)
                
                # Thread-safe logging
                if logging_lock:
                    with logging_lock:
                        logging.info(f'Invocation: {invocation["model"]} - Success: {performance_metrics.get('judge_success', None)}')
                else:
                    logging.info(f'Invocation: {invocation["model"]} - Success: {performance_metrics.get('judge_success', None)}')
                
                post_iteration(scenario_config=scenario_config)
                
            except Exception as e:
                if logging_lock:
                    with logging_lock:
                        logging.error(f"Error while processing scenario: {scenario.get('model_id', 'unknown')}. Error: {e}")
                else:
                    logging.error(f"Error while processing scenario: {scenario.get('model_id', 'unknown')}. Error: {e}")
                
        return local_invocations

    # Execute scenarios in parallel
    with ThreadPoolExecutor(max_workers=num_parallel_calls) as executor:
        # Submit all scenarios and store futures
        future_to_scenario = {executor.submit(process_scenario, scenario): scenario 
                            for scenario in scenarios}
        
        # Print initial state
        print(f"Total scenarios submitted: {len(future_to_scenario)}")
        print(f"Number of parallel workers: {num_parallel_calls}")
        
        # Monitor futures as they complete
        for future in concurrent.futures.as_completed(future_to_scenario):

            try:
                result = future.result()
                all_invocations.extend(result)
            except Exception as e:
                if logging_lock:
                    with logging_lock:
                        logging.error(f"Scenario failed: {e}")
                else:
                    logging.error(f"Scenario failed: {e}")

        return all_invocations

#----------------------------------------
# MAIN FUNCTION
#----------------------------------------
#
# def main():
#
#     # Parse command line arguments
#     parser = argparse.ArgumentParser(description='Advanced Unified LLM Benchmarking Tool')
#     parser.add_argument('--input', type=str, required=True, help='Input JSONL file with benchmark prompts')
#
#     parser.add_argument('--output-dir', type=str, default='benchmark_results', help='Directory to save results')
#     parser.add_argument('--parallel-calls', type=int, default=4, help='Number of parallel API calls')
#     parser.add_argument('--invocations-per-scenario', type=int, default=5, help='Number of times to run each prompt')
#     parser.add_argument('--sleep-between-invocations', type=int, default=60, help='Sleep time between invocations (seconds)')
#     parser.add_argument('--experiment-counts', type=int, default=1, help='How many times to run the experiment')
#     parser.add_argument('--experiment-name', type=str, default=f"Benchmark-{datetime.now().strftime('%Y%m%d')}",
#                         help='Name of the experiment')
#     parser.add_argument('--use-llm-judge', action='store_true', help='Enable LLM-as-judge evaluation')
#     parser.add_argument('--judge-model', type=str, default=DEFAULT_JUDGE_MODEL,
#                         help='Model to use for LLM-as-judge evaluation')
#     parser.add_argument('--judge-region', type=str, default=DEFAULT_JUDGE_REGION,
#                         help='AWS region for judge model')
#
#     args = parser.parse_args()
#
#     # Initialize timestamp and logging
#     timestamp = setup_logging()
#     logging_lock = Lock()
#
#     # Create output directories
#     os.makedirs(args.output_dir, exist_ok=True)
#
#     # Initialize configuration
#     scenario_config = {
#         "sleep_between_invocations": args.sleep_between_invocations,
#         "invocations_per_scenario": args.invocations_per_scenario,
#         "TEMPERATURE": 1,
#         "TOP_P": 1,
#         "TOP_K": 250,
#         "EXPERIMENT_NAME": args.experiment_name,
#         "use_llm_judge": args.use_llm_judge,
#         "judge_model_id": args.judge_model,
#         "judge_region": args.judge_region
#     }
#
#     # Download nltk resources
#     try:
#         nltk.download('punkt', quiet=True)
#         nltk.download('stopwords', quiet=True)
#     except Exception as e:
#         logging.error(f"Error downloading NLTK resources: {e}")
#         print(f"Error downloading NLTK resources: {e}")
#
#     # Load scenarios from input file
#     use_cases_scenarios = []
#
#     try:
#         with open(args.input, 'r', encoding='utf-8') as f:
#             for line in f:
#                 file = json.loads(line.strip())
#                 prompt = file.get('text_prompt')
#                 task_types = file.get('task_type')
#                 golden_answer = file.get('golden_answer')
#                 model_id = file.get('model_id')
#                 region = file.get('region')
#                 latency_inference_profile = file.get('inference_profile', 'optimized')
#                 input_token_cost = file.get('input_token_cost', 0.0)
#                 output_token_cost = file.get('output_token_cost', 0.0)
#
#                 # Ensure task_types is a list
#                 if isinstance(task_types, str):
#                     task_types = [task_types]
#
#                 # Validate task types
#                 valid_task_types = [task for task in task_types if task in SUPPORTED_TASK_TYPES]
#
#                 if not valid_task_types:
#                     logging.warning(f"Skipping scenario with unsupported task types: {task_types}")
#                     continue
#
#                 out_tokens = file.get('expected_output_tokens', 100)
#                 use_cases_scenarios.append({
#                     "prompt": prompt,
#                     "configured_output_tokens_for_request": out_tokens,
#                     "task_types": valid_task_types,
#                     "golden_answer": golden_answer,
#                     "stream": True,
#                     "model_id": model_id,
#                     "region": region,
#                     "latency_inference_profile": latency_inference_profile,
#                     "input_token_cost": input_token_cost,
#                     "output_token_cost": output_token_cost
#                 })
#     except Exception as e:
#         logging.error(f"Error loading scenarios: {e}")
#         print(f"Error loading scenarios: {e}")
#         return
#
#     if not use_cases_scenarios:
#         logging.error("No valid scenarios found in the input file.")
#         print("No valid scenarios found in the input file.")
#         return
#
#     print(f"Loaded {len(use_cases_scenarios)} scenarios from {args.input}")
#     print(f"Using LLM-as-judge evaluation: {args.use_llm_judge}")
#     if args.use_llm_judge:
#         print(f"Judge model: {args.judge_model} in {args.judge_region}")
#
#     # Main experiment loop
#     run_count = 1
#     while run_count <= args.experiment_counts:
#         selected_scenarios = random.sample(
#             use_cases_scenarios,
#             k=min(len(use_cases_scenarios), len(use_cases_scenarios))
#         )
#
#         with logging_lock:
#             logging.info(f"{len(selected_scenarios)} scenarios x {scenario_config['invocations_per_scenario']} invocations = {len(selected_scenarios) * scenario_config['invocations_per_scenario']} total invocations")
#
#         logging.info(f"Running iteration {run_count}")
#
#         try:
#             # Create a new client for the main thread
#             config = Config(
#                 retries = dict(
#                     max_attempts = 5
#                 )
#             )
#
#             # Default to first region in scenarios
#             default_region = selected_scenarios[0]['region'] if selected_scenarios else 'us-east-1'
#
#             client = boto3.client(
#                 service_name='bedrock-runtime',
#                 region_name=default_region,
#                 config=config
#             )
#
#             # Run the scenarios and measure times
#             invocations = execute_benchmark(
#                 client,
#                 selected_scenarios,
#                 scenario_config,
#                 num_parallel_calls=args.parallel_calls,
#                 early_break=False,
#                 logging_lock=logging_lock
#             )
#
#             # Convert the invocations list to a pandas DataFrame
#             df = pd.DataFrame(invocations)
#             df['timestamp'] = pd.Timestamp.now()
#             df['run_count'] = run_count
#
#             # Write the DataFrame to a CSV file
#             output_file = os.path.join(args.output_dir, f"invocations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
#             df.to_csv(output_file, index=False)
#
#             with logging_lock:
#                 logging.info(f"Results written to {output_file}")
#                 logging.info(f"Completed run {run_count} of {args.experiment_counts}")
#         except Exception as e:
#             logging.error(f"Error in run {run_count}: {e}")
#             print(f"Error in run {run_count}: {e}")
#
#         run_count += 1
#
#     # Create the final report
#     try:
#         report_file = create_report(args.output_dir, timestamp, args.use_llm_judge)
#         print(f"Benchmark complete! Final report saved to: {report_file}")
#     except Exception as e:
#         logging.error(f"Error creating report: {e}")
#         print(f"Error creating report: {e}")

# if __name__ == "__main__":
    # main()

def main(
        input_file,
        output_dir='benchmark_results',
        parallel_calls=4,
        invocations_per_scenario=5,
        sleep_between_invocations=60,
        experiment_counts=1,
        experiment_name=f"Benchmark-{datetime.now().strftime('%Y%m%d')}",
        use_llm_judge=False,
        judge_model=DEFAULT_JUDGE_MODEL,
        judge_region=DEFAULT_JUDGE_REGION
):
    # Initialize timestamp and logging
    timestamp = setup_logging()
    logging_lock = Lock()

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)

    # Initialize configuration
    scenario_config = {
        "sleep_between_invocations": sleep_between_invocations,
        "invocations_per_scenario": invocations_per_scenario,
        "TEMPERATURE": 1,
        "TOP_P": 1,
        "EXPERIMENT_NAME": experiment_name,
        "use_llm_judge": use_llm_judge,
        "judge_model_id": judge_model,
        "judge_region": judge_region
    }

    # Load scenarios from input file
    use_cases_scenarios = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                file = json.loads(line.strip())
                prompt = file.get('text_prompt')
                task_types = file.get('task')
                golden_answer = file.get('golden_answer')
                model_id = file.get('model_id')
                region = file.get('region')
                latency_inference_profile = file.get('inference_profile', 'optimized')
                input_token_cost = file.get('input_token_cost', 0.0)
                output_token_cost = file.get('output_token_cost', 0.0)

                task_type = task_types['task_type']
                task_criteria = task_types['task_criteria']

                out_tokens = file.get('expected_output_tokens', 100)
                use_cases_scenarios.append({
                    "prompt": prompt,
                    "configured_output_tokens_for_request": out_tokens,
                    "task_types": task_type,
                    "task_criteria": task_criteria,
                    "golden_answer": golden_answer,
                    "stream": True,
                    "model_id": model_id,
                    "region": region,
                    "latency_inference_profile": latency_inference_profile,
                    "input_token_cost": input_token_cost,
                    "output_token_cost": output_token_cost
                })
    except Exception as e:
        logging.error(f"Error loading scenarios: {e}")
        print(f"Error loading scenarios: {e}")
        return

    if not use_cases_scenarios:
        logging.error("No valid scenarios found in the input file.")
        print("No valid scenarios found in the input file.")
        return

    print(f"Loaded {len(use_cases_scenarios)} scenarios from {input_file}")
    print(f"Using LLM-as-judge evaluation: {use_llm_judge}")
    if use_llm_judge:
        print(f"Judge model: {judge_model} in {judge_region}")

    # Main experiment loop
    run_count = 1
    while run_count <= experiment_counts:
        selected_scenarios = random.sample(
            use_cases_scenarios,
            k=min(len(use_cases_scenarios), len(use_cases_scenarios))
        )

        with logging_lock:
            logging.info(
                f"{len(selected_scenarios)} scenarios x {scenario_config['invocations_per_scenario']} invocations = {len(selected_scenarios) * scenario_config['invocations_per_scenario']} total invocations")

        logging.info(f"Running iteration {run_count}")

        try:
            # Create a new client for the main thread
            config = Config(
                retries=dict(
                    max_attempts=5
                )
            )

            # Default to first region in scenarios
            default_region = selected_scenarios[0]['region'] if selected_scenarios else 'us-east-1'

            client = boto3.client(
                service_name='bedrock-runtime',
                region_name=default_region,
                config=config
            )

            # Run the scenarios and measure times
            invocations = execute_benchmark(
                client,
                selected_scenarios,
                scenario_config,
                num_parallel_calls=parallel_calls,
                early_break=False,
                logging_lock=logging_lock
            )

            # Convert the invocations list to a pandas DataFrame
            df = pd.DataFrame(invocations)
            df['timestamp'] = pd.Timestamp.now()
            df['run_count'] = run_count

            # Write the DataFrame to a CSV file
            output_file = os.path.join(output_dir, f"invocations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df.to_csv(output_file, index=False)

            with logging_lock:
                logging.info(f"Results written to {output_file}")
                logging.info(f"Completed run {run_count} of {experiment_counts}")
        except Exception as e:
            logging.error(f"Error in run {run_count}: {e}")
            print(f"Error in run {run_count}: {e}")

        run_count += 1

    # Create the final report
    try:
        report_file = create_report(output_dir, timestamp, use_llm_judge)
        print(f"Benchmark complete! Final report saved to: {report_file}")
    except Exception as e:
        logging.error(f"Error creating report: {e}")
        print(f"Error creating report: {e}")


if __name__ == "__main__":
    # main(
    #     input_file="prompt-evaluations/full-benchmark-prompts-mini.jsonl",
    #     output_dir="results",
    #     parallel_calls=2,
    #     invocations_per_scenario=1,
    #     use_llm_judge=True
    # )
    report_file = create_report('/Users/claumazz/PycharmProjects/amazon-bedrock-samples/model-full-benchmarking-llm-judge/results/',
                                datetime.now().strftime('%Y%m%d_%H%M%S'),
                                True)

