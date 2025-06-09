import json

def generate_model_info(file_path='/Users/claumazz/Projects/amazon-bedrock-samples/poc-to-prod/360-eval/config/models_profiles.jsonl'):
    try:
        # Initialize empty structures
        bedrock_models = []
        openai_models = []
        cost_map = {}
        print(file_path)
        # Read and process the JSONL file
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    data = json.loads(line)
                    model_id = data['model_id']

                    # Categorize models based on prefix
                    if model_id.startswith('openai/'):
                        openai_models.append(model_id)
                    else:
                        bedrock_models.append(model_id)

                    # Build cost map entry
                    # Handle the case where 'input_token_cost' might be misspelled as 'input'
                    input_cost_key = 'input_token_cost' if 'input_token_cost' in data else ('input_cost_per_1k' if 'input_cost_per_1k' in data else 'input')
                    output_token_key = 'output_token_cost' if 'output_token_cost' in data else ('output_cost_per_1k' if 'output_cost_per_1k' in data else 'output')
                    cost_map[model_id] = {
                        "input": data[input_cost_key],
                        "output": data[output_token_key]
                    }
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line: {line}")
                except KeyError as e:
                    print(f"Warning: Missing key in data: {e} for line: {line}")

        # Print the results in the requested format
        print("DEFAULT_BEDROCK_MODELS = [")
        for model in bedrock_models:
            print(f"    \"{model}\",")
        print("]")
        print()

        print("DEFAULT_OPENAI_MODELS = [")
        for model in openai_models:
            print(f"    \"{model}\",")
        print("]")
        print()

        print("# Default token costs (per 1000 tokens)")
        print("DEFAULT_COST_MAP = {")
        for model, costs in cost_map.items():
            print(f"    \"{model}\": {{\"input\": {costs['input']}, \"output\": {costs['output']}}},")
        print("}")

        # Return the generated structures
        return {
            "DEFAULT_BEDROCK_MODELS": bedrock_models,
            "DEFAULT_OPENAI_MODELS": openai_models,
            "DEFAULT_COST_MAP": cost_map
        }

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None



"""Constants for the Streamlit dashboard."""

# App title and information
APP_TITLE = "LLM Benchmarking Dashboard"
SIDEBAR_INFO = """
### LLM Benchmarking Dashboard

This dashboard provides an intuitive interface for:
- Setting up evaluations from CSV files
- Configuring model parameters
- Selecting judge models
- Monitoring evaluation progress
- Viewing results and reports

For more details, see the [README.md](https://github.com/aws-samples/amazon-bedrock-samples/tree/360-eval/poc-to-prod/360-eval)
"""

# Default directories
DEFAULT_OUTPUT_DIR = "benchmark_results"
DEFAULT_PROMPT_EVAL_DIR = "prompt-evaluations"
CONFIG_DIR = "config"

# Evaluation parameters
DEFAULT_PARALLEL_CALLS = 4
DEFAULT_INVOCATIONS_PER_SCENARIO = 5
DEFAULT_SLEEP_BETWEEN_INVOCATIONS = 3
DEFAULT_EXPERIMENT_COUNTS = 1
DEFAULT_TEMPERATURE_VARIATIONS = 2

# Default model regions
AWS_REGIONS = [
        # North America
        'us-east-1',  # N. Virginia
        'us-east-2',  # Ohio
        'us-west-1',  # N. California
        'us-west-2',  # Oregon

        # Africa
        'af-south-1',  # Cape Town

        # Asia Pacific
        'ap-east-1',  # Hong Kong
        'ap-south-2',  # Hyderabad
        'ap-southeast-3',  # Jakarta
        'ap-southeast-5',  # Malaysia
        'ap-southeast-4',  # Melbourne
        'ap-south-1',  # Mumbai
        'ap-northeast-3',  # Osaka
        'ap-northeast-2',  # Seoul
        'ap-southeast-1',  # Singapore
        'ap-southeast-2',  # Sydney
        'ap-southeast-7',  # Thailand
        'ap-northeast-1',  # Tokyo

        # Canada
        'ca-central-1',  # Central
        'ca-west-1',  # Calgary

        # Europe
        'eu-central-1',  # Frankfurt
        'eu-west-1',  # Ireland
        'eu-west-2',  # London
        'eu-south-1',  # Milan
        'eu-west-3',  # Paris
        'eu-south-2',  # Spain
        'eu-north-1',  # Stockholm
        'eu-central-2',  # Zurich

        # Israel
        'il-central-1',  # Tel Aviv

        # Mexico
        'mx-central-1',  # Central

        # Middle East
        'me-south-1',  # Bahrain
        'me-central-1',  # UAE

        # South America
        'sa-east-1',  # São Paulo

        # AWS GovCloud
        'us-gov-east-1',  # US-East
        'us-gov-west-1',  # US-West
    ]


defaults = generate_model_info()

DEFAULT_BEDROCK_MODELS = defaults['DEFAULT_BEDROCK_MODELS']
DEFAULT_OPENAI_MODELS = defaults['DEFAULT_OPENAI_MODELS']
DEFAULT_COST_MAP = defaults['DEFAULT_COST_MAP']

judges = generate_model_info("/Users/claumazz/Projects/amazon-bedrock-samples/poc-to-prod/360-eval/config/judge_profiles.jsonl")

DEFAULT_JUDGES = judges['DEFAULT_BEDROCK_MODELS']
DEFAULT_JUDGES_COST = judges['DEFAULT_COST_MAP']

# # Default model list - can be extended
# DEFAULT_BEDROCK_MODELS = [
#     "amazon.nova-pro-v1:0",
#     "amazon.nova-lite-v1:0",
#     "us.anthropic.claude-3-5-haiku-20241022-v1:0",
#     "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
#     "us.meta.llama3-3-70b-instruct-v1:0",
#     "meta.llama3-70b-instruct-v1:0",
#     "mistral.mixtral-8x7b-instruct-v0:1",
# ]
#
# DEFAULT_OPENAI_MODELS = [
#     "openai/gpt-4.1",
#     "openai/gpt-4.1-mini",
#     "openai/gpt-4o",
#     "openai/gpt-4o-mini",
# ]
#
# # Default token costs (per 1000 tokens)
# DEFAULT_COST_MAP = {
#     "amazon.nova-pro-v1:0": {"input": 0.0008, "output": 0.0032},
#     "amazon.nova-lite-v1:0": {"input": 0.00006, "output": 0.000015},
#     "us.anthropic.claude-3-5-haiku-20241022-v1:0": {"input": 0.000001, "output": 0.000015},
#     "us.anthropic.claude-3-5-sonnet-20241022-v2:0": {"input": 0.003, "output": 0.015},
#     "us.meta.llama3-3-70b-instruct-v1:0": {"input": 0.00072, "output": 0.00072},
#     "mistral.mixtral-8x7b-instruct-v0:1": {"input": 0.00045, "output": 0.0007},
#     "openai/gpt-4.1": {"input": 0.002, "output": 0.012},
#     "openai/gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},
#     "openai/gpt-4o": {"input": 0.005, "output": 0.02},
#     "openai/gpt-4o-mini": {"input": 0.0006, "output": 0.0024},
# }