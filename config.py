import os
import logging
import sys
from models.TOT_10 import TreeOfThoughts

# API Configuration
# API_KEY = os.getenv("DEEPINFRA_TOKEN", "sk-005ecb1ccb3a41798f5a00465b7292c1")
API_KEY = os.getenv("DEEPINFRA_TOKEN", "6CsmsskJ9LlwYPUMXnsy2LX3u3VgfqIi")
BASE_URL = "https://api.deepinfra.com/v1/openai"
# BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# DEFAULT_MODEL = "qwen2.5-math-7b-instruct"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


evaluation_file = "dataset/cs5260_val_random300.jsonl"
inference_file = "dataset/cs5260_test_random300.jsonl"

# Tree of Thoughts Configuration
TOT_CONFIG = {
    'temperature': 1,
    'max_depth': 4,
    'n_samples_per_step': 3,
    'k_best_thoughts': 1
}

# Parallel Processing Configuration
NUM_PROCESSES = os.cpu_count() - 1 

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(message)s',
    stream=sys.stdout
)