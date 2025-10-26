# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import openai
import yaml
import os

from typing import List, Dict, Optional, Any
from loguru import logger

from verify import config

# Load environment variables from .env file
if os.path.exists('.env'):
    with open('.env', 'r') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                key, value = line.strip().split('=', 1)
                # Remove quotes if present
                value = value.strip('\'"')
                os.environ[key] = value


# Load prompt
with open(config["prompt_yaml"], 'r', encoding='utf-8') as f:
    prompt = yaml.safe_load(f)


class Model:
    """LLM using OpenAI API."""
    
    def __init__(self, api_config: Dict[str, Any], sampling_config: Dict[str, Any]):
        self.model_name = api_config.pop("model")
        self.client = openai.OpenAI(**api_config)
        self.sampling_params = {k: v for k, v in sampling_config.items()}
        
        logger.info(f"LLM provider initialized with model: {self.model_name}")
    
    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Generate text using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **self.sampling_params
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise


def build_prompt(theorem_statement: str, is_replan: bool = False, 
                proven_subgoals: Optional[List[str]] = None, 
                stuck_subgoal: Optional[str] = None) -> List[Dict[str, str]]:
    """Build messages for LLM based on planning type"""
    prompt_type = "Dynamic Replanning" if is_replan else "Initial Planning"
    prompt_config = prompt[prompt_type]
    
    # Build user message based on planning type
    if is_replan:
        if stuck_subgoal is None:
            raise ValueError("stuck_subgoal is required for replanning")
        
        # proven_subgoals can be empty list if stuck at the first step
        proven_subgoals_str = "\n".join(proven_subgoals) if proven_subgoals else ""
        
        user_content = prompt_config["template"].format(
            theorem=theorem_statement,
            examples=prompt_config["examples"],
            proven_subgoals=proven_subgoals_str,
            stuck_subgoal=stuck_subgoal
        )
    else:
        user_content = prompt_config["template"].format(
            theorem=theorem_statement, 
            examples=prompt_config["examples"]
        )
    
    return [
        {"role": "system", "content": prompt_config["system"]},
        {"role": "user", "content": user_content}
    ]


def get_model() -> Model:
    """Initialize and return appropriate model based on config"""
    model_type = (config.get("model_type") or "api").lower()
    sampling = config.get("sampling", {})
    
    if model_type == "local":
        return Model(config.get("local_model", {}), sampling)
    elif model_type == "api":
        api_config = config.get("api_model", {}).copy()
        # Replace placeholders with environment variables
        if api_config.get("api_key") == "YOUR_OPENAI_API_KEY":
            api_config["api_key"] = os.getenv("OPENAI_API_KEY")
        return Model(api_config, sampling)
    elif model_type == "azure":
        return Model(config.get("azure_model", {}), sampling)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: 'local', 'api', 'azure'")


def parse_have(text: str) -> List[str]:
    """Parse have statements from model output text"""
    results = []

    # Match from 'have' to next 'have' or end of text
    have_pattern = r'have\s(.*?)(?=\n*have\s|\Z)'
    matches = re.finditer(have_pattern, text, re.DOTALL)
    
    for match in matches:
        # Reconstruct full statement and remove := clause
        statement = f"have {match.group(1).strip()}"
        statement = re.sub(r'\s*:=.*$', '', statement, flags=re.DOTALL)
        statement = re.sub(r'\s+', ' ', statement).strip()
        
        if statement and statement != 'have':
            results.append(statement)
    
    return results
