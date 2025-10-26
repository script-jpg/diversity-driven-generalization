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

import yaml
import json

from typing import Dict, Any, Optional, List, Tuple
from lean_dojo import Dojo, Theorem, LeanGitRepo, TacticState, LeanError
from loguru import logger


# Load config once at module level
with open("config.yaml", 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


class TheoremData:
    """Theorem data and dojo interaction management"""
    
    def __init__(self):
        self._statements = None
        self._dojo_data = None
    
    @property
    def statements(self) -> Dict[str, Any]:
        """Get theorem statements (id -> statement)"""
        if self._statements is None:
            self._statements = {}
            with open(config["statement_file"], 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        self._statements[entry["id"]] = entry
        return self._statements
    
    @property
    def dojo_data(self) -> Dict[str, Any]:
        """Get dojo data (full_name -> dojo info)"""
        if self._dojo_data is None:
            self._dojo_data = {}
            with open(config["dojo_data_file"], 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        self._dojo_data[entry["full_name"]] = entry
        return self._dojo_data
    
    def get_statement(self, theorem_id: str) -> str:
        """Get theorem statement by id"""
        return self.statements[theorem_id]["statement"]
    
    def verify_plan(self, theorem_id: str, plan: List[str]) -> Tuple[bool, Optional[str]]:
        """Verify if plan can execute successfully for theorem
        
        Returns:
            (success, error_msg): True if all steps pass, False with error msg otherwise
        """
        # Empty plan is invalid
        if not plan:
            error_msg = "Plan is empty"
            logger.warning(f"Plan verification failed for {theorem_id}: {error_msg}, steps: {plan}")
            return False, error_msg
        
        # Check if theorem exists in dojo data
        if theorem_id not in self.dojo_data:
            error_msg = f"Theorem '{theorem_id}' not found in dojo data file. Please ensure the theorem exists in both statement_file and dojo_data_file."
            logger.error(f"Plan verification failed for {theorem_id}: {error_msg}")
            return False, error_msg
        
        dojo_info = self.dojo_data[theorem_id]
        theorem = Theorem(
            repo=LeanGitRepo(url=dojo_info["url"], commit=dojo_info["commit"]),
            file_path=dojo_info["file_path"],
            full_name=dojo_info["full_name"]
        )
        
        try:
            with Dojo(theorem, timeout=600) as (dojo, state):
                for idx, step in enumerate(plan):
                    state = dojo.run_tac(state, step)
                    if isinstance(state, LeanError):
                        error_msg = f"Step {idx} failed: {step} -> {state.error}"
                        logger.warning(f"Plan verification failed for {theorem_id}: {error_msg}, steps: {plan}")
                        return False, error_msg
                    if idx != len(plan) - 1 and not isinstance(state, TacticState):
                        error_msg = f"Step {idx} unexpected state: {step} -> {type(state).__name__}"
                        logger.warning(f"Plan verification failed for {theorem_id}: {error_msg}, steps: {plan}")
                        return False, error_msg
                    state = dojo.run_tac(state, "sorry")
                assert isinstance(state, TacticState), "Final state must be a TacticState"
                logger.info(f"Plan verification succeeded for {theorem_id}, steps: {plan}")
                return True, None
                    
        except Exception as ex:
            error_msg = f"Dojo exception: {ex}"
            logger.warning(f"Plan verification failed for {theorem_id}: {error_msg}, steps: {plan}")
            return False, error_msg


# Global instances
theorem_data = TheoremData()
