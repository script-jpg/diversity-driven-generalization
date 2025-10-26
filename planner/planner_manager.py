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

import time
import json
import ray
import atexit

from typing import List, Dict, Optional, Tuple
from loguru import logger
from tqdm import tqdm
from ray.util import ActorPool

from generate import build_prompt, parse_have, get_model
from verify import theorem_data, config

# Timeout for model readiness check
MODEL_READINESS_TIMEOUT = 300


@ray.remote
class PlanIOActor:
    """Centralized Plan I/O service using Ray actor for efficient file operations"""
    
    def __init__(self):
        """Initialize by loading the plan file once"""
        plan_file = config["plan_file"]
        
        try:
            with open(plan_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content:
                    logger.warning(f"Plan file '{plan_file}' is empty. Starting with empty plans.")
                    self.plans = {}
                else:
                    self.plans = json.loads(content)
                    logger.info(f"Loaded {len(self.plans)} existing plans from file")
        except FileNotFoundError:
            logger.warning(f"Plan file '{plan_file}' does not exist. Will create it when saving.")
            self.plans = {}
        except json.JSONDecodeError as e:
            logger.warning(f"Plan file '{plan_file}' contains invalid JSON: {e}. Starting with empty plans.")
            self.plans = {}
        
        logger.info("PlanIOActor initialized")
    
    def get_theorem_info(self, theorem_id: str) -> Dict:
        """Get plan info for a theorem (fast, in-memory lookup)"""
        return self.plans[theorem_id]
    
    def get_all_theorem_ids(self) -> List[str]:
        """Get all theorem IDs that have plans"""
        return list(self.plans.keys())
    
    def update_plan(self, theorem_id: str, plan_steps: List[str]) -> None:
        """Update plan in memory"""
        existing = self.plans.get(theorem_id, {})
        progress = existing.get("progress", 0)
        self.plans[theorem_id] = {"plan": plan_steps, "progress": progress}
    
    def save_plans(self) -> None:
        """Write all plans to disk"""
        try:
            with open(config["plan_file"], 'w', encoding='utf-8') as f:
                json.dump(self.plans, f, indent=2, ensure_ascii=False)
            logger.info(f"Plans saved to {config['plan_file']}")
        except Exception as e:
            logger.error(f"Error saving plans: {e}")


@ray.remote
class PlanGenerator:
    """Ray actor wrapping the LLM for plan generation"""
    
    def __init__(self):
        """Initialize the model"""
        self.model = get_model()
        logger.info(f"PlanGenerator initialized with model type: {self.model.model_name}")
    
    def is_ready(self) -> bool:
        """Check if the model is ready"""
        return True
    
    def generate_plan(
        self,
        theorem_statement: str,
        is_replan: bool = False,
        proven_subgoals: Optional[List[str]] = None,
        stuck_subgoal: Optional[str] = None
    ) -> List[str]:
        """Generate plan steps using the model"""
        messages = build_prompt(
            theorem_statement,
            is_replan=is_replan,
            proven_subgoals=proven_subgoals,
            stuck_subgoal=stuck_subgoal
        )
        
        content = self.model.generate(messages)
        steps = parse_have(content)
        return steps
    
    def shutdown(self) -> None:
        """Clean up model resources"""
        if hasattr(self.model, 'cleanup'):
            self.model.cleanup()


@ray.remote
class Planner:
    """Ray actor that handles planning for theorems with retry logic"""
    
    def __init__(
        self,
        plan_gen: PlanGenerator,
        plan_io_actor: PlanIOActor,
        max_retries: int = 3
    ):
        self.plan_gen = plan_gen
        self.plan_io_actor = plan_io_actor
        self.max_retries = max_retries
    
    def plan_theorem(
        self,
        theorem_id: str,
        is_replan: bool = False
    ) -> Tuple[str, Dict]:
        """
        Plan a single theorem (initial planning or replanning)
        
        Returns:
            (theorem_id, result_dict)
            result_dict contains: {plan, progress, verified, error (optional)}
        """
        try:
            # Get theorem statement
            statement = theorem_data.get_statement(theorem_id)
            
            # For replanning, get existing plan info
            proven_subgoals = None
            stuck_subgoal = None
            progress = 0
            
            if is_replan:
                theorem_info = ray.get(self.plan_io_actor.get_theorem_info.remote(theorem_id))
                
                full_plan = theorem_info["plan"]
                progress = theorem_info["progress"]
                
                if progress >= len(full_plan):
                    logger.error(f"Progress {progress} exceeds plan length {len(full_plan)}")
                    return theorem_id, {
                        "plan": full_plan,
                        "progress": progress,
                        "verified": False,
                        "error": "Progress exceeds plan length"
                    }
                
                stuck_subgoal = full_plan[progress]
                proven_subgoals = full_plan[:progress]  # Only the proven subgoals before stuck
                logger.info(f"[{theorem_id}] Stuck at progress {progress}, subgoal: {stuck_subgoal}")
            
            # Generate and verify plan with retries
            plan, success, error_msg = self._get_verified_plan(
                theorem_id,
                statement,
                is_replan,
                proven_subgoals,
                stuck_subgoal
            )
            
            # Build result
            result = {
                "plan": plan,
                "progress": progress,
                "verified": success
            }
            if not success:
                result["error"] = error_msg
            
            plan_type = "replan" if is_replan else "initial plan"
            if success:
                logger.info(f"[{theorem_id}] Successfully verified {plan_type}, result: {result}")
            else:
                logger.error(f"[{theorem_id}] Failed to verify {plan_type}: {error_msg}, result: {result}")
            return theorem_id, result
            
        except Exception as e:
            logger.error(f"[{theorem_id}] Exception during planning: {e}")
            return theorem_id, {
                "plan": [],
                "progress": 0,
                "verified": False,
                "error": str(e)
            }
    
    def _get_verified_plan(
        self,
        theorem_id: str,
        statement: str,
        is_replan: bool,
        proven_subgoals: Optional[List[str]],
        stuck_subgoal: Optional[str]
    ) -> Tuple[Optional[List[str]], bool, Optional[str]]:
        """Generate plan with verification and retry logic"""
        plan_type = "replanning" if is_replan else "plan"
        
        # Generate plan first
        try:
            plan = ray.get(self.plan_gen.generate_plan.remote(statement, is_replan, proven_subgoals, stuck_subgoal))
            logger.info(f"[{theorem_id}] Generated plan with {len(plan)} steps: {plan}")
        except Exception as e:
            logger.error(f"[{theorem_id}] Plan generation failed: {e}")
            return [], False, f"Plan generation failed: {e}"
        
        # Skip verification for now (to see generated plans)
        logger.info(f"[{theorem_id}] Skipping verification - returning generated plan")
        return plan, False, "Verification skipped"
        
        # Retry logic for failed verifications
        for retry_count in range(1, self.max_retries + 1):
            if success:
                break
            logger.warning(
                f"[{theorem_id}] {plan_type.capitalize()} verification failed. "
                f"Retry {retry_count}/{self.max_retries}: {error_msg}"
            )
            try:
                plan = ray.get(self.plan_gen.generate_plan.remote(statement, is_replan, proven_subgoals, stuck_subgoal))
                success, error_msg = theorem_data.verify_plan(theorem_id, plan)
            except Exception as e:
                logger.warning(f"[{theorem_id}] Retry {retry_count} failed due to system error: {e}")
                break
        
        return plan, success, error_msg


class PlannerManager:
    """Manager for parallel planning across theorems"""
    
    def __init__(
        self,
        num_plan_gen: int = 1,
        num_planners: int = 4,
        num_gpus_per_plan_gen: int = 0,
        num_cpus_per_planner: int = 1,
        max_retries: int = 3
    ):
        """
        Initialize PlannerManager
        
        Args:
            num_plan_gen: Number of plan generator actors (models)
            num_planners: Number of planner actors
            num_cpus_per_planner: CPUs allocated per planner
            num_gpus_per_plan_gen: GPUs per plan generator (for local models)
            max_retries: Maximum retries for failed verification
        """
        logger.info(
            f"Initializing PlannerManager with {num_plan_gen} plan generators "
            f"and {num_planners} planners..."
        )
        
        # Create centralized PlanIO actor
        logger.info("Creating PlanIO actor...")
        self.plan_io_actor = PlanIOActor.remote()
        
        # Create plan generators (with GPU support for local models)
        logger.info("Creating plan generators...")
        if num_gpus_per_plan_gen > 0:
            self.plan_generators = [
                PlanGenerator.options(num_gpus=num_gpus_per_plan_gen).remote()
                for _ in range(num_plan_gen)
            ]
        else:
            self.plan_generators = [
                PlanGenerator.remote()
                for _ in range(num_plan_gen)
            ]
        
        # Wait for all generators to be ready
        logger.info("Waiting for plan generators to be ready...")
        self._wait_for_generators_ready(timeout=MODEL_READINESS_TIMEOUT)
        
        # Create planners (all share the same PlanIO actor)
        logger.info("Creating planners...")
        self.planners = [
            Planner.options(num_cpus=num_cpus_per_planner).remote(
                plan_gen=self.plan_generators[i % num_plan_gen],
                plan_io_actor=self.plan_io_actor,
                max_retries=max_retries
            )
            for i in range(num_planners)
        ]
        self.planner_pool = ActorPool(self.planners)
        
        gpu_info = f", each with {num_gpus_per_plan_gen} GPU(s)" if num_gpus_per_plan_gen > 0 else ""
        logger.info(
            f"Initialized {num_plan_gen} plan generator(s){gpu_info} and "
            f"{num_planners} planner(s), each with {num_cpus_per_planner} CPU(s)"
        )
        
        # Register cleanup to run on exit (handles normal exit and Ctrl+C)
        atexit.register(self.cleanup)
    
    def _wait_for_generators_ready(self, timeout: int = 300) -> None:
        """Wait for all plan generators to be ready"""
        logger.info(
            f"Waiting for all {len(self.plan_generators)} plan generators to be ready..."
        )
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                ready_futures = [
                    gen.is_ready.remote() for gen in self.plan_generators
                ]
                ready_results = ray.get(ready_futures, timeout=10)
                if all(ready_results):
                    elapsed_time = time.time() - start_time
                    logger.info(
                        f"All {len(self.plan_generators)} plan generators are ready! "
                        f"(took {elapsed_time:.2f}s)"
                    )
                    return
                else:
                    ready_count = sum(ready_results)
                    elapsed_time = time.time() - start_time
                    logger.info(
                        f"{ready_count}/{len(self.plan_generators)} plan generators ready, "
                        f"waiting... (elapsed: {elapsed_time:.2f}s)"
                    )
                    time.sleep(10)
            except Exception as e:
                elapsed_time = time.time() - start_time
                logger.warning(
                    f"Plan generators not ready yet, elapsed: {elapsed_time:.2f}s, "
                    f"log: {e}"
                )
                time.sleep(10)
        
        logger.warning(
            f"Not all plan generators became ready within {timeout} seconds"
        )
    
    def cleanup(self) -> None:
        """Clean up all plan generator resources"""
        logger.info("Cleaning up plan generators...")
        try:
            # Shutdown all generators
            shutdown_futures = [gen.shutdown.remote() for gen in self.plan_generators]
            ray.get(shutdown_futures, timeout=30)
            logger.info("All plan generators cleaned up successfully")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    def plan_theorems(
        self,
        theorem_ids: List[str],
        is_replan: bool = False
    ) -> Dict[str, Dict]:
        """
        Plan theorems in parallel
        
        Args:
            theorem_ids: List of theorem IDs to plan
            is_replan: If True, do replanning; if False, do initial planning
        
        Returns:
            Dict mapping theorem_id -> result_dict
        """
        mode = "replanning" if is_replan else "initial planning"
        logger.info(f"Starting parallel {mode} for {len(theorem_ids)} theorems...")
        
        # Map theorems across planner pool
        future_results = self.planner_pool.map_unordered(
            lambda planner, thm_id: planner.plan_theorem.remote(thm_id, is_replan),
            theorem_ids
        )
        
        results = {}
        num_verified = 0
        update_futures = []  # Track async update operations
        
        for total, (theorem_id, result) in enumerate(
            tqdm(future_results, total=len(theorem_ids), desc=mode.capitalize()), 
            start=1
        ):
            results[theorem_id] = result
            
            # Update plan asynchronously (fast, in-memory) - save both verified and unverified plans
            if result.get("plan"):  # Save plan if it exists, regardless of verification status
                future = self.plan_io_actor.update_plan.remote(theorem_id, result["plan"])
                update_futures.append(future)
                
            if result["verified"]:
                num_verified += 1
            
            logger.info(
                f"Completed {total}/{len(theorem_ids)} theorems, "
                f"{num_verified} verified, acc: {num_verified/total:.2%}"
            )
        
        # Wait for all async updates to complete
        if update_futures:
            ray.get(update_futures)
            logger.info(f"All {len(update_futures)} plan updates completed")
        
        # Save all changes to disk atomically
        ray.get(self.plan_io_actor.save_plans.remote())
        
        # Summary
        total = len(results)
        logger.info(
            f"{mode.capitalize()} Complete: {num_verified}/{total} plans verified "
            f"({num_verified/total if total > 0 else 0:.2%})"
        )
        
        return results
