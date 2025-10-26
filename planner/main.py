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

import argparse
import ray

from typing import List, Dict, Optional
from loguru import logger

from planner_manager import PlannerManager
from verify import theorem_data


def _run_planning(
    is_replan: bool,
    num_plan_gen: int,
    num_planners: int,
    num_gpus_per_plan_gen: int,
    num_cpus_per_planner: int,
    max_retries: int,
    specific_theorems: Optional[List[str]]
) -> Dict[str, Dict]:
    """
    Common implementation for both initial planning and replanning
    
    Args:
        is_replan: If True, do replanning; if False, do initial planning
        num_plan_gen: Number of plan generator actors
        num_planners: Number of planner actors
        num_gpus_per_plan_gen: GPUs per plan generator
        num_cpus_per_planner: CPUs per planner
        max_retries: Maximum retries for verification
        specific_theorems: Optional list of specific theorem IDs
    
    Returns:
        Dict mapping theorem_id -> result_dict
    
    Note:
        The planning system only updates plans, not progress. Progress is
        managed by the search system and is automatically preserved when
        saving plans.
    """
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init()
    
    # Create planner manager (which creates the PlanIO actor)
    manager = PlannerManager(
        num_plan_gen=num_plan_gen,
        num_planners=num_planners,
        num_gpus_per_plan_gen=num_gpus_per_plan_gen,
        num_cpus_per_planner=num_cpus_per_planner,
        max_retries=max_retries
    )
    
    # Get theorem IDs to process
    if specific_theorems:
        theorem_ids = specific_theorems
        logger.info(f"Processing {len(theorem_ids)} specific theorems")
    else:
        if is_replan:
            # Reuse manager's PlanIO actor to get all theorem IDs
            theorem_ids = ray.get(manager.plan_io_actor.get_all_theorem_ids.remote())
            if not theorem_ids:
                logger.error("No existing plans found. Cannot perform replanning on empty plan file. Please run initial planning first.")
                ray.shutdown()
                return {}
            logger.info(f"Processing all {len(theorem_ids)} theorems with existing plans")
        else:
            theorem_ids = list(theorem_data.statements.keys())
            logger.info(f"Processing all {len(theorem_ids)} theorems")
    
    return manager.plan_theorems(theorem_ids, is_replan=is_replan)


def initial_planning(
    num_plan_gen: int = 1,
    num_planners: int = 4,
    num_gpus_per_plan_gen: int = 0,
    num_cpus_per_planner: int = 1,
    max_retries: int = 3,
    specific_theorems: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Generate initial plans for theorems in parallel
    
    Args:
        num_plan_gen: Number of plan generator actors
        num_planners: Number of planner actors
        num_gpus_per_plan_gen: GPUs per plan generator (for local models)
        num_cpus_per_planner: CPUs per planner
        max_retries: Maximum retries for verification
        specific_theorems: Optional list of specific theorem IDs
    
    Returns:
        Dict mapping theorem_id -> result_dict
    """
    return _run_planning(
        is_replan=False,
        num_plan_gen=num_plan_gen,
        num_planners=num_planners,
        num_gpus_per_plan_gen=num_gpus_per_plan_gen,
        num_cpus_per_planner=num_cpus_per_planner,
        max_retries=max_retries,
        specific_theorems=specific_theorems
    )


def dynamic_replanning(
    num_plan_gen: int = 1,
    num_planners: int = 4,
    num_gpus_per_plan_gen: int = 0,
    num_cpus_per_planner: int = 1,
    max_retries: int = 3,
    specific_theorems: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Generate dynamic replanning for theorems in parallel
    
    Args:
        num_plan_gen: Number of plan generator actors
        num_planners: Number of planner actors
        num_gpus_per_plan_gen: GPUs per plan generator (for local models)
        num_cpus_per_planner: CPUs per planner
        max_retries: Maximum retries for verification
        specific_theorems: Optional list of specific theorem IDs
    
    Returns:
        Dict mapping theorem_id -> result_dict
    """
    return _run_planning(
        is_replan=True,
        num_plan_gen=num_plan_gen,
        num_planners=num_planners,
        num_gpus_per_plan_gen=num_gpus_per_plan_gen,
        num_cpus_per_planner=num_cpus_per_planner,
        max_retries=max_retries,
        specific_theorems=specific_theorems
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Parallel Theorem Planning System")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["initial", "replan"],
        help="Planning mode: 'initial' for initial planning, 'replan' for dynamic replanning"
    )
    parser.add_argument(
        "--theorems",
        type=str,
        nargs="+",
        help="Specific theorem IDs to process (optional)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for failed verifications (default: 3)"
    )
    parser.add_argument(
        "--num-plan-gen",
        type=int,
        default=1,
        help="Number of plan generator actors (default: 1)"
    )
    parser.add_argument(
        "--num-planners",
        type=int,
        default=4,
        help="Number of planner actors (default: 4)"
    )
    parser.add_argument(
        "--num-gpus-per-plan-gen",
        type=int,
        default=0,
        help="GPUs per plan generator actor, for local models (default: 0)"
    )
    parser.add_argument(
        "--num-cpus-per-planner",
        type=int,
        default=1,
        help="CPUs per planner actor (default: 1)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    planning_func = initial_planning if args.mode == "initial" else dynamic_replanning
    planning_func(
        num_plan_gen=args.num_plan_gen,
        num_planners=args.num_planners,
        num_gpus_per_plan_gen=args.num_gpus_per_plan_gen,
        num_cpus_per_planner=args.num_cpus_per_planner,
        max_retries=args.max_retries,
        specific_theorems=args.theorems
    )


if __name__ == "__main__":
    main()
