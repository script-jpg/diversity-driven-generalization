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

# Initial Planning Script
# Generates initial plans for all theorems

python src/plan/main.py \
    --mode initial \
    --num-plan-gen 7 \
    --num-planners 7 \
    --num-cpus-per-planner 10 \
    --max-retries 3 > src/plan/plan.log 2>&1

# Optional: Specify specific theorems to plan
# Uncomment and modify the line below to plan specific theorems
# python src/plan/main.py \
#     --mode initial \
#     --theorems theorem_id_1 theorem_id_2 theorem_id_3 \
#     --num-plan-gen 7 \
#     --num-planners 7 \
#     --num-cpus-per-planner 10 \
#     --max-retries 3 > src/plan/plan.log 2>&1
