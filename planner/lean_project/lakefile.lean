import Lake
open Lake DSL

package «minimal_theorem_project» where
  -- add package configuration options here

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.10.0"

@[default_target]
lean_lib «MinimalTheoremProject» where
  -- add library configuration options here