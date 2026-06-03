# Unresolved dependencies and implementation boundaries

This file records verified boundaries that remain after the repository update.

- Full online Wilson inclusion probabilities for a changing per-step candidate set are not available in closed form in the current training loop. The online normalized-weight implementation therefore provides finite mean-one weights for uniform and degree sampling, and a documented mean-one proxy for Wilson and Adaptive Wilson. Static Wilson weights still support exact root marginals and Monte Carlo forest estimates.
- Adaptive Wilson currently implements the supervisor-recommended baseline `q_i = alpha` as the first supported adaptive mode. The code exposes `alpha` and `alpha_grid` in YAML; richer loss-, uncertainty-, or coverage-adaptive `q_i,t` schedules are left as future modes because they require a policy for updating node scores during training.
- Wilson exact checks solve dense linear systems and are guarded by `wilson_checks.max_exact_nodes`. Large graphs should use smaller diagnostic subgraphs or increase the limit explicitly after checking memory.
- Dataset downloads still depend on PyTorch Geometric dataset classes and external dataset availability.
- The final ZIP intentionally omits macOS `__MACOSX` resource-fork metadata from the original upload because those are archive artifacts, not repository source files.
