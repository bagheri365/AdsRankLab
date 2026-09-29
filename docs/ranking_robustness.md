# Tie-aware ranking robustness

The value-ranking experiment uses deterministic row-order tie breaking so exact
top-k sets are reproducible. Some policies, however, have large exact-score
ties at important cutoffs. A deterministic tie breaker should not be
interpreted as evidence that one tied row is economically preferred to another.

This robustness experiment therefore leaves the existing value-ranking
experiment unchanged and adds a separate tie-aware analysis.

At the top 0.1%, 1%, and 5% cutoffs it reports:

- rows strictly above the cutoff (`guaranteed_selected`);
- rows tied at the cutoff (`ambiguous_boundary_rows`);
- selected slots that must be filled from the cutoff tie;
- guaranteed-set Jaccard overlap;
- possible-set Jaccard overlap when all cutoff-tied rows are admitted.

Rank shifts are also normalized by the full held-out population size.

The experiment remains restricted to the historically observed won-impression
population and does not claim counterfactual auction outcomes.

Run:

```bash
python -m ads_rank_lab.experiments.ranking_robustness
```

Output:

```text
artifacts/results/ranking_robustness_season2.json
```
