# Method

## Grids and candidate points

The method uses three nested sparse grids:

- An auxiliary grid provides a coarse interpolant.
- A baseline grid contains all initially evaluated points.
- A target grid defines the candidate points for refinement.

The auxiliary grid is contained in the baseline, so its interpolant reuses
baseline evaluations. Candidates are target nodes absent from the baseline.
Only selected candidates require additional simulations.

The total-degree index set is `sum(i_j - 1) <= level`, with one-dimensional
levels starting at one. Choose `auxiliary_level < base_level < target_level`;
the auxiliary level defaults to `base_level - 1`. Clenshaw-Curtis and Leja rules
use different node growth, so equal levels need not imply equal costs.

## Discrepancy indicators

Evaluate the baseline and auxiliary interpolants at every candidate. The
absolute indicator is:

```text
eta(x) = |I_base(x) - I_aux(x)|
```

The stabilized symmetric relative indicator is:

```text
eta(x) = 2*|I_base(x)-I_aux(x)| / (|I_base(x)|+|I_aux(x)|+epsilon*S)
S = max over candidates of max(|I_base(x)|, |I_aux(x)|)
```

Set `indicator_type` to `absolute` or `symmetric_relative`. The stabilizer
`relative_epsilon` defaults to `1e-5`. Identically zero predictions produce
zero indicators.

For a vector criterion, interpolate each component first. The numerator uses
the Euclidean norm of the component differences; prediction magnitudes in the
denominator and scale also use Euclidean norms.

## Point selection

A percentage budget requests `floor(q*N_candidates)` evaluations, where
`q=budget_percent/100`. For a fixed count, omit `budget_percent` and set `max_budget`.
A requested count of zero selects no points. Count-based selection ranks
candidates by decreasing indicator.

Threshold selection uses the following refinement settings:

```json
{
    "use_auto_budget": true,
    "auto_budget_method": "threshold",
    "auto_budget_threshold": 0.05,
    "include_ties": true,
    "tie_tolerance": 0.001
}
```

The threshold cutoff is `auto_budget_threshold*max(eta)`. Count-based selection
uses the indicator at the last requested rank as its cutoff. With ties enabled,
points satisfying `eta >= cutoff - tie_tolerance*abs(cutoff)` are included.
The actual selected count can therefore exceed the nominal budget.

Set `include_ties=false` for a strict count limit. An optional `max_budget`
caps the nominal threshold count; when active, that cap uses a rank cutoff.
The configuration reference describes selection precedence and elbow selection.

## Surrogate reconstruction

Keep evaluated values at baseline nodes and selected target nodes. Fill each
remaining target node with the baseline prediction, then interpolate the
completed target grid. Optional additional target evaluations are also used
when supplied in the Phase 2 file.

With no additional evaluations, the reconstructed model reproduces the
baseline interpolant. With every target node evaluated, it reproduces the
fully evaluated target interpolant. Accuracy between these limits depends on
the response, grid, and selected points.

## Validation

Independent reference values provide RMSE, mean absolute error (`mae`), maximum
absolute error (`max_absolute_error`), the 95th percentile absolute error, and
signed bias. These metrics have the output's physical units.

Percentage error plots use:

```text
100*abs(prediction-reference)/(max(reference)-min(reference))
```

The reference range is computed over the validation set for each output
component. For constant references, absolute metrics remain available and
percentage plots are skipped.
