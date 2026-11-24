# Robotic clutching: statistical analysis

Analysis code for *Effects of Clutching-Related Visuomotor Misalignments on
Performance in Simulated Robotic Surgery*.

## Contents

- `analyse.py`: mixed ANOVAs, participant-level permutation checks for peg drops,
  and directional Bayesian completion-time comparisons (X, Y and Z versus aligned
  across trials 1–10, plus Z versus aligned across trials 1–5). Includes prior-width
  and log-time sensitivity checks.
- `requirements.txt`: the three Python dependencies.

The code produces numerical tables only. Data, figures, exploratory hierarchical
models and figure-generation code are not included. De-identified data can be
requested from the corresponding author by email.

## Input data

Supply a CSV with one row per participant per trial and these columns:

| Column | Description |
| --- | --- |
| `participant` | Anonymous participant identifier, unique across the study |
| `group` | `Aligned`, `X-misaligned`, `Y-misaligned` or `Z-misaligned` |
| `trial` | Integer from 1 to 10 |
| `completion_time` | Completion time in seconds, greater than zero |
| `drops` | Nonnegative integer peg-drop count |

The script expects the complete analysis dataset: 40 participants, 10 in each
condition, with all 10 trials per participant (400 rows). Each participant must
belong to exactly one condition. Missing values, duplicate trials, additional
trials and incomplete records cause an error; they are not silently excluded.
Additional columns are ignored. Participant IDs should retain their original
anonymous values to reproduce the exact seeded permutation p values.

## Setup and run

Tested with Python 3.12.4. Python package versions are recorded with each run.

From this directory, install the Python dependencies, preferably in a virtual
environment:

```sh
python -m pip install -r requirements.txt
```

Run all analyses with the path to the requested dataset:

```sh
python analyse.py /path/to/data.csv --out results
```

## Analyses and outputs

### Main frequentist analysis

`mixed_anova.csv` contains the condition, trial and condition-by-trial tests for
both outcomes. Condition is between participants; trial is within participants.
Greenhouse–Geisser corrections apply to trial and interaction degrees of freedom
and p values. Effect sizes are partial eta squared. There are no post-hoc
pairwise tests. `descriptives.csv` summarises participant averages, with sample
SDs calculated between participants.

`drops_permutation.csv` contains robustness checks for the condition and
condition-by-trial effects on drops. Each of 99,999 permutations reassigns
condition labels to entire participant trajectories, keeping 10 participants per
condition. The random seed is 20261030. The p value is `(exceedances + 1) / 100000`.

### Supplementary Bayesian analysis

Each participant contributes one average completion time, so the tests have
10 independent observations per condition, not 100. The three planned
misaligned-versus-aligned comparisons are tested separately; there are no
comparisons between misalignment conditions.

The null model fixes the population mean difference at zero. The directional
alternative assigns a positive half-Cauchy prior to the standardised mean
difference, defined as misaligned minus aligned divided by the common population
SD. Positive effects represent slowdowns. The standard prior scale is
`r = sqrt(2)/2`, approximately 0.707. These are standard JZS independent-samples
t-tests, with a common variance and shared noninformative nuisance priors.
SciPy quadrature integrates the noncentral-t density over the effect-size
prior. No posterior sampling is needed.

`bayesian_main.csv` reports BF0+ at the standard width for the three ten-trial
comparisons and the five-trial Z comparison. BF0+ greater than 1 favours exact
zero difference over the specified slowdown model. It is not a posterior
probability or a test of practical equivalence. The comparisons share an aligned
group, and the five-trial check uses the same participants; their Bayes factors
should not be multiplied.

`bayesian_sensitivity.csv` repeats these tests across prior widths 0.2–1.5 and
on natural-log-transformed participant averages. Averaging precedes the log
transformation. Logged results describe log participant means, not seconds.

Output files are overwritten when rerunning with the same output directory.
The supplied `.gitignore` excludes data CSVs and generated results from Git
by default.

## Statistical reference

The Bayesian tests implement the JZS model described in the
[`ttestBF` documentation](https://search.r-project.org/CRAN/refmans/BayesFactor/html/ttestBF.html),
using [SciPy numerical integration](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html)
and its [noncentral-t density](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html).

## Verification

Numerical checks covered zero, positive and negative effects, balanced and
unequal group sizes, and multiple prior widths. An independent variance-mixture
formula also reproduced the symmetric JZS Bayes factors. These checks assess
implementation accuracy; they do not assess whether the model assumptions hold
in the data.
