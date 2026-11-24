"""Frequentist and Bayesian statistical analyses for the clutching study.

Run: python analyse.py /path/to/data.csv --out results
Only numerical results are written; no figures are generated.
"""

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.integrate import quad

GROUPS = ["Aligned", "X-misaligned", "Y-misaligned", "Z-misaligned"]
SEED = 20261030
N_PERMUTATIONS = 99_999
STANDARD_R = np.sqrt(2) / 2


def load_data(path: Path) -> pd.DataFrame:
    """Validate the complete, balanced study dataset before calculating results."""
    columns = ["participant", "group", "trial", "completion_time", "drops"]
    data = pd.read_csv(path)
    missing = set(columns) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    data = data[columns].copy()
    if data.isna().any().any():
        raise ValueError("Required columns must not contain missing values.")
    if set(data.group) != set(GROUPS):
        raise ValueError(f"Group labels must be exactly: {GROUPS}")
    values = data[["trial", "completion_time", "drops"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Trial numbers and outcomes must be finite numbers.")
    if (data.completion_time <= 0).any() or (data.drops < 0).any():
        raise ValueError("Times must be positive and drops nonnegative.")
    if not (data.drops == np.floor(data.drops)).all():
        raise ValueError("Drops must be integer counts.")
    if data.duplicated(["participant", "trial"]).any():
        raise ValueError("Duplicate participant/trial rows.")
    if not data.groupby("participant").group.nunique().eq(1).all():
        raise ValueError("Each participant must belong to exactly one group.")
    trials = data.groupby("participant").trial.apply(lambda x: set(x) == set(range(1, 11)))
    counts = data.groupby("group").participant.nunique()
    if len(data) != 400 or not trials.all() or not counts.eq(10).all():
        raise ValueError("Expected 40 participants, 10 per group, each with trials 1–10 only.")
    # Preserve the original ordering for seeded permutations: pandas infers the
    # study's numeric participant IDs as numbers, and groups have a fixed order.
    data["group"] = pd.Categorical(data.group, categories=GROUPS, ordered=True)
    return data


def mixed_anova(dat: pd.DataFrame, outcome: str) -> tuple[pd.DataFrame, float]:
    """Balanced 4 x 10 mixed ANOVA with exact split-plot sums of squares."""
    dat = dat.copy()
    dat["participant"] = dat["participant"].astype(str)
    wide = dat.pivot(index=["participant", "group"], columns="trial", values=outcome)
    y = wide.to_numpy(dtype=float)
    groups = np.asarray(wide.index.get_level_values("group"), dtype=str)
    n_subjects, n_trials = y.shape
    grand = y.mean()
    subject_means = y.mean(axis=1)
    trial_means = y.mean(axis=0)
    group_means = {g: y[groups == g].mean() for g in GROUPS}
    group_ns = {g: int(np.sum(groups == g)) for g in GROUPS}

    ss_group = n_trials * sum(
        group_ns[g] * (group_means[g] - grand) ** 2 for g in GROUPS
    )
    ss_subject_group = n_trials * sum(
        (subject_means[i] - group_means[groups[i]]) ** 2 for i in range(n_subjects)
    )
    ss_trial = n_subjects * np.sum((trial_means - grand) ** 2)
    ss_interaction = sum(
        group_ns[g]
        * np.sum((y[groups == g].mean(axis=0) - group_means[g] - trial_means + grand) ** 2)
        for g in GROUPS
    )
    ss_total = np.sum((y - grand) ** 2)
    ss_error = ss_total - ss_group - ss_subject_group - ss_trial - ss_interaction

    df_group = len(GROUPS) - 1
    df_subject_group = n_subjects - len(GROUPS)
    df_trial = n_trials - 1
    df_interaction = df_group * df_trial
    df_error = df_subject_group * df_trial

    f_group = (ss_group / df_group) / (ss_subject_group / df_subject_group)
    f_trial = (ss_trial / df_trial) / (ss_error / df_error)
    f_interaction = (ss_interaction / df_interaction) / (ss_error / df_error)

    # Greenhouse--Geisser epsilon from residual covariance after group means
    # have been removed separately at each trial.
    residuals = y.copy()
    for g in GROUPS:
        residuals[groups == g] -= y[groups == g].mean(axis=0)
    covariance = np.cov(residuals, rowvar=False, ddof=1)
    centering = np.eye(n_trials) - np.ones((n_trials, n_trials)) / n_trials
    projected = centering @ covariance @ centering
    epsilon = np.trace(projected) ** 2 / (
        (n_trials - 1) * np.trace(projected @ projected)
    )

    rows = [
        {
            "outcome": outcome,
            "effect": "group",
            "df1": df_group,
            "df2": df_subject_group,
            "F": f_group,
            "p": stats.f.sf(f_group, df_group, df_subject_group),
            "partial_eta_squared": ss_group / (ss_group + ss_subject_group),
            "gg_epsilon": np.nan,
        },
        {
            "outcome": outcome,
            "effect": "trial",
            "df1": df_trial * epsilon,
            "df2": df_error * epsilon,
            "F": f_trial,
            "p": stats.f.sf(f_trial, df_trial * epsilon, df_error * epsilon),
            "partial_eta_squared": ss_trial / (ss_trial + ss_error),
            "gg_epsilon": epsilon,
        },
        {
            "outcome": outcome,
            "effect": "group_x_trial",
            "df1": df_interaction * epsilon,
            "df2": df_error * epsilon,
            "F": f_interaction,
            "p": stats.f.sf(
                f_interaction, df_interaction * epsilon, df_error * epsilon
            ),
            "partial_eta_squared": ss_interaction / (ss_interaction + ss_error),
            "gg_epsilon": epsilon,
        },
    ]
    return pd.DataFrame(rows), float(epsilon)


def split_plot_statistics(y: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Return the group and group-by-trial F statistics."""
    n_subjects, n_trials = y.shape
    n_groups = len(GROUPS)
    grand = y.mean()
    subject_means = y.mean(axis=1)
    trial_means = y.mean(axis=0)
    group_means = np.array([y[labels == g].mean() for g in range(n_groups)])
    group_trial_means = np.array(
        [y[labels == g].mean(axis=0) for g in range(n_groups)]
    )

    ss_group = n_trials * sum(
        np.sum(labels == g) * (group_means[g] - grand) ** 2
        for g in range(n_groups)
    )
    ss_subject_group = n_trials * np.sum(
        (subject_means - group_means[labels]) ** 2
    )
    ss_trial = n_subjects * np.sum((trial_means - grand) ** 2)
    ss_interaction = sum(
        np.sum(labels == g)
        * np.sum(
            (
                group_trial_means[g]
                - group_means[g]
                - trial_means
                + grand
            )
            ** 2
        )
        for g in range(n_groups)
    )
    ss_total = np.sum((y - grand) ** 2)
    ss_error = ss_total - ss_group - ss_subject_group - ss_trial - ss_interaction

    df_group = n_groups - 1
    df_subject_group = n_subjects - n_groups
    df_interaction = (n_groups - 1) * (n_trials - 1)
    df_error = (n_subjects - n_groups) * (n_trials - 1)

    f_group = (ss_group / df_group) / (ss_subject_group / df_subject_group)
    f_interaction = (ss_interaction / df_interaction) / (ss_error / df_error)
    return float(f_group), float(f_interaction)


def permutation_test(
    data: pd.DataFrame,
    outcome: str,
    rng: np.random.Generator,
) -> list[dict[str, float | int | str]]:
    wide = data.pivot(index=["participant", "group"], columns="trial", values=outcome)
    y = wide.to_numpy(dtype=float)
    observed_groups = wide.index.get_level_values("group")
    observed_labels = np.array([GROUPS.index(group) for group in observed_groups])
    observed_group, observed_interaction = split_plot_statistics(y, observed_labels)

    exceed_group = 0
    exceed_interaction = 0
    balanced_labels = np.repeat(np.arange(len(GROUPS)), 10)
    for _ in range(N_PERMUTATIONS):
        permuted_labels = rng.permutation(balanced_labels)
        f_group, f_interaction = split_plot_statistics(y, permuted_labels)
        exceed_group += f_group >= observed_group
        exceed_interaction += f_interaction >= observed_interaction

    return [
        {
            "outcome": outcome,
            "effect": "alignment_condition",
            "observed_F": observed_group,
            "permutations": N_PERMUTATIONS,
            "permutation_p": (exceed_group + 1) / (N_PERMUTATIONS + 1),
        },
        {
            "outcome": outcome,
            "effect": "alignment_condition_x_trial",
            "observed_F": observed_interaction,
            "permutations": N_PERMUTATIONS,
            "permutation_p": (exceed_interaction + 1) / (N_PERMUTATIONS + 1),
        },
    ]

def jzs_bayes_factors(x: np.ndarray, y: np.ndarray, r: float) -> dict:
    """JZS independent-samples t-test, with positive effects defined as x > y.

    H0 fixes delta = 0. H+ assigns delta a positive half-Cauchy(0, r) prior.
    delta is the population mean difference divided by the common population SD.
    These are the same models as BayesFactor::ttestBF(nullInterval=c(0, Inf)).

    Under a fixed delta, the t statistic has a noncentral t distribution with
    noncentrality delta * sqrt(n_effective). Integrate that density over the
    effect prior, then divide by the central t density under H0. This implements
    the standard JZS test with the shared noninformative nuisance priors.
    Substituting delta = r*tan(theta) makes the half-Cauchy prior uniform on
    theta in (0, pi/2), avoiding integration over an infinite effect-size range.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (x.ndim != 1 or y.ndim != 1 or min(len(x), len(y)) < 2
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not np.isfinite(r) or r <= 0):
        raise ValueError("Expected two finite samples and a positive prior scale.")
    df = len(x) + len(y) - 2
    pooled_variance = ((len(x)-1)*x.var(ddof=1) + (len(y)-1)*y.var(ddof=1)) / df
    if pooled_variance <= 0:
        raise ValueError("The pooled variance must be positive.")
    n_effective = 1 / (1/len(x) + 1/len(y))
    t_value = (x.mean()-y.mean()) / np.sqrt(pooled_variance/n_effective)
    null_density = stats.t.pdf(t_value, df)

    def integrate_direction(sign: int) -> tuple[float, float]:
        def integrand(theta: float) -> float:
            noncentrality = sign * np.sqrt(n_effective) * r * np.tan(theta)
            return (2/np.pi) * stats.nct.pdf(t_value, df, noncentrality) / null_density

        value, error = quad(integrand, 0, np.pi/2, epsabs=1e-11, epsrel=1e-10)
        if not np.isfinite(value) or value <= 0 or error/value > 1e-7:
            raise ArithmeticError("Bayes-factor integration did not meet accuracy tolerance.")
        return value, error/value

    bf_plus0, positive_error = integrate_direction(1)
    bf_minus0, negative_error = integrate_direction(-1)
    # A symmetric Cauchy prior assigns equal mass to the two half-Cauchy priors.
    bf_10 = (bf_plus0 + bf_minus0) / 2
    return dict(BF0plus=1/bf_plus0, BF01=1/bf_10,
                quadrature_relative_error=max(positive_error, negative_error))


def bayesian_analyses(data: pd.DataFrame, out: Path) -> None:
    """Participant-average directional comparisons and sensitivity checks."""
    widths = np.unique(np.r_[np.linspace(.2, 1.5, 53), STANDARD_R, 1, np.sqrt(2)])
    rows = []
    for n_trials in [10, 5]:
        means = (data.loc[data.trial <= n_trials]
                 .groupby(["participant", "group"], observed=True).completion_time
                 .mean().reset_index())
        comparisons = GROUPS[1:] if n_trials == 10 else ["Z-misaligned"]
        for group in comparisons:
            for scale in ["seconds", "log_participant_mean"]:
                # Average in seconds first, then log the participant average.
                value = means.completion_time if scale == "seconds" else np.log(means.completion_time)
                x = value[means.group == group].to_numpy()
                y = value[means.group == "Aligned"].to_numpy()
                for r in widths:
                    result = jzs_bayes_factors(x, y, r)
                    rows.append(dict(trials=n_trials, scale=scale,
                                     comparison=f"{group} minus Aligned",
                                     n_misaligned=len(x), n_aligned=len(y), prior_r=r,
                                     mean_misaligned=x.mean(), mean_aligned=y.mean(),
                                     difference=x.mean()-y.mean(), BF0plus=result["BF0plus"],
                                     quadrature_relative_error=result["quadrature_relative_error"]))
    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(out / "bayesian_sensitivity.csv", index=False)
    main = sensitivity.loc[(sensitivity.scale == "seconds") &
                           np.isclose(sensitivity.prior_r, STANDARD_R, rtol=0, atol=1e-12)]
    main.to_csv(out / "bayesian_main.csv", index=False)
    print(main.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Trial-level CSV; see README for columns.")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    data = load_data(args.data)
    args.out.mkdir(parents=True, exist_ok=True)

    # One average per participant: trial observations are not independent people.
    means = data.groupby(["participant", "group"], observed=True)[["completion_time", "drops"]].mean()
    summary = means.groupby("group", observed=True).agg(
        n=("completion_time", "size"),
        completion_time_mean=("completion_time", "mean"),
        completion_time_sd=("completion_time", "std"),
        drops_mean=("drops", "mean"), drops_sd=("drops", "std"))
    summary.to_csv(args.out / "descriptives.csv")

    # The group effect uses between-participant error. Trial and interaction use
    # within-participant error, with Greenhouse–Geisser corrected df and p values.
    anova = pd.concat([mixed_anova(data, outcome)[0]
                       for outcome in ["completion_time", "drops"]], ignore_index=True)
    anova.to_csv(args.out / "mixed_anova.csv", index=False)
    print(anova.to_string(index=False))

    # Shuffle whole participant trajectories, preserving repeated measures and
    # the balanced group allocation. The +1 correction avoids zero p values.
    permutation = pd.DataFrame(permutation_test(data, "drops", np.random.default_rng(SEED)))
    permutation.to_csv(args.out / "drops_permutation.csv", index=False)
    print(permutation.to_string(index=False))
    bayesian_analyses(data, args.out)
    versions = dict(Python=platform.python_version(), numpy=np.__version__,
                    pandas=pd.__version__, scipy=scipy.__version__,
                    permutation_seed=SEED, permutations=N_PERMUTATIONS)
    (args.out / "python_versions.json").write_text(json.dumps(versions, indent=2) + "\n")


if __name__ == "__main__":
    main()
