#!/usr/bin/env python3
"""Exact and Monte Carlo sensitivity search for threshold-feasible Campy NBS.

The script distinguishes three questions:

1. Does any feasible package exist inside the stated perturbation bounds?
2. How often does the original unconditional Monte Carlo design find one?
3. Conditional on a specified package being feasible, which package is the NBS?

Issue weights vary inside relative bounds, remain nonnegative, and sum to the
stakeholder's original total. Option scores are scaled proportionally within
each positive-weight issue. An originally zero-weight issue receives [0, 1]
and keeps all of its options tied at the new issue weight, matching the prior
adaptation script.

Perturbation conventions are shared with table2_proportional_floor.py: the
per-issue tolerance is max(--weight-floor, cap * weight), each stakeholder's
total budget is preserved, weights are drawn by uniform hit-and-run, and
within-issue option ratios, rankings, ties and structural zeros are preserved.

The acceptance threshold and the no-agreement payoff are the same quantity in
this framework. Unless --disagreement-point is given explicitly, the Nash
product in each test therefore measures gains above that test's own acceptance
threshold, and above each draw's own threshold when the threshold varies.
Pinning the disagreement point above the feasibility threshold clamps every
gain to zero and makes the arg-max return the lowest-index feasible package
rather than a bargaining solution.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Problem:
    stakeholders: list[str]
    issues: list[str]
    options_by_issue: dict[str, list[str]]
    packages: list[tuple[str, ...]]
    package_labels: list[str]
    original_weights: dict[str, np.ndarray]
    ratio_matrices: dict[str, np.ndarray]
    baseline_utilities: np.ndarray


def parse_problem(path: Path, separator: str = ";") -> Problem:
    df = pd.read_csv(path, sep=separator)
    if "Stakeholder" not in df.columns:
        raise ValueError("Input must contain a Stakeholder column.")
    if df["Stakeholder"].duplicated().any():
        duplicates = df.loc[df["Stakeholder"].duplicated(), "Stakeholder"].tolist()
        raise ValueError(f"Duplicate stakeholder rows: {duplicates}")

    option_columns = [
        str(column)
        for column in df.columns
        if re.fullmatch(r"[A-Za-z]+\d+", str(column))
    ]
    if not option_columns:
        raise ValueError("No issue-option columns such as A1 or B2 were found.")
    if df[option_columns].isna().any().any():
        raise ValueError("Score matrix contains missing option scores.")
    if (df[option_columns] < 0).any().any():
        raise ValueError("Scores must be nonnegative.")

    issues: list[str] = []
    options_by_issue: dict[str, list[str]] = {}
    for option in option_columns:
        issue = re.match(r"^([A-Za-z]+)", option).group(1)
        if issue not in options_by_issue:
            issues.append(issue)
            options_by_issue[issue] = []
        options_by_issue[issue].append(option)

    packages = list(itertools.product(*(options_by_issue[i] for i in issues)))
    labels = [",".join(package) for package in packages]
    stakeholders = df["Stakeholder"].astype(str).tolist()
    original_weights: dict[str, np.ndarray] = {}
    ratio_matrices: dict[str, np.ndarray] = {}
    baseline_utilities = np.empty((len(packages), len(stakeholders)), dtype=float)

    for stakeholder_index, stakeholder in enumerate(stakeholders):
        row = df.loc[df["Stakeholder"].eq(stakeholder)].iloc[0]
        weights = np.array(
            [float(row[options_by_issue[issue]].max()) for issue in issues],
            dtype=float,
        )
        if weights.sum() <= 0:
            raise ValueError(f"{stakeholder} has no positive issue weight.")
        original_weights[stakeholder] = weights

        ratios = np.empty((len(packages), len(issues)), dtype=float)
        for package_index, package in enumerate(packages):
            for issue_index, option in enumerate(package):
                weight = weights[issue_index]
                score = float(row[option])
                ratios[package_index, issue_index] = (
                    1.0 if math.isclose(weight, 0.0, abs_tol=1e-12)
                    else score / weight
                )
            baseline_utilities[package_index, stakeholder_index] = sum(
                float(row[option]) for option in package
            )
        ratio_matrices[stakeholder] = ratios

    return Problem(
        stakeholders=stakeholders,
        issues=issues,
        options_by_issue=options_by_issue,
        packages=packages,
        package_labels=labels,
        original_weights=original_weights,
        ratio_matrices=ratio_matrices,
        baseline_utilities=baseline_utilities,
    )


def bounds_for_weights(
    original: np.ndarray,
    relative_cap: float,
    minimum_half_width: float = 1.0,
    zero_upper: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    half_width = np.maximum(relative_cap * original, minimum_half_width)
    lower = np.where(original <= 1e-12, 0.0, np.maximum(0.0, original - half_width))
    upper = np.where(original <= 1e-12, zero_upper, original + half_width)
    total = float(original.sum())
    if lower.sum() > total + 1e-9 or upper.sum() < total - 1e-9:
        raise ValueError("Weight bounds do not intersect the normalization plane.")
    return lower, upper


def linear_extreme_on_bounded_simplex(
    coefficients: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    total: float,
    maximize: bool,
) -> tuple[float, np.ndarray]:
    """Exact linear optimum by filling coordinates in coefficient order."""
    point = lower.astype(float).copy()
    remainder = total - float(point.sum())
    order = np.argsort(coefficients)
    if maximize:
        order = order[::-1]
    for index in order:
        addition = min(remainder, float(upper[index] - point[index]))
        point[index] += addition
        remainder -= addition
        if remainder <= 1e-10:
            break
    if abs(remainder) > 1e-7:
        raise RuntimeError("Could not fill the bounded simplex to its target total.")
    return float(coefficients @ point), point


def exact_cap_result(problem: Problem, cap: float, weight_floor: float = 1.0) -> dict:
    number_packages = len(problem.packages)
    maxima = np.empty((number_packages, len(problem.stakeholders)), dtype=float)
    maximizing_weights: dict[str, np.ndarray] = {}

    for stakeholder_index, stakeholder in enumerate(problem.stakeholders):
        original = problem.original_weights[stakeholder]
        lower, upper = bounds_for_weights(original, cap, weight_floor)
        ratio_matrix = problem.ratio_matrices[stakeholder]
        stakeholder_weights = np.empty((number_packages, len(problem.issues)))
        for package_index in range(number_packages):
            value, weights = linear_extreme_on_bounded_simplex(
                ratio_matrix[package_index], lower, upper, float(original.sum()), True
            )
            maxima[package_index, stakeholder_index] = value
            stakeholder_weights[package_index] = weights
        maximizing_weights[stakeholder] = stakeholder_weights

    attainable_common = maxima.min(axis=1)
    best_index = int(np.argmax(attainable_common))
    witness = {
        stakeholder: maximizing_weights[stakeholder][best_index].tolist()
        for stakeholder in problem.stakeholders
    }
    witness_utilities = maxima[best_index].tolist()
    return {
        "relative_cap": cap,
        "best_package_index": best_index,
        "best_package": problem.package_labels[best_index],
        "highest_attainable_common_threshold": float(attainable_common[best_index]),
        "stakeholder_maxima_for_best_package": dict(
            zip(problem.stakeholders, map(float, witness_utilities))
        ),
        "maximizing_weights": witness,
    }


def critical_cap_for_threshold(
    problem: Problem,
    threshold: float,
    lower_cap: float = 0.0,
    upper_cap: float = 2.0,
    iterations: int = 45,
    weight_floor: float = 1.0,
) -> dict:
    low = lower_cap
    high = upper_cap
    high_result = exact_cap_result(problem, high, weight_floor)
    if high_result["highest_attainable_common_threshold"] < threshold - 1e-9:
        return {"exists": False, "threshold": threshold}
    for _ in range(iterations):
        middle = (low + high) / 2.0
        result = exact_cap_result(problem, middle, weight_floor)
        if result["highest_attainable_common_threshold"] >= threshold:
            high = middle
            high_result = result
        else:
            low = middle
    high_result = dict(high_result)
    high_result.update({"exists": True, "threshold": threshold, "critical_cap": high})
    return high_result


def hit_and_run(
    start: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    total: float,
    n_samples: int,
    rng: np.random.Generator,
    burn_in: int,
    thinning: int,
    utility_coefficients: np.ndarray | None = None,
    minimum_utility: float | None = None,
) -> np.ndarray:
    """Sample a bounded simplex, optionally conditioned on one utility inequality."""
    current = np.asarray(start, dtype=float).copy()
    if not np.isclose(current.sum(), total, atol=1e-7):
        raise ValueError("Hit-and-run starting point is not normalized.")
    if np.any(current < lower - 1e-8) or np.any(current > upper + 1e-8):
        raise ValueError("Hit-and-run starting point violates weight bounds.")
    if utility_coefficients is not None:
        if minimum_utility is None:
            raise ValueError("minimum_utility is required for conditional sampling.")
        if float(utility_coefficients @ current) < minimum_utility - 1e-7:
            raise ValueError("Conditional hit-and-run start is not utility feasible.")

    draws = np.empty((n_samples, len(current)), dtype=float)
    draw_index = 0
    total_steps = burn_in + n_samples * thinning
    for step in range(total_steps):
        direction = rng.normal(size=len(current))
        direction -= direction.mean()
        norm = np.linalg.norm(direction)
        if norm <= 1e-14:
            continue
        direction /= norm
        step_low = -np.inf
        step_high = np.inf
        for value, delta, low_bound, high_bound in zip(
            current, direction, lower, upper
        ):
            if abs(delta) <= 1e-14:
                continue
            first = (low_bound - value) / delta
            second = (high_bound - value) / delta
            step_low = max(step_low, min(first, second))
            step_high = min(step_high, max(first, second))

        if utility_coefficients is not None:
            current_utility = float(utility_coefficients @ current)
            utility_direction = float(utility_coefficients @ direction)
            if abs(utility_direction) <= 1e-14:
                if current_utility < minimum_utility - 1e-9:
                    raise RuntimeError("Conditional chain left its feasible half-space.")
            else:
                crossing = (minimum_utility - current_utility) / utility_direction
                if utility_direction > 0:
                    step_low = max(step_low, crossing)
                else:
                    step_high = min(step_high, crossing)

        if step_low > step_high + 1e-10:
            raise RuntimeError("Numerically empty hit-and-run interval.")
        if step_high - step_low <= 1e-12:
            midpoint = (step_low + step_high) / 2.0
            step_low = midpoint
            step_high = midpoint
        move = step_low if step_low == step_high else rng.uniform(step_low, step_high)
        current += move * direction
        if step >= burn_in and (step - burn_in) % thinning == 0:
            draws[draw_index] = current
            draw_index += 1
            if draw_index == n_samples:
                break
    if draw_index != n_samples:
        raise RuntimeError(f"Generated {draw_index} of {n_samples} requested draws.")
    return draws


def evaluate_draws(
    problem: Problem,
    weights_by_stakeholder: dict[str, np.ndarray],
    thresholds: np.ndarray,
    disagreement_point,
    chunk_size: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples = len(thresholds)
    # The disagreement point may be a scalar or one value per draw. When the
    # acceptance threshold is treated as a sensitivity parameter it IS the
    # no-agreement payoff, so it must enter the Nash product as well; holding it
    # at a value above the threshold clamps every gain to zero and makes the
    # arg-max meaningless.
    disagreement = np.broadcast_to(
        np.asarray(disagreement_point, dtype=float), (n_samples,)
    )
    any_feasible = np.zeros(n_samples, dtype=bool)
    feasible_counts = np.zeros(n_samples, dtype=int)
    winner_indices = np.full(n_samples, -1, dtype=int)
    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        size = stop - start
        feasible = np.ones((size, len(problem.packages)), dtype=bool)
        nash = np.ones((size, len(problem.packages)), dtype=float)
        for stakeholder in problem.stakeholders:
            utilities = (
                weights_by_stakeholder[stakeholder][start:stop]
                @ problem.ratio_matrices[stakeholder].T
            )
            feasible &= utilities >= thresholds[start:stop, None]
            nash *= np.maximum(utilities - disagreement[start:stop, None], 0.0)
        finite_nash = np.where(feasible, nash, -np.inf)
        chunk_any = feasible.any(axis=1)
        any_feasible[start:stop] = chunk_any
        feasible_counts[start:stop] = feasible.sum(axis=1)
        if chunk_any.any():
            local_rows = np.flatnonzero(chunk_any)
            winner_indices[start + local_rows] = np.argmax(
                finite_nash[local_rows], axis=1
            )
    return any_feasible, feasible_counts, winner_indices


def run_uniform_monte_carlo(
    problem: Problem,
    cap: float,
    threshold_center: float,
    threshold_variation: float,
    samples: int,
    seed: int,
    burn_in: int,
    thinning: int,
    disagreement_point: float,
    weight_floor: float = 1.0,
) -> dict:
    rng = np.random.default_rng(seed)
    low_threshold = threshold_center * (1.0 - threshold_variation)
    high_threshold = threshold_center * (1.0 + threshold_variation)
    thresholds = rng.uniform(low_threshold, high_threshold, size=samples)
    if disagreement_point is None:
        disagreement_point = thresholds
    weights: dict[str, np.ndarray] = {}
    for stakeholder in problem.stakeholders:
        original = problem.original_weights[stakeholder]
        lower, upper = bounds_for_weights(original, cap, weight_floor)
        weights[stakeholder] = hit_and_run(
            original, lower, upper, float(original.sum()), samples, rng, burn_in, thinning
        )
    any_feasible, feasible_counts, winners = evaluate_draws(
        problem, weights, thresholds, disagreement_point
    )
    winner_counts = pd.Series(winners[winners >= 0]).value_counts().to_dict()
    return {
        "method": "unconditional_uniform_hit_and_run",
        "relative_cap": cap,
        "threshold_center": threshold_center,
        "threshold_relative_variation": threshold_variation,
        "threshold_interval": [low_threshold, high_threshold],
        "samples": samples,
        "disagreement_point": ("follows the per-draw acceptance threshold"
                               if np.ndim(disagreement_point) else float(disagreement_point)),
        "iterations_with_any_feasible_package": int(any_feasible.sum()),
        "probability_any_feasible_package": float(any_feasible.mean()),
        "maximum_feasible_package_count_in_one_iteration": int(feasible_counts.max()),
        "nbs_counts": {
            problem.package_labels[int(index)]: int(count)
            for index, count in winner_counts.items()
        },
    }


def run_conditional_monte_carlo(
    problem: Problem,
    cap_result: dict,
    threshold: float,
    samples: int,
    seed: int,
    burn_in: int,
    thinning: int,
    disagreement_point: float,
    weight_floor: float = 1.0,
) -> dict:
    cap = float(cap_result["relative_cap"])
    package_index = int(cap_result["best_package_index"])
    if cap_result["highest_attainable_common_threshold"] < threshold - 1e-8:
        return {
            "method": "conditional_on_frontier_package_feasibility",
            "relative_cap": cap,
            "threshold": threshold,
            "possible": False,
            "reason": "Threshold exceeds exact attainable frontier.",
        }
    rng = np.random.default_rng(seed)
    weights: dict[str, np.ndarray] = {}
    for stakeholder in problem.stakeholders:
        original = problem.original_weights[stakeholder]
        lower, upper = bounds_for_weights(original, cap, weight_floor)
        coefficients = problem.ratio_matrices[stakeholder][package_index]
        start = np.asarray(cap_result["maximizing_weights"][stakeholder], dtype=float)
        weights[stakeholder] = hit_and_run(
            start,
            lower,
            upper,
            float(original.sum()),
            samples,
            rng,
            burn_in,
            thinning,
            utility_coefficients=coefficients,
            minimum_utility=threshold,
        )
    thresholds = np.full(samples, threshold, dtype=float)
    if disagreement_point is None:
        disagreement_point = threshold
    any_feasible, feasible_counts, winners = evaluate_draws(
        problem, weights, thresholds, disagreement_point
    )
    winner_counts = pd.Series(winners[winners >= 0]).value_counts().to_dict()
    return {
        "method": "conditional_on_frontier_package_feasibility",
        "relative_cap": cap,
        "conditioning_package": problem.package_labels[package_index],
        "threshold": threshold,
        "possible": True,
        "samples": samples,
        "disagreement_point": disagreement_point,
        "iterations_with_any_feasible_package": int(any_feasible.sum()),
        "minimum_feasible_package_count_in_one_iteration": int(feasible_counts.min()),
        "maximum_feasible_package_count_in_one_iteration": int(feasible_counts.max()),
        "nbs_counts": {
            problem.package_labels[int(index)]: int(count)
            for index, count in winner_counts.items()
        },
        "interpretation": (
            "Conditional search establishes and explores existence; its NBS frequencies "
            "are not unconditional probabilities under the original perturbation prior."
        ),
    }


def baseline_summary(
    problem: Problem, threshold: float, disagreement_point: float = 0.0
) -> dict:
    """Unperturbed characterization of the outcome space.

    The Nash product is measured above the disagreement point, matching the
    disagreement-adjusted product used in the Table 2 analysis. When no package
    clears the threshold for every stakeholder, no disagreement-adjusted Nash
    bargaining solution exists: every gain vector has a nonpositive component,
    so every product is zero and an argmax would be meaningless. In that case
    the maximizer is reported as absent, together with the infeasibility gap
    min_d max_s (threshold - U_s(d)), the number of utility points by which the
    preference structure misses feasibility.
    """
    utilities = problem.baseline_utilities
    feasible = np.all(utilities >= threshold, axis=1)
    gaps = np.max(threshold - utilities, axis=1)
    infeasibility_gap = float(gaps.min())
    binding_index = int(np.argmin(gaps))
    max_min_index = int(np.argmax(utilities.min(axis=1)))

    summary = {
        "stakeholders": problem.stakeholders,
        "issues": problem.issues,
        "number_of_packages": len(problem.packages),
        "issue_weights": {
            stakeholder: dict(
                zip(problem.issues, map(float, problem.original_weights[stakeholder]))
            )
            for stakeholder in problem.stakeholders
        },
        "threshold": threshold,
        "disagreement_point": disagreement_point,
        "feasible_package_count": int(feasible.sum()),
        "infeasibility_gap": infeasibility_gap,
        "closest_to_feasible_package": problem.package_labels[binding_index],
        "binding_stakeholders": [
            problem.stakeholders[i]
            for i in np.flatnonzero(
                threshold - utilities[binding_index] >= infeasibility_gap - 1e-9
            )
        ],
        "baseline_maximin_package": problem.package_labels[max_min_index],
        "baseline_maximin_utilities": dict(
            zip(problem.stakeholders, map(float, utilities[max_min_index]))
        ),
        "baseline_highest_common_utility": float(utilities[max_min_index].min()),
    }

    if feasible.any():
        gains = np.maximum(utilities - disagreement_point, 0.0)
        nash = np.prod(np.where(feasible[:, None], gains, 0.0), axis=1)
        nbs_index = int(np.argmax(nash))
        summary.update({
            "nash_bargaining_solution_exists": True,
            "nash_bargaining_solution": problem.package_labels[nbs_index],
            "nash_bargaining_solution_utilities": dict(
                zip(problem.stakeholders, map(float, utilities[nbs_index]))
            ),
        })
    else:
        summary.update({
            "nash_bargaining_solution_exists": False,
            "nash_bargaining_solution": None,
            "nash_bargaining_solution_utilities": None,
            "note": (
                "No package is individually rational at this threshold, so no "
                "disagreement-adjusted Nash bargaining solution is defined."
            ),
        })

    # Reported for continuity with the legacy analysis only: the maximizer of the
    # product of raw utilities, which ignores the disagreement point entirely.
    raw_index = int(np.argmax(np.prod(np.maximum(utilities, 0.0), axis=1)))
    summary["legacy_raw_utility_nash_maximizer"] = problem.package_labels[raw_index]
    summary["legacy_raw_utility_nash_maximizer_utilities"] = dict(
        zip(problem.stakeholders, map(float, utilities[raw_index]))
    )
    return summary


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--separator", default=";")
    parser.add_argument("--threshold", type=float, default=65.0)
    parser.add_argument(
        "--disagreement-point",
        type=float,
        default=None,
        help=(
            "No-agreement payoff used in the Nash product. Left unset it follows the "
            "acceptance threshold of each individual test, which is the coherent "
            "choice because the two are the same quantity. Fixing it detaches the "
            "Nash benchmark from the feasibility test. "
            "matching the disagreement-adjusted Nash product of the Table 2 analysis. "
            "Pass 0 for the legacy raw-utility product."
        ),
    )
    parser.add_argument(
        "--weight-floor",
        type=float,
        default=1.0,
        help=(
            "Absolute floor on the issue-weight half-width, in points, so the "
            "tolerance is max(weight_floor, cap * weight) as in Table 2."
        ),
    )
    parser.add_argument("--caps", default="0.10,0.15,0.20,0.25,0.30")
    parser.add_argument(
        "--threshold-variation",
        type=float,
        default=0.10,
        help=(
            "Relative variation of the shared acceptance threshold, held independent "
            "of the issue-weight cap. Used for the default --uniform-tests and for "
            "the exploratory threshold of the conditional search."
        ),
    )
    parser.add_argument(
        "--uniform-tests",
        default=None,
        help=(
            "Comma-separated issue_cap:threshold_relative_variation pairs. "
            "Defaults to every cap in --caps paired with --threshold-variation."
        ),
    )
    parser.add_argument(
        "--fixed-tests",
        default="0.10:58.5,0.15:60.0,0.20:63.0,0.25:65.0,0.30:65.0",
        help="Comma-separated issue_cap:fixed_common_threshold pairs.",
    )
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--conditional-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--burn-in", type=int, default=500)
    parser.add_argument("--thinning", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("campy_nbs_results"))
    args = parser.parse_args()

    problem = parse_problem(args.input_csv, args.separator)
    caps = parse_float_list(args.caps)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # The no-agreement payoff and the acceptance threshold are the same quantity
    # in the negotiation design, so the Nash product is measured above the
    # threshold unless the legacy raw-utility product is requested explicitly.
    # args.disagreement_point stays None so that each test uses its own
    # acceptance threshold as the no-agreement payoff.

    # Threshold variation is a controlled parameter, not a function of the cap.
    if args.uniform_tests is None:
        args.uniform_tests = ",".join(
            f"{cap}:{args.threshold_variation}" for cap in caps
        )

    baseline = baseline_summary(
        problem, args.threshold,
        args.threshold if args.disagreement_point is None else args.disagreement_point)
    cap_results = [
        exact_cap_result(problem, cap, args.weight_floor) for cap in caps
    ]
    for result in cap_results:
        result["minimum_threshold_reduction_from_central"] = max(
            0.0,
            1.0 - result["highest_attainable_common_threshold"] / args.threshold,
        )
    critical = critical_cap_for_threshold(
        problem, args.threshold, weight_floor=args.weight_floor
    )

    uniform_results = []
    for offset, specification in enumerate(args.uniform_tests.split(",")):
        if not specification.strip():
            continue
        cap_text, variation_text = specification.split(":", maxsplit=1)
        uniform_results.append(
            run_uniform_monte_carlo(
                problem,
                float(cap_text),
                args.threshold,
                float(variation_text),
                args.samples,
                args.seed + offset,
                args.burn_in,
                args.thinning,
                args.disagreement_point,
                args.weight_floor,
            )
        )

    for offset, specification in enumerate(args.fixed_tests.split(",")):
        if not specification.strip():
            continue
        cap_text, threshold_text = specification.split(":", maxsplit=1)
        fixed_result = run_uniform_monte_carlo(
            problem,
            float(cap_text),
            float(threshold_text),
            0.0,
            args.samples,
            args.seed + 50 + offset,
            args.burn_in,
            args.thinning,
            args.disagreement_point,
            args.weight_floor,
        )
        fixed_result["method"] = "unconditional_uniform_hit_and_run_fixed_threshold"
        uniform_results.append(fixed_result)

    conditional_results = []
    for offset, cap_result in enumerate(cap_results):
        cap = float(cap_result["relative_cap"])
        # Lower edge of the same shared-threshold interval used everywhere else,
        # rather than a variation tied to the issue-weight cap.
        exploratory_threshold = args.threshold * (1.0 - args.threshold_variation)
        if cap_result["highest_attainable_common_threshold"] >= exploratory_threshold:
            conditional_results.append(
                run_conditional_monte_carlo(
                    problem,
                    cap_result,
                    exploratory_threshold,
                    args.conditional_samples,
                    args.seed + 100 + offset,
                    args.burn_in,
                    args.thinning,
                    args.disagreement_point,
                    args.weight_floor,
                )
            )
        if cap_result["highest_attainable_common_threshold"] >= args.threshold:
            conditional_results.append(
                run_conditional_monte_carlo(
                    problem,
                    cap_result,
                    args.threshold,
                    args.conditional_samples,
                    args.seed + 200 + offset,
                    args.burn_in,
                    args.thinning,
                    args.disagreement_point,
                    args.weight_floor,
                )
            )

    report = {
        "input_file": str(args.input_csv),
        "disagreement_point": (
            "follows each test's acceptance threshold"
            if args.disagreement_point is None else args.disagreement_point),
        "acceptance_threshold": args.threshold,
        "issue_weight_floor_points": args.weight_floor,
        "threshold_relative_variation": args.threshold_variation,
        "issue_weight_tolerance": "max(weight_floor, cap * issue weight), budget preserved",
        "zero_weight_issue_handling": "redrawn on [0, 1]; all options take the perturbed issue maximum",
        "baseline": baseline,
        "exact_issue_weight_frontier": cap_results,
        "critical_issue_cap_at_central_threshold": critical,
        "unconditional_monte_carlo": uniform_results,
        "conditional_feasibility_search": conditional_results,
        "method_notes": [
            "Exact frontier results are deterministic and cannot be improved by more MC draws.",
            "Unconditional Monte Carlo estimates prevalence under uniform bounded-simplex weights and a uniform common threshold.",
            "Conditional search samples only profiles where the frontier package clears the selected common threshold.",
            "Conditional NBS frequencies demonstrate possible outcomes but are not unconditional probabilities.",
            "Nash gains are measured above the disagreement point, which defaults to the acceptance threshold.",
            "The shared threshold varies by +/- threshold_relative_variation independently of the issue-weight cap.",
        ],
    }

    json_path = args.output_dir / "campy_nbs_strategy_results.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    frontier_rows = []
    for result in cap_results:
        row = {
            "Issue_weight_relative_cap": result["relative_cap"],
            "Best_package": result["best_package"],
            "Highest_attainable_common_threshold": result[
                "highest_attainable_common_threshold"
            ],
            "Minimum_threshold_reduction_from_65": result[
                "minimum_threshold_reduction_from_central"
            ],
        }
        row.update(
            {
                f"Maximum_{stakeholder}": value
                for stakeholder, value in result[
                    "stakeholder_maxima_for_best_package"
                ].items()
            }
        )
        frontier_rows.append(row)
    pd.DataFrame(frontier_rows).to_csv(
        args.output_dir / "exact_feasibility_frontier.csv", index=False
    )

    if critical.get("exists"):
        critical_cap = float(critical["critical_cap"])
        critical_package_index = int(critical["best_package_index"])
        witness_rows = []
        for stakeholder in problem.stakeholders:
            original = problem.original_weights[stakeholder]
            lower, upper = bounds_for_weights(original, critical_cap, args.weight_floor)
            adapted = np.asarray(
                critical["maximizing_weights"][stakeholder], dtype=float
            )
            utility = float(
                problem.ratio_matrices[stakeholder][critical_package_index]
                @ adapted
            )
            for issue_index, issue in enumerate(problem.issues):
                witness_rows.append(
                    {
                        "Threshold": args.threshold,
                        "Critical_relative_cap": critical_cap,
                        "Frontier_package": critical["best_package"],
                        "Stakeholder": stakeholder,
                        "Issue": issue,
                        "Original_weight": original[issue_index],
                        "Lower_bound": lower[issue_index],
                        "Adapted_weight": adapted[issue_index],
                        "Upper_bound": upper[issue_index],
                        "Stakeholder_package_utility": utility,
                    }
                )
        pd.DataFrame(witness_rows).to_csv(
            args.output_dir / "critical_threshold_witness.csv", index=False
        )

    mc_rows = []
    for result in uniform_results:
        mc_rows.append(
            {
                key: value
                for key, value in result.items()
                if key != "nbs_counts"
            }
            | {"nbs_counts_json": json.dumps(result["nbs_counts"], sort_keys=True)}
        )
    pd.DataFrame(mc_rows).to_csv(
        args.output_dir / "unconditional_mc_summary.csv", index=False
    )

    conditional_rows = []
    for result in conditional_results:
        conditional_rows.append(
            {
                key: value
                for key, value in result.items()
                if key not in {"nbs_counts", "interpretation"}
            }
            | {"nbs_counts_json": json.dumps(result.get("nbs_counts", {}), sort_keys=True)}
        )
    pd.DataFrame(conditional_rows).to_csv(
        args.output_dir / "conditional_nbs_summary.csv", index=False
    )

    print(json.dumps({
        "baseline": baseline,
        "frontier": [
            {
                "cap": result["relative_cap"],
                "package": result["best_package"],
                "highest_common_threshold": result[
                    "highest_attainable_common_threshold"
                ],
            }
            for result in cap_results
        ],
        "critical_cap_for_threshold": critical.get("critical_cap"),
        "unconditional_mc": uniform_results,
        "conditional_search": conditional_results,
        "output": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
