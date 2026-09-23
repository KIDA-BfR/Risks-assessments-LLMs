#!/usr/bin/env python3
"""Table-2 robustness analysis: proportional issue-weight caps with an absolute
floor, plus perturbation of the disagreement (acceptance) threshold.

Perturbation of issue weights
-----------------------------
For stakeholder p and issue i with baseline issue weight w_pi, the admissible
interval is set proportionally with an absolute floor:

    c_pi   = max(FLOOR_PTS, FRAC * w_pi)                 if w_pi > 0
             (reported specification: FLOOR_PTS = 1 point, FRAC = 0.25)
    w'_pi in [max(0, w_pi - c_pi), w_pi + c_pi]          if w_pi > 0
    w'_pi in [0, ZERO_UPPER]                             if w_pi = 0

A provisional shift is drawn uniformly on each issue's own interval and the
resulting vector is projected onto the feasible set that keeps the
stakeholder's total budget (normally 100 points) and every issue inside its
interval. Confidence therefore scales with the stated weight: issue-importance
reversals remain possible only between issues whose intervals overlap, i.e.
where the elicitation never identified the ordering, and are impossible between
issues the stakeholder clearly separated.

A baseline weight of 0 is treated as "could plausibly have been 1" and is given
the one-sided interval [0, ZERO_UPPER]. Because all options in such an issue
were scored 0, no within-issue preference was expressed; every option is
therefore assigned the perturbed issue maximum (ratio 1.0), which is what the
scoring rules prescribe for indifference. Such an issue shifts a stakeholder's
utility level without altering their ranking of packages.

All option scores inside a non-zero issue are rescaled proportionally, so
within-issue preference ratios, rankings, ties, and structural zeros are
preserved exactly.

Perturbation of the acceptance threshold
----------------------------------------
The disagreement point r (65 utility points in the main analysis) is a single
acceptance threshold agreed for all participants, so it is perturbed as one
shared draw rather than independently per stakeholder:

    u ~ U(-THRESHOLD_FRAC, +THRESHOLD_FRAC),   r'_p = r_p * (1 + u) for every p

Every stakeholder therefore moves together and the common threshold remains
common in each sample. Because a raised threshold can empty the individually
rational set, samples in which no NBS exists are recorded rather than raising:
the analysis reports how often that happens and the infeasibility gap
min_d max_p (r'_p - U_p(d)) for those samples.

Outputs include Table2_values.csv, Table2_sensitivity.tex/.pdf,
mc_weight_adjustments.csv, mc_threshold_draws.csv, target_deal_mc_samples.csv,
mc_summary.csv, order_reversals.csv, and the alternative-NBS distribution.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import re
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import negotiation_sensitivity_core_promptzip as core

DEAL_TAG_RE = re.compile(r"<DEAL>\s*([^<]+?)\s*</DEAL>", re.I | re.S)
GENERIC_DEAL_RE = re.compile(r"\b[A-Za-z]+\d+(?:\s*,\s*[A-Za-z]+\d+){1,}\b")


def normalize_deal(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).upper()


def deal_tokens(deal: str) -> List[str]:
    return [x.strip().upper() for x in str(deal).split(",") if x.strip()]


def hamming_count(a: str, b: str) -> int:
    aa, bb = deal_tokens(a), deal_tokens(b)
    if len(aa) != len(bb):
        raise ValueError(f"Deal lengths differ: {a!r} vs {b!r}")
    return int(sum(x != y for x, y in zip(aa, bb)))


ANSWER_BLOCK_RE = re.compile(r"<ANSWER>(.*?)(?:</ANSWER>|\Z)", re.S | re.I)
SCRATCHPAD_BLOCK_RE = re.compile(r"<SCRATCHPAD>(.*?)(?:</SCRATCHPAD>|\Z)", re.S | re.I)


def answer_body(text: str) -> str:
    """Public part of a deal_suggestion response.

    A no-op for the answers*.json files, whose deal_suggestion is already the
    public text alone. Retained so full_conversation*.json can also be parsed:

    there, the agent prompt states that the scratchpad is secret and not seen by
    other parties, so <DEAL> tags inside <SCRATCHPAD> are private reasoning
    rather than proposals and must not be counted. When no <ANSWER> block is
    present the scratchpad is stripped and the remainder used.
    """
    if not text:
        return ""
    m = ANSWER_BLOCK_RE.search(str(text))
    if m:
        return m.group(1)
    return SCRATCHPAD_BLOCK_RE.sub("", str(text))


def extract_answer_deals(text: str) -> List[str]:
    """Every distinct deal put forward in the public answer, in order.

    The first entry is the primary proposal. Later entries are the fallbacks or
    compromises the proposer also signalled willingness to accept, e.g. "I
    propose X; if X is not accepted I will support Y". Repetitions of the same
    deal within one answer collapse to a single entry, so a deal restated for
    emphasis is not double-counted.
    """
    body = answer_body(text)
    if not body:
        return []
    found = DEAL_TAG_RE.findall(body) or GENERIC_DEAL_RE.findall(body)
    deals, seen = [], set()
    for raw in found:
        d = normalize_deal(raw)
        if d and d not in seen:
            seen.add(d)
            deals.append(d)
    return deals


def extract_last_deal(text: str) -> Optional[str]:
    """Legacy single-deal rule, retained for --deal-selection last."""
    deals = extract_answer_deals(text)
    return deals[-1] if deals else None


ANSWERS_STEM_RE = re.compile(r"^answers", re.I)
FULL_STEM_RE = re.compile(r"^full_conversation", re.I)


def _zip_members(zf: zipfile.ZipFile, source: str):
    """Log members to parse, and the source actually used.

    `answers` files hold only the public response text, so no private reasoning
    is present to be mistakenly parsed. `full_conversation` files hold the
    scratchpad as well and require the <ANSWER> block to be isolated first.
    With source="auto" the answers files are used when available.
    """
    answers = sorted(n for n in zf.namelist()
                     if ANSWERS_STEM_RE.match(Path(n).name) and n.lower().endswith(".json"))
    full = sorted(n for n in zf.namelist()
                  if FULL_STEM_RE.match(Path(n).name) and n.lower().endswith(".json"))
    if source == "answers":
        if not answers:
            raise ValueError("No answers*.json members found in the logs zip")
        return answers, "answers"
    if source == "full_conversation":
        if not full:
            raise ValueError("No full_conversation*.json members found in the logs zip")
        return full, "full_conversation"
    if answers:
        return answers, "answers"
    if not full:
        raise ValueError("Logs zip contains neither answers*.json nor full_conversation*.json")
    return full, "full_conversation"


def extract_final_deals_from_logs_zip(log_zip: str | Path,
                                      selection: str = "all",
                                      source: str = "auto") -> tuple:
    """Final proposals from every run in a logs zip.

    selection:
      all     - every distinct deal in the public answer (primary + fallbacks)
      primary - the first deal only
      last    - the last deal only, reproducing the original behaviour

    Returns (counts, per_deal, info). `counts` has one row per distinct deal with
    its number of occurrences, ordered by descending count and then by deal so
    the ordering is reproducible rather than dependent on zip member order.
    `per_deal` is the long record of one row per extracted deal. `info` reports
    which source was parsed and whether any private reasoning was encountered.
    """
    records = []
    scratchpad_seen = 0
    with zipfile.ZipFile(log_zip, "r") as zf:
        names, used = _zip_members(zf, source)
        for name in names:
            try:
                obj = json.loads(zf.read(name))
            except Exception:
                continue
            sessions = obj.get("voting_sessions", {})
            if not sessions:
                continue
            keys = sorted(sessions.keys(),
                          key=lambda x: int(x) if str(x).isdigit() else str(x))
            ds = sessions[keys[-1]].get("deal_suggestion", "")
            # answers files store a plain string; full_conversation a dict
            text = ds if isinstance(ds, str) else str(ds.get("answer", ""))
            if SCRATCHPAD_BLOCK_RE.search(text):
                scratchpad_seen += 1
            deals = extract_answer_deals(text)
            if not deals:
                continue
            if selection == "primary":
                deals = deals[:1]
            elif selection == "last":
                deals = deals[-1:]
            run = FULL_STEM_RE.sub("", ANSWERS_STEM_RE.sub("", Path(name).stem))
            for rank, deal in enumerate(deals, start=1):
                records.append({
                    "run": run,
                    "rank": rank,
                    "role": "primary" if rank == 1 else "fallback",
                    "deals_in_answer": len(deals),
                    "deal": deal,
                })

    per_deal = pd.DataFrame(records)
    tally = collections.Counter(per_deal["deal"]) if len(per_deal) else collections.Counter()
    counts = pd.DataFrame(
        sorted(({"deal": d, "count": n} for d, n in tally.items()),
               key=lambda r: (-r["count"], r["deal"]))
    )
    info = {
        "source": used,
        "files_parsed": len(names),
        "files_containing_private_scratchpad": scratchpad_seen,
    }
    return counts, per_deal, info


def deals_long_to_results_format(per_deal: pd.DataFrame,
                                 issue_order: Sequence[str]) -> pd.DataFrame:
    """One row per extracted deal in the Round;A;B;...  layout of Results_*.csv."""
    rows = []
    for _, r in per_deal.iterrows():
        toks = {t[0].upper(): t[1:] for t in deal_tokens(r["deal"])}
        row = {"Round": r["run"]}
        for issue in issue_order:
            row[issue] = int(toks[issue.upper()])
        rows.append(row)
    return pd.DataFrame(rows, columns=["Round", *issue_order])


def write_results_format(df: pd.DataFrame, path: Path) -> None:
    """Write with the separator, BOM and line endings of the supplied Results file."""
    df.to_csv(path, sep=";", index=False, encoding="utf-8-sig", lineterminator="\r\n")


def load_target_deals(deals_file, logs_zip, top_k, baseline_nbs, out_dir,
                      selection="all", issue_order=None, source="auto"):
    """Assemble the deals whose robustness Table 2 reports.

    Recurrent deals come from the logs zip, ordered by how many runs proposed
    them. The baseline NBS is prepended as N0 whether or not it was ever
    proposed, since every distance in the table is measured against it.
    """
    observed = None
    per_deal = None
    if logs_zip:
        observed, per_deal, info = extract_final_deals_from_logs_zip(
            logs_zip, selection=selection, source=source)
        print(f"Parsed {info['files_parsed']} {info['source']} files from the logs zip"
              + (f"; {info['files_containing_private_scratchpad']} contained a private "
                 "scratchpad, whose deals were excluded"
                 if info["files_containing_private_scratchpad"] else
                 "; none contained private reasoning"))
        observed.to_csv(out_dir / "observed_final_deal_counts.csv", index=False)
        per_deal.to_csv(out_dir / "observed_final_deals_long.csv", index=False)
        if issue_order:
            results = deals_long_to_results_format(per_deal, issue_order)
            write_results_format(results, out_dir / "Results_extracted_all_deals.csv")
            primary = per_deal[per_deal["rank"] == 1]
            write_results_format(
                deals_long_to_results_format(primary, issue_order),
                out_dir / "Results_extracted_primary_only.csv")
            n_multi = int((per_deal.groupby("run")["rank"].max() > 1).sum())
            print(f"Extracted {len(per_deal)} deals from {per_deal['run'].nunique()} runs "
                  f"({n_multi} runs proposed a fallback in addition to a primary deal)")

    if deals_file:
        df = pd.read_csv(deals_file)
        lower = {c.lower(): c for c in df.columns}
        if "deal" not in lower:
            raise ValueError("Target-deals CSV needs a Deal/deal column")
        deal_col = lower["deal"]
        id_col = lower.get("id")
        rows = []
        for i, row in df.iterrows():
            extra = {k: row[k] for k in df.columns if k not in {deal_col, id_col}}
            rows.append({"ID": str(row[id_col]) if id_col else f"R{i+1}",
                         "Deal": normalize_deal(row[deal_col]), **extra})
        targets = pd.DataFrame(rows)
        if observed is not None and "Runs" not in targets.columns:
            tally = dict(zip(observed["deal"], observed["count"]))
            targets["Runs"] = [int(tally.get(d, 0)) for d in targets["Deal"]]
    elif observed is not None:
        chosen = observed.head(int(top_k)).copy()
        targets = pd.DataFrame({
            "ID": [f"R{i+1}" for i in range(len(chosen))],
            "Deal": chosen["deal"].tolist(),
            "Runs": chosen["count"].astype(int).tolist(),
        })
    else:
        raise ValueError("Provide either --target-deals or --llm-logs-zip")

    targets = targets.loc[targets["ID"].astype(str).str.upper() != "N0"].copy()
    # Order the recurrent deals by how often they were proposed, with the deal
    # string as a deterministic tie-break, then renumber R1..Rk to match.
    if "Runs" in targets.columns:
        targets = targets.sort_values(["Runs", "Deal"], ascending=[False, True])
        targets["ID"] = [f"R{i+1}" for i in range(len(targets))]
    targets = targets.reset_index(drop=True)

    n0 = {c: np.nan for c in targets.columns}
    n0["ID"] = "N0"
    n0["Deal"] = normalize_deal(baseline_nbs)
    if "Runs" in targets.columns:
        tally = dict(zip(observed["deal"], observed["count"])) if observed is not None else {}
        n0["Runs"] = int(tally.get(normalize_deal(baseline_nbs), 0))
    targets = pd.concat([pd.DataFrame([n0]), targets], ignore_index=True)
    targets.to_csv(out_dir / "target_deals_used.csv", index=False)
    return targets


def target_indices(deals_df, targets):
    lookup = {normalize_deal(d): int(i) for i, d in enumerate(deals_df["deal"].tolist())}
    out = {}
    for _, r in targets.iterrows():
        d = normalize_deal(r["Deal"])
        if d not in lookup:
            raise ValueError(f"Target deal {d} is not valid for this scenario")
        out[str(r["ID"])] = lookup[d]
    return out


def nash_loss_percent(target_vec, nbs_vec, thresholds):
    tg = target_vec - thresholds
    ng = nbs_vec - thresholds
    if np.any(tg < -1e-9) or np.any(ng < -1e-9):
        return np.nan
    tp = float(np.prod(np.maximum(tg, 0.0)))
    np_ = float(np.prod(np.maximum(ng, 0.0)))
    if np_ <= 0:
        return np.nan
    return float(100.0 * (1.0 - tp / np_))


def fmt_interval(med, lo, hi, decimals=1):
    if any(pd.isna(x) for x in [med, lo, hi]):
        return "--"
    f = f"{{:.{decimals}f}}"
    return f"{f.format(med)} [{f.format(lo)}--{f.format(hi)}]"


def summarize_target_metrics(sample_metrics, baseline_rows, min_loss_samples=20):
    rows = []
    for _, b in baseline_rows.iterrows():
        ident = str(b["ID"])
        sub = sample_metrics[sample_metrics["ID"] == ident]
        loss_valid = sub["nash_loss"].dropna()
        loss_n = int(len(loss_valid))
        loss_stats = (np.nan, np.nan, np.nan)
        if loss_n >= min_loss_samples:
            loss_stats = tuple(float(loss_valid.quantile(q)) for q in [0.5, 0.05, 0.95])
        row = {
            "ID": ident,
            "Deal": b["Deal"],
            "IRpct": float(sub["IR"].mean() * 100),
            "ParetoPct": float(sub["Pareto"].mean() * 100),
            "H0": int(b["H0"]),
            "Hmed": float(sub["H"].median()),
            "Hlo": float(sub["H"].quantile(0.05)),
            "Hhi": float(sub["H"].quantile(0.95)),
            "L10": float(b["L10"]),
            "L1med": float(sub["L1"].median()),
            "L1lo": float(sub["L1"].quantile(0.05)),
            "L1hi": float(sub["L1"].quantile(0.95)),
            "L20": float(b["L20"]),
            "L2med": float(sub["L2"].median()),
            "L2lo": float(sub["L2"].quantile(0.05)),
            "L2hi": float(sub["L2"].quantile(0.95)),
            "Loss0": b["Loss0"],
            "Lossmed": loss_stats[0],
            "Losslo": loss_stats[1],
            "Losshi": loss_stats[2],
            "LossN": loss_n,
        }
        for c in b.index:
            if c not in row and c not in {"H0", "L10", "L20", "Loss0"}:
                row[c] = b[c]
        rows.append(row)
    return pd.DataFrame(rows)


def scenario_from_scores_csv(path, threshold: float, separator: str = ";"):
    """Build the scenario dict from a stakeholder-by-option score matrix.

    Expects a `Stakeholder` column followed by issue-option columns named like
    A1, A2, B1 ... The issue weight is the maximum option score within that
    issue, exactly as in the elicitation rules, and every stakeholder is given
    the same acceptance threshold. This is the same input format the
    Campylobacter feasibility search consumes, so both analyses can be driven
    from one file.
    """
    df = pd.read_csv(path, sep=separator)
    if "Stakeholder" not in df.columns:
        raise ValueError("Scores CSV must contain a 'Stakeholder' column")
    if df["Stakeholder"].duplicated().any():
        dup = df.loc[df["Stakeholder"].duplicated(), "Stakeholder"].tolist()
        raise ValueError(f"Duplicate stakeholder rows: {dup}")

    option_cols = [c for c in df.columns if re.fullmatch(r"[A-Za-z]+\d+", str(c))]
    if not option_cols:
        raise ValueError("No issue-option columns such as A1 or B2 were found")
    if df[option_cols].isna().any().any():
        raise ValueError("Score matrix contains missing option scores")
    if (df[option_cols] < 0).any().any():
        raise ValueError("Option scores must be nonnegative")

    issue_order, option_order = [], {}
    for c in option_cols:
        issue = re.match(r"^([A-Za-z]+)", str(c)).group(1)
        if issue not in option_order:
            issue_order.append(issue)
            option_order[issue] = []
        option_order[issue].append(str(c))

    parties = df["Stakeholder"].astype(str).tolist()
    utilities, thresholds = {}, {}
    for _, row in df.iterrows():
        p = str(row["Stakeholder"])
        utilities[p] = {}
        for issue in issue_order:
            opts = {o: float(row[o]) for o in option_order[issue]}
            utilities[p][issue] = {"max": float(max(opts.values())), "options": opts}
        thresholds[p] = float(threshold)

    totals = {p: sum(utilities[p][i]["max"] for i in issue_order) for p in parties}
    return {
        "parties": parties,
        "issue_order": issue_order,
        "option_order": option_order,
        "utilities": utilities,
        "thresholds": thresholds,
        "issue_weight_totals": totals,
    }


def load_scenario_any(args):
    """Load from a scores CSV when given, otherwise from the prompts zip."""
    if args.scores_csv:
        sc = scenario_from_scores_csv(args.scores_csv, args.threshold, args.separator)
        print("Issue-weight totals per stakeholder:",
              {k: round(v, 3) for k, v in sc["issue_weight_totals"].items()})
        off = {k: v for k, v in sc["issue_weight_totals"].items() if abs(v - 100.0) > 1e-6}
        if off:
            print(f"  WARNING: totals differ from 100 for {off}; "
                  "the budget is preserved at each stakeholder's own total.")
        return sc, sc["thresholds"]
    sc = core.load_scenario(args.prompts_zip, Path(args.out).resolve() / "_work",
                            default_model=args.default_model)
    return sc, core.thresholds_dict(sc)


def _prepare_fast_mc(scenario: dict, base_u: dict):
    issue_order = list(scenario["issue_order"])
    parties = list(scenario["parties"])
    deals = core.enumerate_deals(issue_order, scenario["option_order"])
    ratios, base_weights = {}, {}
    for p in parties:
        R = np.zeros((len(deals), len(issue_order)), dtype=float)
        W = np.zeros(len(issue_order), dtype=float)
        for j, issue in enumerate(issue_order):
            idata = base_u[p][issue]
            w = float(idata["max"])
            W[j] = w
            if w > 0:
                mapping = {o: float(v) / w for o, v in idata["options"].items()}
                R[:, j] = deals[issue].map(mapping).to_numpy(dtype=float)
            else:
                # Baseline weight 0: every option was scored 0, so no within-issue
                # preference was expressed. Under the scoring rules indifference
                # means every option carries the issue maximum, hence ratio 1.0.
                # Such an issue shifts utility level, never package ranking.
                R[:, j] = 1.0
        ratios[p] = R
        base_weights[p] = W
    return deals, parties, issue_order, ratios, base_weights


def issue_caps(w: np.ndarray, frac: float, floor_pts: float, zero_upper: float):
    """Per-issue admissible interval [lower, upper] for one stakeholder.

    Non-zero weight : half-width max(floor_pts, frac*w), clipped at 0 below.
    Zero weight     : one-sided interval [0, zero_upper].

    Identical to bounds_for_weights() in campy_nbs_strategy_search.py.
    """
    w = np.asarray(w, dtype=float)
    half = np.maximum(floor_pts, frac * w)
    lower = np.maximum(0.0, w - half)
    upper = w + half
    zero = w <= 0.0
    lower[zero] = 0.0
    upper[zero] = zero_upper
    return lower, upper


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


def sample_weight_chains(base_weights, parties, frac, floor_pts, zero_upper,
                         n_samples, rng, burn_in, thinning):
    """Uniform draws on each stakeholder's bounded, budget-preserving weight set.

    Uses the same hit-and-run sampler as the Campylobacter feasibility search, so
    both analyses draw from one distribution over the same support: uniform on
    {w' : lower <= w' <= upper, sum(w') = sum(w)}. One chain per stakeholder,
    started at the unperturbed weights.
    """
    chains = {}
    for p in parties:
        w = base_weights[p].astype(float)
        lower, upper = issue_caps(w, frac, floor_pts, zero_upper)
        chains[p] = hit_and_run(w, lower, upper, float(w.sum()),
                                n_samples, rng, burn_in, thinning)
    return chains


def scores_from_proportional_draws(parties, issue_order, ratios, base_weights,
                                   chains=None, sample_index=None):
    """Build package scores from one weight draw.

    With chains=None the unperturbed weights are used, which reproduces the
    baseline landscape. Otherwise row `sample_index` of each stakeholder's
    hit-and-run chain supplies the perturbed weights.
    """
    n = next(iter(ratios.values())).shape[0]
    scores = np.zeros((n, len(parties)), dtype=float)
    new_weights = {}
    for pi, p in enumerate(parties):
        w = base_weights[p].astype(float)
        nw = w.copy() if chains is None else chains[p][sample_index]
        new_weights[p] = nw
        scores[:, pi] = ratios[p] @ nw
    return scores, new_weights


def sample_thresholds(base_thresholds, rng, frac, mode="common"):
    """Multiplicative +/-frac perturbation of the shared acceptance threshold.

    The acceptance threshold is established jointly for all participants, so a
    single shift is drawn per sample and applied to every stakeholder. Passing
    mode="off" (or frac <= 0) leaves the thresholds unperturbed.
    """
    r = np.asarray(base_thresholds, dtype=float)
    if frac <= 0 or mode == "off":
        return r.copy()
    return r * (1.0 + float(rng.uniform(-frac, frac)))


def strict_order_pairs(w: np.ndarray):
    """Index pairs (i, j) whose baseline weights are strictly ordered w_i > w_j."""
    m = len(w)
    return [(i, j) for i in range(m) for j in range(m) if w[i] > w[j]]


def infeasibility_gap(scores, thresholds):
    """min over packages of max over stakeholders of (threshold - utility).

    Negative or zero means at least one package satisfies every threshold, so an
    NBS exists. A positive value is the number of utility points by which the
    preference structure misses feasibility.
    """
    return float(np.min(np.max(thresholds[None, :] - scores, axis=1)))


def choose_nbs_safe(scores, thresholds):
    """As choose_nbs_fast, but returns None instead of raising when no NBS exists."""
    try:
        return choose_nbs_fast(scores, thresholds)
    except RuntimeError:
        return None


def choose_nbs_fast(scores, thresholds):
    unanimous = np.all(scores >= thresholds[None, :] - 1e-9, axis=1)
    idx = np.flatnonzero(unanimous)
    if len(idx) == 0:
        raise RuntimeError("No unanimous individually-rational deal exists; NBS undefined")
    gains = np.maximum(scores[idx] - thresholds[None, :], 0.0)
    prod = np.prod(gains, axis=1)
    max_prod = np.max(prod)
    cpos = np.flatnonzero(np.isclose(prod, max_prod, rtol=1e-12, atol=1e-12))
    if len(cpos) > 1:
        welfare = scores[idx[cpos]].sum(axis=1)
        chosen_pos = int(cpos[int(np.argmax(welfare))])
    else:
        chosen_pos = int(cpos[0])
    return int(idx[chosen_pos]), float(prod[chosen_pos])


def make_table2_tex(summary, alt_nbs, baseline_nbs, n_samples, settings, out_tex):
    def esc(s):
        return str(s).replace("_", r"\_").replace("%", r"\%")

    frac_pct = settings["frac"] * 100.0
    floor_pts = settings["floor_pts"]
    thr_pct = settings["threshold_frac"] * 100.0
    base_thr = settings.get("base_threshold")
    thr_txt = (rf"$r={base_thr:g}$" if base_thr is not None else "each stakeholder's $r$")
    no_nbs = settings["no_nbs_pct"]
    mode_txt = {"common": "as a single shared draw applied to all stakeholders",
                "off": "not perturbed"}[settings["threshold_mode"]]

    lines = [
        r"\documentclass[9pt,a4paper]{article}",
        r"\usepackage[a4paper,margin=9mm]{geometry}",
        r"\usepackage[T1]{fontenc}",
        # lmodern is optional so the table also compiles on minimal TeX installs;
        # without scalable fonts microtype cannot do font expansion.
        r"\IfFileExists{lmodern.sty}{\usepackage{lmodern}\usepackage{microtype}}"
        r"{\usepackage[expansion=false]{microtype}}",
        r"\usepackage{amsmath,amssymb,booktabs,array,pdflscape}",
        r"\setlength{\parindent}{0pt}",
        r"\newcommand{\NBS}{\mathrm{NBS}}",
        r"\begin{document}\begin{landscape}\thispagestyle{empty}",
        r"{\Large\bfseries Table 2: Sensitivity of recurrent deals to the contemporaneous NBS}\\",
        rf"{{\small {n_samples:,} simultaneous Monte Carlo perturbations of the stakeholder "
        rf"preference landscape (per-issue interval $\pm\max({floor_pts:g}\text{{ point}},\,{frac_pct:g}\%$ "
        rf"of the issue weight$)$, total budget preserved) and of the acceptance threshold "
        rf"($\pm{thr_pct:g}\%$ of {thr_txt}, drawn {mode_txt}); "
        rf"baseline NBS: {esc(baseline_nbs)}.}}\par\vspace{{2mm}}",
        r"\centering\scriptsize\setlength{\tabcolsep}{2.3pt}\renewcommand{\arraystretch}{1.16}",
        r"\begin{tabular}{@{}llccc|cc|cc|cc|cc@{}}",
        r"\toprule",
        r"&&&&&\multicolumn{2}{c|}{\textbf{Hamming $H$}}&\multicolumn{2}{c|}{\textbf{$L_1$}}&\multicolumn{2}{c|}{\textbf{$L_2$}}&\multicolumn{2}{c}{\textbf{Nash-efficiency loss $\mathcal L_{\NBS}$ (\%)}}\\",
        r"\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}",
        r"\textbf{ID}&\textbf{Deal $(A,B,C,D,E,F)$}&\textbf{Runs}&\textbf{IR in MC}&\textbf{Pareto in MC}&\textbf{Base}&\textbf{MC med. [5--95\%]}&\textbf{Base}&\textbf{MC med. [5--95\%]}&\textbf{Base}&\textbf{MC med. [5--95\%]}&\textbf{Base}&\textbf{MC med. [5--95\%]}\\",
        r"\midrule",
    ]
    for _, r in summary.iterrows():
        deal = "(" + ",".join(x[1:] if x and x[0].isalpha() else x for x in deal_tokens(r["Deal"])) + ")"
        loss0 = "--" if pd.isna(r["Loss0"]) else f"{float(r['Loss0']):.1f}"
        lossmc = fmt_interval(r["Lossmed"], r["Losslo"], r["Losshi"], 1)
        runs = r.get("Runs", np.nan)
        runs_txt = "--" if pd.isna(runs) else f"{int(runs)}"
        row = [
            esc(r["ID"]), f"${deal}$", runs_txt,
            f"{r['IRpct']:.1f}\\%", f"{r['ParetoPct']:.1f}\\%",
            f"{int(r['H0'])}", fmt_interval(r["Hmed"], r["Hlo"], r["Hhi"], 0),
            f"{r['L10']:.1f}", fmt_interval(r["L1med"], r["L1lo"], r["L1hi"], 1),
            f"{r['L20']:.1f}", fmt_interval(r["L2med"], r["L2lo"], r["L2hi"], 1),
            loss0, lossmc,
        ]
        lines.append(" & ".join(row) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\raggedright\vspace{2mm}"]
    lines += [
        rf"\textbf{{Perturbation rule.}} Each issue weight $w_{{pi}}$ is redrawn uniformly from its own "
        rf"admissible interval $[\max(0,w_{{pi}}-c_{{pi}}),\,w_{{pi}}+c_{{pi}}]$ with "
        rf"$c_{{pi}}=\max({floor_pts:g},\,{settings['frac']:g}\,w_{{pi}})$, so tolerance scales with the stated "
        rf"weight and never falls below the {floor_pts:g}-point granularity of the elicitation instrument. "
        rf"A baseline weight of $0$ is treated as ``could plausibly have been {settings['zero_upper']:g}'' and "
        rf"redrawn from $[0,{settings['zero_upper']:g}]$; since all options in such an issue were scored $0$, "
        rf"no within-issue preference was expressed and every option receives the perturbed issue maximum, "
        rf"which shifts the stakeholder's utility level without altering their ranking of packages. "
        rf"Weights are drawn by hit-and-run, uniformly on the set of vectors that stay inside every "
        rf"issue interval and preserve the stakeholder's original total budget (100 points); the same "
        rf"sampler is used in the companion Campylobacter feasibility search. All option utilities within a non-zero issue are rescaled "
        rf"proportionally, so within-issue ratios, rankings, ties, and structural zeros are preserved exactly. "
        rf"Issue-importance reversals are therefore possible only between issues whose intervals overlap; "
        rf"across all draws the largest relative weight gap ever reversed was "
        rf"{settings['max_reversal_gap_pct']:.0f}\%, over {settings['n_reversing_pairs']} issue pairs. "
        rf"The acceptance threshold is established jointly for all participants and is "
        rf"therefore perturbed as a single shared draw: "
        rf"$r'_p=r_p(1+u)$ for every stakeholder $p$, with one $u\sim U(-{settings['threshold_frac']:g},{settings['threshold_frac']:g})$ drawn per sample.",
        r"\par\vspace{1mm}",
        rf"\textbf{{Definitions.}} For perturbation $s$ with contemporaneous NBS $d_s^*$ and thresholds $r^{{(s)}}$,",
        r"\[H_s(d)=\sum_j \mathbf 1[d_j\neq d^*_{s,j}],\quad L_{1,s}(d)=\sum_i|U_i^{(s)}(d)-U_i^{(s)}(d_s^*)|,\]",
        r"\[L_{2,s}(d)=\sqrt{\sum_i\left(U_i^{(s)}(d)-U_i^{(s)}(d_s^*)\right)^2},\quad",
        r"\mathcal L_{\NBS,s}(d)=100\left[1-\frac{\prod_i\left(U_i^{(s)}(d)-r_i^{(s)}\right)}{\prod_i\left(U_i^{(s)}(d_s^*)-r_i^{(s)}\right)}\right].\]",
        rf"\textbf{{Runs}} counts how many of the {settings['n_runs']} simulation runs put the deal "
        rf"forward in their final public proposal; recurrent deals are listed in descending order of "
        rf"that count, with the deal string as a tie-break. Every distinct deal in a run's public "
        rf"\texttt{{<ANSWER>}} block is counted, so a run proposing a primary deal and a fallback "
        rf"contributes both; deals appearing only in the private \texttt{{<SCRATCHPAD>}} are excluded. "
        rf"N0 is the baseline Nash bargaining solution, included because every distance is measured "
        rf"against it, and was not itself proposed in any run. "
        rf"IR and Pareto shares are over all {n_samples:,} draws. Because a raised threshold can empty the "
        rf"individually rational set, the NBS may fail to exist: this occurred in {no_nbs:.1f}\% of draws, and "
        rf"the $H$, $L_1$, $L_2$ and Nash-loss columns are computed over the remaining draws. "
        rf"Brackets are empirical Monte Carlo robustness intervals (5th--95th percentiles), not "
        rf"sampling-theory confidence intervals. Nash loss is summarized only when enough perturbed draws "
        rf"keep the target deal individually rational, and is then conditional on that subset.",
        r"\par\vspace{3mm}",
        r"\textbf{Alternative NBS distribution} (over draws in which an NBS exists).\par\vspace{1mm}",
        r"\begin{tabular}{@{}lrrr@{}}\toprule \textbf{Deal}&\textbf{Count}&\textbf{Frequency}&\textbf{$H$ vs baseline}\\\midrule",
    ]
    for _, r in alt_nbs.iterrows():
        d = "(" + ",".join(x[1:] for x in deal_tokens(r["Deal"])) + ")"
        lines.append(f"${d}$ & {int(r['Count'])} & {float(r['Pct']):.1f}\\% & {int(r['H_vs_N0'])}\\\\")
    lines += [r"\bottomrule\end{tabular}"]

    lines += [r"\end{landscape}\end{document}"]
    out_tex.write_text("\n".join(lines), encoding="utf-8")


def compile_tex(tex_path):
    exe = shutil.which("pdflatex")
    if not exe:
        print("[info] pdflatex unavailable; .tex created but PDF skipped")
        return None
    try:
        subprocess.run([exe, "-interaction=nonstopmode", "-halt-on-error", tex_path.name], cwd=tex_path.parent,
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        print(e.stdout[-4000:])
        return None
    return tex_path.with_suffix(".pdf")


def run_analysis(args):
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario, thresholds_d = load_scenario_any(args)
    base_u = (copy.deepcopy(scenario["utilities"]) if args.scores_csv
              else core.utility_copy(scenario))
    deals, parties, issue_order, ratios, base_weights = _prepare_fast_mc(scenario, base_u)
    th_arr = np.array([thresholds_d[p] for p in parties], dtype=float)

    # Baseline: unperturbed weights and unperturbed thresholds.
    bscores, _ = scores_from_proportional_draws(
        parties, issue_order, ratios, base_weights)
    nbs_idx0, nbs_prod0 = choose_nbs_fast(bscores, th_arr)
    baseline_nbs = normalize_deal(deals.loc[nbs_idx0, "deal"])
    nbs_vec0 = bscores[nbs_idx0]

    targets = load_target_deals(args.target_deals, args.llm_logs_zip, args.top_k,
                                baseline_nbs, out_dir,
                                selection=args.deal_selection, issue_order=issue_order,
                                source=args.logs_source)
    long_path = out_dir / "observed_final_deals_long.csv"
    observed_runs = (pd.read_csv(long_path)["run"].nunique() if long_path.exists() else 0)
    tindex = target_indices(deals, targets)

    baseline_rows = []
    for _, tr in targets.iterrows():
        ident, d = str(tr["ID"]), normalize_deal(tr["Deal"])
        idx = tindex[ident]
        tv = bscores[idx]
        ir = bool(np.all(tv >= th_arr - 1e-9))
        row = {
            "ID": ident, "Deal": d,
            "H0": hamming_count(d, baseline_nbs),
            "L10": float(np.abs(tv - nbs_vec0).sum()),
            "L20": float(np.sqrt(np.square(tv - nbs_vec0).sum())),
            "Loss0": nash_loss_percent(tv, nbs_vec0, th_arr) if ir else np.nan,
        }
        for c in targets.columns:
            if c not in {"ID", "Deal"}:
                row[c] = tr[c]
        baseline_rows.append(row)
    baseline_rows = pd.DataFrame(baseline_rows)

    # Report the admissible interval actually used for every parameter.
    cap_rows = []
    order_pairs = {}
    for p_ in parties:
        w = base_weights[p_]
        lo, hi = issue_caps(w, args.frac, args.floor_pts, args.zero_upper)
        order_pairs[p_] = strict_order_pairs(w)
        for j, issue in enumerate(issue_order):
            cap_rows.append({
                "party": p_, "issue": issue, "baseline_weight": float(w[j]),
                "lower": float(lo[j]), "upper": float(hi[j]),
                "half_width": float((hi[j] - lo[j]) / 2.0),
                "floor_binding": bool(w[j] > 0 and args.floor_pts >= args.frac * w[j]),
                "zero_weight_rule": bool(w[j] <= 0),
            })
    pd.DataFrame(cap_rows).to_csv(out_dir / "issue_weight_intervals.csv", index=False)

    rng = np.random.default_rng(args.seed)
    chains = sample_weight_chains(base_weights, parties, args.frac, args.floor_pts,
                                  args.zero_upper, args.mc_samples, rng,
                                  args.burn_in, args.thinning)

    target_ids = targets["ID"].astype(str).tolist()
    target_deals = [normalize_deal(x) for x in targets["Deal"]]
    target_idx_arr = np.array([tindex[x] for x in target_ids], dtype=int)

    target_rows, mc_rows, final_adjustment_rows, threshold_rows = [], [], [], []
    reversal_counter = collections.Counter()
    reversal_gaps = []
    no_nbs_samples = 0

    for s_i in range(args.mc_samples):
        scores, new_weights = scores_from_proportional_draws(
            parties, issue_order, ratios, base_weights, chains, s_i)
        th_s = sample_thresholds(th_arr, rng, args.threshold_frac, args.threshold_mode)

        adj_row = {"sample_id": s_i}
        for p_ in parties:
            for j, issue in enumerate(issue_order):
                adj_row[f"{p_}|{issue}"] = float(new_weights[p_][j] - base_weights[p_][j])
        final_adjustment_rows.append(adj_row)
        threshold_rows.append({"sample_id": s_i,
                               **{p_: float(th_s[k]) for k, p_ in enumerate(parties)}})

        # Issue-importance reversals: expected only between overlapping intervals.
        for p_ in parties:
            w, nw = base_weights[p_], new_weights[p_]
            for i, j in order_pairs[p_]:
                if nw[i] < nw[j] - 1e-9:
                    reversal_counter[(p_, issue_order[i], issue_order[j])] += 1
                    reversal_gaps.append((w[i] - w[j]) / max(w[i], 1.0))

        chosen = choose_nbs_safe(scores, th_s)
        tvs = scores[target_idx_arr]

        ge = np.all(scores[:, None, :] >= (tvs[None, :, :] - 1e-9), axis=2)
        gt = np.any(scores[:, None, :] > (tvs[None, :, :] + 1e-9), axis=2)
        pareto_flags = ~np.any(ge & gt, axis=0)
        ir_flags = np.all(tvs >= th_s[None, :] - 1e-9, axis=1)

        if chosen is None:
            no_nbs_samples += 1
            mc_rows.append({
                "sample_id": s_i, "nbs_exists": False, "nbs_deal": None,
                "nbs_same": False, "nbs_hamming_count": np.nan, "nbs_product": np.nan,
                "infeasibility_gap": infeasibility_gap(scores, th_s),
            })
            for k, (ident, d) in enumerate(zip(target_ids, target_deals)):
                target_rows.append({
                    "sample_id": s_i, "ID": ident, "Deal": d, "nbs_exists": False,
                    "nbs_deal": None, "IR": bool(ir_flags[k]), "Pareto": bool(pareto_flags[k]),
                    "H": np.nan, "L1": np.nan, "L2": np.nan, "nash_loss": np.nan,
                })
            if (s_i + 1) % max(1, args.progress_every) == 0:
                print(f"[MC] {s_i+1}/{args.mc_samples}")
            continue

        nbs_idx, nbs_prod = chosen
        nbs_deal = normalize_deal(deals.loc[nbs_idx, "deal"])
        nbs_vec = scores[nbs_idx]

        mc_rows.append({
            "sample_id": s_i, "nbs_exists": True, "nbs_deal": nbs_deal,
            "nbs_same": nbs_deal == baseline_nbs,
            "nbs_hamming_count": hamming_count(nbs_deal, baseline_nbs),
            "nbs_product": nbs_prod,
            "infeasibility_gap": infeasibility_gap(scores, th_s),
        })
        for k, (ident, d) in enumerate(zip(target_ids, target_deals)):
            tv = tvs[k]
            ir = bool(ir_flags[k])
            target_rows.append({
                "sample_id": s_i, "ID": ident, "Deal": d, "nbs_exists": True,
                "nbs_deal": nbs_deal, "IR": ir, "Pareto": bool(pareto_flags[k]),
                "H": hamming_count(d, nbs_deal),
                "L1": float(np.abs(tv - nbs_vec).sum()),
                "L2": float(np.sqrt(np.square(tv - nbs_vec).sum())),
                "nash_loss": nash_loss_percent(tv, nbs_vec, th_s) if ir else np.nan,
            })
        if (s_i + 1) % max(1, args.progress_every) == 0:
            print(f"[MC] {s_i+1}/{args.mc_samples}")

    mc_df = pd.DataFrame(mc_rows)
    target_sample_df = pd.DataFrame(target_rows)
    adjustment_df = pd.DataFrame(final_adjustment_rows)
    threshold_df = pd.DataFrame(threshold_rows)

    mc_df.to_csv(out_dir / "mc_summary.csv", index=False)
    target_sample_df.to_csv(out_dir / "target_deal_mc_samples.csv", index=False)
    adjustment_df.to_csv(out_dir / "mc_weight_adjustments.csv", index=False)
    threshold_df.to_csv(out_dir / "mc_threshold_draws.csv", index=False)

    rev_rows = [{"party": k[0], "issue_higher": k[1], "issue_lower": k[2],
                 "reversals": v, "pct_of_samples": 100.0 * v / args.mc_samples}
                for k, v in reversal_counter.most_common()]
    pd.DataFrame(rev_rows).to_csv(out_dir / "order_reversals.csv", index=False)
    max_rev_gap = 100.0 * max(reversal_gaps) if reversal_gaps else 0.0

    # ---- validation of the requested sensitivity design ----
    interval_violations = 0
    for p_ in parties:
        w = base_weights[p_]
        lo, hi = issue_caps(w, args.frac, args.floor_pts, args.zero_upper)
        for j, issue in enumerate(issue_order):
            col = adjustment_df[f"{p_}|{issue}"].to_numpy(dtype=float) + w[j]
            interval_violations += int(np.sum((col < lo[j] - 1e-7) | (col > hi[j] + 1e-7)))
    if interval_violations:
        raise RuntimeError(f"{interval_violations} weight draws fell outside their issue interval")

    max_sum_error = 0.0
    for _, row in adjustment_df.iterrows():
        for p_ in parties:
            max_sum_error = max(max_sum_error,
                                abs(sum(float(row[f"{p_}|{issue}"]) for issue in issue_order)))
    if max_sum_error > 1e-7:
        raise RuntimeError(f"Budget conservation failed; max stakeholder delta sum = {max_sum_error}")

    adj_cols = [c for c in adjustment_df.columns if c != "sample_id"]
    max_abs = float(adjustment_df[adj_cols].abs().to_numpy().max())
    thr_cols = [c for c in threshold_df.columns if c != "sample_id"]
    thr_rel = (threshold_df[thr_cols].to_numpy(dtype=float) / th_arr[None, :]) - 1.0
    max_thr_rel = float(np.abs(thr_rel).max()) if thr_rel.size else 0.0
    if max_thr_rel > args.threshold_frac + 1e-9:
        raise RuntimeError(f"Threshold draw exceeded +/-{args.threshold_frac}: {max_thr_rel}")

    summary = summarize_target_metrics(target_sample_df, baseline_rows,
                                       min_loss_samples=args.min_loss_samples)
    summary.to_csv(out_dir / "Table2_values.csv", index=False)

    feasible = mc_df[mc_df["nbs_exists"]]
    nbs_counts = feasible["nbs_deal"].value_counts().rename_axis("Deal").reset_index(name="Count")
    nbs_counts["Pct"] = nbs_counts["Count"] / max(len(feasible), 1) * 100.0
    nbs_counts["H_vs_N0"] = nbs_counts["Deal"].apply(lambda d: hamming_count(d, baseline_nbs))
    nbs_counts.to_csv(out_dir / "alternative_NBS_distribution.csv", index=False)

    changed = feasible[feasible["nbs_deal"] != baseline_nbs]
    base_toks = deal_tokens(baseline_nbs)
    issue_change_rows = []
    for j, issue in enumerate(issue_order):
        pct = (np.mean([deal_tokens(d)[j] != base_toks[j] for d in changed["nbs_deal"]]) * 100.0) if len(changed) else 0.0
        issue_change_rows.append({"issue": issue, "pct_changed_conditional_on_NBS_change": pct})
    pd.DataFrame(issue_change_rows).to_csv(out_dir / "alternative_NBS_issue_changes.csv", index=False)

    settings = {
        "n_runs": int(observed_runs),
        "frac": args.frac, "floor_pts": args.floor_pts, "zero_upper": args.zero_upper,
        "threshold_frac": args.threshold_frac, "threshold_mode": args.threshold_mode,
        "threshold_perturbation": ("single shared draw applied to all stakeholders"
                                   if args.threshold_mode == "common" else "none"),
        "base_threshold": float(th_arr[0]) if len(set(th_arr)) == 1 else None,
        "no_nbs_pct": 100.0 * no_nbs_samples / args.mc_samples,
        "max_reversal_gap_pct": max_rev_gap,
        "n_reversing_pairs": len(rev_rows),
    }

    tex = out_dir / "Table2_sensitivity.tex"
    make_table2_tex(summary, nbs_counts, baseline_nbs, args.mc_samples, settings, tex)
    pdf = compile_tex(tex)

    metadata = {
        "score_source": (str(Path(args.scores_csv).resolve()) if args.scores_csv
                         else str(Path(args.prompts_zip).resolve())),
        "target_deals_file": str(Path(args.target_deals).resolve()) if args.target_deals else None,
        "llm_logs_zip": str(Path(args.llm_logs_zip).resolve()) if args.llm_logs_zip else None,
        "mc_samples": args.mc_samples,
        "deal_selection": args.deal_selection,
        "logs_source": args.logs_source,
        "deal_extraction_scope": ("public response only; answers*.json contain no private "
                                  "reasoning, and when full_conversation*.json is parsed "
                                  "instead only the <ANSWER> block is read"),
        "issue_weight_frac": args.frac,
        "issue_weight_floor_points": args.floor_pts,
        "zero_weight_upper": args.zero_upper,
        "threshold_frac": args.threshold_frac,
        "threshold_mode": args.threshold_mode,
        "threshold_perturbation": ("single shared draw applied to all stakeholders"
                                   if args.threshold_mode == "common" else "none"),
        "seed": args.seed,
        "perturbation": ("uniform hit-and-run on {w' : lower <= w' <= upper, "
                         "sum(w') = sum(w)} with lower/upper from "
                         "[max(0, w-max(floor, frac*w)), w+max(floor, frac*w)] and "
                         "zero-weight issues on [0, zero_upper]"),
        "sampler": "hit-and-run, shared with campy_nbs_strategy_search.py",
        "hit_and_run_burn_in": args.burn_in,
        "hit_and_run_thinning": args.thinning,
        "normalization": "each stakeholder's final issue weights retain the original total (normally 100)",
        "within_issue": "option scores rescaled proportionally; ratios/rankings/ties/structural zeros preserved",
        "zero_weight_issue_handling": "all options assigned the perturbed issue maximum (indifference)",
        "max_observed_abs_final_issue_weight_change": max_abs,
        "max_observed_abs_stakeholder_weight_budget_error": max_sum_error,
        "max_observed_relative_threshold_change": max_thr_rel,
        "samples_without_NBS": no_nbs_samples,
        "samples_without_NBS_pct": 100.0 * no_nbs_samples / args.mc_samples,
        "issue_order_reversals_pairs": rev_rows,
        "max_relative_gap_reversed_pct": max_rev_gap,
        "baseline_nbs": baseline_nbs,
        "parties": parties,
        "issue_order": issue_order,
        "baseline_thresholds": thresholds_d,
        "table2_pdf": str(pdf) if pdf else None,
    }
    (out_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Analysis written to {out_dir}")
    print(f"Baseline NBS: {baseline_nbs}")
    print(f"Samples with no NBS: {no_nbs_samples}/{args.mc_samples} "
          f"({100.0*no_nbs_samples/args.mc_samples:.1f}%)")
    if len(feasible):
        print(f"Baseline NBS unchanged (of feasible): {int(feasible['nbs_same'].sum())}/{len(feasible)} "
              f"= {feasible['nbs_same'].mean()*100:.1f}%")
    print(f"Max final |issue-weight change|: {max_abs:.4f} pts")
    print(f"Max budget error: {max_sum_error:.3e}; max relative threshold change: {max_thr_rel:.4f}")
    print(f"Issue-order reversals: {len(rev_rows)} pairs; largest relative gap reversed {max_rev_gap:.1f}%")


def build_parser():
    p = argparse.ArgumentParser(description="Table-2 sensitivity: proportional issue-weight caps with absolute floor + threshold perturbation")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--scores-csv",
                     help="stakeholder-by-option score matrix (same format as the "
                          "Campylobacter search); issue weight = max option score per issue")
    src.add_argument("--prompts-zip",
                     help="agent prompt zip; scores are parsed out of the prompts")
    p.add_argument("--separator", default=";", help="field separator of --scores-csv")
    p.add_argument("--logs-source", choices=["answers", "full_conversation", "auto"],
                   default="answers",
                   help=("which members of the logs zip to parse. 'answers' uses the "
                         "answers*.json files, which contain only the public response "
                         "and therefore cannot expose private reasoning; "
                         "'full_conversation' falls back to the full transcripts and "
                         "isolates the <ANSWER> block; 'auto' prefers answers when present"))
    p.add_argument("--deal-selection", choices=["all", "primary", "last"], default="all",
                   help=("which deals to take from each run's public answer: every "
                         "distinct one (default), the primary proposal only, or the "
                         "last one only, which reproduces the original behaviour"))
    p.add_argument("--threshold", type=float, default=65.0,
                   help="common acceptance threshold applied with --scores-csv")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--target-deals")
    src.add_argument("--llm-logs-zip")
    p.add_argument("--top-k", type=int, default=7)
    p.add_argument("--out", default="table2_absolute5")
    p.add_argument("--mc-samples", type=int, default=1500)
    p.add_argument("--frac", type=float, default=0.25,
                   help="proportional half-width of the issue-weight interval (0.25 = 25%%)")
    p.add_argument("--floor-pts", type=float, default=1.0,
                   help="absolute floor on the half-width, in points")
    p.add_argument("--burn-in", type=int, default=500,
                   help="hit-and-run burn-in steps (matches the Campylobacter search)")
    p.add_argument("--thinning", type=int, default=100,
                   help=("hit-and-run thinning interval. 100 brings the mean lag-1 "
                         "autocorrelation of the weight draws to ~0.005, i.e. "
                         "effectively independent; the Campylobacter default of 5 "
                         "leaves ~0.53 and biases the retention estimate low"))
    p.add_argument("--zero-upper", type=float, default=1.0,
                   help="upper bound for an issue whose baseline weight is 0")
    p.add_argument("--threshold-frac", type=float, default=0.10,
                   help="relative perturbation of the shared acceptance threshold (0.10 = +/-10%%)")
    p.add_argument("--threshold-mode", choices=["common", "off"], default="common",
                   help="perturb the common acceptance threshold as one shared draw, or not at all")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--default-model", default="gpt-4o")
    p.add_argument("--min-loss-samples", type=int, default=20)
    p.add_argument("--progress-every", type=int, default=100)
    return p


if __name__ == "__main__":
    run_analysis(build_parser().parse_args())
