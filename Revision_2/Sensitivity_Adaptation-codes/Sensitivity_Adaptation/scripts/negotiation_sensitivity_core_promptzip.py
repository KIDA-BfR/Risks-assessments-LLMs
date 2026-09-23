from __future__ import annotations

import copy
import itertools
import json
import math
import os
import re
import shutil
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable, Any

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


NUMBER_RE = r"[-+]?\d+(?:\.\d+)?"
ISSUE_LINE_RE = re.compile(
    rf"^\s*Issue\s+([A-Za-z0-9_\-]+)\s*\(\s*max\s+score\s*({NUMBER_RE})\s*\)\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
OPTION_SCORE_RE = re.compile(rf"([A-Za-z]+\d+)\s*\(\s*({NUMBER_RE})\s*\)")
THRESHOLD_PATTERNS = [
    re.compile(rf"cannot\s+accept\s+any\s+deal\s+with\s+a\s+score\s+less\s+than\s+({NUMBER_RE})", re.I),
    re.compile(rf"minimum\s+score[^\d\-+]*({NUMBER_RE})", re.I),
]
MIN_APPROVALS_RE = re.compile(r"at\s+least\s+(\d+)\s+part(?:y|ies)\s+agree", re.I)
REPRESENT_PATTERNS = [
    re.compile(r'You\s+represent\s+the\s+["“]([^"”]+)["”]', re.I),
    re.compile(r'You\s+represent\s+["“]([^"”]+)["”]', re.I),
]
VETO_NAME_PATTERNS = [
    re.compile(r'Ensuring\s+(.+?)\s+approval\s+is\s+crucial\s+because\s+they\s+have\s+veto\s+power', re.I),
    re.compile(r'must\s+include\s+(.+?)\s+approval', re.I),
]


def _fmt_number(x: float, decimals: int = 2) -> str:
    x = float(x)
    if abs(x - round(x)) < 10 ** (-(decimals + 2)):
        return str(int(round(x)))
    return f"{x:.{decimals}f}".rstrip("0").rstrip(".")


def extract_archive(zip_path: str | Path, work_dir: str | Path) -> Path:
    zip_path = Path(zip_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / zip_path.stem
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out)
    # If archive has exactly one top-level folder, return it.
    children = [p for p in out.iterdir() if p.name != "__MACOSX"]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return out


def find_manifest(root: str | Path, manifest_name: Optional[str] = None) -> Optional[Path]:
    """Return a manifest when present, but do not require one.

    Prompt-only ZIP archives are supported.  When no manifest is available,
    ``load_scenario`` discovers stakeholder prompt files by looking for utility
    score tables inside .txt files.
    """
    root = Path(root)
    if manifest_name:
        p = root / manifest_name
        if p.exists():
            return p
        candidates = list(root.rglob(Path(manifest_name).name))
        if candidates:
            return candidates[0]
        raise FileNotFoundError(f"Could not find manifest {manifest_name!r} under {root}")
    candidates = [p for p in root.rglob("*.txt") if "initial_prompts" in p.name.lower()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: (p.name.lower() != "initial_prompts.txt", len(str(p))))
    return candidates[0]


def _resolve_prompt_path(path_text: str, manifest_path: Path, scenario_root: Path) -> Path:
    raw = Path(path_text.strip())
    tries = []
    if raw.is_absolute():
        tries.append(raw)
    tries.extend([
        manifest_path.parent / raw,
        scenario_root / raw,
        manifest_path.parent / raw.name,
        scenario_root / raw.name,
    ])
    for p in tries:
        if p.exists():
            return p.resolve()
    by_name = list(scenario_root.rglob(raw.name))
    if len(by_name) == 1:
        return by_name[0].resolve()
    if len(by_name) > 1:
        sibling = [p for p in by_name if p.parent == manifest_path.parent]
        if sibling:
            return sibling[0].resolve()
        return by_name[0].resolve()
    raise FileNotFoundError(f"Could not resolve prompt path {path_text!r} from manifest {manifest_path}")


def parse_manifest(manifest_path: str | Path, scenario_root: str | Path) -> List[dict]:
    manifest_path = Path(manifest_path)
    scenario_root = Path(scenario_root)
    rows = []
    for line in manifest_path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 4:
            raise ValueError(f"Manifest line needs 4 comma-separated fields: {line}")
        name, file_text, role, model = parts[:4]
        prompt_path = _resolve_prompt_path(file_text, manifest_path, scenario_root)
        rows.append({"name": name.strip(), "prompt_path": prompt_path, "role": role.strip(), "model": model.strip()})
    return rows


def parse_threshold(text: str) -> float:
    for pat in THRESHOLD_PATTERNS:
        m = pat.search(text)
        if m:
            return float(m.group(1))
    raise ValueError("Could not parse the minimum acceptable score from a prompt.")


def parse_utility_table(text: str) -> OrderedDict:
    issues = OrderedDict()
    for m in ISSUE_LINE_RE.finditer(text):
        issue = m.group(1)
        max_score = float(m.group(2))
        tail = m.group(3)
        opts = OrderedDict((om.group(1), float(om.group(2))) for om in OPTION_SCORE_RE.finditer(tail))
        if not opts:
            continue
        issues[issue] = {"max": max_score, "options": opts}
    if not issues:
        raise ValueError("No scoring lines like 'Issue A (max score 50): A1 (50), ...' were found.")
    return issues


def infer_party_name(text: str, prompt_path: str | Path) -> str:
    for pat in REPRESENT_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).strip()
    return Path(prompt_path).stem.replace("_", " ").strip()


def discover_prompt_rows(root: str | Path, default_model: str = "gpt-4o") -> List[dict]:
    """Discover stakeholder prompts when the ZIP contains no manifest."""
    root = Path(root)
    rows = []
    seen = set()
    for path in sorted(root.rglob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8-sig")
            parse_utility_table(text)
        except Exception:
            continue
        name = infer_party_name(text, path)
        key = name.casefold()
        if key in seen:
            raise ValueError(
                f"More than one scored prompt appears to represent {name!r}. "
                "Add an initial_prompts.txt manifest to disambiguate the files."
            )
        seen.add(key)
        rows.append({"name": name, "prompt_path": path.resolve(), "role": "player", "model": default_model})
    if not rows:
        raise FileNotFoundError(
            "No stakeholder prompt files were found. Put the scored stakeholder .txt prompts in the ZIP "
            "(an initial_prompts.txt manifest is optional)."
        )
    return rows


def _canonical_party_match(candidate: str, parties: Iterable[str]) -> Optional[str]:
    c = re.sub(r"\\s+", " ", candidate).strip().casefold()
    for p in parties:
        if re.sub(r"\\s+", " ", str(p)).strip().casefold() == c:
            return p
    return None


def infer_veto_parties_from_prompts(parties: OrderedDict) -> List[str]:
    found = []
    names = list(parties)
    for pdata in parties.values():
        text = pdata["prompt_text"]
        for pat in VETO_NAME_PATTERNS:
            for m in pat.finditer(text):
                matched = _canonical_party_match(m.group(1), names)
                if matched and matched not in found:
                    found.append(matched)
    return found


def load_scenario(
    zip_path: str | Path,
    work_dir: str | Path,
    manifest_name: Optional[str] = None,
    threshold_override: Optional[float] = None,
    default_model: str = "gpt-4o",
) -> dict:
    """Load a scenario from a ZIP.

    The ZIP only needs the stakeholder prompt text files.  A legacy
    initial_prompts*.txt manifest is used when present but is no longer required;
    initial_deal*.txt files are ignored because opening deals are generated later
    from the current moderator's utility table.
    """
    root = extract_archive(zip_path, work_dir)
    manifest = find_manifest(root, manifest_name)
    manifest_rows = parse_manifest(manifest, root) if manifest is not None else discover_prompt_rows(root, default_model)

    parties = OrderedDict()
    for row in manifest_rows:
        text = Path(row["prompt_path"]).read_text(encoding="utf-8-sig")
        issues = parse_utility_table(text)
        threshold = float(threshold_override) if threshold_override is not None else parse_threshold(text)
        parties[row["name"]] = {
            "name": row["name"],
            "role": row.get("role", "player"),
            "model": row.get("model") or default_model,
            "prompt_path": Path(row["prompt_path"]),
            "prompt_text": text,
            "threshold": threshold,
            "issues": issues,
        }

    first = next(iter(parties.values()))
    def natural_key(x):
        parts = re.split(r"(\\d+)", str(x))
        return [int(p) if p.isdigit() else p.lower() for p in parts]
    issue_order = sorted(first["issues"].keys(), key=natural_key)
    option_order = OrderedDict((i, sorted(first["issues"][i]["options"].keys(), key=natural_key)) for i in issue_order)
    for pname, pdata in parties.items():
        if set(pdata["issues"].keys()) != set(issue_order):
            raise ValueError(f"Issue set differs for party {pname}: {list(pdata['issues'])} vs {issue_order}")
        for issue in issue_order:
            if set(pdata["issues"][issue]["options"]) != set(option_order[issue]):
                raise ValueError(f"Option set differs for party {pname}, issue {issue}")

    inferred_veto = [p for p, d in parties.items() if "veto" in d["role"].lower()]
    for p in infer_veto_parties_from_prompts(parties):
        if p not in inferred_veto:
            inferred_veto.append(p)

    inferred_min_approvals = None
    for pdata in parties.values():
        m = MIN_APPROVALS_RE.search(pdata["prompt_text"])
        if m:
            inferred_min_approvals = int(m.group(1))
            break

    return {
        "root": root,
        "manifest": manifest,
        "manifest_rows": manifest_rows,
        "parties": parties,
        "issue_order": issue_order,
        "option_order": option_order,
        "inferred_veto_parties": inferred_veto,
        "inferred_moderators": list(parties),  # every stakeholder is tested as moderator
        "inferred_min_approvals": inferred_min_approvals,
    }

def utility_copy(scenario: dict) -> OrderedDict:
    out = OrderedDict()
    for p, pdata in scenario["parties"].items():
        out[p] = OrderedDict()
        for issue, idata in pdata["issues"].items():
            out[p][issue] = {
                "max": float(idata["max"]),
                "options": OrderedDict((o, float(v)) for o, v in idata["options"].items()),
            }
    return out


def thresholds_dict(scenario: dict) -> OrderedDict:
    return OrderedDict((p, float(d["threshold"])) for p, d in scenario["parties"].items())


def enumerate_deals(issue_order: List[str], option_order: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    for combo in itertools.product(*(option_order[i] for i in issue_order)):
        rows.append({**dict(zip(issue_order, combo)), "deal": ",".join(combo)})
    return pd.DataFrame(rows)


def score_deals(deals: pd.DataFrame, utility: dict, issue_order: List[str]) -> pd.DataFrame:
    scores = pd.DataFrame(index=deals.index)
    for party, ptable in utility.items():
        total = np.zeros(len(deals), dtype=float)
        for issue in issue_order:
            mapping = ptable[issue]["options"]
            total += deals[issue].map(mapping).to_numpy(dtype=float)
        scores[party] = total
    return scores


def acceptance_matrix(scores: pd.DataFrame, thresholds: dict, inclusive: bool = True) -> pd.DataFrame:
    acc = pd.DataFrame(index=scores.index)
    for party in scores.columns:
        if inclusive:
            acc[party] = scores[party] >= thresholds[party]
        else:
            acc[party] = scores[party] > thresholds[party]
    return acc


def vote_mask(acc: pd.DataFrame, veto_parties: Iterable[str], min_approvals: int) -> np.ndarray:
    veto_parties = list(veto_parties or [])
    mask = (acc.sum(axis=1).to_numpy() >= int(min_approvals))
    for p in veto_parties:
        if p not in acc.columns:
            raise KeyError(f"Veto party {p!r} not in parties {list(acc.columns)}")
        mask &= acc[p].to_numpy(dtype=bool)
    return mask


def deal_hamming(deal_a: Optional[str], deal_b: Optional[str]) -> float:
    if deal_a is None or deal_b is None:
        return np.nan
    a = deal_a.split(",")
    b = deal_b.split(",")
    if len(a) != len(b):
        return np.nan
    return float(np.mean([x != y for x, y in zip(a, b)]))


def jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def choose_nbs(deals: pd.DataFrame, scores: pd.DataFrame, thresholds: dict, unanimous_mask: np.ndarray) -> Tuple[Optional[str], Optional[int], float]:
    idx = np.flatnonzero(unanimous_mask)
    if len(idx) == 0:
        return None, None, np.nan
    gains = scores.iloc[idx].copy()
    for p in gains.columns:
        gains[p] = np.maximum(gains[p].to_numpy() - thresholds[p], 0.0)
    # log(product) handles large products and zeros; add tiny epsilon only for ranking zero-gain ties.
    prod = gains.prod(axis=1).to_numpy(dtype=float)
    welfare = scores.iloc[idx].sum(axis=1).to_numpy(dtype=float)
    order = np.lexsort((welfare, prod))
    chosen_pos = order[-1]
    chosen_idx = int(idx[chosen_pos])
    return deals.loc[chosen_idx, "deal"], chosen_idx, float(prod[chosen_pos])


def choose_welfare_pass(deals: pd.DataFrame, scores: pd.DataFrame, pass_mask: np.ndarray) -> Tuple[Optional[str], Optional[int], float]:
    idx = np.flatnonzero(pass_mask)
    if len(idx) == 0:
        return None, None, np.nan
    welfare = scores.iloc[idx].sum(axis=1).to_numpy(dtype=float)
    chosen_pos = int(np.argmax(welfare))
    chosen_idx = int(idx[chosen_pos])
    return deals.loc[chosen_idx, "deal"], chosen_idx, float(welfare[chosen_pos])


def analyze_utility(
    scenario: dict,
    utility: dict,
    veto_parties: Optional[List[str]] = None,
    min_approvals: Optional[int] = None,
    inclusive: bool = True,
    deals: Optional[pd.DataFrame] = None,
) -> dict:
    if deals is None:
        deals = enumerate_deals(scenario["issue_order"], scenario["option_order"])
    thresholds = thresholds_dict(scenario)
    scores = score_deals(deals, utility, scenario["issue_order"])
    acc = acceptance_matrix(scores, thresholds, inclusive=inclusive)
    veto = scenario["inferred_veto_parties"] if veto_parties is None else veto_parties
    minimum = scenario["inferred_min_approvals"] if min_approvals is None else min_approvals
    if minimum is None:
        minimum = len(scenario["parties"])
    p_mask = vote_mask(acc, veto, minimum)
    u_mask = acc.all(axis=1).to_numpy(dtype=bool)
    nbs_deal, nbs_idx, nbs_product = choose_nbs(deals, scores, thresholds, u_mask)
    welfare_deal, welfare_idx, welfare_value = choose_welfare_pass(deals, scores, p_mask)
    return {
        "deals": deals,
        "scores": scores,
        "acceptance": acc,
        "pass_mask": p_mask,
        "unanimous_mask": u_mask,
        "pass_count": int(p_mask.sum()),
        "unanimous_count": int(u_mask.sum()),
        "party_acceptable_counts": {p: int(acc[p].sum()) for p in acc.columns},
        "nbs_deal": nbs_deal,
        "nbs_idx": nbs_idx,
        "nbs_product": nbs_product,
        "welfare_deal": welfare_deal,
        "welfare_idx": welfare_idx,
        "welfare_value": welfare_value,
        "veto_parties": list(veto),
        "min_approvals": int(minimum),
    }


def compare_analysis(base: dict, alt: dict) -> dict:
    out = {
        "vote_jaccard": jaccard(base["pass_mask"], alt["pass_mask"]),
        "unanimous_jaccard": jaccard(base["unanimous_mask"], alt["unanimous_mask"]),
        "pass_count": alt["pass_count"],
        "pass_count_delta": alt["pass_count"] - base["pass_count"],
        "unanimous_count": alt["unanimous_count"],
        "unanimous_count_delta": alt["unanimous_count"] - base["unanimous_count"],
        "nbs_deal": alt["nbs_deal"],
        "nbs_same": alt["nbs_deal"] == base["nbs_deal"],
        "nbs_hamming": deal_hamming(base["nbs_deal"], alt["nbs_deal"]),
        "welfare_deal": alt["welfare_deal"],
        "welfare_same": alt["welfare_deal"] == base["welfare_deal"],
        "welfare_hamming": deal_hamming(base["welfare_deal"], alt["welfare_deal"]),
    }
    rank_corrs = []
    for p in base["scores"].columns:
        x = base["scores"][p].to_numpy()
        y = alt["scores"][p].to_numpy()
        if spearmanr is not None:
            corr = float(spearmanr(x, y).statistic)
        else:
            corr = float(pd.Series(x).rank().corr(pd.Series(y).rank(), method="pearson"))
        out[f"rank_corr_{p}"] = corr
        rank_corrs.append(corr)
    out["mean_rank_corr"] = float(np.nanmean(rank_corrs))
    return out


def _renormalize_party_weights(party_table: dict, multipliers: Dict[str, float]) -> dict:
    out = copy.deepcopy(party_table)
    issues = list(out.keys())
    old_weights = np.array([float(out[i]["max"]) for i in issues], dtype=float)
    target_total = old_weights.sum()
    raw = np.array([old_weights[k] * float(multipliers.get(i, 1.0)) for k, i in enumerate(issues)], dtype=float)
    if raw.sum() <= 0:
        return out
    new_weights = raw * (target_total / raw.sum())
    for k, issue in enumerate(issues):
        old_w = old_weights[k]
        new_w = float(new_weights[k])
        opts = out[issue]["options"]
        if old_w > 0:
            ratios = {o: float(v) / old_w for o, v in opts.items()}
        else:
            ratios = {o: 0.0 for o in opts}
        out[issue]["max"] = new_w
        out[issue]["options"] = OrderedDict((o, ratios[o] * new_w) for o in opts)
    return out


def perturb_weight(utility: dict, party: str, issue: str, rel_change: float) -> OrderedDict:
    out = copy.deepcopy(utility)
    out[party] = _renormalize_party_weights(out[party], {issue: 1.0 + float(rel_change)})
    return out


def apply_weight_multipliers(utility: dict, factors: Dict[Tuple[str, str], float]) -> OrderedDict:
    out = copy.deepcopy(utility)
    parties = list(out.keys())
    for p in parties:
        p_factors = {issue: factors.get((p, issue), 1.0) for issue in out[p]}
        out[p] = _renormalize_party_weights(out[p], p_factors)
    return out


def shape_parameters(utility: dict) -> List[dict]:
    """Return tied intermediate score levels as shape parameters. Structural 0 and max levels are excluded."""
    params = []
    for party, ptable in utility.items():
        for issue, idata in ptable.items():
            mx = float(idata["max"])
            if mx <= 0:
                continue
            groups = {}
            for opt, score in idata["options"].items():
                s = float(score)
                if s <= 0 or abs(s - mx) < 1e-12:
                    continue
                key = round(s, 12)
                groups.setdefault(key, []).append(opt)
            for score, options in sorted(groups.items(), reverse=True):
                params.append({"party": party, "issue": issue, "score": float(score), "options": options})
    return params


def perturb_shape_level(utility: dict, party: str, issue: str, options: List[str], rel_change: float) -> OrderedDict:
    out = copy.deepcopy(utility)
    idata = out[party][issue]
    mx = float(idata["max"])
    current = float(idata["options"][options[0]])
    proposed = current * (1.0 + float(rel_change))

    # Preserve original ordinal structure: keep the tied group between nearest lower/higher distinct levels.
    original_values = sorted(set(float(v) for v in idata["options"].values()))
    lower = max([v for v in original_values if v < current], default=0.0)
    upper = min([v for v in original_values if v > current], default=mx)
    eps = max(mx, 1.0) * 1e-6
    lo = lower + eps if lower < current else lower
    hi = upper - eps if upper > current else upper
    proposed = min(max(proposed, lo), hi)
    for opt in options:
        idata["options"][opt] = float(proposed)
    return out


def oat_screen(
    scenario: dict,
    base_utility: dict,
    rel_delta: float = 0.10,
    include_shape: bool = True,
    veto_parties: Optional[List[str]] = None,
    min_approvals: Optional[int] = None,
    inclusive: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    deals = enumerate_deals(scenario["issue_order"], scenario["option_order"])
    base = analyze_utility(scenario, base_utility, veto_parties, min_approvals, inclusive, deals=deals)
    rows = []

    # Issue-weight parameters: skip zero-weight issues.
    for party, ptable in base_utility.items():
        for issue, idata in ptable.items():
            if float(idata["max"]) <= 0:
                continue
            for sign in (-1, +1):
                alt_u = perturb_weight(base_utility, party, issue, sign * rel_delta)
                alt = analyze_utility(scenario, alt_u, veto_parties, min_approvals, inclusive, deals=deals)
                comp = compare_analysis(base, alt)
                rows.append({
                    "parameter_type": "issue_weight",
                    "party": party,
                    "issue": issue,
                    "options": "ALL",
                    "direction": f"{sign*rel_delta:+.1%}",
                    "rel_change": sign * rel_delta,
                    **comp,
                })

    if include_shape:
        for par in shape_parameters(base_utility):
            for sign in (-1, +1):
                alt_u = perturb_shape_level(base_utility, par["party"], par["issue"], par["options"], sign * rel_delta)
                alt = analyze_utility(scenario, alt_u, veto_parties, min_approvals, inclusive, deals=deals)
                comp = compare_analysis(base, alt)
                rows.append({
                    "parameter_type": "within_issue_shape",
                    "party": par["party"],
                    "issue": par["issue"],
                    "options": "+".join(par["options"]),
                    "direction": f"{sign*rel_delta:+.1%}",
                    "rel_change": sign * rel_delta,
                    **comp,
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["vote_set_change"] = 1.0 - df["vote_jaccard"]
        df["unanimous_set_change"] = 1.0 - df["unanimous_jaccard"]
        df = df.sort_values(["vote_set_change", "nbs_hamming", "unanimous_set_change"], ascending=False).reset_index(drop=True)
    return df, base


def monte_carlo_weight_sensitivity(
    scenario: dict,
    base_utility: dict,
    n_samples: int = 5000,
    rel_delta: float = 0.10,
    seed: int = 123,
    veto_parties: Optional[List[str]] = None,
    min_approvals: Optional[int] = None,
    inclusive: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    deals = enumerate_deals(scenario["issue_order"], scenario["option_order"])
    base = analyze_utility(scenario, base_utility, veto_parties, min_approvals, inclusive, deals=deals)
    params = [(p, i) for p, pt in base_utility.items() for i, idata in pt.items() if float(idata["max"]) > 0]
    factor_mat = rng.uniform(1.0 - rel_delta, 1.0 + rel_delta, size=(n_samples, len(params)))
    metric_rows = []
    for s in range(n_samples):
        factor_dict = {params[j]: float(factor_mat[s, j]) for j in range(len(params))}
        alt_u = apply_weight_multipliers(base_utility, factor_dict)
        alt = analyze_utility(scenario, alt_u, veto_parties, min_approvals, inclusive, deals=deals)
        comp = compare_analysis(base, alt)
        metric_rows.append({"sample_id": s, **comp})
    metrics = pd.DataFrame(metric_rows)
    metrics["vote_set_change"] = 1.0 - metrics["vote_jaccard"]
    metrics["unanimous_set_change"] = 1.0 - metrics["unanimous_jaccard"]
    factor_cols = [f"{p}|{i}" for p, i in params]
    factors = pd.DataFrame(factor_mat, columns=factor_cols)
    factors.insert(0, "sample_id", np.arange(n_samples))
    return metrics, factors, base


def utility_from_mc_row(base_utility: dict, factor_row: pd.Series) -> OrderedDict:
    factors = {}
    for col, val in factor_row.items():
        if col == "sample_id":
            continue
        party, issue = col.split("|", 1)
        factors[(party, issue)] = float(val)
    return apply_weight_multipliers(base_utility, factors)


def select_mc_representatives(metrics: pd.DataFrame, factors: pd.DataFrame, n_random_representatives: int = 3) -> List[dict]:
    if len(metrics) == 0:
        return []
    selected = []
    # Quantiles across vote-set change: stable / median / sensitive by default.
    qs = np.linspace(0.05, 0.95, max(n_random_representatives, 1))
    used = set()
    for q in qs:
        target = float(metrics["vote_set_change"].quantile(q))
        idx = (metrics["vote_set_change"] - target).abs().sort_values().index
        chosen = next(int(i) for i in idx if int(metrics.loc[i, "sample_id"]) not in used)
        sid = int(metrics.loc[chosen, "sample_id"])
        used.add(sid)
        selected.append({
            "name": f"mc_q{int(round(q*100)):02d}_sample{sid}",
            "sample_id": sid,
            "metric_row": metrics.loc[chosen].to_dict(),
            "factor_row": factors.loc[factors["sample_id"] == sid].iloc[0].copy(),
        })
    return selected


def select_oat_scenarios(oat_df: pd.DataFrame, base_utility: dict, n_top: int = 4) -> List[dict]:
    if oat_df.empty or n_top <= 0:
        return []
    # Keep the strongest direction for each unique parameter.
    best = (
        oat_df.sort_values(["vote_set_change", "nbs_hamming", "unanimous_set_change"], ascending=False)
        .drop_duplicates(subset=["parameter_type", "party", "issue", "options"], keep="first")
        .head(n_top)
    )
    out = []
    for _, row in best.iterrows():
        if row["parameter_type"] == "issue_weight":
            u = perturb_weight(base_utility, row["party"], row["issue"], float(row["rel_change"]))
        else:
            options = str(row["options"]).split("+")
            u = perturb_shape_level(base_utility, row["party"], row["issue"], options, float(row["rel_change"]))
        safe_party = re.sub(r"\W+", "_", str(row["party"])).strip("_")
        safe_opts = re.sub(r"\W+", "_", str(row["options"])).strip("_")
        name = f"oat_{row['parameter_type']}_{safe_party}_{row['issue']}_{safe_opts}_{row['direction'].replace('%','pct').replace('+','plus').replace('-','minus')}"
        out.append({"name": name, "utility": u, "metric_row": row.to_dict()})
    return out


def rewrite_prompt_scores(prompt_text: str, party_utility: dict, threshold: Optional[float] = None, decimals: int = 2) -> str:
    def repl(m: re.Match) -> str:
        issue = m.group(1)
        if issue not in party_utility:
            return m.group(0)
        idata = party_utility[issue]
        options = ", ".join(f"{o} ({_fmt_number(v, decimals)})" for o, v in idata["options"].items())
        return f"Issue {issue} (max score {_fmt_number(idata['max'], decimals)}): {options}"

    out = ISSUE_LINE_RE.sub(repl, prompt_text)
    if threshold is not None:
        # Replace only the explicit minimum acceptance statement first.
        out, n = re.subn(
            rf"(cannot\s+accept\s+any\s+deal\s+with\s+a\s+score\s+less\s+than\s+)({NUMBER_RE})",
            lambda m: m.group(1) + _fmt_number(threshold, decimals),
            out,
            count=1,
            flags=re.I,
        )
        # Also keep the no-deal score aligned if the common template is present.
        out = re.sub(
            rf"(If\s+no\s+deal\s+is\s+achieved,\s+your\s+score\s+is\s+)({NUMBER_RE})",
            lambda m: m.group(1) + _fmt_number(threshold, decimals),
            out,
            count=1,
            flags=re.I,
        )
    return out


def _party_max_total(party_utility: dict) -> float:
    return float(sum(max(float(v) for v in idata["options"].values()) for idata in party_utility.values()))


def construct_preferred_initial_deal(
    scenario: dict,
    utility: dict,
    moderator: str,
    thresholds: Optional[dict] = None,
    tie_break: str = "other_welfare",
) -> dict:
    """Construct the moderator's opening deal directly from stakeholder preferences.

    Primary rule: maximize the moderator's own additive utility.  Small score
    perturbations therefore only alter the opening deal when they genuinely alter
    the moderator's preferred option set.

    If several full deals give exactly the same maximum moderator utility, the
    default ``other_welfare`` tie-break uses the other stakeholders' preferences
    *without sacrificing any moderator utility*: first maximize the number of
    parties meeting their threshold, then the minimum normalized utility among
    other parties, then normalized social welfare, and finally lexical deal order
    for deterministic reproducibility.
    """
    if moderator not in utility:
        raise KeyError(f"Unknown moderator {moderator!r}")
    thresholds = thresholds or thresholds_dict(scenario)
    deals = enumerate_deals(scenario["issue_order"], scenario["option_order"])
    scores = score_deals(deals, utility, scenario["issue_order"])
    mod_scores = scores[moderator].to_numpy(dtype=float)
    best_score = float(np.max(mod_scores))
    candidate_idx = np.where(np.isclose(mod_scores, best_score, rtol=0.0, atol=1e-9))[0]

    if tie_break not in {"other_welfare", "lexical"}:
        raise ValueError("tie_break must be 'other_welfare' or 'lexical'")

    candidates = []
    max_totals = {p: max(_party_max_total(utility[p]), 1e-12) for p in utility}
    for idx in candidate_idx:
        row_scores = scores.loc[idx]
        accept_count = int(sum(float(row_scores[p]) >= float(thresholds[p]) for p in utility))
        others = [p for p in utility if p != moderator]
        normalized = {p: float(row_scores[p]) / max_totals[p] for p in utility}
        other_min = min((normalized[p] for p in others), default=normalized[moderator])
        social = sum(normalized.values())
        candidates.append({
            "idx": int(idx),
            "deal": str(deals.loc[idx, "deal"]),
            "moderator_score": float(row_scores[moderator]),
            "accept_count": accept_count,
            "other_min_normalized": float(other_min),
            "normalized_social_welfare": float(social),
            "scores": {p: float(row_scores[p]) for p in utility},
        })

    if tie_break == "lexical":
        chosen = sorted(candidates, key=lambda x: x["deal"])[0]
    else:
        chosen = sorted(
            candidates,
            key=lambda x: (-x["accept_count"], -x["other_min_normalized"], -x["normalized_social_welfare"], x["deal"]),
        )[0]

    chosen = dict(chosen)
    chosen.update({
        "moderator": moderator,
        "n_equally_best_deals": len(candidates),
        "tie_break": tie_break,
        "moderator_threshold": float(thresholds[moderator]),
        "moderator_meets_threshold": bool(chosen["moderator_score"] >= float(thresholds[moderator])),
    })
    return chosen


def construct_all_initial_deals(
    scenario: dict,
    utility: dict,
    tie_break: str = "other_welfare",
) -> pd.DataFrame:
    rows = [construct_preferred_initial_deal(scenario, utility, p, tie_break=tie_break) for p in utility]
    return pd.DataFrame(rows)


def _moderator_role(base_role: str, is_moderator: bool, is_veto: bool = False) -> str:
    roles = [r.strip() for r in str(base_role or "").split("&") if r.strip()]
    roles = [r for r in roles if r not in {"voting_moderator", "veto"}]
    if not roles:
        roles = ["player"]
    if is_veto:
        roles.append("veto")
    if is_moderator:
        roles.append("voting_moderator")
    return "&".join(dict.fromkeys(roles))


def write_prompt_package(
    scenario: dict,
    utility: dict,
    out_dir: str | Path,
    moderator: str,
    thresholds: Optional[dict] = None,
    decimals: int = 2,
    initial_deal_tie_break: str = "other_welfare",
) -> dict:
    """Write one runnable package for one condition × one moderator.

    No external initial-deal file is consumed.  The opening deal and moderator
    role are generated automatically from the current (possibly perturbed)
    stakeholder preferences.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir = out_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    thresholds = thresholds or thresholds_dict(scenario)

    initial = construct_preferred_initial_deal(
        scenario, utility, moderator, thresholds=thresholds, tie_break=initial_deal_tie_break
    )

    manifest_lines = []
    for party, pdata in scenario["parties"].items():
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", party).strip("_") or "party"
        ppath = (prompt_dir / f"{safe_name}.txt").resolve()
        rewritten = rewrite_prompt_scores(pdata["prompt_text"], utility[party], threshold=thresholds[party], decimals=decimals)
        ppath.write_text(rewritten, encoding="utf-8")
        role = _moderator_role(
            pdata.get("role", "player"),
            is_moderator=(party == moderator),
            is_veto=(party in scenario.get("inferred_veto_parties", [])),
        )
        manifest_lines.append(f"{party},{ppath},{role},{pdata['model']}")

    manifest_path = (out_dir / "initial_prompts.txt").resolve()
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    initial_path = (out_dir / "initial_deal.txt").resolve()
    initial_path.write_text(initial["deal"] + "\n", encoding="utf-8")

    metadata = {
        "moderator": moderator,
        "initial_deal": initial["deal"],
        "initial_deal_construction": initial,
        "thresholds": thresholds,
        "utility": utility,
        "manifest": str(manifest_path),
        "initial_deal_file": str(initial_path),
    }
    meta_path = out_dir / "scenario.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "dir": out_dir,
        "manifest": manifest_path,
        "initial_deal": initial_path,
        "metadata": meta_path,
        "initial_deal_info": initial,
    }


def generate_selected_prompt_packages(
    scenario: dict,
    base_utility: dict,
    oat_df: pd.DataFrame,
    mc_metrics: Optional[pd.DataFrame],
    mc_factors: Optional[pd.DataFrame],
    out_root: str | Path,
    n_top_oat: int = 4,
    n_mc: int = 3,
    include_baseline: bool = True,
    moderators: Optional[Iterable[str]] = None,
    initial_deal_tie_break: str = "other_welfare",
) -> pd.DataFrame:
    """Export selected sensitivity conditions for *every* requested moderator."""
    out_root = Path(out_root)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    moderators = list(moderators) if moderators is not None else list(scenario["parties"])
    unknown = [m for m in moderators if m not in scenario["parties"]]
    if unknown:
        raise KeyError(f"Unknown moderators: {unknown}")

    conditions = []
    if include_baseline:
        conditions.append({"scenario": "baseline", "type": "baseline", "utility": base_utility, "metric_row": {}})
    for spec in select_oat_scenarios(oat_df, base_utility, n_top=n_top_oat):
        conditions.append({"scenario": spec["name"], "type": "OAT", "utility": spec["utility"], "metric_row": spec["metric_row"]})
    if mc_metrics is not None and mc_factors is not None and n_mc > 0:
        for spec in select_mc_representatives(mc_metrics, mc_factors, n_random_representatives=n_mc):
            conditions.append({
                "scenario": spec["name"],
                "type": "MC",
                "utility": utility_from_mc_row(base_utility, spec["factor_row"]),
                "metric_row": spec["metric_row"],
            })

    rows = []
    for cond in conditions:
        for moderator in moderators:
            safe_mod = re.sub(r"[^A-Za-z0-9_.-]+", "_", moderator).strip("_") or "moderator"
            pkg = write_prompt_package(
                scenario,
                cond["utility"],
                out_root / cond["scenario"] / f"moderator_{safe_mod}",
                moderator=moderator,
                initial_deal_tie_break=initial_deal_tie_break,
            )
            info = pkg["initial_deal_info"]
            row = {
                "scenario": cond["scenario"],
                "type": cond["type"],
                "moderator": moderator,
                "manifest": str(pkg["manifest"]),
                "initial_deal": str(pkg["initial_deal"]),
                "initial_deal_text": info["deal"],
                "initial_deal_moderator_score": info["moderator_score"],
                "initial_deal_n_ties": info["n_equally_best_deals"],
                "initial_deal_accept_count": info["accept_count"],
            }
            row.update({f"metric_{k}": v for k, v in cond["metric_row"].items() if np.isscalar(v)})
            rows.append(row)

    manifest_df = pd.DataFrame(rows)
    manifest_df.to_csv(out_root / "selected_scenarios.csv", index=False)
    return manifest_df

def baseline_summary_df(base: dict) -> pd.DataFrame:
    rows = [{"metric": f"acceptable_deals_{p}", "value": n} for p, n in base["party_acceptable_counts"].items()]
    rows += [
        {"metric": "unanimous_acceptable_deals", "value": base["unanimous_count"]},
        {"metric": "voting_rule_passing_deals", "value": base["pass_count"]},
        {"metric": "all_party_NBS", "value": base["nbs_deal"]},
        {"metric": "passing_social_welfare_max", "value": base["welfare_deal"]},
    ]
    return pd.DataFrame(rows)


def save_analysis_outputs(out_dir: str | Path, base: dict, oat_df: pd.DataFrame, mc_metrics: Optional[pd.DataFrame] = None, mc_factors: Optional[pd.DataFrame] = None) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_summary_df(base).to_csv(out_dir / "baseline_summary.csv", index=False)
    base["scores"].assign(deal=base["deals"]["deal"]).to_csv(out_dir / "baseline_all_deal_scores.csv", index=False)
    oat_df.to_csv(out_dir / "oat_sensitivity.csv", index=False)
    if mc_metrics is not None:
        mc_metrics.to_csv(out_dir / "mc_metrics.csv", index=False)
    if mc_factors is not None:
        mc_factors.to_csv(out_dir / "mc_weight_factors.csv", index=False)
