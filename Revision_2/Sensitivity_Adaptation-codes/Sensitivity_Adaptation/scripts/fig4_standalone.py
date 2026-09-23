"""Fig. 4 equilibrium characterization, reading the extraction package outputs."""
import itertools
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory

SCORES_FILE = "/mnt/user-data/uploads/Case2_OPM_Workshop_Alex_final_scores.csv"
DEALS_FILE = "/home/claude/out_v3/Results_extracted_all_deals.csv"
SEPARATOR = ";"
DISAGREEMENT = 65
EXPORT = "/home/claude/out_v3/Fig4_deal_metrics.csv"
FIG_OUT = "/home/claude/out_v3/Fig_4.png"

pref_df = pd.read_csv(SCORES_FILE, sep=SEPARATOR)
res_df = pd.read_csv(DEALS_FILE, sep=SEPARATOR)

stakeholders = pref_df["Stakeholder"].astype(str).tolist()
disagreement = {sh: DISAGREEMENT for sh in stakeholders}

issues, options_per_issue, pref = [], {}, {sh: {} for sh in stakeholders}
for col in pref_df.columns[1:]:
    issue, option = col[0], int(col[1:])
    if issue not in options_per_issue:
        issues.append(issue)
        options_per_issue[issue] = []
    options_per_issue[issue].append(option)
    for _, row in pref_df.iterrows():
        sh = str(row["Stakeholder"])
        pref[sh].setdefault(issue, {})[option] = float(row[col])
for issue in issues:
    options_per_issue[issue] = sorted(options_per_issue[issue])

all_packages = list(itertools.product(*[options_per_issue[i] for i in issues]))

packages_data = []
for pkg in all_packages:
    utils = tuple(sum(pref[sh][issues[i]][pkg[i]] for i in range(len(issues)))
                  for sh in stakeholders)
    gains = tuple(utils[i] - disagreement[stakeholders[i]] for i in range(len(stakeholders)))
    is_ir = all(g >= 0 for g in gains)
    nash = float(np.prod(gains)) if is_ir else None
    packages_data.append({"Package": pkg, "Utilities": utils, "Disagreement_Gains": gains,
                          "Is_Individually_Rational": is_ir, "Nash_Product": nash})

# NBS over individually rational packages; ties broken by total welfare, as in
# choose_nbs_fast of the Table 2 script, so both report the same benchmark.
ir_packages = [p for p in packages_data if p["Is_Individually_Rational"]]
nbs_package, max_nash = None, None
if ir_packages:
    max_nash = max(p["Nash_Product"] for p in ir_packages)
    tied = [p for p in ir_packages if p["Nash_Product"] == max_nash]
    nbs_package = max(tied, key=lambda p: sum(p["Utilities"]))
    if len(tied) > 1:
        print(f"NOTE: {len(tied)} packages tie on the Nash product; broken by total welfare")
    if max_nash <= 0:
        print("WARNING: the maximal Nash product is 0; at least one stakeholder sits "
              "exactly on the disagreement point, so the benchmark is degenerate")
else:
    gaps = [max(disagreement[stakeholders[i]] - p["Utilities"][i]
                for i in range(len(stakeholders))) for p in packages_data]
    j = int(np.argmin(gaps))
    print(f"No individually rational package exists; no NBS. Infeasibility gap "
          f"{gaps[j]:.1f} utility points at {packages_data[j]['Package']}")

pareto_front = []
for pkg1 in packages_data:
    u1 = pkg1["Utilities"]
    if any(all(f >= x for f, x in zip(pf["Utilities"], u1))
           and any(f > x for f, x in zip(pf["Utilities"], u1)) for pf in pareto_front):
        continue
    pareto_front = [pf for pf in pareto_front
                    if not (all(x >= f for x, f in zip(u1, pf["Utilities"]))
                            and any(x > f for x, f in zip(u1, pf["Utilities"])))]
    pareto_front.append(pkg1)
pareto_packages_set = {p["Package"] for p in pareto_front}

print(f"{len(all_packages)} packages | {len(pareto_front)} Pareto optimal | "
      f"{len(ir_packages)} individually rational")
if nbs_package:
    print(f"NBS {nbs_package['Package']} utilities {nbs_package['Utilities']} "
          f"Nash product {max_nash:,.0f}")

by_package = {p["Package"]: p for p in packages_data}
rows = []
for _, row in res_df.iterrows():
    pkg = tuple(int(row[i]) for i in issues)
    info = by_package.get(pkg)
    if info is None:
        print(f"WARNING: deal {pkg} in round {row['Round']} is not a valid package")
        continue
    rec = {"Round": row["Round"], "Package": pkg, "Utilities": info["Utilities"],
           "Is_Individually_Rational": info["Is_Individually_Rational"],
           "Nash_Product": info["Nash_Product"],
           "Is_Pareto_Optimal": pkg in pareto_packages_set}
    if nbs_package is not None:
        d = [nbs_package["Utilities"][i] - info["Utilities"][i] for i in range(len(stakeholders))]
        rec.update({
            "Is_NBS": pkg == nbs_package["Package"],
            "Hamming_to_NBS": sum(a != b for a, b in zip(pkg, nbs_package["Package"])),
            "Manhattan_to_NBS": sum(abs(x) for x in d),
            "Euclidean_to_NBS": math.sqrt(sum(x * x for x in d)),
            "Nash_Loss_Pct": (abs((info["Nash_Product"] - max_nash) / max_nash) * 100
                              if info["Is_Individually_Rational"] and max_nash else np.nan),
        })
    rows.append(rec)

res_analysis_df = pd.DataFrame(rows)
res_analysis_df.to_csv(EXPORT, index=False)
n = len(res_analysis_df)
print(f"\n{n} deals across {res_analysis_df['Round'].nunique()} rounds | "
      f"Pareto {res_analysis_df['Is_Pareto_Optimal'].sum()} "
      f"({res_analysis_df['Is_Pareto_Optimal'].mean()*100:.1f}%) | "
      f"IR {res_analysis_df['Is_Individually_Rational'].sum()} "
      f"({res_analysis_df['Is_Individually_Rational'].mean()*100:.1f}%)")

# ---------------------------------------------------------------- figure
FS = dict(sub=30, tick=25, bar=30, ham=35, badge=40, leg=30)
plt.rcParams["font.family"] = "sans-serif"
fmt_num = lambda x: f"{x:.1f}".rstrip("0").rstrip(".")

counts = res_analysis_df["Package"].value_counts()
deal_counts = counts[counts > 1]
if nbs_package is None or len(deal_counts) == 0:
    raise SystemExit("Nothing to plot")

top = []
for pkg, count in deal_counts.items():
    info = by_package[pkg]
    is_ir = info["Is_Individually_Rational"]
    loss = (abs((info["Nash_Product"] - max_nash) / max_nash) * 100
            if is_ir and max_nash else np.nan)
    d = [nbs_package["Utilities"][i] - info["Utilities"][i] for i in range(len(stakeholders))]
    top.append({"Package": str(pkg), "Count": count, "Loss_Pct": loss,
                "Loss_For_Plot": 0.0 if np.isnan(loss) else loss,
                "Loss_Label": "Not IR" if np.isnan(loss) else f"{loss:.1f}%",
                "Is_Pareto": pkg in pareto_packages_set, "Is_IR": is_ir,
                "Hamming": sum(a != b for a, b in zip(pkg, nbs_package["Package"])),
                "Euclidean": math.sqrt(sum(x * x for x in d)),
                "Manhattan": sum(abs(x) for x in d)})
df = (pd.DataFrame(top)
      .sort_values(["Count", "Loss_For_Plot", "Package"], ascending=[True, False, True])
      .reset_index(drop=True))

H = 3.8 + len(df) * 1.05
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, H),
                               sharey=True, gridspec_kw={"wspace": 0.0})
y = np.arange(len(df))
bh, c_loss, c_euc, c_man, c_nir = 0.35, "steelblue", "grey", "coral", "#9ebbd0"

max_loss = max(df["Loss_Pct"].dropna().max() if df["Loss_Pct"].notna().any() else 1.0, 1.0)
bars1 = ax1.barh(y, df["Loss_For_Plot"], height=0.6, color=c_loss,
                 edgecolor="black", linewidth=1.2)
ax1.axvline(0, color="black", linewidth=2)
for i, bar in enumerate(bars1):
    lab = df.loc[i, "Loss_Label"]
    x = max_loss * 0.05 if lab == "Not IR" else bar.get_width() + max_loss * 0.05
    ax1.text(x, bar.get_y() + bar.get_height() / 2, lab, va="center", ha="right",
             fontsize=FS["bar"], fontweight="bold")
ax1.set_title("Efficiency Loss vs.\nOptimal Nash (%)", x=0.36, pad=14,
              fontsize=FS["sub"], fontweight="bold")
ax1.set_xlim(max_loss * 1.35, 0)
ax1.grid(axis="x", linestyle="--", alpha=0.7)
ax1.tick_params(axis="x", labelsize=FS["tick"])

max_dist = max(df["Euclidean"].max(), df["Manhattan"].max(), 1.0)
b_euc = ax2.barh(y - bh / 2, df["Euclidean"], height=bh, color=c_euc,
                 edgecolor="black", linewidth=1.2)
b_man = ax2.barh(y + bh / 2, df["Manhattan"], height=bh, color=c_man,
                 edgecolor="black", linewidth=1.2)
ax2.axvline(0, color="black", linewidth=2)
tr = blended_transform_factory(ax2.transAxes, ax2.transData)
for i, (e, m, h) in enumerate(zip(df["Euclidean"], df["Manhattan"], df["Hamming"])):
    ax2.text(e + max_dist * 0.02, y[i] - bh / 2, fmt_num(e), va="center", ha="left",
             fontsize=FS["bar"], fontweight="bold")
    ax2.text(m + max_dist * 0.02, y[i] + bh / 2, fmt_num(m), va="center", ha="left",
             fontsize=FS["bar"], fontweight="bold")
    ax2.text(1.04, y[i], f"Hamming: {h}", transform=tr, va="center", ha="left",
             color="#333333", fontsize=FS["ham"], fontstyle="italic", clip_on=False,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.5))
ax2.set_title("Distance from Optimal Nash\n(Utility Points)", x=0.64, pad=14,
              fontsize=FS["sub"], fontweight="bold")
ax2.set_xlim(0, max_dist * 1.25)
ax2.grid(axis="x", linestyle="--", alpha=0.7)
ax2.tick_params(axis="x", labelsize=FS["tick"])

for i, row in df.iterrows():
    if not row["Is_Pareto"]:
        for b in (bars1[i], b_euc[i], b_man[i]):
            b.set_hatch("///")
    if not row["Is_IR"]:
        bars1[i].set_facecolor(c_nir)
        bars1[i].set_alpha(0.65)
        b_euc[i].set_alpha(0.45)
        b_man[i].set_alpha(0.45)

ax1.set_yticks(y)
ax1.set_yticklabels(
    [f"{r['Package']}\n(Occurred {r['Count']}x) | "
     f"[{'PARETO OPTIMAL' if r['Is_Pareto'] else 'SUB-OPTIMAL'}; "
     f"{'IR' if r['Is_IR'] else 'NOT IR'}]" for _, r in df.iterrows()],
    fontsize=FS["tick"])
ax1.tick_params(axis="y", pad=40)
plt.setp(ax1.get_yticklabels(), ha="right")
ax2.tick_params(axis="y", left=False, labelleft=False)
for ax in (ax1, ax2):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

legend = fig.legend(handles=[
    mpatches.Patch(facecolor=c_euc, edgecolor="black", label="Euclidean Dist. (Fairness Penalty)"),
    mpatches.Patch(facecolor=c_man, edgecolor="black", label="Manhattan Dist. (Absolute Points Lost)"),
    mpatches.Patch(facecolor="white", edgecolor="black", hatch="////", label="Not Pareto Optimal"),
    mpatches.Patch(facecolor=c_nir, edgecolor="black", alpha=0.65, label="Not individually rational"),
], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08), fontsize=FS["leg"],
    frameon=True, columnspacing=3.0, handlelength=1.8, handletextpad=0.8,
    borderpad=0.8, labelspacing=1.2)
legend.get_frame().set_edgecolor("#dddddd")
legend.get_frame().set_facecolor("white")

plt.tight_layout(rect=[0, 0.02, 1, 0.94])
fig.subplots_adjust(left=0.34, right=0.82, bottom=1.85 / H, top=1 - 2.1 / H)
fig.text(0.5, 1 - 0.38 / H, f"Nash Bargaining Solution Deal: {nbs_package['Package']}",
         ha="center", va="center", fontsize=FS["badge"], fontweight="bold", color="white",
         bbox=dict(facecolor="#34495e", edgecolor="none", boxstyle="round,pad=0.5", alpha=0.9))
fig.savefig(FIG_OUT, dpi=110, bbox_inches="tight")
fig.savefig(FIG_OUT.replace(".png", ".pdf"), bbox_inches="tight")
print(f"\nFigure -> {FIG_OUT}")
print(df[["Package", "Count", "Loss_Label", "Hamming", "Euclidean", "Manhattan",
          "Is_Pareto", "Is_IR"]].to_string(index=False))
