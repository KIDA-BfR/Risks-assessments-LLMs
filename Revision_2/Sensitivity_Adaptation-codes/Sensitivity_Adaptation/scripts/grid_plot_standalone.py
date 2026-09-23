import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GRID_DIR = "/content/campy_nbs_grid"
FIG_OUT = f"{GRID_DIR}/Fig_adaptation_grid.png"

plt.rcParams["font.family"] = "sans-serif"
FS = dict(title=19, sub=16, lab=15, tick=13, ann=12, leg=12)

res = json.load(open(f"{GRID_DIR}/campy_nbs_strategy_results.json"))
mc = pd.read_csv(f"{GRID_DIR}/unconditional_mc_summary.csv")
front = pd.read_csv(f"{GRID_DIR}/exact_feasibility_frontier.csv")
base = res["baseline"]
crit = res["critical_issue_cap_at_central_threshold"]

fixed = (mc[mc["threshold_relative_variation"] == 0]
         .sort_values("threshold_center").reset_index(drop=True))
varying = mc[mc["threshold_relative_variation"] > 0]

cap = float(fixed["relative_cap"].iloc[0])
attainable = float(front["Highest_attainable_common_threshold"].iloc[0])
gap = float(base["infeasibility_gap"])
n = int(fixed["samples"].iloc[0])

fig, (axA, axB) = plt.subplots(1, 2, figsize=(16.5, 6.6),
                               gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.28})

# -------------------------------------------------- A: feasibility vs threshold
x = fixed["threshold_center"].to_numpy(float)
hits = fixed["iterations_with_any_feasible_package"].to_numpy(float)
rate = 100.0 * hits / n
floor = 100.0 / n / 3.0                      # plotting floor for zero-hit rows
yplot = np.where(hits > 0, rate, floor)

axA.plot(x, yplot, "-o", color="steelblue", linewidth=2.4, markersize=11,
         markeredgecolor="black", zorder=3, label="Weights only, threshold fixed")
for xi, yi, h in zip(x, yplot, hits):
    axA.annotate(f"{int(h)}/{n:,}", (xi, yi), textcoords="offset points",
                 xytext=(0, 14), ha="center", fontsize=FS["ann"], fontweight="bold")

if len(varying):
    vr = 100.0 * float(varying["probability_any_feasible_package"].iloc[0])
    lo, hi = [float(v) for v in str(varying["threshold_interval"].iloc[0]).strip("[]()").split(",")]
    axA.hlines(vr, lo, hi, color="coral", linewidth=3.5, zorder=2,
               label=f"Threshold also drawn on [{lo:g}, {hi:g}]: {vr:.2f}%")

axA.axvline(attainable, color="darkgreen", linestyle="--", linewidth=2, zorder=1)
axA.annotate(f"exact limit {attainable:g}\n(no solution above)", (attainable, floor * 2.2),
             xytext=(-10, 0), textcoords="offset points", ha="right",
             fontsize=FS["ann"], color="darkgreen", fontweight="bold")

axA.set_yscale("log")
axA.set_xlabel("Common acceptance threshold (utility points)", fontsize=FS["lab"])
axA.set_ylabel(f"Draws with a feasible deal (%, log scale)", fontsize=FS["lab"])
axA.set_title(f"A  Feasibility vs acceptance level\n"
              f"issue-weight tolerance max(1 pt, {cap*100:g}% of weight)",
              fontsize=FS["sub"], fontweight="bold", loc="left")
axA.set_ylim(floor / 2.5, 100)
axA.invert_xaxis()
axA.grid(alpha=0.35, linestyle="--", which="both")
axA.tick_params(labelsize=FS["tick"])
axA.legend(fontsize=FS["leg"], loc="upper left", framealpha=0.95)

# -------------------------------------------------- B: which deal is the NBS
comp = {}
for _, r in fixed.iterrows():
    comp[float(r["threshold_center"])] = json.loads(r["nbs_counts_json"] or "{}")
totals = {t: sum(c.values()) for t, c in comp.items()}
overall = pd.Series({d: sum(c.get(d, 0) for c in comp.values())
                     for d in {k for c in comp.values() for k in c}}).sort_values(ascending=False)
top = list(overall.index[:3])
short = lambda d: "(" + ",".join(p[1:] for p in d.split(",")) + ")"
colors = ["steelblue", "coral", "grey", "#c9c9c9"]

ys = sorted(comp, reverse=True)
ypos = np.arange(len(ys))
left = np.zeros(len(ys))
for k, deal in enumerate(top + ["other"]):
    vals = []
    for t in ys:
        tot = totals[t]
        if tot == 0:
            vals.append(0.0)
        elif deal == "other":
            vals.append(100.0 * (tot - sum(comp[t].get(d, 0) for d in top)) / tot)
        else:
            vals.append(100.0 * comp[t].get(deal, 0) / tot)
    vals = np.array(vals)
    axB.barh(ypos, vals, left=left, height=0.62, color=colors[k],
             edgecolor="black", linewidth=1.0,
             label=short(deal) if deal != "other" else "other deals")
    for i, v in enumerate(vals):
        if v >= 9:
            axB.text(left[i] + v / 2, ypos[i], f"{v:.0f}%", ha="center", va="center",
                     fontsize=FS["ann"], fontweight="bold",
                     color="white" if k < 2 else "black")
    left += vals

axB.set_yticks(ypos)
axB.set_yticklabels([f"{t:g}\n({totals[t]:,} feasible)" for t in ys], fontsize=FS["tick"])
axB.set_xlabel("Share of feasible draws in which the deal is the NBS (%)", fontsize=FS["lab"])
axB.set_title("B  Which deal is the NBS, given feasibility\nrow label: acceptance threshold",
              fontsize=FS["sub"], fontweight="bold", loc="left")
axB.set_xlim(0, 100)
axB.grid(axis="x", alpha=0.35, linestyle="--")
axB.tick_params(axis="x", labelsize=FS["tick"])
axB.legend(fontsize=FS["leg"], loc="upper center", bbox_to_anchor=(0.5, -0.13),
           ncol=4, framealpha=0.95, columnspacing=1.4, handlelength=1.4)
for s in ("top", "right"):
    axB.spines[s].set_visible(False)
    axA.spines[s].set_visible(False)

fig.suptitle(
    f"Campylobacter adaptation search: no deal is feasible at the elicited scores "
    f"(gap {gap:g} points, {', '.join(base['binding_stakeholders'])} binding);\n"
    f"minimum issue-weight tolerance admitting a threshold-65 solution: "
    f"{crit['critical_cap']*100:.3f}%",
    fontsize=FS["title"], fontweight="bold", y=1.045)

fig.savefig(FIG_OUT, dpi=200, bbox_inches="tight")
fig.savefig(FIG_OUT.replace(".png", ".pdf"), bbox_inches="tight")
plt.show()
print(f"Saved {FIG_OUT} and .pdf")
