"""Flowchart schemas for the sensitivity (OPM) and adaptation (Campylobacter) analyses."""
from graphviz import Digraph

OUT_DIR = "/home/claude"

FILL = {
    "input": "#cfe0fb",      # source data
    "derive": "#d6ecfb",     # deterministic derivation
    "base": "#c9ecd4",       # baseline references
    "perturb": "#fde2b3",    # perturbation
    "exact": "#fff3c4",      # exact / deterministic search
    "metric": "#fbd0e4",     # recomputed metrics
    "result": "#e3d5f5",     # interpretation
}
EDGE = {"#7f8c9b"}


def new_graph(name):
    g = Digraph(name, format="png")
    g.attr(rankdir="TB", bgcolor="white", splines="spline", nodesep="0.30", ranksep="0.42")
    g.attr("node", shape="box", style="rounded,filled", fontname="DejaVu Sans",
           fontsize="15", penwidth="1.6", margin="0.20,0.12", height="0.1")
    g.attr("edge", color="#7f8c9b", penwidth="1.8", arrowsize="0.9",
       fontname="DejaVu Sans", fontsize="12", fontcolor="#5b6770")
    return g


def node(g, key, label, kind, color=None):
    edge_col = {"input": "#4a7fd4", "derive": "#5aa9dd", "base": "#3fae68",
                "perturb": "#e8a33d", "exact": "#d4b73a", "metric": "#dd5fa0",
                "result": "#8e6bc4"}[kind]
    g.node(key, label, fillcolor=FILL[kind], color=color or edge_col)


# ==================================================================== SENSITIVITY
s = new_graph("sensitivity_opm")
node(s, "csv", "Score matrix (OPM)\\n5 stakeholders × 6 issues", "input")
node(s, "logs", "Simulation logs\\npublic answers*.json", "input")
node(s, "util", "Extract utility functions\\nU\u1d62(d)=Σ\u2096u\u1d62(d\u2096)\\nissue weight = max option score", "derive")
node(s, "enum", "Enumerate all OPM deals\\n11,250 complete agreements", "derive")
node(s, "extract", "Extract proposed deals\\nevery distinct deal per final\\nproposal: 39 from 35 runs", "derive")
node(s, "base", "Baseline references\\nNBS (3,2,3,1,1,2) at r=65\\n+ 7 recurrent LLM deals", "base")
node(s, "pert", "Perturb the preference landscape\\n10,000 simultaneous draws", "perturb")
node(s, "w", "Issue weights\\nmax(1 pt, 25% of weight)\\n100-pt budget preserved\\nuniform hit-and-run", "perturb")
node(s, "t", "Acceptance threshold\\nr = 65 ± 10%\\none shared draw per sample\\nalso the disagreement point", "perturb")
node(s, "recomp", "Recompute per draw\\ncontemporaneous NBS, IR, Pareto,\\nHamming / L\u2081 / L\u2082, Nash loss", "metric")
node(s, "out", "Robustness of the achieved deal\\nNBS retained 25.0%, IR 92.2%,\\nPareto 100%  →  Table 2", "result")

s.edge("csv", "util")
s.edge("util", "enum")
s.edge("logs", "extract")
s.edge("enum", "base")
s.edge("extract", "base")
s.edge("base", "pert")
s.edge("pert", "w")
s.edge("pert", "t")
s.edge("w", "recomp")
s.edge("t", "recomp")
s.edge("recomp", "out")
with s.subgraph() as r:
    r.attr(rank="same")
    r.node("csv"); r.node("logs")
with s.subgraph() as r:
    r.attr(rank="same")
    r.node("w"); r.node("t")
s.render(f"{OUT_DIR}/Schema_sensitivity_OPM", cleanup=True)
s.format = "pdf"
s.render(f"{OUT_DIR}/Schema_sensitivity_OPM", cleanup=True)

# ==================================================================== ADAPTATION
a = new_graph("adaptation_campy")
node(a, "csv", "Score matrix (Campylobacter)\\n4 stakeholders × 5 issues", "input")
node(a, "util", "Extract utility functions\\nU\u1d62(d)=Σ\u2096u\u1d62(d\u2096)\\nissue weight = max option score", "derive")
node(a, "enum", "Enumerate all Campylobacter deals\\n1,200 complete agreements", "derive")
node(a, "check", "Baseline check at r = 65\\n0 feasible deals\\ninfeasibility gap 10.0 pts\\nFood Safety Authorities binding\\n→ no NBS exists", "base")
node(a, "adapt", "Adapt the preference landscape\\nmax(1 pt, 25% of weight)\\n100-pt budget preserved", "perturb")
node(a, "exact", "Exact frontier\\ndeterministic search\\nbest deal (4,5,4,1,1)\\nhighest threshold 66.25\\ncritical tolerance 22.222%", "exact")
node(a, "uncond", "Unconditional Monte Carlo\\n100,000 uniform draws per\\nthreshold 65 … 57.5\\n→ how prevalent", "perturb")
node(a, "cond", "Conditional Monte Carlo\\n2,000 draws restricted to the\\nfeasible slice\\n→ what it is", "perturb")
node(a, "recomp", "Recompute per draw\\nfeasibility, NBS above r\\n(disagreement point = r),\\npackage counts", "metric")
node(a, "out", "Infeasible but marginally adaptable\\n1 in 100,000 draws at r=65;\\nunique deal (4,5,4,1,1);\\n×3–10 region per 1.25 pts relaxed", "result")

a.edge("csv", "util")
a.edge("util", "enum")
a.edge("enum", "check")
a.edge("check", "adapt")
a.edge("adapt", "exact")
a.edge("adapt", "uncond")
# the exact search supplies the package the conditional sampler conditions on
a.edge("exact", "cond", label="  witness\n  deal  ")
a.edge("exact", "recomp")
a.edge("uncond", "recomp")
a.edge("cond", "recomp")
a.edge("recomp", "out")
with a.subgraph() as r:
    r.attr(rank="same")
    r.node("exact"); r.node("uncond")
a.render(f"{OUT_DIR}/Schema_adaptation_Campylobacter", cleanup=True)
a.format = "pdf"
a.render(f"{OUT_DIR}/Schema_adaptation_Campylobacter", cleanup=True)

print("rendered both schemas")
