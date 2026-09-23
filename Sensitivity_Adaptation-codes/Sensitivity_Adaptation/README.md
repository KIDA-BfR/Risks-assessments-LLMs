# What to upload to /content

All six files the notebook needs, from `scripts/` and `inputs/`:

| File | Folder | Used by |
|---|---|---|
| `table2_proportional_floor.py` | scripts | section 1 (Table 2) |
| `negotiation_sensitivity_core_promptzip.py` | scripts | imported by the above, must sit beside it |
| `campy_nbs_strategy_search.py` | scripts | section 2 (Campylobacter grid) |
| `Case2_OPM_Workshop_Alex_final_scores.csv` | inputs | sections 1 and 3 |
| `Case1_Campy_Workshop_Alex_final_scores.csv` | inputs | section 2 |
| `OPM_thoughts.zip` | inputs | section 1, deal extraction (needs the `OPM/Answers/` members) |

Flat upload, no subfolders: the notebook expects `/content/<filename>`.
Cell 2 of the notebook checks all six are present before anything runs.

`scripts/fig4_standalone.py` is optional. It is the Fig. 4 pipeline as a single
script, for running outside the notebook; the notebook has the same code inline
in its last two code cells.

## Expected outputs

`expected_outputs/` holds the results of the two runs, for comparison:

- `Table2_sensitivity.pdf`, `Table2_values.csv` - Table 2 at 10,000 draws
- `Results_extracted_all_deals.csv` (39 rows), `Results_extracted_primary_only.csv` (35 rows)
- `observed_final_deal_counts.csv`, `observed_final_deals_long.csv` - extraction detail
- `Fig_4.png`, `Fig_4.pdf`, `Fig4_deal_metrics.csv`
- `campy_*.csv` - the Campylobacter frontier and Monte Carlo summaries
- `EXTRACTION.md`, `campy_RUN.md` - what each analysis does and how to read it

Key numbers to check a reproduction against:

| Quantity | Value |
|---|---|
| Deals extracted | 39 from 35 runs, 4 with a fallback |
| Baseline NBS | `A3,B2,C3,D1,E1,F2`, utilities 75/85/83/71/83, Nash product 388,800 |
| Baseline NBS retained (Table 2) | 25.0% of 10,000 draws |
| Campylobacter infeasibility gap | 10.0 utility points, Food Safety Authorities binding |
| Campylobacter critical cap | 22.222% |
| Campylobacter feasible draws per 100,000 | 1 / 30 / 347 / 1,554 / 5,176 / 25,705 at thresholds 65 to 57.5 |

Section 2 is the slow cell, around seven tests of 100,000 hit-and-run draws in
pure Python. Drop `--samples` to 10000 for a quick structural check; the grid
shape still shows, with the threshold-65 row empty.
