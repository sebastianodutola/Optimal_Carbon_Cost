# Carbon-aware EV charging optimisation

Eco-warriors rejoice!
MPC scheduling reduces EV charging carbon cost by **43–47%** using the NESO 
48-hour carbon intensity forecast. Analysis across four load profiles shows 
forecast accuracy is not the binding constraint (>92% forecast efficiency); 
forecast horizon is.

![Comparison graphs: optimal vs naive vs clairvoyant carbon-cost]<img width="1189" height="812" alt="image" src="https://github.com/user-attachments/assets/4d4b21d7-5afe-430b-ade2-fb857ead0e70" />

Full Analysis: [Carbon efficient charging of battery powered loads](https://sebastianodutola.github.io/Optimal_Carbon_Cost/carbon_saving_analysis.html)

## Tech Stack 
- **Optimisation**: SciPy.optimise.linprog — Linear Programming HiGHS C++ Wrapper (Dual-Simplex / Interior-point)
- **Data** NESO API, Pandas, Parquet / PyArrow, Httpx (asynchronous client)
- **Validation** Arch, SciPy.stats

## Further Research Directions
- Stochastic Control for non-deterministic loads subject to chance constraints.
- Dual carbon-cost optimisation.
- Forecast horizon extension.

## Repo Contains
- `battery/` - python package with optimisers, ingest code, and models.
- `src/` — simulation loop (can only be run after data ingested with ingest.py and ingest_actual.py)
- `tests/` — a smokescreen test for the lp_optimiser.

src and tests for completeness purposes only.

