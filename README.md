# Carbon-aware EV charging optimisation

Eco-warriors rejoice!
MPC scheduling reduces EV charging carbon cost by **43–47%** using the NESO 
48-hour carbon intensity forecast. Analysis across four load profiles shows 
forecast accuracy is not the binding constraint (>92% forecast efficiency); 
forecast horizon is.

<img width="1189" height="812" alt="image" src="https://github.com/user-attachments/assets/bff9936c-313d-43f2-a92f-6c1099c5e6c3"/>

Full Analysis: [Carbon efficient charging of battery powered loads](https://htmlpreview.github.io/?https://github.com/sebastianodutola/Optimal_Carbon_Cost/blob/main/carbon_saving_analysis.html)

## Repo Contains
- `battery/` - python package with optimisers, ingest code, and models.
- `src/` — simulation loop (can only be run after data ingested with ingest.py and ingest_actual.py)
- `tests/` — a smokescreen test for the lp_optimiser.

src and tests for completeness purposes only.

