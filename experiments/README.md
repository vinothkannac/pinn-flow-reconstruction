# experiments

The scripts that train the models and collect the numbers.

* `run_experiments.py` the full grid (seeds, noise levels, sensor counts, both models and the inverse problem)
* `run_tuning.py` the loss-weight sweep, chosen on held-out sensors
* `summarize_results.py` mean and std tables and the error-versus-noise plots

Run them from the main folder, for example `python experiments/run_experiments.py --case venturi`.
