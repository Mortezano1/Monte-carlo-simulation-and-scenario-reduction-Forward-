# Monte Carlo Scenario Reduction

This repository contains a compact Python implementation of scenario reduction for three-dimensional scenarios. The workflow reads a CSV file of scenarios with probabilities, reduces the set to a smaller representative subset, and saves the results in an output folder.

## Files

- scenario_reduction.py: main script for loading input, reducing scenarios, and writing outputs.
- example/input/scenarios.csv: example input data.
- results/: folder for generated outputs such as CSV files, JSON summary, and a plot.

## How to run

1. Create a virtual environment and install dependencies:
   - python3 -m venv .venv
   - . .venv/bin/activate
   - pip install -r requirements.txt
2. Run the script:
   - python scenario_reduction.py --output-dir results --target-count 50 --count 1000 --seed 42

## Outputs

The script writes:

- reduced_scenarios.csv
- summary.json
- scenario_reduction_plot.png
