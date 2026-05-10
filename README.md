# SRFBench

This repository contains the code, data, ground-truth files, experiment outputs, and figures for the project report:

**SRFBench: A Reasoning-First Benchmark for Semantic Operators**

## Query Number
For Econ dataset, the Query 3 in the code base is refering to Query 2 in the paper, and Query 5 in the code base is refering to Query 3 in the paper.

## Important Reproduction Note

This repository is **not a standalone one-command package**.

To fully rerun the experiments, you must separately install and configure the three evaluated systems:

- LOTUS
- Palimpzest
- ThalamusDB

You must also set up the required Python environment, DuckDB databases, package dependencies, and OpenAI API keys. API keys and private credentials are **not included** in this repository.

Because the full experiments require external systems and paid LLM API calls, this repository includes saved result files, metrics, and figures so the reported results can be inspected without rerunning every experiment from scratch.

## Repository Structure

```text
SRFBench/
├── econ/
│   ├── Econ_data/              # Econ benchmark data and DuckDB files
│   ├── Econ_ground_truth/      # Econ ground-truth labels
│   ├── Econ_results/           # Econ metrics, and plots
│   └── Econ_runner/            # Scripts used to run Econ experiments
│
├── logic/
│   ├── Logic_data/             # Logic benchmark data and DuckDB files
│   ├── Logic_ground_truth/     # Logic ground-truth labels
│   ├── Logic_results/          # Logic  metrics, and plots
│   └── Logic_runner/           # Scripts used to run Logic experiments
│
├── paper/
│   ├── Paper_data/             # Scientific Paper benchmark data and DuckDB files
│   ├── Paper_ground_truth/     # Scientific Paper ground-truth labels
│   ├── Paper_results/          # Scientific Paper metrics, and plots
│   └── Paper_runner/           # Scripts used to run Scientific Paper experiments
│
├── figures/                    # Final figures used in the project report
└── README.md
```



To reproduce the full experiments, you need:

- Python environment with required packages installed
- LOTUS installed and configured
- Palimpzest installed and configured
- ThalamusDB installed and configured
- DuckDB support
- OpenAI API access
- Local API key configuration




## Running Experiments

The experiment scripts are organized by dataset:

```text
econ/Econ_runner/
logic/Logic_runner/
paper/Paper_runner/
```

These scripts assume that the required systems and API keys are already set up.

The saved outputs are stored in:

```text
econ/Econ_results/
logic/Logic_results/
paper/Paper_results/
```


## Code Availability

Repository:

https://github.com/billyypp/SRFBench
