# TSAM Notebook Workflows

This repository contains two Jupyter notebooks that demonstrate practical
[ETHOS.TSAM](https://github.com/FZJ-IEK3-VSA/tsam) workflows for energy
time-series aggregation.

The notebooks are not copies of the upstream TSAM quickstart examples. They
extend the basic usage pattern with input validation, diagnostic plots, output
inspection, and one custom grouped aggregation approach.

## Upstream TSAM

- Original repository: https://github.com/FZJ-IEK3-VSA/tsam
- Documentation: https://tsam.readthedocs.io/
- Upstream example notebooks:
  [basic example](https://tsam.readthedocs.io/en/latest/notebooks/quickstart/)
  and
  [visualization example](https://tsam.readthedocs.io/en/latest/notebooks/visualization/)

## Notebooks

- `src/quickstart.ipynb` - annual baseline aggregation with validation,
  diagnostics, plotting, output inspection, and saved results.
- `src/approach_1_ALL.ipynb` - monthly working/non-working grouped aggregation with
  representative-day outputs and group-level diagnostics.

## Installation

Install the project dependencies with `uv`:

```bash
uv sync
```

Start JupyterLab:

```bash
uv run jupyter lab
```

Or start Jupyter Notebook:

```bash
uv run jupyter notebook
```

Then open the notebooks from `src/`.

## Grouped CLI

Run the original 5-working/2-non-working representative configuration with:

```bash
uv run tsam-workflows grouped \
  --data-dir data \
  --output-dir outputs/approach_1 \
  --year 2025 \
  --countries DE \
  --working-clusters 5 \
  --non-working-clusters 2 \
  --cluster-method hierarchical \
  --overwrite
```

Open `charts/index.html`, whose direct local URL is printed when the command
finishes. It provides a responsive sidebar for every chart in one workspace.
The only other item in `charts/` is the internal `assets/` folder, which holds
the individual chart pages and shared `plotly.min.js` bundle. The report works
offline and requires no additional dependency or server.

The feature drilldown embeds each group's reusable TSAM result arrays once and
renders the selected group/country/feature view in the browser. Selecting more
countries therefore increases one dashboard's data instead of creating
thousands of HTML files.
Runs are generated in a staging directory. With `--overwrite`, artifacts are
then replaced inside the existing output folders so open Finder and IDE windows
remain attached to the populated directories.

## Data

Input CSV files live in `data/`. The notebooks use the checked-in sample
datasets through relative paths, so they can be run from the repository root.

## Acknowledgement

This work was prepared for Helmholtz-Zentrum Berlin für Materialien und Energie (**HZB**), a member of the Helmholtz Association, in the context of the Green Deal Ukraïna project.

<img src="assets/HZB_logo.png" alt="Sponsor logo" width=200>
