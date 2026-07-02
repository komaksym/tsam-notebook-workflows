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
  --countries ALL \
  --working-clusters 5 \
  --non-working-clusters 2 \
  --cluster-method hierarchical \
  --overwrite
```

Open `charts/index.html`, whose direct local URL is printed when the command
finishes. It provides a responsive sidebar for every chart in one workspace.

### Custom YAML Configuration

Use the optional YAML configuration for custom input files, CSV separators,
timestamp parsing, feature labels, chart feature groups, calendar rules, and
mean-preserved representatives:

```bash
uv run tsam-workflows grouped \
  --config examples/grouped-workflow.yaml \
  --output-dir outputs/custom \
  --overwrite
```

Relative dataset paths resolve from the YAML file's directory. Omitting
`countries` or setting `countries: ALL` selects every available country; use a
list such as `countries: [DE, FR]` for a subset. Explicit CLI options such as
`--year`, `--countries`, and cluster counts override YAML values.

Each dataset's `feature` becomes the canonical output token, for example raw
column `DE` with `feature: solar` becomes `DE_solar_2025`. `feature_group`
organizes related features in chart selectors and does not change clustering.
`unit_interval: true` validates that every value is between zero and one; it
does not normalize the data.

The grouped workflow always creates representative 24-hour days. Sampling
frequency is inferred from timestamps, must be regular and identical across
datasets, and is recorded in `manifest.json`. Enabling
`preserve_column_means` exports TSAM's rescaled synthetic representatives while
retaining each selected medoid date as provenance.

## Data

Input CSV files live in `data/`. The notebooks use the checked-in sample
datasets through relative paths, so they can be run from the repository root.

## Acknowledgement

This work was prepared for Helmholtz-Zentrum Berlin für Materialien und Energie (**HZB**), a member of the Helmholtz Association, in the context of the Green Deal Ukraïna project.

<img src="assets/HZB_logo.png" alt="Sponsor logo" width=200>
