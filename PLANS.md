# Grouped Workflow CLI Plan

## Summary

Extract `src/approach_1_ALL.ipynb` into a reusable Python package and expose it
through `uv run tsam-workflows grouped`. Keep the notebook as an interactive
thin client, remove `src/approach_1_DE.ipynb`, and leave the annual quickstart
and advanced TSAM preservation options for later milestones.

## Milestones

1. Package the grouped workflow under `src/tsam_workflows/` with config, data,
   grouped aggregation, chart export, CLI, and module entrypoint modules.
2. Add a Hatchling build backend and `tsam-workflows` console script while using
   `argparse` for CLI parsing.
3. Preserve the existing grouped output CSV schemas:
   `reduced_hourly_df.csv`, `representative_days.csv`, and
   `day_assignments_df.csv`.
4. Write artifacts through a staging directory before publishing to the final
   output directory.
5. Export offline Plotly HTML charts, always including summary charts and adding
   scoped group/country/feature drilldowns only for valid selected
   combinations.
6. Refactor `src/approach_1_ALL.ipynb` into a thin client that imports the
   shared workflow and chart functions, then clear stale outputs.
7. Remove `src/approach_1_DE.ipynb`.
8. Update README and CI for the new CLI/package workflow.

## Validation

- `uv run ruff check src/tsam_workflows tests src/approach_1_ALL.ipynb`
- `uv run mypy src/tsam_workflows`
- `uv run pytest tests`
- `uv run pytest --nbmake src/approach_1_ALL.ipynb`
- `uv run tsam-workflows grouped --help`
- `uv build`
- `git diff --check`

## Chart Fidelity And Default Export Follow-up

### Summary

Restore the chart appearance and 5/2 clustering behavior from the original
notebook-only workflow, and include group-level TSAM diagnostics in ordinary
CLI output without enabling the much larger country/feature export matrix.

### Milestones

1. Lock the original calendar and heatmap palette behavior with regression
   tests, then restore deterministic Plotly defaults where extraction changed
   them.
2. Export cluster-weight and cluster-accuracy TSAM charts for every workflow
   group by default; retain explicit selectors for feature-level drilldowns.
3. Restore the notebook's 5 working-day and 2 non-working-day clusters so the
   representative-share chart shows meaningful distributions instead of
   mathematically trivial 100% rows.
4. Execute focused tests after each milestone, then run lint, typecheck, the
   full test suite, notebook execution, CLI smoke validation, and package build.

## Render All Charts By Default

### Summary

Remove plot-selection controls from the CLI and notebook. Every valid summary,
group-level, and group/country/feature TSAM chart should render without users
editing flags or supplying chart-selector arguments.

### Milestones

1. Replace selector-driven chart planning with deterministic enumeration of all
   valid groups, selected countries, and available feature groups.
2. Remove `--chart-*` CLI arguments and chart-selection data from the manifest.
3. Remove notebook `RUN_CHART_OUTPUTS` and `RUN_WIDGET_OUTPUTS` switches and
   render every chart/widget cell unconditionally.
4. Update CLI documentation and regression tests, then run the full validation
   gate and inspect an end-to-end export count.

## Consolidated Group Diagnostics

### Summary

Replace per-group cluster-weight and cluster-accuracy files with one offline
`group_diagnostics.html` dashboard. A group dropdown should update both Plotly
panels from figure JSON embedded during export.

### Milestones

1. Add a failing export regression test for the dashboard file, group selector,
   embedded figure registry, and absence of per-group diagnostic files.
2. Add the minimal HTML/JavaScript writer using `Plotly.react()` and wire it into
   chart export.
3. Update documentation, run the complete validation gate, and verify dropdown
   behavior in the rendered local dashboard.

## On-demand Feature Drilldown Dashboard

### Summary

Replace thousands of pre-rendered group/country/feature chart files with one
offline `drilldown_dashboard.html`. Python should serialize each TSAM group's
reusable result arrays once; plain JavaScript should build only the chart chosen
through group, country, feature-group, and chart-type controls.

### Milestones

1. Add failing export tests proving that one drilldown dashboard replaces all
   individual feature chart files and contains every selector and reusable data
   payload needed for client-side rendering.
2. Add a focused dashboard module that serializes original, reconstructed,
   representative, assignment, and weight data once per group and renders the
   four existing drilldown views with Plotly in the browser.
3. Wire the dashboard into chart export, update the README output contract, and
   verify focused tests before running lint, typecheck, all tests, notebook
   execution, CLI help, package build, and an end-to-end ALL-country smoke run.

## Finder-safe Output Publication

### Summary

Preserve the existing output and `charts` directory identities during
`--overwrite` so open Finder and IDE windows continue tracking the populated
folders. Keep staging-generation safety and atomically replace individual
artifacts, publishing `manifest.json` last.

### Milestones

1. Add a failing regression test proving overwrite preserves both directory
   inodes while replacing new artifacts and removing obsolete artifacts.
2. Replace whole-tree deletion with a recursive staged-file merge that keeps
   existing directories and Finder metadata intact.
3. Run focused and full validation, then perform two real CLI runs against the
   same output path and verify stable directory inodes and complete artifacts.

## Restore Full Country Selectors

### Summary

Restore the original notebook-only country-name mapping and reuse its
`Full name (CODE)` labels in both Option 1 notebook widgets and the offline
drilldown dashboard. Run the Option 1 workflow with all available countries so
its selectors are not limited to Germany.

### Milestones

1. Add failing regression tests for the original country labels, missing-name
   validation, dashboard payload labels, and all-country notebook default.
2. Move the exact original mapping into shared configuration, wire both selector
   surfaces to it, and retain Germany as the initial selected value.
3. Run focused checks, execute the notebook, and browser-test the exported
   selector labels before the full validation gate.

## CLI Phase Progress

### Summary

Print four immediate, dependency-free progress phases to stderr during grouped
CLI execution while retaining the final success line on stdout.

### Milestones

1. Add a failing CLI regression test for exact progress ordering and output
   streams.
2. Emit flushed messages before workflow execution, table writing, chart
   generation, and artifact publication.
3. Run focused and full validation plus one real CLI smoke run to confirm users
   see progress before completion.

## Modern Local Chart Navigation

### Summary

Replace the plain chart-link list with a responsive offline dashboard shell.
A persistent sidebar should group overview and exploration charts, load the
selected existing HTML file into an embedded frame, and retain direct fallback
links plus an open-separately action.

### Milestones

1. Add a failing export regression test for grouped navigation, the embedded
   chart frame, active selection, direct links, and absence of external assets.
2. Replace only the index writer with semantic HTML, embedded CSS, and minimal
   JavaScript; preserve every existing chart file and output contract.
3. Browser-test selection, responsive layout, direct opening, local-file/server
   loading, keyboard navigation, and console errors before the full validation
   gate.

## Chart Output Packaging And Handoff

### Summary

Keep `charts/` focused on the user entrypoint by moving standalone chart pages
and `plotly.min.js` into `charts/assets/`. Update the CLI completion message to
print a direct local URL to `charts/index.html` before the artifact directory.

### Milestones

1. Add failing regressions for the two-item chart root, nested asset links, and
   exact success-message dashboard guidance.
2. Export chart pages and Plotly into `assets/`, keep the modern index at the
   root, and preserve relative offline loading.
3. Run an overwrite smoke test to verify obsolete root files are removed, then
   browser-test the dashboard and run the full validation gate.

## Optional YAML Workflow Configuration

### Summary

Add an optional `--config` YAML file for custom dataset paths, separators,
canonical feature names, chart feature groups, unit-interval validation,
timestamp parsing, workflow selection, calendar overrides, and TSAM mean
preservation. Preserve the current no-config CLI behavior and let explicitly
provided CLI flags override YAML values.

### Milestones

1. Add failing configuration tests for strict YAML schema validation, paths
   relative to the YAML file, `countries: ALL`, subset countries, and defaults;
   then add PyYAML as a direct dependency and implement the loader.
2. Add failing CLI tests for optional `--config` loading and precedence
   `built-in defaults < YAML < explicit CLI flags`; keep output publication
   settings CLI-only.
3. Add failing data tests for configurable timestamp columns/formats, inferred
   common sampling frequency, complete-year validation, and a fixed 24-hour
   TSAM period whose timestep count follows the inferred resolution.
4. Add failing aggregation tests proving `preserve_column_means: true` exports
   TSAM's rescaled representatives while retaining medoid-date provenance and
   recording the effective setting and inferred frequency in the manifest.
5. Add an example YAML file and README usage documentation, then run lint,
   typecheck, all tests, notebook execution, CLI help, package build, diff
   checks, and real default/configured CLI smoke runs.
