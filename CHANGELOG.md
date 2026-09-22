# Changelog

## Presentation update

- Lead the README with computed particle motion and a separate cavity-startup GIF, with documented timing and interpolation.
- Add a reproducible animation builder and record its source fields, output hashes and tracer integration checks.
- Rewrite the report introduction with historical sources, physical applications, and concrete explanations of the mathematical and computational difficulties.

## 1.3.0 — quantified cavity accuracy

- Regenerate all production states from rest with three grids per Reynolds number and a common steady criterion.
- Record numerical-source, runner and cache hashes, timestamps, software versions and fresh intermediate snapshots.
- Add actual-cavity transient time refinement, ten-time-unit continuations at two steps, and an unfiltered steady-equation defect.
- Add conditional Richardson/GCI fits for multiple observables, including unequal refinement ratios and explicit rejection of inadmissible fits.
- Independently check the Re = 400 three-grid prediction with a fourth, 513² integration and overlapping extrapolations.
- Extend MMS evidence with an asymmetric field and a finest-grid time-contamination check.
- Generate the accuracy tables and figures, synchronize the report and executed notebook, and test the error-estimation mathematics.

The numerical evolution scheme is unchanged.

## 1.2.0 — repository review

- Validate numerical inputs, initialise consistent field pairs, and account for the configured lid speed in automatic step selection.
- Version checkpoint files and reject incompatible restarts before replacing state.
- Exclude inert vorticity corners consistently from verification norms.
- Restore printed Ghia values, fix a transcription error, report anomaly sensitivity, and add the reflected Botella–Peyret reference.
- Regenerate numerical figures and evidence; replace the README and report with a methods-focused presentation.
- Add a portable executed walkthrough, demo, reproducible result/report builders, regression tests and GitHub CI.

Earlier local drafts and the original assignment remain in the ignored `report/archive/` directory.
