# Reproducing the study

Run commands from the project root. Python 3.10+ is supported. Install with `python -m pip install -e '.[dev,notebook]'` in a virtual environment. NumPy, SciPy, Matplotlib, SymPy and Pillow are the numerical dependencies; Jupyter dependencies are optional outside the notebook workflow. The reviewed run used Python 3.14.3, NumPy 2.5.2 and SciPy 1.18.0. Exact measured versions are recorded in `data/verification.json`.

## Small demonstration

```bash
python scripts/run_demo.py
python scripts/run_demo.py --re 1000 --n 65 --time 120
```

Outputs go to ignored `output/demo/` by default. The script prints the residual and whether the steady criterion was met. A high-Re coarse-grid demo is not a replacement for the production study.

## Production fields

```bash
python scripts/run_production.py --force
python scripts/run_production.py --case Re400_N129 --force
```

| Re | Grids | Time cap | dt | Stop threshold |
|---:|---|---:|---|---:|
| 400 | 65², 129², 257² | 150, 100, 100 | 0.0025, 0.0025, 0.001953125 | 1e-6 |
| 1000 | 65², 129², 257² | 400 | 0.001 | 1e-6 |
| 3200 | 129², 193², 257² | 1000 | 0.0003125 | 1e-6 |

All cases start from rest and stop when the common update-rate threshold is met. The runner exits unsuccessfully if a time cap is reached first. The caps are not the actual final times: those are recorded in each JSON and in the generated results table.

The full sequential suite can take an hour or more. `--fft-workers 4` accelerates the larger transforms on suitable hardware; it changes execution parallelism, not the discretisation. The recorded study used four workers for Re3200_N257, two for Re3200_N193 and one for the other cases. Independent cases can run in separate processes, with separate logs. Avoid oversubscribing the machine.

Caches are reused only when case parameters, tolerance, numerical-source and runner hashes, initial-condition label and FFT worker count match. Use `--force` to regenerate a mismatched case. Each record includes UTC start/finish times, software versions and field/cache SHA-256 hashes. The final array and diagnostic histories come from the same integration. Intermediate states at t = 100, 150 and 300 are retained whenever reached before convergence. Production NPZ files are study caches; use `Solver.save` for strict restart checkpoints.

An independent Re = 400, N = 513 run tests the three-grid model. Run `python scripts/check_fourth_grid.py` after the main production suite; it records `data/fourth_grid.json`, including the prediction error and the change between overlapping three-grid fits. It uses four FFT workers by default.

## Accuracy measurements

```bash
python scripts/run_accuracy.py
# Independent of production fields:
python scripts/run_accuracy.py --temporal-only
# Reuse completed per-case studies and recombine:
python scripts/run_accuracy.py --assemble-only
```

The temporal study uses N = 129, T = 5 and three halved time steps at each Re. The steady study continues all nine states for ten time units at dt and dt/2. It records drift in the primary vortex and centerlines, plus the steady PDE defect. Three-grid fits use actual refinement ratios and retain non-monotone sequences without an invented uncertainty estimate. See [accuracy.md](accuracy.md) for assumptions and formulas.

Per-case accuracy records live in `data/accuracy/`; `data/accuracy.json` assembles them with the grid study. Cached studies are accepted only when analysis/numerical-source fingerprints and input-cache hashes agree. `--refresh` forces new measurements. `--steady-only --case Re3200_N257 --fft-workers 4` runs one continuation study.

## Evidence and figures

```bash
python scripts/build_results.py
python scripts/build_results.py --skip-verification
```

The first command verifies all nine production caches and their provenance, loads the accuracy study, reruns the MMS and small-grid stability studies, and builds the PNG/PDF figures and table inputs. The MMS rerun is the slow part: about an hour on one core, because the spatial studies integrate to T = 0.5 with dt = 5e-6. The second command reuses the existing MMS JSON when only rebuilding visualisations. Neither starts a missing production run silently.

Outputs include:

- `data/validation.json`: recomputed numerical comparisons, equation checks, next-step residual, run provenance and SHA-256 of each final field pair.
- `data/validation.csv`: compact results table.
- `data/centerline_comparisons.csv`: every reference station, numerical value and anomaly flag.
- `data/verification.json`: all refinement errors and orders, wall truncation errors and Stokes thresholds.
- `data/accuracy.json` and `accuracy_summary.md`: grid fits, transient refinement and settling measurements.
- `data/fourth_grid.json`: independent Re = 400, 513² test of the three-grid extrapolation.
- `report/*_rows.tex`: measured tables consumed by the report, including grid, temporal, settling, eddy and spectral-reference comparisons.

All production states were regenerated from rest for this study. Builders verify the numerical-source and file hashes, finiteness, shape and Poisson residual, and recompute a continuation step. They reject stale or unconverged fields. The previous inherited states are preserved only in the ignored local archive; they are not used for current figures or tables.

When changing the numerical method, rerun the production study and update the narrative conclusions as well as the generated tables. Small residuals alone do not establish grid independence. Raw Ghia data are preserved; full and secondary RMS metrics must be kept distinct.

## README animations

Run `python scripts/build_animations.py` after the production caches are available; it needs Manim (`pip install -e ".[animation]"`, or the conda-forge environment described in [animations.md](animations.md)). The README uses an animation of passive tracers in the converged 257² fields and a separate animation of the recorded Re = 400 startup, both drawn by the Manim scenes in `scripts/manim_scenes.py`. Input hashes, video and poster hashes, rendering settings, decoded-frame PSNR and tracer-step checks are recorded in `data/animations.json`. See [animations.md](animations.md) for the interpolation and timing conventions. The MP4s and posters are included in the repository and play without the field caches.

## Notebook

```bash
python scripts/build_notebook.py --execute
jupyter lab notebooks/lid_driven_cavity.ipynb
```

The generator executes with the active Python interpreter. It runs a 65² demo and a short MMS study, and reads the committed results and figures for the larger cases. It does not require production `.npz` files. Running without `--execute` rewrites a clean notebook with no outputs; use the executed version when preparing it for GitHub.

## PDF report

```bash
python scripts/build_report.py
# Or select a compiler explicitly:
python scripts/build_report.py --engine /path/to/tectonic
```

Install Tectonic or TeX Live with XeLaTeX/LuaLaTeX separately, and install Times New Roman for both the report and figure builds. The report uses the system font through `fontspec`, with Times-style mathematical symbols from `newtxmath`; pdfLaTeX is not supported. Tectonic may download standard packages on its first run. The source must be compiled from `report/`; the helper does this automatically. It outputs `report/report.pdf`. The report and its figures use LaTeX equations and vector plots. Render the PDF and inspect page breaks, equations, figures and tables after changing the source.

## Restart semantics

```python
from ns2d import Grid, Cavity, Solver
solver = Solver(Grid.square(65), Cavity(400), dt=0.0025)
solver.run(n_steps=100, progress=False)
solver.save('checkpoint.npz')
continued = Solver(Grid.square(65), Cavity(400), dt=0.0025)
continued.load('checkpoint.npz')
continued.run(n_steps=100, progress=False)
```

`T` is an absolute target time; `n_steps` is an additional step count. Supply exactly one. The fixed step is not shortened to land on `T`. Strict loading checks time step, domain, Reynolds number, built-in problem signature and wall conventions before changing state. `strict=False` imports a legacy state or deliberately changes configuration; shape and finite-state checks still apply. With a custom `Problem`, override `checkpoint_signature()` to include every physical parameter. Restart equivalence is bitwise for the tested DST path, and within solver tolerance when changing linear backends.
