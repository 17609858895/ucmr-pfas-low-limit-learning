# PFAS reporting limits and censored mixture learning

Important scientific code accompanying **Low-limit observations inform multi-threshold PFAS assessment with censored mixture learning**.

This release contains the data-processing, concentration-model, validation and observational-summary code used for the manuscript, together with small aggregate results used to check its reported numbers. Large source files, row-level water-system records, fitted models, manuscript files and unrelated exploratory analyses are excluded.

## Quick verification

The aggregate checks use only the Python standard library:

```bash
python tools/verify_retained_results.py
```

They check the six recalibration ratios; the reported 4 ng/L ROC area, average precision and Brier skill; the 20 ng/L anchor comparison; probability consistency; the dioxane rate ratio; normalized attribution shares; screening proportions; and the two historical cohort sizes. These checks inspect retained aggregate summaries. They do not refit the models or constitute independent validation of the underlying observations.

For numerical checks of the concentration-distribution and pooling code:

```bash
python -m pip install -r requirements.txt
python tools/smoke_core.py
```

The recorded analysis environment is Python 3.12. The requirements record the installed scientific-library versions at release preparation. Neither verification command trains models. Training the full benchmark and repeated stress experiments is computationally expensive.

## Inputs and source versions

Public data should be obtained from their owning agencies:

| Source | Official access | Analysis input |
| --- | --- | --- |
| UCMR1–5 analytical and additional-data files | [EPA UCMR occurrence data](https://www.epa.gov/dwucmr/occurrence-data-unregulated-contaminant-monitoring-rule) | UCMR5 final release with July 2026 summary; historical chemistry through 2021 |
| UCMR5 archive used locally | [EPA occurrence-data ZIP](https://www.epa.gov/system/files/other-files/2023-08/ucmr5-occurrence-data.zip) | Downloaded 15 October 2026; expected archive SHA256 in `data_manifest.json` |
| SDWIS public-water-system table | [EPA ECHO SDWA downloads and dictionary](https://echo.epa.gov/tools/data-downloads/sdwa-download-summary) | 2026Q3 registry snapshot; obtain `SDWA_PUB_WATER_SYSTEMS.csv` |
| Served-area geographical proxies | [Census 2023 Gazetteer files](https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.2023.html), [ZCTA ZIP](https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2023_Gazetteer/2023_Gaz_zcta_national.zip) | 2023 ZCTA representative points |

Agency download links can serve a later revision; compare file hashes and release dates before interpreting a changed result. The historical input used in the analysis was an existing merged UCMR1–5 CSV mirror. Its exact bytes are identified in `data_manifest.json`; the merged file is not redistributed here. The scripts expect its recorded schema. Downloading and concatenating the current agency files without a documented schema and version reconciliation is not an exact substitute for this input. Consequently this code-only release does not provide an independently complete, one-command reproduction from newly downloaded raw data.

Create this input layout:

```text
inputs/
  combined_ucmr_data.csv
  ucmr5/
    UCMR5_All.txt
    UCMR5_AddtlDataElem.txt
    UCMR5_ZIPCodes.txt
  sdwa/
    SDWA_PUB_WATER_SYSTEMS.csv
  gazetteer/
    2023_Gaz_zcta_national.txt
```

`PFAS_RAW_DIR` may point to a different directory with that layout. `PFAS_WORK_DIR` selects the cache/output directory; the default is `work/` inside this repository. All generated results go to that work directory, leaving `retained_results/` unchanged. Neither input files nor generated row-level outputs are included in Git tracking.

## Analysis order and manuscript mapping

Run scripts from the repository root with `python analysis/<filename>`. The scientific fitting, thresholds, seeds and formulas are preserved from the source scripts; input and output paths have been made portable. Selected historical comments now use the manuscript's observational interpretation.

| Stage | Scripts, in order | Current manuscript / SI |
| --- | --- | --- |
| Data and response preparation | `s00_profile.py` → `s01_support.py` → `s02_build_features.py`; loader `common_data.py` | Methods; Supplementary Methods S1–S2; Tables 1 and S1 |
| Base concentration models | `censored_models.py` | S3–S4; Table S2 |
| Common-MRL benchmark | `s10_common_benchmark.py` → `s10b_monotonicity.py` | Fig. 3; Tables S3 and S11 |
| Distributional recalibration | `s17_recalibration.py` | Fig. 4; Tables S4 and S10 |
| Reporting-limit and low-limit anchor experiment | `s11_anchor_stress.py` | Fig. 5; Table S5 |
| Geographical evaluation | `s12_spatial_cv.py`; repeat with argument `nostate` | Fig. 4; Table S6 |
| Two-learner attribution | `s15_shap_heads.py` | Figs. 6–7; Table S7 |
| Empirical prediction sets | `s13_conformal.py` after recalibration | Fig. 8; Table S8 |
| Treatment-report comparisons | `s05_causal_dml.py` → `s07_causal_robust.py` → `s08_within_pws.py` → `s14_causal_process.py` | Figs. 9–10; Table S9 |
| Historical marker associations | `historical_markers.py` after data preparation | Fig. 7; Table S12 |

`s12_spatial_cv.py nostate` omits the State and Region categorical predictors. The internal `causal_*`, `ATT` and `ATE` filenames/columns are retained for traceability; in the manuscript these are adjusted observational contrasts between report groups. They do not identify treatment effects. Likewise mixture membership does not establish true PFAS absence, historical chemical associations do not establish shared sources, and the retrospective assessments do not demonstrate prospective deployment performance.

## Which version is represented

The current manuscript and its ten main figures use the original analysis outputs (`e1`–`e6`) retained by the compact figure package, plus the eligible-cohort historical marker associations adopted from the restructuring stage. `source_manifest.json` records the local source-relative provenance and hashes for copied files. Its `release_sha256` values refer to tracked repository bytes after Git text normalization; working-copy line endings may differ.

The restructuring stage also contains separate nested train/calibration/test experiments and same-family anchor sensitivities. Those results were not substituted for the current manuscript's reported results and are not included here. In particular, `s17_recalibration.py` estimates each target fold's calibration transform from base-model out-of-fold predictions in the other four folds; this is the procedure described in Supplementary Methods S5, not the separate nested experiment. The empirical prediction sets inherit that retained calibration structure and should be read as retrospective descriptive summaries.

The plotting stages rearranged the original analysis into ten figures without retraining. Their large source and panel exports are outside this focused code release. `retained_results/` provides aggregate data underlying the important reported numbers; full layout reproduction and all manuscript/SI exports are not claimed.

Fig. 2's analyte-specific observed comparison uses 4,435 systems, whereas its joint common-threshold comparison uses 4,439. Both original denominators are preserved in the aggregate checks. The released aggregate files alone cannot identify the four-system membership difference. The historical model-based 4 ng/L extrapolation explored earlier is excluded because the current figure reports observable high-threshold comparisons.

## Contents and preservation

- `analysis/`: 16 adapted original analysis modules and two release helpers.
- `retained_results/`: small CSV/JSON summaries without PWS identifiers or row-level data.
- `tools/`: aggregate-number verification and a deterministic numerical smoke check.
- `data_manifest.json`: source-version records and input hashes; raw inputs are excluded.
- `source_manifest.json`: correspondence to the retained project versions.

Original project files were preserved. No full model training was repeated when preparing this release.
