# COVID-19 Global Impact Analysis

An exploratory analysis of historical COVID-19 cases, deaths, vaccination and policy indicators. Correlations in this project do not establish causal effects of vaccination or policy. The outputs are not current public-health guidance.

## Inputs and provenance

`covid_analysis.py` retrieves Johns Hopkins CSSE confirmed/deaths/recovered series and Our World in Data COVID-19 data, including population, vaccination and policy fields, when files are absent. It does not ingest Oxford's separate policy CSV. These URLs refer to upstream files that can change or disappear. Each run writes `data_manifest.json` with the exact URL, SHA-256 and retrieval or verification time for every input. Archive that manifest and the corresponding inputs to reproduce a run. Review each publisher's attribution and reuse terms before redistribution.

The tracked `archive (2).zip` is a legacy dataset archive; the script currently uses the named downloaded CSVs, not that archive. The charts were regenerated on 10 November 2026 with the input hashes in `data_manifest.json`. `SUMMARY.md` describes interpretation limits for that historical snapshot, not current public-health conditions.

## Run

```bash
pip install -r requirements.txt
python covid_analysis.py
python -m unittest discover -s tests
```

Run from the repository root. The script fails if a required download fails instead of continuing with an incomplete set. It writes `merged_covid_dataset.csv`, `data_manifest.json`, and plots under `visualization/`. Check the manifest, merge coverage, missingness, and country-name alignment before interpreting plots. The `data_manifest.json` records a snapshot; the code does not pin upstream server revisions automatically.

## Limitations

Johns Hopkins and OWID country names and coverage may differ. Data can be revised retrospectively; recovered cases have uneven availability. Missing daily case, death, vaccination, and policy observations remain missing rather than being copied from earlier dates. The scripts are descriptive and do not adjust for testing, reporting, population composition or policy confounders. The repository has no software license file; data rights belong to their respective publishers.

**Repository description recommendation:** “Historical COVID-19 data exploration with source hashes and explicit merge limitations.”
