# Model card — 30-day readmission risk for diabetic inpatients

## Model details
- **Type**: binary classifier, `HistGradientBoostingClassifier` (scikit-learn) with balanced class weights, tuned by randomized search on cross-validated PR-AUC.
- **Pipeline**: median imputation and standardisation (numeric), constant imputation and one-hot encoding with rare-level grouping (categorical), all fitted on training data only.
- **Probability outputs**: isotonic calibration (`CalibratedClassifierCV`, 5-fold). Raw scores are for ranking only.
- **Operating points**: F2-optimal threshold 0.43 (recall-oriented); capacity threshold 0.559 (top 20 % of patients).
- **Authors**: Ranim Rtimi, Nour Elhouda Louhichi.

## Intended use
- **Primary use**: research and education; ranking diabetic inpatients at discharge by 30-day readmission risk to prioritise **low-cost** follow-up (phone calls, early outpatient visits).
- **Out of scope**: denying care or resources; clinical decisions without clinician review; use on recent data, ICD-10 coded data or non-US settings without re-validation and recalibration.

## Data
- **Source**: UCI *Diabetes 130-US Hospitals for Years 1999–2008* (101,766 encounters, 130 hospitals).
- **Cohort**: first encounter of each patient; deaths and hospice discharges excluded → 69,987 patients, 9.0 % readmitted within 30 days.
- **Split**: stratified 80/20 (55,989 train / 13,998 test); no patient in both sets.
- **Features (31)**: demographics, admission type/source, discharge disposition, specialty, ICD-9 diagnosis groups, stay counts, prior-year utilisation, HbA1c/glucose test results, diabetes medications.

## Performance (held-out test set)

| Metric | Value (95 % bootstrap CI) |
|--------|---------------------------|
| ROC-AUC | 0.653 (0.638–0.670) |
| PR-AUC | 0.179 (0.163–0.199) — chance 0.090 |
| Recall @ 0.43 | 0.742 (0.718–0.767) |
| Precision @ 0.43 | 0.116 (0.110–0.124) |
| Patients flagged @ 0.43 | 57.3 % |
| Recall / precision @ 0.559 (20 % flagged) | 0.375 / 0.173 |
| Top-decile readmission rate | 21.2 % (2.4× average) |
| Brier score (calibrated) | 0.079 (0.075–0.082) — reference 0.082 |

Decision curve analysis: positive net benefit over "follow up everyone" and "follow up no one" for threshold probabilities from 3 % to 40 %.

## Fairness (threshold 0.43)

| Group | Recall | Precision | ROC-AUC |
|-------|--------|-----------|---------|
| Female / Male | 0.758 / 0.725 | 0.114 / 0.120 | 0.657 / 0.650 |
| African-American / Caucasian / Other-Unknown | 0.727 / 0.742 / 0.792 | 0.128 / 0.113 / 0.123 | 0.673 / 0.643 / 0.725 |
| Age < 60 / 60–79 / 80+ | 0.573 / 0.771 / 0.893 | 0.119 / 0.116 / 0.114 | 0.670 / 0.638 / 0.636 |

A single threshold gives markedly lower recall to younger patients (lower prevalence and scores). Age-specific thresholds should be considered and subgroup metrics monitored in any deployment.

## Explainability
Discharge disposition and prior-year inpatient visits dominate permutation importance, SHAP values and logistic-regression odds ratios. HbA1c, glucose and individual medications contribute little. These are associations, not causal effects.

## Limitations and risks
- Old (1999–2008) US administrative data; treatments, coding (ICD-9 → ICD-10) and policies have changed.
- Readmissions to other hospitals are not observed (possible false negatives in the label).
- No vital signs, detailed laboratory values or social determinants of health.
- Internal validation only; temporal and external validation required before any real-world use.
