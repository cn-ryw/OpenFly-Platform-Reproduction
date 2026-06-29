# AirSim Smoke Eval Analysis

## Scope

This is a local OpenFly-Agent 4bit AirSim smoke-eval analysis. It is not a full OpenFly benchmark report.

## Run Metadata

- Run dir: `demo_results/airsim_smoke/20260629_005458`
- Env: `env_airsim_16`
- Unnorm key: `vlnv1`
- Vehicle: `drone_1`
- Max steps: `20`
- Contact sheet: `demo_results/airsim_smoke/20260629_005458/contact_sheet.png`

## Aggregate

- Samples: `30`
- Mean NE: `76.457`
- Mean SR: `0.1667`
- Mean OSR: `0.5667`
- Mean SPL: `0.1346`
- Action histogram: `{0: 18, 1: 15, 2: 1, 3: 13, 9: 309}`
- Sample classes: `{'fail': 10, 'max_steps_fail': 3, 'near_miss_osr': 12, 'success': 5}`
- Stop predicted count: `18`
- Max-step count: `13`
- Image error count: `0`

## Per Sample

| sample | class | steps | GT len | NE | SR | OSR | SPL | stop | predicted actions |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | success | 12 | 16 | 7.436 | 1 | 1 | 1.0461 | True | `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0]` |
| 1 | near_miss_osr | 14 | 4 | 90.853 | 0 | 1 | 0.0000 | True | `[9, 9, 9, 9, 9, 9, 9, 9, 1, 1, 9, 9, 1, 0]` |
| 2 | near_miss_osr | 20 | 17 | 37.882 | 0 | 1 | 0.0000 | False | `[9, 9, 9, 9, 9, 9, 9, 3, 9, 9, 3, 9, 9, 9, 3, 9, 3, 3, 9, 9]` |
| 3 | fail | 1 | 55 | 136.392 | 0 | 0 | 0.0000 | True | `[0]` |
| 4 | max_steps_fail | 20 | 17 | 83.588 | 0 | 0 | 0.0000 | False | `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]` |
| 5 | fail | 1 | 28 | 104.498 | 0 | 0 | 0.0000 | True | `[0]` |
| 6 | success | 11 | 9 | 19.726 | 1 | 1 | 0.7246 | True | `[9, 9, 9, 9, 9, 9, 9, 1, 1, 2, 0]` |
| 7 | success | 9 | 9 | 19.042 | 1 | 1 | 0.7400 | True | `[9, 9, 9, 9, 9, 9, 9, 9, 0]` |
| 8 | fail | 8 | 21 | 101.425 | 0 | 0 | 0.0000 | True | `[9, 9, 9, 9, 1, 1, 1, 0]` |
| 9 | max_steps_fail | 20 | 15 | 90.254 | 0 | 0 | 0.0000 | False | `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]` |
| 10 | near_miss_osr | 20 | 17 | 64.856 | 0 | 1 | 0.0000 | False | `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 3, 1, 9]` |
| 11 | near_miss_osr | 12 | 7 | 57.000 | 0 | 1 | 0.0000 | True | `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0]` |
| 12 | near_miss_osr | 20 | 15 | 71.608 | 0 | 1 | 0.0000 | False | `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 3]` |
| 13 | max_steps_fail | 20 | 18 | 60.472 | 0 | 0 | 0.0000 | False | `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]` |
| 14 | fail | 1 | 37 | 105.256 | 0 | 0 | 0.0000 | True | `[0]` |
| 15 | near_miss_osr | 20 | 22 | 26.776 | 0 | 1 | 0.0000 | True | `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0]` |
| 16 | fail | 1 | 57 | 104.505 | 0 | 0 | 0.0000 | True | `[0]` |
| 17 | fail | 1 | 50 | 164.224 | 0 | 0 | 0.0000 | True | `[0]` |
| 18 | near_miss_osr | 20 | 5 | 166.716 | 0 | 1 | 0.0000 | False | `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]` |
| 19 | success | 7 | 11 | 7.565 | 1 | 1 | 1.0277 | True | `[9, 9, 9, 9, 9, 9, 0]` |
| 20 | near_miss_osr | 20 | 22 | 43.823 | 0 | 1 | 0.0000 | False | `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 9, 9, 9, 9, 9, 9]` |
| 21 | fail | 1 | 62 | 80.007 | 0 | 0 | 0.0000 | True | `[0]` |
| 22 | fail | 10 | 18 | 39.079 | 0 | 0 | 0.0000 | True | `[9, 9, 9, 9, 3, 9, 9, 9, 9, 0]` |
| 23 | near_miss_osr | 20 | 13 | 41.173 | 0 | 1 | 0.0000 | False | `[9, 9, 9, 9, 9, 9, 9, 9, 3, 9, 9, 3, 1, 3, 9, 9, 9, 1, 3, 3]` |
| 24 | near_miss_osr | 20 | 11 | 115.410 | 0 | 1 | 0.0000 | False | `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]` |
| 25 | fail | 1 | 55 | 139.102 | 0 | 0 | 0.0000 | True | `[0]` |
| 26 | fail | 1 | 41 | 89.080 | 0 | 0 | 0.0000 | True | `[0]` |
| 27 | success | 5 | 4 | 18.000 | 1 | 1 | 0.5000 | True | `[9, 9, 9, 9, 0]` |
| 28 | near_miss_osr | 20 | 9 | 131.224 | 0 | 1 | 0.0000 | False | `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]` |
| 29 | near_miss_osr | 20 | 16 | 76.726 | 0 | 1 | 0.0000 | False | `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 1, 1, 9, 9, 9, 1]` |

## Interpretation

- `success` samples end within the 20m success radius.
- `near_miss_osr` samples entered the radius at some point but did not finish there.
- `max_steps_fail` samples reached the step budget without success.
