# AirSim Smoke Eval Analysis

## Scope

This is a local OpenFly-Agent 4bit AirSim smoke-eval analysis. It is not a full OpenFly benchmark report.

## Run Metadata

- Run dir: `demo_results/airsim_smoke/20260629_004035`
- Env: `env_airsim_16`
- Unnorm key: `vlnv1`
- Vehicle: `drone_1`
- Max steps: `20`
- Contact sheet: `demo_results/airsim_smoke/20260629_004035/contact_sheet.png`

## Aggregate

- Samples: `3`
- Mean NE: `65.526`
- Mean SR: `0.3333`
- Mean OSR: `0.6667`
- Mean SPL: `0.3487`
- Action histogram: `{0: 1, 1: 4, 2: 2, 3: 5, 9: 40}`
- Sample classes: `{'max_steps_fail': 1, 'near_miss_osr': 1, 'success': 1}`
- Stop predicted count: `1`
- Max-step count: `2`
- Image error count: `0`

## Per Sample

| sample | class | steps | GT len | NE | SR | OSR | SPL | stop | predicted actions |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 0 | success | 12 | 16 | 7.436 | 1 | 1 | 1.0461 | True | `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0]` |
| 1 | near_miss_osr | 20 | 4 | 131.391 | 0 | 1 | 0.0000 | False | `[9, 9, 9, 9, 9, 9, 9, 9, 1, 1, 9, 9, 1, 9, 9, 9, 9, 2, 2, 9]` |
| 2 | max_steps_fail | 20 | 17 | 57.750 | 0 | 0 | 0.0000 | False | `[9, 3, 9, 9, 9, 9, 9, 9, 9, 9, 9, 3, 1, 9, 3, 3, 3, 9, 9, 9]` |

## Interpretation

- `success` samples end within the 20m success radius.
- `near_miss_osr` samples entered the radius at some point but did not finish there.
- `max_steps_fail` samples reached the step budget without success.
