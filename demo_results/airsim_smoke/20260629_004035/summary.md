# AirSim 4bit Smoke Eval

## Scope

This is a minimal AirSim-first smoke evaluation for OpenFly-Agent on local 4bit inference. It is not a full OpenFly benchmark run.

## Preflight

- Environment: `env_airsim_16`
- Scene start script exists: `True`
- Python airsim import: `True`
- CUDA available: `True`
- Samples selected: `3`
- AirSim settings matches project: `True`
- Dry run: `False`

## Settings

- Model: `IPEC-COMMUNITY/openfly-agent-7b`
- Unnorm key: `vlnv1`
- Max steps per sample: `20`
- Success radius: `20.0`
- Vehicle name: `drone_1`

## Results

| sample | steps | actions | NE | SR | OSR | SPL | image_error |
|---:|---:|---|---:|---:|---:|---:|---|
| 0 | 12 | `[9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 0]` | 7.436 | 1 | 1 | 1.0461 | `None` |
| 1 | 20 | `[9, 9, 9, 9, 9, 9, 9, 9, 1, 1, 9, 9, 1, 9, 9, 9, 9, 2, 2, 9]` | 131.391 | 0 | 1 | 0.0000 | `None` |
| 2 | 20 | `[9, 3, 9, 9, 9, 9, 9, 9, 9, 9, 9, 3, 1, 9, 3, 3, 3, 9, 9, 9]` | 57.750 | 0 | 0 | 0.0000 | `None` |

## Aggregate

- Mean NE: `65.526`
- Mean SR: `0.3333`
- Mean OSR: `0.6667`
- Mean SPL: `0.3487`

## Notes

- Official `train/eval.py` uses `unnorm_key="vlnv1"`; this script follows that by default.
- The action-id conversion intentionally mirrors official eval: unmatched rounded 8D actions default to action id 0.
