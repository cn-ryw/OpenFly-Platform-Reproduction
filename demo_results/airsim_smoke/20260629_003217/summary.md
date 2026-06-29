# AirSim 4bit Smoke Eval

## Scope

This is a minimal AirSim-first smoke evaluation for OpenFly-Agent on local 4bit inference. It is not a full OpenFly benchmark run.

## Preflight

- Environment: `env_airsim_16`
- Scene start script exists: `True`
- Python airsim import: `True`
- CUDA available: `True`
- Samples selected: `1`
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

## Aggregate

- Mean NE: `7.436`
- Mean SR: `1.0000`
- Mean OSR: `1.0000`
- Mean SPL: `1.0461`

## Notes

- Official `train/eval.py` uses `unnorm_key="vlnv1"`; this script follows that by default.
- The action-id conversion intentionally mirrors official eval: unmatched rounded 8D actions default to action id 0.
