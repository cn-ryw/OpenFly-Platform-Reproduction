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
- Dry run: `True`

## Settings

- Model: `IPEC-COMMUNITY/openfly-agent-7b`
- Unnorm key: `vlnv1`
- Max steps per sample: `5`
- Success radius: `20.0`
- Vehicle name: `drone_1`

## Dry-run result

No simulator or model execution was run. Resolve failed preflight items before actual smoke evaluation.

## Notes

- Official `train/eval.py` uses `unnorm_key="vlnv1"`; this script follows that by default.
- The action-id conversion intentionally mirrors official eval: unmatched rounded 8D actions default to action id 0.
