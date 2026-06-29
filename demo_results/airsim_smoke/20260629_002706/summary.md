# AirSim 4bit Smoke Eval

## Scope

This is a minimal AirSim-first smoke evaluation for OpenFly-Agent on local 4bit inference. It is not a full OpenFly benchmark run.

## Preflight

- Environment: `env_airsim_16`
- Scene start script exists: `True`
- Python airsim import: `True`
- CUDA available: `True`
- Samples selected: `1`
- Dry run: `False`

## Settings

- Model: `IPEC-COMMUNITY/openfly-agent-7b`
- Unnorm key: `vlnv1`
- Max steps per sample: `20`
- Success radius: `20.0`

## Results

| sample | steps | actions | NE | SR | OSR | SPL | image_error |
|---:|---:|---|---:|---:|---:|---:|---|
| 0 | 0 | `[]` | 103.566 | 0 | 0 | 0.0000 | `RPCError: rpclib: function 'simGetImages' (called with 3 arg(s)) threw an exception. The exception is not derived from std::exception. No further information available.` |

## Aggregate

- Mean NE: `103.566`
- Mean SR: `0.0000`
- Mean OSR: `0.0000`
- Mean SPL: `0.0000`

## Notes

- Official `train/eval.py` uses `unnorm_key="vlnv1"`; this script follows that by default.
- The action-id conversion intentionally mirrors official eval: unmatched rounded 8D actions default to action id 0.
