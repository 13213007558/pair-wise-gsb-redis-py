# Feedback task starting environment

Upstream: https://github.com/redis/redis-py
Upstream commit: `e80066215123c111ce08b38d7f52ee04d86ba2ed`
Prepared on macOS with CPython 3.12.11. No external database, queue, or cloud account is required.
The dependency file pins the installed runtime and test packages for the selected baseline. It is not a claim that all optional upstream test suites are installed.

From the repository root:

```sh
export NO_INTERNET=1 TZ=UTC
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r .feedback/requirements.txt -e .
.venv/bin/python -m pytest -q -o addopts= --noconftest tests/test_crc.py tests/test_backoff.py tests/test_retry.py -k 'not TestRedisClientRetry'
```

Scope note: The scope is the command below, not all optional upstream integrations.

Observed baseline (2026-11-10): `  warnings.warn(`
Whole-command wall time: 0.243 seconds, exit code 0. This is one local measurement, not a runtime guarantee. The first install is excluded.

This seed adds preparation metadata only. Application sources and upstream tests are unchanged; there is no task solution in this commit.
