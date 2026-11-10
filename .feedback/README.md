# Feedback task starting environment

Upstream: https://github.com/Yubico/python-fido2
Upstream commit: `9d5991d6785e20bd487c8b6c996df6ce2226642a`
Prepared on macOS with CPython 3.12.11. No external database, queue, or cloud account is required.
The dependency file pins the installed runtime and test packages for the selected baseline. It is not a claim that all optional upstream test suites are installed.

From the repository root:

```sh
export NO_INTERNET=1 TZ=UTC
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r .feedback/requirements.txt -e .
.venv/bin/python -m pytest -q -o addopts= tests/test_server.py tests/test_attestation.py tests/test_webauthn.py tests/test_mds3.py tests/test_cose.py tests/test_cbor.py
```

Scope note: The scope is the command below, not all optional upstream integrations.

Observed baseline (2026-11-10): `112 passed in 1.86s`
Whole-command wall time: 2.189 seconds, exit code 0. This is one local measurement, not a runtime guarantee. The first install is excluded.

This seed adds preparation metadata only. Application sources and upstream tests are unchanged; there is no task solution in this commit.
