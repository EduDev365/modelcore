# Release process

ModelCore releases should use GitHub environments and PyPI Trusted Publishing (OIDC), without long-lived API tokens.
Configure separate `testpypi` and `pypi` environments in GitHub, require approval for production, and register this
repository and [publish workflow](.github/workflows/publish.yml) as a trusted publisher in each package index. The
workflow is manual and grants `id-token: write` only to the publication job.

The current release candidate remains version `1.3.0`. After approving the final diff, update the version to `1.4.0`
and repeat every validation before creating a release.

## Validation

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
python -m mypy src/modelcore
python -m compileall -q src/modelcore
python -m pip check
python -m build
python -m twine check dist/*
```

Inspect both artifacts before upload. Test installation from the wheel and sdist in clean environments, including
`py.typed`, public imports, and an offline provider composition.

## Publication sequence

1. Approve the final diff.
2. Bump the version to `1.4.0`.
3. Run validation again.
4. Commit.
5. Push.
6. Tag `v1.4.0`.
7. Publish to TestPyPI through a protected GitHub environment using Trusted Publishing.
8. Install from TestPyPI and run the offline smoke test.
9. Create the GitHub Release with the `1.4.0` changelog notes.
10. Publish to production PyPI through a separately protected environment using Trusted Publishing.
11. Install from PyPI in a clean environment.
12. Run the final offline smoke test.

Do not upload artifacts built before the version bump or reuse artifacts between TestPyPI and production unless their
hashes were reviewed and the release procedure deliberately promotes the exact same immutable files.
