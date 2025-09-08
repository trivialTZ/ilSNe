from __future__ import annotations


def test_slsim_importable():
    try:
        import slsim  # type: ignore  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise AssertionError(
            "slsim should be importable in this environment for SLSIM-related tests"
        ) from e
