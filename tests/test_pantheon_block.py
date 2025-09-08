import pytest

import ilsne.hierarc_glue as glue


def test_pantheon_block_requires_hierarc(monkeypatch):
    # Simulate hierArc not installed by nulling the imported class
    monkeypatch.setattr(glue, "_PantheonPlusData", None, raising=True)
    with pytest.raises(ImportError):
        glue.PantheonPivotBlock(0.1)
