from __future__ import annotations

from ilsne.roman_interface import load_hltds_stub, to_survey_config


def test_roman_to_survey_config_mapping():
    cad = load_hltds_stub()
    cfg = to_survey_config(cad)
    assert cfg.name == "roman-hltds"
    assert cfg.epochs == cad.epochs
    assert cfg.days_between == cad.days_between
    # depths keys should match
    assert set(cfg.depths.keys()) == set(cad.per_epoch_depth_5sig.keys())

