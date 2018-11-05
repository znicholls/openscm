import pytest
from pathlib import Path
from openscm.cli import load_scenario, load_parameters

test_root = Path(__file__).parents[1]


def test_load_scenario():
    scenario = load_scenario(test_root / "data/rcp26.csv")
    assert scenario.index.name == "Year"
    assert "CO2ffi" in scenario.columns
    assert "CH4" in scenario.columns


def test_load_parameters():
    params = load_parameters(test_root / "data/params.yaml")
    assert params["ecs"] == 3


def test_default_parameters():
    params = load_parameters()
    assert len(params) == 0
