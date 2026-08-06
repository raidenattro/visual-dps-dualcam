import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experts.linear_expert import LinearPickExpert

MODEL = {
    "feature_keys": ["a", "b"],
    "scaler_mean": [0.0, 0.0],
    "scaler_scale": [1.0, 1.0],
    "coef": [1.0, 1.0],
    "intercept": 0.0,
}


def _expert(**extra):
    return LinearPickExpert({"model": {**MODEL, **extra}})


def test_missing_feature_uses_impute_not_fail_open():
    exp = _expert(impute=[0.5, -0.5])
    with_value, _ = exp.score({"a": 0.5, "b": -0.5})
    imputed, detail = exp.score({"a": None, "b": None})
    assert imputed == with_value
    assert detail["n_imputed"] == 2
    assert not detail.get("missing")


def test_old_model_without_impute_keeps_fail_open():
    exp = _expert()
    prob, detail = exp.score({"a": 1.0, "b": None})
    assert prob == 0.55
    assert detail["missing"] is True


def test_unparseable_value_treated_as_missing():
    exp = _expert(impute=[0.0, 0.0])
    _, detail = exp.score({"a": "abc", "b": 0.0})
    assert detail["n_imputed"] == 1


def test_impute_dimension_mismatch_rejected():
    try:
        _expert(impute=[0.0])
    except ValueError:
        return
    raise AssertionError("impute 维度不一致应当报错")
