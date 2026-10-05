from src.olist_inference.artifacts import load_artifacts


def test_artifacts_load():
    artifacts = load_artifacts()

    assert artifacts["preprocessor"] is not None
    assert artifacts["model"] is not None

    assert len(
        artifacts["input_features"]
    ) == 26

    assert len(
        artifacts["feature_list"]
    ) == 52


def test_threshold():
    artifacts = load_artifacts()

    assert artifacts["threshold"] == 0.4