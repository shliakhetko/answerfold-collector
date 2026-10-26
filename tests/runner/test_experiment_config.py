from pathlib import Path

from collector.runner.experiment_config import load_experiment_config

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "experiment.yaml"


def test_load_experiment_config():
    config = load_experiment_config(FIXTURE)

    assert config.reps_per_day == 1
    assert config.days == 1
    assert len(config.engines) == 1
    assert config.engines[0].engine == "openai_chatgpt"
    assert config.engines[0].model_requested == "gpt-4o-search-preview"
