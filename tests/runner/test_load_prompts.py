from pathlib import Path

from collector.runner.load_prompts import load_prompts_csv

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "prompts.csv"


def test_load_prompts_csv():
    prompts = load_prompts_csv(FIXTURE)

    assert len(prompts) == 1
    prompt = prompts[0]
    assert prompt.intent_id == 1
    assert prompt.paraphrase_id == 1
    assert prompt.prompt_text == "what is the capital of France?"
    assert prompt.branded is False
