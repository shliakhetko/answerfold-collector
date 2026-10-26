import json

from analysis.phase_b.figure_shape import FigureResult
from analysis.phase_b.serialize import write_seam_file


def test_write_seam_file_shape(tmp_path):
    result = FigureResult(
        figure_id="S3",
        status="ok",
        value={"openai_chatgpt": 0.1},
        denominator_before_exclusion={"openai_chatgpt": 105},
        denominator_after_exclusion={"openai_chatgpt": 100},
        n_included={"openai_chatgpt": 100},
        n_excluded={"openai_chatgpt": {}},
        threshold=None,
        reason_untestable=None,
        supporting_table=[{"run_id": "r1"}],
    )
    path = tmp_path / "figures.json"
    write_seam_file(path, inputs_manifest={"instrument_commit": "x"}, figure_results={"S3": result})

    written = json.loads(path.read_text())
    assert set(written) == {"generator", "inputs", "figures", "intermediates"}
    assert written["inputs"] == {"instrument_commit": "x"}
    assert written["figures"]["S3"]["value"] == {"openai_chatgpt": 0.1}
    assert written["figures"]["S3"]["threshold"] is None
    assert written["intermediates"] == {}


def test_write_seam_file_embeds_intermediates_when_given(tmp_path):
    result = FigureResult("S3", "ok", 1, 2, 2, 2, {}, None, None, [])
    path = tmp_path / "figures.json"
    write_seam_file(
        path,
        inputs_manifest={},
        figure_results={"S3": result},
        intermediates={"source_ownership": {"row_count": 5}},
    )
    written = json.loads(path.read_text())
    assert written["intermediates"] == {"source_ownership": {"row_count": 5}}


def test_write_seam_file_is_byte_identical_and_generated_at_is_in_sidecar(tmp_path):
    result = FigureResult("S3", "ok", 1, 2, 2, 2, {}, None, None, [])
    path1, path2 = tmp_path / "a.json", tmp_path / "b.json"
    write_seam_file(path1, inputs_manifest={}, figure_results={"S3": result})
    write_seam_file(path2, inputs_manifest={}, figure_results={"S3": result})

    assert path1.read_bytes() == path2.read_bytes()
    assert "generated_at" not in path1.read_text()
    meta = json.loads(path1.with_suffix(".meta.json").read_text())
    assert "generated_at" in meta
