import os
import tempfile
from csv_qwen_suite.src.nodes import _load_csv_prompts, _to_int32, _sanitize_filename_token

def test_to_int32_wrap():
    assert _to_int32(0) == 0
    assert _to_int32(2**31) == -2**31
    assert _to_int32(2**31 - 1) == 2**31 - 1

def test_sanitize():
    assert _sanitize_filename_token("a b") == "a_b"
    assert _sanitize_filename_token("x:y*?") == "x_y"

def test_load_csv_autodetect():
    csv_text = "positive,negative\nhello,world\n"
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "x.csv")
        with open(p, "w", encoding="utf-8") as f:
            f.write(csv_text)
        data = _load_csv_prompts(p, "", "")
        assert data.positive_col == "positive"
        assert data.negative_col == "negative"
        assert len(data.rows) == 1
