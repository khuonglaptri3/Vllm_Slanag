import ast
import json
from pathlib import Path


def test_kaggle_notebooks_are_valid_and_contain_t4_flags():
    root = Path(__file__).resolve().parent.parent
    vllm_nb_path = root / "notebooks" / "01_vllm_kaggle.ipynb"
    sglang_nb_path = root / "notebooks" / "02_sglang_kaggle.ipynb"

    assert vllm_nb_path.exists()
    assert sglang_nb_path.exists()
    assert (root / "01_vllm_kaggle.ipynb").exists()
    assert (root / "02_sglang_kaggle.ipynb").exists()

    vllm_nb = json.loads(vllm_nb_path.read_text(encoding="utf-8"))
    sglang_nb = json.loads(sglang_nb_path.read_text(encoding="utf-8"))

    assert vllm_nb["nbformat"] == 4
    assert sglang_nb["nbformat"] == 4

    for nb in (vllm_nb, sglang_nb):
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                src = "".join(cell["source"]).strip()
                if not src.startswith("!"):
                    compile(
                        src,
                        "<notebook_cell>",
                        "exec",
                        flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
                    )

    vllm_src = "\n".join("".join(c["source"]) for c in vllm_nb["cells"])
    sglang_src = "\n".join("".join(c["source"]) for c in sglang_nb["cells"])

    # Verify required Kaggle T4 flags
    assert "--dtype float16" in vllm_src
    assert "--enable-prefix-caching" in vllm_src
    assert "Qwen/Qwen2.5-1.5B-Instruct" in vllm_src
    assert "vllm_results.json" in vllm_src
    assert "format_single_engine_markdown" in vllm_src
    assert "vllm_summary.md" in vllm_src
    assert "pip uninstall -y -q torchaudio" in vllm_src

    assert "--dtype half" in sglang_src
    assert "--attention-backend triton" in sglang_src
    assert "--sampling-backend pytorch" in sglang_src
    assert "LD_LIBRARY_PATH" in sglang_src
    assert "sys.executable" in sglang_src
    assert "/tmp/cuda_link" in sglang_src
    assert 'uv pip install --system --prerelease=allow "sglang[all]"' in sglang_src
    assert "Qwen/Qwen2.5-1.5B-Instruct" in sglang_src
    assert "sglang_results.json" in sglang_src
    assert "format_single_engine_markdown" in sglang_src
    assert "sglang_summary.md" in sglang_src
    assert "pip uninstall -y -q torchaudio" in sglang_src


def test_report_and_readme_exist_and_cover_mentor_questions():
    root = Path(__file__).resolve().parent.parent
    report_path = root / "REPORT.md"
    readme_path = root / "README.md"

    assert report_path.exists()
    assert readme_path.exists()

    report_text = report_path.read_text(encoding="utf-8")
    assert "PagedAttention" in report_text
    assert "RadixAttention" in report_text
    assert "Concurrent Users" in report_text
    assert "Multi-Turn Agent" in report_text
