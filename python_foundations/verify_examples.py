"""复跑教材示例并记录输出；练习模板只检查语法，不当作已完成的程序。"""

import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent


def main():
    parsed = []
    for path in sorted(ROOT.rglob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parsed.append(str(path.relative_to(ROOT)))

    records = {}
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    with TemporaryDirectory() as temp_dir:
        programs = sorted((ROOT / "examples").glob("[0-9][0-9]_*.py"))
        programs.append(ROOT / "answers" / "config_report.py")
        for path in programs:
            result = subprocess.run(
                [sys.executable, str(path)], cwd=temp_dir, env=environment,
                text=True, capture_output=True, timeout=60, check=True,
            )
            records[str(path.relative_to(ROOT))] = {
                "stdout": result.stdout, "stderr": result.stderr, "exit_code": result.returncode,
            }

        package = subprocess.run(
            [sys.executable, "-m", "mini_package"], cwd=ROOT / "examples", env=environment,
            text=True, capture_output=True, timeout=15, check=True,
        )
        records["python -m mini_package"] = {"stdout": package.stdout, "exit_code": package.returncode}

        cli_path = ROOT / "examples" / "12_errors_cli.py"
        cli = subprocess.run(
            [sys.executable, str(cli_path), "--token-id", "50256", "--all"],
            cwd=temp_dir, env=environment, text=True, capture_output=True, timeout=15, check=True,
        )
        assert "50256" in cli.stdout and "True" in cli.stdout
        invalid_cli = subprocess.run(
            [sys.executable, str(cli_path), "--token-id", "-1"],
            cwd=temp_dir, env=environment, text=True, capture_output=True, timeout=15,
        )
        assert invalid_cli.returncode == 2 and "token_id 必须" in invalid_cli.stderr

        output_path = Path(temp_dir) / "nested" / "report.json"
        subprocess.run(
            [sys.executable, str(ROOT / "answers" / "config_report.py"), "--output", str(output_path)],
            cwd=temp_dir, env=environment, text=True, capture_output=True, timeout=15, check=True,
        )
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        assert saved["report"]["token_embedding_parameters"] == 38597376

    inspect_config = runpy.run_path(str(ROOT / "answers" / "config_report.py"))["inspect_config"]
    config = json.loads((ROOT / "fixtures" / "demo_gpt2_config.json").read_text(encoding="utf-8"))
    report, info = inspect_config(config, output_info=True)
    assert report["head_dim"] == 64 and report["position_embedding_parameters"] == 786432
    assert info["missing_keys"] == [] and inspect_config(config) == report
    invalid_configs = [{}, dict(config, n_head=13), dict(config, n_embd=True)]
    for invalid in invalid_configs:
        try:
            inspect_config(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("错误配置未被拒绝")

    source = ROOT.parent / "inspect_checkpoint.py"
    verification = {
        "audience": "熟悉 Java、C++、Dart 的资深程序员；Python 迁移版",
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "safetensors")},
        "syntax_checked": parsed,
        "runs": records,
        "extra_checks": ["包入口", "命令行参数覆盖", "非法 ID", "JSON 保存后读回", "配置缺字段/不能整除/bool 冒充整数", "各示例内的绑定/复制/闭包/视图/注册语义断言"],
        "source_reading_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "status": "passed",
    }
    (ROOT / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("通过：{} 个 Python 文件的语法；{} 个独立脚本；包入口和边界检查。".format(len(parsed), len(programs)))
    print("未运行待完成的练习模板，未下载或加载真实 GPT-2 权重。")


if __name__ == "__main__":
    main()
