#!/usr/bin/env python3
"""Checks static supply-chain invariants without importing ComfyUI or PyTorch."""

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fail(message):
    raise SystemExit(message)


requirements = sorted(
    line.strip()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
)
project_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
dependencies_block = re.search(r"(?ms)^dependencies\s*=\s*\[(.*?)^\]", project_text)
if not dependencies_block:
    fail("pyproject.toml não declara a lista project.dependencies")
project_dependencies = sorted(re.findall(r'^\s*"([^\"]+)"\s*,?\s*$', dependencies_block.group(1), re.M))
if requirements != project_dependencies:
    fail("requirements.txt e project.dependencies do pyproject.toml divergem")

manifest = json.loads((ROOT / "model_checksums.json").read_text(encoding="utf-8"))
models = manifest.get("models", {})
nodes_tree = ast.parse((ROOT / "nodes.py").read_text(encoding="utf-8"))
model_map = next(
    (node.value for node in nodes_tree.body if isinstance(node, ast.Assign)
     and any(isinstance(target, ast.Name) and target.id == "MODEL_ID_MAP" for target in node.targets)),
    None,
)
if model_map is None:
    fail("MODEL_ID_MAP ausente em nodes.py")
model_ids = set(ast.literal_eval(model_map).values())
if model_ids != set(models):
    fail("MODEL_ID_MAP e model_checksums.json precisam conter os mesmos modelos")

for model_id, info in models.items():
    if not model_id.startswith("Qwen/"):
        fail(f"Modelo fora da allowlist Qwen: {model_id}")
    if not re.fullmatch(r"[0-9a-f]{40}", info.get("revision", "")):
        fail(f"Revisão inválida para {model_id}")
    for path, file_info in info.get("files", {}).items():
        if not re.fullmatch(r"[0-9a-f]{64}", file_info.get("sha256", "")):
            fail(f"SHA-256 inválido: {model_id}/{path}")
        if not isinstance(file_info.get("size_bytes"), int) or file_info["size_bytes"] <= 0:
            fail(f"Tamanho inválido: {model_id}/{path}")

for source in ROOT.rglob("*.py"):
    if any(part in {".git", "__pycache__", ".venv"} for part in source.parts):
        continue
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        is_snapshot_download = isinstance(function, ast.Name) and function.id == "snapshot_download"
        if is_snapshot_download and not any(keyword.arg == "revision" for keyword in node.keywords):
            fail(f"snapshot_download sem revisão explícita: {source.relative_to(ROOT)}:{node.lineno}")
        is_torch_load = (
            isinstance(function, ast.Attribute)
            and function.attr == "load"
            and isinstance(function.value, ast.Name)
            and function.value.id == "torch"
        )
        if is_torch_load and not any(
            keyword.arg == "weights_only"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in node.keywords
        ):
            fail(f"torch.load sem weights_only=True: {source.relative_to(ROOT)}:{node.lineno}")

print("Supply-chain manifests e chamadas torch.load estão coerentes.")
