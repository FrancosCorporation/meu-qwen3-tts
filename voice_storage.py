"""Armazenamento e leitura segura de perfis de voz do ComfyUI."""

import json
import inspect
import logging
import os
import tempfile
from pathlib import Path

import torch

logger = logging.getLogger("Qwen3-TTS.voice-storage")
MAX_VOICE_FILE_BYTES = 200 * 1024 * 1024
VOICE_EXTENSIONS = {".pt", ".pth", ".safetensors"}
_VOICE_TENSOR_KEYS = ("ref_code", "ref_spk_embedding")


def _supports_safe_torch_load():
    try:
        return "weights_only" in inspect.signature(torch.load).parameters
    except (TypeError, ValueError):
        return False


if not _supports_safe_torch_load():
    raise RuntimeError("PyTorch incompatível: atualize para uma versão com torch.load(weights_only=True)")


def voice_library_dir(create=True):
    try:
        import folder_paths
        base_dir = folder_paths.get_output_directory()
    except Exception:
        logger.exception("Não foi possível consultar a pasta de saída do ComfyUI")
        base_dir = None
    if not base_dir:
        base_dir = Path(__file__).parent / "voice_library"
    root = (Path(base_dir).expanduser() / "qwen3-tts_voices").resolve()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def _within_root(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def voice_save_dir(save_path=""):
    root = voice_library_dir()
    directory = Path(save_path).expanduser() if save_path and save_path.strip() else root
    if not directory.is_absolute():
        directory = root / directory
    directory = directory.resolve()
    if not _within_root(directory, root):
        raise ValueError(f"A pasta de vozes precisa ficar dentro de {root}")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def validate_voice_file(path):
    if not isinstance(path, (str, os.PathLike)) or not str(path).strip():
        raise ValueError("Informe um arquivo de voz salvo na biblioteca do ComfyUI")

    root = voice_library_dir()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        candidate = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError("Arquivo de voz não encontrado na biblioteca aprovada") from exc

    if not _within_root(candidate, root):
        raise ValueError("Arquivo de voz fora da biblioteca aprovada")
    if candidate.suffix.lower() not in VOICE_EXTENSIONS:
        raise ValueError("A voz precisa usar .pt, .pth ou .safetensors")
    if not candidate.is_file():
        raise ValueError("O caminho informado não é um arquivo de voz")
    tamanho = candidate.stat().st_size
    if tamanho <= 0 or tamanho > MAX_VOICE_FILE_BYTES:
        raise ValueError("Arquivo de voz vazio ou maior que o limite de 200 MiB")
    return candidate


def _voice_tensor_dict(prompt_items):
    if not isinstance(prompt_items, list) or not prompt_items or len(prompt_items) > 64:
        raise ValueError("Perfil de voz inválido: quantidade de itens fora do limite")

    tensors = {}
    rows = []
    for index, item in enumerate(prompt_items):
        if not isinstance(item, dict):
            raise ValueError("Perfil de voz inválido: item precisa ser um objeto")
        row = {
            "x_vector_only_mode": bool(item.get("x_vector_only_mode")),
            "icl_mode": bool(item.get("icl_mode")),
            "ref_text": item.get("ref_text"),
        }
        if row["ref_text"] is not None and (
            not isinstance(row["ref_text"], str) or len(row["ref_text"]) > 100000
        ):
            raise ValueError("Perfil de voz inválido: ref_text precisa ser texto")
        for key in _VOICE_TENSOR_KEYS:
            value = item.get(key)
            if value is None:
                row[key] = None
                continue
            try:
                tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
                tensor = tensor.detach().cpu().contiguous()
            except Exception as exc:
                raise ValueError(f"Perfil de voz inválido: {key} não é um tensor") from exc
            tensor_name = f"item_{index}_{key}"
            tensors[tensor_name] = tensor
            row[key] = tensor_name
        rows.append(row)
    return tensors, rows


def _save_safetensors(path, prompt_items):
    from safetensors.torch import save_file

    tensors, rows = _voice_tensor_dict(prompt_items)
    if not tensors:
        raise ValueError("Perfil de voz inválido: nenhum tensor de referência foi salvo")
    metadata = {
        "format": "qwen3-tts-voice-v1",
        "items": json.dumps(rows, ensure_ascii=False, separators=(",", ":")),
    }
    save_file(tensors, str(path), metadata=metadata)


def _load_safetensors(path):
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as arquivo:
        metadata = arquivo.metadata() or {}
        if metadata.get("format") != "qwen3-tts-voice-v1":
            raise ValueError("Formato safetensors de voz não reconhecido")
        raw_items = metadata.get("items", "")
        if len(raw_items) > 1024 * 1024:
            raise ValueError("Metadados do perfil de voz excedem 1 MiB")
        rows = json.loads(raw_items)
        if not isinstance(rows, list) or not rows or len(rows) > 64:
            raise ValueError("Perfil de voz inválido: quantidade de itens fora do limite")

        nomes = set(arquivo.keys())
        items = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Perfil de voz inválido: item precisa ser um objeto")
            item = {
                "x_vector_only_mode": bool(row.get("x_vector_only_mode")),
                "icl_mode": bool(row.get("icl_mode")),
                "ref_text": row.get("ref_text"),
            }
            if item["ref_text"] is not None and (
                not isinstance(item["ref_text"], str) or len(item["ref_text"]) > 100000
            ):
                raise ValueError("Perfil de voz inválido: ref_text precisa ser texto")
            for key in _VOICE_TENSOR_KEYS:
                nome_tensor = row.get(key)
                if nome_tensor is None:
                    item[key] = None
                elif not isinstance(nome_tensor, str) or nome_tensor not in nomes:
                    raise ValueError("Perfil de voz inválido: tensor ausente")
                else:
                    item[key] = arquivo.get_tensor(nome_tensor)
            items.append(item)
    return items


def _load_torch_voice(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("PyTorch não aceitou weights_only=True; atualize o PyTorch") from exc


def load_voice(path):
    arquivo = validate_voice_file(path)
    if arquivo.suffix.lower() == ".safetensors":
        try:
            payload = _load_safetensors(arquivo)
        except ImportError as exc:
            raise RuntimeError("Instale safetensors para carregar este perfil de voz") from exc
    else:
        payload = _load_torch_voice(arquivo)

    if isinstance(payload, dict) and "prompt" in payload:
        payload = payload["prompt"]
    elif isinstance(payload, dict) and "voice_clone_prompt" in payload:
        payload = payload["voice_clone_prompt"]
    if (not isinstance(payload, list) or not payload or len(payload) > 64
            or any(not isinstance(item, dict) for item in payload)):
        raise ValueError("Conteúdo de perfil de voz inválido")
    return payload


def save_voice(prompt_items, save_path="", name="voice"):
    directory = voice_save_dir(save_path)
    safe_name = "".join(char for char in (name or "voice") if char.isalnum() or char in ("-", "_"))
    safe_name = safe_name.strip() or "voice"
    use_safetensors = False
    try:
        import safetensors.torch  # noqa: F401
        use_safetensors = True
    except ImportError:
        pass

    suffix = ".safetensors" if use_safetensors else ".pt"
    target = directory / f"{safe_name}{suffix}"
    fd, temporary = tempfile.mkstemp(prefix=f".{safe_name}-", suffix=suffix, dir=directory)
    os.close(fd)
    try:
        if use_safetensors:
            _save_safetensors(temporary, prompt_items)
        else:
            torch.save({"version": 2, "prompt": prompt_items}, temporary)
        size = os.path.getsize(temporary)
        if size <= 0 or size > MAX_VOICE_FILE_BYTES:
            raise ValueError("Perfil de voz vazio ou maior que o limite de 200 MiB")
        os.replace(temporary, target)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            logger.warning("Não foi possível remover o arquivo temporário de voz %s", temporary)
        raise
    return str(target)
