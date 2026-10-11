# ComfyUI-QwenTTS
# A clean, efficient ComfyUI custom node for Qwen3-TTS (Text-to-Speech) functionality.
# This implementation provides high-quality speech generation and voice cloning
# capabilities using the Qwen3-TTS model family.
#
# Models License Notice:
# - Qwen3-TTS: Apache-2.0 License (https://huggingface.co/collections/Qwen/qwen3-tts)
#
# This integration script follows GPL-3.0 License.
# When using or modifying this code, please respect both the original model licenses
# and this integration's license terms.
#
# Source: https://github.com/1038lab/ComfyUI-QwenTTS

import os
import sys
import gc
import hashlib
import json
import logging
import shutil
import stat
from pathlib import Path
from typing import Any, Tuple, Optional
from contextlib import nullcontext
from collections import OrderedDict

import numpy as np
import torch
import folder_paths
import comfy.model_management as model_management

try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

current_dir = Path(__file__).parent
logger = logging.getLogger("Qwen3-TTS")
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
try:
    from .voice_storage import load_voice as _load_safe_voice
except ImportError:
    from voice_storage import load_voice as _load_safe_voice

qwen_pkg_dir = current_dir / "qwen_tts"
if qwen_pkg_dir.exists() and str(qwen_pkg_dir) not in sys.path:
    sys.path.insert(0, str(qwen_pkg_dir))

_IMPORT_ERROR = None
Qwen3TTSModel = None

def _ensure_qwen_package():
    if not qwen_pkg_dir.exists():
        return
    pkg_name = "qwen_tts"
    if pkg_name not in sys.modules:
        try:
            import types
            module = types.ModuleType(pkg_name)
            module.__path__ = [str(qwen_pkg_dir)]
            sys.modules[pkg_name] = module
        except Exception:
            return
    core_dir = qwen_pkg_dir / "core"
    if core_dir.exists() and "qwen_tts.core" not in sys.modules:
        try:
            import types
            core_module = types.ModuleType("qwen_tts.core")
            core_module.__path__ = [str(core_dir)]
            sys.modules["qwen_tts.core"] = core_module

            tokenizer_12hz_dir = core_dir / "tokenizer_12hz"
            if tokenizer_12hz_dir.exists():
                token_pkg = types.ModuleType("qwen_tts.core.tokenizer_12hz")
                token_pkg.__path__ = [str(tokenizer_12hz_dir)]
                sys.modules["qwen_tts.core.tokenizer_12hz"] = token_pkg

                import importlib.util
                cfg_path = tokenizer_12hz_dir / "configuration_qwen3_tts_tokenizer_v2.py"
                mdl_path = tokenizer_12hz_dir / "modeling_qwen3_tts_tokenizer_v2.py"
                if cfg_path.exists():
                    spec = importlib.util.spec_from_file_location(
                        "qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2",
                        str(cfg_path),
                    )
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        sys.modules[spec.name] = mod
                        spec.loader.exec_module(mod)
                        core_module.Qwen3TTSTokenizerV2Config = getattr(mod, "Qwen3TTSTokenizerV2Config")
                if mdl_path.exists():
                    spec = importlib.util.spec_from_file_location(
                        "qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2",
                        str(mdl_path),
                    )
                    if spec and spec.loader:
                        mod = importlib.util.module_from_spec(spec)
                        sys.modules[spec.name] = mod
                        spec.loader.exec_module(mod)
                        core_module.Qwen3TTSTokenizerV2Model = getattr(mod, "Qwen3TTSTokenizerV2Model")

                try:
                    from transformers import PretrainedConfig, PreTrainedModel

                    class Qwen3TTSTokenizerV1Config(PretrainedConfig):
                        model_type = "qwen3_tts_tokenizer_25hz"

                    class Qwen3TTSTokenizerV1Model(PreTrainedModel):
                        config_class = Qwen3TTSTokenizerV1Config

                        def __init__(self, config):
                            super().__init__(config)

                        def forward(self, *args, **kwargs):
                            raise RuntimeError("Tokenizer 25Hz is not supported in this node.")

                    core_module.Qwen3TTSTokenizerV1Config = Qwen3TTSTokenizerV1Config
                    core_module.Qwen3TTSTokenizerV1Model = Qwen3TTSTokenizerV1Model
                except Exception:
                    pass
        except Exception:
            return

def _load_qwen3_model():
    global Qwen3TTSModel, _IMPORT_ERROR
    if Qwen3TTSModel is not None:
        return Qwen3TTSModel
    _ensure_qwen_package()
    try:
        import importlib.util
        model_path = qwen_pkg_dir / "inference" / "qwen3_tts_model.py"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing qwen3_tts_model.py at {model_path}")
        spec = importlib.util.spec_from_file_location(
            "qwen_tts.inference.qwen3_tts_model",
            str(model_path),
        )
        if not spec or not spec.loader:
            raise ImportError("Failed to create spec for qwen3_tts_model")
        module = importlib.util.module_from_spec(spec)
        sys.modules["qwen_tts.inference.qwen3_tts_model"] = module
        spec.loader.exec_module(module)
        Qwen3TTSModel = getattr(module, "Qwen3TTSModel")
        return Qwen3TTSModel
    except Exception as e:
        _IMPORT_ERROR = str(e)
        print(f"[Qwen3-TTS] Failed to import qwen3_tts_model: {e}")
        return None

LANGUAGE_CHOICES = [
    "Auto",
    "Chinese",
    "English",
    "Japanese",
    "Korean",
    "French",
    "German",
    "Spanish",
    "Portuguese",
    "Russian",
    "Italian",
]
LANGUAGE_MAP = {
    "Auto": "auto",
    "Chinese": "chinese",
    "English": "english",
    "Japanese": "japanese",
    "Korean": "korean",
    "French": "french",
    "German": "german",
    "Spanish": "spanish",
    "Portuguese": "portuguese",
    "Russian": "russian",
    "Italian": "italian",
}

SPEAKER_CHOICES = [
    "Aiden",
    "Dylan",
    "Eric",
    "Ono_Anna",
    "Ryan",
    "Serena",
    "Sohee",
    "Uncle_Fu",
    "Vivian",
]

MODEL_ID_MAP = {
    ("Base", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    ("Base", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    ("CustomVoice", "0.6B"): "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    ("CustomVoice", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    ("VoiceDesign", "1.7B"): "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}
with open(current_dir / "model_checksums.json", "r", encoding="utf-8") as manifest_file:
    MODEL_MANIFEST = json.load(manifest_file)["models"]

ATTENTION_OPTIONS = ["auto", "sage_attn", "flash_attn", "sdpa", "eager"]
MAX_CACHED_MODELS = 2
_MODEL_CACHE: OrderedDict = OrderedDict()
_VERIFIED_MODEL_FILES = {}


def _available_devices():
    devices = ["auto"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


def _flash_attn_available():
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


def _sage_attn_available():
    try:
        import sageattention  # noqa: F401
        return True
    except Exception:
        return False


def _resolve_attention_impl(attention: str, device: str):
    if not device.startswith("cuda"):
        return None
    if attention == "auto":
        if _flash_attn_available():
            return "flash_attention_2"
        if _sage_attn_available():
            return "sage_attn"
        return "sdpa"
    if attention == "sage_attn":
        return "sage_attn"
    if attention == "flash_attn":
        if _flash_attn_available():
            return "flash_attention_2"
        return None
    if attention == "sdpa":
        return "sdpa"
    if attention == "eager":
        return "eager"
    return None


def _patch_sage_attention(model):
    try:
        from sageattention import sageattn
    except Exception:
        raise RuntimeError("sageattention is not installed. Please install it to use sage_attn.")

    patched = 0
    for name, module in model.model.named_modules():
        if getattr(module, "_qwen_sage_patched", False):
            continue
        if "attn" not in name.lower() and "Attention" not in type(module).__name__:
            continue
        if not hasattr(module, "forward"):
            continue

        original_forward = module.forward

        def make_sage_forward(orig_forward):
            def sage_forward(*args, **kwargs):
                if len(args) >= 3:
                    q, k, v = args[0], args[1], args[2]
                    if hasattr(q, "shape") and hasattr(k, "shape") and hasattr(v, "shape"):
                        attn_mask = kwargs.get("attention_mask", None)
                        try:
                            return sageattn(q, k, v, is_causal=False, attn_mask=attn_mask)
                        except Exception:
                            return orig_forward(*args, **kwargs)
                return orig_forward(*args, **kwargs)

            return sage_forward

        module.forward = make_sage_forward(original_forward)
        module._qwen_sage_patched = True
        patched += 1

    print(f"[Qwen3-TTS] sage_attn patched modules: {patched}")


def _resolve_device(device_choice: str):
    if device_choice == "auto":
        if torch.cuda.is_available():
            return str(model_management.get_torch_device())
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device_choice == "cuda" and torch.cuda.is_available():
        return str(model_management.get_torch_device())
    if device_choice == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(precision: str, device: str):
    if device == "mps":
        if precision in ("bf16", "fp16"):
            return torch.float16
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def _maybe_autocast(device: str, precision: str):
    if not device.startswith("cuda"):
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _model_store_root():
    tts_paths = _get_tts_paths()
    if tts_paths:
        return os.path.join(tts_paths[0], "Qwen3-TTS")
    base = folder_paths.models_dir
    return os.path.join(base, "TTS", "Qwen3-TTS")


def _get_tts_paths():
    folders = getattr(folder_paths, "folder_names_and_paths", {}) or {}
    for key in ("tts", "TTS"):
        if key in folders:
            return folder_paths.get_folder_paths(key) or []
    return []


def _verify_model_files(model_path: str, model_id: str):
    model_info = MODEL_MANIFEST.get(model_id)
    if not model_info:
        raise ValueError(f"Modelo sem revisão/checksum aprovado: {model_id}")

    root = Path(model_path).resolve()
    fingerprints = []
    for relative_path, expected in model_info["files"].items():
        file_path = (root / relative_path).resolve()
        if not file_path.is_relative_to(root) or not file_path.is_file():
            raise RuntimeError(f"Arquivo obrigatório ausente ou fora do modelo: {relative_path}")
        stat = file_path.stat()
        if stat.st_size != expected["size_bytes"]:
            raise RuntimeError(f"Tamanho não confere para {relative_path} em {root}")
        fingerprints.append((relative_path, stat.st_size, stat.st_mtime_ns))

    verification_key = (model_id, str(root), model_info["revision"], tuple(fingerprints))
    if verification_key in _VERIFIED_MODEL_FILES:
        return

    for relative_path, expected in model_info["files"].items():
        file_path = root / relative_path
        digest = hashlib.sha256()
        with file_path.open("rb") as model_file:
            for chunk in iter(lambda: model_file.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected["sha256"]:
            raise RuntimeError(f"SHA-256 não confere para {relative_path} em {root}; remova a pasta e baixe novamente")

    _VERIFIED_MODEL_FILES[verification_key] = True


def _find_local_model(model_id: str) -> Optional[str]:
    if model_id not in MODEL_MANIFEST:
        raise ValueError(f"Modelo fora da allowlist: {model_id}")

    model_name = model_id.rsplit("/", 1)[-1]
    try:
        candidates = [os.path.join(_model_store_root(), model_name)]
        tts_roots = _get_tts_paths()
    except Exception:
        logger.exception("Falha ao resolver os diretórios locais de modelo para %s", model_id)
        raise
    for root in tts_roots:
        candidates.append(os.path.join(root, "Qwen3-TTS", model_name))

    for path in dict.fromkeys(candidates):
        try:
            contents = os.listdir(path)
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError:
            logger.exception("Falha ao verificar a pasta local de modelo: %s", path)
            raise
        if contents:
            model_root = Path(path)
            arquivos_obrigatorios = MODEL_MANIFEST[model_id]["files"]
            try:
                status_arquivos = [
                    (model_root / relative_path).stat()
                    for relative_path in arquivos_obrigatorios
                ]
            except (FileNotFoundError, NotADirectoryError):
                logger.warning("Download incompleto em %s; retomando a revisão fixa", path)
                continue
            except OSError:
                logger.exception("Falha ao verificar pesos em %s", path)
                raise
            if any(not stat.S_ISREG(item.st_mode) for item in status_arquivos):
                logger.warning("Download incompleto em %s; retomando a revisão fixa", path)
                continue
            _verify_model_files(path, model_id)
            return path
    return None


def _download_model(model_id: str) -> str:
    if model_id not in MODEL_MANIFEST:
        raise ValueError(f"Modelo fora da allowlist: {model_id}")
    if snapshot_download is None:
        raise RuntimeError("huggingface_hub não está instalado; instale as dependências do plugin")

    model_info = MODEL_MANIFEST[model_id]
    target_root = _model_store_root()
    target_dir = os.path.join(target_root, model_id.rsplit("/", 1)[-1])
    try:
        os.makedirs(target_root, exist_ok=True)
        livre = shutil.disk_usage(target_root).free
        minimo = int(model_info["size_bytes"] * 1.1) + 512 * 1024 * 1024
        if livre < minimo:
            raise OSError(
                f"espaço insuficiente: {livre / 1024**3:.1f} GiB livres; "
                f"são necessários aproximadamente {minimo / 1024**3:.1f} GiB"
            )
        snapshot_download(
            repo_id=model_id,
            revision=model_info["revision"],
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )
        _verify_model_files(target_dir, model_id)
        return target_dir
    except Exception as exc:
        logger.exception("Falha ao preparar o modelo %s na pasta %s", model_id, target_dir)
        try:
            livre = shutil.disk_usage(target_root).free
            detalhe_espaco = f" Espaço livre: {livre / 1024**3:.1f} GiB."
        except OSError:
            detalhe_espaco = " Não foi possível consultar o espaço livre."
        raise RuntimeError(
            f"Não foi possível baixar/verificar {model_id} na revisão "
            f"{model_info['revision']}: {exc}.{detalhe_espaco}"
        ) from exc


def _resolve_model_source(model_id: str) -> str:
    if model_id not in MODEL_MANIFEST:
        raise ValueError(f"Modelo fora da allowlist: {model_id}")
    local_path = _find_local_model(model_id)
    if local_path:
        return local_path
    return _download_model(model_id)


def _load_model(model_type: str, model_size: str, device_choice: str, precision: str, attention: str = "auto"):
    tipos = {"base": "Base", "customvoice": "CustomVoice", "voicedesign": "VoiceDesign"}
    tamanhos = {"0.6b": "0.6B", "1.7b": "1.7B"}
    precisao = {"fp32": "fp32", "float32": "fp32", "fp16": "fp16", "float16": "fp16",
                "bf16": "bf16", "bfloat16": "bf16"}
    dispositivos = {"auto", "cuda", "mps", "cpu"}
    model_type = tipos.get(str(model_type).strip().lower())
    model_size = tamanhos.get(str(model_size).strip().lower())
    precision = precisao.get(str(precision).strip().lower())
    device_choice = str(device_choice).strip().lower()
    attention = str(attention).strip().lower()
    if model_type is None or model_size is None:
        raise ValueError("Tipo ou tamanho de modelo inválido")
    if precision is None:
        raise ValueError("Precisão inválida; use fp32, fp16 ou bf16")
    if device_choice not in dispositivos:
        raise ValueError("Dispositivo inválido; use auto, cuda, mps ou cpu")
    if attention not in ATTENTION_OPTIONS:
        raise ValueError(f"Backend de atenção inválido: {attention}")
    if model_type == "VoiceDesign" and model_size != "1.7B":
        raise ValueError("VoiceDesign only supports 1.7B models.")
    if Qwen3TTSModel is None:
        _load_qwen3_model()
    if Qwen3TTSModel is None:
        hint = _IMPORT_ERROR or "unknown import error"
        raise RuntimeError(
            "qwen_tts is not available. Please install dependencies in your ComfyUI environment "
            f"and check the package path. Import error: {hint}"
        )

    model_id = MODEL_ID_MAP.get((model_type, model_size))
    if model_id is None:
        raise ValueError(f"Unsupported model type/size: {model_type}/{model_size}")
    revision = MODEL_MANIFEST[model_id]["revision"]

    device = _resolve_device(device_choice)
    dtype = _resolve_dtype(precision, device)

    attn_impl = _resolve_attention_impl(attention, device)
    if attn_impl == "sage_attn" and not _sage_attn_available():
        logger.warning("sage_attn não está disponível; usando o backend padrão")
        attn_impl = None
    cache_key = (model_type, model_size, device, precision, attn_impl or "default")
    if cache_key in _MODEL_CACHE:
        _MODEL_CACHE.move_to_end(cache_key)
        print(f"[Qwen3-TTS] Using cached model: {model_type} {model_size} on {device} ({precision})")
        return _MODEL_CACHE[cache_key]
    logger.info("Cache miss de modelo; carregando chave normalizada %s", cache_key)

    source = _resolve_model_source(model_id)
    if device.startswith("cuda"):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    try:
        if attn_impl == "sage_attn":
            model = Qwen3TTSModel.from_pretrained(
                source, device_map=device, dtype=dtype, revision=revision
            )
            _patch_sage_attention(model)
            print("[Qwen3-TTS] attention: sage_attn")
        else:
            kwargs = {"device_map": device, "dtype": dtype, "revision": revision}
            if attn_impl:
                kwargs["attn_implementation"] = attn_impl
            model = Qwen3TTSModel.from_pretrained(source, **kwargs)
            if device.startswith("cuda") and attn_impl:
                print(f"[Qwen3-TTS] attention: {attn_impl}")
    except Exception as e:
        if "attn_implementation" in str(e) or "flash" in str(e).lower():
            print("[Qwen3-TTS] attention unavailable, fallback to default attention")
            cache_key = (model_type, model_size, device, precision, "default")
            if cache_key in _MODEL_CACHE:
                _MODEL_CACHE.move_to_end(cache_key)
                return _MODEL_CACHE[cache_key]
            model = Qwen3TTSModel.from_pretrained(
                source, device_map=device, dtype=dtype, revision=revision
            )
        else:
            raise

    _MODEL_CACHE[cache_key] = model
    if len(_MODEL_CACHE) > MAX_CACHED_MODELS:
        evicted_key, evicted_model = _MODEL_CACHE.popitem(last=False)
        del evicted_model
        logger.info("Modelo removido do cache LRU: %s", evicted_key)
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                logger.exception("Falha ao liberar o cache CUDA após evicção")
    return model


def _clear_model_cache():
    _MODEL_CACHE.clear()
    gc.collect()
    try:
        model_management.soft_empty_cache()
    except Exception:
        logger.exception("Falha ao liberar o cache do ComfyUI")
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            logger.exception("Falha ao liberar o cache CUDA")


def _set_seed(seed: int):
    if seed is None or seed < 0:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2 ** 32))


def _get_prompt_item_class():
    try:
        _load_qwen3_model()
        module = __import__("qwen_tts.inference.qwen3_tts_model", fromlist=["VoiceClonePromptItem"])
        return getattr(module, "VoiceClonePromptItem", None)
    except Exception:
        return None


def _deserialize_prompt_item(data):
    cls = _get_prompt_item_class()
    if cls is None:
        return data
    return cls(
        ref_code=data.get("ref_code"),
        ref_spk_embedding=data.get("ref_spk_embedding"),
        x_vector_only_mode=bool(data.get("x_vector_only_mode")),
        icl_mode=bool(data.get("icl_mode")),
        ref_text=data.get("ref_text"),
    )


def _deserialize_prompt_items(items):
    return [_deserialize_prompt_item(item) for item in items]


def _load_voice_from_file(path: str):
    if not path:
        return None
    return _deserialize_prompt_items(_load_safe_voice(path))


def _audio_to_tuple(audio: Any) -> Tuple[np.ndarray, int]:
    waveform = None
    sr = None

    if isinstance(audio, dict):
        if "waveform" in audio:
            waveform = audio.get("waveform")
            sr = audio.get("sample_rate") or audio.get("sr") or audio.get("sampling_rate")
        elif "data" in audio and "sampling_rate" in audio:
            waveform = audio.get("data")
            sr = audio.get("sampling_rate")
        elif "audio" in audio and isinstance(audio["audio"], (tuple, list)):
            a0, a1 = audio["audio"]
            if isinstance(a0, (int, float)):
                sr, waveform = int(a0), a1
            else:
                waveform, sr = a0, int(a1)
    elif isinstance(audio, tuple) and len(audio) == 2:
        a0, a1 = audio
        if isinstance(a0, (int, float)):
            sr, waveform = int(a0), a1
        else:
            waveform, sr = a0, int(a1)
    elif isinstance(audio, list) and len(audio) == 2:
        waveform, sr = audio[0], int(audio[1])

    if sr is None or waveform is None:
        raise ValueError("Invalid AUDIO input")

    try:
        sample_rate = float(sr.item() if hasattr(sr, "item") else sr)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("AUDIO sample rate must be between 8000 and 192000 Hz") from exc
    if not np.isfinite(sample_rate) or not sample_rate.is_integer() or not 8000 <= sample_rate <= 192000:
        raise ValueError("AUDIO sample rate must be between 8000 and 192000 Hz")
    sample_rate = int(sample_rate)
    max_samples = sample_rate * 30

    if isinstance(waveform, torch.Tensor):
        wav = waveform.detach()
        if not wav.is_floating_point():
            wav = wav.float()
        if wav.dim() == 3:
            if wav.shape[0] != 1:
                raise ValueError("AUDIO reference must contain exactly one audio item")
            wav = wav.squeeze(0)
        if wav.dim() == 2:
            if wav.shape[0] <= 8 and wav.shape[1] > 8:
                wav = wav.mean(dim=0)
            elif wav.shape[1] <= 8 and wav.shape[0] > 8:
                wav = wav.mean(dim=1)
            else:
                raise ValueError("AUDIO reference must contain mono audio or at most 8 channels")
        if wav.dim() != 1:
            raise ValueError("AUDIO reference has an unsupported shape")
        if wav.numel() > max_samples:
            raise ValueError("AUDIO reference exceeds the 30-second limit")
        wav = wav.float().cpu().numpy()
    else:
        try:
            wav = np.asarray(waveform)
        except (TypeError, ValueError) as exc:
            raise ValueError("AUDIO reference has an unsupported shape") from exc
        if wav.ndim == 3:
            if wav.shape[0] != 1:
                raise ValueError("AUDIO reference must contain exactly one audio item")
            wav = np.squeeze(wav, axis=0)
        if wav.ndim == 2:
            if wav.shape[0] <= 8 and wav.shape[1] > 8:
                wav = np.mean(wav, axis=0)
            elif wav.shape[1] <= 8 and wav.shape[0] > 8:
                wav = np.mean(wav, axis=1)
            else:
                raise ValueError("AUDIO reference must contain mono audio or at most 8 channels")
        if wav.ndim != 1:
            raise ValueError("AUDIO reference has an unsupported shape")
        if wav.size > max_samples:
            raise ValueError("AUDIO reference exceeds the 30-second limit")

    wav = wav.astype(np.float32)
    if not np.isfinite(wav).all():
        raise ValueError("AUDIO reference contains invalid numeric values")
    if wav.size < 1024:
        pad = 1024 - wav.size
        wav = np.concatenate([wav, np.zeros(pad, dtype=np.float32)])

    return wav, sample_rate


def _to_comfy_audio(wavs, sr: int):
    if isinstance(wavs, list) and len(wavs) > 0:
        wav = wavs[0]
    else:
        wav = wavs
    if isinstance(wav, np.ndarray):
        tensor = torch.from_numpy(wav).float()
    else:
        tensor = wav.float()
    if tensor.dim() > 1:
        tensor = tensor.squeeze()
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    return {"waveform": tensor, "sample_rate": int(sr)}


def _custom_voice_generate(
    text: str,
    speaker: str,
    model_size: str,
    device: str,
    precision: str,
    language: str,
    instruct: str = "",
    seed: int = -1,
    max_new_tokens: int = 2048,
    do_sample: bool = True,
    top_p: float = 0.9,
    top_k: int = 50,
    temperature: float = 0.9,
    repetition_penalty: float = 1.0,
    attention: str = "auto",
    unload_models: bool = False,
):
    if not text or not text.strip():
        raise ValueError("Text is required")
    _set_seed(seed)
    model = _load_model("CustomVoice", model_size, device, precision, attention)
    mapped_lang = LANGUAGE_MAP.get(language, "auto")
    with _maybe_autocast(_resolve_device(device), precision):
        wavs, sr = model.generate_custom_voice(
            text=text,
            language=mapped_lang,
            speaker=str(speaker).lower().replace(" ", "_"),
            instruct=instruct if instruct and instruct.strip() else None,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
    audio = _to_comfy_audio(wavs, sr)
    if unload_models:
        del model
        _clear_model_cache()
    return (audio,)


def _voice_design_generate(
    text: str,
    instruct: str,
    model_size: str,
    device: str,
    precision: str,
    language: str,
    seed: int = -1,
    max_new_tokens: int = 2048,
    do_sample: bool = True,
    top_p: float = 0.9,
    top_k: int = 50,
    temperature: float = 0.9,
    repetition_penalty: float = 1.0,
    attention: str = "auto",
    unload_models: bool = False,
):
    if not text or not text.strip() or not instruct or not instruct.strip():
        raise ValueError("Text and instruct are required")
    _set_seed(seed)
    model = _load_model("VoiceDesign", model_size, device, precision, attention)
    mapped_lang = LANGUAGE_MAP.get(language, "auto")
    with _maybe_autocast(_resolve_device(device), precision):
        wavs, sr = model.generate_voice_design(
            text=text,
            language=mapped_lang,
            instruct=instruct,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
    audio = _to_comfy_audio(wavs, sr)
    if unload_models:
        del model
        _clear_model_cache()
    return (audio,)


def _voice_clone_generate(
    reference_audio: Any,
    target_text: str,
    model_size: str,
    device: str,
    precision: str,
    language: str,
    reference_text: str = "",
    x_vector_only: bool = False,
    allow_empty_ref_text: bool = False,
    voice: Any = None,
    seed: int = -1,
    max_new_tokens: int = 2048,
    do_sample: bool = True,
    top_p: float = 0.9,
    top_k: int = 50,
    temperature: float = 0.9,
    repetition_penalty: float = 1.0,
    attention: str = "auto",
    unload_models: bool = False,
):
    if not target_text or not target_text.strip():
        raise ValueError("Target text is required")
    prompt = None
    audio_tuple = None
    if voice is not None:
        if isinstance(voice, str) and voice.strip():
            prompt = _load_voice_from_file(voice.strip())
        elif isinstance(voice, list):
            prompt = voice
    if prompt is None:
        if (not reference_text or not reference_text.strip()) and not x_vector_only:
            if allow_empty_ref_text:
                x_vector_only = True
            else:
                raise ValueError("reference_text is required unless x_vector_only is enabled")
        if reference_audio is None:
            raise ValueError("reference_audio is required when voice is not provided")
        audio_tuple = _audio_to_tuple(reference_audio)
    _set_seed(seed)
    model = _load_model("Base", model_size, device, precision, attention)
    mapped_lang = LANGUAGE_MAP.get(language, "auto")
    with _maybe_autocast(_resolve_device(device), precision):
        if prompt is not None:
            wavs, sr = model.generate_voice_clone(
                text=target_text,
                language=mapped_lang,
                voice_clone_prompt=prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )
        else:
            wavs, sr = model.generate_voice_clone(
                text=target_text,
                language=mapped_lang,
                ref_audio=audio_tuple,
                ref_text=reference_text.strip() if reference_text and reference_text.strip() else None,
                x_vector_only_mode=bool(x_vector_only),
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )
    audio = _to_comfy_audio(wavs, sr)
    if unload_models:
        del model
        _clear_model_cache()
    return (audio,)


class Qwen3TTSCustomVoiceBasic:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": OrderedDict([
                ("target_text", ("STRING", {"multiline": True, "default": "Hello from Qwen3-TTS.", "tooltip": "Text to synthesize"})),
                ("speaker", (SPEAKER_CHOICES, {"default": "Ryan"})),
                ("model_size", (["0.6B", "1.7B"], {"default": "1.7B"})),
                ("language", (LANGUAGE_CHOICES, {"default": "Auto"})),
            ]),
            "optional": OrderedDict([
                ("instruct", ("STRING", {"multiline": True, "default": "", "tooltip": "Style instruction"})),
                ("unload_models", ("BOOLEAN", {"default": True, "tooltip": "Unload cached models after generation"})),
                ("seed", ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff})),
            ]),
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "🧪AILab/🎙️QwenTTS"

    def generate(self, text, speaker, model_size, language, instruct="", unload_models=True, seed=-1):
        import torch
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        return _custom_voice_generate(
            text=text,
            speaker=speaker,
            model_size=model_size,
            device="auto",
            precision="bf16",
            language=language,
            instruct=instruct,
            seed=seed,
            max_new_tokens=2048,
            do_sample=False,
            top_p=0.9,
            top_k=50,
            temperature=0.9,
            repetition_penalty=1.0,
            unload_models=unload_models,
        )


class Qwen3TTSCustomVoiceAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": OrderedDict([
                ("target_text", ("STRING", {"multiline": True, "default": "Hello from Qwen3-TTS.", "tooltip": "Text to synthesize"})),
                ("speaker", (SPEAKER_CHOICES, {"default": "Ryan"})),
                ("model_size", (["0.6B", "1.7B"], {"default": "1.7B"})),
                ("device", (_available_devices(), {"default": "auto"})),
                ("precision", (["bf16", "fp16", "fp32"], {"default": "bf16"})),
                ("language", (LANGUAGE_CHOICES, {"default": "Auto"})),
            ]),
            "optional": OrderedDict([
                ("instruct", ("STRING", {"multiline": True, "default": "", "tooltip": "Style instruction"})),
                ("max_new_tokens", ("INT", {"default": 2048, "min": 256, "max": 4096, "step": 256})),
                ("do_sample", ("BOOLEAN", {"default": False})),
                ("top_p", ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05})),
                ("top_k", ("INT", {"default": 50, "min": 0, "max": 200, "step": 1})),
                ("temperature", ("FLOAT", {"default": 0.9, "min": 0.1, "max": 2.0, "step": 0.05})),
                ("repetition_penalty", ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05})),
                ("attention", (ATTENTION_OPTIONS, {"default": "auto"})),
                ("unload_models", ("BOOLEAN", {"default": True, "tooltip": "Unload cached models after generation"})),
                ("seed", ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff})),
            ]),
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "🧪AILab/🎙️QwenTTS"

    def generate(self, text, speaker, model_size, device, precision, language, instruct="", max_new_tokens=2048, do_sample=True, top_p=0.9, top_k=50, temperature=0.9, repetition_penalty=1.0, attention="auto", unload_models=True, seed=-1):
        import torch
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        return _custom_voice_generate(
            text=text,
            speaker=speaker,
            model_size=model_size,
            device=device,
            precision=precision,
            language=language,
            instruct=instruct,
            seed=seed,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            attention=attention,
            unload_models=unload_models,
        )


class Qwen3TTSVoiceDesignBasic:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": OrderedDict([
                ("target_text", ("STRING", {"multiline": True, "default": "Hello from Qwen3-TTS VoiceDesign.", "tooltip": "Text to synthesize"})),
                ("instruct", ("STRING", {"multiline": True, "default": "A warm, gentle female voice.", "tooltip": "Voice description"})),
                ("model_size", (["1.7B"], {"default": "1.7B"})),
                ("language", (LANGUAGE_CHOICES, {"default": "Auto"})),
            ]),
            "optional": OrderedDict([
                ("unload_models", ("BOOLEAN", {"default": True, "tooltip": "Unload cached models after generation"})),
                ("seed", ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff})),
            ]),
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "🧪AILab/🎙️QwenTTS"

    def generate(self, text, instruct, model_size, language, unload_models=True, seed=-1):
        import torch
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        return _voice_design_generate(
            text=text,
            instruct=instruct,
            model_size=model_size,
            device="auto",
            precision="bf16",
            language=language,
            seed=seed,
            max_new_tokens=2048,
            do_sample=False,
            top_p=0.9,
            top_k=50,
            temperature=0.9,
            repetition_penalty=1.0,
            unload_models=unload_models,
        )


class Qwen3TTSVoiceDesignAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": OrderedDict([
                ("target_text", ("STRING", {"multiline": True, "default": "Hello from Qwen3-TTS VoiceDesign.", "tooltip": "Text to synthesize"})),
                ("instruct", ("STRING", {"multiline": True, "default": "A warm, gentle female voice.", "tooltip": "Voice description"})),
                ("model_size", (["1.7B"], {"default": "1.7B"})),
                ("device", (_available_devices(), {"default": "auto"})),
                ("precision", (["bf16", "fp16", "fp32"], {"default": "bf16"})),
                ("language", (LANGUAGE_CHOICES, {"default": "Auto"})),
            ]),
            "optional": OrderedDict([
                ("max_new_tokens", ("INT", {"default": 2048, "min": 256, "max": 4096, "step": 256})),
                ("do_sample", ("BOOLEAN", {"default": False})),
                ("top_p", ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05})),
                ("top_k", ("INT", {"default": 50, "min": 0, "max": 200, "step": 1})),
                ("temperature", ("FLOAT", {"default": 0.9, "min": 0.1, "max": 2.0, "step": 0.05})),
                ("repetition_penalty", ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05})),
                ("attention", (ATTENTION_OPTIONS, {"default": "auto"})),
                ("unload_models", ("BOOLEAN", {"default": True})),
                ("seed", ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff})),
            ]),
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "🧪AILab/🎙️QwenTTS"

    def generate(self, text, instruct, model_size, device, precision, language, max_new_tokens=2048, do_sample=True, top_p=0.9, top_k=50, temperature=0.9, repetition_penalty=1.0, attention="auto", unload_models=True, seed=-1):
        import torch
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        return _voice_design_generate(
            text=text,
            instruct=instruct,
            model_size=model_size,
            device=device,
            precision=precision,
            language=language,
            seed=seed,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            attention=attention,
            unload_models=unload_models,
        )


class Qwen3TTSVoiceCloneBasic:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": OrderedDict([
                ("target_text", ("STRING", {"multiline": True, "default": "Hello, this is a cloned voice.", "tooltip": "Text to speak"})),
                ("model_size", (["0.6B", "1.7B"], {"default": "1.7B"})),
                ("language", (LANGUAGE_CHOICES, {"default": "Auto"})),
            ]),
            "optional": OrderedDict([
                ("reference_audio", ("AUDIO", {"tooltip": "Reference audio for cloning (not needed if voice is provided)"})),
                ("reference_text", ("STRING", {"multiline": True, "default": "", "tooltip": "Transcript of reference audio"})),
                ("x_vector_only", ("BOOLEAN", {"default": False, "tooltip": "Skip ref_text by using speaker embedding only"})),
                ("voice", ("VOICE",)),
                ("unload_models", ("BOOLEAN", {"default": True, "tooltip": "Unload cached models after generation"})),
                ("seed", ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff})),
            ]),
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "🧪AILab/🎙️QwenTTS"

    def generate(self, target_text, model_size, language, reference_audio=None, reference_text="", x_vector_only=False, voice=None, unload_models=True, seed=-1):
        import torch
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        return _voice_clone_generate(
            reference_audio=reference_audio,
            target_text=target_text,
            model_size=model_size,
            device="auto",
            precision="bf16",
            language=language,
            reference_text=reference_text,
            x_vector_only=x_vector_only,
            allow_empty_ref_text=True,
            voice=voice,
            seed=seed,
            max_new_tokens=2048,
            do_sample=False,
            top_p=0.9,
            top_k=50,
            temperature=0.9,
            repetition_penalty=1.0,
            unload_models=unload_models,
        )


class Qwen3TTSVoiceCloneAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": OrderedDict([
                ("target_text", ("STRING", {"multiline": True, "default": "Hello, this is a cloned voice.", "tooltip": "Text to speak"})),
                ("model_size", (["0.6B", "1.7B"], {"default": "1.7B"})),
                ("device", (_available_devices(), {"default": "auto"})),
                ("precision", (["bf16", "fp16", "fp32"], {"default": "bf16"})),
                ("language", (LANGUAGE_CHOICES, {"default": "Auto"})),
            ]),
            "optional": OrderedDict([
                ("reference_audio", ("AUDIO", {"tooltip": "Reference audio for cloning (not needed if voice is provided)"})),
                ("reference_text", ("STRING", {"multiline": True, "default": "", "tooltip": "Transcript of reference audio"})),
                ("x_vector_only", ("BOOLEAN", {"default": False, "tooltip": "Skip ref_text by using speaker embedding only"})),
                ("voice", ("VOICE",)),
                ("max_new_tokens", ("INT", {"default": 2048, "min": 256, "max": 4096, "step": 256})),
                ("do_sample", ("BOOLEAN", {"default": False})),
                ("top_p", ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05})),
                ("top_k", ("INT", {"default": 50, "min": 0, "max": 200, "step": 1})),
                ("temperature", ("FLOAT", {"default": 0.9, "min": 0.1, "max": 2.0, "step": 0.05})),
                ("repetition_penalty", ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05})),
                ("attention", (ATTENTION_OPTIONS, {"default": "auto"})),
                ("unload_models", ("BOOLEAN", {"default": True})),
                ("seed", ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff})),
            ]),
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "🧪AILab/🎙️QwenTTS"

    def generate(self, target_text, model_size, device, precision, language, reference_audio=None, reference_text="", x_vector_only=False, voice=None, max_new_tokens=2048, do_sample=True, top_p=0.9, top_k=50, temperature=0.9, repetition_penalty=1.0, attention="auto", unload_models=True, seed=-1):
        import torch
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        return _voice_clone_generate(
            reference_audio=reference_audio,
            target_text=target_text,
            model_size=model_size,
            device=device,
            precision=precision,
            language=language,
            reference_text=reference_text,
            x_vector_only=x_vector_only,
            allow_empty_ref_text=False,
            voice=voice,
            seed=seed,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            attention=attention,
            unload_models=unload_models,
        )


NODE_CLASS_MAPPINGS = {
    "AILab_Qwen3TTSCustomVoice": Qwen3TTSCustomVoiceBasic,
    "AILab_Qwen3TTSCustomVoice_Advanced": Qwen3TTSCustomVoiceAdvanced,
    "AILab_Qwen3TTSVoiceDesign": Qwen3TTSVoiceDesignBasic,
    "AILab_Qwen3TTSVoiceDesign_Advanced": Qwen3TTSVoiceDesignAdvanced,
    "AILab_Qwen3TTSVoiceClone": Qwen3TTSVoiceCloneBasic,
    "AILab_Qwen3TTSVoiceClone_Advanced": Qwen3TTSVoiceCloneAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_Qwen3TTSCustomVoice": "Custom Voice (QwenTTS)",
    "AILab_Qwen3TTSCustomVoice_Advanced": "Custom Voice (QwenTTS) Advanced",
    "AILab_Qwen3TTSVoiceDesign": "Voice Design (QwenTTS)",
    "AILab_Qwen3TTSVoiceDesign_Advanced": "Voice Design (QwenTTS) Advanced",
    "AILab_Qwen3TTSVoiceClone": "Voice Clone (QwenTTS)",
    "AILab_Qwen3TTSVoiceClone_Advanced": "Voice Clone (QwenTTS) Advanced",
}
