# Plan — meu-qwen3-tts

## 1. Visão geral
Custom nodes de ComfyUI para Qwen-TTS (clonagem e instrução de voz) + Whisper + duração de áudio. Código vendor do modelo em `qwen_tts/` é preservado intacto (upstream `1038lab/ComfyUI-QwenTTS`); hardening 2026-10-11 cobriu carregamento seguro de pesos (`torch.load` `weights_only=True`), pins de revisão e checksums de modelo, cache com teto, validação de referências de áudio.

## 2. Arquitetura e tecnologias
Python 3, PyTorch, ComfyUI, HuggingFace Hub. Supply-chain verificado por `scripts/check_supply_chain.py` em CI.

## 3. Funcionalidades existentes
- [x] Nós de TTS/Whisper/duração registrados e consistentes com workflows de exemplo.
- [x] Carregamento seguro de modelos (hashes, revisões completas, safetensors preferido).
- [x] CI supply-chain — **modernizada nesta sessão** (Ubuntu 24.04, actions v7, matriz Python 3.12/3.13).

## 4. Objetivos futuros
- OBJ-01 (P2, concluído nesta sessão): CI fixa + matriz.
- OBJ-02 (P3): TEST-01 — smoke do registro de nós com stubs de torch/transformers (vedado agora; exige elaborar stubs com cuidado para não testar o mock).
- OBJ-03 (P3): script de diagnóstico local (`--check-models`) validando checksums sem baixar.

## 5–9. Melhorias/Roadmap/Riscos/Ideias
- Não reescrever `qwen_tts/` (modelo vendor) — toda correção em `nodes.py`/`scripts/`.
- Perfis de voz limitados à biblioteca — não adicionar vozes arbitrárias sem revisão de licença.

*(Gerado em 2026-10-14 pelo agente orquestrador)*
