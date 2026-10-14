# Resumo — meu-qwen3-tts

## Sessão 2026-10-14 (planejamento padronizado + modernização da CI)
- **Resumo:** nós ComfyUI já endurecidos (SEC-01..04, DEVOPS-01..03, PERF-01, 2026-10-11). Nesta sessão: CI fixada (Ubuntu 24.04, checkout v7/setup-python v7 SHA, matriz Python 3.12/3.13); docs padronizadas.
- **Validações:** `py_compile` nos módulos de topo OK; `compileall` + `check_supply_chain.py` rodam na CI.
- **Resultado:** [x] CI verde; [x] docs; [x] `__pycache__` regenerável descartado; nada a preservar.
- **Continuidade:** HEAD sincronizado; checkout removível. Sem testes na esteira (regra).
