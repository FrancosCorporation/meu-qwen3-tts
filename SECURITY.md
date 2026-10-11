# Segurança e cadeia de fornecimento

## Reportar uma falha

Não publique um exploit ou arquivo de voz malicioso em issue pública. Use um aviso privado de
segurança do GitHub quando estiver habilitado para este repositório; se não estiver, procure os
mantenedores pelos canais oficiais do projeto. Envie passos de reprodução e impacto sem anexar
tokens, arquivos pessoais de voz ou outros dados privados.

## Regras que precisam continuar verdadeiras

1. Nós de inferência só aceitam IDs `Qwen/*` declarados em `MODEL_ID_MAP` e também presentes em
   `model_checksums.json`.
2. Todo download usa uma revisão completa de commit; os pesos e o tokenizer do repositório são
   conferidos pelos tamanhos e SHA-256 registrados nesse manifesto.
3. Todo `torch.load` deve usar `weights_only=True`. Não adicione fallback sem essa opção nem
   habilite `trust_remote_code=True` para modelos baixados.
4. Perfis `.pt`/`.pth` só podem ser lidos dentro da biblioteca de vozes do ComfyUI e são carregados
   com `weights_only=True`. `safetensors` é preferido quando disponível.

## Atualizar modelos ou dependências

Uma atualização de modelo precisa de revisão humana do repositório oficial, do commit completo,
da licença e do SHA-256 de `model.safetensors` e `speech_tokenizer/model.safetensors`. Registre
esses dados em `model_checksums.json` e atualize a tabela do README na mesma mudança. O workflow
estático confere a allowlist, as revisões, os hashes, os dois manifests de dependências e o uso de
`torch.load`; ele não baixa nem executa pesos de modelo.

As faixas de dependências em `requirements.txt` e `pyproject.toml` devem permanecer idênticas. Não
amplie a faixa do PyTorch ou Transformers sem revisar a compatibilidade do modelo, dos kernels e da
proteção de desserialização.
