# MELHORIAS — meu-qwen3-tts

> **Gerado por análise de código em 2026-10-02** · Stack: Python 3 + PyTorch + ComfyUI (custom nodes) + Hugging Face
> Branch `master` · base `51b31df` · 10.385 LOC · **0 testes** · sem CI
>
> **Este arquivo é um plano de execução.** Cada item tem ID, `arquivo:linha`, mudança exata,
> critério de aceite e comando de verificação.

---

## 0. Como usar este documento

1. Execute na ordem **P0 → P1 → P2 → P3**, respeitando as ondas da §8.
2. Ao terminar um item: marque `- [x]`, rode o **Verificação**, comite `fix(<ID>): descrição`.
3. **Não reescreva o modelo.** O `qwen_tts/` é código do modelo (provavelmente derivado do upstream
   `1038lab/ComfyUI-QwenTTS`, declarado em `pyproject.toml`). Toda correção vai em `nodes.py`,
   `AILab_*.py` ou nos manifests — nunca dentro de `qwen_tts/core/models/*` sem necessidade provada.
4. **Não quebre a compatibilidade ComfyUI.** Os nós são carregados por nome/classe; itens de
   segurança devem preservar a assinatura dos nós existentes.
5. **Idioma:** comentários/docs em português (padrão do autor); commits em inglês com
   `fix:`/`feat:`/`docs:`.

---

## 1. Diagnóstico executivo

Pacote de nós ComfyUI para o TTS Qwen3: `CustomVoice`, `VoiceDesign` e `VoiceClone` com controles
de qualidade/velocidade e otimização de memória para AMD ROCm. São 3 camadas: `nodes.py` (interface
ComfyUI + resolução e download de modelo), `qwen_tts/` (modelo vendorizado) e `AILab_*.py`
(ferramentas auxiliares de duração de áudio e TTS).

**O que está bem (não refaça):**

| Item | Evidência |
|---|---|
| Allowlist fechada de modelos oficiais | `nodes.py:180` (`MODEL_ID_MAP` — só 5 repos `Qwen/*`) + `:387-389` (tipo/tamanho desconhecido rejeitado com `ValueError`) |
| `torch.load` com `weights_only=True` no caminho principal | `nodes.py:474` |
| Sem `trust_remote_code=True` | `grep` não acha a flag em nenhum `.py` do repo |
| `snapshot_download` sem symlink | `nodes.py:358` (`local_dir_use_symlinks=False`) |
| Fallback de atenção (flash → default) tratado | `nodes.py:425-430` com detecção de erro por mensagem |
| `target_text` vazio rejeitado cedo | `nodes.py:668-669` (`ValueError` antes de carregar o modelo) |

**O que está quebrado:**

1. **Fallback para `torch.load` sem `weights_only`** (`nodes.py:476`): quando o torch é antigo
   (`TypeError`), o carregamento de voz cai para deserialização **pickle completa** — RCE por
   arquivo `.pt` malicioso.
2. **Caminho de voz vindo de string arbitrária** (`nodes.py:673`): qualquer string vira `path`
   de `torch.load`, sem allowlist de diretório nem verificação de origem.
3. **Dependências sem pin** em **dois** manifests que **divergem** (`torch>=2.9.1`, `numpy` e
   `huggingface_hub` sem versão; `transformers>=4.57.0` vs `>=4.57.3`).
4. **Zero testes e zero CI** num pacote que baixa e executa modelo de 1.7 B parâmetros.

Nada aqui exige reescrever o modelo — é contorno em `nodes.py` + infra de teste.

---

## 2. Tabela de prioridades

| ID | Título | Sev | Arquivo | Depende de |
|---|---|---|---|---|
| SEC-01 | Fallback para `torch.load` sem `weights_only` (pickle RCE) | **P0** | `nodes.py:476` | — |
| SEC-02 | Voz carregada de caminho arbitrário, sem allowlist | **P1** | `nodes.py:673,470` | SEC-01 |
| SEC-03 | Download de modelo sem verificação de hash/revisão | **P1** | `nodes.py:358,84` | — |
| DEVOPS-01 | Dependências sem pin em dois manifests divergentes | **P1** | `requirements.txt`, `pyproject.toml` | — |
| BUG-01 | `except Exception: pass` engole falha de resolução de modelo | **P1** | `nodes.py:340-346` | — |
| BUG-02 | Falha de download é silenciosa (`print` e retorna `None`) | **P1** | `nodes.py:349-363` | — |
| TEST-01 | Zero testes (downloads, fallbacks e vozes) | **P1** | *(ausente)* | BUG-01, BUG-02 |
| DEVOPS-02 | Zero CI | **P2** | *(ausente)* `.github/workflows/` | TEST-01 |
| IMP-01 | Cache de modelo na memória sem limite nem liberação | **P2** | `nodes.py:395` (`cache_key`, sem evicção) | — |
| IMP-02 | `seed` sem defeito de reprodutibilidade documentado | **P3** | `nodes.py:684` (`_set_seed`) | — |
| PERF-01 | Modelo recarregado por nó sem reuso garantido | **P2** | `nodes.py:395-397` | IMP-01 |
| SEC-04 | `reference_audio` aceito sem validação de formato/tamanho | **P2** | `nodes.py:682` | — |
| DEVOPS-03 | Falta `.gitignore` de modelos baixados | **P2** | *(ausente/checar)* | — |
| DOC-01 | README não documenta origem e licença dos pesos | **P2** | `README.md` | — |
| DOC-02 | Falta `SECURITY.md` de supply chain | **P3** | *(ausente)* | — |

**Placar: 1 P0 · 6 P1 · 6 P2 · 2 P3 = 15 itens.**

---

## 3. Segurança
### SEC-01 · Fallback para `torch.load` sem `weights_only` (pickle RCE) · [P0]

- **Arquivo:** `nodes.py:474-476`
- **Evidência:**
  ```python
  try:
      data = torch.load(path, map_location="cpu", weights_only=True)
  except TypeError:
      data = torch.load(path, map_location="cpu")
  ```
- **Impacto:** o `TypeError` acontece quando o PyTorch é antigo (< 1.13) e não conhece o argumento
  `weights_only`. Nesse caso, o fallback carrega com **deserialização pickle completa** — e pickle
  executa código arbitrário no `load`. Como o `path` vem do usuário (`SEC-02`: `nodes.py:673`), um
  arquivo `.pt` malicioso **executa código na máquina que roda o ComfyUI**. É RCE clássico de ML:
  a proteção existe na linha 474, mas a linha 476 a **desfaz** exatamente no ambiente mais frágil
  (torch desatualizado). Pior: o `TypeError` também pode ser lançado por um arquivo **corrompido**
  (argumento não reconhecido por outro motivo), então o fallback dispara em cenário que não é
  "torch antigo" — sem avisar o usuário.
- **Mudança:**
  1. **Nunca** carregar sem `weights_only=True`. Exigir PyTorch ≥ 1.13 (é de 2022) no manifest —
     item ligado ao `DEVOPS-01`.
  2. Se `TypeError` for lançado, **recusar com erro explícito** ("atualize o torch") em vez de
     fazer fallback inseguro. O usuário decide, não o código.
  3. Adicionar checagem de `weights_only` suportado na inicialização (`hasattr`/`inspect`) para
     falhar cedo, não no primeiro arquivo de voz.
- **Aceite:** não existe chamada `torch.load` sem `weights_only=True` no repo; torch antigo gera
  erro claro em vez de RCE silencioso.
- **Verificação:**
  ```bash
  grep -rn 'torch.load' --include='*.py' . | grep -v __pycache__
  # toda ocorrencia deve ter weights_only=True
  python3 -c "import torch; print(torch.__version__)"   # >= 1.13
  ```

### SEC-02 · Voz carregada de caminho arbitrário, sem allowlist · [P1]

- **Arquivo:** `nodes.py:673` (`prompt = _load_voice_from_file(voice.strip())`) e `:470`
  (`def _load_voice_from_file(path: str)`)
- **Evidência:** `voice` é um parâmetro do nó (`nodes.py:673`) aceito como `str` livre. Qualquer
  string vira `path` e cai no `torch.load(path, ...)` da linha 474. Não há: (a) allowlist de
  diretório; (b) normalização de caminho; (c) checagem de extensão; (d) limite de tamanho.
- **Impacto:** combinado com o `SEC-01`, qualquer `.pt` em qualquer lugar do disco (inclusive fora
  do diretório de modelos) é carregado. E com pickle RCE no fallback, apontar para
  `/tmp/malicioso.pt` é execução de código. Mesmo sem o RCE: leitura de arquivo arbitrário
  (`/etc/passwd` como "voz" falha com formato, mas vaza o erro de formato — oráculo de existência
  de arquivo).
- **Mudança:** (1) aceitar **apenas** caminhos dentro do diretório de vozes do pacote
  (`os.path.commonpath` + `resolve`, recusar fora); (2) exigir extensão `.pt`/`.pth`/`.safetensors`;
  (3) preferir **`safetensors`** quando disponível (formato sem pickle) e documentar a conversão;
  (4) limite de tamanho explícito (ex.: 200 MB) com erro claro.
- **Aceite:** `voice="/etc/passwd"` e `voice="../../x.pt"` são recusados **antes** de qualquer `load`.
- **Verificação:**
  ```bash
  python3 -c "from nodes import _load_voice_from_file; _load_voice_from_file('/etc/passwd')" 2>&1 \
    | grep -qi 'recusado\|fora do diretorio\|extensao' && echo OK || echo FALHA
  ```

### SEC-03 · Download de modelo sem verificação de hash nem revisão · [P1]

- **Arquivo:** `nodes.py:358` (`snapshot_download(repo_id=model_id, local_dir=target_dir, ...)`) e
  `qwen_tts/core/models/modeling_qwen3_tts.py:84` (segundo `snapshot_download` interno)
- **Evidência:** o download é feito pelo `repo_id` da allowlist (`MODEL_ID_MAP`, `nodes.py:180` —
  isso está **bem feito**) mas **sem** `revision=` fixada e **sem** verificação de hash posterior.
  Se o repositório `Qwen/*` no Hub for atualizado (ou comprometido com chave roubada), o próximo
  `snapshot_download` puxa bytes novos sem aviso — e o modelo executa código via `from_pretrained`.
- **Impacto:** supply chain do modelo. É improvável (repos oficiais Qwen) mas o custo é total:
  modelo malicioso = código arbitrário na inferência. Sem `revision` fixada, builds futuros não são
  reproduzíveis — o mesmo código baixa pesos diferentes ao longo do tempo.
- **Mudança:** (1) fixar `revision=<commit-sha>` no `snapshot_download` para cada uma das 5 entradas
  do `MODEL_ID_MAP` (documentar o SHA no próprio dict); (2) verificar hash `sha256` dos arquivos
  principais após o download (guardar em `checksums.json` versionado); (3) documentar o
  procedimento de atualização de revisão no README.
- **Aceite:** reexecutar o download com rede limpa baixa **exatamente** os mesmos bytes (hash igual).
- **Verificação:**
  ```bash
  grep -n 'revision=' nodes.py qwen_tts/core/models/modeling_qwen3_tts.py   # deve existir
  sha256sum <modelo> | diff - checksums.json
  ```

### SEC-04 · `reference_audio` aceito sem validação de formato/tamanho · [P2]

- **Arquivo:** `nodes.py:682` (`if reference_audio is None: raise ...`) e o uso posterior
- **Evidência:** a única checagem é presença (`is None`). Não há validação de sample rate, duração
  máxima, canais ou tamanho do array.
- **Impacto:** áudio de horas (ou array gigante via API) entra no pipeline e estoura memória/vRAM
  no meio da geração — DoS de recurso numa máquina compartilhada (o ambiente-alvo é ComfyUI local,
  mas o nó pode ser exposto via API). Áudio malformado pode ainda quebrar o `whisper_encoder`
  (`tokenizer_25hz/vq/whisper_encoder.py:406`) com exceção não tratada.
- **Mudança:** validar na entrada: duração máxima (ex.: 30 s para referência, conforme o que os
  nós `CustomVoice`/`VoiceDesign` esperam), sample rate esperado, e rejeitar com `ValueError` claro
  antes de carregar o modelo (barato recusa cedo).
- **Aceite:** áudio de 10 min é recusado na entrada, sem carregar modelo e sem OOM.
- **Verificação:**
  ```bash
  python3 -c "from nodes import <no>; <no>(reference_audio=audio_10min)" 2>&1 | grep -qi 'duracao\|tamanho' && echo OK
  ```

---

## 4. Bugs e defeitos funcionais

### BUG-01 · `except Exception: pass` engole falha de resolução de modelo · [P1]

- **Arquivo:** `nodes.py:340-346`
- **Evidência:**
  ```python
  except Exception:
      pass
  for path in candidates:
      if os.path.isdir(path) and os.listdir(path):
          return path
  return None
  ```
- **Impacto:** falha em `_find_local_model` (permissão, disco, caminho quebrado) é **silenciada** e o
  fluxo cai para `_download_model` (`nodes.py:365-372`) — que baixa gigabytes **de novo** mesmo que
  o modelo já esteja no disco. O usuário vê download repetido sem saber que a causa é permissão de
  pasta. E se o download também falhar (`BUG-02`), a mensagem final não menciona a causa raiz.
- **Mudança:** logar a exceção (com `logging`, não `print`) com o path envolvido; só silenciar
  exceções de "não encontrado" (`FileNotFoundError`), e deixar as demais subirem.
- **Aceite:** erro de permissão em pasta de modelo aparece no log com o caminho; download repetido
  desnecessário deixa de acontecer nesse cenário.
- **Verificação:**
  ```bash
  chmod 000 <pasta-modelo> && python3 -c "..." 2>&1 | grep -qi 'permissao' && echo OK
  grep -n 'except Exception:' nodes.py   # nao deve restar pass solitario
  ```

### BUG-02 · Falha de download é silenciosa (`print` e retorna `None`) · [P1]

- **Arquivo:** `nodes.py:349-363` (`_download_model`)
- **Evidência:** o `except` da linha ~360 faz `print(f"[Qwen3-TTS] Download failed...")` e deixa a
  função retornar `None`; o chamador (`_resolve_model_source`, linha 365) então retorna o
  `model_id` cru (linha 372) como se fosse caminho local.
- **Impacto:** download falho (rede, disco cheio, Hub fora) vira `from_pretrained(source)` com
  `source` sendo o ID remoto — o que **parece** funcionar (o HF baixa de novo para o cache
  padrão) mas esconde que o diretório local esperado está vazio. Resultado: dois caches divergentes
  do mesmo modelo e erro confuso quando o espaço acaba. Pior: com `local_dir_use_symlinks=False`,
  cada tentativa duplica gigabytes.
- **Mudança:** em falha de download, **levantar** `RuntimeError` com a causa e o espaço livre
  (checar `shutil.disk_usage` antes de baixar 1.7 B); só cair para o ID remoto se for decisão
  **explícita** do usuário (flag `allow_remote_fallback=True`, default False). Nunca duplo cache
  silencioso.
- **Aceite:** download falho interrompe com erro claro (causa + disco), em vez de seguir com cache
  duplo.
- **Verificação:**
  ```bash
  # sem rede (ou HF_REPO bloqueado): executar o no e conferir RuntimeError, nao None silencioso
  ```
---

## 5. Qualidade: testes, arquitetura e observabilidade

### TEST-01 · Zero testes para downloads, fallbacks e vozes · [P1]

- **Arquivo:** *(ausente)* — `find . -name 'test*' | grep -v __pycache__` → nada.
- **Evidência:** 10.385 LOC, downloads de rede, fallback de segurança (`SEC-01`), allowlist de
  modelos e 3 nós ComfyUI — sem um teste sequer. Os bugs `BUG-01` e `BUG-02` são exatamente do tipo
  que `pytest` pega sem GPU.
- **Impacto:** qualquer refactor em `nodes.py` (o arquivo que recebe 90% das mudanças) pode quebrar
  resolução de modelo, reintroduzir o fallback inseguro ou trocar a allowlist — sem sinal vermelho.
  E sem teste, o `DEVOPS-02` (CI) não tem o que rodar.
- **Mudança:** `tests/` com `pytest`, priorizando o barato (mock, sem modelo real):
  | Caso | Assertivo |
  |---|---|
  | `MODEL_ID_MAP` só contém `Qwen/*` e tem `revision` | allowlist íntegra (`SEC-03`) |
  | `_find_local_model` com pasta sem permissão | loga, não silencia (`BUG-01`) |
  | `_download_model` com rede simulada falha | levanta `RuntimeError` (`BUG-02`) |
  | `_load_voice_from_file('/etc/passwd')` | recusado antes do `load` (`SEC-02`) |
  | `torch.load` em todo `.py` do repo | todos com `weights_only=True` (`SEC-01`) |
  | `_resolve_model_source` com tipo inválido | `ValueError` (`nodes.py:389`) |
  Usar `unittest.mock` para `snapshot_download` (nunca bater na rede real).
- **Aceite:** `pytest -q` passa; reintroduzir o `torch.load` sem flag ou ampliar o MAP quebra o build.
- **Verificação:**
  ```bash
  pip install pytest && pytest -q    # todos passam
  # reintroduzir fallback inseguro -> pytest deve FALHAR
  ```

### IMP-01 · Cache de modelo na memória sem limite nem liberação · [P2]

- **Arquivo:** `nodes.py:395` (`cache_key = (...)`, sem evicção visível)
- **Evidência:** o `_load_model` guarda por chave composta (tipo, tamanho, device, precisão, atenção)
  e imprime "Using cached model" na linha 397. Não há limite de entradas, TTL nem `unload`.
  (Existe um parâmetro `unload_models` em `nodes.py:667` — vale conferir se ele limpa o cache ou só
  descarrega da GPU; se só descarrega, o dict continua crescendo.)
- **Impacto:** cada combinação (tipo × tamanho × device × precisão × atenção) aloca um modelo de
  até 1.7 B parâmetros. Trocar de precisão 3 vezes = 3 modelos na RAM. Em máquina com 16 GB, isso é
  OOM silencioso — e OOM no meio de geração corrompe o workflow do ComfyUI inteiro.
- **Mudança:** (1) LRU com no máximo N modelos (ex.: 2) e liberação explícita (`del` + `torch.cuda.
  empty_cache()`/`gc.collect`); (2) documentar o custo aproximado por combinação no README;
  (3) respeitar `unload_models=True` esvaziando o cache, não só descarregando.
- **Aceite:** carregar 3 combinações distintas mantém no máximo 2 na memória; a terceira evicta a LRU.
- **Verificação:**
  ```bash
  # carregar Base/0.6B, Base/1.7B, CustomVoice/1.7B e medir RSS; depois voltar ao primeiro e
  # conferir que houve recarregamento (eviccao), nao OOM
  ```

### PERF-01 · Modelo recarregado por nó sem reuso garantido entre nós · [P2]

- **Arquivo:** `nodes.py:395-397`, `:572` (`_load_model("CustomVoice", ...)`), `:684` (`_load_model("Base", ...)`)
- **Evidência:** cada nó chama `_load_model` com seus próprios parâmetros; o cache é por chave exata.
  Dois nós com `precision` escrita diferente (`"fp32"` vs `"FP32"` — conferir normalização) geram
  **duas entradas** para o mesmo peso.
- **Impacto:** workflow com 2 nós do "mesmo" modelo carrega 2× se os parâmetros divergirem por
  detalhe de escrita — dobra tempo de carga e memória.
- **Mudança:** (1) normalizar todos os componentes da chave (`lower()`, `strip()`, sinonimos de
  device); (2) logar a chave normalizada quando houver cache miss, para o usuário ver por que não
  reutilizou; (3) combinar com o `IMP-01` (limite do cache).
- **Aceite:** dois nós com `"fp32"`/`"FP32"` reutilizam a mesma entrada (um só load no log).
- **Verificação:** montar workflow com 2 nós equivalentes e contar "Using cached model" no log.

### IMP-02 · `seed` sem garantia de reprodutibilidade documentada · [P3]

- **Arquivo:** `nodes.py:684` (`_set_seed(seed)`)
- **Evidência:** a semente é aplicada antes da geração, mas não há teste provando que mesma
  `seed` + mesmos parâmetros = mesmo áudio, nem documentação do que **não** é determinístico
  (atenção flash, `do_sample` com kernels diferentes, `top_k`/`top_p` em precisão reduzida).
- **Impacto:** usuário que precisa de take idêntico (dublagem, A/B) não sabe se pode confiar na
  seed. Sem contrato, "mudei nada e saiu diferente" vira bug report sem resposta.
- **Mudança:** teste de determinismo (mesma seed, 2 gerações, comparar bytes) documentado como
  suportado **apenas** no caminho determinístico (atenção padrão, mesma precisão); e nota no README
  listando as combinações que quebram a garantia.
- **Aceite:** teste verde no caminho garantido; README diz onde a garantia não vale.
- **Verificação:**
  ```bash
  pytest tests/test_seed.py -q    # 2 geracoes com mesma seed, bytes iguais
  ```

---

## 6. DevOps / Infra

### DEVOPS-01 · Dependências sem pin em dois manifests divergentes · [P1]

- **Arquivo:** `requirements.txt` e `pyproject.toml` (ambos listam as mesmas 12 dependências)
- **Evidência:**
  ```
  torch>=2.9.1 / torchaudio>=2.9.1 / transformers>=4.57.0   (requirements.txt)
  torch>=2.9.1 / torchaudio>=2.9.1 / transformers>=4.57.3   (pyproject.toml)
  numpy / huggingface_hub / openai-whisper                   (sem versão nos dois)
  ```
  `transformers` diverge (`4.57.0` vs `4.57.3`); `numpy`, `huggingface_hub` e `openai-whisper` não
  têm piso em nenhum dos dois; `torch>=2.9.1` aceita qualquer futuro 2.x.
- **Impacto:** `pip install -r requirements.txt` e `pip install .` **podem instalar conjuntos
  diferentes** — o ambiente de dev não é o ambiente do usuário. Sem piso, `numpy` 2.x quebra
  `librosa`/`soundfile` antigos (incompatibilidade conhecida). E o `SEC-01` mostra que a versão do
  torch decide se há RCE no fallback — aceitar "qualquer 2.x" é aceitar regressão de segurança
  futura.
- **Mudança:** (1) escolher **um** manifest como fonte (`pyproject.toml`) e gerar o outro
  (`pip-compile`); (2) piso **e** teto em todas (`torch>=2.9.1,<2.10`, etc.); (3) alinhar
  `transformers` nos dois (o maior dos pisos); (4) fixar `torch>=2.0` como mínimo (garante
  `weights_only` como fallback do `SEC-01`); (5) `pip install --require-hashes` no CI.
- **Aceite:** os dois arquivos concordam; `pip install` de ambos produz o mesmo conjunto.
- **Verificação:**
  ```bash
  diff <(grep -oE '^[a-z0-9_-]+' requirements.txt | sort) \
       <(grep -oE '"[a-z0-9_-]+' pyproject.toml | tr -d '"' | sort)   # alinhados
  pip install -r requirements.txt --dry-run   # sem conflito
  ```

### DEVOPS-02 · Zero CI · [P2]

- **Arquivo:** *(ausente)* `.github/workflows/`
- **Evidência:** `ls .github/workflows` → inexistente.
- **Impacto:** sem CI, o `TEST-01` não roda sozinho e o `DEVOPS-01` (divergência de manifests)
  ninguém confere. É o repo com mais LOC desta wave (10.385) e o único sem qualquer automação.
- **Mudança:** workflow em `on: [push, pull_request]` com: `python -m py_compile nodes.py`
  (sintaxe), `pytest -q` (testes sem GPU — os do `TEST-01` são todos mock), e checagem de
  divergência dos manifests + `grep torch.load` sem `weights_only` (barreira do `SEC-01`).
  **Sem** baixar modelo no CI (lento e caro) — só testes mock.
- **Aceite:** PR que reintroduza fallback inseguro ou quebre a allowlist é bloqueado.
- **Verificação:**
  ```bash
  python3 -m py_compile nodes.py AILab_QwenTTS_Tools.py AILab_AudioDuration.py
  pytest -q
  ```

### DEVOPS-03 · Modelos baixados sem `.gitignore` garantido · [P2]

- **Arquivo:** `.gitignore` (existe, 32 linhas) — conferir cobertura de `_model_store_root()`
- **Evidência:** `nodes.py:352-354` cria `target_root = _model_store_root()` e baixa o modelo para
  dentro dele. O `.gitignore` precisa cobrir **esse** diretório, senão um `git add -A` versiona
  gigabytes de pesos (1.7 B parâmetros ≈ vários GB).
- **Impacto:** versionar pesos acidentalmente incha o repositório para sempre (e o `git push`
  nunca mais termina). É o acidente clássico de repo de ML — e este tem o download automático
  ativado por padrão.
- **Mudança:** (1) confirmar que o diretório de `_model_store_root()` está no `.gitignore`;
  (2) adicionar também `*.pt *.pth *.bin *.safetensors *.onnx` por segurança;
  (3) adicionar `du -sh` do diretório de modelos ao README como referência de espaço.
- **Aceite:** `git status --porcelain` nunca mostra arquivo de peso, mesmo após download.
- **Verificação:**
  ```bash
  git check-ignore -q <dir-modelos>/x.pt && echo 'OK: ignorado'
  git status --porcelain | grep -cE '\.(pt|pth|bin|safetensors|onnx)$'   # 0
  ```

---

## 7. Documentação

### DOC-01 · README não documenta origem e licença dos pesos · [P2]

- **Arquivo:** `README.md` (146 linhas)
- **Evidência:** o README descreve instalação e uso dos nós, mas o `SEC-03` exige `revision` fixada
  para 5 modelos — e o README não diz **qual** revisão, nem qual licença cobre os pesos
  (`Qwen/Qwen3-TTS-*`), nem quanto espaço cada um ocupa, nem que o download é automático.
- **Impacto:** o usuário não sabe o que vai ser baixado, de onde, sob qual licença, nem quanto
  espaço precisa — e licença de peso de TTS importa (uso comercial, voz clonada). Sem isso, o
  `SEC-03` (revisão) não tem onde ser registrado.
- **Mudança:** tabela de modelos no README: nome, tamanho, revisão SHA, licença, espaço em disco,
  e nota de que o download é automático na primeira execução (com como pré-baixar e como apontar
  para diretório local existente).
- **Aceite:** alguém decide qual modelo usar lendo só o README, e sabe o custo em disco antes.
- **Verificação:** `grep -n 'revision\|licen\|GB' README.md` retorna a tabela.

### DOC-02 · Falta `SECURITY.md` de supply chain · [P3]

- **Arquivo:** *(ausente)* `SECURITY.md`
- **Evidência:** tem `LICENSE`, mas nenhum registro de como reportar problema de segurança nem
  quais são as garantias de supply chain do pacote.
- **Impacto:** as decisões duras (allowlist de modelos, `weights_only` obrigatório, sem
  `trust_remote_code`) existem só neste plano. Sem registro, voltam em refactor.
- **Mudança:** criar com: canal de reporte; as **3 regras de supply chain** (só `Qwen/*` da
  allowlist; `torch.load` sempre com `weights_only`; sem `trust_remote_code`); e a política de
  atualização de revisão (quem aprova, como registra).
- **Aceite:** arquivo existe com as 3 regras e o canal.
- **Verificação:** `ls SECURITY.md && grep -c 'weights_only\|allowlist' SECURITY.md`

---

## 8. Ordem de execução (waves)

### Wave 1 — Fechar o RCE (P0)
1. **`SEC-01`** — eliminar o fallback sem `weights_only`; exigir torch ≥ 1.13 com erro claro.
   (Exige o `DEVOPS-01` para o piso de versão — faça o piso junto, na mesma mudança.)

> Sem a Wave 1, qualquer `.pt` apontado como voz é execução potencial de código.

### Wave 2 — Cercar o carregamento (P1)
2. **`SEC-02`** — allowlist de diretório + extensão + tamanho para vozes.
3. **`SEC-03`** — `revision` fixada + `checksums.json` dos 5 modelos.
4. **`DEVOPS-01`** — unificar manifests com piso e teto (se ainda não feito na Wave 1).
5. **`BUG-01`** — logar exceção de resolução em vez de `pass`.
6. **`BUG-02`** — falhar com `RuntimeError` em download quebrado (sem cache duplo silencioso).
7. **`TEST-01`** — travar tudo isso (allowlist, fallback, `torch.load`, resolução).

### Wave 3 — Operação (P2)
8. **`DEVOPS-02`** — CI com testes mock (sem baixar modelo) + barreira do `SEC-01`.
9. **`IMP-01`** → **`PERF-01`** — limite do cache, depois normalização de chave.
10. **`SEC-04`** — validação de `reference_audio`.
11. **`DEVOPS-03`** — gitignore de pesos + espaço documentado.
12. **`DOC-01`** — tabela de modelos no README.

### Wave 4 — Garantia (P3)
13. **`IMP-02`** — teste de determinismo + nota de onde a garantia não vale.
14. **`DOC-02`**.

**Dependências que não podem ser invertidas:**
`SEC-01` antes de `SEC-02` (não adianta cercar caminho se o loader ainda aceita pickle) ·
`DEVOPS-01` (piso do torch) junto com `SEC-01` · `SEC-03` antes de `DOC-01` (a tabela precisa das
revisões) · `TEST-01` depois de `BUG-01`/`BUG-02` (o teste precisa do comportamento como alvo).

---

## 9. Fora de escopo / riscos

| Item | Decisão | Motivo |
|---|---|---|
| Reescrever o modelo (`qwen_tts/`) | **Não** | É código do modelo; as correções são de contorno em `nodes.py`. Só mexa dentro com causa provada. |
| Trocar `snapshot_download` por download manual | **Não** | O HF Hub é o canal certo; o defeito é falta de `revision`, não de ferramenta. |
| Habilitar `trust_remote_code=True` | **Nunca** | Regra de supply chain: código executável vindo do Hub sem revisão é RCE por design. |
| Adicionar modo servidor/API REST | **Não** | Fora do escopo. Se um dia existir, reabrir threat model inteiro. |
| Suportar modelo arbitrário do usuário (fora da allowlist) | **Não** | Quebraria o `SEC-03`. Novos modelos entram por PR que atualiza MAP + revisão + checksum. |

**Riscos desta execução:**

- **`SEC-01` pode quebrar usuário com torch antigo.** Exigir ≥ 1.13 (2022) é razoável, mas
  documente no README qual versão instalar — senão o erro claro vira barreira sem saída.
- **`SEC-03` congela os pesos.** Fixar `revision` significa que melhorias do upstream param de
  chegar automaticamente. Defina cadência de revisão (ex.: trimestral, com PR dedicado).
- **`BUG-02` (`RuntimeError` em download falho) muda comportamento.** Hoje o fluxo "parece
  funcionar" caindo para o cache padrão; depois da correção, falha visível. Comunique que falha
  visível **é** o comportamento certo.
- **`IMP-01` (limite de cache) pode recarregar modelo em workflow grande.** Escolha N=2 com
  margem e meça antes de reduzir — evicção agressiva piora o tempo.

---

## 10. Definição de pronto (DoD)

**Segurança**
- [ ] `SEC-01` — nenhum `torch.load` sem `weights_only=True`; torch < 1.13 gera erro claro
- [ ] `SEC-02` — voz fora do diretório/extensão/tamanho é recusada antes do `load`
- [ ] `SEC-03` — 5 modelos com `revision` fixada + `checksums.json` verificado
- [ ] `SEC-04` — `reference_audio` fora do limite é recusado na entrada

**Funcional**
- [ ] `BUG-01` — erro de resolução aparece no log com o caminho
- [ ] `BUG-02` — download falho levanta `RuntimeError` com causa e disco

**Testes e qualidade**
- [ ] `TEST-01` — `pytest` com os 6 casos, todos passando, sem GPU
- [ ] `IMP-01` — cache com limite e liberação; sem crescimento indefinido
- [ ] `PERF-01` — parâmetros equivalentes reutilizam a mesma entrada
- [ ] `IMP-02` — determinismo testado no caminho garantido e documentado

**Infra e documentação**
- [ ] `DEVOPS-01` — manifests unificados, com piso e teto, sem divergência
- [ ] `DEVOPS-02` — CI verde (sintaxe + testes + barreira do `SEC-01`)
- [ ] `DEVOPS-03` — nenhum peso aparece em `git status` após download
- [ ] `DOC-01` — README com tabela de modelos (revisão, licença, espaço)
- [ ] `DOC-02` — `SECURITY.md` com as 3 regras de supply chain

**Validação final:**
```bash
python3 -m py_compile nodes.py && pytest -q
grep -rn 'torch.load' --include='*.py' . | grep -v __pycache__ | grep -v 'weights_only=True' && echo FALHA || echo OK
git status --porcelain | grep -cE '\\.(pt|pth|bin|safetensors|onnx)$'   # 0
```

---

*Fim do plano. Gerado por leitura direta do código em 2026-10-02. Nenhum item já estava corrigido*
*— todos apontam para defeitos ainda presentes.*
