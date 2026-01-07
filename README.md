# Transcrição de Áudio com WhisperX

Transcreve áudio em português utilizando **WhisperX** (Whisper + alinhamento de palavras + diarização). Identifica falantes, alinha palavras e conta ocorrências de palavras-chave por falante.

## 📋 Recursos

- ✅ **Transcrição de áudio** em português com Whisper (modelos: tiny, base, small, medium, large-v3)
- ✅ **Alinhamento de palavras** em nível de palavra (sabe exatamente quando cada palavra foi dita)
- ✅ **Diarização** (identificação automática de falantes)
- ✅ **Contagem de palavras-chave** por falante
- ✅ **Fallback para CPU** (se GPU falhar)
- ✅ **Timer de progresso** (mostra quanto tempo cada etapa leva)
- ✅ **Saídas estruturadas** (transcrição + estatísticas)

## 🚀 Instalação Rápida

### 1. Clonar repositório
```bash
git clone https://github.com/seu-usuario/transcricao-audio.git
cd transcricao-audio
```

### 2. Criar ambiente virtual (Python 3.11)
```powershell
# Windows (PowerShell)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS/Linux
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependências
```bash
python -m pip install --upgrade pip setuptools wheel
```

**Opção A: CPU (simples, mais lento)**
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**Opção B: GPU CUDA 12.1** (requer NVIDIA + drivers atualizados)
```bash
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

**Opção C: GPU CUDA 11.8**
```bash
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 📖 Uso

### 1. Configurar token Hugging Face (opcional)

Copie `.env.example` para `.env` e configure seu token:

```bash
cp .env.example .env
# Edite .env e cole seu token
```

Ou use variável de ambiente:
```powershell
$env:HUGGINGFACE_TOKEN = "hf_seu_token_aqui"
```

### 2. Customizar palavras-chave

Edite `palavras.json` para adicionar/remover palavras a contar:

```json
{
  "palavras": [
    "mano",
    "aburso",
    "sua_palavra",
    "outra_palavra"
  ]
}
```

Deixe array vazio para desabilitar contagem:
```json
{
  "palavras": []
}
```

### 3. Rodar

```bash
# Básico
python transcricao_audio.py audio.mp3

# Com modelo específico
python transcricao_audio.py audio.mp3 --model small

# Com modelo mais preciso (lento)
python transcricao_audio.py audio.mp3 --model large-v3
```

## 📊 Saída

Dois arquivos são gerados (baseado no nome de entrada):

### 1. `audio.transcricao.txt`
```
Speaker 1: Olá, tudo bem?
Speaker 1: Como você está?
Speaker 2: Tudo ótimo!
```

### 2. `audio.estatisticas.txt`
```
CONTAGEM DE PALAVRAS POR FALANTE
Speaker 1: 245 palavras
Speaker 2: 189 palavras

CONTAGEM DE PALAVRAS ESPECÍFICAS POR FALANTE

Speaker 1:
  'mano': 3 vez(es)
  'aburso': 1 vez(es)

Speaker 2:
  'mano': 2 vez(es)
  'aburso': 0 vez(es)
```

## ⚙️ Configuração

### `.env` - Token Hugging Face

Copie `.env.example` para `.env`:
```bash
cp .env.example .env
```

Edite e configure seu token (pegue em https://huggingface.co/settings/tokens):
```
HUGGINGFACE_TOKEN=hf_seu_token_aqui
```

O token é carregado automaticamente. Para segurança, **nunca** commit `.env` no GitHub (já está em `.gitignore`).

### `palavras.json` - Palavras-chave a contar

Edite para adicionar/remover palavras:
```json
{
  "palavras": [
    "mano",
    "aburso",
    "sua_palavra"
  ]
}
```

Saída mostrará contagem por falante para cada palavra.

### Código - Outros parâmetros

Edite `transcricao_audio.py` para customizar:

```python
# Tamanho de lote para transcrição
BATCH_SIZE = 32  # aumente para ~64 se tiver muita VRAM

# Modelo padrão
DEFAULT_MODEL = "small"  # tiny, base, small, medium, large-v3
```

## 🎯 Performance

### Tempos estimados (áudio de 1 hora, GPU Tesla T4):

| Modelo | Carga | Transcrição | Alinhamento | Diarização | **Total** |
|--------|-------|-------------|-------------|-----------|----------|
| tiny   | 1m    | 2m          | 1m          | 2m        | **6m**   |
| base   | 1m    | 4m          | 1m          | 2m        | **8m**   |
| small  | 1m    | 8m          | 2m          | 2m        | **13m**  |
| medium | 2m    | 15m         | 3m          | 2m        | **22m**  |
| large  | 3m    | 25m         | 4m          | 2m        | **34m**  |

**CPU é ~10-15x mais lento** que GPU.

### 💡 Dicas de otimização:
- Use modelo `small` ou `tiny` para audios longos
- Aumente `BATCH_SIZE` para 64 se tiver 8GB+ VRAM GPU
- Reduza `BATCH_SIZE` para 16 se tiver pouca memória
- Diarização é lenta; desative se não precisar (não configure token)

## 🔧 Requisitos

- **Python 3.11+**
- **PyTorch 2.1+** (CPU ou GPU)
- **FFmpeg** (para carregar áudio)
- **Hugging Face Token** (opcional, para diarização)

### Instalar FFmpeg
```bash
# Windows (se não tiver)
winget install -e --id Gyan.FFmpeg

# macOS
brew install ffmpeg

# Linux (Ubuntu/Debian)
sudo apt-get install ffmpeg
```

### Instalar dependências Python
```bash
pip install -r requirements.txt
# Instale também pytorch (CPU ou GPU)
# Ver seção "Instalação Rápida" acima
```

## 🐛 Troubleshooting

### `ModuleNotFoundError: No module named 'whisperx'`
→ Ative o venv: `.\.venv\Scripts\Activate.ps1`

### `OSError: [WinError 127] Não foi possível encontrar o procedimento`
→ Pytorch incompatível com GPU. O script tenta CPU automaticamente. Se quiser forçar CPU:
```python
device = "cpu"  # Edite a linha 37
```

### Diarização desativada / token inválido
→ Configure token Hugging Face:
```powershell
$env:HUGGINGFACE_TOKEN = "hf_seu_token"
```
Pegue seu token em https://huggingface.co/settings/tokens

### Áudio muito longo (demora muitas horas)
→ Use modelo mais rápido:
```bash
python transcricao_audio.py audio.mp3 --model tiny
```

## 📝 Exemplo Completo

```powershell
# 1. Ativar ambiente
.\.venv\Scripts\Activate.ps1

# 2. Configurar arquivo .env (copiar template)
Copy-Item .env.example .env
# Editar .env e colar token Hugging Face

# 3. Customizar palavras (editar palavras.json se quiser)
# (opcional - já vem com "mano" e "aburso")

# 4. Rodar
python transcricao_audio.py meu_video.mp3 --model small

# Saída:
# ============================================================
# Dispositivo: cuda (compute_type=float16)
# Modelo: small
# Batch size: 32
# ============================================================
#
# ✓ 2 palavra(s)-chave carregada(s) de palavras.json
# [1/5] Carregando áudio...
# ✓ Áudio carregado em 0:00:05
# [2/5] Carregando modelo Whisper (small)...
# ✓ Modelo carregado em 0:01:30
# [3/5] Transcrevendo áudio...
# ✓ Transcrição concluída em 0:15:45
# [4/5] Alinhando palavras...
# ✓ Alinhamento concluído em 0:02:10
# [5/5] Identificando falantes (diarização)...
# ✓ Diarização concluída em 0:03:20
#
# ============================================================
# ✓ Processamento concluído em 0:22:50
# TRANSCRIÇÃO: meu_video.transcricao.txt
# ESTATÍSTICAS: meu_video.estatisticas.txt
# ============================================================
```

## 📜 Licença

MIT License - Veja `LICENSE` para detalhes.

## 🤝 Contribuindo

Melhorias e sugestões são bem-vindas! Abra uma issue ou pull request.

---

**Nota:** O primeiro uso baixa modelos (~3-5 GB para `small`, ~10 GB para `large`). Isso é feito uma única vez.
