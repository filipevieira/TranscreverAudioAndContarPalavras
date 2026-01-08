# Changelog

## [1.0.0] - 2026-01-07

### ✨ Adicionado

- Transcrição de áudio em português com **WhisperX**
- Suporte para múltiplos modelos Whisper (tiny, base, small, medium, large-v3)
- **Alinhamento de palavras** em nível granular
- **Diarização automática** para identificação de falantes
- Contagem de palavras-chave por falante
- Fallback automático para CPU se GPU não estiver disponível
- Timer de progresso para monitorar tempo de cada etapa
- Suporte para arquivos de áudio em **WAV, MP3 e outros formatos comuns**
- Argumentos de linha de comando para seleção de modelo
- Carregamento de token Hugging Face via variável de ambiente

### 📁 Saídas geradas

- `arquivo.transcricao.txt` - Transcrição completa com identificação de falantes
- `arquivo.estatisticas.txt` - Contagem de palavras por falante

### 🔧 Dependências principais

- **whisperx** (>=0.10.1) - Transcrição e alinhamento
- **torch** (2.1.2) - Framework de ML
- **torchaudio** (2.1.2) - Processamento de áudio
- **python-dotenv** (>=1.0.0) - Gerenciamento de variáveis de ambiente

### 📝 Configuração

- Modelo padrão: `small` (rápido e boa qualidade)
- Batch size: 32 palavras
- Requer token Hugging Face para diarização

### ⚠️ Limitações conhecidas

- Tratamento de erros ainda básico (melhorias em versões futuras)
- Pensado para uso local em linha de comando
- Requer Python 3.11+

### 🎯 Como usar

```bash
# Uso básico
python TranscreverAudioAndContarPalavras.py seu_audio.mp3

# Com modelo específico
python TranscreverAudioAndContarPalavras.py seu_audio.wav --model large-v3
```
