"""Transcrição de áudio com WhisperX, alinhamento e diarização (identificação de falantes).

Utiliza Whisper-X (faster-whisper + alignment + pyannote) para gerar transcrições,
alinhamento em nível de palavra e identificação de falantes. Conta palavras-chave por falante.

Uso:
    python transcricao_audio.py <arquivo_audio> [--model {tiny|base|small|medium|large}]
    python transcricao_audio.py audio.mp3
    python transcricao_audio.py audio.mp3 --model small

Saída:
    - arquivo.transcricao.txt: transcrição com falantes
    - arquivo.estatisticas.txt: contagem de palavras

Requisitos:
    - Python 3.11+
    - PyTorch (GPU ou CPU)
    - whisperx, torch, torchaudio
    - Token Hugging Face (para diarização)
"""

import os
import json
import whisperx
import torch
import re
import time
import argparse
from collections import defaultdict
from datetime import timedelta
from dotenv import load_dotenv
import gc
import sys
from pathlib import Path

# ------------ CONFIGURAÇÕES PADRÃO ------------
# Carrega variáveis de ambiente (.env)
load_dotenv()

# Modelos disponíveis (tamanho crescente = qualidade crescente, tempo crescente)
MODELS = {"tiny": "tiny", "base": "base", "small": "small", "medium": "medium", "large-v3": "large-v3"}
DEFAULT_MODEL = "small"  # rápido e com boa qualidade

# Token Hugging Face (carrega de variável de ambiente ou .env)
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Tamanho de lote para transcrição (aumentado para mais velocidade)
BATCH_SIZE = 32
# -----------------------------------------------


def load_target_words(filepath: str = "palavras.json") -> list:
    """Carrega palavras-chave de arquivo JSON externo.
    
    Args:
        filepath: Caminho do arquivo JSON com palavras.
    
    Returns:
        Lista de palavras ou [] se arquivo não existir.
    """
    path = Path(filepath)
    if not path.exists():
        print(f"⚠️  Arquivo {filepath} não encontrado. Contagem de palavras desativada.")
        return []
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            words = data.get("palavras", [])
            if words:
                print(f"✓ {len(words)} palavra(s)-chave carregada(s) de {filepath}")
            return words
    except json.JSONDecodeError as e:
        print(f"⚠️  Erro ao ler {filepath}: {e}. Contagem de palavras desativada.")
        return []


def normalize(text: str) -> str:
    """Normaliza texto para comparação de palavras.
    
    Remove acentos e caracteres especiais, converte para minúsculas.
    
    Args:
        text: Texto a normalizar.
    
    Returns:
        Texto normalizado.
    """
    return re.sub(r"\W+", " ", text.lower()).strip()


def format_time(seconds: float) -> str:
    """Formata tempo em segundos para HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))



def process_audio(
    audio_file: str,
    whisper_model: str = DEFAULT_MODEL,
    hf_token: str = None,
    target_words: list = None,
    device: str = None,
    batch_size: int = BATCH_SIZE
) -> dict:
    """Função principal de processamento para ser usada por API ou CLI.
    
    Returns:
        Dict com chaves: 'transcription' (lista de segmentos), 'stats' (textos), 'files' (caminhos)
    """
    if target_words is None:
        target_words = []
        
    start_time = time.time()
    
    # Detecção de dispositivo
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    compute_type = "float16" if device == "cuda" else "float32"
    print(f"Dispositivo: {device} (compute_type={compute_type})")

    audio_path = Path(audio_file)
    if not audio_path.exists():
        raise FileNotFoundError(f"Arquivo {audio_path} não encontrado.")

    # 1. Carrega áudio
    print(f"[1/5] Carregando áudio...")
    audio = whisperx.load_audio(str(audio_path))

    # 2. Carrega modelo
    print(f"[2/5] Carregando modelo Whisper ({whisper_model})...")
    try:
        model = whisperx.load_model(whisper_model, device, compute_type=compute_type)
    except OSError as exc:
        if device == "cuda":
            print(f"⚠️  Falha na GPU ({exc}); tentando CPU...")
            device = "cpu"
            compute_type = "float32"
            model = whisperx.load_model(whisper_model, device, compute_type=compute_type)
        else:
            raise exc

    # 3. Transcreve
    print(f"[3/5] Transcrevendo áudio...")
    result = model.transcribe(audio, batch_size=batch_size, language="pt")

    # 4. Alinhamento
    print(f"[4/5] Alinhando palavras...")
    try:
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
    except Exception as e:
        print(f"⚠️  Erro no alinhamento: {e}")

    # 5. Diarização
    print(f"[5/5] Identificando falantes (diarização)...")
    if hf_token:
        try:
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
        except Exception as exc:
            print(f"⚠️  Erro na diarização: {exc}")
    else:
        print("⚠️  Diarização ignorada (sem token).")

    # Processar resultados
    word_count = defaultdict(int)
    specific_word_count = defaultdict(lambda: defaultdict(int))
    transcript_lines = []

    for segment in result["segments"]:
        speaker = segment.get("speaker", "Unknown")
        text = segment["text"].strip()
        transcript_lines.append({"speaker": speaker, "text": text, "start": segment["start"], "end": segment["end"]})

        # contagem
        words = re.findall(r"\w+", text, flags=re.UNICODE)
        word_count[speaker] += len(words)

        norm_text = normalize(text)
        for w in target_words:
            pattern = r"\b" + re.escape(w.lower()) + r"\b"
            specific_word_count[speaker][w] += len(re.findall(pattern, norm_text))

    # Gera estatísticas formatadas
    stats_text = "CONTAGEM DE PALAVRAS POR FALANTE\n"
    for speaker, count in word_count.items():
        stats_text += f"{speaker}: {count} palavras\n"

    if target_words:
        stats_text += "\nCONTAGEM DE PALAVRAS ESPECÍFICAS POR FALANTE\n"
        for speaker, words_dict in specific_word_count.items():
            stats_text += f"\n{speaker}:\n"
            for w, c in words_dict.items():
                stats_text += f"  '{w}': {c} vez(es)\n"
    
    # Limpeza
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "transcription": transcript_lines,
        "stats": stats_text,
        "word_count": dict(word_count),
        "specific_word_count": {k: dict(v) for k, v in specific_word_count.items()}
    }


def main(audio_file: str, whisper_model: str = DEFAULT_MODEL):
    """Entrypoint CLI."""
    # Carrega configs
    target_words = load_target_words("palavras.json")
    
    print(f"\n{'='*60}")
    print(f"Iniciando processamento: {audio_file}")
    print(f"{'='*60}\n")
    
    try:
        # Processa
        results = process_audio(
            audio_file=audio_file,
            whisper_model=whisper_model,
            hf_token=HF_TOKEN,
            target_words=target_words,
            batch_size=BATCH_SIZE
        )
        
        # Salva arquivos
        audio_path = Path(audio_file)
        transcript_file = audio_path.with_suffix(".transcricao.txt")
        stats_file = audio_path.with_suffix(".estatisticas.txt")
        
        with transcript_file.open("w", encoding="utf-8") as f:
            for line in results["transcription"]:
                f.write(f"{line['speaker']}: {line['text']}\n")
                
        with stats_file.open("w", encoding="utf-8") as f:
            f.write(results["stats"])
            
        print(f"\n{'='*60}")
        print(f"✓ Concluído!")
        print(f"Arquivo Transcrição: {transcript_file}")
        print(f"Arquivo Estatísticas: {stats_file}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"❌ Erro fatal: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcreve, alinha e identifica falantes em arquivo de áudio.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("audio", help="Caminho do arquivo de áudio")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=list(MODELS.keys()),
        help=f"Modelo Whisper (padrão: {DEFAULT_MODEL})"
    )
    
    args = parser.parse_args()
    main(args.audio, args.model)

