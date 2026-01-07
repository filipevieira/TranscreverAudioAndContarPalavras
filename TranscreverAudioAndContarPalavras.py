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


def main(audio_file: str, whisper_model: str = DEFAULT_MODEL):
    """Processa arquivo de áudio: transcrição, alinhamento e diarização.
    
    Args:
        audio_file: Caminho do arquivo de áudio.
        whisper_model: Modelo Whisper a usar ('tiny', 'base', 'small', 'medium', 'large-v3').
    """
    # Carrega palavras-chave do arquivo externo
    target_words = load_target_words("palavras.json")
    
    start_time = time.time()
    
    # Dispositivo (GPU se tiver)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"
    print(f"\n{'='*60}")
    print(f"Dispositivo: {device} (compute_type={compute_type})")
    print(f"Modelo: {whisper_model}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"{'='*60}\n")

    audio_path = Path(audio_file)
    if not audio_path.exists():
        print(f"❌ ERRO: arquivo {audio_path} não encontrado.")
        return

    # Carrega áudio
    print(f"[1/5] Carregando áudio...")
    step_start = time.time()
    audio = whisperx.load_audio(str(audio_path))
    print(f"✓ Áudio carregado em {format_time(time.time() - step_start)}")

    # Modelo WhisperX (tenta GPU, cai para CPU se falhar)
    print(f"[2/5] Carregando modelo Whisper ({whisper_model})...")
    step_start = time.time()
    try:
        model = whisperx.load_model(whisper_model, device, compute_type=compute_type)
    except OSError as exc:
        if device == "cuda":
            print(f"⚠️  Falha na GPU ({exc}); tentando CPU...")
            device = "cpu"
            compute_type = "float32"
            try:
                model = whisperx.load_model(whisper_model, device, compute_type=compute_type)
            except Exception as exc2:
                print(f"❌ ERRO ao carregar modelo na CPU: {exc2}")
                return
        else:
            print(f"❌ ERRO ao carregar modelo: {exc}")
            return

    if device == "cpu":
        print("⚠️  Executando em CPU (mais lento; GPU recomendada para audios longos).")

    print(f"✓ Modelo carregado em {format_time(time.time() - step_start)}")
    
    print(f"[3/5] Transcrevendo áudio...")
    step_start = time.time()
    result = model.transcribe(audio, batch_size=BATCH_SIZE, language="pt")
    print(f"✓ Transcrição concluída em {format_time(time.time() - step_start)}")

    # Alinhamento
    print(f"[4/5] Alinhando palavras...")
    step_start = time.time()
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
    print(f"✓ Alinhamento concluído em {format_time(time.time() - step_start)}")

    # Diarização (falantes)
    print(f"[5/5] Identificando falantes (diarização)...")
    step_start = time.time()
    diarize_model = None
    if HF_TOKEN:
        try:
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
        except Exception as exc:
            print(f"⚠️  Diarização desativada (token inválido/sem acesso)")

    if diarize_model:
        try:
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            print(f"✓ Diarização concluída em {format_time(time.time() - step_start)}")
        except Exception as exc:
            print(f"⚠️  Erro na diarização: {exc}")
    else:
        print("⚠️  Diarização não executada (token não configurado).")

    # Contagens
    word_count = defaultdict(int)  # total de palavras por falante
    specific_word_count = defaultdict(lambda: defaultdict(int))  # falante -> palavra -> qtd

    # Arquivos de saída
    transcript_file = audio_path.with_suffix(".transcricao.txt")
    stats_file = audio_path.with_suffix(".estatisticas.txt")

    with transcript_file.open("w", encoding="utf-8") as f:
        for segment in result["segments"]:
            speaker = segment.get("speaker", "Unknown")
            text = segment["text"].strip()
            f.write(f"{speaker}: {text}\n")

            # contagem total de palavras
            words = re.findall(r"\w+", text, flags=re.UNICODE)
            word_count[speaker] += len(words)

            # contagem de palavras específicas
            norm_text = normalize(text)
            for w in target_words:
                pattern = r"\b" + re.escape(w.lower()) + r"\b"
                specific_word_count[speaker][w] += len(re.findall(pattern, norm_text))

    # Salva estatísticas
    with stats_file.open("w", encoding="utf-8") as f:
        f.write("CONTAGEM DE PALAVRAS POR FALANTE\n")
        for speaker, count in word_count.items():
            f.write(f"{speaker}: {count} palavras\n")

        if target_words:
            f.write("\nCONTAGEM DE PALAVRAS ESPECÍFICAS POR FALANTE\n")
            for speaker, words_dict in specific_word_count.items():
                f.write(f"\n{speaker}:\n")
                for w, c in words_dict.items():
                    f.write(f"  '{w}': {c} vez(es)\n")
        else:
            f.write("\n(Nenhuma palavra-chave configurada em palavras.json)\n")

    # limpar GPU
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✓ Processamento concluído em {format_time(total_time)}")
    print(f"TRANSCRIÇÃO: {transcript_file}")
    print(f"ESTATÍSTICAS: {stats_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcreve, alinha e identifica falantes em arquivo de áudio.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Exemplos:
  python transcricao_audio.py audio.mp3
  python transcricao_audio.py audio.wav --model small
  python transcricao_audio.py audio.mp3 --model large-v3

Modelos disponíveis (do mais rápido ao mais preciso):
  tiny    → muito rápido, menos preciso
  base    → rápido, boa qualidade
  small   → padrão, equilibrado (recomendado)
  medium  → lento, melhor qualidade
  large-v3→ muito lento, melhor qualidade

Token Hugging Face:
  Configure via variável de ambiente: $env:HUGGINGFACE_TOKEN = "seu_token"
        """)
    parser.add_argument("audio", help="Caminho do arquivo de áudio (mp3, wav, m4a, etc.)")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=list(MODELS.keys()),
        help=f"Modelo Whisper a usar (padrão: {DEFAULT_MODEL})"
    )
    
    args = parser.parse_args()
    main(args.audio, args.model)
