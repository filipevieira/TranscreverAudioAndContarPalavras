import shutil
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
import json
from typing import List

# Importa a lógica do nosso script principal
# (Assumindo que o arquivo se chama TranscreverAudioAndContarPalavras.py)
from TranscreverAudioAndContarPalavras import process_audio

app = FastAPI()

# Configuração CORS (caso necessário)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Diretório para salvar uploads temporários
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Servir arquivos estáticos (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return JSONResponse(content={"message": "Server running. Go to /static/index.html"})

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("small"),
    token: str = Form(None),
    keywords: str = Form(None)
):
    """
    Endpoint para transcrever áudio.
    Recebe arquivo, modelo, token HF e palavras-chave (string separada por vírgula).
    """
    try:
        # Salva o arquivo temporariamente
        file_location = UPLOAD_DIR / file.filename
        with file_location.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"Arquivo recebido: {file.filename}")
        print(f"Modelo: {model}")
        
        # Processa palavras-chave
        target_words_list = []
        if keywords:
            # Separa por vírgula e limpa espaços
            target_words_list = [w.strip() for w in keywords.split(",") if w.strip()]
            
        print(f"Palavras-chave: {target_words_list}")

        # Chama a função de processamento
        # (Isso vai bloquear o thread, mas para uso local único é aceitável.
        # Em produção, usaria BackgroundTasks ou fila async)
        result = process_audio(
            audio_file=str(file_location),
            whisper_model=model,
            hf_token=token if token else None,
            target_words=target_words_list
        )
        
        # Opcional: remover o arquivo após uso
        # file_location.unlink()
        
        return JSONResponse(content={
            "status": "success",
            "data": result
        })

    except Exception as e:
        print(f"Erro no servidor: {e}")
        return JSONResponse(content={
            "status": "error",
            "message": str(e)
        }, status_code=500)

if __name__ == "__main__":
    import uvicorn
    # Abre o navegador automaticamente
    import webbrowser
    webbrowser.open("http://127.0.0.1:8000/static/index.html")
    
    print("Iniciando servidor...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
