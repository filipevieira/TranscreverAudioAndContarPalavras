document.addEventListener('DOMContentLoaded', () => {
    // --- Elements ---
    const form = document.getElementById('transcribeForm');
    const dropArea = document.getElementById('dropArea');
    const fileInput = document.getElementById('audioFile');
    const fileNameDisplay = document.getElementById('fileName');
    const submitBtn = document.getElementById('submitBtn');
    const btnText = submitBtn.querySelector('.btn-text');
    const spinner = submitBtn.querySelector('.loading-spinner');
    
    // States
    const emptyState = document.getElementById('emptyState');
    const loadingState = document.getElementById('loadingState');
    const resultsContent = document.getElementById('resultsContent');
    const timerDisplay = document.getElementById('timer');
    
    // Tabs
    const tabs = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    // Results
    const transcriptionList = document.getElementById('transcriptionList');
    const statsContent = document.getElementById('statsContent');

    let processingTimer = null;
    let secondsElapsed = 0;

    // --- File Drag & Drop ---
    dropArea.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            updateFileName(fileInput.files[0].name);
        }
    });

    dropArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropArea.classList.add('dragover');
    });

    dropArea.addEventListener('dragleave', () => {
        dropArea.classList.remove('dragover');
    });

    dropArea.addEventListener('drop', (e) => {
        e.preventDefault();
        dropArea.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            updateFileName(fileInput.files[0].name);
        }
    });

    function updateFileName(name) {
        fileNameDisplay.textContent = name;
        // Basic validation
        if(!name.match(/\.(mp3|wav|m4a|mp4)$/i)){
            alert('Por favor, selecione um arquivo de áudio ou vídeo.');
        }
    }

    // --- Tabs Logic ---
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            tab.classList.add('active');
            const tabId = tab.dataset.tab;
            document.getElementById(tabId + 'Tab').classList.add('active');
        });
    });

    // --- Form Submission ---
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (fileInput.files.length === 0) {
            alert("Por favor, selecione um arquivo de áudio.");
            return;
        }

        // Set Loading State
        setLoading(true);
        startTimer();

        const formData = new FormData(form);

        try {
            const response = await fetch('/transcribe', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Falha na requisição');

            const result = await response.json();
            
            if (result.status === 'success') {
                renderResults(result.data);
            } else {
                alert('Erro: ' + (result.message || 'Erro desconhecido'));
                setLoading(false);
            }

        } catch (error) {
            console.error(error);
            alert('Erro ao conectar com o servidor. Verifique se ele está rodando.');
            setLoading(false);
        } finally {
            stopTimer();
        }
    });

    // --- Rendering ---
    function renderResults(data) {
        setLoading(false);
        resultsContent.classList.remove('hidden');
        emptyState.classList.add('hidden');
        
        // 1. Transcription
        transcriptionList.innerHTML = '';
        const speakers = new Set();
        
        data.transcription.forEach(item => {
            speakers.add(item.speaker);
            
            const bubble = document.createElement('div');
            bubble.className = 'chat-bubble';
            
            // Generate avatar initials/color based on speaker
            const isAlt = Array.from(speakers).indexOf(item.speaker) % 2 !== 0; // alternate colors
            const initials = item.speaker.slice(0,2).toUpperCase();
            
            bubble.innerHTML = `
                <div class="speaker-avatar ${isAlt ? 'alt' : ''}">${initials}</div>
                <div class="message-content">
                    <div class="speaker-name">${item.speaker}</div>
                    <div class="text">${item.text}</div>
                </div>
            `;
            transcriptionList.appendChild(bubble);
        });

        // 2. Stats
        statsContent.textContent = data.stats;
    }

    // --- Utilities ---
    function setLoading(isLoading) {
        if (isLoading) {
            submitBtn.disabled = true;
            btnText.textContent = 'Processando...';
            // Show spinner via CSS if needed, or stick to simple text
            loadingState.classList.remove('hidden');
            emptyState.classList.add('hidden');
            resultsContent.classList.add('hidden');
        } else {
            submitBtn.disabled = false;
            btnText.textContent = 'Iniciar Transcrição';
            loadingState.classList.add('hidden');
            // Do not unhide resultsContent here, renderResults does that
        }
    }

    function startTimer() {
        secondsElapsed = 0;
        timerDisplay.textContent = "00:00";
        if (processingTimer) clearInterval(processingTimer);
        
        processingTimer = setInterval(() => {
            secondsElapsed++;
            const mins = Math.floor(secondsElapsed / 60).toString().padStart(2, '0');
            const secs = (secondsElapsed % 60).toString().padStart(2, '0');
            timerDisplay.textContent = `${mins}:${secs}`;
        }, 1000);
    }

    function stopTimer() {
        clearInterval(processingTimer);
    }
});

function copyTranscription() {
    const texts = [];
    document.querySelectorAll('.chat-bubble').forEach(bubble => {
        const speaker = bubble.querySelector('.speaker-name').innerText;
        const text = bubble.querySelector('.text').innerText;
        texts.push(`${speaker}: ${text}`);
    });
    
    navigator.clipboard.writeText(texts.join('\n')).then(() => {
        alert('Copiado para a área de transferência!');
    });
}
