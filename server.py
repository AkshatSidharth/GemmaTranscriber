import subprocess
import tempfile
import threading
from pathlib import Path

import librosa
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_ID = "google/gemma-4-E2B-it"
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, device_map="auto")

STATIC_DIR = Path(__file__).parent

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Gemma 4 Transcription Tester</title>
  <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { background: #0f0f13; }
  </style>
</head>
<body>
  <div id="root"></div>
  <script type="text/babel" data-type="module">
    const { useState, useRef, useCallback, useEffect } = React;

    __JSX_CODE__

    ReactDOM.createRoot(document.getElementById("root")).render(
      React.createElement(Gemma4TranscribeTester)
    );
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    jsx_code = (STATIC_DIR / "gemma4-transcribe-tester.jsx").read_text()
    jsx_code = jsx_code.replace(
        'import { useState, useRef, useCallback, useEffect } from "react";', ""
    )
    jsx_code = jsx_code.replace("export default function", "function")
    return HTML_TEMPLATE.replace("__JSX_CODE__", jsx_code)


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    system_prompt: str = Form("Transcribe the audio."),
):
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        tmp.write(await audio.read())
        webm_path = tmp.name

    wav_path = webm_path.replace(".webm", ".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", webm_path, "-ar", "16000", "-ac", "1", wav_path],
        check=True,
        capture_output=True,
    )

    audio_array, sr = librosa.load(wav_path, sr=16000, mono=True)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_array},
                {"type": "text", "text": system_prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_tensors="pt", return_dict=True,
        add_generation_prompt=True,
    ).to(model.device)

    streamer = TextIteratorStreamer(
        processor.tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    print(f"[DEBUG] system_prompt={system_prompt!r}")
    print(f"[DEBUG] Input keys: {list(inputs.keys())}")
    for k, v in inputs.items():
        if hasattr(v, "shape"):
            print(f"[DEBUG] {k}: shape={v.shape}, dtype={v.dtype}")

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=500,
        streamer=streamer,
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    def token_stream():
        try:
            for token in streamer:
                if token:
                    print(f"[DEBUG] Streamed token: '{token}'")
                    yield f"data: {token}\n\n"
        finally:
            thread.join()
            Path(webm_path).unlink(missing_ok=True)
            Path(wav_path).unlink(missing_ok=True)
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")


def main():
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)


if __name__ == "__main__":
    main()
