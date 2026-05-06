import json
import re
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


def build_system_prompt(base_prompt: str, vocabulary: list[dict]) -> str:
    """Inject vocabulary corrections into the system prompt."""
    if not vocabulary:
        return base_prompt
    vocab_lines = "\n".join(
        f'- If you hear "{v["spoken"]}", always write "{v["corrected"]}"'
        for v in vocabulary
        if v.get("spoken", "").strip() and v.get("corrected", "").strip()
    )
    if not vocab_lines:
        return base_prompt
    return (
        f"{base_prompt}\n\n"
        f"Important vocabulary corrections — apply these exactly:\n{vocab_lines}"
    )


def apply_replacements(text: str, replacements: list[dict]) -> str:
    for rule in replacements:
        find = rule.get("find", "").strip()
        replace = rule.get("replace", "").strip()
        if find:
            text = re.sub(re.escape(find), replace, text, flags=re.IGNORECASE)
    return text


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    system_prompt: str = Form("Transcribe the audio."),
    replacements: str = Form("[]"),   # JSON [{find, replace}] — post-processing
    vocabulary: str = Form("[]"),     # JSON [{spoken, corrected}] — injected into prompt
):
    replacement_rules = json.loads(replacements)
    vocabulary_rules = json.loads(vocabulary)

    # Build the final prompt with vocabulary hints baked in
    final_prompt = build_system_prompt(system_prompt, vocabulary_rules)

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
                {"type": "text", "text": final_prompt},
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

    print(f"[DEBUG] final_prompt={final_prompt!r}")
    print(f"[DEBUG] replacements={replacement_rules}")

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=500,
        streamer=streamer,
        repetition_penalty=1.3,
        no_repeat_ngram_size=4,
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    def token_stream():
        full_text = ""
        try:
            for token in streamer:
                if token:
                    full_text += token
                    # Send cumulative text so far — client replaces its preview each event
                    yield f"data: {full_text}\n\n"
        finally:
            thread.join()
            Path(webm_path).unlink(missing_ok=True)
            Path(wav_path).unlink(missing_ok=True)

        # Apply vocabulary corrections as post-processing too (prompt injection alone is unreliable on small models)
        vocab_replacements = [
            {"find": v["spoken"], "replace": v["corrected"]}
            for v in vocabulary_rules
            if v.get("spoken", "").strip() and v.get("corrected", "").strip()
        ]
        final_text = apply_replacements(full_text, vocab_replacements + replacement_rules)
        print(f"[DEBUG] raw={full_text!r}")
        print(f"[DEBUG] final={final_text!r}")

        # Send post-processed final result as a tagged event, then done
        yield f"data: [FINAL]{final_text}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")


def main():
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)


if __name__ == "__main__":
    main()
