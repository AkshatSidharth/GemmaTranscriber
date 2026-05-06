import { useState, useRef, useCallback, useEffect } from "react";

// ─── constants ────────────────────────────────────────────────────────────────

const VISUALIZER_BARS = 32;
const DEFAULT_ENDPOINT = "/transcribe";

const PRESET_PROMPTS = [
  { label: "Basic", prompt: "Transcribe the audio." },
  {
    label: "Punctuated",
    prompt:
      "Transcribe the audio with accurate punctuation, capitalization, and paragraph breaks.",
  },
  {
    label: "Verbatim",
    prompt:
      "Transcribe every word exactly as spoken, including filler words like 'um', 'uh', and false starts.",
  },
  {
    label: "Medical",
    prompt:
      "Transcribe this medical audio accurately. Preserve all medical terminology, drug names, and dosages exactly as spoken.",
  },
  {
    label: "Speakers",
    prompt:
      "Transcribe the audio and label each different speaker as Speaker 1, Speaker 2, etc. Start a new line for each speaker turn.",
  },
  {
    label: "Summary",
    prompt:
      "Transcribe the audio, then on a new line write 'Summary:' followed by a one-sentence summary of what was said.",
  },
  {
    label: "Translate EN",
    prompt: "Transcribe the audio and translate it into English.",
  },
  {
    label: "Timestamps",
    prompt:
      "Transcribe the audio and add approximate timestamps in [mm:ss] format at the start of each sentence.",
  },
];

// ─── helpers ──────────────────────────────────────────────────────────────────

function computeWER(reference, hypothesis) {
  const r = reference.trim().toLowerCase().split(/\s+/);
  const h = hypothesis.trim().toLowerCase().split(/\s+/);
  const m = r.length;
  const n = h.length;
  const dp = Array.from({ length: m + 1 }, (_, i) =>
    Array.from({ length: n + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0))
  );
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] =
        r[i - 1] === h[j - 1]
          ? dp[i - 1][j - 1]
          : 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
    }
  }
  return m === 0 ? 0 : dp[m][n] / m;
}

async function streamTranscription(audioBlob, endpoint, systemPrompt, replacements, vocabulary, onToken, signal) {
  const formData = new FormData();
  formData.append("audio", audioBlob, "recording.webm");
  formData.append("system_prompt", systemPrompt);
  formData.append("replacements", JSON.stringify(replacements));
  formData.append("vocabulary", JSON.stringify(vocabulary));

  if (!endpoint || !endpoint.trim()) {
    // Demo mode
    const demos = [
      "This week I traveled to Chicago to deliver my farewell address to the nation.",
      "Gemma 4 supports native audio understanding for speech-to-text transcription.",
      "The patient presents with acute onset chest pain radiating to the left arm.",
      "Speaker 1: Good morning everyone. Speaker 2: Morning! Ready to start?",
    ];
    const text = demos[Math.floor(Math.random() * demos.length)];
    const words = text.split(" ");
    let partial = "";
    for (const word of words) {
      if (signal && signal.aborted) return null;
      partial += (partial ? " " : "") + word;
      if (onToken) onToken(partial);
      await new Promise((r) => setTimeout(r, 55));
    }
    return text;
  }

  const res = await fetch(endpoint, { method: "POST", body: formData, signal });
  if (!res.ok) throw new Error(`Server error: ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let full = "";
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const data = line.slice(6);
        if (data === "[DONE]") return full;
        full = data;
        if (onToken) onToken(full);
      }
    }
  }
  return full;
}

function formatDuration(secs) {
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

// ─── WaveformVisualizer ───────────────────────────────────────────────────────

function WaveformVisualizer({ analyser, isRecording }) {
  const canvasRef = useRef(null);
  const animRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;

    const draw = () => {
      ctx.clearRect(0, 0, W, H);
      if (analyser && isRecording) {
        const data = new Uint8Array(analyser.frequencyBinCount);
        analyser.getByteFrequencyData(data);
        const step = Math.floor(data.length / VISUALIZER_BARS);
        const barW = W / VISUALIZER_BARS - 2;
        for (let i = 0; i < VISUALIZER_BARS; i++) {
          const v = data[i * step] / 255;
          const h = Math.max(4, v * H * 0.85);
          const x = i * (barW + 2) + 1;
          const y = (H - h) / 2;
          const g = ctx.createLinearGradient(x, y, x, y + h);
          g.addColorStop(0, `rgba(251,146,60,${0.6 + v * 0.4})`);
          g.addColorStop(1, `rgba(217,70,239,${0.5 + v * 0.5})`);
          ctx.beginPath();
          ctx.roundRect(x, y, barW, h, 3);
          ctx.fillStyle = g;
          ctx.fill();
        }
      } else {
        const t = Date.now() / 1200;
        const barW = W / VISUALIZER_BARS - 2;
        for (let i = 0; i < VISUALIZER_BARS; i++) {
          const v = 0.15 + 0.1 * Math.sin(t + i * 0.35);
          const h = Math.max(4, v * H * 0.5);
          const x = i * (barW + 2) + 1;
          const y = (H - h) / 2;
          ctx.beginPath();
          ctx.roundRect(x, y, barW, h, 3);
          ctx.fillStyle = `rgba(148,163,184,${0.2 + 0.1 * Math.sin(t + i * 0.5)})`;
          ctx.fill();
        }
      }
      animRef.current = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(animRef.current);
  }, [analyser, isRecording]);

  return (
    <canvas
      ref={canvasRef}
      width={440}
      height={80}
      style={{ width: "100%", maxWidth: 440, height: 80 }}
    />
  );
}

// ─── RunCard ─────────────────────────────────────────────────────────────────

function RunCard({ run, index, total, referenceText, onPin, isPinned }) {
  const wer = referenceText.trim() ? computeWER(referenceText, run.transcription) : null;

  const werColor =
    wer === null
      ? "rgba(148,163,184,0.5)"
      : wer < 0.1
      ? "rgb(74,222,128)"
      : wer < 0.25
      ? "rgb(250,204,21)"
      : "rgb(248,113,113)";

  return (
    <div
      style={{
        background: isPinned ? "rgba(251,146,60,0.06)" : "rgba(25,25,32,0.8)",
        border: `1px solid ${isPinned ? "rgba(251,146,60,0.35)" : "rgba(255,255,255,0.06)"}`,
        borderRadius: 14,
        padding: "18px 20px",
        marginBottom: 12,
        animation: "fadeSlideIn 0.4s ease both",
      }}
    >
      {/* header row */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 10, gap: 8 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
            <span style={{ fontFamily: "JetBrains Mono,monospace", fontSize: 10, color: "rgba(251,146,60,0.7)", letterSpacing: 1.5, textTransform: "uppercase" }}>
              Run #{total - index}
            </span>
            {wer !== null && (
              <span
                style={{
                  fontFamily: "JetBrains Mono,monospace",
                  fontSize: 10,
                  color: werColor,
                  border: `1px solid ${werColor}`,
                  borderRadius: 4,
                  padding: "1px 7px",
                  letterSpacing: 0.5,
                }}
              >
                WER {(wer * 100).toFixed(1)}%
              </span>
            )}
          </div>
          <div
            style={{
              fontFamily: "JetBrains Mono,monospace",
              fontSize: 11,
              color: "rgba(148,163,184,0.6)",
              whiteSpace: "nowrap",
              overflow: "hidden",
              textOverflow: "ellipsis",
            }}
          >
            prompt: {run.systemPrompt}
          </div>
        </div>
        <div style={{ display: "flex", gap: 6, flexShrink: 0 }}>
          <button
            onClick={() => onPin(run.id)}
            title={isPinned ? "Unpin" : "Pin for comparison"}
            style={{
              background: isPinned ? "rgba(251,146,60,0.2)" : "rgba(255,255,255,0.04)",
              border: `1px solid ${isPinned ? "rgba(251,146,60,0.4)" : "rgba(255,255,255,0.08)"}`,
              borderRadius: 6,
              color: isPinned ? "rgb(251,146,60)" : "rgba(148,163,184,0.5)",
              padding: "3px 10px",
              fontSize: 11,
              cursor: "pointer",
              fontFamily: "JetBrains Mono,monospace",
            }}
          >
            {isPinned ? "★ Pinned" : "☆ Pin"}
          </button>
          <button
            onClick={() => navigator.clipboard?.writeText(run.transcription)}
            style={{
              background: "rgba(255,255,255,0.04)",
              border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 6,
              color: "rgba(148,163,184,0.6)",
              padding: "3px 10px",
              fontSize: 11,
              cursor: "pointer",
              fontFamily: "JetBrains Mono,monospace",
            }}
          >
            Copy
          </button>
        </div>
      </div>

      <p
        style={{
          margin: 0,
          fontFamily: "Source Serif 4,Georgia,serif",
          fontSize: 15,
          lineHeight: 1.7,
          color: "rgba(237,233,228,0.9)",
          wordBreak: "break-word",
        }}
      >
        {run.transcription}
      </p>

      <div
        style={{
          marginTop: 10,
          fontFamily: "JetBrains Mono,monospace",
          fontSize: 10,
          color: "rgba(148,163,184,0.35)",
          display: "flex",
          gap: 16,
        }}
      >
        <span>{new Date(run.timestamp).toLocaleTimeString()}</span>
        {run.audioDuration != null && <span>{run.audioDuration.toFixed(1)}s audio</span>}
      </div>
    </div>
  );
}

// ─── ComparePanel ─────────────────────────────────────────────────────────────

function ComparePanel({ runs, referenceText }) {
  if (runs.length < 2) return null;

  return (
    <div
      style={{
        background: "rgba(20,20,28,0.9)",
        border: "1px solid rgba(56,189,248,0.2)",
        borderRadius: 16,
        padding: "22px 24px",
        marginBottom: 28,
        animation: "fadeSlideIn 0.4s ease",
      }}
    >
      <div
        style={{
          fontFamily: "JetBrains Mono,monospace",
          fontSize: 11,
          color: "rgba(56,189,248,0.7)",
          letterSpacing: 2,
          textTransform: "uppercase",
          marginBottom: 16,
        }}
      >
        ⊞ Pinned Comparison ({runs.length})
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: `repeat(${Math.min(runs.length, 3)}, 1fr)`,
          gap: 14,
        }}
      >
        {runs.map((run) => {
          const wer = referenceText.trim() ? computeWER(referenceText, run.transcription) : null;
          return (
            <div
              key={run.id}
              style={{
                background: "rgba(30,30,38,0.8)",
                border: "1px solid rgba(255,255,255,0.07)",
                borderRadius: 10,
                padding: "14px 16px",
              }}
            >
              <div
                style={{
                  fontFamily: "JetBrains Mono,monospace",
                  fontSize: 10,
                  color: "rgba(251,146,60,0.6)",
                  marginBottom: 6,
                  letterSpacing: 1,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {run.systemPrompt}
              </div>
              {wer !== null && (
                <div
                  style={{
                    fontFamily: "JetBrains Mono,monospace",
                    fontSize: 11,
                    color: wer < 0.1 ? "rgb(74,222,128)" : wer < 0.25 ? "rgb(250,204,21)" : "rgb(248,113,113)",
                    marginBottom: 8,
                    fontWeight: 500,
                  }}
                >
                  WER: {(wer * 100).toFixed(1)}%
                </div>
              )}
              <p
                style={{
                  margin: 0,
                  fontFamily: "Source Serif 4,Georgia,serif",
                  fontSize: 13,
                  lineHeight: 1.65,
                  color: "rgba(237,233,228,0.85)",
                  wordBreak: "break-word",
                }}
              >
                {run.transcription}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── main component ───────────────────────────────────────────────────────────

export default function Gemma4TranscribeTester() {
  const [endpoint, setEndpoint] = useState(DEFAULT_ENDPOINT);
  const [systemPrompt, setSystemPrompt] = useState(PRESET_PROMPTS[0].prompt);
  const [referenceText, setReferenceText] = useState("");
  const [showConfig, setShowConfig] = useState(true);
  const [replacements, setReplacements] = useState([{ find: "", replace: "" }]);
  const [vocabulary, setVocabulary] = useState([{ spoken: "", corrected: "" }]);

  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [streamingText, setStreamingText] = useState("");
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState(null);

  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadedFileName, setUploadedFileName] = useState("");

  const [runs, setRuns] = useState([]);
  const [pinnedIds, setPinnedIds] = useState(new Set());

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const streamRef = useRef(null);
  const analyserRef = useRef(null);
  const timerRef = useRef(null);
  const abortRef = useRef(null);
  const fileInputRef = useRef(null);

  // ── recording ──────────────────────────────────────────────────────────────

  const startRecording = useCallback(async () => {
    setError(null);
    setUploadedFile(null);
    setUploadedFileName("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      analyserRef.current = analyser;

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      mediaRecorder.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        await runTranscription(blob);
      };
      mediaRecorder.start(250);
      setIsRecording(true);
      setDuration(0);
      timerRef.current = setInterval(() => setDuration((d) => d + 1), 1000);
    } catch {
      setError("Microphone access denied.");
    }
  }, [endpoint, systemPrompt]);

  const stopRecording = useCallback(() => {
    clearInterval(timerRef.current);
    if (mediaRecorderRef.current?.state !== "inactive") mediaRecorderRef.current.stop();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    analyserRef.current = null;
    setIsRecording(false);
  }, []);

  // ── file upload ────────────────────────────────────────────────────────────

  const handleFileSelect = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadedFile(file);
    setUploadedFileName(file.name);
    setError(null);
  }, []);

  const transcribeUpload = useCallback(async () => {
    if (!uploadedFile) return;
    await runTranscription(uploadedFile);
  }, [uploadedFile, endpoint, systemPrompt]);

  // ── core transcription ────────────────────────────────────────────────────

  const runTranscription = useCallback(
    async (audioBlob) => {
      if (abortRef.current) abortRef.current.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      setIsProcessing(true);
      setStreamingText("");

      let audioDuration = null;
      try {
        const buf = await audioBlob.arrayBuffer();
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const decoded = await audioCtx.decodeAudioData(buf.slice(0));
        audioDuration = decoded.duration;
      } catch {
        // non-critical
      }

      try {
        const activeReplacements = replacements.filter((r) => r.find.trim());
        const activeVocabulary = vocabulary.filter((v) => v.spoken.trim() && v.corrected.trim());
        const text = await streamTranscription(
          audioBlob,
          endpoint,
          systemPrompt,
          activeReplacements,
          activeVocabulary,
          (partial) => setStreamingText(partial),
          controller.signal
        );
        if (text) {
          setRuns((prev) => [
            {
              id: Date.now(),
              systemPrompt,
              transcription: text,
              audioDuration,
              timestamp: Date.now(),
            },
            ...prev,
          ]);
        }
      } catch (e) {
        if (e.name !== "AbortError") setError(e.message);
      } finally {
        setStreamingText("");
        setIsProcessing(false);
      }
    },
    [endpoint, systemPrompt]
  );

  const togglePin = useCallback((id) => {
    setPinnedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const exportJSON = useCallback(() => {
    const data = JSON.stringify(
      runs.map((r) => ({
        timestamp: new Date(r.timestamp).toISOString(),
        systemPrompt: r.systemPrompt,
        transcription: r.transcription,
        audioDuration: r.audioDuration,
        wer: referenceText.trim()
          ? parseFloat((computeWER(referenceText, r.transcription) * 100).toFixed(2))
          : null,
      })),
      null,
      2
    );
    const url = URL.createObjectURL(new Blob([data], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `gemma4-test-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [runs, referenceText]);

  const pinnedRuns = runs.filter((r) => pinnedIds.has(r.id));

  // ── render ────────────────────────────────────────────────────────────────

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "linear-gradient(160deg,#0f0f13 0%,#1a1a24 40%,#12121a 100%)",
        color: "#ede9e4",
        fontFamily: "Source Serif 4,Georgia,serif",
        display: "flex",
        gap: 0,
      }}
    >
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500&family=Source+Serif+4:ital,opsz,wght@0,8..60,300;0,8..60,400;0,8..60,600;1,8..60,400&family=Outfit:wght@300;500;700;900&display=swap');
        @keyframes fadeSlideIn { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
        @keyframes pulseGlow { 0%,100%{box-shadow:0 0 30px rgba(239,68,68,.25)} 50%{box-shadow:0 0 45px rgba(239,68,68,.45)} }
        @keyframes spin { to{transform:rotate(360deg)} }
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }
        textarea:focus,input:focus { outline: none; }
        ::-webkit-scrollbar { width:6px } ::-webkit-scrollbar-track{background:transparent} ::-webkit-scrollbar-thumb{background:rgba(255,255,255,.1);border-radius:3px}
      `}</style>

      {/* ── left panel: config ── */}
      <div
        style={{
          width: 340,
          minWidth: 280,
          flexShrink: 0,
          borderRight: "1px solid rgba(255,255,255,0.05)",
          padding: "32px 24px",
          overflowY: "auto",
          display: "flex",
          flexDirection: "column",
          gap: 24,
        }}
      >
        {/* title */}
        <div>
          <div style={{ fontFamily: "JetBrains Mono,monospace", fontSize: 10, letterSpacing: 4, textTransform: "uppercase", color: "rgba(251,146,60,.55)", marginBottom: 8 }}>
            Gemma 4
          </div>
          <h1
            style={{
              fontFamily: "Outfit,sans-serif",
              fontSize: 22,
              fontWeight: 900,
              margin: "0 0 4px",
              background: "linear-gradient(135deg,#fb923c 0%,#e879f9 60%,#38bdf8 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
            }}
          >
            Transcription Tester
          </h1>
          <p style={{ fontSize: 12, color: "rgba(237,233,228,.4)", margin: 0, lineHeight: 1.5 }}>
            Test how system prompts affect transcription quality.
          </p>
        </div>

        {/* endpoint */}
        <div>
          <label style={labelStyle}>API Endpoint</label>
          <input
            type="text"
            value={endpoint}
            onChange={(e) => setEndpoint(e.target.value)}
            placeholder="leave blank for demo mode"
            style={inputStyle}
          />
        </div>

        {/* system prompt */}
        <div>
          <label style={labelStyle}>System Prompt</label>

          {/* presets */}
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 10 }}>
            {PRESET_PROMPTS.map((p) => (
              <button
                key={p.label}
                onClick={() => setSystemPrompt(p.prompt)}
                style={{
                  background:
                    systemPrompt === p.prompt
                      ? "rgba(251,146,60,.2)"
                      : "rgba(255,255,255,.04)",
                  border: `1px solid ${
                    systemPrompt === p.prompt
                      ? "rgba(251,146,60,.5)"
                      : "rgba(255,255,255,.08)"
                  }`,
                  borderRadius: 6,
                  color:
                    systemPrompt === p.prompt
                      ? "rgb(251,146,60)"
                      : "rgba(148,163,184,.7)",
                  padding: "4px 10px",
                  fontSize: 11,
                  cursor: "pointer",
                  fontFamily: "JetBrains Mono,monospace",
                  transition: "all 0.15s",
                }}
              >
                {p.label}
              </button>
            ))}
          </div>

          <textarea
            value={systemPrompt}
            onChange={(e) => setSystemPrompt(e.target.value)}
            rows={4}
            style={{
              ...inputStyle,
              resize: "vertical",
              lineHeight: 1.55,
            }}
          />
        </div>

        {/* vocabulary corrections — injected into prompt */}
        <div>
          <label style={labelStyle}>
            Vocabulary Corrections{" "}
            <span style={{ color: "rgba(74,222,128,.5)", fontWeight: 400, textTransform: "none" }}>
              (injected into prompt — most reliable)
            </span>
          </label>
          {vocabulary.map((rule, i) => (
            <div key={i} style={{ display: "flex", gap: 6, marginBottom: 6, alignItems: "center" }}>
              <input
                type="text"
                value={rule.spoken}
                onChange={(e) => {
                  const next = [...vocabulary];
                  next[i] = { ...next[i], spoken: e.target.value };
                  setVocabulary(next);
                }}
                placeholder="Spoken (e.g. mole mature)"
                style={{ ...inputStyle, flex: 1, fontSize: 11 }}
              />
              <span style={{ color: "rgba(74,222,128,.4)", fontFamily: "JetBrains Mono,monospace", fontSize: 12 }}>→</span>
              <input
                type="text"
                value={rule.corrected}
                onChange={(e) => {
                  const next = [...vocabulary];
                  next[i] = { ...next[i], corrected: e.target.value };
                  setVocabulary(next);
                }}
                placeholder="Write as (e.g. chole bhature)"
                style={{ ...inputStyle, flex: 1, fontSize: 11 }}
              />
              <button
                onClick={() => setVocabulary(vocabulary.filter((_, j) => j !== i))}
                style={{ background: "none", border: "none", color: "rgba(248,113,113,.5)", cursor: "pointer", fontSize: 16, padding: "0 2px", lineHeight: 1 }}
              >×</button>
            </div>
          ))}
          <button
            onClick={() => setVocabulary([...vocabulary, { spoken: "", corrected: "" }])}
            style={{ background: "rgba(74,222,128,.03)", border: "1px dashed rgba(74,222,128,.15)", borderRadius: 6, color: "rgba(74,222,128,.4)", padding: "5px 12px", fontSize: 11, cursor: "pointer", fontFamily: "JetBrains Mono,monospace", width: "100%", marginTop: 2 }}
          >
            + Add vocabulary rule
          </button>
        </div>

        {/* post-processing replacements */}
        <div>
          <label style={labelStyle}>
            Find &amp; Replace{" "}
            <span style={{ color: "rgba(148,163,184,.4)", fontWeight: 400, textTransform: "none" }}>
              (applied after transcription)
            </span>
          </label>
          {replacements.map((rule, i) => (
            <div key={i} style={{ display: "flex", gap: 6, marginBottom: 6, alignItems: "center" }}>
              <input
                type="text"
                value={rule.find}
                onChange={(e) => {
                  const next = [...replacements];
                  next[i] = { ...next[i], find: e.target.value };
                  setReplacements(next);
                }}
                placeholder="Find…"
                style={{ ...inputStyle, flex: 1, fontSize: 11 }}
              />
              <span style={{ color: "rgba(148,163,184,.3)", fontFamily: "JetBrains Mono,monospace", fontSize: 12 }}>→</span>
              <input
                type="text"
                value={rule.replace}
                onChange={(e) => {
                  const next = [...replacements];
                  next[i] = { ...next[i], replace: e.target.value };
                  setReplacements(next);
                }}
                placeholder="Replace…"
                style={{ ...inputStyle, flex: 1, fontSize: 11 }}
              />
              <button
                onClick={() => setReplacements(replacements.filter((_, j) => j !== i))}
                style={{ background: "none", border: "none", color: "rgba(248,113,113,.5)", cursor: "pointer", fontSize: 16, padding: "0 2px", lineHeight: 1 }}
              >×</button>
            </div>
          ))}
          <button
            onClick={() => setReplacements([...replacements, { find: "", replace: "" }])}
            style={{ background: "rgba(255,255,255,.03)", border: "1px dashed rgba(255,255,255,.1)", borderRadius: 6, color: "rgba(148,163,184,.4)", padding: "5px 12px", fontSize: 11, cursor: "pointer", fontFamily: "JetBrains Mono,monospace", width: "100%", marginTop: 2 }}
          >
            + Add rule
          </button>
        </div>

        {/* reference text (for WER) */}
        <div>
          <label style={labelStyle}>
            Reference Transcript{" "}
            <span style={{ color: "rgba(148,163,184,.4)", fontWeight: 400, textTransform: "none" }}>
              (optional — enables WER)
            </span>
          </label>
          <textarea
            value={referenceText}
            onChange={(e) => setReferenceText(e.target.value)}
            rows={3}
            placeholder="Paste the ground-truth text to compute Word Error Rate…"
            style={{ ...inputStyle, resize: "vertical", lineHeight: 1.55 }}
          />
        </div>

        {/* export */}
        {runs.length > 0 && (
          <button
            onClick={exportJSON}
            style={{
              background: "rgba(56,189,248,.08)",
              border: "1px solid rgba(56,189,248,.2)",
              borderRadius: 8,
              color: "rgba(56,189,248,.85)",
              padding: "10px 16px",
              fontSize: 12,
              cursor: "pointer",
              fontFamily: "JetBrains Mono,monospace",
              letterSpacing: 0.5,
            }}
          >
            ↓ Export {runs.length} run{runs.length !== 1 ? "s" : ""} as JSON
          </button>
        )}
      </div>

      {/* ── right panel: input + results ── */}
      <div
        style={{
          flex: 1,
          padding: "32px 28px",
          overflowY: "auto",
          display: "flex",
          flexDirection: "column",
          gap: 24,
        }}
      >
        {/* audio input card */}
        <div
          style={{
            background: "rgba(20,20,28,.8)",
            border: `1px solid ${isRecording ? "rgba(239,68,68,.35)" : "rgba(255,255,255,.06)"}`,
            borderRadius: 18,
            padding: "24px 28px",
            transition: "border-color .3s",
          }}
        >
          <div style={{ fontFamily: "JetBrains Mono,monospace", fontSize: 11, letterSpacing: 2, textTransform: "uppercase", color: "rgba(237,233,228,.3)", marginBottom: 18 }}>
            Audio Input
          </div>

          <WaveformVisualizer analyser={analyserRef.current} isRecording={isRecording} />

          {isRecording && (
            <div style={{ textAlign: "center", marginTop: 8, fontFamily: "JetBrains Mono,monospace", fontSize: 18, fontWeight: 500, color: "rgba(239,68,68,.9)", letterSpacing: 2 }}>
              {formatDuration(duration)}
            </div>
          )}

          <div style={{ display: "flex", alignItems: "center", gap: 14, marginTop: 18 }}>
            {/* mic button */}
            <button
              onClick={isRecording ? stopRecording : startRecording}
              disabled={isProcessing}
              style={{
                width: 56,
                height: 56,
                borderRadius: "50%",
                border: isRecording ? "2px solid rgba(239,68,68,.6)" : "2px solid rgba(251,146,60,.3)",
                background: isRecording
                  ? "radial-gradient(circle,rgba(239,68,68,.2) 0%,rgba(239,68,68,.05) 70%)"
                  : "radial-gradient(circle,rgba(251,146,60,.12) 0%,rgba(251,146,60,.02) 70%)",
                cursor: isProcessing ? "wait" : "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                flexShrink: 0,
                transition: "all .25s",
                animation: isRecording ? "pulseGlow 2s ease-in-out infinite" : "none",
                opacity: isProcessing ? 0.4 : 1,
              }}
              title={isRecording ? "Stop recording" : "Start recording"}
            >
              {isRecording ? (
                <div style={{ width: 18, height: 18, borderRadius: 4, background: "rgb(239,68,68)" }} />
              ) : (
                <div style={{ width: 22, height: 22, borderRadius: "50%", background: "linear-gradient(135deg,#fb923c,#e879f9)" }} />
              )}
            </button>

            <div style={{ fontFamily: "JetBrains Mono,monospace", fontSize: 11, color: "rgba(148,163,184,.45)", letterSpacing: 1 }}>
              {isProcessing
                ? <span style={{ display: "flex", alignItems: "center", gap: 7 }}>
                    <span style={{ width: 12, height: 12, border: "2px solid rgba(251,146,60,.3)", borderTopColor: "rgb(251,146,60)", borderRadius: "50%", display: "inline-block", animation: "spin .8s linear infinite" }} />
                    Transcribing…
                  </span>
                : isRecording
                ? "Recording — tap to stop"
                : "Tap to record"}
            </div>

            <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 10 }}>
              <span style={{ fontFamily: "JetBrains Mono,monospace", fontSize: 11, color: "rgba(148,163,184,.4)" }}>or</span>
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isRecording || isProcessing}
                style={{
                  background: "rgba(255,255,255,.04)",
                  border: "1px solid rgba(255,255,255,.09)",
                  borderRadius: 8,
                  color: "rgba(148,163,184,.7)",
                  padding: "8px 14px",
                  fontSize: 12,
                  cursor: "pointer",
                  fontFamily: "JetBrains Mono,monospace",
                  opacity: isRecording || isProcessing ? 0.4 : 1,
                }}
              >
                Upload file
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*,video/webm"
                style={{ display: "none" }}
                onChange={handleFileSelect}
              />
              {uploadedFile && (
                <button
                  onClick={transcribeUpload}
                  disabled={isProcessing}
                  style={{
                    background: "rgba(251,146,60,.12)",
                    border: "1px solid rgba(251,146,60,.3)",
                    borderRadius: 8,
                    color: "rgb(251,146,60)",
                    padding: "8px 14px",
                    fontSize: 12,
                    cursor: "pointer",
                    fontFamily: "JetBrains Mono,monospace",
                    opacity: isProcessing ? 0.4 : 1,
                    maxWidth: 160,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                  title={`Transcribe ${uploadedFileName}`}
                >
                  ▶ {uploadedFileName || "Transcribe"}
                </button>
              )}
            </div>
          </div>
        </div>

        {/* current prompt badge */}
        <div
          style={{
            fontFamily: "JetBrains Mono,monospace",
            fontSize: 11,
            color: "rgba(148,163,184,.45)",
            background: "rgba(255,255,255,.02)",
            border: "1px solid rgba(255,255,255,.05)",
            borderRadius: 8,
            padding: "8px 14px",
            letterSpacing: 0.3,
          }}
        >
          <span style={{ color: "rgba(251,146,60,.5)", marginRight: 8 }}>prompt:</span>
          {systemPrompt}
        </div>

        {/* error */}
        {error && (
          <div style={{ background: "rgba(239,68,68,.09)", border: "1px solid rgba(239,68,68,.22)", borderRadius: 10, padding: "10px 16px", fontFamily: "JetBrains Mono,monospace", fontSize: 12, color: "rgba(239,68,68,.85)" }}>
            {error}
          </div>
        )}

        {/* streaming */}
        {isProcessing && streamingText && (
          <div
            style={{
              background: "rgba(25,25,34,.8)",
              border: "1px solid rgba(251,146,60,.25)",
              borderRadius: 14,
              padding: "18px 20px",
            }}
          >
            <div style={{ fontFamily: "JetBrains Mono,monospace", fontSize: 10, color: "rgba(251,146,60,.6)", letterSpacing: 2, textTransform: "uppercase", marginBottom: 8, display: "flex", alignItems: "center", gap: 7 }}>
              <span style={{ width: 7, height: 7, borderRadius: "50%", background: "rgb(251,146,60)", display: "inline-block", animation: "blink 1s step-end infinite" }} />
              Streaming…
            </div>
            <p style={{ margin: 0, fontFamily: "Source Serif 4,Georgia,serif", fontSize: 15, lineHeight: 1.7, color: "rgba(237,233,228,.9)" }}>
              {streamingText}
              <span style={{ display: "inline-block", width: 2, height: 16, background: "rgb(251,146,60)", marginLeft: 2, verticalAlign: "text-bottom", animation: "blink 1s step-end infinite" }} />
            </p>
          </div>
        )}

        {/* comparison panel */}
        <ComparePanel runs={pinnedRuns} referenceText={referenceText} />

        {/* run history */}
        {runs.length > 0 && (
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
              <div style={{ fontFamily: "JetBrains Mono,monospace", fontSize: 11, letterSpacing: 2, textTransform: "uppercase", color: "rgba(237,233,228,.3)" }}>
                Results ({runs.length})
              </div>
              {pinnedIds.size > 0 && (
                <span style={{ fontFamily: "JetBrains Mono,monospace", fontSize: 10, color: "rgba(251,146,60,.5)" }}>
                  {pinnedIds.size} pinned
                </span>
              )}
            </div>
            {runs.map((run, i) => (
              <RunCard
                key={run.id}
                run={run}
                index={i}
                total={runs.length}
                referenceText={referenceText}
                onPin={togglePin}
                isPinned={pinnedIds.has(run.id)}
              />
            ))}
          </div>
        )}

        {runs.length === 0 && !isProcessing && (
          <div
            style={{
              flex: 1,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              padding: "60px 20px",
              color: "rgba(148,163,184,.25)",
              fontFamily: "JetBrains Mono,monospace",
              fontSize: 13,
              textAlign: "center",
              gap: 10,
            }}
          >
            <div style={{ fontSize: 36, marginBottom: 6 }}>🎙</div>
            <div>Record or upload audio to start testing</div>
            <div style={{ fontSize: 11, color: "rgba(148,163,184,.15)", marginTop: 4 }}>
              Swap the system prompt on the left to compare results
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// shared styles
const labelStyle = {
  display: "block",
  fontFamily: "JetBrains Mono,monospace",
  fontSize: 10,
  letterSpacing: 1.5,
  textTransform: "uppercase",
  color: "rgba(148,163,184,.5)",
  marginBottom: 8,
  fontWeight: 500,
};

const inputStyle = {
  width: "100%",
  background: "rgba(0,0,0,.3)",
  border: "1px solid rgba(255,255,255,.07)",
  borderRadius: 8,
  padding: "9px 12px",
  color: "#ede9e4",
  fontFamily: "JetBrains Mono,monospace",
  fontSize: 12,
  boxSizing: "border-box",
};
