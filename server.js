require('dotenv').config();

const express = require('express');
const cors = require('cors');
const multer = require('multer');
const http = require('http');
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { Anthropic } = require('@anthropic-ai/sdk');
const db = require('./db');

const app = express();

const PORT = process.env.PORT || 3000;
const WHISPER_MODEL = process.env.WHISPER_MODEL || 'NbAiLab/nb-whisper-large-v3';
const CLAUDE_MODEL = process.env.CLAUDE_MODEL || 'claude-sonnet-4-6';
const DEFAULT_LANGUAGE = 'no';
// Base URL of the portal frontend, used to build deep links in notification
// emails. Overridable via env for local/staging deployments.
const PORTAL_URL = process.env.PORTAL_URL || 'https://meling-portal-production.up.railway.app';

// Anthropic client — reads ANTHROPIC_API_KEY from the environment. The SDK
// constructor throws when no key is present, so only construct it when the key
// exists; analyzeTranscript no-ops when the client is null, letting transcription
// run without an Anthropic key.
const anthropic = process.env.ANTHROPIC_API_KEY ? new Anthropic() : null;
const GDPR_NOTE =
  'All audio is processed in-memory on EU infrastructure (Railway EU-West, Amsterdam) ' +
  'using the open-source NB-Whisper Norwegian model. No audio or transcript leaves the EU, ' +
  'and all temporary files are deleted immediately after processing.';

const UPLOAD_DIR = '/tmp/whisper-uploads/';
const OUTPUT_DIR = '/tmp/whisper-output/';

// Ensure working directories exist.
for (const dir of [UPLOAD_DIR, OUTPUT_DIR]) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

const ALLOWED_EXTENSIONS = ['.mp3', '.wav', '.mp4', '.m4a', '.ogg', '.webm', '.flac'];
const MAX_FILE_SIZE = 500 * 1024 * 1024; // 500 MB — large legal recordings (several hours)

// Timeouts sized for large legal audio. Uploading up to 500 MB over a slow
// connection can take many minutes, and CPU transcription of a multi-hour
// recording runs far longer than the old 10-minute cap.
const UPLOAD_TIMEOUT_MS = 60 * 60 * 1000;            // 60 min — HTTP upload socket/request
const TRANSCRIPTION_TIMEOUT_MS = 4 * 60 * 60 * 1000; // 4 h   — Whisper inference call

// Transcription jobs are persisted in PostgreSQL (see db.js) so they survive
// server restarts/redeploys. Rows older than 6 hours are purged by a periodic
// sweep (see CLEANUP_INTERVAL_MS below).
const CLEANUP_INTERVAL_MS = 30 * 60 * 1000; // run TTL cleanup every 30 minutes

app.use(cors({
  origin: [
    'https://meling-portal-production.up.railway.app',
    'http://localhost:5173',
    'http://localhost:3000'
  ],
  methods: ['GET', 'POST'],
  allowedHeaders: ['Content-Type']
}));
app.use((req, res, next) => {
  res.setTimeout(UPLOAD_TIMEOUT_MS);
  next();
});
// 100 MB headroom for JSON endpoints (/chat, /search) that carry the full
// transcript of a multi-hour recording. File uploads do not go through here.
app.use(express.json({ limit: '100mb' }));

// ---------------------------------------------------------------------------
// Multer configuration
// ---------------------------------------------------------------------------
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, UPLOAD_DIR),
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    const unique = `${Date.now()}-${Math.round(Math.random() * 1e9)}${ext}`;
    cb(null, unique);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: MAX_FILE_SIZE },
  fileFilter: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    if (ALLOWED_EXTENSIONS.includes(ext)) {
      cb(null, true);
    } else {
      cb(new Error(`Unsupported file type "${ext}". Allowed: ${ALLOWED_EXTENSIONS.join(', ')}`));
    }
  },
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Convert a number of seconds into an HH:MM:SS string. */
function toTimestamp(seconds) {
  const total = Math.floor(Number(seconds) || 0);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const pad = (n) => String(n).padStart(2, '0');
  return `${pad(h)}:${pad(m)}:${pad(s)}`;
}

/** Safely delete a file, ignoring "not found" errors. */
function safeUnlink(filePath) {
  if (!filePath) return;
  fs.unlink(filePath, (err) => {
    if (err && err.code !== 'ENOENT') {
      console.error(`Failed to delete ${filePath}:`, err.message);
    }
  });
}

/** Escape a string for safe interpolation into HTML email bodies. */
function escapeHtml(value) {
  return String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/** Format a processing duration (milliseconds) into a Norwegian-friendly string. */
function formatProcessingTime(ms) {
  const totalSeconds = Math.max(0, Math.round(ms / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return minutes > 0 ? `${minutes} min ${seconds} sek` : `${seconds} sek`;
}

/**
 * Send a notification email through the Resend API. Fire-and-forget: never
 * throws, logs failures. No-ops if Resend is not configured or no recipient
 * was provided.
 */
async function sendResendEmail({ to, subject, html }) {
  if (!to) return;

  const apiKey = process.env.RESEND_API_KEY;
  const from = process.env.RESEND_FROM_EMAIL;
  if (!apiKey || !from) {
    console.warn('[email] RESEND_API_KEY / RESEND_FROM_EMAIL not configured; skipping email.');
    return;
  }

  try {
    const resp = await fetch('https://api.resend.com/emails', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ from, to, subject, html }),
    });

    if (!resp.ok) {
      const body = await resp.text();
      console.error(`[email] Resend returned ${resp.status}: ${body}`);
    } else {
      console.log(`[email] Notification sent to ${to} ("${subject}")`);
    }
  } catch (err) {
    console.error('[email] Failed to send notification:', err.message);
  }
}

/** Notify the requester that their transcription job finished successfully. */
function notifyJobSuccess(email, jobId, result, processingMs) {
  if (!email) return;

  const preview = (result.text || '').slice(0, 300);
  const filename = result.metadata && result.metadata.original_filename;
  const link = `${PORTAL_URL}/transcription?jobId=${encodeURIComponent(jobId)}`;
  const button =
    `<a href="${escapeHtml(link)}" ` +
    `style="display:inline-block;padding:12px 24px;background:#1a2b4a;color:#ffffff;` +
    `text-decoration:none;border-radius:8px;font-weight:600;font-size:15px;">` +
    `Åpne transkripsjonen →</a>`;

  const html =
    `<h2>Transkripsjon ferdig ✓</h2>` +
    (filename ? `<p><strong>Fil:</strong> ${escapeHtml(filename)}</p>` : '') +
    `<p><strong>Behandlingstid:</strong> ${escapeHtml(formatProcessingTime(processingMs))}</p>` +
    `<p style="margin:24px 0;">${button}</p>` +
    `<p><strong>Forhåndsvisning:</strong></p>` +
    `<blockquote>${escapeHtml(preview)}${result.text.length > 300 ? '…' : ''}</blockquote>`;

  sendResendEmail({ to: email, subject: 'Transkripsjon ferdig ✓', html });
}

/** Notify the requester that their transcription job failed. */
function notifyJobError(email, jobId, errorMsg, processingMs) {
  if (!email) return;

  const html =
    `<h2>Transkripsjon feilet</h2>` +
    `<p><strong>Jobb-ID:</strong> ${escapeHtml(jobId)}</p>` +
    `<p><strong>Behandlingstid:</strong> ${escapeHtml(formatProcessingTime(processingMs))}</p>` +
    `<p><strong>Feilmelding:</strong></p>` +
    `<blockquote>${escapeHtml(errorMsg)}</blockquote>`;

  sendResendEmail({ to: email, subject: 'Transkripsjon feilet', html });
}

// ---------------------------------------------------------------------------
// GET /health
// ---------------------------------------------------------------------------
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'meling-whisper-service',
    model: WHISPER_MODEL,
    default_language: DEFAULT_LANGUAGE,
    gdpr_note: GDPR_NOTE,
    timestamp: new Date().toISOString(),
  });
});

// ---------------------------------------------------------------------------
// Whisper server HTTP client
// ---------------------------------------------------------------------------

/**
 * POST to the co-located Python faster-whisper server (127.0.0.1:8765).
 * The server holds the model in memory between calls, so there is no
 * per-request model load penalty.
 */
function callWhisperServer(inputPath, language, context) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify({ input: inputPath, language, context: context || '' });
    const options = {
      hostname: '127.0.0.1',
      port: 8765,
      path: '/transcribe',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(body),
      },
      timeout: TRANSCRIPTION_TIMEOUT_MS,
    };

    const req = http.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => {
        if (res.statusCode !== 200) {
          reject(new Error(`Whisper server returned ${res.statusCode}: ${data}`));
          return;
        }
        try {
          resolve(JSON.parse(data));
        } catch (e) {
          reject(new Error(`Failed to parse Whisper response: ${e.message}`));
        }
      });
    });

    req.on('error', reject);
    req.on('timeout', () => req.destroy(new Error('Whisper transcription timed out after 4 hours')));
    req.write(body);
    req.end();
  });
}

// ---------------------------------------------------------------------------
// Claude post-processing
// ---------------------------------------------------------------------------

// JSON schema constraining Claude's response. Every property is required and
// objects disallow extras, as the structured-outputs API requires. For HIGH
// confidence segments, `suggestion` and `reason` are returned as empty strings.
const ANALYSIS_SCHEMA = {
  type: 'object',
  properties: {
    segments: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'integer' },
          confidence: { type: 'string', enum: ['HIGH', 'LOW'] },
          suggestion: { type: 'string' },
          reason: { type: 'string' },
        },
        required: ['id', 'confidence', 'suggestion', 'reason'],
        additionalProperties: false,
      },
    },
    summary: { type: 'string' },
    flagged_count: { type: 'integer' },
  },
  required: ['segments', 'summary', 'flagged_count'],
  additionalProperties: false,
};

/**
 * Ask Claude to review the transcript segments. Returns the structured analysis
 * object, or null if analysis is unavailable (no API key, no segments). Throws
 * on an actual API/parse failure so the caller can record it on the job.
 */
async function analyzeTranscript({ segments, context }) {
  if (!anthropic) {
    console.warn('[claude] ANTHROPIC_API_KEY not configured; skipping analysis.');
    return null;
  }
  if (!Array.isArray(segments) || segments.length === 0) {
    return null;
  }

  // Send only the id + text of each segment — timestamps aren't needed for review.
  const segmentInput = segments.map((seg) => ({ id: seg.id, text: seg.text }));

  const contextBlock =
    context && context.trim()
      ? `\n\nKontekst oppgitt av brukeren (navn, steder, fagtermer som kan forekomme):\n${context.trim()}\n`
      : '';

  const system =
    'Du er en assistent som kvalitetssikrer norske transkripsjoner for et advokatfirma. ' +
    'Du vurderer hvert segment og flagger de som sannsynligvis inneholder transkripsjonsfeil ' +
    '(feilstavede navn/steder, ord som ikke gir mening i konteksten, sannsynlig feilhørte ord). ' +
    'For hvert segment setter du confidence til "HIGH" når teksten virker korrekt, eller "LOW" ' +
    'når den bør gjennomgås. For "LOW"-segmenter gir du en korrigert versjon i "suggestion" og ' +
    'en kort begrunnelse i "reason" (på norsk). For "HIGH"-segmenter skal "suggestion" og "reason" ' +
    'være tomme strenger. Du skriver også et kort sammendrag av hele transkripsjonen på norsk i ' +
    '"summary", og angir antall flaggede segmenter i "flagged_count".';

  const userText =
    'Gjennomgå følgende transkripsjonssegmenter.' +
    contextBlock +
    '\n\nSegmenter (JSON):\n' +
    JSON.stringify(segmentInput);

  // Stream the response so a large transcript doesn't risk an HTTP timeout.
  const stream = anthropic.messages.stream({
    model: CLAUDE_MODEL,
    max_tokens: 32000,
    thinking: { type: 'adaptive' },
    system,
    output_config: { format: { type: 'json_schema', schema: ANALYSIS_SCHEMA } },
    messages: [{ role: 'user', content: userText }],
  });

  const message = await stream.finalMessage();

  if (message.stop_reason === 'refusal') {
    throw new Error('Claude refused to analyze the transcript.');
  }

  const textBlock = message.content.find((block) => block.type === 'text');
  if (!textBlock) {
    throw new Error('Claude response contained no text block.');
  }

  const analysis = JSON.parse(textBlock.text);

  // Recompute flagged_count server-side so it's always consistent with segments.
  analysis.flagged_count = analysis.segments.filter((s) => s.confidence === 'LOW').length;

  return analysis;
}

// ---------------------------------------------------------------------------
// Claude automatic summary
// ---------------------------------------------------------------------------

// JSON schema constraining the structured Norwegian legal summary. Every key is
// required; arrays may be empty when a section has nothing to report.
const SUMMARY_SCHEMA = {
  type: 'object',
  properties: {
    hvem: { type: 'array', items: { type: 'string' } },
    hovedpunkter: { type: 'array', items: { type: 'string' } },
    motsigelser: { type: 'array', items: { type: 'string' } },
    datoerOgTall: { type: 'array', items: { type: 'string' } },
    anbefalteSporsmal: { type: 'array', items: { type: 'string' } },
  },
  required: ['hvem', 'hovedpunkter', 'motsigelser', 'datoerOgTall', 'anbefalteSporsmal'],
  additionalProperties: false,
};

/**
 * Generate a structured Norwegian legal summary of the transcript. Unlike the
 * opt-in analysis, this always runs — it only uses the transcript text that
 * already lives on our EU servers. Returns the parsed summary object, or null
 * when summarisation is unavailable (no API key, no transcript text). Throws on
 * an actual API/parse failure so the caller can swallow it.
 */
async function generateSummary({ text, context }) {
  if (!anthropic) {
    console.warn('[summary] ANTHROPIC_API_KEY not configured; skipping summary.');
    return null;
  }
  if (!text || !text.trim()) {
    return null;
  }

  const contextBlock =
    context && context.trim()
      ? `\n\nKontekst oppgitt av brukeren (navn, steder, fagtermer som kan forekomme):\n${context.trim()}\n`
      : '';

  const system =
    'Du er en juridisk assistent for et norsk advokatfirma. Du lager et kort, ' +
    'strukturert sammendrag av en transkripsjon av et lydopptak. Du svarer alltid ' +
    'på norsk og baserer deg kun på innholdet i transkripsjonen. Du fyller ut feltene:\n' +
    '- "hvem": personer/parter som er til stede eller nevnes i opptaket.\n' +
    '- "hovedpunkter": 3-5 punkter med de viktigste utsagnene.\n' +
    '- "motsigelser": motsigelser eller uklarheter i opptaket (tom liste hvis ingen).\n' +
    '- "datoerOgTall": viktige datoer, beløp, saksnumre og tall som nevnes (tom liste hvis ingen).\n' +
    '- "anbefalteSporsmal": 2-3 anbefalte oppfølgingsspørsmål advokaten kan stille.\n' +
    'Hvert felt er en liste med korte strenger. Ikke dikt opp informasjon som ikke ' +
    'finnes i transkripsjonen.';

  const userText =
    'Lag et sammendrag av følgende transkripsjon.' +
    contextBlock +
    '\n\nTranskripsjon:\n' +
    text.trim();

  // Stream the response so a long transcript doesn't risk an HTTP timeout.
  const stream = anthropic.messages.stream({
    model: CLAUDE_MODEL,
    max_tokens: 8000,
    system,
    output_config: { format: { type: 'json_schema', schema: SUMMARY_SCHEMA } },
    messages: [{ role: 'user', content: userText }],
  });

  const message = await stream.finalMessage();

  if (message.stop_reason === 'refusal') {
    throw new Error('Claude refused to summarize the transcript.');
  }

  const textBlock = message.content.find((block) => block.type === 'text');
  if (!textBlock) {
    throw new Error('Claude summary response contained no text block.');
  }

  return JSON.parse(textBlock.text);
}

// ---------------------------------------------------------------------------
// Background transcription runner
// ---------------------------------------------------------------------------

/**
 * Submit a transcription job to the persistent Python Whisper server, then
 * persist the outcome to the jobs table when it completes or fails. Never
 * throws; all outcomes are recorded on the job row.
 */
async function runWhisperJob({ jobId, inputPath, originalFilename, language, model, email, context, analyze }) {
  // Read created_at once (set when the row was inserted) so processing time can
  // be reported even after a long transcription. Fall back to now if the row or
  // DB is unavailable.
  const existing = await db.getJob(jobId).catch(() => null);
  const createdAtMs = existing && existing.created_at
    ? new Date(existing.created_at).getTime()
    : Date.now();

  console.log(`[transcribe:${jobId}] Sending to whisper server: ${inputPath}`);

  let whisperResult;
  try {
    whisperResult = await callWhisperServer(inputPath, language, context);
  } catch (err) {
    safeUnlink(inputPath);
    const processingMs = Date.now() - createdAtMs;
    console.error(`[transcribe:${jobId}] Whisper failed:`, err.message);
    const errorMsg = err.message || 'Transcription failed.';
    try {
      await db.updateJob(jobId, { status: 'error', error: errorMsg });
    } catch (dbErr) {
      console.error(`[transcribe:${jobId}] Failed to persist error:`, dbErr.message);
    }
    notifyJobError(email, jobId, errorMsg, processingMs);
    return;
  }

  safeUnlink(inputPath);

  const processingMs = Date.now() - createdAtMs;

  const rawSegments = Array.isArray(whisperResult.segments) ? whisperResult.segments : [];
  const segments = rawSegments.map((seg, index) => ({
    id: typeof seg.id === 'number' ? seg.id : index,
    start: toTimestamp(seg.start),
    end: toTimestamp(seg.end),
    start_seconds: Number(seg.start) || 0,
    end_seconds: Number(seg.end) || 0,
    text: (seg.text || '').trim(),
  }));

  const durationSeconds =
    segments.length > 0 ? segments[segments.length - 1].end_seconds : 0;

  const result = {
    text: (whisperResult.text || '').trim(),
    language: whisperResult.language || language,
    duration_seconds: durationSeconds,
    segments,
    metadata: {
      original_filename: originalFilename,
      model_used: model,
      processed_at: new Date().toISOString(),
      gdpr_note: GDPR_NOTE,
    },
  };

  // Always generate an automatic Norwegian legal summary. Unlike the opt-in
  // analysis, this has no GDPR implication beyond what the transcript already
  // does — it only reuses the transcript text already on our EU servers. Kick
  // it off now so it runs concurrently with any analysis below; it's awaited
  // before the job is marked done. Failures are non-fatal: the transcript is
  // still delivered with the summary error recorded on the job.
  let summary = null;
  let summaryError = null;
  console.log(`[summary:${jobId}] Generating summary with ${CLAUDE_MODEL}`);
  const summaryPromise = generateSummary({ text: result.text, context })
    .then((s) => {
      summary = s;
    })
    .catch((err) => {
      console.error(`[summary:${jobId}] Summary failed:`, err.message);
      summaryError = err.message || 'Claude summary failed.';
    });

  // Post-process the transcript with Claude only when the client opted in.
  // Analysis sends the transcript to the Anthropic API (outside the EU), so
  // when it runs we surface a GDPR notice on the result. Failures here are
  // non-fatal — the transcript is still delivered, with the analysis error
  // recorded on the job.
  let analysis = null;
  let analysisError = null;
  if (analyze) {
    result.analysis_gdpr_notice =
      'Legal analysis processed via Anthropic API (outside EU). ' +
      'Enable only if you accept this.';
    try {
      console.log(`[claude:${jobId}] Analyzing ${segments.length} segments with ${CLAUDE_MODEL}`);
      analysis = await analyzeTranscript({ segments, context });
    } catch (err) {
      console.error(`[claude:${jobId}] Analysis failed:`, err.message);
      analysisError = err.message || 'Claude analysis failed.';
    }
  }

  // Wait for the summary to settle before marking the job done so the result
  // and summary land together. summaryPromise never rejects (it records its
  // own error), so this can't throw.
  await summaryPromise;

  try {
    await db.updateJob(jobId, {
      status: 'done',
      transcript: JSON.stringify(result),
      summary: summary === null ? null : JSON.stringify(summary),
      summary_error: summaryError,
      analysis: analysis === null ? null : JSON.stringify(analysis),
      analysis_error: analysisError,
    });
  } catch (dbErr) {
    console.error(`[transcribe:${jobId}] Failed to persist result:`, dbErr.message);
  }

  notifyJobSuccess(email, jobId, result, processingMs);
}

// ---------------------------------------------------------------------------
// POST /transcribe  — accepts an upload, starts a background job, returns its ID
// ---------------------------------------------------------------------------
app.post('/transcribe', (req, res) => {
  upload.single('audio')(req, res, async (uploadErr) => {
    if (uploadErr) {
      const code = uploadErr.code === 'LIMIT_FILE_SIZE' ? 413 : 400;
      return res.status(code).json({ error: uploadErr.message });
    }

    if (!req.file) {
      return res.status(400).json({ error: 'No audio file provided (field name must be "audio").' });
    }

    const inputPath = req.file.path;
    const originalFilename = req.file.originalname;
    const language = (req.body && req.body.language) || DEFAULT_LANGUAGE;
    const model = WHISPER_MODEL;
    const email = (req.body && typeof req.body.email === 'string' && req.body.email.trim()) || null;
    const context =
      (req.body && typeof req.body.context === 'string' && req.body.context.trim()) || null;
    // Claude analysis is opt-in. It sends the transcript to the Anthropic API
    // (outside the EU), so it only runs when the client explicitly requests it.
    const analyze = Boolean(req.body && req.body.analyze === true) ||
      Boolean(req.body && req.body.analyze === 'true');

    const jobId = crypto.randomUUID();

    // Persist the job before kicking off work so it survives a restart and so
    // /status can find it immediately.
    try {
      await db.createJob({ id: jobId, status: 'processing', filename: originalFilename });
    } catch (dbErr) {
      safeUnlink(inputPath);
      console.error(`[transcribe:${jobId}] Failed to create job:`, dbErr.message);
      return res.status(500).json({ error: 'Could not create transcription job.' });
    }

    // Kick off Whisper in the background — do NOT await it.
    runWhisperJob({ jobId, inputPath, originalFilename, language, model, email, context, analyze })
      .catch((err) => console.error(`[job:${jobId}] Unhandled error:`, err.message));

    res.json({ job_id: jobId, status: 'processing' });
  });
});

// ---------------------------------------------------------------------------
// GET /status/:job_id  — poll the status/result of a transcription job
// ---------------------------------------------------------------------------
app.get('/status/:job_id', async (req, res) => {
  let job;
  try {
    job = await db.getJob(req.params.job_id);
  } catch (dbErr) {
    console.error(`[status:${req.params.job_id}] DB lookup failed:`, dbErr.message);
    return res.status(500).json({ status: 'error', error: 'Could not look up job status.' });
  }

  if (!job) {
    return res.status(404).json({ status: 'error', error: 'Unknown or expired job ID.' });
  }

  if (job.status === 'done') {
    return res.json({
      status: 'done',
      result: job.result,
      summary: job.summary,
      summary_error: job.summary_error || null,
      analysis: job.analysis,
      analysis_error: job.analysis_error || null,
    });
  }
  if (job.status === 'error') {
    return res.json({ status: 'error', error: job.error });
  }
  return res.json({ status: 'processing' });
});

// ---------------------------------------------------------------------------
// POST /search
// ---------------------------------------------------------------------------
app.post('/search', (req, res) => {
  const { transcript, query } = req.body || {};

  if (!transcript || !Array.isArray(transcript.segments)) {
    return res
      .status(400)
      .json({ error: 'Body must include a "transcript" object with a "segments" array.' });
  }
  if (!query || typeof query !== 'string' || !query.trim()) {
    return res.status(400).json({ error: 'Body must include a non-empty "query" string.' });
  }

  const needle = query.trim().toLowerCase();
  const segments = transcript.segments;

  const matches = [];
  segments.forEach((seg, index) => {
    const text = (seg.text || '').toLowerCase();
    if (text.includes(needle)) {
      const before = index > 0 ? segments[index - 1] : null;
      const after = index < segments.length - 1 ? segments[index + 1] : null;

      matches.push({
        id: seg.id,
        start: seg.start,
        end: seg.end,
        start_seconds: seg.start_seconds,
        end_seconds: seg.end_seconds,
        text: seg.text,
        context: {
          before: before ? before.text : null,
          after: after ? after.text : null,
        },
      });
    }
  });

  res.json({
    query,
    match_count: matches.length,
    matches,
  });
});

// ---------------------------------------------------------------------------
// POST /chat  — answer a lawyer's question about a transcript via Claude
// ---------------------------------------------------------------------------

const CHAT_SYSTEM_PROMPT =
  'Du er en juridisk assistent. Brukeren er en norsk advokat. ' +
  'Svar alltid på norsk. Du har tilgang til en transkripsjon av et lydopptak. ' +
  'Svar på spørsmål basert kun på innholdet i transkripsjonen.';

app.post('/chat', async (req, res) => {
  const { jobId, question, transcript } = req.body || {};

  if (!question || typeof question !== 'string' || !question.trim()) {
    return res.status(400).json({ error: 'Body must include a non-empty "question" string.' });
  }
  if (!Array.isArray(transcript)) {
    return res.status(400).json({ error: 'Body must include a "transcript" array of segments.' });
  }
  if (!anthropic) {
    return res.status(503).json({ error: 'Chat is unavailable: ANTHROPIC_API_KEY is not configured.' });
  }

  // Flatten the segments into a timestamped transcript Claude can read.
  const transcriptText = transcript
    .map((seg) => {
      const stamp = seg && seg.start != null ? `[${seg.start}] ` : '';
      return `${stamp}${(seg && seg.text) || ''}`.trim();
    })
    .filter(Boolean)
    .join('\n');

  const userText =
    'Her er transkripsjonen av lydopptaket:\n\n' +
    transcriptText +
    '\n\nSpørsmål fra advokaten:\n' +
    question.trim();

  try {
    console.log(`[chat:${jobId || 'n/a'}] Answering question over ${transcript.length} segments with ${CLAUDE_MODEL}`);

    const message = await anthropic.messages.create({
      model: CLAUDE_MODEL,
      max_tokens: 2000,
      system: CHAT_SYSTEM_PROMPT,
      messages: [{ role: 'user', content: userText }],
    });

    const textBlock = message.content.find((block) => block.type === 'text');
    const answer = textBlock ? textBlock.text : 'Beklager, jeg klarte ikke å svare på spørsmålet.';

    res.json({ answer });
  } catch (err) {
    console.error(`[chat:${jobId || 'n/a'}] Failed:`, err.message);
    res.status(500).json({ error: 'Kunne ikke hente svar fra assistenten.' });
  }
});

// ---------------------------------------------------------------------------
// Error handler (catches stray multer / express errors)
// ---------------------------------------------------------------------------
app.use((err, req, res, next) => {
  console.error('[error]', err.message);
  res.status(500).json({ error: err.message });
});

// Initialise the database (create the jobs table) and start the periodic TTL
// sweep. A DB failure here is logged but never crashes the server — the rest of
// the service stays up and individual DB operations fail gracefully.
db.initDb()
  .then(() => {
    console.log('✅ Database ready (jobs table ensured).');
  })
  .catch((err) => {
    console.error('[db] Initialisation failed (server will continue):', err.message);
  });

// Purge jobs older than 6 hours every 30 minutes. Self-contained try/catch so a
// transient DB error doesn't take down the interval.
setInterval(async () => {
  try {
    const removed = await db.deleteOldJobs();
    if (removed > 0) console.log(`[db] Cleaned up ${removed} job(s) older than 6 hours.`);
  } catch (err) {
    console.error('[db] TTL cleanup failed:', err.message);
  }
}, CLEANUP_INTERVAL_MS);

const server = app.listen(PORT, () => {
  console.log(`✅ Meling Whisper Service running on port ${PORT}`);
  console.log(`   Model: ${process.env.WHISPER_MODEL || 'medium'}`);
  console.log(`   GDPR: EU-only processing`);
});
// Socket inactivity timeout. Large uploads stream continuously so they stay
// active, but allow generous headroom. requestTimeout overrides Node's ~5 min
// default, which would otherwise abort a large/slow upload (e.g. 500 MB, or a
// 67 MB file on a slow connection) mid-stream. headersTimeout left at default.
server.timeout = UPLOAD_TIMEOUT_MS;
server.requestTimeout = UPLOAD_TIMEOUT_MS;
