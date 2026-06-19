require('dotenv').config();

const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { exec } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');
const crypto = require('crypto');

const app = express();

const PORT = process.env.PORT || 3000;
const WHISPER_MODEL = process.env.WHISPER_MODEL || 'medium';
const DEFAULT_LANGUAGE = 'no';
const GDPR_NOTE =
  'All audio is processed in-memory on EU infrastructure (Railway EU-West, Amsterdam) ' +
  'using the open-source Whisper model. No audio or transcript leaves the EU, and all ' +
  'temporary files are deleted immediately after processing.';

const UPLOAD_DIR = '/tmp/whisper-uploads/';
const OUTPUT_DIR = '/tmp/whisper-output/';

// Ensure working directories exist.
for (const dir of [UPLOAD_DIR, OUTPUT_DIR]) {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

const ALLOWED_EXTENSIONS = ['.mp3', '.wav', '.mp4', '.m4a', '.ogg', '.webm', '.flac'];
const MAX_FILE_SIZE = 500 * 1024 * 1024; // 500 MB

// In-memory registry of async transcription jobs. Each entry:
//   { status: 'processing' | 'done' | 'error', result, error, created_at }
const jobs = new Map();
const JOB_TTL_MS = 60 * 60 * 1000; // 1 hour

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
  res.setTimeout(600000); // 10 minutes
  next();
});
app.use(express.json({ limit: '50mb' }));

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

/** Shell-escape a path for safe use inside a double-quoted argument. */
function shellQuote(value) {
  // Wrap in double quotes and escape any embedded double quotes / backslashes.
  return `"${String(value).replace(/(["\\$`])/g, '\\$1')}"`;
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
// Background transcription runner
// ---------------------------------------------------------------------------

/**
 * Run Whisper for a single job in the background, updating the job entry in
 * `jobs` when it completes or fails. Never throws; all outcomes are recorded
 * on the job. The input audio and output JSON are deleted once Whisper is done.
 */
function runWhisperJob({ jobId, inputPath, originalFilename, language, model }) {
  const command =
    `python3 -m whisper ${shellQuote(inputPath)} ` +
    `--model ${model} ` +
    `--language ${language} ` +
    `--output_format json ` +
    `--output_dir ${shellQuote(OUTPUT_DIR)} ` +
    `--verbose False`;

  console.log(`[transcribe:${jobId}] Running: ${command}`);

  exec(command, { maxBuffer: 1024 * 1024 * 64 }, (execErr, stdout, stderr) => {
    console.log(`[Whisper:${jobId}] stdout:`, stdout?.substring(0, 500));
    console.log(`[Whisper:${jobId}] stderr:`, stderr?.substring(0, 500));

    // Input audio is no longer needed once Whisper has finished reading it.
    safeUnlink(inputPath);

    const job = jobs.get(jobId);
    if (!job) return; // Job already expired and was cleaned up.

    if (execErr) {
      console.error(`[transcribe:${jobId}] Whisper failed:`, stderr || execErr.message);
      job.status = 'error';
      job.error = (stderr || execErr.message || 'Transcription failed.').trim();
      return;
    }

    // Whisper writes "<input-basename>.json" into OUTPUT_DIR. Deriving the path
    // deterministically (rather than picking the newest file) keeps concurrent
    // jobs from grabbing each other's output.
    const jsonPath = path.join(
      OUTPUT_DIR,
      `${path.basename(inputPath, path.extname(inputPath))}.json`
    );

    let whisperResult;
    try {
      whisperResult = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
    } catch (parseErr) {
      safeUnlink(jsonPath);
      job.status = 'error';
      job.error = `Failed to parse Whisper output: ${parseErr.message}`;
      return;
    }

    // Output JSON is no longer needed once read.
    safeUnlink(jsonPath);

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

    job.status = 'done';
    job.result = {
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
  });
}

// ---------------------------------------------------------------------------
// POST /transcribe  — accepts an upload, starts a background job, returns its ID
// ---------------------------------------------------------------------------
app.post('/transcribe', (req, res) => {
  upload.single('audio')(req, res, (uploadErr) => {
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

    const jobId = crypto.randomUUID();
    jobs.set(jobId, { status: 'processing', result: null, error: null, created_at: Date.now() });

    // Drop the job entry after one hour so the Map does not grow unbounded.
    setTimeout(() => jobs.delete(jobId), JOB_TTL_MS);

    // Kick off Whisper in the background — do NOT await it.
    runWhisperJob({ jobId, inputPath, originalFilename, language, model });

    res.json({ job_id: jobId, status: 'processing' });
  });
});

// ---------------------------------------------------------------------------
// GET /status/:job_id  — poll the status/result of a transcription job
// ---------------------------------------------------------------------------
app.get('/status/:job_id', (req, res) => {
  const job = jobs.get(req.params.job_id);

  if (!job) {
    return res.status(404).json({ status: 'error', error: 'Unknown or expired job ID.' });
  }

  if (job.status === 'done') {
    return res.json({ status: 'done', result: job.result });
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
// Error handler (catches stray multer / express errors)
// ---------------------------------------------------------------------------
app.use((err, req, res, next) => {
  console.error('[error]', err.message);
  res.status(500).json({ error: err.message });
});

const server = app.listen(PORT, () => {
  console.log(`✅ Meling Whisper Service running on port ${PORT}`);
  console.log(`   Model: ${process.env.WHISPER_MODEL || 'medium'}`);
  console.log(`   GDPR: EU-only processing`);
});
server.timeout = 600000; // 10 minutes
