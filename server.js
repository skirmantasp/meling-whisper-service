require('dotenv').config();

const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { exec } = require('child_process');
const fs = require('fs');
const path = require('path');
const os = require('os');

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

app.use(cors({
  origin: [
    'https://meling-portal-production.up.railway.app',
    'http://localhost:5173',
    'http://localhost:3000'
  ],
  methods: ['GET', 'POST'],
  allowedHeaders: ['Content-Type']
}));
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

/** Find the most recently modified JSON file in a directory. */
function findLatestJson(dir) {
  const files = fs
    .readdirSync(dir)
    .filter((f) => f.toLowerCase().endsWith('.json'))
    .map((f) => {
      const full = path.join(dir, f);
      return { full, mtime: fs.statSync(full).mtimeMs };
    })
    .sort((a, b) => b.mtime - a.mtime);

  return files.length > 0 ? files[0].full : null;
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
// POST /transcribe
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

    const command =
      `whisper ${shellQuote(inputPath)} ` +
      `--model ${model} ` +
      `--language ${language} ` +
      `--output_format json ` +
      `--output_dir ${shellQuote(OUTPUT_DIR)} ` +
      `--verbose False`;

    console.log(`[transcribe] Running: ${command}`);

    exec(command, { maxBuffer: 1024 * 1024 * 64 }, (execErr, stdout, stderr) => {
      // Input audio is no longer needed regardless of outcome.
      safeUnlink(inputPath);

      if (execErr) {
        console.error('[transcribe] Whisper failed:', stderr || execErr.message);
        return res.status(500).json({
          error: 'Transcription failed.',
          details: (stderr || execErr.message || '').trim(),
        });
      }

      const jsonPath = findLatestJson(OUTPUT_DIR);
      if (!jsonPath) {
        return res.status(500).json({ error: 'Whisper produced no JSON output.' });
      }

      let whisperResult;
      try {
        whisperResult = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
      } catch (parseErr) {
        safeUnlink(jsonPath);
        return res.status(500).json({
          error: 'Failed to parse Whisper output.',
          details: parseErr.message,
        });
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

      res.json({
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
      });
    });
  });
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

app.listen(PORT, () => {
  console.log(`meling-whisper-service listening on port ${PORT}`);
  console.log(`Model: ${WHISPER_MODEL} | Default language: ${DEFAULT_LANGUAGE}`);
  console.log(`Host: ${os.hostname()}`);
});
