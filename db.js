// ---------------------------------------------------------------------------
// PostgreSQL job store
// ---------------------------------------------------------------------------
// Replaces the previous in-memory Map so transcription jobs survive server
// restarts/redeploys. Connects using the DATABASE_URL that Railway injects
// automatically — credentials are never hardcoded here.
//
// The exported helpers keep the same job object shape the rest of the service
// (and the frontend, via /status) already expects:
//   { status, result, summary, summary_error, analysis, analysis_error, error,
//     created_at, updated_at }
//
// Column mapping notes:
//   - `transcript` (TEXT) stores the full structured result object as JSON —
//     the frontend reads `result` (text, segments, duration, metadata), so the
//     whole object is persisted, not just the plain text.
//   - `summary` is JSONB (the structured Norwegian summary object).
//   - `analysis` (TEXT) stores the analysis object as JSON.
// ---------------------------------------------------------------------------

const { Pool } = require('pg');

// Railway's internal Postgres connection (postgres.railway.internal) does not
// require/support TLS, so SSL is left off by default. Set DATABASE_SSL=true if
// you ever connect over the public proxy, which does require it.
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: process.env.DATABASE_SSL === 'true' ? { rejectUnauthorized: false } : false,
});

// Without this listener, an error on an idle client (e.g. the DB restarting)
// would be emitted as an uncaught 'error' event and crash the process.
pool.on('error', (err) => {
  console.error('[db] Unexpected idle client error:', err.message);
});

/**
 * Create the jobs table if it doesn't already exist. Awaited at startup; the
 * caller logs and continues if it throws so a DB outage doesn't crash the
 * server.
 */
async function initDb() {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS jobs (
      id             TEXT PRIMARY KEY,
      status         TEXT NOT NULL,
      filename       TEXT,
      transcript     TEXT,
      summary        JSONB,
      analysis       TEXT,
      summary_error  TEXT,
      analysis_error TEXT,
      error          TEXT,
      case_number    TEXT,
      created_at     TIMESTAMPTZ DEFAULT NOW(),
      updated_at     TIMESTAMPTZ DEFAULT NOW()
    )
  `);
  // Migration for databases created before case assignment existed.
  await pool.query('ALTER TABLE jobs ADD COLUMN IF NOT EXISTS case_number TEXT');
}

/** Insert a freshly-created job. created_at/updated_at default to NOW(). */
async function createJob({ id, status, filename }) {
  await pool.query(
    'INSERT INTO jobs (id, status, filename) VALUES ($1, $2, $3)',
    [id, status, filename]
  );
}

/**
 * Fetch a job by id and map it back to the in-memory job shape, or return null
 * if it doesn't exist. JSONB (`summary`) comes back already parsed; the TEXT
 * JSON columns (`transcript`/`analysis`) are parsed here.
 */
async function getJob(id) {
  const { rows } = await pool.query('SELECT * FROM jobs WHERE id = $1', [id]);
  if (rows.length === 0) return null;

  const r = rows[0];
  return {
    id: r.id,
    status: r.status,
    filename: r.filename,
    result: r.transcript ? JSON.parse(r.transcript) : null,
    summary: r.summary || null,
    summary_error: r.summary_error || null,
    analysis: r.analysis ? JSON.parse(r.analysis) : null,
    analysis_error: r.analysis_error || null,
    error: r.error || null,
    case_number: r.case_number || null,
    created_at: r.created_at,
    updated_at: r.updated_at,
  };
}

// Columns updateJob is allowed to set. Names are validated against this list so
// the dynamic SQL can never include caller-controlled identifiers.
const UPDATABLE_COLUMNS = new Set([
  'status',
  'transcript',
  'summary',
  'analysis',
  'summary_error',
  'analysis_error',
  'error',
  'case_number',
]);

/**
 * Partially update a job. `fields` is a plain object of column → value; pass
 * the already-serialized JSON string for `transcript`/`analysis`, and either a
 * JS object or JSON string for the JSONB `summary` column. Always bumps
 * updated_at.
 */
async function updateJob(id, fields) {
  const keys = Object.keys(fields).filter((k) => UPDATABLE_COLUMNS.has(k));
  if (keys.length === 0) return;

  const sets = keys.map((k, i) => `${k} = $${i + 2}`);
  sets.push('updated_at = NOW()');
  const values = keys.map((k) => fields[k]);

  await pool.query(
    `UPDATE jobs SET ${sets.join(', ')} WHERE id = $1`,
    [id, ...values]
  );
}

/**
 * List all jobs (lightweight — no transcript/summary payloads) for the cases
 * overview. Newest first. Each row: { id, filename, status, case_number,
 * created_at }.
 */
async function listJobs() {
  const { rows } = await pool.query(
    `SELECT id, filename, status, case_number, created_at
       FROM jobs
       ORDER BY created_at DESC`
  );
  return rows.map((r) => ({
    id: r.id,
    filename: r.filename || null,
    status: r.status,
    case_number: r.case_number || null,
    created_at: r.created_at,
  }));
}

/** Delete a single job by id. Returns true if a row was removed, false if no
 * job with that id existed. */
async function deleteJob(id) {
  const { rowCount } = await pool.query('DELETE FROM jobs WHERE id = $1', [id]);
  return rowCount > 0;
}

module.exports = {
  pool,
  initDb,
  createJob,
  getJob,
  updateJob,
  listJobs,
  deleteJob,
};
