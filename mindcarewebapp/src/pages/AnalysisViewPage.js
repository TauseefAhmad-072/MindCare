import { useState } from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, ChevronDown, ChevronUp, CheckCircle2, Sparkles, Shield } from 'lucide-react';

/* ─────────────────────────────────────────────────────────────────────────────
   Shared API client + result components for the MindCare dual-head model.
   Backend contract: see format_result() in app.py (POST /api/predict -> { status, result }).
   Set REACT_APP_API_BASE (CRA) or edit the fallback below (Vite etc.) to point at your Flask server.
   ───────────────────────────────────────────────────────────────────────────── */
export const API_BASE =
  (typeof process !== 'undefined' && process.env && process.env.REACT_APP_API_BASE) ||
  'http://127.0.0.1:5000/api';

export async function apiRequest(path, body, { method } = {}) {
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      method: method || (body === undefined ? 'GET' : 'POST'),
      headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new Error(`Cannot reach the MindCare server at ${API_BASE}. Is the Flask backend running?`);
  }
  let data = null;
  try { data = await res.json(); } catch { /* non-JSON error page */ }
  if (!res.ok || !data || data.status === 'error') {
    throw new Error((data && (data.error || data.detail)) || `Server error (${res.status}).`);
  }
  return data;
}

export const analyzeText = async (text) => (await apiRequest('/predict', { text })).result;

export const EMOTION_META = {
  normal:     { icon: '😐', color: '#64748B' },
  happy:      { icon: '😊', color: '#22C55E' },
  sad:        { icon: '😢', color: '#3B82F6' },
  angry:      { icon: '😠', color: '#EF4444' },
  frustrated: { icon: '😤', color: '#F97316' },
  anxious:    { icon: '😰', color: '#A78BFA' },
};

export const DISORDER_LABELS = {
  adhd: 'ADHD', anxiety_disorder: 'Anxiety Disorder', aspergers: "Asperger's / Autism",
  bipolar: 'Bipolar', bpd: 'Borderline (BPD)', depression: 'Depression',
  ocd: 'OCD', ptsd: 'PTSD', schizophrenia: 'Schizophrenia',
};

const barColor = (p) => (p < 0.35 ? '#14B8A6' : p < 0.65 ? '#F59E0B' : '#EF4444');
const pct = (p, d = 1) => `${(p * 100).toFixed(d)}%`;

/* ── Safety banner ─────────────────────────────────────────────────────────── */
export function SafetyAlert({ notice }) {
  if (!notice) return null;
  return (
    <div role="alert" style={{
      background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.35)', borderRadius: 12,
      padding: 16, display: 'flex', gap: 12, alignItems: 'flex-start',
    }}>
      <AlertTriangle size={20} color="#EF4444" style={{ flexShrink: 0, marginTop: 2 }} />
      <div>
        <div style={{ color: '#EF4444', fontWeight: 700, marginBottom: 4 }}>Please reach out for support</div>
        <div style={{ color: '#FCA5A5', fontSize: 14, lineHeight: 1.6 }}>{notice}</div>
      </div>
    </div>
  );
}

/* ── Single winning emotion ────────────────────────────────────────────────── */
export function EmotionBadge({ result, size = 'lg' }) {
  const meta = EMOTION_META[result.emotion] || { icon: '🙂', color: result.color || '#94A3B8' };
  const big = size === 'lg';
  return (
    <motion.div initial={{ opacity: 0, scale: 0.85 }} animate={{ opacity: 1, scale: 1 }}
      transition={{ type: 'spring', stiffness: 260, damping: 18 }}
      data-testid="emotion-badge"
      style={{
        display: 'inline-flex', alignItems: 'center', gap: big ? 12 : 8,
        padding: big ? '10px 22px' : '5px 12px', borderRadius: 999, background: meta.color, color: '#fff',
        fontWeight: 800, fontSize: big ? 20 : 13, letterSpacing: '0.02em',
        boxShadow: `0 4px 18px ${meta.color}55`,
      }}>
      <span style={{ fontSize: big ? 26 : 16 }}>{meta.icon}</span>
      <span>{String(result.emotion).toUpperCase()}</span>
      <span style={{ fontWeight: 500, fontSize: big ? 14 : 11, opacity: 0.92 }}>{pct(result.confidence)} confidence</span>
    </motion.div>
  );
}

/* ── Nine independent probability bars ─────────────────────────────────────── */
export function DisorderBars({ result, limit }) {
  const untrained = new Set(result.untrained_disorders || []);
  let entries = Object.entries(result.disorder_probabilities || {});
  if (limit) {
    entries = entries.filter(([k]) => !untrained.has(k)).sort((a, b) => b[1] - a[1]).slice(0, limit);
  }
  return (
    <div data-testid="disorder-bars" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {entries.map(([key, p], i) => {
        const off = untrained.has(key);
        return (
          <div key={key} data-testid={off ? 'bar-untrained' : 'bar-row'} style={{ opacity: off ? 0.5 : 1 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, marginBottom: 5 }}>
              <span style={{ color: '#E2E8F0', fontWeight: 600 }}>{DISORDER_LABELS[key] || key}</span>
              {off
                ? <span style={{ fontSize: 11, color: '#94A3B8', background: 'rgba(255,255,255,0.06)', padding: '2px 8px', borderRadius: 99 }}>not trained — no data</span>
                : <span style={{ color: '#F1F5F9', fontWeight: 700, fontVariantNumeric: 'tabular-nums' }}>{pct(p)}</span>}
            </div>
            <div style={{ height: 10, borderRadius: 99, background: 'rgba(255,255,255,0.06)', overflow: 'hidden' }}>
              {!off && (
                <motion.div initial={{ width: 0 }} animate={{ width: `${Math.max(p * 100, 0.8)}%` }}
                  transition={{ duration: 0.9, delay: i * 0.05, ease: [0.22, 1, 0.36, 1] }}
                  style={{ height: '100%', borderRadius: 99, background: barColor(p) }} />
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ── Guidance card ─────────────────────────────────────────────────────────── */
export function RecommendationsCard({ recs }) {
  if (!recs || !recs.steps) return null;
  const urgent = !!recs.urgent;
  const accent = urgent ? '#EF4444' : '#2DD4BF';
  return (
    <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }}
      data-testid="recommendations"
      style={{
        background: urgent ? 'rgba(239,68,68,0.07)' : 'linear-gradient(135deg, rgba(20,184,166,0.07), rgba(139,92,246,0.05))',
        border: `1px solid ${urgent ? 'rgba(239,68,68,0.3)' : 'rgba(45,212,191,0.2)'}`, borderRadius: 14, padding: '18px 20px',
      }}>
      <div style={{ fontSize: 12, color: accent, fontWeight: 700, marginBottom: 8, letterSpacing: '0.04em' }}>
        💡 {String(recs.title || 'WHAT YOU CAN DO NEXT').toUpperCase()}
      </div>
      <p style={{ fontSize: 14, color: '#94A3B8', lineHeight: 1.65, margin: '0 0 14px' }}>{recs.consolation}</p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {recs.steps.map((step, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
            <CheckCircle2 size={16} color={accent} style={{ flexShrink: 0, marginTop: 2 }} />
            <span style={{ fontSize: 14, color: '#94A3B8', lineHeight: 1.55 }}>{step}</span>
          </div>
        ))}
      </div>
    </motion.div>
  );
}

/* ── Collapsible computation trace ─────────────────────────────────────────── */
export function TokenTrace({ result }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ border: '1px solid rgba(255,255,255,0.06)', borderRadius: 12 }}>
      <button type="button" onClick={() => setOpen(!open)} aria-expanded={open}
        style={{ width: '100%', display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 16px',
          background: 'transparent', border: 'none', color: '#94A3B8', cursor: 'pointer', fontSize: 13, fontWeight: 600 }}>
        <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}><Sparkles size={14} color="#2DD4BF" /> Model trace (tokens → ids → scores)</span>
        {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
      </button>
      {open && (
        <div style={{ padding: '0 16px 16px', fontSize: 12, color: '#94A3B8', lineHeight: 1.7 }}>
          <div style={{ marginBottom: 8 }}>
            <b style={{ color: '#CBD5E1' }}>Clean tokens ({result.clean_tokens.length}{result.truncated ? `, truncated to ${result.padded_length}` : ''}):</b>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 6 }}>
              {result.clean_tokens.slice(0, result.padded_length).map((t, i) => (
                <span key={i} style={{ background: 'rgba(255,255,255,0.05)', borderRadius: 6, padding: '2px 7px' }}>
                  {t}<sub style={{ color: '#2DD4BF', marginLeft: 3 }}>{result.token_ids[i]}</sub>
                </span>
              ))}
            </div>
          </div>
          <b style={{ color: '#CBD5E1' }}>Emotion probabilities:</b>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, marginTop: 4 }}>
            {Object.entries(result.emotion_probabilities).map(([k, v]) => <span key={k}>{k}: {pct(v)}</span>)}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Full result (Assessment page) ─────────────────────────────────────────── */
export default function AnalysisView({ result }) {
  const top = result.top_disorder;
  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="glass-card"
      data-testid="analysis-view"
      style={{ padding: 28, marginTop: 24, display: 'flex', flexDirection: 'column', gap: 22,
        border: `2px solid ${result.urgent ? 'rgba(239,68,68,0.4)' : 'var(--border-card)'}` }}>
      <SafetyAlert notice={result.safety_notice} />

      <div>
        <div style={{ fontSize: 12, color: '#94A3B8', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 10 }}>Detected emotion</div>
        <EmotionBadge result={result} />
      </div>

      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 14, flexWrap: 'wrap', gap: 6 }}>
          <div style={{ fontSize: 12, color: '#94A3B8', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Clinical-pattern probabilities</div>
          {top && top.probability >= 0.5 && (
            <div style={{ fontSize: 13, color: '#F1F5F9' }}>Strongest match: <b>{DISORDER_LABELS[top.name] || top.name}</b> ({pct(top.probability)})</div>
          )}
        </div>
        <DisorderBars result={result} />
        <p style={{ fontSize: 12, color: '#64748B', margin: '14px 0 0', lineHeight: 1.6 }}>
          <Shield size={12} style={{ display: 'inline', marginRight: 5, verticalAlign: '-1px' }} />
          Each bar is an independent probability, so they do not sum to 100%. This is a screening signal about how the text reads,
          <b> not a diagnosis</b>. Please speak to a qualified professional about any concern.
        </p>
      </div>

      <RecommendationsCard recs={result.recommendations} />
      <TokenTrace result={result} />
    </motion.div>
  );
}

/* ── Compact card body (scrapers) ──────────────────────────────────────────── */
export function PredictionSummary({ prediction }) {
  const top = (prediction.top_disorder && prediction.top_disorder.probability >= 0.5) ? prediction.top_disorder : null;
  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', marginBottom: 14 }}>
        <EmotionBadge result={prediction} size="sm" />
        {top && <span style={{ fontSize: 13, color: '#CBD5E1' }}>Pattern: <b>{DISORDER_LABELS[top.name] || top.name}</b> ({pct(top.probability, 0)})</span>}
        {prediction.urgent && <span style={{ fontSize: 12, color: '#EF4444', fontWeight: 700 }}>⚠ safety language detected</span>}
      </div>
      <DisorderBars result={prediction} limit={3} />
    </div>
  );
}