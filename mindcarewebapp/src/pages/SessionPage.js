/*
SessionPage.js — In-memory multimodal ingestion + results viewer
Backend: Flask on http://127.0.0.1:5000
*/
import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
    FileVideo,
    Upload,
    Loader2,
    AlertCircle,
    TrendingUp,
    Users,
    Eye,
    Image,
    Mic,
} from 'lucide-react';

// ─────────────────────────────────────────────────────────────
// Backend endpoints
// ─────────────────────────────────────────────────────────────
const ENDPOINTS = {
    video: 'http://127.0.0.1:5000/api/video/predict',
    image: 'http://127.0.0.1:5000/api/image/predict',
    audio: 'http://127.0.0.1:5000/api/audio/predict',
};

const EMOTION_COLORS = {
    angry: '#EF4444',
    disgust: '#84CC16',
    fear: '#A855F7',
    happy: '#FACC15',
    neutral: '#94A3B8',
    sad: '#3B82F6',
    surprise: '#F97316',
};

const emotionColor = (e) => EMOTION_COLORS[e] || '#2DD4BF';

export default function SessionPage() {
    const [mode, setMode] = useState('video');
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const fileInputRef = useRef(null);

    const resetSelection = (newMode) => {
        setMode(newMode);
        setFile(null);
        setPreview(null);
        setResult(null);
        setError(null);
    };

    const handleFileChange = (e) => {
        if (!e.target.files || e.target.files.length === 0) return;
        const selected = e.target.files[0];

        if (selected.size > 25 * 1024 * 1024) {
            setError('File size exceeds production upload threshold limit of 25MB.');
            return;
        }

        setFile(selected);
        setError(null);
        setResult(null);

        if (mode === 'image') {
            setPreview(URL.createObjectURL(selected));
        } else {
            setPreview(null);
        }
    };

    const handleAnalyze = async (e) => {
        e.preventDefault();
        if (!file) return;

        setLoading(true);
        setError(null);
        setResult(null);

        let targetUrl;
        const formData = new FormData();

        if (mode === 'video') {
            targetUrl = ENDPOINTS.video;
            formData.append('video', file);
        } else if (mode === 'image') {
            targetUrl = ENDPOINTS.image;
            formData.append('image', file);
        } else {
            targetUrl = ENDPOINTS.audio;
            formData.append('audio', file);
        }

        try {
            const response = await fetch(targetUrl, { method: 'POST', body: formData });

            if (!response.ok) {
                let detail = '';
                try {
                    const ct = response.headers.get('content-type') || '';
                    if (ct.includes('application/json')) {
                        const j = await response.json();
                        detail = j.error || j.message || j.detail || JSON.stringify(j);
                    } else {
                        detail = await response.text();
                    }
                } catch {
                    detail = response.statusText;
                }
                throw new Error(`Backend returned ${response.status}. ${detail}`);
            }

            const data = await response.json();
            setResult(data);
        } catch (err) {
            console.error(err);
            setError(err.message || 'Failed to reach backend.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div
            style={{
                paddingTop: 120,
                minHeight: '100vh',
                paddingBottom: 80,
                position: 'relative',
            }}
        >
            <div className="container" style={{ maxWidth: 1000, padding: '0 20px', zIndex: 1 }}>
                <div style={{ textAlign: 'center', marginBottom: 40 }}>
                    <h1 style={{ fontSize: 36, fontWeight: 900, marginBottom: 8 }}>
                        Smart-Room Multi-Modal Chamber
                    </h1>
                    <p style={{ color: '#94A3B8' }}>
                        Select ingestion nodes to map unified audio-vision sessions or individual streams.
                    </p>
                </div>

                {/* Mode Tabs */}
                <div
                    style={{
                        display: 'flex',
                        gap: 10,
                        marginBottom: 24,
                        background: 'rgba(255,255,255,0.03)',
                        padding: 6,
                        borderRadius: 12,
                        border: '1px solid rgba(255,255,255,0.06)',
                    }}
                >
                    {[
                        { id: 'video', label: 'Session Video (.mp4)', Icon: FileVideo },
                        { id: 'image', label: 'Capture Frame', Icon: Image },
                        { id: 'audio', label: 'Pure Audio SER', Icon: Mic },
                    ].map(({ id, label, Icon }) => (
                        <button
                            key={id}
                            type="button"
                            onClick={() => resetSelection(id)}
                            style={{
                                flex: 1,
                                padding: '10px',
                                borderRadius: 8,
                                background: mode === id ? '#0D9488' : 'transparent',
                                color: '#FFF',
                                border: 'none',
                                cursor: 'pointer',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                gap: 8,
                                fontWeight: 600,
                            }}
                        >
                            <Icon size={16} /> {label}
                        </button>
                    ))}
                </div>

                {/* Dropzone */}
                <div className="glass-card" style={{ padding: 40, textAlign: 'center', marginBottom: 24 }}>
                    <input
                        type="file"
                        onChange={handleFileChange}
                        ref={fileInputRef}
                        style={{ display: 'none' }}
                        accept={
                            mode === 'video' ? 'video/*' : mode === 'image' ? 'image/*' : 'audio/*'
                        }
                    />

                    {!file ? (
                        <div onClick={() => fileInputRef.current.click()} style={{ cursor: 'pointer' }}>
                            <Upload size={48} color="#2DD4BF" style={{ margin: '0 auto 16px' }} />
                            <p style={{ fontSize: 16, fontWeight: 600, color: '#F1F5F9' }}>
                                Click to select your standard {mode.toUpperCase()} session file
                            </p>
                        </div>
                    ) : (
                        <div>
                            <p style={{ color: '#2DD4BF', fontWeight: 600, marginBottom: 16 }}>
                                Target Item Selected: {file.name}
                            </p>
                            {preview && (
                                <img
                                    src={preview}
                                    alt="Preview"
                                    style={{
                                        maxWidth: '100%',
                                        maxHeight: 240,
                                        borderRadius: 8,
                                        marginBottom: 16,
                                    }}
                                />
                            )}
                            <div style={{ display: 'flex', gap: 12, justifyContent: 'center' }}>
                                <button
                                    type="button"
                                    className="btn-secondary"
                                    onClick={() => setFile(null)}
                                    disabled={loading}
                                >
                                    Clear
                                </button>
                                <button
                                    type="button"
                                    className="btn-primary"
                                    onClick={handleAnalyze}
                                    disabled={loading}
                                >
                                    {loading && (
                                        <Loader2 className="spin" size={16} style={{ marginRight: 6 }} />
                                    )}
                                    {loading ? 'Extracting Subprocesses...' : 'Run Framework Analytics'}
                                </button>
                            </div>
                        </div>
                    )}
                </div>

                {/* Error */}
                <AnimatePresence>
                    {error && (
                        <motion.div
                            initial={{ opacity: 0, y: -10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0 }}
                            style={{
                                color: '#FECACA',
                                background: 'rgba(239,68,68,0.08)',
                                border: '1px solid rgba(239,68,68,0.2)',
                                padding: 16,
                                borderRadius: 12,
                                display: 'flex',
                                gap: 12,
                                marginBottom: 32,
                            }}
                        >
                            <AlertCircle color="#EF4444" style={{ flexShrink: 0 }} />
                            <div>
                                <span style={{ fontWeight: 700, display: 'block', marginBottom: 4 }}>
                                    Orchestration Warning
                                </span>
                                <span style={{ fontSize: 13.5, lineHeight: 1.5 }}>{error}</span>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Results */}
                {result && (
                    <motion.div
                        initial={{ opacity: 0, y: 15 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="glass-card"
                        style={{ padding: 24 }}
                    >
                        {mode === 'video' && <VideoResults data={result} />}
                        {mode === 'image' && <ImageResults data={result} />}
                        {mode === 'audio' && <AudioResults data={result} />}
                    </motion.div>
                )}
            </div>
        </div>
    );
}

/* ─────────────────────────────────────────────────────────────
   Shared UI atoms
   ───────────────────────────────────────────────────────────── */

function StatCard({ label, value, accent }) {
    return (
        <div
            style={{
                background: 'rgba(255,255,255,0.02)',
                padding: 16,
                borderRadius: 10,
                border: '1px solid rgba(255,255,255,0.05)',
            }}
        >
            <span
                style={{
                    fontSize: 11,
                    color: '#64748B',
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em',
                }}
            >
                {label}
            </span>
            <p
                style={{
                    fontSize: 22,
                    fontWeight: 800,
                    margin: '6px 0 0 0',
                    color: accent || '#2DD4BF',
                }}
            >
                {value}
            </p>
        </div>
    );
}

function EmotionBar({ emotion, value }) {
    const pct = Math.round(value * 100);
    return (
        <div style={{ marginBottom: 8 }}>
            <div
                style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    fontSize: 12,
                    marginBottom: 4,
                    color: '#CBD5E1',
                }}
            >
                <span style={{ textTransform: 'capitalize' }}>{emotion}</span>
                <span style={{ color: '#94A3B8' }}>{pct}%</span>
            </div>
            <div
                style={{
                    height: 6,
                    background: 'rgba(255,255,255,0.05)',
                    borderRadius: 999,
                    overflow: 'hidden',
                }}
            >
                <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${pct}%` }}
                    transition={{ duration: 0.5 }}
                    style={{
                        height: '100%',
                        background: emotionColor(emotion),
                        borderRadius: 999,
                    }}
                />
            </div>
        </div>
    );
}

function SectionTitle({ icon: Icon, children }) {
    return (
        <h4
            style={{
                color: '#F1F5F9',
                marginBottom: 12,
                fontSize: 14,
                fontWeight: 700,
                display: 'flex',
                alignItems: 'center',
                gap: 8,
            }}
        >
            {Icon && <Icon size={16} color="#2DD4BF" />}
            {children}
        </h4>
    );
}

/* ─────────────────────────────────────────────────────────────
   Video Results
   ───────────────────────────────────────────────────────────── */

function VideoResults({ data }) {
    if (data.status !== 'success') {
        return (
            <div style={{ color: '#FECACA', fontSize: 14 }}>
                Backend returned non-success status: <strong>{data.status || 'unknown'}</strong>
                {data.error && <div style={{ marginTop: 6 }}>{data.error}</div>}
            </div>
        );
    }

    const tracks = data.summary?.video_by_track
        ? Object.entries(data.summary.video_by_track)
        : [];
    const audio = data.summary?.audio_overall;
    const observations = data.observations || [];

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 28 }}>
            {/* Session stats */}
            <div>
                <SectionTitle icon={TrendingUp}>Session Overview</SectionTitle>
                <div
                    style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
                        gap: 12,
                    }}
                >
                    <StatCard label="Duration" value={`${data.duration_sec ?? '—'}s`} accent="#2DD4BF" />
                    <StatCard
                        label="Observations"
                        value={`${data.num_observations ?? 0}`}
                        accent="#8B5CF6"
                    />
                    <StatCard
                        label="Frames Processed"
                        value={`${data.num_frames_total ?? 0}`}
                        accent="#38BDF8"
                    />
                    <StatCard
                        label="Native FPS"
                        value={`${data.native_fps ?? '—'}`}
                        accent="#F59E0B"
                    />
                </div>
            </div>

            {/* Per-track emotion summary */}
            {tracks.length > 0 && (
                <div>
                    <SectionTitle icon={Users}>Face Expression Tracker Summary</SectionTitle>
                    <div
                        style={{
                            display: 'grid',
                            gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))',
                            gap: 14,
                        }}
                    >
                        {tracks.map(([trackId, log]) => (
                            <div
                                key={trackId}
                                style={{
                                    background: 'rgba(255,255,255,0.02)',
                                    padding: 16,
                                    borderRadius: 12,
                                    border: '1px solid rgba(255,255,255,0.05)',
                                }}
                            >
                                <div
                                    style={{
                                        display: 'flex',
                                        justifyContent: 'space-between',
                                        alignItems: 'center',
                                        marginBottom: 12,
                                    }}
                                >
                                    <span style={{ fontSize: 13, fontWeight: 700, color: '#F1F5F9' }}>
                                        Person Track #{trackId}
                                    </span>
                                    <span
                                        style={{
                                            fontSize: 11,
                                            color: '#94A3B8',
                                            background: 'rgba(255,255,255,0.04)',
                                            padding: '2px 8px',
                                            borderRadius: 999,
                                        }}
                                    >
                                        {log.num_observations} frames
                                    </span>
                                </div>

                                <div style={{ marginBottom: 12 }}>
                                    <span style={{ fontSize: 11, color: '#64748B' }}>DOMINANT</span>
                                    <div
                                        style={{
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: 8,
                                            marginTop: 4,
                                        }}
                                    >
                                        <span
                                            style={{
                                                width: 10,
                                                height: 10,
                                                borderRadius: '50%',
                                                background: emotionColor(log.dominant_emotion),
                                            }}
                                        />
                                        <strong
                                            style={{
                                                fontSize: 16,
                                                textTransform: 'capitalize',
                                                color: emotionColor(log.dominant_emotion),
                                            }}
                                        >
                                            {log.dominant_emotion}
                                        </strong>
                                    </div>
                                </div>

                                {log.emotion_distribution && (
                                    <div>
                                        {Object.entries(log.emotion_distribution).map(([emo, val]) => (
                                            <EmotionBar key={emo} emotion={emo} value={val} />
                                        ))}
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Audio summary (if present) */}
            {audio && (
                <div>
                    <SectionTitle icon={Mic}>Acoustic Vocal SER Summary</SectionTitle>
                    <div
                        style={{
                            background: 'rgba(139,92,246,0.05)',
                            border: '1px solid rgba(139,92,246,0.18)',
                            padding: 16,
                            borderRadius: 12,
                        }}
                    >
                        {audio.dominant_emotion && (
                            <p style={{ margin: '0 0 10px 0', fontSize: 14, color: '#F1F5F9' }}>
                                Dominant:{' '}
                                <strong style={{ color: '#A78BFA', textTransform: 'capitalize' }}>
                                    {audio.dominant_emotion}
                                </strong>
                            </p>
                        )}
                        {audio.emotion_distribution &&
                            Object.entries(audio.emotion_distribution).map(([emo, val]) => (
                                <EmotionBar key={emo} emotion={emo} value={val} />
                            ))}
                    </div>
                </div>
            )}

            {/* Observations timeline */}
            {observations.length > 0 && (
                <div>
                    <SectionTitle icon={Eye}>Frame-by-Frame Observations</SectionTitle>
                    <div
                        style={{
                            display: 'grid',
                            gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
                            gap: 14,
                        }}
                    >
                        {observations.map((obs, idx) => {
                            const v = obs.video || {};
                            const face = v.faces?.[0];
                            const skipped = face?.skip_reason;
                            return (
                                <div
                                    key={idx}
                                    style={{
                                        background: 'rgba(255,255,255,0.02)',
                                        borderRadius: 12,
                                        border: '1px solid rgba(255,255,255,0.05)',
                                        overflow: 'hidden',
                                    }}
                                >
                                    {obs.frame_b64_uri ? (
                                        <img
                                            src={obs.frame_b64_uri}
                                            alt={`Frame ${obs.frame_index}`}
                                            style={{
                                                width: '100%',
                                                height: 160,
                                                objectFit: 'cover',
                                                display: 'block',
                                                opacity: skipped ? 0.5 : 1,
                                            }}
                                        />
                                    ) : (
                                        <div
                                            style={{
                                                height: 160,
                                                display: 'grid',
                                                placeItems: 'center',
                                                background: 'rgba(255,255,255,0.02)',
                                                color: '#475569',
                                                fontSize: 12,
                                            }}
                                        >
                                            No frame
                                        </div>
                                    )}

                                    <div style={{ padding: 12 }}>
                                        <div
                                            style={{
                                                display: 'flex',
                                                justifyContent: 'space-between',
                                                alignItems: 'center',
                                                marginBottom: 6,
                                            }}
                                        >
                                            <span style={{ fontSize: 11, color: '#64748B' }}>
                                                t = {obs.timestamp_sec?.toFixed(2) ?? '—'}s
                                            </span>
                                            <span
                                                style={{
                                                    fontSize: 10,
                                                    color: '#94A3B8',
                                                    background: 'rgba(255,255,255,0.04)',
                                                    padding: '2px 6px',
                                                    borderRadius: 4,
                                                }}
                                            >
                                                {obs.trigger}
                                            </span>
                                        </div>

                                        {v.emotion ? (
                                            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                                <span
                                                    style={{
                                                        width: 8,
                                                        height: 8,
                                                        borderRadius: '50%',
                                                        background: emotionColor(v.emotion),
                                                    }}
                                                />
                                                <strong
                                                    style={{
                                                        fontSize: 13,
                                                        textTransform: 'capitalize',
                                                        color: emotionColor(v.emotion),
                                                    }}
                                                >
                                                    {v.emotion}
                                                </strong>
                                                <span style={{ fontSize: 11, color: '#64748B', marginLeft: 'auto' }}>
                                                    {(v.confidence * 100).toFixed(0)}%
                                                </span>
                                            </div>
                                        ) : (
                                            <span style={{ fontSize: 12, color: '#64748B' }}>
                                                {skipped === 'quality_gate_failed'
                                                    ? 'Frame skipped (low quality)'
                                                    : 'No emotion detected'}
                                            </span>
                                        )}
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}

            {/* Disclaimers */}
            {(data.video_disclaimer || data.audio_disclaimer) && (
                <div
                    style={{
                        marginTop: 8,
                        padding: 14,
                        borderRadius: 10,
                        background: 'rgba(250,204,21,0.05)',
                        border: '1px solid rgba(250,204,21,0.15)',
                        fontSize: 11.5,
                        lineHeight: 1.6,
                        color: '#CBD5E1',
                    }}
                >
                    <strong style={{ color: '#FACC15' }}>Disclaimer · </strong>
                    {data.video_disclaimer || data.audio_disclaimer}
                </div>
            )}
        </div>
    );
}

/* ─────────────────────────────────────────────────────────────
   Image Results
   ───────────────────────────────────────────────────────────── */

function ImageResults({ data }) {
    // The /api/image/predict endpoint wraps its payload: { status, result: {...} }
    if (data.status && data.status !== 'success') {
        return <div style={{ color: '#FECACA' }}>Analysis failed: {data.error || data.status}</div>;
    }

    // Unwrap the result envelope
    const r = data.result || data;

    if (r.face_detected === false) {
        return (
            <div style={{ color: '#94A3B8', textAlign: 'center', padding: 24 }}>
                <span style={{ fontSize: 36 }}>🙈</span>
                <p style={{ marginTop: 12 }}>{r.reason || 'No face detected in the image.'}</p>
            </div>
        );
    }

    const dominant = r.emotion || r.dominant_emotion;
    const distribution = r.probabilities || r.emotion_distribution || {};
    const confidence = r.confidence;
    const numFaces = r.num_faces;
    const disclaimer = r.disclaimer;

    return (
        <div>
            <SectionTitle icon={Image}>Frame Expression Analysis</SectionTitle>
            <div
                style={{
                    background: 'rgba(255,255,255,0.02)',
                    padding: 16,
                    borderRadius: 12,
                    border: '1px solid rgba(255,255,255,0.05)',
                    marginBottom: 12,
                }}
            >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14, flexWrap: 'wrap', gap: 10 }}>
                    {dominant && (
                        <div>
                            <span style={{ fontSize: 11, color: '#64748B', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Detected Expression</span>
                            <div style={{ fontSize: 22, fontWeight: 800, color: emotionColor(dominant), textTransform: 'capitalize', marginTop: 2 }}>
                                {dominant}
                            </div>
                        </div>
                    )}
                    <div style={{ display: 'flex', gap: 14 }}>
                        {confidence !== undefined && (
                            <div style={{ textAlign: 'right' }}>
                                <span style={{ fontSize: 11, color: '#64748B' }}>Confidence</span>
                                <div style={{ fontSize: 18, fontWeight: 800, color: '#F1F5F9' }}>{(confidence * 100).toFixed(1)}%</div>
                            </div>
                        )}
                        {numFaces !== undefined && (
                            <div style={{ textAlign: 'right' }}>
                                <span style={{ fontSize: 11, color: '#64748B' }}>Faces</span>
                                <div style={{ fontSize: 18, fontWeight: 800, color: '#38BDF8' }}>{numFaces}</div>
                            </div>
                        )}
                    </div>
                </div>
                {Object.keys(distribution).length > 0 &&
                    Object.entries(distribution)
                        .sort((a, b) => b[1] - a[1])
                        .map(([emo, val]) => (
                            <EmotionBar key={emo} emotion={emo} value={val} />
                        ))}
            </div>
            {disclaimer && (
                <div
                    style={{
                        marginTop: 8,
                        padding: 14,
                        borderRadius: 10,
                        background: 'rgba(250,204,21,0.05)',
                        border: '1px solid rgba(250,204,21,0.15)',
                        fontSize: 11.5,
                        lineHeight: 1.6,
                        color: '#CBD5E1',
                    }}
                >
                    <strong style={{ color: '#FACC15' }}>Disclaimer · </strong>
                    {disclaimer}
                </div>
            )}
        </div>
    );
}

/* ─────────────────────────────────────────────────────────────
   Audio Results
   ───────────────────────────────────────────────────────────── */

function AudioResults({ data }) {
    if (data.status && data.status !== 'success' && !data.result) {
        return <div style={{ color: '#FECACA' }}>Analysis failed: {data.status || data.error}</div>;
    }

    const payload = data.result || data;
    const dominant = (payload.dominant_emotion || payload.emotion || '').toLowerCase();
    const distribution = payload.emotion_distribution || payload.probabilities || {};
    const waveform = payload.waveform_analysis || payload.waveform_tone_analysis || {};
    const toneExplanation = payload.tone_explanation || '';
    const capturedText = payload.captured_text || payload.transcript?.text || '';
    const urduSentiment = payload.urdu_sentiment || null;
    const disclaimer = payload.audio_disclaimer || payload.disclaimer;

    const toneColors = {
        happy: '#22C55E',
        neutral: '#94A3B8',
        sad: '#3B82F6',
        angry: '#EF4444',
    };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            <SectionTitle icon={Mic}>Vocal Emotion & Waveform Tone Analysis</SectionTitle>

            {/* 1. Main Tone Emotion Card */}
            <div
                style={{
                    background: 'rgba(139,92,246,0.06)',
                    border: '1px solid rgba(139,92,246,0.22)',
                    padding: 20,
                    borderRadius: 14,
                }}
            >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
                    <div>
                        <span style={{ fontSize: 12, color: '#A78BFA', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                            Dominant Tone Affect
                        </span>
                        <h3 style={{ margin: '4px 0 0 0', fontSize: 24, fontWeight: 800, color: toneColors[dominant] || '#F1F5F9', textTransform: 'uppercase' }}>
                            {dominant || 'Neutral'}
                        </h3>
                    </div>
                    {payload.confidence && (
                        <div style={{ textAlign: 'right' }}>
                            <span style={{ fontSize: 11, color: '#94A3B8' }}>Tone Confidence</span>
                            <div style={{ fontSize: 20, fontWeight: 800, color: '#F1F5F9' }}>
                                {(payload.confidence * 100).toFixed(1)}%
                            </div>
                        </div>
                    )}
                </div>

                {toneExplanation && (
                    <div style={{ fontSize: 13, color: '#CBD5E1', marginBottom: 16, lineHeight: 1.5, background: 'rgba(0,0,0,0.2)', padding: '10px 14px', borderRadius: 8 }}>
                        💡 {toneExplanation}
                    </div>
                )}

                {/* 4-Class Tone Probabilities */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {Object.entries(distribution).map(([emo, val]) => (
                        <div key={emo}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12.5, marginBottom: 4 }}>
                                <span style={{ textTransform: 'capitalize', color: '#E2E8F0', fontWeight: 600 }}>{emo}</span>
                                <span style={{ color: '#F8FAFC', fontWeight: 700 }}>{(Number(val) * 100).toFixed(1)}%</span>
                            </div>
                            <div style={{ height: 8, background: 'rgba(255,255,255,0.06)', borderRadius: 99, overflow: 'hidden' }}>
                                <div
                                    style={{
                                        width: `${Math.max(Number(val) * 100, 1)}%`,
                                        height: '100%',
                                        background: toneColors[emo.toLowerCase()] || '#A78BFA',
                                        borderRadius: 99,
                                        transition: 'width 0.6s ease',
                                    }}
                                />
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* 2. Waveform Acoustic Tone Physics */}
            {waveform && Object.keys(waveform).length > 0 && (
                <div
                    style={{
                        background: 'rgba(255,255,255,0.02)',
                        border: '1px solid rgba(255,255,255,0.07)',
                        padding: 18,
                        borderRadius: 14,
                    }}
                >
                    <div style={{ fontSize: 13, fontWeight: 700, color: '#38BDF8', marginBottom: 12, display: 'flex', alignItems: 'center', gap: 6 }}>
                        📊 Acoustic Waveform Tone Metrics
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 12 }}>
                        <div style={{ background: 'rgba(0,0,0,0.25)', padding: 12, borderRadius: 10 }}>
                            <div style={{ fontSize: 11, color: '#94A3B8' }}>Fundamental Pitch (F0)</div>
                            <div style={{ fontSize: 16, fontWeight: 800, color: '#F8FAFC', marginTop: 4 }}>
                                {waveform.pitch_mean_hz ? `${waveform.pitch_mean_hz} Hz` : 'N/A'}
                            </div>
                            <div style={{ fontSize: 10, color: '#64748B', marginTop: 2 }}>
                                Var: ±{waveform.pitch_std_hz || 0} Hz
                            </div>
                        </div>
                        <div style={{ background: 'rgba(0,0,0,0.25)', padding: 12, borderRadius: 10 }}>
                            <div style={{ fontSize: 11, color: '#94A3B8' }}>Loudness / RMS Energy</div>
                            <div style={{ fontSize: 16, fontWeight: 800, color: '#F8FAFC', marginTop: 4 }}>
                                {waveform.energy_rms !== undefined ? Number(waveform.energy_rms).toFixed(3) : 'N/A'}
                            </div>
                            <div style={{ fontSize: 10, color: '#64748B', marginTop: 2 }}>
                                {waveform.energy_level || 'Normal'}
                            </div>
                        </div>
                        <div style={{ background: 'rgba(0,0,0,0.25)', padding: 12, borderRadius: 10 }}>
                            <div style={{ fontSize: 11, color: '#94A3B8' }}>Tone Brightness</div>
                            <div style={{ fontSize: 16, fontWeight: 800, color: '#F8FAFC', marginTop: 4 }}>
                                {waveform.spectral_centroid_hz ? `${waveform.spectral_centroid_hz} Hz` : 'N/A'}
                            </div>
                            <div style={{ fontSize: 10, color: '#64748B', marginTop: 2 }}>
                                {waveform.tone_brightness || 'Spectral Centroid'}
                            </div>
                        </div>
                        <div style={{ background: 'rgba(0,0,0,0.25)', padding: 12, borderRadius: 10 }}>
                            <div style={{ fontSize: 11, color: '#94A3B8' }}>Pitch Contour</div>
                            <div style={{ fontSize: 14, fontWeight: 700, color: '#38BDF8', marginTop: 4 }}>
                                {waveform.pitch_contour || 'Stable'}
                            </div>
                            <div style={{ fontSize: 10, color: '#64748B', marginTop: 2 }}>
                                ZCR: {waveform.zero_crossing_rate || 0}
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* 3. Captured Urdu Speech & Sentiment Card */}
            {capturedText && (
                <div
                    style={{
                        background: 'rgba(20,184,166,0.06)',
                        border: '1px solid rgba(20,184,166,0.25)',
                        padding: 18,
                        borderRadius: 14,
                    }}
                >
                    <div style={{ fontSize: 13, fontWeight: 700, color: '#2DD4BF', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
                        🎙️ Captured Urdu Speech (صوتی نقل اور جذباتی تجزیہ)
                    </div>
                    <div
                        style={{
                            fontSize: 18,
                            lineHeight: 1.8,
                            color: '#F0FDFA',
                            fontFamily: 'Noto Nastaliq Urdu, Tahoma, sans-serif',
                            direction: 'rtl',
                            textAlign: 'right',
                            background: 'rgba(0,0,0,0.3)',
                            padding: 14,
                            borderRadius: 10,
                            marginBottom: 12,
                        }}
                    >
                        "{capturedText}"
                    </div>

                    {urduSentiment && (
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center', marginTop: 10 }}>
                            <span style={{ fontSize: 12, color: '#94A3B8' }}>Urdu Sentiment:</span>
                            <span
                                style={{
                                    padding: '4px 12px',
                                    borderRadius: 99,
                                    fontSize: 12,
                                    fontWeight: 700,
                                    textTransform: 'uppercase',
                                    background:
                                        urduSentiment.sentiment === 'positive'
                                            ? 'rgba(34,197,94,0.2)'
                                            : urduSentiment.sentiment === 'negative'
                                            ? 'rgba(239,68,68,0.2)'
                                            : 'rgba(148,163,184,0.2)',
                                    color:
                                        urduSentiment.sentiment === 'positive'
                                            ? '#4ADE80'
                                            : urduSentiment.sentiment === 'negative'
                                            ? '#F87171'
                                            : '#CBD5E1',
                                    border: `1px solid ${
                                        urduSentiment.sentiment === 'positive'
                                            ? '#22C55E'
                                            : urduSentiment.sentiment === 'negative'
                                            ? '#EF4444'
                                            : '#94A3B8'
                                    }`,
                                }}
                            >
                                {urduSentiment.sentiment} ({urduSentiment.selected_emotion || urduSentiment.emotion})
                            </span>
                            {urduSentiment.confidence && (
                                <span style={{ fontSize: 12, color: '#CBD5E1' }}>
                                    Confidence: {(urduSentiment.confidence * 100).toFixed(0)}%
                                </span>
                            )}
                        </div>
                    )}
                </div>
            )}

            {/* 4. Safety Alert Banner (if crisis detected) */}
            {(payload.safety_hit || payload.safety_notice) && (
                <div
                    style={{
                        padding: 14,
                        borderRadius: 10,
                        background: 'rgba(239,68,68,0.1)',
                        border: '1px solid rgba(239,68,68,0.35)',
                        fontSize: 13,
                        lineHeight: 1.6,
                        color: '#FCA5A5',
                    }}
                >
                    <strong style={{ color: '#EF4444' }}>⚠️ Safety Notice · </strong>
                    {payload.safety_notice}
                </div>
            )}

            {/* 5. Clinical Disclaimer */}
            {disclaimer && (
                <div
                    style={{
                        padding: 12,
                        borderRadius: 10,
                        background: 'rgba(250,204,21,0.05)',
                        border: '1px solid rgba(250,204,21,0.15)',
                        fontSize: 11.5,
                        lineHeight: 1.6,
                        color: '#CBD5E1',
                    }}
                >
                    <strong style={{ color: '#FACC15' }}>Disclaimer · </strong>
                    {disclaimer}
                </div>
            )}
        </div>
    );
}