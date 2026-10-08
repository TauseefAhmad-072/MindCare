import { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import {
    Camera,
    Upload,
    Loader2,
    AlertTriangle,
    ImageOff,
} from 'lucide-react';

const API_BASE = 'http://127.0.0.1:5000/api';

export default function ImagePage() {
    const [file, setFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const inputRef = useRef(null);

    const handleFile = (f) => {
        if (!f) return;

        setFile(f);
        setPreviewUrl(URL.createObjectURL(f));
        setResult(null);
        setError(null);
    };

    const handleAnalyze = async () => {
        if (!file || loading) return;

        setLoading(true);
        setError(null);
        setResult(null);

        const formData = new FormData();
        formData.append('image', file);

        try {
            const res = await fetch(`${API_BASE}/image/predict`, {
                method: 'POST',
                body: formData,
            });

            const data = await res.json();

            if (data.status !== 'success') {
                throw new Error(data.error || 'Image analysis failed.');
            }

            setResult(data.result);
        } catch (err) {
            setError(
                err.message ||
                'Could not connect to the API. Is the Flask server running?'
            );
        } finally {
            setLoading(false);
        }
    };

    return (
        <div
            className="container"
            style={{
                paddingTop: 140,
                paddingBottom: 100,
            }}
        >
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                style={{
                    maxWidth: 720,
                    margin: '0 auto',
                }}
            >
                <div
                    style={{
                        textAlign: 'center',
                        marginBottom: 40,
                    }}
                >
                    <div
                        style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            width: 60,
                            height: 60,
                            borderRadius: 20,
                            background: 'linear-gradient(135deg, #14B8A6, #8B5CF6)',
                            marginBottom: 20,
                        }}
                    >
                        <Camera size={30} color="#fff" />
                    </div>

                    <h1
                        style={{
                            fontSize: 42,
                            fontWeight: 800,
                            color: '#F8FAFC',
                            marginBottom: 12,
                        }}
                    >
                        Facial <span className="gradient-text">Emotion</span> Analysis
                    </h1>

                    <p
                        style={{
                            fontSize: 16,
                            color: '#94A3B8',
                        }}
                    >
                        Upload a photo to detect the observed facial expression using our
                        ViT emotion classifier.
                    </p>
                </div>

                <div
                    className="glass-card"
                    style={{
                        padding: 30,
                        marginBottom: 30,
                    }}
                >
                    <div
                        onClick={() => inputRef.current?.click()}
                        style={{
                            border: '2px dashed rgba(45,212,191,0.3)',
                            borderRadius: 14,
                            padding: 30,
                            textAlign: 'center',
                            cursor: 'pointer',
                            background: 'rgba(45,212,191,0.03)',
                        }}
                    >
                        <input
                            ref={inputRef}
                            type="file"
                            accept=".jpg,.jpeg,.png,.bmp,.webp"
                            hidden
                            onChange={(e) => handleFile(e.target.files[0])}
                        />

                        {previewUrl ? (
                            <img
                                src={previewUrl}
                                alt="preview"
                                style={{
                                    maxHeight: 260,
                                    maxWidth: '100%',
                                    borderRadius: 10,
                                }}
                            />
                        ) : (
                            <>
                                <Upload
                                    size={32}
                                    color="#2DD4BF"
                                    style={{
                                        marginBottom: 10,
                                    }}
                                />

                                <div
                                    style={{
                                        color: '#94A3B8',
                                        fontSize: 14,
                                    }}
                                >
                                    Click to select an image (JPG, PNG, WEBP, BMP)
                                </div>
                            </>
                        )}
                    </div>

                    <button
                        className="btn-primary"
                        onClick={handleAnalyze}
                        disabled={!file || loading}
                        style={{
                            width: '100%',
                            marginTop: 20,
                            justifyContent: 'center',
                            opacity: !file || loading ? 0.5 : 1,
                        }}
                    >
                        {loading ? (
                            <>
                                <Loader2 size={18} className="spin" />
                                Analyzing...
                            </>
                        ) : (
                            'Analyze Expression'
                        )}
                    </button>
                </div>

                {error && (
                    <div
                        style={{
                            background: 'rgba(239,68,68,0.1)',
                            border: '1px solid rgba(239,68,68,0.2)',
                            borderRadius: 12,
                            padding: 18,
                            display: 'flex',
                            gap: 14,
                            color: '#FECACA',
                            marginBottom: 24,
                        }}
                    >
                        <AlertTriangle size={22} color="#EF4444" />
                        <p>{error}</p>
                    </div>
                )}

                {result && !result.face_detected && (
                    <div
                        className="glass-card"
                        style={{
                            padding: 24,
                            textAlign: 'center',
                        }}
                    >
                        <ImageOff
                            size={26}
                            color="#64748B"
                            style={{
                                marginBottom: 10,
                            }}
                        />

                        <div
                            style={{
                                color: '#94A3B8',
                            }}
                        >
                            {result.reason || 'No face detected in the provided image.'}
                        </div>
                    </div>
                )}

                {result && result.face_detected && (
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="glass-card"
                        style={{
                            padding: 26,
                        }}
                    >
                        <div
                            style={{
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'space-between',
                                marginBottom: 20,
                            }}
                        >
                            <div>
                                <div
                                    style={{
                                        fontSize: 12,
                                        color: '#64748B',
                                        textTransform: 'uppercase',
                                        letterSpacing: '0.08em',
                                    }}
                                >
                                    Observed Expression
                                </div>

                                <div
                                    style={{
                                        fontSize: 26,
                                        fontWeight: 800,
                                        color: '#2DD4BF',
                                        textTransform: 'capitalize',
                                    }}
                                >
                                    {result.emotion}
                                </div>
                            </div>

                            <div
                                style={{
                                    textAlign: 'right',
                                }}
                            >
                                <div
                                    style={{
                                        fontSize: 12,
                                        color: '#64748B',
                                        textTransform: 'uppercase',
                                    }}
                                >
                                    Confidence
                                </div>

                                <div
                                    style={{
                                        fontSize: 22,
                                        fontWeight: 800,
                                        color: '#F1F5F9',
                                    }}
                                >
                                    {(result.confidence * 100).toFixed(1)}%
                                </div>
                            </div>
                        </div>

                        <div
                            style={{
                                display: 'flex',
                                flexDirection: 'column',
                                gap: 8,
                                marginBottom: 20,
                            }}
                        >
                            {Object.entries(result.probabilities || {})
                                .sort((a, b) => b[1] - a[1])
                                .map(([label, prob]) => (
                                    <div key={label}>
                                        <div
                                            style={{
                                                display: 'flex',
                                                justifyContent: 'space-between',
                                                fontSize: 12,
                                                color: '#94A3B8',
                                                marginBottom: 4,
                                                textTransform: 'capitalize',
                                            }}
                                        >
                                            <span>{label}</span>
                                            <span>{(prob * 100).toFixed(1)}%</span>
                                        </div>

                                        <div
                                            style={{
                                                height: 5,
                                                background: 'rgba(255,255,255,0.05)',
                                                borderRadius: 10,
                                                overflow: 'hidden',
                                            }}
                                        >
                                            <div
                                                style={{
                                                    height: '100%',
                                                    width: `${prob * 100}%`,
                                                    background:
                                                        'linear-gradient(90deg, #14B8A6, #8B5CF6)',
                                                    borderRadius: 10,
                                                }}
                                            />
                                        </div>
                                    </div>
                                ))}
                        </div>

                        <p
                            style={{
                                fontSize: 12,
                                color: '#64748B',
                                lineHeight: 1.6,
                                borderTop: '1px solid rgba(255,255,255,0.05)',
                                paddingTop: 14,
                            }}
                        >
                            {result.disclaimer}
                        </p>
                    </motion.div>
                )}
            </motion.div>
        </div>
    );
}