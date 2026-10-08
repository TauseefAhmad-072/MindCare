import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import AnalysisView, { API_BASE, analyzeText, apiRequest } from '../pages/AnalysisViewPage';

// 24 negation-heavy / mixed-sentiment / symptom-rich stress-test sentences (click to load)
const EXAMPLES = [
    ['Bipolar · mania then crash', "I stay up for 48 hours straight cleaning my entire house and coding a brand new app because I feel like an absolute genius who can conquer the universe, but this morning I woke up crashing into a total wall of complete irritability, absolutely furious and frustrated for no reason."],
    ['Bipolar · mood swing', "One day I'm completely unstoppable, mapping out impossible academic goals, and the next day I can't even force myself to get out of bed because the sudden downswing leaves me entirely hollow and empty."],
    ['Schizophrenia · whispers', "It's getting incredibly terrifying to sit in my quiet lectures because the ambient background noise keeps morphing into distinct whispers, and I feel this constant, overwhelming paranoia that there are hidden internal patterns in the static on my screen."],
    ['Schizophrenia · fragmented', "My thoughts feel completely unraveled, disorganized, and fragmented, making me incredibly anxious because I can no longer tell what is real and what is just a vivid illusion inside my own head."],
    ['BPD · abandonment', "I am in absolute agony right now because my closest friend didn't reply to my text message for three hours, and I am completely convinced they secretly hate me and are planning to abandon me forever."],
    ['BPD · splitting', "One minute I love my teammates more than anything, and the next minute I feel this intense, burning splitting hatred toward them for leaving me alone. This constant fear of abandonment completely hijacks my day."],
    ['Anxiety · fight-or-flight', "My physical body is trapped in a constant state of fight-or-flight 24/7 and my mind refuses to rest. There isn't even a specific crisis happening right now, but I have this lingering, non-stop sense of impending doom."],
    ['Anxiety · worst case', "My muscles are incredibly tight, my stomach is completely in knots, and I am constantly overthinking every single worst-case scenario for my final project until I am paralyzed by chronic panic."],
    ['ADHD · fifty tabs', "I have been sitting at my desk for four hours trying to write this report, but I haven't gotten past the introduction. My mind feels like it has fifty browser tabs open at the same time and they are all playing different music."],
    ["ADHD · can't lock in", "I want to lock in so badly because the deadline is tomorrow, but my executive functioning is totally paralyzed and I just keep bouncing from one distraction to another while losing complete track of time."],
    ['OCD · the appliance', "I was halfway down the stairs when this terrifying thought hit me that I left my appliance plugged in, and even though I knew I checked it twice, I had to walk all the way back up to confirm."],
    ['OCD · door ritual', "I had to lock and unlock the door exactly four times in a specific sequence to make sure nothing bad happens today. If I don't perform the compulsion perfectly, the intense anxiety makes it impossible to focus."],
    ['Autism · decoding cues', "I spend the whole hour manually processing social cues, forcing myself to make standard eye contact, and trying to decipher the unwritten rules of the conversation so I don't look out of place."],
    ['Autism · too blunt', "People often tell me I sound monotone or overly blunt when I talk about my hyperfixations, but it feels like everyone else was given an implicit social instruction manual at birth while I am stuck simulating human interaction."],
    ['Depression · flatness', "It's not even that I am actively sad; it's just this heavy, crushing weight of complete emotional flatness where absolutely nothing brings me joy anymore. Going to lectures feels entirely pointless."],
    ['Depression · hollow', "I just spend days staring blankly at the ceiling feeling entirely hollow and disconnected from my own body. The mental paralysis is so deep that I've just completely given up on trying to fix it."],
    ['PTSD · dropped book', "A heavy textbook accidentally dropped off a shelf behind me and slammed onto the floor. In less than a second, my heart started racing uncontrollably, and I was entirely trapped in a vivid flashback reliving that accident."],
    ['PTSD · triggers', "These intense sensory triggers keep catching me off guard, waking me up in cold sweats from nightmares and making me feel constantly on edge even when I am physically safe inside a quiet university room."],
    ['Negation · expected to fail', "I cannot say I didn't expect to fail this class, but it's not like I'm completely broken over it."],
    ['Negation · not a disaster', "There is no way this custom code isn't a mess, though it's not a total disaster either."],
    ['Negation · not hopeless', "No one can deny the presentation wasn't a nightmare, but it's not a reason to lose hope."],
    ['Mixed · gorgeous nothing', 'They gave me a gorgeous, beautifully wrapped box containing absolutely nothing but disappointment.'],
    ['Mixed · pretty but unusable', 'The user interface looks incredibly gorgeous, sleek, and beautifully engineered, but it is totally unusable.'],
    ['Healthy · productive day', 'I had such a genuinely productive day studying at the campus library with my classmates this afternoon. We grabbed some warm coffee, sat down, and cleanly mapped out our full presentation slides.'],
];

export default function AssessmentPage() {
    const [text, setText] = useState('');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [server, setServer] = useState({ state: 'checking' });

    useEffect(() => {
        let alive = true;
        apiRequest('/health')
            .then((h) => alive && setServer({ state: 'ok', ...h }))
            .catch((e) => alive && setServer({ state: 'down', message: e.message }));
        return () => { alive = false; };
    }, []);

    const run = async (value) => {
        const t = (value ?? text).trim();
        if (!t || loading) return;
        setLoading(true); setError(null); setResult(null);
        try {
            setResult(await analyzeText(t));
        } catch (e) {
            setError(e.message);
        } finally {
            setLoading(false);
        }
    };

    const pick = (sentence) => { setText(sentence); run(sentence); };

    return (
        <div style={{ paddingTop: 120, minHeight: '100vh', paddingBottom: 80, position: 'relative' }}>
            <div className="orb orb-teal" style={{ width: 600, height: 600, top: -100, right: -200 }} />
            <div className="noise-overlay" />

            <div className="container" style={{ maxWidth: 820, position: 'relative', zIndex: 1, padding: '0 20px' }}>
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                    <h1 style={{ fontSize: 36, fontWeight: 900, marginBottom: 8 }}>AI Assessment Tool</h1>
                    <p style={{ color: '#94A3B8', marginBottom: 28 }}>
                        One emotion and nine independent clinical-pattern probabilities from a token-level self-attention model.
                    </p>

                    {server.state === 'down' && (
                        <div role="status" style={{ background: 'rgba(251,146,60,0.08)', border: '1px solid rgba(251,146,60,0.25)', color: '#FDBA74', borderRadius: 12, padding: 14, marginBottom: 18, fontSize: 14 }}>
                            ⚠️ {server.message}
                        </div>
                    )}
                    {server.state === 'ok' && server.untrained_disorders?.length > 0 && (
                        <div style={{ background: 'rgba(148,163,184,0.08)', border: '1px solid rgba(148,163,184,0.2)', color: '#94A3B8', borderRadius: 12, padding: 12, marginBottom: 18, fontSize: 13 }}>
                            ℹ️ The loaded model has no training data for: <b>{server.untrained_disorders.join(', ')}</b>. Those bars are disabled.
                        </div>
                    )}

                    <div className="glass-card" style={{ padding: 24 }}>
                        <textarea
                            aria-label="Text to analyse" value={text} onChange={(e) => setText(e.target.value)}
                            onKeyDown={(e) => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') run(); }}
                            maxLength={8000}
                            placeholder="Enter social media posts, journal entries, or any free-form text…"
                            style={{
                                width: '100%', minHeight: 150, background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-subtle)',
                                borderRadius: 12, padding: 16, color: '#F1F5F9', fontSize: 16, resize: 'vertical', outline: 'none', boxSizing: 'border-box'
                            }}
                        />
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 16, gap: 12, flexWrap: 'wrap' }}>
                            <div style={{ fontSize: 12, color: '#64748B' }}>{text.length}/8000 · Ctrl+Enter to analyse</div>
                            <button className="btn-primary" onClick={() => run()} disabled={!text.trim() || loading || server.state === 'down'}
                                style={{ opacity: !text.trim() || loading || server.state === 'down' ? 0.5 : 1 }}>
                                {loading ? 'Analyzing…' : 'Analyze Text'}
                            </button>
                        </div>
                    </div>

                    <div style={{ marginTop: 22 }}>
                        <div style={{ fontSize: 12, color: '#94A3B8', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 10 }}>
                            🧪 Stress-test examples — click to analyse
                        </div>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                            {EXAMPLES.map(([label, sentence]) => (
                                <button key={label} type="button" onClick={() => pick(sentence)} disabled={loading || server.state === 'down'}
                                    title={sentence}
                                    style={{
                                        fontSize: 12, padding: '6px 11px', borderRadius: 20, background: 'rgba(45,212,191,0.06)',
                                        border: '1px solid rgba(45,212,191,0.18)', color: '#94A3B8', cursor: loading ? 'wait' : 'pointer'
                                    }}>
                                    {label}
                                </button>
                            ))}
                        </div>
                    </div>
                </motion.div>

                {error && (
                    <div role="alert" style={{ marginTop: 24, background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.25)', borderRadius: 12, padding: 16, color: '#FECACA', fontSize: 14 }}>
                        {error}
                    </div>
                )}
                {result && <AnalysisView result={result} />}
                <div style={{ marginTop: 30, fontSize: 11, color: '#475569' }}>API: {API_BASE}</div>
            </div>
        </div>
    );
}