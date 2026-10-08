import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Brain, Sparkles, Shield, Activity, BarChart2, ChevronRight, Zap, Lock, ArrowRight, CheckCircle2, Star } from 'lucide-react';
import { apiRequest, EMOTION_META, DISORDER_LABELS } from '../pages/AnalysisViewPage';

const fadeUp = { hidden: { opacity: 0, y: 30 }, show: { opacity: 1, y: 0, transition: { duration: 0.6, ease: [0.4, 0, 0.2, 1] } } };
const stagger = { hidden: {}, show: { transition: { staggerChildren: 0.12 } } };

function hexToRgbStr(hex) {
  return [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)).join(',');
}

function HeroSection({ onTry, navigate }) {
  return (
    <section style={{
      minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center',
      background: 'radial-gradient(ellipse 80% 60% at 50% 0%, rgba(20,184,166,0.12) 0%, transparent 60%), radial-gradient(ellipse 60% 50% at 80% 50%, rgba(139,92,246,0.1) 0%, transparent 60%), var(--bg-900)',
      padding: '120px 20px 80px', position: 'relative', overflow: 'hidden', textAlign: 'center',
    }}>
      <div className="orb orb-teal" style={{ width: 500, height: 500, top: -100, left: '50%', transform: 'translateX(-50%)', opacity: 0.3 }} />
      <div className="orb orb-violet" style={{ width: 400, height: 400, bottom: -80, right: -100, opacity: 0.2 }} />
      <div className="noise-overlay" />

      <div className="container" style={{ position: 'relative', zIndex: 1 }}>
        <motion.div initial="hidden" animate="show" variants={stagger}>
          <motion.div variants={fadeUp}>
            <span className="label-tag" style={{ marginBottom: 24, display: 'inline-flex' }}>
              <Sparkles size={11} /> FAST-NUCES FYP · AI-Powered Assessment
            </span>
          </motion.div>

          <motion.h1 variants={fadeUp} style={{ fontSize: 'clamp(42px, 7vw, 80px)', fontWeight: 900, letterSpacing: '-0.04em', lineHeight: 1.05, marginBottom: 24, color: '#F1F5F9' }}>
            Mental Health <span className="text-gradient">AI Analysis</span><br />for Students
          </motion.h1>

          <motion.p variants={fadeUp} style={{ fontSize: 'clamp(16px, 2vw, 20px)', color: '#94A3B8', maxWidth: 580, margin: '0 auto 40px', lineHeight: 1.7 }}>
            Your companion for emotional well-being. A custom token-level self-attention model reads your text and reports one emotion plus nine independent clinical-pattern probabilities.
          </motion.p>

          <motion.div variants={fadeUp} style={{ display: 'flex', gap: 14, justifyContent: 'center', flexWrap: 'wrap' }}>
            <button className="btn-primary" onClick={onTry} style={{ fontSize: 15, padding: '15px 32px' }}>
              <Brain size={18} /> Start Free Assessment
            </button>
            <button className="btn-secondary" onClick={() => navigate('/chat')} style={{ fontSize: 15 }}>
              Talk to MindCare <ChevronRight size={16} />
            </button>
          </motion.div>
        </motion.div>

        {/* Illustrative preview (static example — not a live result) */}
        <motion.div initial={{ opacity: 0, y: 40 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5, duration: 0.8 }}
          style={{ marginTop: 64, animation: 'float 4s ease-in-out infinite' }}>
          <div className="glass-card" style={{
            maxWidth: 540, margin: '0 auto', padding: 24, textAlign: 'left', background: 'rgba(15,23,42,0.7)', backdropFilter: 'blur(24px)',
            border: '1px solid rgba(45,212,191,0.2)', boxShadow: '0 0 60px rgba(20,184,166,0.15), 0 40px 80px rgba(0,0,0,0.5)',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
              <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#22C55E', boxShadow: '0 0 8px #22C55E' }} />
              <span style={{ fontSize: 11, color: '#64748B', fontWeight: 600 }}>EXAMPLE OUTPUT · TOKEN-ATTENTION DUAL-HEAD MODEL</span>
            </div>
            <div style={{ background: 'rgba(255,255,255,0.03)', borderRadius: 10, padding: '12px 14px', fontSize: 13, color: '#94A3B8', lineHeight: 1.6, border: '1px solid rgba(255,255,255,0.05)', marginBottom: 16 }}>
              "A textbook slammed onto the floor and my <mark style={{ background: 'rgba(167,139,250,0.2)', color: '#C4B5FD', padding: '1px 3px', borderRadius: 3 }}>heart started racing</mark>, trapped in a vivid <mark style={{ background: 'rgba(167,139,250,0.2)', color: '#C4B5FD', padding: '1px 3px', borderRadius: 3 }}>flashback</mark>…"
            </div>
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8, padding: '6px 14px', borderRadius: 99, background: EMOTION_META.anxious.color, color: '#fff', fontWeight: 800, fontSize: 14, marginBottom: 16 }}>
              {EMOTION_META.anxious.icon} ANXIOUS
            </div>
            {[['PTSD', 91], ['Anxiety Disorder', 34], ['Depression', 12]].map(([n, v]) => (
              <div key={n} style={{ marginBottom: 9 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#CBD5E1', marginBottom: 4 }}><span>{n}</span><span>{v}%</span></div>
                <div style={{ height: 6, background: 'rgba(255,255,255,0.05)', borderRadius: 99, overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${v}%`, background: v > 60 ? '#EF4444' : v > 30 ? '#F59E0B' : '#14B8A6', borderRadius: 99 }} />
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}

function FeaturesSection() {
  const features = [
    { icon: Brain, color: '#2DD4BF', title: 'Token-Level Self-Attention', desc: 'A scratch-built attention layer weighs every word against every other word, so context and negation matter.' },
    { icon: BarChart2, color: '#A78BFA', title: 'Dual-Head Output', desc: 'One head picks the single most likely emotion; the other scores nine clinical patterns independently from 0 to 100%.' },
    { icon: Activity, color: '#34D399', title: 'Transparent Results', desc: 'See the cleaned tokens, vocabulary ids and every class score, not just a label.' },
    { icon: Shield, color: '#FB7185', title: 'Practical Guidance', desc: 'Each result comes with supportive, plain-language next steps matched to the emotion or pattern detected.' },
    { icon: Zap, color: '#FBBF24', title: 'Safety Notice', desc: 'Language about self-harm triggers an immediate banner with crisis-line information.' },
    { icon: Lock, color: '#60A5FA', title: 'Privacy-Minded', desc: 'The analysis endpoint does not store your text. The chat companion may use an external AI service, so avoid sharing identifying details.' },
  ];
  return (
    <section id="features" className="section" style={{ background: 'var(--bg-800)' }}>
      <div className="orb orb-violet" style={{ width: 600, height: 600, top: -100, right: -150, opacity: 0.12, position: 'absolute' }} />
      <div className="noise-overlay" />
      <div className="container" style={{ position: 'relative', zIndex: 1 }}>
        <motion.div className="section-header" initial="hidden" whileInView="show" viewport={{ once: true, margin: '-80px' }} variants={stagger}>
          <motion.span variants={fadeUp} className="label-tag"><Star size={11} /> Core Capabilities</motion.span>
          <motion.h2 variants={fadeUp} style={{ marginTop: 12 }}>Built for <span className="text-gradient">your well-being</span></motion.h2>
          <motion.p variants={fadeUp}>A focused toolkit for understanding how your words read, with care for privacy and honesty about limits.</motion.p>
        </motion.div>
        <motion.div initial="hidden" whileInView="show" viewport={{ once: true, margin: '-60px' }} variants={stagger}
          style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 20 }}>
          {features.map(({ icon: Icon, color, title, desc }) => (
            <motion.div key={title} variants={fadeUp} className="glass-card" style={{ padding: 28 }}>
              <div style={{ width: 48, height: 48, borderRadius: 14, marginBottom: 20, background: `rgba(${hexToRgbStr(color)}, 0.1)`, border: `1px solid rgba(${hexToRgbStr(color)}, 0.2)`, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Icon size={22} color={color} />
              </div>
              <div style={{ fontSize: 16, fontWeight: 700, color: '#F1F5F9', marginBottom: 10 }}>{title}</div>
              <div style={{ fontSize: 13, color: '#64748B', lineHeight: 1.7 }}>{desc}</div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}

function HowItWorksSection() {
  const steps = [
    { num: '01', color: '#2DD4BF', title: 'Input Your Text', desc: 'Paste a post, journal entry or any free-form text, or pick one of the built-in stress-test sentences.' },
    { num: '02', color: '#A78BFA', title: 'Clean & Tokenise', desc: 'Text is cleaned, mapped to vocabulary ids and padded to a fixed 50-token window.' },
    { num: '03', color: '#34D399', title: 'Self-Attention', desc: 'Scaled dot-product attention (key size 32) contextualises each token; max-pooling keeps the strongest signals.' },
    { num: '04', color: '#FBBF24', title: 'Two Heads, One Result', desc: 'A softmax head chooses one emotion; a sigmoid head scores nine clinical patterns independently. Guidance and any safety notice follow.' },
  ];
  return (
    <section id="how-it-works" className="section" style={{ background: 'var(--bg-900)' }}>
      <div className="orb orb-teal" style={{ width: 500, height: 500, bottom: -100, left: -100, opacity: 0.1, position: 'absolute' }} />
      <div className="noise-overlay" />
      <div className="container" style={{ position: 'relative', zIndex: 1 }}>
        <motion.div className="section-header" initial="hidden" whileInView="show" viewport={{ once: true, margin: '-80px' }} variants={stagger}>
          <motion.span variants={fadeUp} className="label-tag"><Zap size={11} /> Process</motion.span>
          <motion.h2 variants={fadeUp} style={{ marginTop: 12 }}>How it <span className="text-gradient">works</span></motion.h2>
          <motion.p variants={fadeUp}>From raw text to emotion and pattern scores in a fraction of a second.</motion.p>
        </motion.div>
        <div style={{ position: 'relative', maxWidth: 760, margin: '0 auto' }}>
          <div style={{ position: 'absolute', left: 36, top: 0, bottom: 0, width: 2, background: 'linear-gradient(180deg, #14B8A6, #8B5CF6, #34D399, #FBBF24)', opacity: 0.3, borderRadius: 2 }} />
          <motion.div initial="hidden" whileInView="show" viewport={{ once: true, margin: '-60px' }} variants={stagger} style={{ display: 'flex', flexDirection: 'column', gap: 28 }}>
            {steps.map(({ num, color, title, desc }) => (
              <motion.div key={num} variants={fadeUp} style={{ display: 'flex', gap: 24, alignItems: 'flex-start', paddingLeft: 12 }}>
                <div style={{ width: 52, height: 52, borderRadius: 16, flexShrink: 0, zIndex: 1, background: `rgba(${hexToRgbStr(color)}, 0.12)`, border: `2px solid rgba(${hexToRgbStr(color)}, 0.3)`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 13, fontWeight: 900, color }}>
                  {num}
                </div>
                <div className="glass-card" style={{ flex: 1, padding: '20px 24px', borderRadius: 16 }}>
                  <div style={{ fontSize: 16, fontWeight: 700, color: '#F1F5F9', marginBottom: 6 }}>{title}</div>
                  <div style={{ fontSize: 13, color: '#64748B', lineHeight: 1.7 }}>{desc}</div>
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </div>
    </section>
  );
}

const PATTERN_COLORS = ['#2DD4BF', '#A78BFA', '#34D399', '#FB7185', '#FBBF24', '#60A5FA', '#F97316', '#EF4444', '#84CC16'];

function OutputsSection() {
  const [untrained, setUntrained] = useState(null);       // null until /health answers
  useEffect(() => {
    let alive = true;
    apiRequest('/health').then((h) => alive && setUntrained(h.untrained_disorders || [])).catch(() => { });
    return () => { alive = false; };
  }, []);

  return (
    <section id="about" className="section" style={{ background: 'var(--bg-800)' }}>
      <div className="noise-overlay" />
      <div className="container" style={{ position: 'relative', zIndex: 1 }}>
        <motion.div className="section-header" initial="hidden" whileInView="show" viewport={{ once: true, margin: '-80px' }} variants={stagger}>
          <motion.span variants={fadeUp} className="label-tag"><Activity size={11} /> What it reports</motion.span>
          <motion.h2 variants={fadeUp} style={{ marginTop: 12 }}>6 Emotions · <span className="text-gradient">9 Clinical Patterns</span></motion.h2>
          <motion.p variants={fadeUp}>Results describe how the text reads. They are screening signals, not diagnoses.</motion.p>
        </motion.div>

        <div style={{ textAlign: 'center', fontSize: 12, color: '#94A3B8', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 12 }}>Emotion head (one winner)</div>
        <motion.div initial="hidden" whileInView="show" viewport={{ once: true }} variants={stagger}
          style={{ display: 'flex', flexWrap: 'wrap', gap: 12, justifyContent: 'center', marginBottom: 40 }}>
          {Object.entries(EMOTION_META).map(([name, { icon, color }]) => (
            <motion.div key={name} variants={fadeUp} whileHover={{ scale: 1.05 }}
              style={{ padding: '10px 20px', borderRadius: 99, background: `${color}22`, border: `1px solid ${color}55`, color: '#F1F5F9', fontWeight: 700 }}>
              {icon} {name[0].toUpperCase() + name.slice(1)}
            </motion.div>
          ))}
        </motion.div>

        <div style={{ textAlign: 'center', fontSize: 12, color: '#94A3B8', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 12 }}>Disorder head (nine independent probabilities)</div>
        <motion.div initial="hidden" whileInView="show" viewport={{ once: true, margin: '-60px' }} variants={stagger}
          style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 14 }}>
          {Object.entries(DISORDER_LABELS).map(([key, label], i) => {
            const color = PATTERN_COLORS[i % PATTERN_COLORS.length];
            const off = untrained?.includes(key);
            return (
              <motion.div key={key} variants={fadeUp} whileHover={{ scale: 1.03 }}
                style={{ background: `rgba(${hexToRgbStr(color)},0.08)`, border: `1px solid ${color}33`, borderRadius: 16, padding: '18px 20px', opacity: off ? 0.45 : 1 }}>
                <div style={{ fontSize: 15, fontWeight: 700, color: '#F1F5F9', marginBottom: 4 }}>{label}</div>
                <div style={{ fontSize: 11, color: '#64748B' }}>{off ? 'Not trained on this server yet' : '0–100% independent score'}</div>
              </motion.div>
            );
          })}
        </motion.div>
      </div>
    </section>
  );
}

function CTASection({ onTry }) {
  return (
    <section className="section" style={{ background: 'var(--bg-900)', padding: '80px 20px' }}>
      <div className="container" style={{ textAlign: 'center', position: 'relative' }}>
        <div style={{ maxWidth: 680, margin: '0 auto', padding: '60px 40px', background: 'linear-gradient(135deg, rgba(20,184,166,0.08) 0%, rgba(139,92,246,0.08) 100%)', border: '1px solid rgba(45,212,191,0.2)', borderRadius: 28, boxShadow: '0 0 80px rgba(20,184,166,0.08)' }}>
          <motion.div initial="hidden" whileInView="show" viewport={{ once: true }} variants={stagger}>
            <motion.div variants={fadeUp} style={{ fontSize: 40, marginBottom: 12 }}>🧠</motion.div>
            <motion.h2 variants={fadeUp} style={{ fontSize: 'clamp(28px, 4vw, 40px)', fontWeight: 900, letterSpacing: '-0.03em', marginBottom: 14, color: '#F1F5F9' }}>Start your assessment</motion.h2>
            <motion.p variants={fadeUp} style={{ fontSize: 16, color: '#64748B', marginBottom: 32, lineHeight: 1.7 }}>
              No account needed. Paste some text and see the emotion and clinical-pattern scores right away.
            </motion.p>
            <motion.div variants={fadeUp} style={{ display: 'flex', gap: 14, justifyContent: 'center', flexWrap: 'wrap' }}>
              <button className="btn-primary" onClick={onTry} style={{ fontSize: 15, padding: '15px 32px' }}>
                <Brain size={18} /> Try Now — It's Free <ArrowRight size={16} />
              </button>
            </motion.div>
            <motion.div variants={fadeUp} style={{ display: 'flex', gap: 20, justifyContent: 'center', marginTop: 24, flexWrap: 'wrap' }}>
              {['No sign-up required', 'Instant results', 'Not a medical diagnosis'].map((f) => (
                <div key={f} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: '#64748B' }}>
                  <CheckCircle2 size={13} color="#2DD4BF" /> {f}
                </div>
              ))}
            </motion.div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}

export default function LandingPage() {
  const navigate = useNavigate();
  const goToApp = () => navigate('/assessment');
  return (
    <div>
      <HeroSection onTry={goToApp} navigate={navigate} />
      <FeaturesSection />
      <HowItWorksSection />
      <OutputsSection />
      <CTASection onTry={goToApp} />
    </div>
  );
}