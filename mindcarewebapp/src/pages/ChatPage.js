import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Bot, Sparkles, AlertCircle, MessageSquare, Trash2, Shield, Heart } from 'lucide-react';
import { apiRequest, EmotionBadge, DisorderBars, RecommendationsCard, SafetyAlert, DISORDER_LABELS } from '../pages/AnalysisViewPage';

const QUICK_SUGGESTIONS = [
  "I'm feeling really stressed lately",
  'What should I do to feel better?',
  'I feel lonely and disconnected',
  'How can I manage my anxiety?',
  "I'm overwhelmed with my studies",
];

const GREETING = "Hello! I'm your MindCare Companion. I'm here to listen and support you. How are you feeling today?";
const newSessionId = () => `web-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
const timeOf = (d) => d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

function BotMessage({ msg }) {
  return (
    <motion.div initial={{ opacity: 0, y: 10, scale: 0.97 }} animate={{ opacity: 1, y: 0, scale: 1 }}
      style={{ alignSelf: 'flex-start', maxWidth: '85%', display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{
        padding: '12px 16px', borderRadius: '18px 18px 18px 2px',
        background: msg.isError ? 'rgba(239,68,68,0.12)' : 'rgba(30,41,59,0.85)',
        color: msg.isError ? '#FECACA' : '#CBD5E1', fontSize: 14, lineHeight: 1.65,
        border: `1px solid ${msg.isError ? 'rgba(239,68,68,0.3)' : 'rgba(255,255,255,0.06)'}`, whiteSpace: 'pre-wrap',
      }}>
        {msg.text}
      </div>
      {msg.safetyNotice && <SafetyAlert notice={msg.safetyNotice} />}
      {msg.recommendations && msg.recommendations.steps && <RecommendationsCard recs={msg.recommendations} />}
      <div style={{ fontSize: 10, color: '#475569', display: 'flex', alignItems: 'center', gap: 4 }}>
        <Sparkles size={10} color="#2DD4BF" />{timeOf(msg.time)}
      </div>
    </motion.div>
  );
}

export default function ChatPage() {
  const [messages, setMessages] = useState([{ id: '1', type: 'bot', text: GREETING, time: new Date() }]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [analysis, setAnalysis] = useState(null);
  const scrollRef = useRef(null);
  const sessionRef = useRef(newSessionId());

  useEffect(() => {
    scrollRef.current?.scrollIntoView?.({ behavior: 'smooth' });
  }, [messages, isTyping]);

  const sendMessage = async (raw) => {
    const text = (raw || '').trim();
    if (!text || isTyping) return;
    setMessages((prev) => [...prev, { id: `u${Date.now()}`, type: 'user', text, time: new Date() }]);
    setInput('');
    setIsTyping(true);
    try {
      const data = await apiRequest('/chat', { message: text, session_id: sessionRef.current });
      setMessages((prev) => [...prev, {
        id: `b${Date.now()}`, type: 'bot', text: data.reply, time: new Date(),
        recommendations: data.recommendations, safetyNotice: data.safety_notice,
      }]);
      if (data.analysis) setAnalysis(data.analysis);
    } catch (e) {
      setMessages((prev) => [...prev, {
        id: `e${Date.now()}`, type: 'bot', isError: true, time: new Date(),
        text: `I'm having trouble connecting right now. ${e.message}`,
      }]);
    } finally {
      setIsTyping(false);
    }
  };

  const clearChat = async () => {
    if (!window.confirm('Clear conversation?')) return;
    const old = sessionRef.current;
    sessionRef.current = newSessionId();
    setMessages([{ id: '1', type: 'bot', text: 'Hello again! How can I support you?', time: new Date() }]);
    setAnalysis(null);
    try { await apiRequest('/chat/reset', { session_id: old }); } catch { /* server memory will expire on its own */ }
  };

  const top = analysis?.top_disorder && analysis.top_disorder.probability >= 0.5 ? analysis.top_disorder : null;

  return (
    <div style={{
      minHeight: '100vh', paddingTop: 100, paddingBottom: 40, display: 'flex', flexDirection: 'column', alignItems: 'center',
      background: 'radial-gradient(circle at 50% 0%, rgba(20,184,166,0.1) 0%, transparent 50%), var(--bg-900)',
    }}>
      <div className="noise-overlay" />
      <div className="container" style={{ maxWidth: 980, position: 'relative', zIndex: 1, width: '100%' }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) 300px', gap: 20, height: '80vh' }}>

          {/* Chat window */}
          <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden', border: '1px solid rgba(255,255,255,0.05)', background: 'rgba(15,23,42,0.6)' }}>
            <div style={{ padding: '14px 20px', borderBottom: '1px solid rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: 'rgba(255,255,255,0.02)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <div style={{ width: 38, height: 38, borderRadius: 12, background: 'linear-gradient(135deg, #14B8A6, #8B5CF6)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Bot size={20} color="#fff" />
                </div>
                <div>
                  <div style={{ fontSize: 15, fontWeight: 700, color: '#F1F5F9' }}>MindCare Companion</div>
                  <div style={{ fontSize: 11, color: '#2DD4BF', fontWeight: 600 }}>
                    <span style={{ display: 'inline-block', width: 6, height: 6, borderRadius: '50%', background: '#22C55E', marginRight: 5, boxShadow: '0 0 6px #22C55E' }} />
                    Always here to listen
                  </div>
                </div>
              </div>
              <button onClick={clearChat} aria-label="Clear chat" title="Clear Chat" style={{ background: 'transparent', border: 'none', color: '#475569', cursor: 'pointer' }}>
                <Trash2 size={17} />
              </button>
            </div>

            <div style={{ flex: 1, overflowY: 'auto', padding: 20, display: 'flex', flexDirection: 'column', gap: 16 }}>
              <AnimatePresence initial={false}>
                {messages.map((msg) => (msg.type === 'user' ? (
                  <motion.div key={msg.id} initial={{ opacity: 0, y: 10, scale: 0.97 }} animate={{ opacity: 1, y: 0, scale: 1 }}
                    style={{ alignSelf: 'flex-end', maxWidth: '80%', display: 'flex', flexDirection: 'column', alignItems: 'flex-end' }}>
                    <div style={{ padding: '12px 16px', borderRadius: '18px 18px 2px 18px', background: 'linear-gradient(135deg, #0D9488, #0F766E)', color: '#fff', fontSize: 14, lineHeight: 1.5, boxShadow: '0 4px 12px rgba(13,148,136,0.2)', whiteSpace: 'pre-wrap' }}>
                      {msg.text}
                    </div>
                    <div style={{ fontSize: 10, color: '#475569', marginTop: 4 }}>{timeOf(msg.time)}</div>
                  </motion.div>
                ) : <BotMessage key={msg.id} msg={msg} />))}
                {isTyping && (
                  <motion.div key="typing" initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ alignSelf: 'flex-start' }} aria-label="Companion is typing">
                    <div className="typing-indicator"><span /><span /><span /></div>
                  </motion.div>
                )}
              </AnimatePresence>
              <div ref={scrollRef} />
            </div>

            <div style={{ padding: '8px 16px', display: 'flex', gap: 8, flexWrap: 'wrap', borderTop: '1px solid rgba(255,255,255,0.04)' }}>
              {QUICK_SUGGESTIONS.map((s) => (
                <button key={s} type="button" onClick={() => sendMessage(s)} disabled={isTyping}
                  style={{ fontSize: 11, padding: '5px 10px', borderRadius: 20, background: 'rgba(45,212,191,0.06)', border: '1px solid rgba(45,212,191,0.15)', color: '#64748B', cursor: 'pointer' }}>
                  {s}
                </button>
              ))}
            </div>

            <form onSubmit={(e) => { e.preventDefault(); sendMessage(input); }}
              style={{ padding: '14px 16px', background: 'rgba(255,255,255,0.02)', borderTop: '1px solid rgba(255,255,255,0.05)', display: 'flex', gap: 10 }}>
              <input type="text" value={input} onChange={(e) => setInput(e.target.value)} maxLength={4000}
                aria-label="Message" placeholder="Share how you're feeling, or ask for advice…"
                style={{ flex: 1, background: 'rgba(15,23,42,0.8)', border: '1px solid rgba(255,255,255,0.08)', borderRadius: 12, padding: '11px 15px', color: '#F1F5F9', fontSize: 14, outline: 'none' }} />
              <button type="submit" aria-label="Send" disabled={!input.trim() || isTyping} className="btn-primary"
                style={{ padding: '0 18px', borderRadius: 12, opacity: !input.trim() || isTyping ? 0.5 : 1 }}>
                <Send size={17} />
              </button>
            </form>
          </div>

          {/* Sidebar */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16, overflowY: 'auto' }}>
            <div className="glass-card" style={{ padding: 22, border: '1px solid rgba(45,212,191,0.15)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
                <Sparkles size={16} color="#2DD4BF" />
                <span style={{ fontSize: 13, fontWeight: 700, color: '#F1F5F9' }}>Mood Analysis</span>
              </div>
              {analysis ? (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} data-testid="chat-analysis">
                  <div style={{ fontSize: 10, color: '#64748B', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 8 }}>Detected emotion</div>
                  <EmotionBadge result={analysis} size="sm" />
                  <div style={{ fontSize: 10, color: '#64748B', textTransform: 'uppercase', letterSpacing: '0.05em', margin: '16px 0 8px' }}>Top patterns</div>
                  <DisorderBars result={analysis} limit={3} />
                  {top && (
                    <div style={{ marginTop: 14, padding: '10px 12px', borderRadius: 10, background: 'rgba(251,146,60,0.08)', border: '1px solid rgba(251,146,60,0.2)', fontSize: 11, color: '#FB923C', lineHeight: 1.5 }}>
                      {analysis.urgent ? <AlertCircle size={12} style={{ display: 'inline', marginRight: 4 }} /> : <Heart size={12} style={{ display: 'inline', marginRight: 4 }} />}
                      Your words resemble <b>{DISORDER_LABELS[top.name] || top.name}</b> discussions. This is a signal, not a diagnosis.
                    </div>
                  )}
                </motion.div>
              ) : (
                <div style={{ textAlign: 'center', padding: '16px 0' }}>
                  <MessageSquare size={28} color="#1E293B" style={{ marginBottom: 10 }} />
                  <div style={{ fontSize: 11, color: '#475569' }}>Start chatting to see your mood analysis</div>
                </div>
              )}
            </div>

            <div className="glass-card" style={{ padding: 18, background: 'rgba(20,184,166,0.04)', border: '1px solid rgba(20,184,166,0.1)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
                <Shield size={14} color="#2DD4BF" />
                <span style={{ fontSize: 12, fontWeight: 700, color: '#F1F5F9' }}>Your privacy</span>
              </div>
              <p style={{ fontSize: 11, color: '#64748B', lineHeight: 1.6, margin: 0 }}>
                Mood analysis runs on the MindCare server and your messages are not saved by it. The companion's replies may be generated by an external AI service, so avoid sharing names or personal identifiers.
              </p>
            </div>

            <div className="glass-card" style={{ padding: 18, background: 'rgba(139,92,246,0.04)', border: '1px solid rgba(139,92,246,0.12)' }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: '#A78BFA', marginBottom: 8 }}>💬 Getting Advice</div>
              <p style={{ fontSize: 11, color: '#64748B', lineHeight: 1.6, margin: 0 }}>
                Ask things like <em style={{ color: '#94A3B8' }}>"What should I do?"</em> or <em style={{ color: '#94A3B8' }}>"How can I feel better?"</em> to receive personalised suggestions.
              </p>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        .typing-indicator { display:flex; padding:12px 16px; background:rgba(30,41,59,0.85); border-radius:18px 18px 18px 2px; gap:4px; }
        .typing-indicator span { width:6px; height:6px; background:#475569; border-radius:50%; animation:mc-bounce 1.4s infinite ease-in-out both; }
        .typing-indicator span:nth-child(1){ animation-delay:-0.32s; } .typing-indicator span:nth-child(2){ animation-delay:-0.16s; }
        @keyframes mc-bounce { 0%,80%,100%{ transform:scale(0);} 40%{ transform:scale(1);} }
        @media (max-width: 820px){ .container > div[style*="grid-template-columns"]{ grid-template-columns:1fr !important; height:auto !important; } }
      `}</style>
    </div>
  );
}