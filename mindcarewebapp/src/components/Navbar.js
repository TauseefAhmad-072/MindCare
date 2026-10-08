import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Brain, Menu, X, Video, Users, Sparkles, Camera } from 'lucide-react';

const LINKS = [
  { to: '/assessment', label: 'Assessment', icon: Sparkles },
  { to: '/session', label: 'Multi-Modal', icon: Video },
  { to: '/image', label: 'Facial Analysis', icon: Camera },
  { to: '/social', label: 'Social Scanner', icon: Users },
  { to: '/chat', label: 'AI Companion', icon: Brain },
];

export default function Navbar() {
  const [open, setOpen] = useState(false);
  const location = useLocation();

  return (
    <nav
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        zIndex: 100,
        background: 'rgba(6,11,20,0.85)',
        backdropFilter: 'blur(16px)',
        borderBottom: '1px solid rgba(255,255,255,0.06)',
      }}
    >
      <div
        className="container"
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          height: 68,
          padding: '0 20px',
        }}
      >
        <Link
          to="/"
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            textDecoration: 'none',
          }}
        >
          <div
            style={{
              width: 34,
              height: 34,
              borderRadius: 10,
              background: 'linear-gradient(135deg, #14B8A6, #8B5CF6)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Brain size={18} color="#fff" />
          </div>
          <span style={{ fontSize: 16, fontWeight: 800, color: '#F1F5F9' }}>
            MindCare
          </span>
        </Link>

        <div
          className="nav-links-desktop"
          style={{ display: 'flex', alignItems: 'center', gap: 4 }}
        >
          {LINKS.map(({ to, label, icon: Icon }) => {
            const active = location.pathname === to;

            return (
              <Link
                key={to}
                to={to}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  padding: '8px 14px',
                  borderRadius: 10,
                  fontSize: 13.5,
                  fontWeight: 600,
                  textDecoration: 'none',
                  color: active ? '#2DD4BF' : '#94A3B8',
                  background: active ? 'rgba(45,212,191,0.08)' : 'transparent',
                  transition: 'all 0.2s',
                }}
              >
                {Icon && <Icon size={14} />}
                {label}
              </Link>
            );
          })}
        </div>

        <button
          className="nav-toggle-mobile"
          onClick={() => setOpen(!open)}
          style={{
            background: 'transparent',
            border: 'none',
            color: '#F1F5F9',
            cursor: 'pointer',
            display: 'none',
          }}
        >
          {open ? <X size={22} /> : <Menu size={22} />}
        </button>
      </div>

      {open && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          style={{
            borderTop: '1px solid rgba(255,255,255,0.06)',
            padding: '10px 20px 16px',
            display: 'flex',
            flexDirection: 'column',
            gap: 4,
          }}
        >
          {LINKS.map(({ to, label, icon: Icon }) => {
            const active = location.pathname === to;
            return (
              <Link
                key={to}
                to={to}
                onClick={() => setOpen(false)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  padding: '10px 12px',
                  borderRadius: 8,
                  fontSize: 14,
                  fontWeight: 600,
                  textDecoration: 'none',
                  color: active ? '#2DD4BF' : '#CBD5E1',
                  background: active ? 'rgba(45,212,191,0.08)' : 'transparent',
                }}
              >
                {Icon && <Icon size={16} />}
                {label}
              </Link>
            );
          })}
        </motion.div>
      )}

      <style>{`
        @media (max-width: 860px) {
          .nav-links-desktop { display: none !important; }
          .nav-toggle-mobile { display: block !important; }
        }
      `}</style>
    </nav>
  );
}