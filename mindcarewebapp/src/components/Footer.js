import { Link } from 'react-router-dom';
import { Brain, Globe, MessageSquareText, Mail, Heart } from 'lucide-react';

export default function Footer() {
  const year = new Date().getFullYear();

  const sections = [
    {
      title: 'Product',
      links: [
        { label: 'Text Analysis', href: '/assessment' },
        { label: 'PHQ-9 Assessment', href: '/assessment' },
        { label: 'GAD-7 Assessment', href: '/assessment' },
        { label: 'Session History', href: '/assessment' },
      ],
    },
    {
      title: 'About',
      links: [
        { label: 'Our Research', href: '/#about' },
        { label: 'Model Architecture', href: '/#about' },
        { label: 'Dataset', href: '/#about' },
        { label: 'Team', href: '/#about' },
      ],
    },
    {
      title: 'Resources',
      links: [
        { label: 'Crisis Support', href: '#!' },
        { label: 'Umang: 0317-4288665', href: 'tel:03174288665' },
        { label: 'Documentation', href: '#!' },
        { label: 'FAST-NUCES FYP', href: '#!' },
      ],
    },
  ];

  return (
    <footer style={{ background: '#060B14', borderTop: '1px solid rgba(255,255,255,0.06)', padding: '60px 20px 28px' }}>
      <div className="container">
        <div style={{ display: 'grid', gridTemplateColumns: '1.6fr 1fr 1fr 1fr', gap: 40, marginBottom: 48 }}>
          {/* Brand column */}
          <div>
            <Link to="/" style={{ display: 'flex', alignItems: 'center', gap: 10, textDecoration: 'none', marginBottom: 16 }}>
              <div style={{
                width: 36, height: 36, borderRadius: 10,
                background: 'linear-gradient(135deg, #14B8A6, #8B5CF6)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <Brain size={18} color="#fff" />
              </div>
              <span style={{ fontSize: 18, fontWeight: 900, color: '#F1F5F9' }}>MindCare</span>
            </Link>
            <p style={{ fontSize: 13, color: '#64748B', lineHeight: 1.7, maxWidth: 260 }}>
              Semantic & temporal analysis of social media text for student mental health assessment. Built at FAST-NUCES.
            </p>
            <div style={{ display: 'flex', gap: 10, marginTop: 20 }}>
              {[Globe, MessageSquareText, Mail].map((Icon, i) => (
                <a
                  key={i}
                  href="#!"
                  style={{
                    width: 34, height: 34, borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center',
                    background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.08)',
                    color: '#64748B', transition: 'all 0.2s', cursor: 'pointer',
                  }}
                  onMouseEnter={e => { e.currentTarget.style.color = '#2DD4BF'; e.currentTarget.style.borderColor = 'rgba(45,212,191,0.3)'; e.currentTarget.style.background = 'rgba(45,212,191,0.06)'; }}
                  onMouseLeave={e => { e.currentTarget.style.color = '#64748B'; e.currentTarget.style.borderColor = 'rgba(255,255,255,0.08)'; e.currentTarget.style.background = 'rgba(255,255,255,0.05)'; }}
                >
                  <Icon size={15} />
                </a>
              ))}
            </div>
          </div>

          {/* Link columns */}
          {sections.map(({ title, links }) => (
            <div key={title}>
              <div style={{ fontSize: 12, fontWeight: 700, color: '#F1F5F9', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 16 }}>
                {title}
              </div>
              <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 10 }}>
                {links.map(({ label, href }) => (
                  <li key={label}>
                    <a
                      href={href}
                      style={{ fontSize: 13, color: '#64748B', textDecoration: 'none', transition: 'color 0.2s' }}
                      onMouseEnter={e => e.currentTarget.style.color = '#2DD4BF'}
                      onMouseLeave={e => e.currentTarget.style.color = '#64748B'}
                    >
                      {label}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div style={{ borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: 24, display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
          <div style={{ fontSize: 12, color: '#334155' }}>
            © {year} MindCare — FAST-NUCES FYP · Built with <Heart size={11} style={{ display: 'inline', color: '#FB7185', verticalAlign: 'middle' }} /> by Tauseef, Ahyan & Hamza
          </div>
          <div style={{ fontSize: 12, color: '#334155' }}>
            If you're in crisis, call <strong style={{ color: '#FB7185' }}>Umang: 0317-4288665</strong>
          </div>
        </div>
      </div>

      <style>{`
        @media (max-width: 768px) {
          footer .container > div:first-child { grid-template-columns: 1fr 1fr !important; }
          footer .container > div:first-child > div:first-child { grid-column: 1 / -1; }
          footer .container > div:last-child { flex-direction: column; text-align: center; }
        }
      `}</style>
    </footer>
  );
}
