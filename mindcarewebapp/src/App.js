/*
App.js — Unified React Application Router
Wires the frontend layout shell to the consolidated 7-page multi-modal 
and core diagnostic evaluation analysis architecture workspace.
*/
import { Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';

// ── Consolidated 7-Page Architecture Imports (FIXED PATHS)
import LandingPage from './pages/LandingPage';
import AssessmentPage from './pages/AssessmentPage';
import ChatPage from './pages/ChatPage';
import SessionPage from './pages/SessionPage';
import ImagePage from './pages/ImagePage';

// 🚀 CRITICAL FIX: Centralizing platforms to our new unified social component
import SocialScraperPage from './pages/SocialScraperPage.js';

// 🚀 CASE-SENSITIVITY FIX: Pointing cleanly to your disk path configuration (lowercase 'p')
import AnalysisViewPage from './pages/AnalysisViewPage';

export default function App() {
  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Navbar />
      <main style={{ flex: 1 }}>
        <Routes>
          {/* 🏡 Core App Entry Gateway */}
          <Route path="/" element={<LandingPage />} />

          {/* 📝 Initial Testing & Screening Surveys */}
          <Route path="/assessment" element={<AssessmentPage />} />

          {/* 💬 Conversational LangChain RAG Chatbot Hub */}
          <Route path="/chat" element={<ChatPage />} />

          {/* 🎥 Multi-Modal Media Sessions (Video & Audio Tracks Isolation) */}
          <Route path="/session" element={<SessionPage />} />

          {/* 🖼️ Visual/Image Data Asset Ingestion */}
          <Route path="/image" element={<ImagePage />} />

          {/* 🌐 Consolidated Social Footprint Scanner Route (Reddit + X Integration) */}
          <Route path="/social" element={<SocialScraperPage />} />

          {/* 🔬 Deep Multi-Task Attention Network Matrix Analytics Trace */}
          <Route path="/analysis-view" element={<AnalysisViewPage />} />
        </Routes>
      </main>
      <Footer />
    </div>
  );
}
