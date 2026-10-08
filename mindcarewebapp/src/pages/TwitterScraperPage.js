import React, { useState } from "react";
import { motion } from "framer-motion";
import { Search, Loader2, AlertTriangle, X, ExternalLink } from "lucide-react";
import { apiRequest, PredictionSummary, EMOTION_META } from "../pages/AnalysisViewPage";

export default function TwitterScraperPage() {
  const [target, setTarget] = useState("");
  const [limit, setLimit] = useState(10);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleScrape = async (e) => {
    e.preventDefault();
    if (!target.trim() || isLoading) return;
    setIsLoading(true);
    setError(null);
    setResults(null);
    try {
      const n = Math.min(Math.max(parseInt(limit, 10) || 10, 1), 50);
      const data = await apiRequest("/scrape/twitter", { target: target.trim(), limit: n });
      setResults(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const counts = results?.summary?.emotion_counts || {};

  return (
    <div className="container" style={{ paddingTop: 140, paddingBottom: 100 }}>
      <style>{`.spin{animation:mc-spin 1s linear infinite}@keyframes mc-spin{to{transform:rotate(360deg)}}`}</style>
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} style={{ maxWidth: 800, margin: "0 auto" }}>
        <div style={{ textAlign: "center", marginBottom: 40 }}>
          <div style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", width: 60, height: 60, borderRadius: 20, background: "linear-gradient(135deg, #1DA1F2, #0ea5e9)", marginBottom: 20 }}>
            <X size={30} color="#fff" />
          </div>
          <h1 style={{ fontSize: 48, fontWeight: 800, color: "#F8FAFC", marginBottom: 16 }}>
            X (Twitter) <span className="gradient-text">Analysis</span>
          </h1>
          <p style={{ fontSize: 18, color: "#94A3B8" }}>Scrape a user's recent tweets and run each one through the MindCare model. Note: scraping X can take 15–30 seconds.</p>
        </div>

        <div className="glass-card" style={{ padding: 30, marginBottom: 40 }}>
          <form onSubmit={handleScrape} style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
            <div style={{ flex: "1 1 300px", position: "relative" }}>
              <div style={{ position: "absolute", left: 16, top: 16, color: "#64748B" }}><Search size={20} /></div>
              <input type="text" aria-label="Target account" value={target} onChange={(e) => setTarget(e.target.value)}
                placeholder="Enter X handle (e.g. @MehwishQamas)"
                style={{ width: "100%", boxSizing: "border-box", padding: "16px 16px 16px 48px", background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, color: "#fff", fontSize: 16, outline: "none" }} />
            </div>
            <div style={{ width: 120 }}>
              <input type="number" aria-label="Maximum items" value={limit} onChange={(e) => setLimit(e.target.value)} min="1" max="50" title="Max tweets"
                style={{ width: "100%", boxSizing: "border-box", padding: 16, background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 12, color: "#fff", fontSize: 16, outline: "none", textAlign: "center" }} />
            </div>
            <button type="submit" disabled={isLoading || !target.trim()}
              style={{ background: "linear-gradient(135deg, #1DA1F2, #0ea5e9)", color: "#fff", padding: "0 32px", minHeight: 52, borderRadius: 12, fontSize: 16, fontWeight: 600, border: "none", cursor: isLoading ? "not-allowed" : "pointer", display: "flex", alignItems: "center", gap: 10, opacity: isLoading || !target.trim() ? 0.7 : 1, transition: "opacity 0.2s" }}>
              {isLoading ? <><Loader2 size={20} className="spin" /> Scraping...</> : "Analyze Tweets"}
            </button>
          </form>
        </div>

        {error && (
          <div role="alert" style={{ background: "rgba(239, 68, 68, 0.1)", border: "1px solid rgba(239, 68, 68, 0.2)", borderRadius: 12, padding: 20, display: "flex", alignItems: "center", gap: 16, color: "#FECACA", marginBottom: 40 }}>
            <AlertTriangle size={24} color="#EF4444" style={{ flexShrink: 0 }} />
            <p style={{ margin: 0 }}>{error}</p>
          </div>
        )}

        {results && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            <h3 style={{ fontSize: 24, fontWeight: 700, color: "#F8FAFC", marginBottom: 12 }}>
              Found {results.count} tweets for @{results.target}
            </h3>
            {results.message && <p style={{ color: "#94A3B8" }}>{results.message}</p>}

            {results.count > 0 && (
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 24 }}>
                {Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([emo, n]) => (
                  <span key={emo} style={{ fontSize: 13, padding: "5px 12px", borderRadius: 99, background: (EMOTION_META[emo]?.color || "#64748B") + "33", color: "#F1F5F9" }}>
                    {EMOTION_META[emo]?.icon} {emo} · {n}
                  </span>
                ))}
                {results.summary?.urgent_count > 0 && (
                  <span style={{ fontSize: 13, padding: "5px 12px", borderRadius: 99, background: "rgba(239,68,68,0.2)", color: "#FCA5A5" }}>
                    ⚠ {results.summary.urgent_count} with safety language
                  </span>
                )}
              </div>
            )}

            <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
              {results.results.map((item, idx) => (
                <div key={item.source_id || idx} className="glass-card" style={{ padding: 24, borderLeft: `4px solid ${item.prediction.color}` }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16, gap: 12, flexWrap: "wrap" }}>
                    <p style={{ fontSize: 13, color: "#94A3B8", margin: 0 }}>Posted: {item.created_utc}</p>
                    {item.url && (
                      <a href={item.url} target="_blank" rel="noreferrer" style={{ color: "#38BDF8", display: "flex", alignItems: "center", gap: 6, fontSize: 14, textDecoration: "none" }}>
                        View Tweet <ExternalLink size={14} />
                      </a>
                    )}
                  </div>
                  <div style={{ background: "rgba(0,0,0,0.3)", padding: 16, borderRadius: 8, color: "#E2E8F0", fontSize: 15, lineHeight: 1.6, marginBottom: 18, whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
                    {item.text}
                  </div>
                  <PredictionSummary prediction={item.prediction} />
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </motion.div>
    </div>
  );
}