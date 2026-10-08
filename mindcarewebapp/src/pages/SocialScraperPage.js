/*
src/pages/SocialScraperPage.jsx — Consolidated Social Footprint Scanner
Unified UI supporting flat single-pass or granular sentence-level matrix breakdowns.
*/

import React, { useState } from "react";
import { motion } from "framer-motion";
import {
  Search,
  Loader2,
  AlertTriangle,
  MessageSquare,
  X,
  ExternalLink,
} from "lucide-react";

import {
  apiRequest,
  EMOTION_META,
  DISORDER_LABELS,
} from "./AnalysisViewPage.js";

// 🔬 Failsafe Compact Disorder Progress Bar Tracker
function CompactDisorderBars({ result }) {
  const rawProbs =
    result?.disorder_probabilities ||
    result?.prediction?.disorder_probabilities ||
    result?.disorders ||
    {};

  const entries = Object.entries(rawProbs)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3);

  if (entries.length === 0) {
    return (
      <span
        style={{
          fontSize: "11px",
          color: "#64748B",
          fontStyle: "italic",
        }}
      >
        Calculating probability parameters...
      </span>
    );
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "8px",
        marginTop: "12px",
      }}
    >
      {entries.map(([key, p]) => {
        const probValue =
          typeof p === "object"
            ? p.probability ?? 0
            : parseFloat(p) || 0;

        return (
          <div
            key={key}
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              fontSize: "12px",
            }}
          >
            <span style={{ color: "#94A3B8" }}>
              {DISORDER_LABELS[key] || key}
            </span>

            <div
              style={{
                flex: 1,
                height: "6px",
                background: "rgba(255,255,255,0.05)",
                borderRadius: "99px",
                margin: "0 10px",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  height: "100%",
                  width: `${probValue * 100}%`,
                  background:
                    probValue > 0.65
                      ? "#EF4444"
                      : probValue > 0.35
                        ? "#F59E0B"
                        : "#14B8A6",
                  borderRadius: "99px",
                }}
              />
            </div>

            <span
              style={{
                fontWeight: 700,
                color:
                  probValue > 0.65
                    ? "#EF4444"
                    : probValue > 0.35
                      ? "#F59E0B"
                      : "#14B8A6",
                fontVariantNumeric: "tabular-nums",
              }}
            >
              {(probValue * 100).toFixed(1)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

export default function SocialScraperPage() {
  const [activeTab, setActiveTab] = useState("reddit");
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
      const n = Math.min(
        Math.max(parseInt(limit, 10) || 10, 1),
        50
      );

      const cleanTarget = target
        .trim()
        .replace("@", "")
        .replace("u/", "");

      const subRoute =
        activeTab === "reddit"
          ? "/api/scrape/reddit"
          : "/api/scrape/twitter";

      const response = await fetch(
        `http://127.0.0.1:5000${subRoute}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            target: cleanTarget,
            limit: n,
          }),
        }
      );

      if (!response.ok) {
        throw new Error(
          `Server returned HTTP Error Status: ${response.status}`
        );
      }

      const data = await response.json();

      if (data.status === "error" || !data.results) {
        throw new Error(
          data.error ||
          data.message ||
          "Failed to parse multi-task extraction data."
        );
      }

      // ADAPTIVE LAYER:
      // Standardizes payload parameters safely regardless
      // of single or sentence-split arrays.
      const normalizedResults = data.results.map((item) => {
        const predObj = item.prediction || {};

        // Handle nested sentence segmentation variables if present
        if (
          predObj.segmentation === "granular" &&
          Array.isArray(predObj.segments)
        ) {
          return {
            ...item,
            text:
              item.text ||
              "Long form document sequence parsed.",
            prediction: {
              segmentation: "granular",
              dominant_emotion:
                predObj.dominant_emotion || "normal",
              color: predObj.color || "#475569",
              icon: predObj.icon || "😐",
              segments: predObj.segments,
            },
          };
        }

        // Standard fallback logic for single-pass inputs
        const extractedEmotion = (
          predObj.emotion || "normal"
        ).toLowerCase();

        const extractedColor =
          predObj.color ||
          EMOTION_META[extractedEmotion]?.color ||
          "#475569";

        return {
          ...item,
          text: item.text || "Empty observation track.",
          prediction: {
            segmentation: "single",
            emotion: extractedEmotion,
            color: extractedColor,
            icon:
              EMOTION_META[extractedEmotion]?.icon ||
              "😐",
            disorder_probabilities:
              predObj.disorder_probabilities || {},
          },
        };
      });

      const emotionCounts = {};

      normalizedResults.forEach((item) => {
        const emo =
          item.prediction.segmentation === "granular"
            ? item.prediction.dominant_emotion
            : item.prediction.emotion;

        emotionCounts[emo] =
          (emotionCounts[emo] || 0) + 1;
      });

      setResults({
        count: normalizedResults.length,
        target: cleanTarget,
        summary: {
          emotion_counts: emotionCounts,
          urgent_count:
            data.summary?.urgent_count || 0,
        },
        results: normalizedResults,
      });
    } catch (err) {
      console.error(err);

      setError(
        err.message ||
        "Failed to sync transaction parameters with backend server."
      );
    } finally {
      setIsLoading(false);
    }
  };

  const counts =
    results?.summary?.emotion_counts || {};

  const activeBrandGradient =
    activeTab === "reddit"
      ? "linear-gradient(135deg, #FF4500, #FF871D)"
      : "linear-gradient(135deg, #1DA1F2, #0ea5e9)";

  return (
    <div
      className="container"
      style={{
        paddingTop: 140,
        paddingBottom: 100,
        background: "#0F172A",
        minHeight: "100vh",
        color: "#F1F5F9",
      }}
    >
      <style>
        {`
          .spin {
            animation: mc-spin 1s linear infinite;
          }

          @keyframes mc-spin {
            to {
              transform: rotate(360deg);
            }
          }
        `}
      </style>

      <div
        style={{
          maxWidth: 850,
          margin: "0 auto",
          padding: "0 20px",
        }}
      >
        {/* HEADER SECTION */}

        <div
          style={{
            textAlign: "center",
            marginBottom: 35,
          }}
        >
          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              width: 55,
              height: 60,
              borderRadius: 20,
              background: activeBrandGradient,
              marginBottom: 15,
            }}
          >
            {activeTab === "reddit" ? (
              <MessageSquare
                size={28}
                color="#fff"
              />
            ) : (
              <X size={28} color="#fff" />
            )}
          </div>

          <h1
            style={{
              fontSize: 42,
              fontWeight: 800,
              color: "#F8FAFC",
              marginBottom: 12,
              letterSpacing: "-0.02em",
            }}
          >
            Unified Social{" "}
            <span
              style={{
                color:
                  activeTab === "reddit"
                    ? "#FF4500"
                    : "#38BDF8",
              }}
            >
              Scraper Engine
            </span>
          </h1>

          <p
            style={{
              fontSize: 16,
              color: "#94A3B8",
              maxWidth: 600,
              margin: "0 auto",
            }}
          >
            Extract clinical footprints across platforms
            and compute spatial attention weights forward
            and backward simultaneously.
          </p>
        </div>

        {/* CONTROLLING TABS MATRIX BAR */}

        <div
          style={{
            display: "flex",
            gap: "16px",
            borderBottom: "1px solid #1E293B",
            marginBottom: "24px",
            paddingBottom: "4px",
          }}
        >
          <button
            onClick={() => {
              setActiveTab("reddit");
              setResults(null);
              setTarget("");
              setError(null);
            }}
            style={{
              background: "transparent",
              border: "none",
              cursor: "pointer",
              fontSize: "14px",
              fontWeight: 700,
              paddingBottom: "10px",
              color:
                activeTab === "reddit"
                  ? "#FF4500"
                  : "#475569",
              borderBottom:
                activeTab === "reddit"
                  ? "2px solid #FF4500"
                  : "none",
            }}
          >
            👽 Reddit Mining Hub
          </button>

          <button
            onClick={() => {
              setActiveTab("twitter");
              setResults(null);
              setTarget("");
              setError(null);
            }}
            style={{
              background: "transparent",
              border: "none",
              cursor: "pointer",
              fontSize: "14px",
              fontWeight: 700,
              paddingBottom: "10px",
              color:
                activeTab === "twitter"
                  ? "#38BDF8"
                  : "#475569",
              borderBottom:
                activeTab === "twitter"
                  ? "2px solid #38BDF8"
                  : "none",
            }}
          >
            🐦 X (Twitter) Mining Hub
          </button>
        </div>

        {/* INPUT TRANSMISSION LAYOUT CARD */}

        <div
          style={{
            padding: 25,
            marginBottom: 30,
            background: "rgba(30, 41, 59, 0.4)",
            border:
              "1px solid rgba(255,255,255,0.06)",
            borderRadius: "16px",
          }}
        >
          <form
            onSubmit={handleScrape}
            style={{
              display: "flex",
              gap: 16,
              flexWrap: "wrap",
            }}
          >
            <div
              style={{
                flex: "1 1 300px",
                position: "relative",
              }}
            >
              <div
                style={{
                  position: "absolute",
                  left: 16,
                  top: 16,
                  color: "#64748B",
                }}
              >
                <Search
                  size={20}
                  style={{
                    position: "absolute",
                    left: 16,
                    top: 0,
                  }}
                />
              </div>

              <input
                type="text"
                value={target}
                onChange={(e) =>
                  setTarget(e.target.value)
                }
                placeholder={
                  activeTab === "reddit"
                    ? "Enter Reddit username (e.g. SamiSha_)"
                    : "Enter X handle (e.g. @MehwishQamas)"
                }
                style={{
                  width: "100%",
                  boxSizing: "border-box",
                  padding:
                    "16px 16px 16px 48px",
                  background:
                    "rgba(15, 23, 42, 0.6)",
                  border:
                    "1px solid rgba(255,255,255,0.1)",
                  borderRadius: 12,
                  color: "#fff",
                  fontSize: 15,
                  outline: "none",
                }}
              />
            </div>

            <div style={{ width: 100 }}>
              <input
                type="number"
                value={limit}
                onChange={(e) =>
                  setLimit(e.target.value)
                }
                min="1"
                max="50"
                style={{
                  width: "100%",
                  boxSizing: "border-box",
                  padding: 16,
                  background:
                    "rgba(15, 23, 42, 0.6)",
                  border:
                    "1px solid rgba(255,255,255,0.1)",
                  borderRadius: 12,
                  color: "#fff",
                  fontSize: 15,
                  outline: "none",
                  textAlign: "center",
                }}
              />
            </div>

            <button
              type="submit"
              disabled={
                isLoading || !target.trim()
              }
              style={{
                background: activeBrandGradient,
                color: "#fff",
                padding: "0 28px",
                minHeight: 52,
                borderRadius: 12,
                fontSize: 15,
                fontWeight: 600,
                border: "none",
                cursor: isLoading
                  ? "not-allowed"
                  : "pointer",
                display: "flex",
                alignItems: "center",
                gap: 10,
                opacity:
                  isLoading || !target.trim()
                    ? 0.7
                    : 1,
              }}
            >
              {isLoading ? (
                <>
                  <Loader2
                    size={18}
                    className="spin"
                  />
                  Extraction Active...
                </>
              ) : (
                "Analyze Account"
              )}
            </button>
          </form>
        </div>

        {/* ERROR MESSAGE */}

        {error && (
          <div
            role="alert"
            style={{
              background:
                "rgba(239, 68, 68, 0.08)",
              border:
                "1px solid rgba(239, 68, 68, 0.2)",
              borderRadius: 12,
              padding: 16,
              display: "flex",
              alignItems: "center",
              gap: 14,
              color: "#FECACA",
              marginBottom: 30,
            }}
          >
            <AlertTriangle
              size={20}
              color="#EF4444"
              style={{ flexShrink: 0 }}
            />

            <p
              style={{
                margin: 0,
                fontSize: "14px",
              }}
            >
              {error}
            </p>
          </div>
        )}

        {/* RESULTS FEED RENDER BLOCKS */}

        {results && (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "24px",
            }}
          >
            <h3
              style={{
                fontSize: 20,
                fontWeight: 700,
                color: "#F8FAFC",
                marginBottom: 4,
              }}
            >
              Extracted Timeline Data for{" "}
              {activeTab === "reddit" ? "u/" : "@"}
              {results.target}
            </h3>

            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                gap: 8,
                marginBottom: 12,
              }}
            >
              {Object.entries(counts).map(
                ([emo, n]) => (
                  <span
                    key={emo}
                    style={{
                      fontSize: 12,
                      padding: "4px 12px",
                      borderRadius: 99,
                      background:
                        (EMOTION_META[emo]?.color ||
                          "#64748B") + "25",
                      color: "#F1F5F9",
                      fontWeight: 600,
                    }}
                  >
                    {EMOTION_META[emo]?.icon ||
                      "😐"}{" "}
                    {emo.toUpperCase()} · {n}
                  </span>
                )
              )}
            </div>

            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 24,
              }}
            >
              {results.results.map(
                (item, idx) => (
                  <div
                    key={idx}
                    style={{
                      padding: 24,
                      background:
                        "rgba(30, 41, 59, 0.2)",
                      border:
                        "1px solid rgba(255,255,255,0.04)",
                      borderRadius: "14px",
                      borderLeft: `4px solid ${item.prediction.color ||
                        "#475569"
                        }`,
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        justifyContent:
                          "space-between",
                        alignItems: "flex-start",
                        marginBottom: 12,
                        gap: 12,
                        flexWrap: "wrap",
                      }}
                    >
                      <p
                        style={{
                          fontSize: 12,
                          color: "#64748B",
                          margin: 0,
                        }}
                      >
                        Timestamp:{" "}
                        {item.created_utc ||
                          "Recent"}
                      </p>

                      {item.url && (
                        <a
                          href={item.url}
                          target="_blank"
                          rel="noreferrer"
                          style={{
                            color: "#38BDF8",
                            display: "flex",
                            alignItems: "center",
                            gap: 5,
                            fontSize: 13,
                            textDecoration: "none",
                            fontWeight: 500,
                          }}
                        >
                          View Original Link
                          <ExternalLink size={14} />
                        </a>
                      )}
                    </div>

                    <div
                      style={{
                        background:
                          "rgba(15, 23, 42, 0.5)",
                        padding: 16,
                        borderRadius: 8,
                        color: "#E2E8F0",
                        fontSize: 14,
                        lineHeight: 1.6,
                        marginBottom: 16,
                      }}
                    >
                      "{item.text}"
                    </div>

                    {/* CONDITIONAL RENDER CHANNELS FOR SINGLE PASS OR GRANULAR BREAKDOWNS */}

                    {item.prediction
                      .segmentation ===
                      "granular" ? (
                      <div
                        style={{
                          display: "flex",
                          flexDirection: "column",
                          gap: "12px",
                          marginTop: "12px",
                        }}
                      >
                        <span
                          style={{
                            fontSize: "11px",
                            color: "#64748B",
                            fontWeight: 700,
                            letterSpacing: "0.1em",
                          }}
                        >
                          🧠 SENTENCE-LEVEL
                          SEGMENTATION TRACE MATRIX
                        </span>

                        {item.prediction.segments.map(
                          (seg, sIdx) => (
                            <div
                              key={sIdx}
                              style={{
                                background: "#0F172A",
                                padding: "14px",
                                borderRadius: "8px",
                                borderLeft: `3px solid ${seg.color
                                  }`,
                                border:
                                  "1px solid rgba(255,255,255,0.03)",
                              }}
                            >
                              <div
                                style={{
                                  display: "flex",
                                  justifyContent:
                                    "space-between",
                                  alignItems:
                                    "center",
                                  marginBottom: "6px",
                                }}
                              >
                                <span
                                  style={{
                                    fontSize: "11px",
                                    color: "#475569",
                                  }}
                                >
                                  Segment #
                                  {
                                    seg.sentence_index
                                  }
                                </span>

                                <span
                                  style={{
                                    fontSize: "11px",
                                    color:
                                      seg.color,
                                    fontWeight: 700,
                                  }}
                                >
                                  {seg.icon}{" "}
                                  {seg.emotion.toUpperCase()}
                                </span>
                              </div>

                              <p
                                style={{
                                  fontSize: "13px",
                                  color: "#CBD5E1",
                                  marginBottom: "8px",
                                  fontStyle: "italic",
                                }}
                              >
                                "{seg.text}"
                              </p>
                            </div>
                          )
                        )}
                      </div>
                    ) : (
                      <div
                        style={{
                          background: "#0F172A",
                          padding: "16px",
                          borderRadius: "10px",
                          border:
                            "1px solid rgba(255,255,255,0.05)",
                        }}
                      >
                        <div
                          style={{
                            display: "flex",
                            justifyContent:
                              "space-between",
                            alignItems: "center",
                            marginBottom: "8px",
                          }}
                        >
                          <span
                            style={{
                              fontSize: "11px",
                              color: "#64748B",
                              fontWeight: 700,
                            }}
                          >
                            TOKEN ATTENTION MATRIX
                            CLASSIFICATION
                          </span>

                          <span
                            style={{
                              fontSize: "11px",
                              color:
                                item.prediction
                                  .color,
                              fontWeight: 700,
                            }}
                          >
                            {item.prediction.icon}{" "}
                            EMOTION:{" "}
                            {item.prediction.emotion?.toUpperCase()}
                          </span>
                        </div>

                        <CompactDisorderBars
                          result={item}
                        />
                      </div>
                    )}
                  </div>
                )
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}