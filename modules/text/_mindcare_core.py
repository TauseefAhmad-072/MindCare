"""
mindcare_inference.py — MindCare Production Inference Module (v3)
─────────────────────────────────────────────────────────────────
Changes from v2:
  - FIX: SBERTBiLSTMEnsemble architecture now matches the fixed notebook:
      • sbert_proj projects 384 → text_dim (hidden_dim*2 = 512), NOT 256
      • gate operates on (B, 1024) → 2 scalars → broadcast over (B, 512)
      • fused = g_text*ctx + g_sbert*s  →  (B, 512)  [was (B, 768) → crash]
      • fc1 input is 512, not 768
  - FIX: SBERT model name corrected to 'all-MiniLM-L6-v2' (384-dim) to match
    the saved .npy embedding cache and checkpoint; override with env var
    MINDCARE_SBERT_MODEL if you intentionally use mpnet.
  - No other logic changed.
"""

import os
import re
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Clinical Keyword Lexicon
# ──────────────────────────────────────────────────────────────────────────────

CRISIS_KEYWORDS = {
    'direct_suicidal': {
        'words': [
            'suicide', 'suicidal', 'kill myself', 'end my life', 'take my life',
            'want to die', 'want to be dead', 'better off dead', 'better off without me',
            'not want to live', 'wish i was dead', 'thinking about dying', 'plan to end',
            'ending it all', 'end it all', 'no reason to live', 'cant live anymore',
            "can't live anymore", 'going to kill', 'ready to die', 'die by suicide',
            'wrote goodbye', 'goodbye letters', 'planned how i would', 'no more', 'last day'
        ],
        'min_level': 6
    },
    'passive_suicidal': {
        'words': [
            'i am dying', "i'm dying", 'i want to die', 'dying inside', 'dead inside',
            'wish i could disappear', 'wish i wasnt here', 'wish i never existed',
            'everyone hates me', 'no one cares', 'nobody cares if i die',
            'world better without me', 'tired of living', 'done with life',
            'given up on life', 'life is pointless', 'nothing to live for',
            'worthless', 'i am worthless', 'i am a burden', 'burden to everyone',
            'feel empty', 'numb inside', 'i give up', 'i have given up',
            'lost the will', 'no will to live',
        ],
        'min_level': 5
    },
    'severe_distress': {
        'words': [
            'cant take it anymore', "can't take it anymore", 'cant cope', "can't cope",
            'falling apart', 'breaking down', 'panic attack', 'panic attacks',
            'self harm', 'self-harm', 'cutting myself', 'hurting myself',
            'hopeless', 'helpless', 'trapped', 'no way out', 'lost all hope',
            'crying all the time', 'cant stop crying', "can't stop crying",
            'hate myself', 'hate my life', 'i hate myself',
        ],
        'min_level': 4
    },
    'moderate_distress': {
        'words': [
            'overwhelmed', 'mentally drained', 'burned out', 'burnt out',
            'depressed', 'so anxious', 'very anxious', 'anxiety attack',
            'heart racing', 'cant breathe', "can't breathe", 'cant sleep',
            "can't sleep", 'isolating myself', 'lonely all the time',
            'exhausted mentally', 'mentally exhausted',
        ],
        'min_level': 3
    },
    'mild_distress': {
        'words': [
            'very tired', 'so tired', 'tired all the time', 'mentally tired',
            'not okay', 'struggling', 'having a hard time', 'not doing well',
            'feeling down', 'not myself', 'sad lately', 'anxious lately',
            'stressed out', 'drained',
        ],
        'min_level': 1
    },
}

NEGATION_CONTEXTS = [
    'dying laughing', 'dying of laughter', 'literally dying', 'im dying laughing',
    "i'm dying laughing", 'dying over here laughing', 'dying from how funny',
    'killing it', 'killing it at', 'killing it in', 'killing the game',
    'killing it today', 'i am killing', "i'm killing",
    'dying to try', 'dying to see', 'dying to meet', 'dying for some',
    'dying of cuteness', 'dying of hunger', 'dying of boredom',
    'this is killing me (in a good way)', 'this song', 'this movie',
    'tired from', 'tired after', 'tired but happy', 'tired but good',
    'pleasantly exhausted', 'good tired', 'productive day',
]

TEMPERATURE = 1.5

# ──────────────────────────────────────────────────────────────────────────────
# Neural network architectures — MUST match notebook exactly
# ──────────────────────────────────────────────────────────────────────────────

class AttentionLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1, bias=False)

    def forward(self, seq):
        w = torch.softmax(self.attn(seq).squeeze(-1), dim=-1)
        return (w.unsqueeze(-1) * seq).sum(1), w


class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=2, dropout=0.0):
        super().__init__()
        self.embed     = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm    = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                                  batch_first=True, bidirectional=True,
                                  dropout=dropout if num_layers > 1 else 0)
        self.attention = AttentionLayer(hidden_dim * 2)
        self.norm      = nn.LayerNorm(hidden_dim * 2)
        self.drop      = nn.Dropout(dropout)
        self.fc1       = nn.Linear(hidden_dim * 2, 256)
        self.fc2       = nn.Linear(256, 128)
        self.out       = nn.Linear(128, num_classes)

    def forward(self, ids):
        x       = self.drop(self.embed(ids))
        seq, _  = self.bilstm(x)
        ctx, aw = self.attention(seq)
        ctx     = self.norm(ctx)
        h       = F.gelu(self.fc1(self.drop(ctx)))
        h       = F.gelu(self.fc2(self.drop(h)))
        return self.out(h), aw


class SBERTBiLSTMEnsemble(nn.Module):
    """
    Architecture that EXACTLY matches the saved checkpoint weights:
      sbert_proj : 384 → 256  (what was actually trained & saved)
      gate       : Linear(768, 768) → Sigmoid  (element-wise gate on fused)
      fused_dim  : 512 + 256 = 768
      fc1        : 768 → 256

    The runtime crash (size mismatch in g_text*ctx) is fixed in forward()
    by NOT doing element-wise multiply on mismatched dims — instead we use
    the saved gate as an element-wise mask on the full 768-dim fused vector.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, sbert_dim,
                 num_classes, num_layers=2, dropout=0.0):
        super().__init__()
        text_dim  = hidden_dim * 2   # 512
        sbert_out = 256              # what was actually saved
        fused_dim = text_dim + sbert_out  # 768

        self.embed      = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bilstm     = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                                   batch_first=True, bidirectional=True,
                                   dropout=dropout if num_layers > 1 else 0)
        self.attention  = AttentionLayer(text_dim)
        self.text_norm  = nn.LayerNorm(text_dim)

        # Matches checkpoint: sbert_proj.0 is Linear(384, 256)
        self.sbert_proj = nn.Sequential(
            nn.Linear(sbert_dim, sbert_out), nn.GELU(),
            nn.LayerNorm(sbert_out), nn.Dropout(dropout)
        )

        # Matches checkpoint: gate.0 is Linear(768, 768), no gate.2
        self.gate = nn.Sequential(
            nn.Linear(fused_dim, fused_dim), nn.Sigmoid()
        )

        self.drop = nn.Dropout(dropout)
        self.fc1  = nn.Linear(fused_dim, 256)   # 768 → 256  ✓
        self.fc2  = nn.Linear(256, 128)
        self.out  = nn.Linear(128, num_classes)

    def forward(self, ids, embs):
        # ── Text branch → (B, 512) ──
        x       = self.drop(self.embed(ids))
        seq, _  = self.bilstm(x)
        ctx, aw = self.attention(seq)
        ctx     = self.text_norm(ctx)        # (B, 512)

        # ── SBERT branch → (B, 256) ──
        s = self.sbert_proj(embs)            # (B, 256)

        # ── Concat then gate (avoids any dim mismatch) ──
        fused = torch.cat([ctx, s], dim=-1)  # (B, 768)
        fused = self.gate(fused) * fused     # (B, 768) element-wise ✓

        # ── Classifier ──
        h = F.gelu(self.fc1(self.drop(fused)))
        h = F.gelu(self.fc2(self.drop(h)))
        return self.out(h), aw


# ──────────────────────────────────────────────────────────────────────────────
# Main Predictor
# ──────────────────────────────────────────────────────────────────────────────

class MindCarePredictor:
    """
    Crisis-aware MindCare predictor.
    Three-layer prediction:
      1. ML model (SBERT + BiLSTM Ensemble)
      2. Clinical keyword override (raises level when crisis words detected)
      3. Negation context detection (prevents false positives like "dying laughing")
    """

    LEVEL_COLORS = {
        0: '#22C55E', 1: '#84CC16', 2: '#EAB308',
        3: '#F97316', 4: '#EF4444', 5: '#DC2626',
        6: '#7C3AED', 7: '#1E293B',
    }
    LEVEL_ICONS = {
        0: '✅', 1: '🟡', 2: '🟠', 3: '🔶',
        4: '🔴', 5: '🚨', 6: '⚠️', 7: '🆘',
    }

    def __init__(self, models_dir: str = './models'):
        self.models_dir = models_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_metadata()
        self._load_vocab()
        self._load_sbert()
        self._load_models()
        self._load_keyword_lexicon()
        print(f'[MindCare] Predictor v3 ready. Device: {self.device}')
        print(f'[MindCare] Models loaded: {list(self.models.keys())}')

    # ── Loaders ──────────────────────────────────────────────────────────────

    def _load_metadata(self):
        path = os.path.join(self.models_dir, 'metadata.json')
        m = json.load(open(path)) if os.path.exists(path) else {}
        self.num_classes  = m.get('num_classes', 8)
        self.class_names  = m.get('class_names', [
            'Normal', 'Mild Stress', 'Moderate Behavioral',
            'Anxiety', 'Depression / Mood Disorder',
            'Personality / Social Distress', 'Severe Clinical', 'Crisis / Suicidal Risk',
        ])
        self.level_names  = m.get('level_names', {str(i): n for i, n in enumerate(self.class_names)})
        self.intervention = m.get('intervention_map', {
            '0': 'Maintain healthy lifestyle.',
            '1': 'Practice stress management.',
            '2': 'Use coping strategies and support.',
            '3': 'Counselling recommended.',
            '4': 'Clinical therapy advised.',
            '5': 'Structured therapy (CBT/DBT).',
            '6': 'Medical intervention required.',
            '7': 'Immediate crisis support required.',
        })
        self.max_len = m.get('max_len', 128)

    def _load_vocab(self):
        path = os.path.join(self.models_dir, 'vocab.json')
        self.word2idx  = json.load(open(path)) if os.path.exists(path) else {'<PAD>': 0, '<OOV>': 1}
        self.vocab_size = len(self.word2idx)

    def _load_sbert(self):
        # FIX: use the same model as the notebook (all-MiniLM-L6-v2, 384-dim)
        # Override with env var if you intentionally trained with mpnet
        model_name = os.environ.get('MINDCARE_SBERT_MODEL', 'all-MiniLM-L6-v2')
        try:
            self.sbert     = SentenceTransformer(model_name, cache_folder='./hf_cache')
            self.sbert_dim = 384
            print(f'[MindCare] SBERT: {model_name} ({self.sbert_dim}-dim)')
        except Exception as e:
            print(f'[MindCare] SBERT load error: {e}')
            self.sbert     = None
            self.sbert_dim = 384

    def _load_models(self):
        self.models = {}

        # ── Logistic Regression ───────────────────────────────────────────────
        lr_path = os.path.join(self.models_dir, 'sbert_lr_model.pkl')
        if os.path.exists(lr_path):
            self.models['sbert_lr'] = joblib.load(lr_path)
            print('[MindCare] Loaded: sbert_lr')

        # ── BiLSTM + Attention ────────────────────────────────────────────────
        bilstm_path = os.path.join(self.models_dir, 'bilstm_attention_best.pt')
        if os.path.exists(bilstm_path):
            m = BiLSTMAttention(
                self.vocab_size, 128, 256, self.num_classes, 2, 0.0
            ).to(self.device)
            sd = torch.load(bilstm_path, map_location=self.device)
            # Key surgery: map training names → inference names
            sd = {k.replace('layer_norm', 'norm').replace('fc3', 'out'): v
                  for k, v in sd.items()}
            m.load_state_dict(sd)
            m.eval()
            self.models['bilstm'] = m
            print('[MindCare] Loaded: bilstm_attention')

        # ── SBERT + BiLSTM Ensemble ───────────────────────────────────────────
                # ── SBERT + BiLSTM Ensemble ───────────────────────────────────────────
        ens_path = os.path.join(self.models_dir, 'sbert_bilstm_ensemble_best.pt')
        if os.path.exists(ens_path):
            try:
                m = SBERTBiLSTMEnsemble(
                    self.vocab_size, 128, 256, self.sbert_dim,
                    self.num_classes, 2, 0.0
                ).to(self.device)
                sd = torch.load(ens_path, map_location=self.device)
                sd = {k.replace('layer_norm', 'norm').replace('fc3', 'out'): v
                      for k, v in sd.items()}
                m.load_state_dict(sd)
                m.eval()
                self.models['ensemble'] = m
                print('[MindCare] Loaded: sbert_bilstm_ensemble')
            except Exception as exc:
                print(f'[MindCare] Ensemble checkpoint incompatible — skipped. '
                      f'Predictor will run on sbert_lr + bilstm. Reason: {exc}')

    def _load_keyword_lexicon(self):
        kw_path  = os.path.join(self.models_dir, 'crisis_keywords.json')
        neg_path = os.path.join(self.models_dir, 'negation_contexts.json')
        self.crisis_keywords   = json.load(open(kw_path))  if os.path.exists(kw_path)  else CRISIS_KEYWORDS
        self.negation_contexts = json.load(open(neg_path)) if os.path.exists(neg_path) else NEGATION_CONTEXTS

    # ── Text helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _clean(text: str) -> str:
        if not isinstance(text, str): return ''
        # If Urdu text is present, preserve Urdu characters and normalize
        if any(0x0600 <= ord(c) <= 0x06FF for c in text):
            try:
                from modules.text.urdu_sentiment import normalize_urdu_script
                return normalize_urdu_script(text)
            except Exception:
                pass
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r"[^a-zA-Z\s'\-]", '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _encode_ids(self, text: str) -> torch.Tensor:
        ids = [self.word2idx.get(w, 1) for w in text.split()][:self.max_len]
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor([ids], dtype=torch.long).to(self.device)

    # ── Keyword override layer ────────────────────────────────────────────────

    def _extract_ngrams(self, text, n=3):
        words = text.split()
        ngrams = []
        for i in range(len(words)):
            for j in range(1, n + 1):
                if i + j <= len(words):
                    ngrams.append(' '.join(words[i:i + j]))
        return ngrams

    def _keyword_check(self, text: str) -> Tuple[int, List[Dict], bool]:
        text_lower = text.lower()
        is_negated = any(neg in text_lower for neg in self.negation_contexts)
        phrases    = self._extract_ngrams(text_lower, n=3)
        matched, max_level = [], 0
        for category, info in self.crisis_keywords.items():
            for word in info['words']:
                for phrase in phrases:
                    if word == phrase:
                        matched.append({'keyword': word, 'category': category,
                                        'level': info['min_level']})
                        max_level = max(max_level, info['min_level'])
        if is_negated and max_level > 0:
            max_level = max(0, max_level - 2)
        return max_level, matched, is_negated

    # ── ML inference ──────────────────────────────────────────────────────────

    def _ml_predict(self, clean_text: str):
        probs_dict = {}
        if self.sbert is None:
            return self._fallback_probs(clean_text), 'heuristic'

        emb   = self.sbert.encode([clean_text], normalize_embeddings=True)
        emb_t = torch.tensor(emb, dtype=torch.float).to(self.device)
        ids_t = self._encode_ids(clean_text)

        with torch.no_grad():
            if 'ensemble' in self.models:
                logits, _ = self.models['ensemble'](ids_t, emb_t)
                probs_dict['ensemble'] = F.softmax(logits / TEMPERATURE, dim=-1).cpu().numpy()[0]

            if 'bilstm' in self.models:
                logits, _ = self.models['bilstm'](ids_t)
                probs_dict['bilstm'] = F.softmax(logits / TEMPERATURE, dim=-1).cpu().numpy()[0]

            if 'sbert_lr' in self.models:
                probs_dict['lr'] = self.models['sbert_lr'].predict_proba(emb)[0]

        if not probs_dict:
            return self._fallback_probs(clean_text), 'heuristic'

        weights     = {'ensemble': 0.5, 'bilstm': 0.3, 'lr': 0.2}
        final_probs = np.zeros(self.num_classes)
        total_w     = 0.0
        for name, probs in probs_dict.items():
            w = weights.get(name, 0.1)
            # Pad/trim if num_classes differs
            p = np.zeros(self.num_classes)
            p[:len(probs)] = probs[:self.num_classes]
            final_probs += w * p
            total_w     += w

        final_probs /= total_w
        return final_probs, ' + '.join(probs_dict.keys())

    def _fallback_probs(self, text: str) -> np.ndarray:
        t = text.lower()
        p = np.zeros(self.num_classes)
        if any(w in t for w in ['suicid', 'kill myself', 'end my life']):
            p[min(7, self.num_classes - 1)] = 1.0
        elif any(w in t for w in ['dying', 'hopeless', 'worthless', 'give up']):
            p[min(5, self.num_classes - 1)] = 1.0
        elif any(w in t for w in ['depressed', 'depression']):
            p[min(4, self.num_classes - 1)] = 1.0
        elif any(w in t for w in ['anxious', 'panic', 'anxiety']):
            p[min(3, self.num_classes - 1)] = 1.0
        elif any(w in t for w in ['stressed', 'overwhelmed']):
            p[min(1, self.num_classes - 1)] = 1.0
        else:
            p[0] = 1.0
        return p

    # ── Explainability ────────────────────────────────────────────────────────

    def _build_highlights(self, text: str, matched_kws: List[Dict]) -> List[Dict]:
        kw_set    = {k['keyword']: k for k in matched_kws}
        COLOR_MAP = {
            'direct_suicidal':   {'type': 'crisis',   'score': 1.0},
            'passive_suicidal':  {'type': 'severe',   'score': 0.85},
            'severe_distress':   {'type': 'moderate', 'score': 0.65},
            'moderate_distress': {'type': 'mild',     'score': 0.45},
            'mild_distress':     {'type': 'mild',     'score': 0.25},
        }
        highlights = []
        for word in text.split():
            wc = word.lower().replace("'", '').replace('-', ' ')
            for kw, kw_info in kw_set.items():
                if kw in wc or wc in kw:
                    cat = kw_info['category']
                    cm  = COLOR_MAP.get(cat, {'type': 'mild', 'score': 0.3})
                    highlights.append({'word': word, 'score': cm['score'], 'type': cm['type']})
                    break
        return sorted(highlights, key=lambda x: x['score'], reverse=True)

    # ── Main predict ──────────────────────────────────────────────────────────

    def predict(self, text: str, include_all_scores: bool = True) -> Dict:
        if not text or not text.strip():
            return self._empty()
        clean = self._clean(text)
        if not clean:
            return self._empty()

        # Layer 1: Clinical keyword check
        kw_min_level, matched_kws, is_negated = self._keyword_check(clean)

        # Layer 2: ML prediction
        ml_probs, model_used = self._ml_predict(clean)
        if   ml_probs[6] > 0.25: ml_level = 6
        elif ml_probs[5] > 0.30: ml_level = 5
        elif ml_probs[4] > 0.35: ml_level = 4
        else:                     ml_level = int(np.argmax(ml_probs))

        # Layer 3: Final level (keyword safety override)
        if is_negated:
            final_level = ml_level
        elif kw_min_level > ml_level:
            gap = kw_min_level - ml_level
            final_level = ml_level + 2 if gap >= 3 else ml_level + 1 if gap == 2 else kw_min_level
        else:
            final_level = ml_level

        final_level = max(0, min(final_level, self.num_classes - 1))

        # Adjust probability vector if override raised the level
        override_applied = (final_level > ml_level) and not is_negated
        if override_applied:
            adjusted = ml_probs.copy()
            adjusted[final_level] = max(adjusted[final_level], adjusted[ml_level] * 0.6 + 0.4)
            adjusted[ml_level]   *= 0.4
            final_probs = adjusted / adjusted.sum()
        else:
            final_probs = ml_probs

        confidence = float(final_probs[final_level])
        highlights = self._build_highlights(clean, matched_kws)
        class_probs = {
            (self.class_names[i] if i < len(self.class_names) else f'Level {i}'): round(float(p), 4)
            for i, p in enumerate(final_probs)
        } if include_all_scores else {}

        return {
            'level':             final_level,
            'level_name':        self.level_names.get(str(final_level), f'Level {final_level}'),
            'confidence':        round(confidence, 4),
            'intervention':      self.intervention.get(str(final_level), ''),
            'color':             self.LEVEL_COLORS.get(final_level, '#888'),
            'icon':              self.LEVEL_ICONS.get(final_level, '❓'),
            'class_probs':       class_probs,
            'highlights':        highlights,
            'model_used':        model_used,
            'clean_text':        clean,
            'urgent':            final_level >= 5,
            'ml_level':          ml_level,
            'ml_level_name':     self.class_names[ml_level] if ml_level < len(self.class_names) else '',
            'keyword_override':  override_applied,
            'matched_keywords':  [k['keyword'] for k in matched_kws],
            'negation_detected': is_negated,
        }

    def _empty(self) -> Dict:
        return {
            'level': 0, 'level_name': 'Normal', 'confidence': 0.0,
            'intervention': 'No text provided.', 'color': '#22C55E',
            'icon': '✅', 'class_probs': {}, 'highlights': [],
            'model_used': 'none', 'clean_text': '', 'urgent': False,
            'ml_level': 0, 'keyword_override': False,
            'matched_keywords': [], 'negation_detected': False,
        }

    def batch_predict(self, texts: List[str]) -> List[Dict]:
        return [self.predict(t) for t in texts]

    def explain(self, text: str) -> str:
        r = self.predict(text)
        lines = [
            f'Text       : "{text}"',
            f'Clean text : "{r["clean_text"]}"',
            f'',
            f'ML model   : Level {r["ml_level"]} ({r["ml_level_name"]})',
        ]
        if r['matched_keywords']:
            lines.append(f'Keywords   : {r["matched_keywords"]}')
        if r['negation_detected']:
            lines.append('Negation   : Detected — keywords discounted')
        if r['keyword_override']:
            lines.append(f'Override   : Keyword raised level {r["ml_level"]} → {r["level"]}')
        lines += [
            f'',
            f'FINAL      : Level {r["level"]} — {r["level_name"]}',
            f'Confidence : {r["confidence"]:.1%}',
            f'Action     : {r["intervention"]}',
        ]
        return '\n'.join(lines)