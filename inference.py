import re
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

MODEL_PATH = Path("model_assets/best_roberta_model.pth")
SCALER_PATH = Path("model_assets/best_scaler.pkl")

if not MODEL_PATH.exists() or not SCALER_PATH.exists():
    raise FileNotFoundError(
        "Model files not found.\n"
        "Download them from Google Drive and place into model_assets/.\n"
        "Required files:\n"
        " - model_assets/best_roberta_model.pth\n"
        " - model_assets/best_scaler.pkl\n"
        "See README.md for details."
    )

try:
    from textblob import TextBlob
    _TEXTBLOB_OK = True
except Exception:
    _TEXTBLOB_OK = False

try:
    import textstat as _textstat
    _TEXTSTAT_OK = True
except Exception:
    _TEXTSTAT_OK = False


def _count_emojis(text: str) -> int:
    count = 0
    for ch in text:
        cat = unicodedata.category(ch)
        cp = ord(ch)
        if cat in ("So", "Sm") or (0x1F300 <= cp <= 0x1FAFF) or (0x2600 <= cp <= 0x27BF):
            count += 1
    return count


def _safe_polarity(text: str) -> float:
    if not _TEXTBLOB_OK:
        return 0.0
    try:
        return float(TextBlob(text).sentiment.polarity)
    except Exception:
        return 0.0


def _flesch(text: str) -> float:
    if _TEXTSTAT_OK:
        try:
            return float(_textstat.flesch_reading_ease(text))
        except Exception:
            pass
    words = text.split()
    sents = max(text.count(".") + text.count("!") + text.count("?"), 1)
    syllables = sum(_syllable_count(w) for w in words)
    wc = max(len(words), 1)
    return 206.835 - 1.015 * (wc / sents) - 84.6 * (syllables / wc)


def _sentence_count(text: str) -> int:
    if _TEXTSTAT_OK:
        try:
            return max(int(_textstat.sentence_count(text)), 1)
        except Exception:
            pass
    return max(text.count(".") + text.count("!") + text.count("?"), 1)


def _syllable_count(word: str) -> int:
    word = word.lower().strip(".,!?")
    if not word:
        return 1
    vowels = "aeiouy"
    count = 0
    prev_v = False
    for ch in word:
        is_v = ch in vowels
        if is_v and not prev_v:
            count += 1
        prev_v = is_v
    return max(count, 1)


class ImprovedEngagementModel(nn.Module):
    def __init__(self, extra_dim: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained("roberta-base")
        hidden = self.bert.config.hidden_size

        for param in list(self.bert.parameters())[:-24]:
            param.requires_grad = False

        self.text_branch = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.tabular_branch = nn.Sequential(
            nn.Linear(extra_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def mean_pool(self, output, mask):
        emb = output.last_hidden_state
        m = mask.unsqueeze(-1).expand(emb.size()).float()
        return torch.sum(emb * m, 1) / torch.clamp(m.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask, extra_feats):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pool(out, attention_mask)
        fused = torch.cat([self.text_branch(pooled), self.tabular_branch(extra_feats)], dim=1)
        return self.fusion(fused).squeeze()


class Predictor:
    SENTIMENT_COLS = ["sent_neg", "sent_neu", "sent_pos"]
    NLP_COLS = [
        "text_len",
        "word_count",
        "hashtag_count",
        "mention_count",
        "exclamation_count",
        "question_count",
        "emoji_count",
    ]
    ACCOUNT_COLS = ["followers", "following", "num_posts", "is_business_account"]
    ALL_FEATURE_COLS = SENTIMENT_COLS + NLP_COLS + ACCOUNT_COLS

    _CTA_WORDS = {
        "click",
        "link",
        "comment",
        "share",
        "drop",
        "check",
        "follow",
        "like",
        "save",
        "tag",
        "visit",
        "join",
        "sign",
        "get",
    }

    def __init__(
        self,
        model_path: str = str(MODEL_PATH),
        scaler_path: str = str(SCALER_PATH),
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        self.scaler = joblib.load(scaler_path)
        if getattr(self.scaler, "n_features_in_", None) != len(self.ALL_FEATURE_COLS):
            raise ValueError(
                f"Scaler expects {self.scaler.n_features_in_} features, "
                f"but ALL_FEATURE_COLS has {len(self.ALL_FEATURE_COLS)}."
            )

        self.model = ImprovedEngagementModel(extra_dim=self.scaler.n_features_in_)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()

    def _extract_features(
        self,
        text: str,
        followers: int,
        account_type: str,
        following: int,
        num_posts: int,
    ) -> np.ndarray:
        polarity = _safe_polarity(text)

        if polarity > 0.1:
            sn, sne, sp = 0.0, 0.0, 1.0
        elif polarity < -0.1:
            sn, sne, sp = 1.0, 0.0, 0.0
        else:
            sn, sne, sp = 0.0, 1.0, 0.0

        emoji_count = _count_emojis(text)

        return np.array(
            [
                [
                    sn,
                    sne,
                    sp,
                    float(len(text)),
                    float(len(text.split())),
                    float(len(re.findall(r"#\w+", text))),
                    float(len(re.findall(r"@\w+", text))),
                    float(text.count("!")),
                    float(text.count("?")),
                    float(emoji_count),
                    float(followers),
                    float(following),
                    float(num_posts),
                    1.0 if str(account_type).upper() == "BUSINESS" else 0.0,
                ]
            ],
            dtype=np.float32,
        )

    def _top_phrases(self, enc: dict) -> list:
        try:
            with torch.no_grad():
                ao = self.model.bert(
                    input_ids=enc["input_ids"].to(self.device),
                    attention_mask=enc["attention_mask"].to(self.device),
                    output_attentions=True,
                )
            avg_attn = torch.mean(ao.attentions[-1], dim=1).squeeze(0)[0].cpu().numpy()
            tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
            return self._parse_impacts(tokens, avg_attn)
        except Exception:
            return []

    def _parse_impacts(self, tokens: list, weights: np.ndarray) -> list:
        wmap = {}
        cur, sc = "", 0.0
        for tok, w in zip(tokens, weights):
            if tok in ["<s>", "</s>", "<pad>"]:
                continue
            if tok.startswith("Ġ"):
                if cur and len(cur) > 2 and cur.isalpha():
                    wmap[cur] = max(wmap.get(cur, 0.0), sc)
                cur, sc = tok.replace("Ġ", ""), float(w)
            else:
                cur += tok
                sc = max(sc, float(w))
        if cur and len(cur) > 2:
            wmap[cur] = max(wmap.get(cur, 0.0), sc)
        top = sorted(wmap.items(), key=lambda x: x[1], reverse=True)[:5]
        return [{"phrase": p, "score": int(s * 1000)} for p, s in top]

    def _language_analysis(self, text: str, polarity: float) -> dict:
        tl = text.lower()
        words = text.split()
        wc = len(words)

        htags = len(re.findall(r"#\w+", text))
        mntns = len(re.findall(r"@\w+", text))
        qmarks = text.count("?")
        emojis = _count_emojis(text)
        cta_hit = any(w in tl for w in self._CTA_WORDS)

        signals = {
            "cta_detected": cta_hit,
            "hashtag_count": htags,
            "mention_count": mntns,
            "question_count": qmarks,
            "emoji_count": emojis,
        }

        flesch = _flesch(text)
        sentences = _sentence_count(text)
        avg_wlen = sum(len(w.strip(".,!?#@")) for w in words) / max(wc, 1)
        avg_slen = wc / sentences

        has_direct = any(w in tl for w in ["you", "your", "you're", "yours"])
        has_positive = polarity > 0.15
        has_question = qmarks > 0

        lang = 50.0
        lang += min((flesch - 30) * 0.3, 20)
        lang += 10 if has_direct else 0
        lang += 8 if cta_hit else 0
        lang += 5 if has_question else 0
        lang += 5 if has_positive else 0
        lang -= 8 if avg_slen > 25 else 0
        lang -= 5 if avg_wlen > 7 else 0
        lang -= 5 if wc < 5 else 0
        lang = float(np.clip(lang, 0, 100))

        aud = 50.0
        if 40 <= wc <= 150:
            aud += 15
        elif wc < 10:
            aud -= 15
        elif wc > 250:
            aud -= 10

        if 3 <= htags <= 10:
            aud += 12
        elif htags == 0:
            aud -= 8
        elif htags > 20:
            aud -= 5

        if 1 <= emojis <= 5:
            aud += 8
        elif emojis > 10:
            aud -= 5

        aud += 8 if cta_hit else 0
        aud += 5 if has_question else 0
        aud += 7 if has_positive else 0
        aud -= 10 if polarity < -0.3 else 0
        aud = float(np.clip(aud, 0, 100))

        diag = []
        if flesch >= 70:
            diag.append(f"Readability {int(flesch)}/100 — excellent, easy to read")
        elif flesch >= 50:
            diag.append(f"Readability {int(flesch)}/100 — acceptable, moderate complexity")
        else:
            diag.append(f"Readability {int(flesch)}/100 — complex, may lose readers")

        if has_positive:
            diag.append("Positive tone — increases share probability")
        elif polarity < -0.3:
            diag.append("Strong negative tone — reduces organic reach")

        diag.append(
            "Call-to-Action detected — good for driving interactions"
            if cta_hit
            else "No Call-to-Action found — missed engagement trigger"
        )

        if has_direct:
            diag.append("Direct address ('you/your') — personal connection")
        if has_question:
            diag.append("Question present — encourages comment activity")

        if htags == 0:
            diag.append("No hashtags — reduces discoverability significantly")
        elif htags > 20:
            diag.append(f"{htags} hashtags — excessive, looks spammy (optimal: 5–10)")
        else:
            diag.append(f"{htags} hashtag(s) — within recommended range")

        if avg_slen > 25:
            diag.append(f"Avg sentence {avg_slen:.0f} words — too long for mobile")
        if wc < 10:
            diag.append("Text too short — lacks context for algorithm ranking")
        elif wc > 250:
            diag.append(f"Long post ({wc} words) — Instagram prefers shorter content")

        recs = []
        if not cta_hit:
            recs.append("Add a CTA: 'Comment below', 'Share this', 'Click the link in bio'")
        if not has_direct:
            recs.append("Use direct address: start with 'You' or ask 'Have you ever…?'")
        if not has_question:
            recs.append("Add a question at the end to boost comments")
        if flesch < 50:
            recs.append("Shorten sentences — aim for max 15 words per sentence")
        if htags == 0:
            recs.append("Add 5–10 niche hashtags to increase reach by 2–3×")
        elif htags > 20:
            recs.append("Reduce hashtags to 5–10 most relevant ones")
        if emojis == 0 and wc > 15:
            recs.append("Add 1–3 relevant emojis to increase visual engagement")
        if polarity < -0.1 and not has_positive:
            recs.append("Reframe with a positive angle — solutions, not problems")
        if avg_slen > 25:
            recs.append("Break long sentences into 2 shorter ones for mobile readability")
        if wc < 20:
            recs.append("Expand the post — add context, a story, or a concrete example")

        return {
            "language_score": lang,
            "audience_fit": aud,
            "diagnostics": diag[:6],
            "recommendations": recs[:5],
            "signals": signals,
        }

    def predict(
        self,
        text: str,
        followers: int = 1000,
        account_type: str = "CREATOR",
        avg_likes: int | None = None,
        following: int = 500,
        num_posts: int = 100,
        return_explanations: bool = True,
    ) -> dict:
        raw = self._extract_features(text, followers, account_type, following, num_posts)
        feats = torch.tensor(self.scaler.transform(raw), dtype=torch.float).to(self.device)

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )

        with torch.no_grad():
            log_out = self.model(
                enc["input_ids"].to(self.device),
                enc["attention_mask"].to(self.device),
                feats,
            )

        raw_val = float(np.expm1(log_out.item()))
        engagement_score = round(float(np.clip(raw_val * 20, 0, 100)), 1)

        avg_likes_int = int(avg_likes or 0)
        predicted_eng = (
            int(avg_likes_int * (raw_val / 2.0)) if avg_likes_int > 0 else int(raw_val * 50)
        )

        polarity = _safe_polarity(text)
        pos_raw = max(0.0, polarity)
        neg_raw = max(0.0, -polarity)
        neu_raw = max(0.0, 1.0 - pos_raw - neg_raw)

        delta_ok = avg_likes_int > 0
        growth_pct = None
        if delta_ok:
            growth_pct = round(
                ((predicted_eng - avg_likes_int) / max(avg_likes_int, 1)) * 100,
                1,
            )

        lang = self._language_analysis(text, polarity)
        top_phrases = self._top_phrases(enc) if return_explanations else []

        return {
            "engagement_score": engagement_score,
            "predicted_engagement": predicted_eng,
            "sentiment_pos": round(pos_raw, 3),
            "sentiment_neu": round(neu_raw, 3),
            "sentiment_neg": round(neg_raw, 3),
            "delta_ok": delta_ok,
            "growth_percent_vs_avg": growth_pct,
            "signals": lang["signals"],
            "diagnostics": lang["diagnostics"],
            "recommendations": lang["recommendations"],
            "language_score": lang["language_score"],
            "audience_fit_score": lang["audience_fit"],
            "top_phrases": top_phrases,
        }