import re
import unicodedata
from typing import List, Dict, Any, Set, Tuple

# Static list of common English stopwords to avoid external package dependencies
STOPWORDS: Set[str] = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "himself", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did",
    "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
    "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
    "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will",
    "just", "don", "should", "shouldn", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain",
    "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn", "mustn",
    "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn"
}

# Regex compilation for PII detection
PII_PATTERNS: Dict[str, re.Pattern] = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "PHONE": re.compile(r"\b(?:\+?\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b"),
    "AADHAAR": re.compile(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"),
    "IP_ADDRESS": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
}

# Vocabulary for feature extraction scoring
URGENCY_KEYWORDS = {
    "urgent", "urgently", "immediately", "block", "blocked", "freeze", "frozen",
    "jail", "arrest", "arrested", "court", "lawyer", "police", "cbi", "supreme",
    "warning", "expire", "expired", "deactivate", "deactivated", "verification",
    "unauthorized", "investigation", "penalty", "fine"
}

FINANCIAL_KEYWORDS = {
    "rupees", "dollars", "amount", "transfer", "bank", "account", "fee", "fees",
    "payment", "card", "cash", "credit", "debit", "crypto", "usdt", "bitcoin",
    "funds", "deposit", "money", "transaction", "pay", "invest", "investment", "returns"
}

class NLPPreprocessingPipeline:
    """Modular, highly reusable NLP preprocessing pipeline for conversation transcripts."""

    def __init__(self, stopwords: Set[str] = STOPWORDS, lowercase: bool = True):
        self.stopwords = stopwords
        self.lowercase = lowercase

    def clean_and_normalize(self, text: str) -> str:
        """Applies Unicode normalization, lowercasing, and standardizes spacing/symbols."""
        if not isinstance(text, str):
            return ""

        # Normalize Unicode characters
        text = unicodedata.normalize("NFKD", text)

        if self.lowercase:
            text = text.lower()

        # Normalize common currency labels
        text = text.replace("₹", " rupees ")
        text = re.sub(r"\brs\.?\b", " rupees ", text)
        text = text.replace("$", " dollars ")

        # Sanitize HTML or system logs
        text = re.sub(r"<[^>]*>", " ", text)

        # Collapse multiple spaces/newlines
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def mask_pii(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Masks common PII categories (Email, Phone, Aadhaar, Cards, IPs) using regex.
        Returns the masked text and counts of masked entities.
        """
        masked_text = text
        counts = {}
        
        for pii_type, pattern in PII_PATTERNS.items():
            matches = pattern.findall(masked_text)
            counts[pii_type] = len(matches)
            if matches:
                masked_text = pattern.sub(f" [{pii_type}] ", masked_text)

        # Re-collapse spaces that might have been introduced by substitutions
        masked_text = re.sub(r"\s+", " ", masked_text).strip()
        return masked_text, counts

    def separate_speakers(self, text: str) -> List[Dict[str, str]]:
        """
        Parses transcripts and splits dialogue into structured speaker turns.
        Supports standard tags like '[Scammer]: text' or 'Speaker 1: text'.
        """
        # Matches Speaker formats at start of line or transcript
        pattern = r"(?:^|\n)\s*\[?([A-Za-z0-9\s_]+)\]?\s*:\s*"
        
        splits = list(re.finditer(pattern, text))
        if not splits:
            return [{"speaker": "Unknown", "text": text.strip()}]
            
        turns = []
        for i in range(len(splits)):
            start = splits[i].end()
            end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
            
            speaker = splits[i].group(1).strip()
            turn_text = text[start:end].strip()
            # Clean turn text
            turn_text = re.sub(r"\s+", " ", turn_text)
            
            turns.append({"speaker": speaker, "text": turn_text})
            
        return turns

    def segment_conversation(self, turns: List[Dict[str, str]], window_size: int = 3, overlap: int = 1) -> List[List[Dict[str, str]]]:
        """
        Segments a list of dialogue turns into overlapping conversation windows.
        Useful for feeding local dialogue context into transformers.
        """
        if not turns:
            return []
            
        segments = []
        step = window_size - overlap
        if step <= 0:
            step = 1

        for i in range(0, len(turns), step):
            window = turns[i : i + window_size]
            segments.append(window)
            if i + window_size >= len(turns):
                break
                
        return segments

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes text into individual lowercase word tokens, omitting punctuation."""
        return re.findall(r"\b\w+\b", text.lower())

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Filters out standard stopwords from a list of tokens."""
        return [t for t in tokens if t not in self.stopwords]

    def extract_features(self, text: str, turns: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Extracts semantic and structural features from the text and dialogue turns.
        Returns a dictionary of numeric features.
        """
        tokens = self.tokenize(text)
        total_chars = len(text)
        total_words = len(tokens)
        
        # Keyword counting
        urgency_count = sum(1 for t in tokens if t in URGENCY_KEYWORDS)
        financial_count = sum(1 for t in tokens if t in FINANCIAL_KEYWORDS)
        
        # Calculate structural characteristics
        digits_count = sum(1 for c in text if c.isdigit())
        exclamations_count = text.count("!")
        questions_count = text.count("?")
        
        # Speaker metrics
        speakers = {turn["speaker"] for turn in turns if turn["speaker"] != "Unknown"}
        
        return {
            "char_length": total_chars,
            "word_count": total_words,
            "digit_density": digits_count / total_chars if total_chars > 0 else 0.0,
            "exclamation_count": exclamations_count,
            "question_count": questions_count,
            "urgency_score": urgency_count,
            "financial_score": financial_count,
            "speaker_count": len(speakers),
            "turn_count": len(turns)
        }

    def process(self, raw_transcript: str) -> Dict[str, Any]:
        """
        Runs the complete preprocessing pipeline end-to-end.
        Returns a structured dictionary ready for feature injection or fine-tuning.
        """
        # 1. Separate speakers first (preserves raw transcript structure)
        turns = self.separate_speakers(raw_transcript)
        
        # 2. Clean and mask PII for each turn and the overall text
        cleaned_overall = self.clean_and_normalize(raw_transcript)
        masked_overall, pii_counts = self.mask_pii(cleaned_overall)
        
        cleaned_turns = []
        for turn in turns:
            c_text = self.clean_and_normalize(turn["text"])
            m_text, _ = self.mask_pii(c_text)
            cleaned_turns.append({
                "speaker": turn["speaker"],
                "text": m_text
            })

        # 3. Segment conversations
        segments = self.segment_conversation(cleaned_turns)
        
        # 4. Tokenization & Stopwords on the masked text
        tokens = self.tokenize(masked_overall)
        filtered_tokens = self.remove_stopwords(tokens)
        
        # 5. Extract metadata features
        features = self.extract_features(masked_overall, cleaned_turns)
        
        return {
            "cleaned_transcript": masked_overall,
            "turns": cleaned_turns,
            "segments": segments,
            "tokens": tokens,
            "filtered_tokens": filtered_tokens,
            "pii_counts": pii_counts,
            "features": features
        }
