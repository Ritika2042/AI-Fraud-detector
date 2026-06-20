import re
import unicodedata
from typing import List, Union
import pandas as pd

class TextPreprocessor:
    """Class to handle text cleaning and normalization for scam detection."""
    
    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase
        
    def clean_text(self, text: str) -> str:
        """
        Cleans raw text from conversation transcripts.
        - Handles non-string/NaN inputs by converting to empty string
        - Normalizes Unicode characters
        - Normalizes currency formats (e.g., converting ₹, Rs., to 'rupees')
        - Standardizes URLs and email placeholders
        - Sanitizes special characters and punctuation
        - Collapses duplicate whitespaces
        """
        if not isinstance(text, str):
            return ""
            
        # Normalize Unicode (NFKD decomposes characters, e.g. accented characters)
        text = unicodedata.normalize("NFKD", text)
        
        # Convert to lowercase if configured
        if self.lowercase:
            text = text.lower()
            
        # Standardize common currency symbols and labels
        text = text.replace("₹", " rupees ")
        text = re.sub(r"\brs\.?\b", " rupees ", text)
        text = text.replace("usd", " dollars ").replace("$", " dollars ")
        
        # Standardize URLs
        text = re.sub(r"https?://\S+|www\.\S+", " url_link ", text)
        
        # Standardize Emails
        text = re.sub(r"\S+@\S+\.\S+", " email_address ", text)
        
        # Remove HTML tags if any
        text = re.sub(r"<[^>]*>", " ", text)
        
        # Retain alphanumeric characters, basic spaces, and common symbols like %, !, ?
        text = re.sub(r"[^a-zA-Z0-9_\s!?%]", " ", text)
        
        # Clean up double/multiple spaces
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

    def preprocess_df(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """
        Applies preprocessing to a text column of a pandas DataFrame.
        Returns a new copy of the DataFrame with cleaned text.
        """
        df = df.copy()
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame.")
            
        df[text_column] = df[text_column].astype(str).apply(self.clean_text)
        return df
