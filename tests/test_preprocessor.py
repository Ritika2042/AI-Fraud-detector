import pandas as pd
from src.data_pipeline.preprocessor import TextPreprocessor

def test_clean_text_basic():
    preprocessor = TextPreprocessor(lowercase=True)
    assert preprocessor.clean_text("Hello World!") == "hello world!"

def test_clean_text_currency():
    preprocessor = TextPreprocessor(lowercase=True)
    # Checks conversion of Rupee symbol
    assert "rupees" in preprocessor.clean_text("Please transfer ₹15000.")
    # Checks conversion of Rs abbreviation
    assert "rupees" in preprocessor.clean_text("Payment of Rs. 500 is pending.")
    # Checks conversion of Dollar symbol
    assert "dollars" in preprocessor.clean_text("Send $100.")

def test_clean_text_urls_and_emails():
    preprocessor = TextPreprocessor(lowercase=True)
    text = "Visit http://secure-update-sbi.com or email support@sbi.com"
    cleaned = preprocessor.clean_text(text)
    assert "url_link" in cleaned
    assert "email_address" in cleaned

def test_clean_text_spaces():
    preprocessor = TextPreprocessor(lowercase=True)
    assert preprocessor.clean_text("This   is   a    test  ") == "this is a test"

def test_clean_text_invalid_inputs():
    preprocessor = TextPreprocessor(lowercase=True)
    assert preprocessor.clean_text(None) == ""
    assert preprocessor.clean_text(12345) == ""

def test_preprocess_df():
    preprocessor = TextPreprocessor(lowercase=True)
    df = pd.DataFrame({"text": ["Hello ₹500", "World!"]})
    processed_df = preprocessor.preprocess_df(df, text_column="text")
    assert processed_df["text"].iloc[0] == "hello rupees 500"
    assert processed_df["text"].iloc[1] == "world!"
