import re
import tiktoken
from typing import List, Dict

def clean_text(text: str) -> str:
    """Clean and normalize text."""
    text = text.lower()
    text = re.sub(r"\$", "", text)
    text = re.sub(r"(?s).*#### ", "", text)
    text = re.sub(r"\.$", "", text)
    text = re.sub(r",", "", text)
    return text if text else "-1000000000"

def extract_value(text: str) -> str:
    """Extract numerical value from text."""
    pattern = r"(-?[$0-9.,]{2,})|(-?[0-9]+)"
    matches = re.findall(pattern, text)
    if matches:
        for match_groups in matches[::-1]:
            for group in match_groups:
                if group:
                    return clean_text(group)
    return "-1000000000"

def count_tokens(text: str, model: str = "p50k_base") -> int:
    """Count tokens in text."""
    encoding = tiktoken.get_encoding(model)
    return len(encoding.encode(text))