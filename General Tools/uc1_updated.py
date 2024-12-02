import requests
from transformers import pipeline
import spacy

# Load SpaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize Hugging Face summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)

def download_eu_ai_act():
    """
    Downloads the full text of the EU AI Act from a specified source URL.
    Replace the URL below with the actual location of the EU AI Act.
    """
    url = "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689"  # Replace with actual URL
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception("Failed to download EU AI Act. Please check the URL or internet connection.")

def split_text(text, max_chars=1000):
    """
    Splits the text into manageable chunks for summarization.
    """
    chunks = []
    while len(text) > max_chars:
        split_point = text.rfind(".", 0, max_chars) + 1
        if split_point == 0:
            split_point = max_chars
        chunks.append(text[:split_point].strip())
        text = text[split_point:]
    chunks.append(text.strip())
    return chunks

# Download the EU AI Act text
try:
    regulatory_text = download_eu_ai_act()
except Exception as e:
    print(e)
    regulatory_text = """
    Sample fallback text in case downloading fails.
    """  # Replace this with an offline version if necessary

# Split the text into chunks for summarization
text_chunks = split_text(regulatory_text)

# Summarize each chunk and perform NER
summaries = []
entities = []

for i, chunk in enumerate(text_chunks):
    # Summarize the chunk
    summary = summarizer(chunk, max_length=60, min_length=30, do_sample=False)[0]['summary_text']
    summaries.append(summary)
    print(f"Summary of Section {i+1}: {summary}")
    
    # Perform NER on each chunk
    doc = nlp(chunk)
    section_entities = {entity.text: entity.label_ for entity in doc.ents}
    entities.append(section_entities)
    print(f"Entities in Section {i+1}: {section_entities}")

print("\n--- Combined Summaries ---")
print(" ".join(summaries))
print("\n--- Named Entities Found ---")
for i, section_entities in enumerate(entities):
    print(f"Entities in Section {i+1}: {section_entities}")