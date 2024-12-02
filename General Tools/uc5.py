from transformers import pipeline
import spacy

# Load pre-trained NER model for legal terms
nlp = spacy.load("en_core_web_sm")

# Sample privacy regulation text (GDPR)
privacy_regulation_text = """
The GDPR establishes rules for the protection of personal data in AI systems. 
It requires that AI systems processing personal data ensure user consent, data minimization, and the right to be forgotten. 
The regulation also mandates transparency in how AI systems handle personal data and imposes penalties for non-compliance.
"""

# Initialize the summarization pipeline
summarizer = pipeline("summarization")

# Summarize the privacy regulation
privacy_summary = summarizer(privacy_regulation_text, max_length=60, min_length=30, do_sample=False)
print("Summary of GDPR Privacy Rules: ", privacy_summary[0]['summary_text'])

# Perform NER to extract legal terms related to privacy
doc = nlp(privacy_regulation_text)
for entity in doc.ents:
    print(f"{entity.text}: {entity.label_}")
