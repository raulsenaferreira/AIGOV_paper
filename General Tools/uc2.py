from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example texts from two different regulations
eu_ai_act = """
The European Union AI Act introduces a risk-based regulatory framework for AI systems. It classifies AI systems into different risk categories such as unacceptable risk, high risk, and limited risk. For high-risk systems, specific requirements such as human oversight, transparency, and risk management are imposed.
"""
us_algorithmic_accountability_act = """
The Algorithmic Accountability Act in the U.S. mandates companies to assess the impacts of automated decision systems, particularly focusing on bias, discrimination, and data security. Companies are required to audit algorithms to ensure transparency and fairness in decision-making processes.
"""

# Initialize summarization pipeline
summarizer = pipeline("summarization")

# Summarize both texts
eu_summary = summarizer(eu_ai_act, max_length=60, min_length=30, do_sample=False)
us_summary = summarizer(us_algorithmic_accountability_act, max_length=60, min_length=30, do_sample=False)

print("Summary of the EU AI Act: ", eu_summary[0]['summary_text'])
print("Summary of the US Algorithmic Accountability Act: ", us_summary[0]['summary_text'])

# Keyword extraction and similarity analysis
documents = [eu_ai_act, us_algorithmic_accountability_act]
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

# Compute cosine similarity between the two regulations
cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
print(f"Cosine Similarity between EU and US regulations: {cosine_sim[0][0]}")
