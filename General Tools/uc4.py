from transformers import AutoTokenizer, AutoModel
import numpy as np
from scipy.spatial.distance import cosine
import torch

# Load Hugging Face model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Compact model for embeddings
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Sample documents
doc1 = """
1. Providers of general-purpose AI models shall:

(a) draw up and keep up-to-date the technical documentation of the model, including its training and testing process and the results of its evaluation, which shall contain, at a minimum, the information set out in Annex XI for the purpose of providing it, upon request, to the AI Office and the national competent authorities;

(b) draw up, keep up-to-date and make available information and documentation to providers of AI systems who intend to integrate the general-purpose AI model into their AI systems. Without prejudice to the need to observe and protect intellectual property rights and confidential business information or trade secrets in accordance with Union and national law, the information and documentation shall:

(i) enable providers of AI systems to have a good understanding of the capabilities and limitations of the general-purpose AI model and to comply with their obligations pursuant to this Regulation; and

(ii) contain, at a minimum, the elements set out in Annex XII;

(c) put in place a policy to comply with Union law on copyright and related rights, and in particular to identify and comply with, including through state-of-the-art technologies, a reservation of rights expressed pursuant to Article 4(3) of Directive (EU) 2019/790;

(d) draw up and make publicly available a sufficiently detailed summary about the content used for training of the general-purpose AI model, according to a template provided by the AI Office.

2. The obligations set out in paragraph 1, points (a) and (b), shall not apply to providers of AI models that are released under a free and open-source licence that allows for the access, usage, modification, and distribution of the model, and whose parameters, including the weights, the information on the model architecture, and the information on model usage, are made publicly available. This exception shall not apply to general-purpose AI models with systemic risks.

3. Providers of general-purpose AI models shall cooperate as necessary with the Commission and the national competent authorities in the exercise of their competences and powers pursuant to this Regulation.

4. Providers of general-purpose AI models may rely on codes of practice within the meaning of Article 56 to demonstrate compliance with the obligations set out in paragraph 1 of this Article, until a harmonised standard is published. Compliance with European harmonised standards grants providers the presumption of conformity to the extent that those standards cover those obligations. Providers of general-purpose AI models who do not adhere to an approved code of practice or do not comply with a European harmonised standard shall demonstrate alternative adequate means of compliance for assessment by the Commission.

5. For the purpose of facilitating compliance with Annex XI, in particular points 2 (d) and (e) thereof, the Commission is empowered to adopt delegated acts in accordance with Article 97 to detail measurement and calculation methodologies with a view to allowing for comparable and verifiable documentation.

6. The Commission is empowered to adopt delegated acts in accordance with Article 97(2) to amend Annexes XI and XII in light of evolving technological developments.

7. Any information or documentation obtained pursuant to this Article, including trade secrets, shall be treated in accordance with the confidentiality obligations set out in Article 78.
"""
doc2 = """
The most common goal for exploiting GenAI capabilities during this period was to shape or influence
public opinion (27% of all reported cases). In those instances, we saw actors deploy a range of tactics
to distort the public perception of political realities. These included impersonating public figures,
using synthetic digital personas to simulate grassroots support for or against a cause (astroturfing)
and creating falsified media.
The majority of cases in our dataset involved the generation of emotionally charged synthetic
images around politically divisive topics, such as war, societal unrest or economic decline. For example,
images and ads shared during electoral campaigns in the US, Canada and New Zealand by party
staffers14 and state-sponsored actors15 alike frequently depicted scenes of urban decay, homelessness
and insecurity. Purportedly leaked AI-generated videos and audio clips of politicians falsely endorsing
controversial political positions — such as Vladimir Putin declaring martial law after Ukrainian forces
entered Russian territory — and privately attacking their political opponents were also common
"""

# Step 1: Sentence Tokenization
import spacy
nlp = spacy.load("en_core_web_sm")

def extract_sentences(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

sentences_doc1 = extract_sentences(doc1)
sentences_doc2 = extract_sentences(doc2)

# Step 2: Generate Embeddings
def generate_embeddings(sentences):
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**inputs)
    # Use mean pooling to get sentence embeddings
    embeddings = model_output.last_hidden_state.mean(dim=1)
    return embeddings.numpy()

embeddings_doc1 = generate_embeddings(sentences_doc1)
embeddings_doc2 = generate_embeddings(sentences_doc2)

# Step 3: Compute Cosine Similarity
def compute_similarity(embeddings1, embeddings2):
    similarity_matrix = np.zeros((len(embeddings1), len(embeddings2)))
    for i, emb1 in enumerate(embeddings1):
        for j, emb2 in enumerate(embeddings2):
            similarity_matrix[i, j] = 1 - cosine(emb1, emb2)
    return similarity_matrix

similarity_matrix = compute_similarity(embeddings_doc1, embeddings_doc2)

# Step 4: Identify Causal Links
threshold = 0.5  # Adjust the threshold as needed
causal_links = []
for i, sent1 in enumerate(sentences_doc1):
    for j, sent2 in enumerate(sentences_doc2):
        if similarity_matrix[i, j] >= threshold:
            causal_links.append((sent1, sent2, similarity_matrix[i, j]))

# Step 5: Output Results
print("Extracted Causal Relations:")
for cause, effect, score in causal_links:
    print(f"Cause: {cause}")
    print(f"Effect: {effect}")
    print(f"Similarity Score: {score:.2f}")
    print("-" * 50)
