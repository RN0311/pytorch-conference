import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer


question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

query = "Give two research papers with a summary on multi-objective optimization for fair and unbiased machine learning models."

documents = [
    "Minimax Pareto Fairness: A Multi Objective Perspective. Summary: This paper proposes a fairness criterion where a classifier achieves minimax risk and is Pareto-efficient with respect to all groups, avoiding unnecessary harm while optimizing for fairness and accuracy. https://proceedings.mlr.press/v119/martinez20a.html",
    "Fair AutoML through Multi-Objective Optimization. Summary: This paper introduces a novel AutoML framework that naturally supports multi-objective optimization to generate higher-dimensional Pareto fronts depicting trade-offs among multiple model fairness and accuracy measures. https://openreview.net/pdf?id=KwLWsm5idpR"
]

encoded_query = question_tokenizer(query, return_tensors='pt')
encoded_docs = [context_tokenizer(doc, return_tensors='pt') for doc in documents]

query_embedding = question_encoder(**encoded_query).pooler_output
doc_embeddings = [context_encoder(**doc).pooler_output for doc in encoded_docs]

cosine_sim = torch.nn.CosineSimilarity(dim=1)
similarities = [cosine_sim(query_embedding, doc_embedding).item() for doc_embedding in doc_embeddings]

ranked_docs = sorted(zip(similarities, documents), reverse=True)

most_relevant_doc = ranked_docs[0][1]
print(f"Most relevant document: {most_relevant_doc}")
