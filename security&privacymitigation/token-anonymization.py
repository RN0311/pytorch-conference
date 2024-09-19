import torch
from transformers import BertTokenizer, BertForMaskedLM

data = [
    "John Doe lives at 123 Main St.",
    "You can reach Alice at alice@example.com.",
    "Bob's phone number is (555) 123-4567."
]

# Predefined patterns for sensitive information
anonymization_patterns = {
    '<NAME>': r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b',   
    '<EMAIL>': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Matches email addresses
    '<PHONE>': r'\(\d{3}\)\s?\d{3}-\d{4}',  # Matches phone numbers like (555) 123-4567
    '<ADDRESS>': r'\d+\s[A-Za-z]+\s(St|Ave|Blvd|Rd)',  # Matches simple addresses like '123 Main St'
}


import re

def anonymize_text(text, patterns):
  '''
  This function replaces sensitive information in the text based on the patterns.
  '''
  for token, pattern in patterns.items():
      text = re.sub(pattern, token, text)
  return text

# Anonymize the training data
anonymized_data = [anonymize_text(sentence, anonymization_patterns) for sentence in data]

print("Original data:")
print(data)
print("\nAnonymized data:")
print(anonymized_data)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer(anonymized_data, return_tensors='pt', padding=True, truncation=True)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

outputs = model(**inputs, labels=inputs['input_ids'])


loss = outputs.loss
print(f"\nLoss: {loss.item()}")
# The model can be further fine-tuned based on the anonymized dataset.