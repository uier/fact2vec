import torch
from training import FactEmbedder
import numpy as np
import sys

sys.modules['FactEmbedder'] = FactEmbedder

# net = torch.load('model/fact2vec.pth')
net = torch.load('./net_300.pth')

from sklearn.metrics.pairwise import cosine_similarity


prompt = "from 2006 to 2010"
texts = [
    prompt,
  "The difference between Action and Drama regarding to their Count of Records is 15 when Year is 2006.",
  "The difference between Action and Drama regarding to their Count of Records is 9 when Year is 2007.",
  "The difference between Action and Drama regarding to their Count of Records is 2 when Year is 2008.",
  "The difference between Action and Drama regarding to their Count of Records is 1 when Year is 2009.",
  "The difference between Action and Drama regarding to their Count of Records is 1 when Year is 2010.",
  "The Action accounts for 20.00% of the Count of Records when Year is 2006.",
  "The Drama accounts for 80.00% of the Count of Records when Year is 2006.",
  "The Action accounts for 33.33% of the Count of Records when Year is 2007.",
  "The Drama accounts for 66.67% of the Count of Records when Year is 2007.",
  "The Action accounts for 43.75% of the Count of Records when Year is 2008.",
  "The Drama accounts for 56.25% of the Count of Records when Year is 2008.",
  "The Action accounts for 53.85% of the Count of Records when Year is 2009.",
  "The Drama accounts for 46.15% of the Count of Records when Year is 2009.",
  "The Action accounts for 57.14% of the Count of Records when Year is 2010.",
  "The Drama accounts for 42.86% of the Count of Records when Year is 2010.",
  "The 2006 accounts for 15.62% of the Count of Records when Genre is Action.",
  "The 2007 accounts for 28.12% of the Count of Records when Genre is Action.",
  "The 2008 accounts for 21.88% of the Count of Records when Genre is Action.",
  "The 2009 accounts for 21.88% of the Count of Records when Genre is Action.",
  "The 2010 accounts for 12.50% of the Count of Records when Genre is Action.",
  "The 2006 accounts for 35.71% of the Count of Records when Genre is Drama.",
  "The 2007 accounts for 32.14% of the Count of Records when Genre is Drama.",
  "The 2008 accounts for 16.07% of the Count of Records when Genre is Drama.",
  "The 2009 accounts for 10.71% of the Count of Records when Genre is Drama.",
  "The 2010 accounts for 5.36% of the Count of Records when Genre is Drama.",
  "In the Count of Records ranking of different Year(s), the top three Year(s) are 2007, followed by 2008, then 2009 when Genre is Action.",
  "In the Count of Records ranking of different Year(s), the last three Year(s) are 2010, followed by 2006, then 2009 when Genre is Action.",
  "In the Count of Records ranking of different Year(s), the top three Year(s) are 2006, followed by 2007, then 2008 when Genre is Drama.",
  "In the Count of Records ranking of different Year(s), the last three Year(s) are 2010, followed by 2009, then 2008 when Genre is Drama.",
  "The minimum value of the Count of Records is Action when Year is 2006.",
  "The maximum value of the Count of Records is Drama when Year is 2006.",
  "The minimum value of the Count of Records is Action when Year is 2007.",
  "The maximum value of the Count of Records is Drama when Year is 2007.",
  "The minimum value of the Count of Records is Action when Year is 2008.",
  "The maximum value of the Count of Records is Drama when Year is 2008.",
  "The minimum value of the Count of Records is Drama when Year is 2009.",
  "The maximum value of the Count of Records is Action when Year is 2009.",
  "The minimum value of the Count of Records is Drama when Year is 2010.",
  "The maximum value of the Count of Records is Action when Year is 2010.",
  "The minimum value of the Count of Records is 2010 when Genre is Action.",
  "The maximum value of the Count of Records is 2007 when Genre is Action.",
  "The minimum value of the Count of Records is 2010 when Genre is Drama.",
  "The maximum value of the Count of Records is 2006 when Genre is Drama.",
  "The increasing trend of Count of Records over Year(s) when Genre is Action from 2006 to 2007.",
  "The decreasing trend of Count of Records over Year(s) when Genre is Action from 2007 to 2010.",
  "The wavering trend of Count of Records over Year(s) when Genre is Action from 2006 to 2010.",
  "The decreasing trend of Count of Records over Year(s) when Genre is Drama from 2006 to 2010."
]


embeddings = net(texts).cpu().detach().numpy()

prompt_embedding = embeddings[0]
sentence_embeddings = embeddings[1:]

print('p', prompt_embedding.shape)
for i in range(len(sentence_embeddings)):
  print(i, sentence_embeddings[i].shape)
#   if sentence_embeddings[i].shape != prompt_embedding.shape:
#     sentence_embeddings[i] = sentence_embeddings[i].reshape(prompt_embedding.shape)

similarities = cosine_similarity([prompt_embedding], sentence_embeddings)[0]

for i, similarity in enumerate(similarities):
    print(f"Sentence: {texts[i+1]} - Similarity: {similarity}")

output = [[texts[i+1], float(similarities[i])] for i in range(len(similarities))]
output = sorted(output, key=lambda x: x[1], reverse=True)

import json
with open("output.json", "w") as f:
    f.write(json.dumps(output, indent=2))