"""
This example uses LexRank (https://www.aaai.org/Papers/JAIR/Vol22/JAIR-2214.pdf)
to create an extractive summarization of a long document.

The document is splitted into sentences using NLTK, then the sentence embeddings are computed. We
then compute the cosine-similarity across all possible sentence pairs.

We then use LexRank to find the most central sentences in the document, which form our summary.

Input document: First section from the English Wikipedia Section
Output summary:
Located at the southern tip of the U.S. state of New York, the city is the center of the New York metropolitan area, the largest metropolitan area in the world by urban landmass.
New York City (NYC), often called simply New York, is the most populous city in the United States.
Anchored by Wall Street in the Financial District of Lower Manhattan, New York City has been called both the world's leading financial center and the most financially powerful city in the world, and is home to the world's two largest stock exchanges by total market capitalization, the New York Stock Exchange and NASDAQ.
New York City has been described as the cultural, financial, and media capital of the world, significantly influencing commerce, entertainment, research, technology, education, politics, tourism, art, fashion, and sports.
If the New York metropolitan area were a sovereign state, it would have the eighth-largest economy in the world.
"""
import nltk
from sentence_transformers import SentenceTransformer, util
import numpy as np
from LexRank import degree_centrality_scores


nltk.download('punkt')
model = SentenceTransformer('all-MiniLM-L6-v2')

# Our input document we want to summarize
# As example, we take the first section from Wikipedia
document = """
In my thesis, I am going to explore the term digital fashion, as it is becoming more and more relevant. The exploration is going to be done with the use of participatory design methods, including interviewees, having strong interest in fashion in general. Interviews will be done in two separate ways. Once by the participants themselves remotely and once face to face. Both times a self-created web app is used. First participants will be introduced to the topic, afterwards they create digital clothing items interactively and also vote on the items others created. Doing so, I hope to capture the broader idea, participants have of digital fashion. Creating the digital clothing items is done by either DALL·E-2 or Stable-Diffusion translating text to image, those artificial neural networks will be part of my generative toolset. Eventually, the created items will be processed using a technology called Dreambooth, synthesising them in different contexts. In the process, I will further emphasise the distinctive look of digital fashion. Meanwhile, I will question participatory design critically. Asking if participatory design might be a way to create more attachment to and understanding of digital products.
"""

#Split the document into sentences
sentences = nltk.sent_tokenize(document)
print("Num sentences:", len(sentences))

#Compute the sentence embeddings
embeddings = model.encode(sentences, convert_to_tensor=True)

#Compute the pair-wise cosine similarities
cos_scores = util.cos_sim(embeddings, embeddings).numpy()

#Compute the centrality for each sentence
centrality_scores = degree_centrality_scores(cos_scores, threshold=None)

#We argsort so that the first element is the sentence with the highest score
most_central_sentence_indices = np.argsort(-centrality_scores)


#Print the 5 sentences with the highest scores
print("\n\nSummary:")
for idx in most_central_sentence_indices[0:5]:
    print(sentences[idx].strip())