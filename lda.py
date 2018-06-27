import pandas as pd

data = pd.read_csv('month.csv');
data_text = data['text']
documents = data_text
# print(len(documents))
# print(documents[:5])


import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)

import nltk
nltk.download('wordnet')


def lemmatize_stemming(text):
	stemmer = SnowballStemmer("english")
	#text = text.decode('utf-8')
	return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    text = text.decode('utf-8')
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result



processed_docs=[]
i = 0
while i < len(documents)-1:
	if documents[i] !=[]:
		processed_docs.append(preprocess((documents[i])))
	else:
		processed_docs.append("N/A")
	#print i
	i=i+1


dictionary = gensim.corpora.Dictionary(processed_docs)

# count = 0
# for k, v in dictionary.iteritems():
#     print(k, v)
#     count += 1
#     if count > 10:
#         break

dictionary.filter_extremes(no_below=15, no_above=0.6, keep_n=10000)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# bow_doc_4310 = bow_corpus[0]

# for i in range(len(bow_doc_4310)):
#     print("Word {} (\"{}\") appears {} time.".format(bow_doc_4310[i][0], 
#                                                dictionary[bow_doc_4310[i][0]], 
# bow_doc_4310[i][1]))


from gensim import corpora, models

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

from pprint import pprint

# for doc in corpus_tfidf:
#     pprint(doc)
#     break


# lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=20, id2word=dictionary, passes=5, workers=5)


# for idx, topic in lda_model.print_topics(-1):
#     print('Topic: {} \nWords: {}'.format(idx, topic))

#Also use TDIF topic modelling in future

lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=32, id2word=dictionary, passes=2, workers=2)

for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))

#sample doc test score

# for index, score in sorted(lda_model_tfidf[bow_corpus[0]], key=lambda tup: -1*tup[1]):
#     print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 30)))

from gensim.models.ldamodel import CoherenceModel 
from gensim.models.ldamodel import LdaModel

coherence_model_lda = CoherenceModel(model=lda_model_tfidf, texts=processed_docs, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()

def compute_coherence_values(dictionary, corpus, texts, limit, start, step):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model=LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=processed_docs, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=bow_corpus, texts=processed_docs, start=2, limit=47, step=6)
# Show graph
import matplotlib.pyplot as plt
limit=47; start=2; step=6;
x = range(start, limit, step)
fig = plt.figure()
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.savefig("coherence.png")
#plt.show()

