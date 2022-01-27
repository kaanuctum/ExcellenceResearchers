from HelperObjects import *
import nltk
import gensim
import re
from tqdm import tqdm

"""
 These are the helper methods used 
"""
sql.connect()
STOPWORDS = nltk.corpus.stopwords.words('english')
STOPWORDS.extend(['however', 'compare', 'differ', 'find', 'year', 'use'])


# just gives the eid and the raw abstracts of all articles
def get_raw_data():
    sql.connect()
    sql.cur.execute(
        "SELECT article_id, raw_abstract FROM article_abstracts WHERE raw_abstract NOTNULL ORDER BY article_id")
    return sql.cur.fetchall()


def clean_copyright(raw_abstracts):
    return [(eid, clean_copyright_just_text(abstract)) for (eid, abstract) in tqdm(raw_abstracts)]


def clean_copyright_just_text(raw_abstract, copyright_threshold=50):
    # if the copyright symbol is not present in the text, there isn't much we can do to find the copyright
    raw_abstract = raw_abstract.strip()
    if raw_abstract.find('©') == -1: return raw_abstract

    phrases_to_delete = ['All rights reserved', 'GmbH & Co. KGaA', 'All Rights Reserved']

    cleaned = ''
    abstract = raw_abstract.replace('B.V', 'BV')
    for phrase in phrases_to_delete:
        abstract = abstract.replace(phrase, '')
    abstract = abstract.replace('The Author(s)', 'The Authors')
    # some copyright sentences are not separated properly

    # add a space to all the dots are followed by a capitol letter
    abstract = re.sub(r'[.](?=[A-Z])(?=[^.])', '. ', abstract).strip()

    # find all the upper case letters or numbers that follow a small case letter without a space
    # abstract = re.sub(r'([a-z](?=[A-Z])(?=[^\s-]))', r'\1. ', abstract)

    # find all the beginnings where the first char is the copyright symbol, followed by some space if need be (usually 1)
    # then having a word that has no uppercase in it(either all small or small plus numbers) then having a capital letter
    # next to it that is not seperated by a space, and itself is not followed by a space\
    # e.g. working:
    # © 2019This paper
    # © asdThis paper
    # not working:
    # © 2019 This paper
    # © 2019.This paper
    abstract = re.sub(r'([a-z0-9](?=[A-Z])(?=[^\s-]))', r'\1. ', abstract)

    # if there is a separation where only integers follow after an copyright sign, this is the date, the rest is also
    # a part of the copy right info, can be removed
    abstract = re.sub(r"(© [0-9]*)(\.)", r"\1", abstract)
    if abstract.find('©') > 0:
        abstract = abstract.replace('©', '. ©')

    # because there are no consistency between the different publications and their copyrights, there will be some cases
    # where the first line is dropped, so that in any other cases the copyright info does not creep into the
    # lemmatized version. This is a sacrifice I am willing to make


    # iterate over the sentences in the abstract
    for i in abstract.split('.'):# nltk.sent_tokenize(abstract):
        i = i.strip()
        if i.find('©') == 0:
            # assume all of the sentence is just copyright info
            continue
        else:
            cleaned += i + ' '
    return cleaned


def remove_punctuation(text):
    # Remove punctuation
    return re.sub(r'[,\.!?]', '', text)


# taken from https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in STOPWORDS] for doc in texts]


def make_bigrams(texts, bigram_mod):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts, bigram_mod, trigram_mod):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


def lemmatization(texts, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in tqdm(texts):
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


def sent_to_words(sentences):
    output = list()
    for sentence in sentences:
        output.append(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
    return output


def bow_to_sentence(word_lists, id2word):
    output = list()
    for word_list in word_lists:
        sen = ''
        for word in word_list:
            try:
                sen += (' ' + id2word[word[0]])
            except KeyError:
                continue
        output.append(sen.strip())
    return output


