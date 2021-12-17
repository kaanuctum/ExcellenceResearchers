from HelperObjects import *
import spacy
import nltk
import gensim
import pandas as pd
import re
from tqdm import tqdm
import pickle
import time
"""
 These are the helper methods used 
"""
sql.connect()
STOPWORDS = nltk.corpus.stopwords.words('english')
STOPWORDS.extend(['however', 'compare', 'differ', 'find', 'year'])


# just gives the eid and the raw abstracts of all articles
def get_raw_data():
    sql.connect()
    sql.cur.execute(
        "SELECT article_id, raw_abstract FROM article_abstracts WHERE raw_abstract NOTNULL ORDER BY article_id")
    return sql.cur.fetchall()


def clean_copyright(raw_abstracts):
    return [(eid, clean_copyright_just_text(abstract)) for (eid, abstract) in tqdm(raw_abstracts)]


def clean_copyright_just_text(raw_abstract, copyright_threshold=50):
    # if the copyright data is not present in the text, there isn't much we can do to find the copyright
    raw_abstract = raw_abstract.strip()
    if raw_abstract.find('©') == -1: return raw_abstract

    phrases_to_delete = ['All rights reserved.', 'GmbH & Co. KGaA', 'All Rights Reserved']

    cleaned = ''
    abstract = raw_abstract.replace('B.V', 'BV')
    for phrase in phrases_to_delete:
        abstract = abstract.replace(phrase, '')

    # some copyright sentences are not separated properly

    # replace all the dots are followed by a capitol letter
    abstract = re.sub(r'[.](?=[A-Z])(?=[^.])', '. ', abstract).strip()

    # find all the upper case letters that follow a small case letter without a space
    abstract = re.sub(r'([a-z](?=[A-Z])(?=[^\s-]))', r'\1. ', abstract)

    # if there is a separation where only integers follow after an copyright sign, this is the date, the rest is also
    # a part of the copy right info, can be removed
    abstract = re.sub(r"(© [0-9]*)(\.)", r"\1", abstract)
    if abstract.find('©') > 0:
        abstract = abstract.replace('©', '. ©')

    # iterate over the sentences in the abstract
    for i in abstract.split('.'):# nltk.sent_tokenize(abstract):
        i = i.strip()
        if i.find('©') == 0 and len(i) < copyright_threshold:
            # if the symbol is the first char and the sentence is less then 50 chars
            # assume all the sentence is just copyright info
            continue
        else:
            cleaned += i




    # this is a less elegant solution, but it at this point I would rather cut too much from the text
    # rather than leave too much copyright in it
    # after a point the amount of If-cases I have to open would be the same as if I just cleaned them by hand
    dirty = cleaned.find("©") >= 0
    if dirty:
        whole = ''
        parts = cleaned.split('.')
        for part in parts:
            if part.strip().find("©") != 0:
                whole += part.strip() + ' '
            else:
                continue  # print(part)
        cleaned = whole
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


def words_to_sent(word_lists):
    output = list()
    for word_list in word_lists:
        sen = ''
        for word in word_list:
            sen += (' ' + word)
        output.append(sen.strip())
    return output


def parse_words():
    grand_beginning = time.time()
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

    start = time.time()
    print("parsing abstracts")
    raw_data = get_raw_data()
    df = pd.DataFrame(raw_data)
    df.columns = ['id', 'raw_data']
    print(f"Raw data gathered in {time.time() - start} seconds")

    start = time.time()
    copy_right_removed = df['raw_data'].map(lambda x: remove_punctuation(clean_copyright_just_text(x)).lower()).tolist()
    print(f"Copyright removed in {time.time() - start} seconds")

    start = time.time()
    data_words = sent_to_words(copy_right_removed)
    print(f"words split in {time.time() - start} seconds")

    start = time.time()
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
    # trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    # trigram_mod = gensim.models.phrases.Phraser(trigram)
    print(f"Bigrams built in {time.time() - start} seconds")

    start = time.time()
    print("lemmatizing")
    # Remove Stop Words
    data_words_no_stops = remove_stopwords(data_words)
    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_no_stops, bigram_mod=bigram_mod)
    data_lemmatized = lemmatization(data_words_bigrams, nlp=nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    df['lemmatized'] = words_to_sent(data_lemmatized)
    print(f"words lemmatized in {time.time() - start} seconds")

    start = time.time()
    id2word = gensim.corpora.Dictionary(data_lemmatized)  # Create Dictionary
    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    # id2word.filter_extremes(no_below=20, no_above=0.5)
    corpus = [id2word.doc2bow(text) for text in data_lemmatized]  # View
    df['bow'] = corpus
    print(f"Corpus created in {time.time() - start} seconds")

    print('Number of unique tokens: %d' % len(id2word))
    print('Number of documents: %d' % len(corpus))

    print(f"Writing into database")
    progressbar = tqdm(total=len(df))
    for i, row in df.iterrows():
        sql.cur.execute('UPDATE article_abstracts SET cleaned_abstract_2=? WHERE article_id=?',
                        (row['lemmatized'], row['id']))
        sql.conn.commit()
        progressbar.update(1)
    progressbar.close()

    # save preliminary results
    pickle.dump(id2word, open(pathManager.get_paths()['id2word'], 'wb'))
    pickle.dump(data_lemmatized, open(pathManager.get_paths()['data_lemmatized'], 'wb'))
    pickle.dump(corpus, open(pathManager.get_paths()['corpus'], 'wb'))

    df.to_pickle(pathManager.get_paths()['df_parsed_articles'])

    print(f"Everything parsed in {time.time() - grand_beginning} seconds")
    return id2word, data_lemmatized, corpus, df


if __name__ == '__main__':
    parse_words()