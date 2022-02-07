from HelperObjects.Parsers.Parse_Abstracts2 import *
import pickle
import time
import spacy
import pandas as pd

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
    print(f"words lemmatized in {time.time() - start} seconds")

    start = time.time()
    id2word = gensim.corpora.Dictionary(data_lemmatized)  # Create Dictionary
    # Filter out words that occur less than 20 documents, or more than in 50% of the documents.
    l1 = len(id2word)
    id2word.filter_extremes(no_below=20, no_above=0.5)
    print(f"{l1 - len(id2word)} Extreme words filtered")
    corpus = [id2word.doc2bow(text) for text in data_lemmatized]  # View
    df['bow'] = corpus
    df['lemmatized'] = bow_to_sentence(corpus, id2word)
    print(f"Corpus created in {time.time() - start} seconds")
    print(df)
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
    id2word, data_lemmatized, corpus, df = parse_words()