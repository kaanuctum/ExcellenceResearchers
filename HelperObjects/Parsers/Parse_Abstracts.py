# THIS IS AN OLD VERSION, SHOULD NOT BE USED ANYMORE


from HelperObjects import sql

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from gensim import corpora

import pickle
import re
from tqdm import tqdm

"""
 These are the helper methods used 
"""










LENGTH_LIMIT = 3
FREQUENCY_LIMIT = 30
UPPER_LIMIT_COMMON = 0.3  # if a word is in more than x% of all the articles, it is ignored
PS = PorterStemmer()
STOPWORDS = stopwords.words('english')
STOPWORDS.extend(['however', 'compare', 'differ', 'find', 'year'])
sql.connect()

# just gives the eid and the raw abstracts of all articles
def get_raw_data():
    sql.connect()
    sql.cur.execute("SELECT article_id, raw_abstract FROM article_abstracts WHERE raw_abstract NOTNULL ORDER BY article_id")
    return sql.cur.fetchall()


def get_sentence_parts(sentence):
    return [t for t in sentence.split() if
            len(t) > LENGTH_LIMIT]  # gensim.utils.simple_preprocess(sentence, deacc=True)


# get the stem of the word given, for now it just passes
def get_stem(raw_abstract, ps=PS):
    return ps.stem(raw_abstract)


def clean_copyright(raw_abstracts):
    return [(eid, clean_copyright_just_text(abstract)) for (eid, abstract) in tqdm(raw_abstracts)]


def clean_copyright_just_text(raw_abstract, copyright_threshold=50):
    # if the copyright data is not present in the text, there isn't much we can do to find the copyright
    if raw_abstract.find('©') == -1: return raw_abstract

    cleaned = ''
    # some copyright sentences are not separated properly
    # find all the upper case letters that follow a small case letter without a space
    # replace all the dots are followed by a capitol letter
    abstract = re.sub('[.](?=[A-Z])(?=[^.])', '. ', raw_abstract)
    abstract = re.sub('([a-z](?=[A-Z])(?=[^\s-]))', r'\1. ', abstract)
    if abstract.find('©') > 0:
        abstract = abstract.replace('©', '. ©')

    # iterate over the sentences in the abstract
    for i in nltk.sent_tokenize(abstract):
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
    return re.sub('[,\.!?]', '', text)



# takes in a list of sentences and simplifies them down to the point where the computer can understand them better
def lemmatization(texts):
    output = []
    for eid, text in tqdm(texts):
        # replace punctuation
        text = re.sub("[^a-zA-Z\s]", " ", text)
        output.append((eid, text.lower()))
    with open('DATA/INPUT_DATA/lemmatized.pickle', 'wb') as pck:
        pickle.dump(output, pck)
    return output


#  insert the words into the database
def fill_words(tokenized):
    # clean the database
    sql.connect()
    sql.cur.execute("DELETE FROM lookup_article_wordcount")
    sql.cur.execute("DELETE FROM words")
    total_stem_count = dict()

    article_count = dict()
    for (eid, abstract) in tqdm(tokenized):
        word_counts = dict()
        stem_counts = dict()

        parts = get_sentence_parts(abstract)
        for word in parts:
            # increment the word count
            stem = get_stem(word)
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1

            if stem in stem_counts:
                stem_counts[stem] += 1
            else:
                stem_counts[stem] = 1

            if stem in total_stem_count:
                total_stem_count[stem] += 1
            else:
                total_stem_count[stem] = 1
            sql.cur.execute(
                "INSERT INTO lookup_article_wordcount (article_id, word, Wcnt, Scnt, stem) VALUES (?, ?, ?, ?, ?)",
                (eid, word, word_counts[word], stem_counts[stem], stem))

        for word in stem_counts:
            if word in article_count:
                article_count[word] += 1
            else:
                article_count[word] = 1

    # insert the word counts
    sql.cur.execute("SELECT word, stem, sum(Wcnt) FROM lookup_article_wordcount GROUP BY word")
    for (w, s, c) in tqdm(sql.cur.fetchall()):
        sql.cur.execute("INSERT INTO words (word, stem, total_count, article_count, ignored) VALUES (?,?,?,0,0)",
                        (w, s, c))

    # insert the stem counts
    sql.cur.execute("SELECT stem, sum(Scnt) FROM lookup_article_wordcount GROUP BY stem")
    for (s, c) in tqdm(sql.cur.fetchall()):
        sql.cur.execute("UPDATE words SET stem_count=?, article_count=? WHERE stem=?", (c, article_count[s], s))

    # update the stem IDS
    sql.cur.execute("SELECT DISTINCT(stem) FROM words")
    i = 0
    for (s) in tqdm(sql.cur.fetchall()):
        sql.cur.execute("UPDATE words SET stem_id=? WHERE stem=?", (i, s[0]))
        i += 1

    sql.conn.commit()
    return total_stem_count



def mark_short_words(len_limit=LENGTH_LIMIT):
    sql.cur.execute("UPDATE words SET ignored=1 where length(word)<=?", (len_limit,))

def mark_stop_words(stop_words=STOPWORDS):
    for stop_word in stop_words:
        sql.cur.execute("UPDATE words SET ignored=2 where stem=?", (get_stem(stop_word),))


def mark_low_frequency(frequency_limit=FREQUENCY_LIMIT):
    sql.cur.execute("UPDATE words SET ignored=3 where stem_count<=?", (frequency_limit,))


def mark_common_words(upper_limit=UPPER_LIMIT_COMMON):
    # mark the words which are in more articles than the threshold allows for
    sql.connect()
    sql.cur.execute("SELECT COUNT(article_id) FROM article_abstracts WHERE abstract NOTNULL")
    total_article_count = sql.cur.fetchall()[0][0]
    threshold = int(total_article_count * upper_limit)
    sql.cur.execute("UPDATE words SET ignored=4 where article_count>=?", (threshold,))
    sql.conn.commit()


def mark_top_words(upper_limit=0.05):
    # mark the words which are in more articles than the threshold allows for
    sql.connect()
    sql.cur.execute("SELECT COUNT(article_id) FROM article_abstracts WHERE abstract NOTNULL")
    total_article_count = sql.cur.fetchall()[0][0]

    sql.cur.execute("SELECT SUM(total_count) from words")
    total_word_count = sql.cur.fetchall()[0][0] # The number of words in the database
    threshold = int(total_word_count * upper_limit)
    i = 0
    while i < threshold:
        sql.cur.execute("SELECT word, total_count from words WHERE total_count = (SELECT max(total_count) from words where ignored = 0)")
        (word, count) = sql.cur.fetchall()[0]  # number of usages of the word that is currently not ignored
        i+=count
        sql.cur.execute("UPDATE words SET ignored=5 where word=?", (word,))
    sql.conn.commit()


def fill_article_count():
    sql.cur.execute("SELECT DISTINCT(stem) FROM words where article_count=NULL")
    words = sql.cur.fetchall()
    for rawWord in tqdm(words):
        word = rawWord[0]
        sql.cur.execute("SELECT COUNT(distinct(article_id)) FROM lookup_article_wordcount WHERE stem=?", (word,))
        num = sql.cur.fetchall()[0][0]
        sql.cur.execute("UPDATE words SET article_count=? where stem=?", (num, word))
        sql.conn.commit()


def get_ignored_word():
    sql.cur.execute("SELECT stem FROM words where ignored>0")
    ignored_word = [i[0] for i in sql.cur.fetchall()]
    with open('DATA/INPUT_DATA/common_words.pickle', 'wb') as pck:
        pickle.dump(ignored_word, pck)
    return ignored_word


def get_common_words():
    sql.cur.execute("SELECT stem FROM words where ignored=4")
    common_words = [i[0] for i in sql.cur.fetchall()]
    return common_words


def get_low_frequency():
    sql.cur.execute("SELECT stem FROM words where ignored=3")
    low_freq = [i[0] for i in sql.cur.fetchall()]
    return low_freq


def tokenize_abstracts(cleaned_abstracts, common_words, low_freq, len_lim = LENGTH_LIMIT, stopwords=STOPWORDS):
    token_cache = []
    for (eid, cleaned) in tqdm(cleaned_abstracts):
        parts = get_sentence_parts(cleaned)
        words = list()
        for word in parts:
            if len(word) < len_lim: continue
            if '$' in word: continue  # filter latex commands
            if word in stopwords: continue

            stem = get_stem(word)  # does nothing for now
            if stem in low_freq: continue
            if stem in common_words: continue
            words.append(stem)
        token_cache.append((eid, words))

    cleaneds = [cleaned for (eid, cleaned) in token_cache]
    dictionary = corpora.Dictionary(cleaneds)
    corpus = [dictionary.doc2bow(cleaned) for cleaned in cleaneds]

    with open('DATA/INPUT_DATA/token_cache.pickle', 'wb') as pck:
        pickle.dump(token_cache, pck)

    with open('DATA/INPUT_DATA/corpus.pickle', 'wb') as pck:
        pickle.dump(corpus, pck)

    with open('DATA/INPUT_DATA/dictionary.pickle', 'wb') as pck:
        pickle.dump(dictionary, pck)

    return token_cache, corpus, dictionary


if __name__ == "__main__":
    mark_common_words()
    sql.conn.commit()
    print(get_common_words())
