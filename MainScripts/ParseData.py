from HelperObjects.Parsers.Parse_Abstracts import *


def parseData():
    # get the raw data
    raw_data = get_raw_data()
    print('removing copyright')
    cleaned_data = clean_copyright(raw_data)

    print('loading cleaned abstracts')
    CLEANED = lemmatization(cleaned_data)
    sql.update_cleaned_abstracts(CLEANED)  # update the db

    print('parsing the words')
    fill_words(CLEANED)  # update the words
    print('filled in words')

    print('marking irrelevant words')
    mark_stop_words()
    mark_short_words()
    mark_low_frequency()
    mark_common_words()
    mark_top_words()
    print('marked irrelevant words')

    _, corpus, dictionary = tokenize_abstracts(CLEANED, common_words=get_common_words(), low_freq=get_low_frequency())
    print('tokenized abstracts')
    return corpus, dictionary

if __name__ == "__main__":
    parseData()