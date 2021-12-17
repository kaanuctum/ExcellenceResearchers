


def train_model_auto(args):
    num_topics, ns = args
    print(f'training with {num_topics} number of topics')

    chunk_size = (2 ** 14)
    passes = 30
    iterations = 600

    lda_model = gensim.models.LdaModel(corpus=ns.CORPUS,
                                       id2word=ns.ID2WORD,
                                       chunksize=chunk_size,
                                       alpha='auto',
                                       eta='auto',
                                       iterations=iterations,
                                       num_topics=num_topics,
                                       passes=passes,
                                       eval_every=None
                                       )
    print(f'finished training with {num_topics} number of topics ')
    return lda_model





if __name__ == '__main__':
    freeze_support()
    mt = ModelTrainer(force_re_parse=False)
    mt.find_optimal_values()
