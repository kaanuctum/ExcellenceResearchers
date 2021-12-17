import pyLDAvis
import pyLDAvis.gensim_models

from HelperObjects.Parsers.Parse_Abstracts2 import parse_words
from HelperObjects import *
import pandas as pd
import pickle
from tqdm import tqdm

import gensim
from multiprocessing import Manager, Pool

class ModelTrainer:
    def __init__(self, force_re_parse, worker_count=1):
        self.min_topics = 20
        self.max_topics = 50
        self.worker_count = worker_count

        self.mn = Manager()
        self.ns = self.mn.Namespace()

        self.model_name = 'best_model'
        self.paths = pathManager.get_paths()

        if force_re_parse:
            self.ns.ID2WORD, self.ns.DATA_LEMMATIZED, self.ns.CORPUS, _ = parse_words()
        else:
            try:
                self.ns.ID2WORD = pickle.load(open(self.paths['id2word'], 'rb'))
                self.ns.DATA_LEMMATIZED = pickle.load(open(self.paths['data_lemmatized'], 'rb'))
                self.ns.CORPUS = pickle.load(open(self.paths['corpus'], 'rb'))

            except FileNotFoundError:
                print("couldn't load preliminary results, re-parsing raw abstracts")
                self.ns.ID2WORD, self.ns.DATA_LEMMATIZED, self.ns.CORPUS, _ = parse_words()

        try:
            self.model_results = pickle.load(open(self.paths['model_results'], 'rb'))
        except FileNotFoundError:
            self.model_results = {'Topics': [],
                                  'Alpha': [],
                                  'Beta': [],
                                  'Coherence': []
                                  }
        try:
            self.prev_done = pickle.load(open(self.paths['prev_done'], 'rb'))
        except FileNotFoundError:
            self.prev_done = []

        pd.DataFrame(self.model_results).to_csv(self.paths['lda_tuning_results'], index=False, sep='\t')
        pickle.dump(self.model_results, open(self.paths['model_results'], 'wb'))
        pickle.dump(self.prev_done, open(self.paths['prev_done'], 'wb'))

    def calc_coherence_of_model(self, lda_model):
        coherence_model_lda = gensim.models.CoherenceModel(model=lda_model,
                                                           texts=self.ns.DATA_LEMMATIZED,
                                                           dictionary=self.ns.ID2WORD,
                                                           coherence='c_v')
        return coherence_model_lda.get_coherence()

    def train_model(self, num_topics, a, b):
        chunk_size = (2 ** 14)
        passes = 20
        iterations = 400

        if a == 'auto' or b == 'auto':
            lda_model = gensim.models.LdaModel(corpus=self.ns.CORPUS,
                                               id2word=self.ns.ID2WORD,
                                               chunksize=chunk_size,
                                               alpha=a,
                                               eta=b,
                                               iterations=iterations,
                                               num_topics=num_topics,
                                               passes=passes,
                                               eval_every=None
                                               )
        else:
            lda_model = gensim.models.LdaMulticore(corpus=self.ns.CORPUS,
                                                   id2word=self.ns.ID2WORD,
                                                   chunksize=chunk_size,
                                                   alpha=a,
                                                   eta=b,
                                                   iterations=iterations,
                                                   num_topics=num_topics,
                                                   passes=passes,
                                                   eval_every=None
                                                   )

        return lda_model

    def create_models_in_parallel(self, training_range):
        executor = Pool(len(training_range))
        results = executor.map(train_model_auto, training_range)
        executor.close()
        executor.join()

        return results

    def save_model(self, ldamodel):
        ldamodel.save(self.paths['model'])
        self.create_visualisation(ldamodel)

    def create_visualisation(self, model):
        vis = pyLDAvis.gensim_models.prepare(model, self.ns.CORPUS, self.ns.ID2WORD, mds="mmds", R=30)
        pyLDAvis.save_html(vis, self.paths['model_vis'])
        print(f'saved visualization for model: {self.model_name}')

    def find_optimal_values(self):

        topics_range = range(self.min_topics, self.max_topics + 1, self.worker_count)

        alpha_range = ['auto']
        beta_range = ['auto']

        progress_bar = tqdm(total=self.max_topics - self.min_topics + 1)

        for alpha in alpha_range:
            for beta in beta_range:
                for topic_number in topics_range:
                    cur_it = (topic_number, alpha, beta)
                    k_max = min(topic_number + self.worker_count, self.max_topics + 1)
                    progress_bar.display(str(topic_number) + ' - ' + str(k_max))
                    # get the coherence score for the given parameters
                    if cur_it in self.prev_done:
                        progress_bar.update(1)
                        continue

                    train_range = [(i, self.ns) for i in
                                   range(topic_number, k_max)]

                    results = self.create_models_in_parallel(training_range=train_range)
                    for lda_model in results:
                        cv = gensim.models.CoherenceModel(model=lda_model, texts=self.ns.DATA_LEMMATIZED,
                                                          dictionary=self.ns.ID2WORD,
                                                          coherence='c_v').get_coherence()

                        k = lda_model.num_topics

                        # Save the model results
                        self.model_results['Topics'].append(k)
                        self.model_results['Alpha'].append(alpha)
                        self.model_results['Beta'].append(alpha)
                        self.model_results['Coherence'].append(cv)
                        self.prev_done.append(cur_it)
                        print(lda_model.num_topics, alpha, beta, cv)

                        if hpController.update(k=k, a=alpha, b=beta, score=cv):
                            self.save_model(ldamodel=lda_model)

                        progress_bar.update(1)
                    pd.DataFrame(self.model_results).to_csv(self.paths['lda_tuning_results'], index=False, sep='\t')
                    pickle.dump(self.model_results, open(self.paths['model_results'], 'wb'))
                    pickle.dump(self.prev_done, open(self.paths['prev_done'], 'wb'))
        progress_bar.close()