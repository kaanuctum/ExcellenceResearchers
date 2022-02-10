import pyLDAvis
import pyLDAvis.gensim_models
import warnings
warnings.filterwarnings('ignore')
from MainScripts.ParseData import parse_words
from HelperObjects import *
import pandas as pd
import pickle
from tqdm import tqdm

import gensim
import concurrent.futures


class ModelTrainer:
    def __init__(self, force_re_parse=False, worker_count=1):

        self.min_topics = 4
        self.max_topics = 50
        self.worker_count = worker_count
        self.model_name = 'best_model'
        self.paths = pathManager.get_paths()

        if force_re_parse:
            self.ID2WORD, self.DATA_LEMMATIZED, self.CORPUS, _ = parse_words()
        else:
            try:
                self.ID2WORD = pickle.load(open(self.paths['id2word'], 'rb'))
                self.DATA_LEMMATIZED = pickle.load(open(self.paths['data_lemmatized'], 'rb'))
                self.CORPUS = pickle.load(open(self.paths['corpus'], 'rb'))

            except FileNotFoundError:
                print("couldn't load preliminary results, re-parsing raw abstracts")
                self.ID2WORD, self.DATA_LEMMATIZED, self.CORPUS, _ = parse_words()

        try:
            self.model_results = pickle.load(open(self.paths['model_results'], 'rb'))
        except FileNotFoundError:
            self.model_results = {'Topics': [],
                                  'Coherence_cv': [],
                                  'Coherence_umass': []
                                  }
        try:
            self.prev_done = pickle.load(open(self.paths['prev_done'], 'rb'))
        except FileNotFoundError:
            self.prev_done = []

        pd.DataFrame(self.model_results).to_csv(self.paths['lda_tuning_results'], index=False, sep='\t')
        pickle.dump(self.model_results, open(self.paths['model_results'], 'wb'))
        pickle.dump(self.prev_done, open(self.paths['prev_done'], 'wb'))

    def train_model_auto(self, num_topics):
        chunk_size = (2 ** 14)
        iterations = 400
        passes = 50
        lda_model = gensim.models.LdaModel(num_topics=num_topics,
                                           corpus=self.CORPUS,
                                           id2word=self.ID2WORD,
                                           chunksize=chunk_size,
                                           iterations=iterations,
                                           passes=passes,
                                           eval_every=None
                                           )
        return lda_model

    def create_models_in_parallel(self, training_range):
        pool = concurrent.futures.ProcessPoolExecutor()
        results = pool.map(self.train_model_auto, training_range)
        return results

    def save_model(self, lda_model, coherence):
        model = f'DATA/MODELS/MODEL/{coherence.upper()}/best_model.model'
        model_vis = f'DATA/MODELS/MODEL/{coherence.upper()}/best_model.html'
        lda_model.save(model)
        vis = pyLDAvis.gensim_models.prepare(lda_model, self.CORPUS, self.ID2WORD, mds="mmds", R=30)
        pyLDAvis.save_html(vis, model_vis)

    def get_coherence_values(self, lda_model):
        cv = gensim.models.CoherenceModel(model=lda_model, texts=self.DATA_LEMMATIZED,
                                          dictionary=self.ID2WORD,
                                          coherence="c_v").get_coherence()

        umass = gensim.models.CoherenceModel(model=lda_model, texts=self.DATA_LEMMATIZED,
                                             dictionary=self.ID2WORD,
                                             coherence="u_mass").get_coherence()

        return cv, umass

    def find_optimal_values(self):

        topics_range = range(self.min_topics, self.max_topics + 1, self.worker_count)

        best_cv = 0
        best_umas = 0

        progress_bar = tqdm(total=self.max_topics - self.min_topics + 1)

        for topic_number in topics_range:
            k_max = min(topic_number + self.worker_count, self.max_topics + 1)
            progress_bar.display(str(topic_number) + ' - ' + str(k_max-1))

            train_range = [i for i in range(topic_number, k_max)]
            prev_done = True
            for i in train_range:
                if i not in self.model_results['Topics']:
                    prev_done = False
                else:
                    print(f"{i} topics previously done")
            if prev_done: continue

            results = self.create_models_in_parallel(training_range=train_range)
            for lda_model in results:

                cv, umass = self.get_coherence_values(lda_model)

                k = lda_model.num_topics

                # Save the model results
                self.model_results['Topics'].append(k)
                self.model_results['Coherence_cv'].append(cv)
                self.model_results['Coherence_umass'].append(umass)
                self.prev_done.append(topic_number)
                print(lda_model.num_topics, cv, umass)

                if cv > best_cv:
                    self.save_model(lda_model, 'cv')
                    print(f"new best coherence for cv")
                    best_cv = cv
                if umass < best_umas:
                    self.save_model(lda_model, 'umass')
                    print(f"new best coherence for umass")
                    best_umas = umass

                progress_bar.update(1)
            pd.DataFrame(self.model_results).to_csv(self.paths['lda_tuning_results'], index=False, sep='\t')
            pickle.dump(self.model_results, open(self.paths['model_results'], 'wb'))
            pickle.dump(self.prev_done, open(self.paths['prev_done'], 'wb'))
        progress_bar.close()
        df = pd.DataFrame(self.model_results)
        print("Best Umass model:\n ", df.iloc[df['Coherence_umass'].idxmin()])
        print("Best CV model   :\n ", df.iloc[df['Coherence_cv'].idxmax()])