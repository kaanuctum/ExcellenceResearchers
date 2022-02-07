class PathManager:
    def __init__(self):
        self.__paths = {
            'lda_tuning_results': 'DATA/MODELS/lda_tuning_results.csv',

            'model_results': 'DATA/MODELS/TRAINING/model_results.pickle',
            'prev_done': 'DATA/MODELS/TRAINING/prev_done.pickle',
            'hyperparams': 'DATA/MODELS/TRAINING/hyperparameters.json',

            'df_parsed_articles': 'DATA/MODELS/INPUT_DATA/df.pickle',

            'corpus': "DATA/MODELS/INPUT_DATA/corpus.pickle",
            'id2word' : 'DATA/MODELS/INPUT_DATA/id2word.pickle',
            'data_lemmatized': 'DATA/MODELS/INPUT_DATA/data_lemmatized.pickle',

            'df_distance': 'DATA/RESULTS/df_distance.pickle',
            'df_distance_csv' : 'DATA/RESULTS/df_distance.csv'
        }

    def get_paths(self):
        return self.__paths
