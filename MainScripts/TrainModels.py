from HelperObjects.ModelTrainer import ModelTrainer
from multiprocessing import freeze_support
import warnings
warnings.filterwarnings('ignore')
if __name__ == '__main__':
    freeze_support()
    mt = ModelTrainer(force_re_parse=False, worker_count=5, min_topics=51,  max_topics=100)
    mt.find_optimal_values()
