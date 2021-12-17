from HelperObjects.ModelTrainer import ModelTrainer
from multiprocessing import freeze_support

if __name__ == '__main__':
    freeze_support()
    mt = ModelTrainer(force_re_parse=False, worker_count = 8)
    mt.find_optimal_values()
