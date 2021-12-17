from multiprocessing import freeze_support

from MainScripts.AnalyzeData import Analyzer
from MainScripts.TrainModels2 import ModelTrainer

if __name__ == '__main__':
    freeze_support()
    # mt = ModelTrainer(force_re_parse=False)
    # mt.find_optimal_values()
    analyzer = Analyzer()
    # analyzer.calc_position_of_documents()
    analyzer.calc_before_after_average_dist()
