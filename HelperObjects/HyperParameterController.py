import json
from HelperObjects.PathManager import PathManager

class HyperParameterController:
    def __init__(self):
        self.a = 0
        self.b = 0
        self.topic_num = 0
        self.score = 0

    def read(self):
        data = json.load(open(PathManager().get_paths()['hyperparams']))
        self.a = data['a']
        self.b = data['b']
        self.topic_num = data['k']
        self.score = data['score']

    def write(self):
        d = {
            'a': self.a,
            'b': self.b,
            'k': self.topic_num,
            'score': self.score
        }
        with open(PathManager().get_paths()['hyperparams'], 'w') as json_file:
            json.dump(d, json_file)

    def update(self, k, a, b, score):
        if score > self.score:
            self.a = a
            self.b = b
            self.topic_num = k
            self.score = score
            self.write()
            return True
        return False
