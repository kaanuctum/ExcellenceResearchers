from HelperObjects.SQLConnector import SQLConnector
from HelperObjects.PathManager import PathManager
import HelperObjects.SensitiveData
import json

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

dataFhand = json.load(open("HelperObjects/SensitiveData/data.json"))
sql = SQLConnector(lite_name=dataFhand["sqlite_name"])

pathManager = PathManager()