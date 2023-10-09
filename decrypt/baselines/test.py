
import sys
import os
from pathlib import Path
path = str(Path(Path.cwd()).parent.absolute())
# sys.path.insert(0, path)
sys.path.append(path) 

print(sys.path)
from decrypt.scrape_parse import (
    load_guardian_splits,
    load_guardian_splits_disjoint,
    load_guardian_splits_disjoint_hash
)

import os
from decrypt import config
from decrypt.common import validation_tools as vt
from decrypt.common.util_data import clue_list_tuple_to_train_split_json
import logging
logging.getLogger(__name__)


k_json_folder = config.DataDirs.Guardian.json_folder

print(k_json_folder)