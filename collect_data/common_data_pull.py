import os
from shutil import copyfileobj
from urllib.request import Request, urlopen

import pandas as pd

# Get path of current file's directory
DIRNAME = os.path.dirname(os.path.realpath(__file__))


def download_csv(url: str, filepath: str):
    req = Request(url, headers={"User-Agent": "XYZ/3.0"})

    with urlopen(req, timeout=10) as in_stream, open(filepath, "wb") as out_file:
        copyfileobj(in_stream, out_file)
