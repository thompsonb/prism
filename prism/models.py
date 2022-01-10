
import hashlib

import os

from torchnlp.download import download_file_maybe_extract

class PrismDataError(Exception):
    pass

def fetch_model():
    if os.environ.get("HOME", None):
        cache_directory = os.environ["HOME"] + "/.cache/prism-mt/"
        if not os.path.exists(cache_directory):
            os.makedirs(cache_directory)
            # Download data.
        if not os.path.exists(cache_directory+'/m39v1/'):
            download_file_maybe_extract(
                url="http://data.statmt.org/prism/m39v1.tar",
                directory=cache_directory,
            )
        return cache_directory
    else:
        raise Exception("HOME environment variable is not defined.")


MODELS = {
    '8412b2044da4b9b2c0a8ce87b305d0d1': {
        'name': 'm39v1',
        'path': 'todo',
        'date': '2020-04-30',
        'description': 'model released with arXiv paper April 2020',
        'langs': ['ar', 'bg', 'bn', 'ca', 'cs', 'da', 'de', 'el', 'en', 'es', 'et', 'eo', 'fi', 'fr', 'he',
                  'hr', 'hu', 'id', 'it', 'ja', 'kk', 'lt', 'lv', 'mk', 'nl', 'no', 'pl', 'pt', 'ro', 'ru',
                  'sk', 'sl', 'sq', 'sr', 'sv', 'tr', 'uk', 'vi', 'zh'],
    }
}

MODEL_DIR = os.environ["HOME"] + "/.cache/prism-mt/m39v1/"

def hash_model(model_dir):
    md5 = hashlib.md5()
    block_size = 2 ** 20
    for fname in ('checkpoint.pt', 'spm.model', 'dict.src.txt', 'dict.tgt.txt'):
        with open(os.path.join(model_dir, fname), "rb") as f:
            while True:
                data = f.read(block_size)
                if not data:
                    break
                md5.update(data)
    md5.digest()
    return md5.hexdigest()
