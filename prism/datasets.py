from typing import List, Dict, Iterator, Any, Tuple

import logging
import sys

from sacrebleu import get_source_file, get_reference_files, DATASETS, get_langpairs_for_testset

class SacrebleuDataError(Exception):
    pass

def parse_sacrebleu_uri(uri: str) -> Tuple[str]:
    """
    Parses the test set and language pair from a URI of the form

        sacrebleu:wmt19:de-en
        sacrebleu:wmt19/google/ar:de-en
    """
    try:
        _, testset, langpair = uri.split(":")
    except ValueError:
        raise SacrebleuDataError('sacrebleu:* flags must take the form "sacrebleu:testset:langpair"')

    testsets = sorted(DATASETS, reverse=True)
    if testset not in testsets:
        raise SacrebleuDataError(f"Test set '{testset}' was not found. Available sacrebleu test sets are:")
        for key in testsets:
            raise SacrebleuDataError(f"  {key:20s}: {DATASETS[key].get('description', '')}")

    lang_pairs = get_langpairs_for_testset(testset)

    if langpair not in lang_pairs:
        raise SacrebleuDataError(
            f"Language pair '{langpair}' not available for testset '{testset}'.\n "
            f"Language pairs available for {testset}: {', '.join(lang_pairs)}")

    return testset, langpair
