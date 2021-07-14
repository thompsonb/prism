# Prism: MT Evaluation in Many Languages via Zero-Shot Paraphrasing

Prism is an automatic MT metric which uses a sequence-to-sequence
paraphraser to score MT system outputs conditioned on their respective
human references.  Prism uses a multilingual NMT model as a zero-shot
paraphraser, which negates the need for synthetic paraphrase data and
results in a single model which works in many languages.

Prism outperforms or statistically ties with all metrics submitted to
the [WMT 2019 metrics shared task](https://www.aclweb.org/anthology/W19-5302/) as
segment-level human correlation.

We provide a large, pre-trained multilingual NMT model which we use as a multilingual paraphraser, 
but the model may also be of use to the research community beyond MT metrics.
We provide examples of using the model for both [multilingual translation](translation/README.md)
and [paraphrase generation](paraphrase_generation/README.md).

Prism scores raw, untokenized text; all preprocessing is applied internally.
This document describes how to install and use Prism.

# Installation

Prism requires a version of [Fairseq](https://github.com/pytorch/fairseq)
compatible with the provided pretrained model.
We recommend starting with a clean environment:

```bash
conda create -n prismenv python=3.7 -y
conda activate prismenv  # older conda versions: source activate prismenv
```

For reasonable speeds, we recommend running on a machine with a GPU
and the [CUDA](https://developer.nvidia.com/cuda-zone) version compatible with the version of fairseq/torch installed above.
Prism will run on a GPU if available; to run on CPU instead, set CUDA_VISIBLE_DEVICES to an empty string.

Download the Prism code and install requirements, including Fairseq:

```bash
git clone https://github.com/thompsonb/prism
cd prism
pip install -r requirements.txt
```

# Download Model

```bash
wget http://data.statmt.org/prism/m39v1.tar
tar xf m39v1.tar
export MODEL_DIR=m39v1/
```

# Metric Usage: Command Line

Create test candidate/reference files:

```bash
echo -e "Hi world.\nThis is a Test." >> cand.en
echo -e "Hello world.\nThis is a test." >> ref.en
echo -e "Bonjour le monde.\nC'est un test." >> src.fr
```

To obtain system-level metric scores, run:
```bash
./prism.py --cand cand.en --ref ref.en --lang en --model-dir $MODEL_DIR/
```
Here, "ref.en" is the (untokenized) human reference, and "cand.en" is the (untokenized) system output.
This command will print some logging information to STDERR, including a model/version identifier,
and print the system-level score (negative, higher is better) to STDOUT:

>Prism identifier: {'version': '0.1', 'model': 'm39v1', 'seg_scores': 'avg_log_prob', 'sys_scores': 'avg_log_prob', 'log_base': 2}  
>-1.0184667  

Candidates can also be piped into prism.py:

```bash
cat cand.en | ./prism.py --ref ref.en --lang en --model-dir $MODEL_DIR/
```

To score output using the source instead of the reference
(i.e., quality estimation as a metric), use the --src flag.
Note that --lang still specifies the target/reference language:

```bash
./prism.py --cand cand.en --src src.fr --lang en --model-dir $MODEL_DIR/ 
```

Prism also has access to all WMT test sets via the
[sacreBLEU](https://github.com/mjpost/sacrebleu) API. These can be
specified as arguments to `--src` and `--ref`, 
for a hypothetical system output $cand, as follows: 

```bash
./prism.py --cand $cand --ref sacrebleu:wmt19:de-en --model-dir $MODEL_DIR/
```
which will cause it to use the English reference from the WMT19 German--English test set.
(Since the language is known, no `--lang` is needed).

To see all options, including segment-level scoring, run:

```bash
./prism.py -h
```

# Metric Usage: Python Module

All functionality is also available in Python, for example:

```python
import os
from prism import Prism

prism = Prism(model_dir=os.environ['MODEL_DIR'], lang='en')

print('Prism identifier:', prism.identifier())

cand = ['Hi world.', 'This is a Test.']
ref = ['Hello world.', 'This is a test.']
src = ['Bonjour le monde.', "C'est un test."]

print('System-level metric:', prism.score(cand=cand, ref=ref))
print('Segment-level metric:', prism.score(cand=cand, ref=ref, segment_scores=True))
print('System-level QE-as-metric:', prism.score(cand=cand, src=src))
print('Segment-level QE-as-metric:', prism.score(cand=cand, src=src, segment_scores=True))
```

Which should produce:

>Prism identifier: {'version': '0.1', 'model': 'm39v1', 'seg_scores': 'avg_log_prob', 'sys_scores': 'avg_log_prob', 'log_base': 2}
>System-level metric: -1.0184666  
>Segment-level metric: [-1.4878583 -0.5490748]  
>System-level QE-as-metric: -1.8306842  
>Segment-level QE-as-metric: [-2.462842  -1.1985264]  


# Multilingual Translation
The Prism model is simply a multilingual NMT model, and can be used for translation --  see the [multilingual translation README](translation/README.md).

# Paraphrase Generation

Attempting to generate paraphrases from the Prism model via naive beam search
(e.g. "translate" from French to French) results in trivial copies most of the time.
However, we provide a simple algorithm to discourage copying
and enable paraphrase generation in many languages -- see the [paraphrase generation README](paraphrase_generation/README.md).


# Supported Languages

Albanian (sq), Arabic (ar), Bengali (bn), Bulgarian (bg), 
Catalan; Valencian (ca), Chinese (zh), Croatian (hr), Czech (cs), 
Danish (da), Dutch (nl), English (en), Esperanto (eo), Estonian (et),
Finnish (fi),  French (fr), German (de), Greek, Modern (el),
Hebrew (modern) (he),  Hungarian (hu), Indonesian (id), Italian (it),
Japanese (ja), Kazakh (kk), Latvian (lv), Lithuanian (lt), Macedonian (mk),
Norwegian (no), Polish (pl), Portuguese (pt), Romanian, Moldavan (ro),
Russian (ru), Serbian (sr), Slovak (sk), Slovene (sl), Spanish; Castilian (es),
Swedish (sv), Turkish (tr), Ukrainian (uk), Vietnamese (vi)

# Data Filtering

The data filtering scripts used to train the Prism model can be found [here](https://github.com/thompsonb/prism_bitext_filter).


# Publications

If you the Prism metric and/or the provided multilingual NMT model, please cite our [EMNLP paper](https://www.aclweb.org/anthology/2020.emnlp-main.8/):
```
@inproceedings{thompson-post-2020-automatic,
    title={Automatic Machine Translation Evaluation in Many Languages via Zero-Shot Paraphrasing},
    author={Brian Thompson and Matt Post},
    year={2020},
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    address = "Online",
    publisher = "Association for Computational Linguistics",
}
```

If you the paraphrase generation algorithm, please also cite our [WMT paper](https://aclanthology.org/2020.wmt-1.67/):
```
@inproceedings{thompson-post-2020-paraphrase,
    title={Paraphrase Generation as Zero-Shot Multilingual Translation: Disentangling Semantic Similarity from Lexical and Syntactic Diversity},
    author={Brian Thompson and Matt Post},
    year={2020},
    booktitle = "Proceedings of the Fifth Conference on Machine Translation (Volume 1: Research Papers)",
    month = nov,
    address = "Online",
    publisher = "Association for Computational Linguistics",
}
```
