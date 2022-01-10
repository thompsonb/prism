from setuptools import setup

setup(
  name = 'prism-mt',
  packages = ['prism'],
  version = '0.0.1',
  description = 'prism-mt',
  long_description = '',
  author = '',
  url = 'https://github.com/thompsonb/prism',
  keywords = [],
  install_requires = ['sentencepiece', 'fairseq', 'sacrebleu',
                      'torch', 'torchvision', 'pytorch-nlp'],
  classifiers = [
    "Programming Language :: Python :: 3",
    "Operating System :: OS Independent",
  ]
)
