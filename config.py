#!/usr/bin/env python

from os.path import abspath, dirname, join

ROOT = dirname(abspath(__file__))
DIR_MODEL = join(ROOT, 'model')
NAME_MODEL = 'AltCLIP-XLMR-L-m18'
FP_MODEL = join(DIR_MODEL, NAME_MODEL)
