#!/usr/bin/env python

from config import MODEL
from baai_modelhub import AutoPull

AutoPull().get_model(model_name='AltCLIP-XLMR-L-m18', model_save_path=MODEL)
