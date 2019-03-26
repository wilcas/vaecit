import cit
import csv
import joblib
import vae
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import data_model as dm
import numpy as np
import tensorflow as tf

from itertools import product
