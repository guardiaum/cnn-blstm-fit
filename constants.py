import os, sys

# SET CONSTANT VARIABLES
ROOT = os.path.dirname(os.path.dirname(__file__))

DATASETS = ROOT + "/datasets"

TRAIN_EXT_DIR = DATASETS + "/train_extractor"
VALIDATION_ARTICLES_DIR = DATASETS + "/articles-validation"
TEST_ARTICLES_DIR = DATASETS + "/articles-test"
FULL_BASE_DIR = DATASETS +"/full-base"