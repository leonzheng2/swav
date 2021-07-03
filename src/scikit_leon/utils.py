import logging
import sys
from pathlib import Path

import yaml


def config_logger(logs_path, verbose):
    logger = logging.getLogger()
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    output_file_handler = logging.FileHandler(logs_path)
    output_file_handler.setFormatter(formatter)
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(output_file_handler)
    logger.addHandler(stdout_handler)
    logging.info(" ".join(sys.argv))
    return logger


def parse_dictionary(dict_file):
    with open(Path(dict_file), 'r', encoding='utf8') as yaml_file:
        yaml_content = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return yaml_content
