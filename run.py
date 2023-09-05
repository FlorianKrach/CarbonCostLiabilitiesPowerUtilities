"""
author: Florian Krach
"""

import numpy as np
import os
import general_structural_model as gsm
from absl import app
from absl import flags
import data_config as data_config
import general_structural_model as gsm


# ------------------------------------------------------------------------------
FUNCTIONS_general_structural_model = {

}


# ------------------------------------------------------------------------------
FLAGS = flags.FLAGS

flags.DEFINE_string("function", None, "name of the function to run")
flags.DEFINE_string("config", None,
                    "config to use (either name of a config dict in data_"
                    "config.py or a dict itself)")


# ------------------------------------------------------------------------------
def main(arg):
    """
    function to run parallel training with flags from command line
    """
    del arg
    function = eval("gsm."+FLAGS.function)
    if FLAGS.config is None:
        config = {}
    else:
        try:
            config = eval("data_config."+FLAGS.config)
        except Exception:
            config = eval(FLAGS.config)
    return function(**config)






if __name__ == '__main__':
    app.run(main)


