from __future__ import absolute_import, unicode_literals, print_function

import numpy as np
import os

from pymultinest.watch import ProgressPlotter 
from pymultinest.watch import ProgressPrinter 


n_params = 8 
prefix = "chains/14-"

watcher = ProgressPlotter(n_params, interval_ms = 10000, outputfiles_basename=prefix)
#watcher = ProgressPrinter(n_params, interval_ms = 10000, outputfiles_basename=prefix)
watcher.run()



