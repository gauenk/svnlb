# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#@nolint

# not linting this file because it imports * from swigfaiss, which
# causes a ton of useless warnings.

import numpy as np
import sys
import inspect
import array
import warnings

# We import * so that the symbol foo can be accessed as faiss.foo.
from .loader import *


__version__ = "%d.%d.%d" % (VNLB_VERSION_MAJOR,
                            VNLB_VERSION_MINOR,
                            VNLB_VERSION_PATCH)
