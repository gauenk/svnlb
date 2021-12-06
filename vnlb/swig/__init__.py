
#@nolint

# not linting this file because it imports * from swigvnlb, which
# causes a ton of useless warnings.

import numpy as np
import sys
import inspect
import array
import warnings

# We import * so that the symbol foo can be accessed as vnlb.foo.
from .loader import *

_swig_enabled=True
__version__ = "%d.%d.%d" % (VNLB_VERSION_MAJOR,
                            VNLB_VERSION_MINOR,
                            VNLB_VERSION_PATCH)

# We import * so that the symbol foo can be accessed as vnlb.foo.
# from .python import *
# import .swig
# import .cpu as cpu
