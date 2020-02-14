"""The purpose of this file is to run the code-generation step wherein Syft
tensors are converted into framework tensors such as Torch, Tensorflow, or Numpy
tensors.

As such, this file is only useful if you are actively developing a new Syft tensor type.

After you run this file, it is good to run the "post_compile_check.py" to see if there
are any potential errors with the compilation.
"""

import syft as sy
from pprintast.pprintast import ppast

sy.compile_torch()
sy.compile_numpy()
