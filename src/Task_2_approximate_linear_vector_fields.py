"""
Task 2: Approximating Linear Vector Fields
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

if __name__ == '__main__':
    cwd = Path.cwd()
    print(cwd)
    path = cwd/"datasets"
    approximateData_LinearFunction(path/"linear_vectorfield_data_x0.txt")
    approximateData_LinearFunction(path/"linear_vectorfield_data_x1.txt")
