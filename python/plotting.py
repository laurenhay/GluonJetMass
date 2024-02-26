# Plotting functions

import awkward as ak
import numpy as np
import pickle
import hist
import coffea
from plugins import *
%matplotlib inline

# from hist import intervals
import matplotlib.pyplot as plt
import mplhep as hep
import matplotlib as mpl

def plotDataMC(data, MC):

        if ratio:
            fig, (ax, rax) = plt.subplots(
                nrows=2,
                ncols=1,
                figsize=(7,7),
                gridspec_kw={"height_ratios": (3, 1)},
                sharex=True
            )
            fig.subplots_adjust(hspace=.07)
        else:
            fig, ax, = plt.subplots(
                nrows=1,
                ncols=1,
                figsize=(7,7)
            )

    