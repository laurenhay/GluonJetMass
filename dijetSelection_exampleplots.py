#!/usr/bin/env python
# coding: utf-8

# # Dijet Selection Example Processor
# 
# We create a dijet selection that selects events that have at least 2 jets with: 
# * $p_{T} > 200 GeV$
# * $|\eta| < 2.5$
# * "Loose" jet ID
# 
# The selection then imposes two topological selections : 
# * $\Delta \phi > 2$
# * $\frac{p_{T,1} - p_{T,2}}{p_{T,1} + p_{T,2}} < 0.3$

# In[1]:


import awkward as ak
import numpy as np
import time
import coffea
import uproot
import hist
print(coffea.__version__)
from coffea import util, processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from collections import defaultdict




# ## Make plots

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


fig, ax = plt.subplots()
output['pt1'].plot1d(ax=ax, overlay='dataset')
ax.set_yscale('log')
ax.set_ylim(1, None)


# In[ ]:


fig, ax = plt.subplots()
output['asym'].plot1d(ax=ax, overlay='dataset')
ax.set_yscale('log')
ax.set_ylim(1, None)


# In[ ]:


fig, ax = plt.subplots()
output['dphi'].plot1d(ax=ax, overlay='dataset')
ax.set_yscale('log')
ax.set_ylim(1, None)
