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


# ## Run the processor over desired files  
# In[ ]:   import time  

tstart = time.time()    

fileset = {'JetHT': [ '/mnt/data/cms/store/data/Run2016B/JetHT/NANOAOD/ver1_HIPM_UL2016_MiniAODv2_NanoAODv9-v2/40000/2B449ED9-2A70-6D4D-9AEB-B2870545D35B.root',] }
   
run = processor.Runner(     executor = processor.FuturesExecutor(compression=None, workers=4),     schema=NanoAODSchema,     chunksize=100_000,     # maxchunks=10,  # total 676 chunks ) 

output = run(     fileset,     "Events",     processor_instance=DijetHists(), )   

 # In[ ]:
