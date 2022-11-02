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


# ## Create the coffea Processor

# In[2]:


class DijetHists(processor.ProcessorABC):
    def __init__(self, ptcut=200., etacut = 2.5):
        # should have separate lower ptcut for gen
        self.ptcut = ptcut
        self.etacut = etacut
        
        dataset_axis = hist.axis.StrCategory([], growth=True, name="dataset", label="Primary dataset")
        mass_axis = hist.axis.Regular(100, 0, 1000, name="mass", label=r"$m$ [GeV]")
        pt_axis = hist.axis.Regular(150, 0, 1500, name="pt", label=r"$p_{T}$ [GeV]") 
        asym_axis = hist.axis.Regular(150, 0, 1, name="asym", label=r"$p_{T}$_asym")
        dphi_axis = hist.axis.Regular(150, 0, np.pi, name="dphi", label=r"$\Delta \phi$") 
        
        h_pt1 = hist.Hist(dataset_axis, pt_axis, storage="weight", label="Counts")
        h_pt2 = hist.Hist(dataset_axis, pt_axis, storage="weight", label="Counts")
        h_asym = hist.Hist(dataset_axis, asym_axis, storage="weight", label="Counts")
        h_dphi = hist.Hist(dataset_axis, dphi_axis, storage="weight", label="Counts")
        
        
        cutflow = defaultdict(int)
        
        self.hists = {
            "pt1":h_pt1, "pt2":h_pt2, "cutflow":cutflow,
            "asym":h_asym,"dphi":h_dphi
        }
        
    
    @property
    def accumulator(self):
        return self._histos
    
    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        dataset = events.metadata['dataset']
        
        
        # Select all jets that satisfy pt, eta, and jet ID cuts
        events.FatJet = events.FatJet[ 
            (events.FatJet.pt > self.ptcut) & 
            (np.abs(events.FatJet.eta) < self.etacut) &
            (events.FatJet.jetId > 0)]

        # Require at least two such jets per event
        dijetEvents = events[(ak.num(events.FatJet) >= 2)]
        
        # Get the leading and subleading jets
        jet1 = dijetEvents.FatJet[:,0]
        jet2 = dijetEvents.FatJet[:,1]
        
        # Select events with delta phi and asymmetry cuts
        dphi12 = np.abs(jet1.delta_phi(jet2))
        asymm = np.abs((jet1.pt - jet2.pt)/(jet1.pt + jet2.pt))
        dphi12_cut = (dphi12 > 2.)
        asymm_cut = (asymm < 0.3)        
        dijetEvents = dijetEvents[ asymm_cut & dphi12_cut ]
        
        # Get the leading and subleading jets of the events that pass selection
        jet1 = dijetEvents.FatJet[:,0]
        jet2 = dijetEvents.FatJet[:,1]
        
        # Make plots
        self.hists["pt1"].fill(dataset=dataset, pt = jet1.pt)
        self.hists["pt2"].fill(dataset=dataset, pt = jet2.pt)
        self.hists["asym"].fill(dataset=dataset, asym = asymm)
        self.hists["dphi"].fill(dataset=dataset, dphi = dphi12)
        

        return self.hists
    
    def postprocess(self, accumulator):
        return accumulator




