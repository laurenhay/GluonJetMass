#### This file contains the processors for dijet and trijet hist selections. Plotting and resulting studies are in separate files.
#### LMH

import argparse

import awkward as ak
import numpy as np
import coffea
import os
import pandas as pd
from plugins import handleData

print(coffea.__version__)
from coffea import util, processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import hist
print(hist.__version__)

parser = argparse.ArgumentParser()

parser.add_argument("year")
parser.add_argument("data")


#####
##### TO DO #####
# find misses --> need to do deltaR matching by hand --> if no reco miss
# need to make rivet routine

##### Do we need underflow/overflow bins?


#### Sal's code --> want to edit to do more than one jet at a time
#### Is this function for gen soft drop mass or 
def get_gen_sd_mass_jet( jet, subjets):
    combs = ak.cartesian( (jet, subjets), axis=1 )
    dr_jet_subjets = combs['0'].delta_r(combs['1'])
    combs = combs[dr_jet_subjets < 0.4]
    total = combs['1'].sum(axis=1)
    return total


bTag_options = ['bbloose', 'bloose', 'bbmed', 'bmed']
def applyBTag(events, btag):
    print('btag input: ', btag, '\n')
    if (btag == 'bbloose'):
        events = events[(events.FatJet[:,0].btagCSVV2 >= 0.460) & (events.FatJet[:,1].btagCSVV2 >= 0.460)]
        print('Loose WP CSV V2 B tag applied to leading two jets')
    elif (btag == 'bloose'):
        events = events[(events.FatJet[:,0].btagCSVV2 >= 0.460)]
        print('Loose WP CSV V2 B tag applied to leading jet only')
    elif (btag == 'bbmed'):
        events = events[(events.FatJet[:,0].btagCSVV2 >= 0.80) & (events.FatJet[:,1].btagCSVV2 >= 0.80)]
        print('Medium WP CSV V2 B tag applied to first two jets')
    elif (btag == 'bmed'):
        events = events[(events.FatJet[:,0].btagCSVV2 >= 0.80)]
        print('Medium WP CSV V2 B tag applied to leading jet only')
    else:
        # printing below when it shoudln't be
        events = events
        print('no btag applied')
    return events

#### currently only for MC --> makes hists and response matrix
class makeDijetHists(processor.ProcessorABC):
    def __init__(self, ptcut=30., etacut = 2.5):
        # should have separate lower ptcut for gen
        self.ptcut = ptcut
        self.etacut = etacut
        dataset_cat = hist.axis.StrCategory([],growth=True,name="dataset", label="Dataset")
        jet_cat = hist.axis.StrCategory([], growth=True, name="jetNumb", label="Jet")
        parton_cat = hist.axis.StrCategory([], growth=True,name="partonFlav", label="Parton Flavour")
        mass_bin = hist.axis.Regular(50, 0, 500.,name="mass", label="Jet Mass (GeV)")
        #### if using specific bin edges use hist.axis.Variable() instead
        mass_gen_bin = hist.axis.Regular(100, 0, 500., name="genmass", label="Gen Jet Mass (GeV)")
        pt_bin = hist.axis.Regular(60, 0, 2400., name="pt",label= "Jet pT (GeV)")
        pt_gen_bin = hist.axis.Regular(60, 0, 2400., name="genpt",label= "Gen Jet pT (GeV)")
        bdisc_bin = hist.axis.Regular(10, 0.0, 1., name="bdisc", label="B-tag discriminator")
        frac_bin = hist.axis.Regular(10, 0.0, 1., name="gfrac", label="Gluon fraction")
        eta_bin = hist.axis.Regular(25, -2.5, 2.5, name="eta", label="Eta")
        hist_dict = {
            'jet_mass':             hist.Hist(dataset_cat, jet_cat, parton_cat, mass_bin, storage="weight", name="Events"),
            'jet_pt':             hist.Hist(dataset_cat, jet_cat, parton_cat, pt_bin, storage="weight", name="Events"),
            'jet_pt_m':           hist.Hist(dataset_cat, jet_cat, pt_bin, mass_bin, storage="weight", name="Events"),
            'jet_eta':            hist.Hist(dataset_cat, jet_cat, parton_cat, eta_bin, storage="weight", name="Events"),
            'jet_sd_mass':        hist.Hist(dataset_cat, jet_cat, parton_cat, mass_bin, storage="weight", name="Events"),
            'gluonPurity':        processor.defaultdict_accumulator(int),
            'nGluonJets':         processor.defaultdict_accumulator(int),
            'cutflow':            processor.defaultdict_accumulator(int),
            'jet_gen_pt':          hist.Hist(dataset_cat, jet_cat, pt_bin, storage="weight", name="Events"),
            'jet_gen_eta':         hist.Hist(dataset_cat, jet_cat, eta_bin,
                                                                 storage="weight",name="Events"),
            'jet_gen_mass':        hist.Hist(dataset_cat, jet_cat, mass_bin,
                                                                 storage="weight",name="Events"),
            'jet_sd_mass':         hist.Hist(dataset_cat, jet_cat, parton_cat, mass_bin, 
                                                                  storage="weight", name="Events"),
            'jet_response':       hist.Hist(dataset_cat, pt_bin, mass_bin, pt_gen_bin,
                                                                mass_gen_bin, storage="weight",
                                                                name="Events"),
                             }
        self._histos = processor.dict_accumulator(hist_dict)
    
    @property
    def accumulator(self):
        return self._histos
    
    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        out = self._histos
        dataset = events.metadata['dataset']
        dijetEvents = events[(ak.num(events.FatJet) >= 2) & (ak.num(events.GenJetAK8) >= 2)]
        
#         for i in range(0,10):
#             print("Check that jets and are ordered by pt: ", events.FatJet[i,:].pt, "\n")
#             print("and that gen jets are order by pt:", events.GenJetAK8[i,:].pt, "\n")
        
        #get leading 2 jets
        fatJets = dijetEvents.FatJet[:,0:2]
        genJets = dijetEvents.GenJetAK8[:,0:2]
        
        print("FatJet fields ", fatJets.fields, "\n")
        print("GenJet fields ", genJets.fields, "\n")
        
        
        jet1 = dijetEvents.FatJet[:,0]
        jet2 = dijetEvents.FatJet[:,1]
        genjet1 = dijetEvents.GenJetAK8[:,0]
        genjet2 = dijetEvents.GenJetAK8[:,1]
        
        print("Initial # of dijet events ", len(fatJets), " and gen dijet events ", len(genJets), "\n")
        
        #calculate dphi_min
        dphi12 = (np.abs(jet1.delta_phi(jet2)) > 2.)
        dphi12_gen = (np.abs(genjet1.delta_phi(genjet2)) > 2.)
        
        #apply dphi gen and reco selection, pt cuts
        #see CMS PAS SMP-20-010 for selections
        
        
        #want to implement pt > 200GeV to be consistent with dijets?
        pt_cut = ak.all(dijetEvents.FatJet.pt > self.ptcut, axis = -1)
        pt_cut_gen = ak.all(dijetEvents.GenJetAK8.pt > self.ptcut, axis = -1)
        
        eta_cut = ak.all(np.abs(dijetEvents.FatJet.eta) < self.etacut, axis = -1)
        eta_cut_gen = ak.all(np.abs(dijetEvents.GenJetAK8.eta) < self.etacut, axis = -1)
        
        asymm = np.abs(jet1.pt - jet2.pt)/(jet1.pt + jet2.pt)
        asymm_gen  = np.abs(genjet1.pt - genjet2.pt)/(genjet1.pt + genjet2.pt)
        
        dijetEvents = dijetEvents[eta_cut & eta_cut_gen & (asymm < 0.3) & (asymm_gen < 0.3) & 
                                  pt_cut & dphi12 & pt_cut_gen & dphi12_gen]
        
        print("# of dijet events after eta cut", len(dijetEvents.FatJet), " and gen dijet events ", len(dijetEvents.GenJetAK8), "\n")
                
        #match jets
        matched = ~ak.any(ak.is_none(dijetEvents.FatJet.matched_gen, axis = -1), axis = -1)
        print("Length of matched ", len(matched))
        # NEED TO MAKE DELTA R MATCHING FOR GEN TO FIND MISSES
        #matched_gen = ~ak.is_none(DNE)
        
        
        # fakes are events reconstructed but does not exist in MC
        fakes = dijetEvents.FatJet[ak.any(ak.is_none(dijetEvents.FatJet.matched_gen, axis = -1), axis = -1)]
        print("Number of fake jets ", len(fakes))
        
        #miss = jet in MC but not reconstructed
        
        #misses = genJets[ak.is_none(DNE)]
        
        dijetEvents = dijetEvents[matched]
        print("Number of matched dijet events", len(dijetEvents))
#         print("Check for none values", ak.any(ak.is_none(dijetEvents, axis = -1)))
        
        jet = dijetEvents.FatJet[:,:2]
        genjet = dijetEvents.GenJetAK8[:,:2]
        
        jet1 = dijetEvents.FatJet[:,0]
        jet2 = dijetEvents.FatJet[:,1]
        genjet1 = dijetEvents.GenJetAK8[:,0]
        genjet2 = dijetEvents.GenJetAK8[:,1]
        
        #flavour --> 21 is gluon
        
        jet2_g     = jet2[np.abs(genjet2.partonFlavour) == 21]
        jet2_uds   = jet2[np.abs(genjet2.partonFlavour) < 4]
        jet2_c     = jet2[np.abs(genjet2.partonFlavour) == 4]
        jet2_b     = jet2[np.abs(genjet2.partonFlavour) == 5]
        jet2_other = jet2[(np.abs(genjet2.partonFlavour) > 5) & (np.abs(genjet2.partonFlavour) != 21)]
        
        jet1_g     = jet1[np.abs(genjet1.partonFlavour) == 21]
        jet1_uds   = jet1[np.abs(genjet1.partonFlavour) < 4]
        jet1_c     = jet1[np.abs(genjet1.partonFlavour) == 4]
        jet1_b     = jet1[np.abs(genjet1.partonFlavour) == 5]
        jet1_other = jet1[(np.abs(genjet1.partonFlavour) > 5) & (np.abs(genjet1.partonFlavour) != 21)]
        
#        #cant do jet_g = jet[jet.partonFlavor == 21] bc this would select events where either jet1 or jet2 is a gluon and keep both jets
#        #jet_g = ak.concatenate((jet1_g, jet2_g), axis = 0)
#        jet_cat = ak.concatenate((jet1, jet2), axis = 0)
#        print("check that total jet cat would work same way. length of concat'd jets", len(ak.concatenate((jet1, jet2), axis = 0)))
#        print(" and flattened jets ",  len(ak.flatten(dijetEvents.FatJet[:,:2])), "\n")
        
#        #make central and forward categories instead of jet1 jet2
#         print("Number of gluon jets = ", len(jet1_g)+len(jet2_g), " or ", len(jet_g))
        
        
        out['jet_mass'].fill(dataset=dataset, jetNumb = "jet1", partonFlav = "Gluon",  mass = jet1_g.mass,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_mass'].fill(dataset=dataset, jetNumb = "jet1", partonFlav = "UDS",    mass = jet1_uds.mass,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_mass'].fill(dataset=dataset, jetNumb = "jet1", partonFlav = "Charm",  mass = jet1_c.mass,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_mass'].fill(dataset=dataset, jetNumb = "jet1", partonFlav = "Bottom", mass = jet1_b.mass,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_mass'].fill(dataset=dataset, jetNumb = "jet1", partonFlav = "Other",  mass = jet1_other.mass,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_mass'].fill(dataset=dataset, jetNumb = "jet2", partonFlav = "Gluon",  mass = jet2_g.mass,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_mass'].fill(dataset=dataset, jetNumb = "jet2", partonFlav = "UDS",    mass = jet2_uds.mass,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_mass'].fill(dataset=dataset, jetNumb = "jet2", partonFlav = "Charm",  mass = jet2_c.mass,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_mass'].fill(dataset=dataset, jetNumb = "jet2", partonFlav = "Bottom", mass = jet2_b.mass,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_mass'].fill(dataset=dataset, jetNumb = "jet2", partonFlav = "Other",  mass = jet2_other.mass,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        
        out['jet_pt'].fill(dataset=dataset, jetNumb = "jet1", partonFlav = "Gluon",  pt = jet1_g.pt,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_pt'].fill(dataset=dataset, jetNumb = "jet1", partonFlav = "UDS",    pt = jet1_uds.pt,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_pt'].fill(dataset=dataset, jetNumb = "jet1", partonFlav = "Charm",  pt = jet1_c.pt,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_pt'].fill(dataset=dataset, jetNumb = "jet1", partonFlav = "Bottom", pt = jet1_b.pt,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_pt'].fill(dataset=dataset, jetNumb = "jet1", partonFlav = "Other",  pt = jet1_other.pt,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_pt'].fill(dataset=dataset, jetNumb = "jet2", partonFlav = "Gluon",  pt = jet2_g.pt,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_pt'].fill(dataset=dataset, jetNumb = "jet2", partonFlav = "UDS",    pt = jet2_uds.pt,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_pt'].fill(dataset=dataset, jetNumb = "jet2", partonFlav = "Charm",  pt = jet2_c.pt,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_pt'].fill(dataset=dataset, jetNumb = "jet2", partonFlav = "Bottom", pt = jet2_b.pt,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_pt'].fill(dataset=dataset, jetNumb = "jet2", partonFlav = "Other",  pt = jet2_other.pt,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )

        out['jet_eta'].fill(dataset=dataset, jetNumb = "jet1", partonFlav = "Gluon",  eta = jet1_g.eta,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_eta'].fill(dataset=dataset, jetNumb = "jet1", partonFlav = "UDS",    eta = jet1_uds.eta,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_eta'].fill(dataset=dataset, jetNumb = "jet1", partonFlav = "Charm",  eta = jet1_c.eta,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_eta'].fill(dataset=dataset, jetNumb = "jet1", partonFlav = "Bottom", eta = jet1_b.eta,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_eta'].fill(dataset=dataset, jetNumb = "jet1", partonFlav = "Other",  eta = jet1_other.eta,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_eta'].fill(dataset=dataset, jetNumb = "jet2", partonFlav = "Gluon",  eta = jet2_g.eta,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_eta'].fill(dataset=dataset, jetNumb = "jet2", partonFlav = "UDS",    eta = jet2_uds.eta,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_eta'].fill(dataset=dataset, jetNumb = "jet2", partonFlav = "Charm",  eta = jet2_c.eta,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_eta'].fill(dataset=dataset, jetNumb = "jet2", partonFlav = "Bottom", eta = jet2_b.eta,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_eta'].fill(dataset=dataset, jetNumb = "jet2", partonFlav = "Other",  eta = jet2_other.eta,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )

        #NOTE --> need gen sd mass eventually --> recluster :( 
#         out['genjet_mass'].fill(
#             mass=ak.flatten(genjet.mass),
#             #weight=ak.flatten(dijetEvents.Generator.weight[~ak.is_none(dijetEvents.FatJet[:,0].matched_gen)])
#         )
            
#         out['weights'].fill(
#             dataset=events.metadata["dataset"],
#             weights=dijetEvents.Generator.weight,
#         )     
#         out['btag'].fill(
#             partonFlav = "Bottom",
#             bdisc=ak.flatten(jet_b.btagCSVV2),
#         )
        out['cutflow']['chunks'] += 1
        out['cutflow']['nGluonJets'] += (len(jet[np.abs(genjet.partonFlavour) == 21]))
        return out
    
    def postprocess(self, accumulator):
        return accumulator

##### TO DO #####
#make mass vs pt and response matrix (pt_gen, mass_gen, pt_reco, mass_reco)
# Add eta/phi/delta_r/pt cuts fully
# Make 2 eta collections --> high eta (>1.7) and central (<1.7)
# add same cuts on GenJetAK8
# find misses --> need to do deltaR matching by hand --> if no reco miss
# do Rivet routine
# make central (eta < 1.7) and high eta bins (1.7 < eta < 2.5)
# try AK4 jets to give low pT ??
# remove phi cuts --  why past me why?? do you mean try with and without phi cuts?

#bcut options: b_loose (apply loose bTag threshold to only hardest jet), bb_loose (apply loose bTag to leading two jets),
#              b_med(apply medium bTag to only the hardest jet), bb_med (apply medium bTag to leading two jets)

class makeTrijetHists(processor.ProcessorABC):
    def __init__(self, ptCut = 200., etaCut = 2.4, btag = 'null'):
        self.ptCut = ptCut
        self.etaCut = etaCut
        self.btag = btag
        dataset_cat = hist.axis.StrCategory([],growth=True,name="dataset", label="Dataset")
        jet_cat = hist.axis.StrCategory([], growth=True, name="jetNumb", label="Jet")
        parton_cat = hist.StrCategory([],growth=True,name="partonFlav", label="Parton Flavour")
        mass_bins = hist.axis.Regular(60, 0, 200,name="mass", label="Jet Mass (GeV)")
        pt_bins = hist.axis.Regular(60, 0, 2400, name="pt", label="Jet pT (GeV)")
        disc_bins = hist.axis.Regular(10, 0.0, 1.,name="bdisc", label="B-tag discriminator")
        eta_bins = hist.axis.Regular(10, -2.5, 2.5, name = "eta", label="Eta")
        self._histos = processor.dict_accumulator({
        'jet_pt':          hist.Hist("Events", jet_cat, parton_cat, pt_bins),
        'jet_eta':         hist.Hist("Events", jet_cat, parton_cat, eta_bins),
        'jet_mass':        hist.Hist("Events", jet_cat, parton_cat, mass_bins),
        'genjet_pt':       hist.Hist("Events", jet_cat, pt_bins),
        'genjet_eta':      hist.Hist("Events", jet_cat, eta_bins),
        'genjet_mass':     hist.Hist("Events", jet_cat, mass_bins),
        'btag':            hist.Hist("Events", jet_cat, parton_cat, disc_bins),
        'gluonPurity':     hist.Hist("Events", parton_cat, disc_bins),
        'nGluonJets':      hist.Hist("Events", jet_cat, ),
        'cutflow':         processor.defaultdict_accumulator(int),
        })
    
    @property
    def accumulator(self):
        return self._histos
    def process(self, events):
        out = self.accumulator.identity()
        trijetEvents = events[
            (ak.num(events.FatJet) >= 3) & (ak.num(events.GenJetAK8) >= 3)
        ]
        #get leading 3 jets
        print("Initial number of matched trijet events after all cuts ", len(trijetEvents), '\n')
        jet1 = trijetEvents.FatJet[:, 0]
        jet2 = trijetEvents.FatJet[:, 1]
        jet3 = trijetEvents.FatJet[:, 2]
        
#        print("Softest jets after >3 selection", len(jet3))
        
        
        #calculate dphi_min
        dphi12 = np.abs(jet1.delta_phi(jet2))
        dphi13 = np.abs(jet1.delta_phi(jet3))
        dphi23 = np.abs(jet2.delta_phi(jet3))
        
        dphi_min = np.amin([dphi12, dphi13, dphi23], axis = 0)
        
#         print("Fields available for FatJets ", trijetEvents.FatJet.fields)
#         print("Fields available for GenJets ", trijetEvents.GenJetAK8.fields)
        
        #do same for gen; jets might not be same order reco and gen
        genjet1 = trijetEvents.GenJetAK8[:,0]
        genjet2 = trijetEvents.GenJetAK8[:,1]
        genjet3 = trijetEvents.GenJetAK8[:,2]
        
        dphi12_gen = np.abs(genjet1.delta_phi(genjet2))
        dphi13_gen = np.abs(genjet1.delta_phi(genjet3))
        dphi23_gen = np.abs(genjet2.delta_phi(genjet3))
        
        dphimin_gen = np.amin([dphi12_gen, dphi13_gen, dphi23_gen], axis = 0)
        
        #revisit pt cut at some point?
        pt_cut = (ak.all(trijetEvents.FatJet.pt > self.ptCut, axis = -1) &
                  ak.all(trijetEvents.GenJetAK8.pt > self.ptCut, axis = -1))
        eta_cut = (ak.all(np.abs(trijetEvents.FatJet.eta) < self.etaCut, axis = -1) &
                   ak.all(np.abs(trijetEvents.GenJetAK8.eta) < self.etaCut, axis = -1))
        asymm = np.abs(jet1.pt - jet2.pt)/(jet1.pt + jet2.pt)
        asymm_gen  = np.abs(genjet1.pt - genjet2.pt)/(genjet1.pt + genjet2.pt)
        
        #apply dphi gen and reco selection, eta, and pt cut
        #selections are based on arXiv:1807.05974v2

        trijetEvents = trijetEvents[(dphimin_gen > 1.0) & (dphi_min > 1.0) & (pt_cut) & (eta_cut) &
                                    (asymm < 0.3) & (asymm_gen < 0.3)
                                    ]
        print(trijetEvents)
        #make matching mask - where there is any none in FatJet give False value
        matched = ~ak.any(ak.is_none(trijetEvents.FatJet.matched_gen, axis = -1), axis = -1)
        # NEED TO MAKE DELTA R MATCHING FOR GEN TO FIND MISSES
        #gen_matched = ~ak.is_none(DNE)
        
        
        # fakes are events reconstructed but does not exist in MC
        fakes = trijetEvents.FatJet[ak.any(ak.is_none(trijetEvents.FatJet.matched_gen, axis = -1), axis =
                                           -1)]
        print("Number of fake jets ", len(fakes))
        
        #miss = jet in MC but not reconstructed
        #misses = genJets[ak.any(ak.is_none(DNE), axis = -1)]
        
        trijetEvents = trijetEvents[matched]
        print("Number of matched trijet events after matching ", len(trijetEvents), '\n')
        
        
        #apply bTag
        trijetEvents = applyBTag(trijetEvents, self.btag)
        print("Number of matched trijet events after all cuts ", len(trijetEvents), '\n')
        
        
        jet1 = trijetEvents.FatJet[:,0]
        jet2 = trijetEvents.FatJet[:,1]
        jet3 = trijetEvents.FatJet[:,2]
        
        genjet1 = jet1.matched_gen
        genjet2 = jet2.matched_gen
        genjet3 = jet3.matched_gen
        

#        print("Softest jets after dphi selection and matching", len(jet3))
        
        #flavour --> 21 is gluon
        jet3_g     = jet3[np.abs(genjet3.partonFlavour) == 21]
        jet3_uds   = jet3[np.abs(genjet3.partonFlavour) < 4]
        jet3_c     = jet3[np.abs(genjet3.partonFlavour) == 4]
        jet3_b     = jet3[np.abs(genjet3.partonFlavour) == 5]
        jet3_other = jet3[(np.abs(genjet3.partonFlavour) > 5) & (np.abs(genjet3.partonFlavour) != 21)]
        
        jet2_g     = jet2[np.abs(genjet2.partonFlavour) == 21]
        jet2_uds   = jet2[np.abs(genjet2.partonFlavour) < 4]
        jet2_c     = jet2[np.abs(genjet2.partonFlavour) == 4]
        jet2_b     = jet2[np.abs(genjet2.partonFlavour) == 5]
        jet2_other = jet2[(np.abs(genjet2.partonFlavour) > 5) & (np.abs(genjet2.partonFlavour) != 21)]
        
        jet1_g     = jet1[np.abs(genjet1.partonFlavour) == 21]
        jet1_uds   = jet1[np.abs(genjet1.partonFlavour) < 4]
        jet1_c     = jet1[np.abs(genjet1.partonFlavour) == 4]
        jet1_b     = jet1[np.abs(genjet1.partonFlavour) == 5]
        jet1_other = jet1[(np.abs(genjet1.partonFlavour) > 5) & (np.abs(genjet1.partonFlavour) != 21)]
        
        print("Check for none values", ak.any(ak.is_none(jet3_g.mass), axis = -1))
        
        print("Gluon purity of third jet for basic selection: ", len(jet3_g)/len(ak.flatten(jet3)))
        
        
        out['jet_mass'].fill(jetNumb = "jet1", partonFlav = "Gluon",  mass = jet1_g.mass,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_mass'].fill(jetNumb = "jet1", partonFlav = "UDS",    mass = jet1_uds.mass,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_mass'].fill(jetNumb = "jet1", partonFlav = "Charm",  mass = jet1_c.mass,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_mass'].fill(jetNumb = "jet1", partonFlav = "Bottom", mass = jet1_b.mass,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_mass'].fill(jetNumb = "jet1", partonFlav = "Other",  mass = jet1_other.mass,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_mass'].fill(jetNumb = "jet2", partonFlav = "Gluon",  mass = jet2_g.mass,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_mass'].fill(jetNumb = "jet2", partonFlav = "UDS",    mass = jet2_uds.mass,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_mass'].fill(jetNumb = "jet2", partonFlav = "Charm",  mass = jet2_c.mass,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_mass'].fill(jetNumb = "jet2", partonFlav = "Bottom", mass = jet2_b.mass,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_mass'].fill(jetNumb = "jet2", partonFlav = "Other",  mass = jet2_other.mass,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_mass'].fill(jetNumb = "jet3", partonFlav = "Gluon",  mass = jet3_g.mass,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_mass'].fill(jetNumb = "jet3", partonFlav = "UDS",    mass = jet3_uds.mass,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_mass'].fill(jetNumb = "jet3", partonFlav = "Charm",  mass = jet3_c.mass,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_mass'].fill(jetNumb = "jet3", partonFlav = "Bottom", mass = jet3_b.mass,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_mass'].fill(jetNumb = "jet3", partonFlav = "Other",  mass = jet3_other.mass,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_pt'].fill(jetNumb = "jet1", partonFlav = "Gluon",  pt = jet1_g.pt,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_pt'].fill(jetNumb = "jet1", partonFlav = "UDS",    pt = jet1_uds.pt,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_pt'].fill(jetNumb = "jet1", partonFlav = "Charm",  pt = jet1_c.pt,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_pt'].fill(jetNumb = "jet1", partonFlav = "Bottom", pt = jet1_b.pt,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_pt'].fill(jetNumb = "jet1", partonFlav = "Other",  pt = jet1_other.pt,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_pt'].fill(jetNumb = "jet2", partonFlav = "Gluon",  pt = jet2_g.pt,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_pt'].fill(jetNumb = "jet2", partonFlav = "UDS",    pt = jet2_uds.pt,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_pt'].fill(jetNumb = "jet2", partonFlav = "Charm",  pt = jet2_c.pt,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_pt'].fill(jetNumb = "jet2", partonFlav = "Bottom", pt = jet2_b.pt,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_pt'].fill(jetNumb = "jet2", partonFlav = "Other",  pt = jet2_other.pt,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_pt'].fill(jetNumb = "jet3", partonFlav = "Gluon",  pt = jet3_g.pt,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_pt'].fill(jetNumb = "jet3", partonFlav = "UDS",    pt = jet3_uds.pt,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_pt'].fill(jetNumb = "jet3", partonFlav = "Charm",  pt = jet3_c.pt,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_pt'].fill(jetNumb = "jet3", partonFlav = "Bottom", pt = jet3_b.pt,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_pt'].fill(jetNumb = "jet3", partonFlav = "Other",  pt = jet3_other.pt,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_eta'].fill(jetNumb = "jet1", partonFlav = "Gluon",  eta = jet1_g.eta,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_eta'].fill(jetNumb = "jet1", partonFlav = "UDS",    eta = jet1_uds.eta,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_eta'].fill(jetNumb = "jet1", partonFlav = "Charm",  eta = jet1_c.eta,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_eta'].fill(jetNumb = "jet1", partonFlav = "Bottom", eta = jet1_b.eta,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_eta'].fill(jetNumb = "jet1", partonFlav = "Other",  eta = jet1_other.eta,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_eta'].fill(jetNumb = "jet2", partonFlav = "Gluon",  eta = jet2_g.eta,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_eta'].fill(jetNumb = "jet2", partonFlav = "UDS",    eta = jet2_uds.eta,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_eta'].fill(jetNumb = "jet2", partonFlav = "Charm",  eta = jet2_c.eta,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_eta'].fill(jetNumb = "jet2", partonFlav = "Bottom", eta = jet2_b.eta,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_eta'].fill(jetNumb = "jet2", partonFlav = "Other",  eta = jet2_other.eta,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_eta'].fill(jetNumb = "jet3", partonFlav = "Gluon",  eta = jet3_g.eta,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_eta'].fill(jetNumb = "jet3", partonFlav = "UDS",    eta = jet3_uds.eta,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_eta'].fill(jetNumb = "jet3", partonFlav = "Charm",  eta = jet3_c.eta,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_eta'].fill(jetNumb = "jet3", partonFlav = "Bottom", eta = jet3_b.eta,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['jet_eta'].fill(jetNumb = "jet3", partonFlav = "Other",  eta = jet3_other.eta,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        #NOTE --> need gen sd mass eventually --> recluster :( 
        out['genjet_mass'].fill(
            jetNumb = "genjet1",
            mass=genjet1.mass,
            #weight=trijetEvents.Generator.weight[~ak.is_none(trijetEvents.FatJet[:,0].matched_gen)]
        )
        out['genjet_mass'].fill(
            jetNumb = "genjet2",
            mass=genjet2.mass,
            #weight=trijetEvents.Generator.weight[matched2]
        )
        out['genjet_mass'].fill(
            jetNumb = "genjet3",
            mass=genjet3.mass,
            #weight=trijetEvents.Generator.weight[matched3]
        )
        out['weights'].fill(
            dataset=events.metadata["dataset"],
            weights=trijetEvents.Generator.weight,
        )     
        out['btag'].fill(jetNumb = "jet1", partonFlav = "Gluon",  bdisc = jet1_g.btagCSVV2,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['btag'].fill(jetNumb = "jet1", partonFlav = "UDS",    bdisc = jet1_uds.btagCSVV2,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['btag'].fill(jetNumb = "jet1", partonFlav = "Charm",  bdisc = jet1_c.btagCSVV2,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['btag'].fill(jetNumb = "jet1", partonFlav = "Bottom", bdisc = jet1_b.btagCSVV2,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['btag'].fill(jetNumb = "jet1", partonFlav = "Other",  bdisc = jet1_other.btagCSVV2,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['btag'].fill(jetNumb = "jet2", partonFlav = "Gluon",  bdisc = jet2_g.btagCSVV2,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['btag'].fill(jetNumb = "jet2", partonFlav = "UDS",    bdisc = jet2_uds.btagCSVV2,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['btag'].fill(jetNumb = "jet2", partonFlav = "Charm",  bdisc = jet2_c.btagCSVV2,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['btag'].fill(jetNumb = "jet2", partonFlav = "Bottom", bdisc = jet2_b.btagCSVV2,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['btag'].fill(jetNumb = "jet2", partonFlav = "Other",  bdisc = jet2_other.btagCSVV2,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['btag'].fill(jetNumb = "jet3", partonFlav = "Gluon",  bdisc = jet3_g.btagCSVV2,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['btag'].fill(jetNumb = "jet3", partonFlav = "UDS",    bdisc = jet3_uds.btagCSVV2,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['btag'].fill(jetNumb = "jet3", partonFlav = "Charm",  bdisc = jet3_c.btagCSVV2,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['btag'].fill(jetNumb = "jet3", partonFlav = "Bottom", bdisc = jet3_b.btagCSVV2,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['btag'].fill(jetNumb = "jet3", partonFlav = "Other",  bdisc = jet3_other.btagCSVV2,
                             #weight = trijetEvents.Generator.weight[matched1]
                            )
        out['cutflow']['chunks'] += 1
        out['cutflow']['nGluonJets'] += (len(jet3[np.abs(genjet3.partonFlavour) == 21]))
        return out
    
    def postprocess(self, accumulator):
        return accumulator
    
def main():
    #### Next run processor with futures executor on all test files
    from dask.distributed import Client
    from plugins import runCoffeaJob
    processor = makeDijetHists()
    result = runCoffeaJob(processor, jsonFile = "fileset_QCD.json", casa = True, testing = True, dask=False)
    util.save(result, "coffeaOutput/dijet_pT" + str(processor.ptcut) + "_eta" + str(processor.etacut) + "_result_test.coffea")
    

if __name__ == '__main__':
    # Execute when the module is not initialized from an import statement: i.e. called from terminal command line
    main()

