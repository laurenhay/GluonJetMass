#### This file contains the processors for dijet and trijet hist selections. Plotting and resulting studies are in separate files.
#### LMH

import argparse

import awkward as ak
import numpy as np
import coffea
import os
import re
import pandas as pd
from plugins import handleData

print(coffea.__version__)
from coffea import util, processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import PackedSelection
from collections import defaultdict
from utils import *
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
    print("Genjet and subjet combinations: ", combs)
    dr_jet_subjets = combs['0'].delta_r(combs['1'])
    combs = combs[dr_jet_subjets < 0.8]
    total = combs['1'].sum(axis=1)
    return total 

def get_dphi( coll0, coll1 ):
    '''
    Find dphi between 3rd jet and , returning none when the event does not have at least two jets
    '''
    combs = ak.cartesian( (coll0, coll1), axis=1 )
    print("Coll. 0: ", len(coll0), '\n', combs['0'])
    print("Coll. 1: ", len(coll1), '\n', combs['1'])
    dphi = np.abs(combs['0'].delta_phi(combs['1']))
    return ak.firsts((combs['1'])).ak.firsts(dphi)


bTag_options = ['bbloose', 'bloose', 'bbmed', 'bmed']
def applyBTag(events, btag):
    print('btag input: ', btag, '\n')
    if (btag == 'bbloose'):
        sel = (events.FatJet[:,0].btagCSVV2 >= 0.460) & (events.FatJet[:,1].btagCSVV2 >= 0.460)
        events = events[sel]
        print('Loose WP CSV V2 B tag applied to leading two jets')
    elif (btag == 'bloose'):
        sel = (events.FatJet[:,0].btagCSVV2 >= 0.460)
        events = events[sel]
        print('Loose WP CSV V2 B tag applied to leading jet only')
    elif (btag == 'bbmed'):
        sel = (events.FatJet[:,0].btagCSVV2 >= 0.80) & (events.FatJet[:,1].btagCSVV2 >= 0.80)
        events = events[sel]
        print('Medium WP CSV V2 B tag applied to first two jets')
    elif (btag == 'bmed'):
        sel = (events.FatJet[:,0].btagCSVV2 >= 0.80)
        events = events[sel]
        print('Medium WP CSV V2 B tag applied to leading jet only')
    else:
        sel = np.ones(len(events), dtype = bool)
        print('no btag applied')
    return events, sel

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
    def __init__(self, ptcut = 200., etacut = 2.4, btag = 'null', data = 'False'):
        self.ptcut = ptcut
        self.etacut = etacut
        self.btag = btag
        self.do_gen = not data
        print("Data: ", data, " gen ", self.do_gen)
        
        dataset_cat = hist.axis.StrCategory([],growth=True,name="dataset", label="Dataset")
        jet_cat = hist.axis.StrCategory([], growth=True, name="jetNumb", label="Jet")
        parton_cat = hist.axis.StrCategory([],growth=True,name="partonFlav", label="Parton Flavour")
        
        #### if using specific bin edges use hist.axis.Variable() instead
        mass_bin =  hist.axis.Variable([0,5,10,20,40,60,80,100,150,200,250,1000], name="mreco", label=r"m_{RECO} (GeV)")
        mass_gen_bin = hist.axis.Variable([0,2.5,5,7.5,10,15,20,30,40,50,60,70,80,90,
                                           100,125,150,175,200,225,250,1000], name="mgen", label=r"m_{GEN} (GeV)")
        pt_bin = hist.axis.Variable([200,280,360,450,520,630,690,750,800,1300,13000], name="ptreco", label=r"p_{T,RECO} (GeV)")   
        pt_gen_bin = hist.axis.Variable([200,280,360,450,520,630,690,750,800,1300,13000], name="ptgen", label=r"p_{T,GEN} (GeV)") 
#         mass_bin = hist.axis.Regular(60, 0, 1000.,name="mreco", label="Jet Mass (GeV)")
#         mass_gen_bin = hist.axis.Regular(60, 0, 1000., name="mgen", label="Gen Jet Mass (GeV)")
#         pt_bin = hist.axis.Regular(60, 0, 3000., name="ptreco",label= "Jet pT (GeV)")
#         pt_gen_bin = hist.axis.Regular(60, 0, 3000., name="ptgen",label= "Gen Jet pT (GeV)")
        eta_bin = hist.axis.Regular(10, -2.5, 2.5, name = "eta", label="Eta")
        frac_axis = hist.axis.Regular(10, 0.0, 1., name="frac", label="Fraction")
        n_axis = hist.axis.Regular(5, 0, 5, name="n", label=r"Number")
        dr_axis = hist.axis.Regular(150, 0, 6.0, name="dr", label=r"$\Delta R$")
        dphi_axis = hist.axis.Regular(150, -2*np.pi, 2*np.pi, name="dphi", label=r"$\Delta \phi$")

        self._histos = {
        #### Old histos
        'jet_mass':        hist.Hist(dataset_cat, jet_cat, parton_cat, mass_gen_bin, storage="weight", name="Events"),
        'jet_pt':          hist.Hist(dataset_cat, jet_cat, parton_cat, pt_gen_bin, storage="weight", name="Events"),
        'jet_eta':         hist.Hist(dataset_cat, jet_cat, parton_cat, eta_bin, storage="weight", name="Events"),
        'btag':            hist.Hist(dataset_cat, jet_cat, parton_cat, frac_axis, storage="weight", name="Events"),
        #### misc.
        'cutflow':            processor.defaultdict_accumulator(int),
            
        #### Plots of things during the selection process / for debugging
        'njet_gen':                  hist.Hist(dataset_cat, n_axis, storage="weight", label="Counts"),
        'dphimin_gen':               hist.Hist(dataset_cat, dphi_axis, storage="weight", label="Counts"),
        'asymm_gen':               hist.Hist(dataset_cat, frac_axis, storage="weight", label="Counts"),
        'njet_reco':                  hist.Hist(dataset_cat, n_axis, storage="weight", label="Counts"),
        'dphimin_reco':               hist.Hist(dataset_cat, dphi_axis, storage="weight", label="Counts"),
        'asymm_reco':               hist.Hist(dataset_cat, frac_axis, storage="weight", label="Counts"),
            
        'jet_dr_reco_gen':           hist.Hist(dataset_cat, dr_axis, storage="weight", label="Counts"),
        'jet_mass_reco_over_gen':    hist.Hist(dataset_cat, frac_axis, storage="weight", label="Counts"),
        'jet_pt_reco':               hist.Hist(dataset_cat, jet_cat, parton_cat, pt_bin, storage="weight", name="Events"),
        'jet_pt_gen':                hist.Hist(dataset_cat, jet_cat, pt_bin, storage="weight", name="Events"),
        'jet_pt_reco_over_gen':      hist.Hist(dataset_cat, frac_axis, storage="weight", label="Counts"),
        'jet_eta_reco':              hist.Hist(dataset_cat, jet_cat, parton_cat, eta_bin, storage="weight", name="Events"),
        'jet_eta_gen':               hist.Hist(dataset_cat, jet_cat, eta_bin, storage="weight",name="Events"),
        #'jet_dr_gen':                hist.Hist(dataset_cat, dr_axis, storage="weight", label="Counts"),
        #'jet_dr_reco':               hist.Hist(dataset_cat, dr_axis, storage="weight", label="Counts"),
        'jet_dphi_reco':             hist.Hist(dataset_cat, dphi_axis, storage="weight", label="Counts"),
        
        'jet_ptasymm_reco':          hist.Hist(dataset_cat, frac_axis, storage="weight", label="Counts"),
        'jet_dr_gen_subjet':         hist.Hist(dataset_cat, dr_axis, storage="weight", label="Counts"),
        'jet_dr_reco_to_gen_subjet': hist.Hist(dataset_cat, dr_axis, storage="weight", label="Counts"),
        'jet_sd_mass_reco':          hist.Hist(dataset_cat, jet_cat, parton_cat, mass_bin, storage="weight", name="Events"),
        'jet_sd_mass_gen':           hist.Hist(dataset_cat, jet_cat, parton_cat, mass_bin, storage="weight", name="Events"),
        'misses':       hist.Hist(dataset_cat, pt_gen_bin, mass_gen_bin, storage="weight", name="Events"),
        'fakes':        hist.Hist(dataset_cat, pt_bin, mass_bin, storage="weight", name="Events"),
         
            
        #### Plots to be unfolded
        'jet_pt_mass_reco_u':       hist.Hist(dataset_cat, pt_bin, mass_bin, storage="weight", name="Events"),
        'jet_pt_mass_reco_g':        hist.Hist(dataset_cat, pt_bin, mass_bin, storage="weight", name="Events"),

        #### Plots for comparison
        'jet_pt_mass_gen_u':        hist.Hist(dataset_cat, pt_gen_bin, mass_gen_bin, storage="weight", label="Counts"),       
        'jet_pt_mass_gen_g':            hist.Hist(dataset_cat, pt_gen_bin, mass_gen_bin, storage="weight", label="Counts"),
        
        #### Plots to get JMR and JMS in MC
        'jet_m_pt_u_reco_over_gen': hist.Hist(dataset_cat, pt_gen_bin, mass_gen_bin, frac_axis, storage="weight",                                                                        label="Counts"),
        'jet_m_pt_g_reco_over_gen':  hist.Hist(dataset_cat, pt_gen_bin, mass_gen_bin, frac_axis, storage="weight",                                                                        label="Counts"),

        #### Plots for the analysis in the proper binning
        'response_matrix_u':    hist.Hist(dataset_cat, pt_bin, mass_bin, pt_gen_bin, mass_gen_bin, storage="weight",                                                         label="Counts"),
        'response_matrix_g':     hist.Hist(dataset_cat, pt_bin, mass_bin, pt_gen_bin, mass_gen_bin, storage="weight",                                                         label="Counts"),
        }
    
    @property
    def accumulator(self):
        return self._histos
    def process(self, events):
        out = self._histos
        dataset = events.metadata['dataset']
        filename = events.metadata['filename']
        print(dataset)
        #####################################
        #### Find the IOV from the dataset name
        #####################################
        IOV = ('2016APV' if ( any(re.findall(r'HIPM',  dataset)))
               else '2018'    if ( any(re.findall(r'UL18', dataset)) or any(re.findall(r'UL2018',    dataset)))
               else '2017'    if ( any(re.findall(r'UL17', dataset)) or any(re.findall(r'UL2017',    dataset)))
               else '2016')
        
        #####################################
        #### Find the era from the file name
        #### Apply the good lumi mask
        #####################################
        weights = np.ones(len(events)) 
        print("Lenght of events ", len(events), "length of weights ", len(weights))
        if (self.do_gen):
            era = None   
        else:
            firstidx = filename.find( "store/data/" )
            fname2 = filename[firstidx:]
            fname_toks = fname2.split("/")
            era = fname_toks[ fname_toks.index("data") + 1]
            print("IOV ", IOV, ", era ", era)
            # apply lumimask and require at least one jet to apply jet trigger prescales
            print("Event runs: ", events.run)
            lumi_mask = getLumiMask(IOV)(events.run, events.luminosityBlock)
            events = events[lumi_mask & (ak.num(events.FatJet) >= 1)]
            print("Event runs: ", events.run)
            trigsel, psweights = applyPrescales(events, year = IOV)
            weights=psweights
            print("Trigger: len of events ", len(events), "len of weights ", len(trigsel))
            print(weights)
            events = events[trigsel]
            weights = weights[trigsel]
        
        #### Need to add PU reweighting for if do_gen
        #### Remove event with very large gen weights???
        sel = PackedSelection()
        sel.add("npv", events.PV.npvsGood>0)

        #####################################
        #### Gen Jet Selection
        ####################################       
        if self.do_gen:
            sel.add("threeGenJets", ak.num(events.GenJetAK8) >= 3)
            pt_cut_gen = ak.all(events.GenJetAK8.pt > 140., axis = -1) ### 70% of reco pt cut
            eta_cut_gen = ak.all(np.abs(events.GenJetAK8.eta) < self.etacut, axis = -1)
            sel.add("ptEtaCutGen", (pt_cut_gen & eta_cut_gen))
            kinesel = sel.all("threeGenJets", "ptEtaCutGen")
            out["njet_gen"].fill(dataset=dataset, n=ak.num(events.GenJetAK8[eta_cut_gen & pt_cut_gen]), 
                                 weight = weights[eta_cut_gen & pt_cut_gen] )
            print("Initial # of events:  ", len(events.GenJetAK8))
                             
            #### Get leading 3 jets
            events = events[kinesel]
            weights = weights[kinesel]
            print("Number of trijet events after kinematic cuts ", len(events.GenJetAK8), '\n')
            genjet1 = events.GenJetAK8[:,0]
            genjet2 = events.GenJetAK8[:,1]
            genjet3 = events.GenJetAK8[:,2]
       
            ### calculate dphi_min
            dphi12_gen = np.abs(genjet1.delta_phi(genjet2))
            dphi13_gen = np.abs(genjet1.delta_phi(genjet3))
            dphi23_gen = np.abs(genjet2.delta_phi(genjet3))
            dphimin_gen = np.amin([dphi12_gen, dphi13_gen, dphi23_gen], axis = 0)
            out["dphimin_gen"].fill(dataset=dataset, dphi = dphimin_gen, weight = weights)
            
            asymm_gen  = np.abs(genjet1.pt - genjet2.pt)/(genjet1.pt + genjet2.pt)
            out["asymm_gen"].fill(dataset=dataset, frac = asymm_gen, weight=weights)
            #### TODO: try to use np.where to add mismatched selections
#             sel.add("toposel", (dphimin_gen < 1.0) & (asymm_gen < 0.3))
            events = events[(dphimin_gen < 1.0) & (asymm_gen < 0.3)]
            weights = weights[(dphimin_gen < 1.0) & (asymm_gen < 0.3)]
            print("Number of matched trijet events after topological selection ", len(events.GenJetAK8), '\n')
            matches = ak.all(events.GenJetAK8.delta_r(events.GenJetAK8.nearest(events.FatJet)) < 0.15, axis = -1)
            misses = ~matches
            out["misses"].fill(dataset=dataset, ptgen = ak.flatten(events[misses].GenJetAK8.pt), 
                                    mgen = ak.flatten(events[misses].GenJetAK8.mass))
            events = events[matches]
            weights = weights[matches]
            print("Gen jet fields: ", events.GenJetAK8.fields)
            print("Number of matched trijet events after matching ", len(events.GenJetAK8), '\n')
            out['cutflow']['misses'] += (len(misses))
#         print("Fields available for FatJets ", trijetEvents.FatJet.fields)
#         print("Fields available for GenJets ", trijetEvents.GenJetAK8.fields)


        #####################################
        #### Reco Jet Selection
        ####################################
    
        #         sel.add("threeRecoJets", ak.num(events.FatJet) >= 3)
        
        pt_cut = (ak.all(events.FatJet.pt > self.ptcut, axis = -1))
        eta_cut = (ak.all(np.abs(events.FatJet.eta) < self.etacut, axis = -1))
        weights = weights[(ak.num(events.FatJet) >= 3) & pt_cut & eta_cut]
        events = events[(ak.num(events.FatJet) >= 3) & pt_cut & eta_cut]
        out["njet_reco"].fill(dataset=dataset, n=ak.num(events.FatJet), weight = weights)
        print("Initial number of reco trijet events ", len(events.FatJet), '\n')
        jet1 = events.FatJet[:, 0]
        jet2 = events.FatJet[:, 1]
        jet3 = events.FatJet[:, 2]

        dphi12 = np.abs(jet1.delta_phi(jet2))
        dphi13 = np.abs(jet1.delta_phi(jet3))
        dphi23 = np.abs(jet2.delta_phi(jet3))
        dphimin = np.amin([dphi12, dphi13, dphi23], axis = 0)
        out["dphimin_reco"].fill(dataset=dataset, dphi = dphimin, weight = weights)
        asymm = np.abs(jet1.pt - jet2.pt)/(jet1.pt + jet2.pt)
        out["asymm_reco"].fill(dataset=dataset, frac = asymm, weight=weights)
        

        events = events[(dphimin > 1.0) & (asymm < 0.3)]
        weights = weights[(dphimin > 1.0) & (asymm < 0.3)]
        #sel.add("recoTopoSel", (dphi_min > 1.0) & (asymm < 0.3))
        print("Number of reco trijet events after topological selection ", len(events.FatJet), '\n')
        #### Apply btag 
        events, btagSel = applyBTag(events, self.btag)
        weights = weights[btagSel]
        print("Number of reco trijet events after btag ", len(events.FatJet), '\n')
        out["jet_pt_mass_reco_u"].fill( dataset=dataset, ptreco=events.FatJet[:,2].pt, mreco=events.FatJet[:,2].mass,
                                       weight=weights )
        out["jet_pt_mass_reco_g"].fill( dataset=dataset, ptreco=events.FatJet[:,2].pt, mreco=events.FatJet[:,2].msoftdrop,
                                       weight=weights )
        
        #### Find fakes and make response matrices
        if self.do_gen:
            #### fakes = reco but no gen
            fakes = ak.any(ak.is_none(events.FatJet.matched_gen, axis = -1), axis = -1)
            out["fakes"].fill(dataset = dataset, ptreco = events[fakes].FatJet[:,2].pt, mreco = events[fakes].FatJet[:,2].mass)
            print("Number of fake jets ", len(fakes), " out of ", len(events), " total events /n")
            matched_reco = ~fakes
            events = events[matched_reco]
            weights = weights[matched_reco]
            print("Number of matched trijet events", len(events))
            #### Get gen soft drop mass
            genjet = events.GenJetAK8[:,2]
            gensubjets = events.SubGenJetAK8
            groomed_genjet = get_gen_sd_mass_jet(genjet, gensubjets)
            out["jet_pt_mass_gen_u"].fill( dataset=dataset, ptgen=genjet.pt, mgen=genjet.mass, weight=weights )
            out["jet_pt_mass_gen_g"].fill( dataset=dataset, ptgen=groomed_genjet.pt, mgen=groomed_genjet.mass, weight=weights )
            #### Get dr between gen and reco jets (should be all < 0.15)
            jet = events.FatJet[:,2]
            out["jet_dr_reco_gen"].fill(dataset=dataset, dr=jet.delta_r(genjet), weight=weights)
            #### Final plots
            out["jet_pt_reco_over_gen"].fill(dataset=dataset, frac=jet.pt/genjet.pt, weight=weights)
            out["jet_m_pt_u_reco_over_gen"].fill(dataset=dataset, ptgen=genjet.pt, mgen = genjet.mass, 
                                                 frac=jet.mass/genjet.mass, weight=weights)
            out["jet_m_pt_g_reco_over_gen"].fill(dataset=dataset, ptgen=groomed_genjet.pt, mgen=groomed_genjet.mass, 
                                                 frac=jet.msoftdrop/groomed_genjet.mass, weight=weights)
            out["response_matrix_u"].fill(dataset=dataset, ptreco=jet.pt, ptgen=genjet.pt, 
                                          mreco=jet.mass, mgen=genjet.mass, weight = weights )
            out["response_matrix_g"].fill(dataset=dataset, ptreco=jet.pt, ptgen=genjet.pt,
                                          mreco=jet.msoftdrop, mgen=groomed_genjet.mass )
            #### Outliers
            weird = (np.abs(jet.msoftdrop - groomed_genjet.mass) > 20.0) & (jet.msoftdrop > 10.)
            print("Number of outliers: , ", len(weird))
            #### Gluon purity plots        
            jet1 = events.FatJet[:,0]
            jet2 = events.FatJet[:,1]
            jet3 = events.FatJet[:,2]

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


            out['jet_mass'].fill(dataset = dataset, jetNumb = "jet1", partonFlav = "Gluon",  mgen = jet1_g.mass,
    #                              weight = trijetEvents.Generator.weight
                                )
            out['jet_mass'].fill(dataset = dataset, jetNumb = "jet1", partonFlav = "UDS",    mgen = jet1_uds.mass,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_mass'].fill(dataset = dataset, jetNumb = "jet1", partonFlav = "Charm",  mgen = jet1_c.mass,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_mass'].fill(dataset = dataset, jetNumb = "jet1", partonFlav = "Bottom", mgen = jet1_b.mass,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_mass'].fill(dataset = dataset, jetNumb = "jet1", partonFlav = "Other",  mgen = jet1_other.mass,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_mass'].fill(dataset = dataset, jetNumb = "jet2", partonFlav = "Gluon",  mgen = jet2_g.mass,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_mass'].fill(dataset = dataset, jetNumb = "jet2", partonFlav = "UDS",    mgen = jet2_uds.mass,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_mass'].fill(dataset = dataset, jetNumb = "jet2", partonFlav = "Charm",  mgen = jet2_c.mass,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_mass'].fill(dataset = dataset, jetNumb = "jet2", partonFlav = "Bottom", mgen = jet2_b.mass,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_mass'].fill(dataset = dataset, jetNumb = "jet2", partonFlav = "Other",  mgen = jet2_other.mass,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_mass'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "Gluon",  mgen = jet3_g.mass,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_mass'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "UDS",    mgen = jet3_uds.mass,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_mass'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "Charm",  mgen = jet3_c.mass,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_mass'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "Bottom", mgen = jet3_b.mass,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_mass'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "Other",  mgen = jet3_other.mass,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_pt'].fill(dataset = dataset, jetNumb = "jet1", partonFlav = "Gluon",  ptgen = jet1_g.pt,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_pt'].fill(dataset = dataset, jetNumb = "jet1", partonFlav = "UDS",    ptgen = jet1_uds.pt,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_pt'].fill(dataset = dataset, jetNumb = "jet1", partonFlav = "Charm",  ptgen = jet1_c.pt,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_pt'].fill(dataset = dataset, jetNumb = "jet1", partonFlav = "Bottom", ptgen = jet1_b.pt,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_pt'].fill(dataset = dataset, jetNumb = "jet1", partonFlav = "Other",  ptgen = jet1_other.pt,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_pt'].fill(dataset = dataset, jetNumb = "jet2", partonFlav = "Gluon",  ptgen = jet2_g.pt,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_pt'].fill(dataset = dataset, jetNumb = "jet2", partonFlav = "UDS",    ptgen = jet2_uds.pt,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_pt'].fill(dataset = dataset, jetNumb = "jet2", partonFlav = "Charm",  ptgen = jet2_c.pt,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_pt'].fill(dataset = dataset, jetNumb = "jet2", partonFlav = "Bottom", ptgen = jet2_b.pt,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_pt'].fill(dataset = dataset, jetNumb = "jet2", partonFlav = "Other",  ptgen = jet2_other.pt,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_pt'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "Gluon",  ptgen = jet3_g.pt,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_pt'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "UDS",    ptgen = jet3_uds.pt,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_pt'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "Charm",  ptgen = jet3_c.pt,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_pt'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "Bottom", ptgen = jet3_b.pt,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_pt'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "Other",  ptgen = jet3_other.pt,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_eta'].fill(dataset = dataset, jetNumb = "jet1", partonFlav = "Gluon",  eta = jet1_g.eta,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_eta'].fill(dataset = dataset, jetNumb = "jet1", partonFlav = "UDS",    eta = jet1_uds.eta,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_eta'].fill(dataset = dataset, jetNumb = "jet1", partonFlav = "Charm",  eta = jet1_c.eta,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_eta'].fill(dataset = dataset, jetNumb = "jet1", partonFlav = "Bottom", eta = jet1_b.eta,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_eta'].fill(dataset = dataset, jetNumb = "jet1", partonFlav = "Other",  eta = jet1_other.eta,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_eta'].fill(dataset = dataset, jetNumb = "jet2", partonFlav = "Gluon",  eta = jet2_g.eta,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_eta'].fill(dataset = dataset, jetNumb = "jet2", partonFlav = "UDS",    eta = jet2_uds.eta,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_eta'].fill(dataset = dataset, jetNumb = "jet2", partonFlav = "Charm",  eta = jet2_c.eta,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_eta'].fill(dataset = dataset, jetNumb = "jet2", partonFlav = "Bottom", eta = jet2_b.eta,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_eta'].fill(dataset = dataset, jetNumb = "jet2", partonFlav = "Other",  eta = jet2_other.eta,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_eta'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "Gluon",  eta = jet3_g.eta,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_eta'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "UDS",    eta = jet3_uds.eta,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_eta'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "Charm",  eta = jet3_c.eta,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_eta'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "Bottom", eta = jet3_b.eta,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_eta'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "Other",  eta = jet3_other.eta,
                                 #weight = trijetEvents.Generator.weight
                                )   
            out['btag'].fill(dataset = dataset, jetNumb = "jet1", partonFlav = "Gluon",  frac = jet1_g.btagCSVV2,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['btag'].fill(dataset = dataset, jetNumb = "jet1", partonFlav = "UDS",    frac = jet1_uds.btagCSVV2,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['btag'].fill(dataset = dataset, jetNumb = "jet1", partonFlav = "Charm",  frac = jet1_c.btagCSVV2,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['btag'].fill(dataset = dataset, jetNumb = "jet1", partonFlav = "Bottom", frac = jet1_b.btagCSVV2,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['btag'].fill(dataset = dataset, jetNumb = "jet1", partonFlav = "Other",  frac = jet1_other.btagCSVV2,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['btag'].fill(dataset = dataset, jetNumb = "jet2", partonFlav = "Gluon",  frac = jet2_g.btagCSVV2,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['btag'].fill(dataset = dataset, jetNumb = "jet2", partonFlav = "UDS",    frac = jet2_uds.btagCSVV2,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['btag'].fill(dataset = dataset, jetNumb = "jet2", partonFlav = "Charm",  frac = jet2_c.btagCSVV2,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['btag'].fill(dataset = dataset, jetNumb = "jet2", partonFlav = "Bottom", frac = jet2_b.btagCSVV2,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['btag'].fill(dataset = dataset, jetNumb = "jet2", partonFlav = "Other",  frac = jet2_other.btagCSVV2,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['btag'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "Gluon",  frac = jet3_g.btagCSVV2,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['btag'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "UDS",    frac = jet3_uds.btagCSVV2,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['btag'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "Charm",  frac = jet3_c.btagCSVV2,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['btag'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "Bottom", frac = jet3_b.btagCSVV2,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['btag'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "Other",  frac = jet3_other.btagCSVV2,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['cutflow']['nGluonJets'] += (len(jet3_g))
            print("Number of jet3's", len(jet3), " and number of gluon jet 3's ", len(jet3_g))
            out['cutflow']['n3Jets'] += (len(jet3.pt))
            print(out['cutflow']['n3Jets'])
        out['cutflow']['chunks'] += 1
        return out
    
    def postprocess(self, accumulator):
        return accumulator
    
def main():
    #### Next run processor with futures executor on all test files
    from dask.distributed import Client
    from plugins import runCoffeaJob
    processor = makeDijetHists()
    result = runCoffeaJob(processor, jsonFile = "QCD_flat_files.json", winterfell = True, testing = True, data = False)
    util.save(result, "coffeaOutput/dijet_pT" + str(processor.ptcut) + "_eta" + str(processor.etacut) + "_result_test.coffea")
    

if __name__ == '__main__':
    # Execute when the module is not initialized from an import statement: i.e. called from terminal command line
    main()
