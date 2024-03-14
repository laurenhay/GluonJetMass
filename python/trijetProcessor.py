#### This file contains the processors for dijet and trijet hist selections. Plotting and resulting studies are in separate files.
#### LMH

#### import outer dependencies
import argparse
import awkward as ak
import numpy as np
import coffea
import os
import re
import pandas as pd
import hist
print(hist.__version__)
print(coffea.__version__)
from coffea import util, processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import PackedSelection
from collections import defaultdict
#### import our python packages
from python.corrections import *
from python.utils import *

parser = argparse.ArgumentParser()

parser.add_argument("year")
parser.add_argument("data")


#####
##### TO DO #####
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


#bTag_options = ['bbloose', 'bloose', 'bbmed', 'bmed']
def applyBTag(events, btag):
    print('btag input: ', btag, '\n')
    if (btag == 'bbloose'):
        sel = (events.FatJet[:,0].btagDeepB >= 0.2027) & (events.FatJet[:,1].btagDeepB >= 0.2027)
        events = events[sel]
        print('Loose WP CSV V2 B tag applied to leading two jets')
    elif (btag == 'bloose'):
        sel = (events.FatJet[:,0].btagDeepB >= 0.2027)
        events = events[sel]
        print('Loose WP CSV V2 B tag applied to leading jet only')
    elif (btag == 'bbmed'):
        sel = (events.FatJet[:,0].btagDeepB >= 0.6001) & (events.FatJet[:,1].btagDeepB >= 0.6001)
        events = events[sel]
        print('Medium WP CSV V2 B tag applied to first two jets')
    elif (btag == 'bmed'):
        sel = (events.FatJet[:,0].btagDeepB >= 0.6001)
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
    def __init__(self, ptcut = 200., ycut = 2.5, btag = 'None', data = 'False', systematics = ['nominal', 'jer', 'jes']):
        self.ptcut = ptcut
        self.ycut = ycut
        self.btag = btag
        self.do_gen = not data
        self.systematics = systematics
        print("Data: ", data, " gen ", self.do_gen)
        dataset_cat = hist.axis.StrCategory([],growth=True,name="dataset", label="Dataset")
        jet_cat = hist.axis.StrCategory([], growth=True, name="jetNumb", label="Jet")
        parton_cat = hist.axis.StrCategory([],growth=True,name="partonFlav", label="Parton Flavour")
        syst_cat = hist.axis.StrCategory([], growth=True, name='syst', label="Systematic")
        #### if using specific bin edges use hist.axis.Variable() instead
        mgen_bin_edges = np.array([0,1,5,10,20,40,60,80,100,150,200,250,1000])
        mreco_bin_edges = np.sort(np.append(mgen_bin_edges,[(mgen_bin_edges[i]+mgen_bin_edges[i+1])/2 for i in range(len(mgen_bin_edges)-1)]))
        print("mreco bins: ", mreco_bin_edges)
        mass_gen_bin =  hist.axis.Variable(mgen_bin_edges, name="mgen", label=r"m_{GEN} (GeV)")                         
        mass_bin = hist.axis.Variable(mreco_bin_edges, name="mreco", label=r"m_{RECO} (GeV)")
        pt_bin = hist.axis.Variable([200,280,360,450,520,630,690,750,800,1300,13000], name="ptreco", label=r"p_{T,RECO} (GeV)")   
        pt_gen_bin = hist.axis.Variable([200,280,360,450,520,630,690,750,800,1300,13000], name="ptgen", label=r"p_{T,GEN} (GeV)") 
#         mass_bin = hist.axis.Regular(60, 0, 1000.,name="mreco", label="Jet Mass (GeV)")
#         mass_gen_bin = hist.axis.Regular(60, 0, 1000., name="mgen", label="Gen Jet Mass (GeV)")
#         pt_bin = hist.axis.Regular(60, 0, 3000., name="ptreco",label= "Jet pT (GeV)")
#         pt_gen_bin = hist.axis.Regular(60, 0, 3000., name="ptgen",label= "Gen Jet pT (GeV)")
        y_bin = hist.axis.Regular(25, 0., 2.5, name="rapidity", label=r"$y$")
        eta_bin = hist.axis.Regular(25, 0., 2.5, name="eta", label=r"$\eta$")
        frac_axis = hist.axis.Regular(10, 0.0, 1., name="frac", label="Fraction")
        n_axis = hist.axis.Regular(5, 0, 5, name="n", label=r"Number")
        dr_axis = hist.axis.Regular(150, 0, 6.0, name="dr", label=r"$\Delta R$")
        dphi_axis = hist.axis.Regular(150, -2*np.pi, 2*np.pi, name="dphi", label=r"$\Delta \phi$")

        self._histos = {
        #### btag study histos
        'alljet_ptreco_mreco':        hist.Hist(dataset_cat, jet_cat, parton_cat, mass_bin, pt_bin, storage="weight", name="Events"),
        'btag_eta':            hist.Hist(dataset_cat, jet_cat, parton_cat, frac_axis, eta_bin, storage="weight", name="Events"),
            
        #### Plots of things during the selection process / for debugging
        'njet_gen':                  hist.Hist(dataset_cat, syst_cat, n_axis, storage="weight", label="Events"),
        'dphimin_gen':               hist.Hist(dataset_cat, syst_cat, dphi_axis, storage="weight", label="Events"),
        'asymm_gen':               hist.Hist(dataset_cat, syst_cat, frac_axis, storage="weight", label="Events"),
        'njet_reco':                  hist.Hist(dataset_cat, syst_cat, n_axis, storage="weight", label="Events"),
        'dphimin_reco':               hist.Hist(dataset_cat, syst_cat, dphi_axis, storage="weight", label="Events"),
        'asymm_reco':               hist.Hist(dataset_cat, syst_cat, frac_axis, storage="weight", label="Events"),
            
        'jet_dr_reco_gen':           hist.Hist(dataset_cat, syst_cat, dr_axis, storage="weight", label="Events"),
        'jet_mass_reco_over_gen':    hist.Hist(dataset_cat, syst_cat, frac_axis, storage="weight", label="Events"),
        'jet_pt_reco':               hist.Hist(dataset_cat, syst_cat, pt_bin, storage="weight", name="Events"),
        'jet_pt_gen':                hist.Hist(dataset_cat, syst_cat, pt_bin, storage="weight", name="Events"),
        'jet_pt_reco_over_gen':      hist.Hist(dataset_cat, syst_cat, frac_axis, storage="weight", label="Events"),
        'jet_eta_reco':              hist.Hist(dataset_cat, syst_cat, eta_bin, storage="weight", name="Events"),
        'jet_eta_gen':               hist.Hist(dataset_cat, syst_cat, eta_bin, storage="weight",name="Events"),
        'jet_rap_reco':              hist.Hist(dataset_cat, syst_cat, y_bin, storage="weight", name="Events"),
        'jet_rap_gen':               hist.Hist(dataset_cat, syst_cat, y_bin, storage="weight",name="Events"),
        #'jet_dr_gen':                hist.Hist(dataset_cat, dr_axis, storage="weight", label="Counts"),
        #'jet_dr_reco':               hist.Hist(dataset_cat, dr_axis, storage="weight", label="Counts"),
        'jet_dphi_reco':             hist.Hist(dataset_cat, syst_cat, dphi_axis, storage="weight", label="Events"),
        
        'jet_ptasymm_reco':          hist.Hist(dataset_cat, syst_cat, frac_axis, storage="weight", label="Events"),
        'jet_dr_gen_subjet':         hist.Hist(dataset_cat, syst_cat, dr_axis, storage="weight", label="Events"),
        'jet_dr_reco_to_gen_subjet': hist.Hist(dataset_cat, syst_cat, dr_axis, storage="weight", label="Events"),
        'jet_sd_mass_reco':          hist.Hist(dataset_cat, syst_cat, jet_cat, parton_cat, mass_bin, storage="weight", name="Events"),
        'jet_sd_mass_gen':           hist.Hist(dataset_cat, syst_cat, jet_cat, parton_cat, mass_bin, storage="weight", name="Events"),
        'misses_g':       hist.Hist(dataset_cat, syst_cat, pt_gen_bin, mass_gen_bin, storage="weight", name="Events"),
        'fakes_g':        hist.Hist(dataset_cat, syst_cat, pt_bin, mass_bin, storage="weight", name="Events"),
        'misses':       hist.Hist(dataset_cat, syst_cat, pt_gen_bin, mass_gen_bin, storage="weight", name="Events"),
        'fakes':        hist.Hist(dataset_cat, syst_cat, pt_bin, mass_bin, storage="weight", name="Events"),
         
            
        #### Plots to be unfolded
        'jet_pt_mass_reco_u':       hist.Hist(dataset_cat, syst_cat, pt_bin, mass_bin, storage="weight", name="Events"),
        'jet_pt_mass_reco_g':        hist.Hist(dataset_cat, syst_cat, pt_bin, mass_bin, storage="weight", name="Events"),

        #### Plots for comparison
        'jet_pt_mass_gen_u':        hist.Hist(dataset_cat, syst_cat, pt_gen_bin, mass_gen_bin, storage="weight", label="Events"),       
        'jet_pt_mass_gen_g':            hist.Hist(dataset_cat, syst_cat, pt_gen_bin, mass_gen_bin, storage="weight", label="Events"),
        
        #### Plots to get JMR and JMS in MC
        'jet_m_pt_u_reco_over_gen': hist.Hist(dataset_cat, syst_cat, pt_gen_bin, mass_gen_bin, frac_axis, storage="weight",                                                                        label="Events"),
        'jet_m_pt_g_reco_over_gen':  hist.Hist(dataset_cat, syst_cat, pt_gen_bin, mass_gen_bin, frac_axis, storage="weight",                                                                        label="Events"),

        #### Plots for the analysis in the proper binning
        'response_matrix_u':    hist.Hist(dataset_cat, syst_cat, pt_bin, mass_bin, pt_gen_bin, mass_gen_bin, storage="weight",                                                         label="Events"),
        'response_matrix_g':     hist.Hist(dataset_cat, syst_cat, pt_bin, mass_bin, pt_gen_bin, mass_gen_bin, storage="weight",                                                         label="Events"),
        # accumulators
        'cutflow': processor.defaultdict_accumulator(int),
        'weights': processor.defaultdict_accumulator(float),
        'systematics': processor.defaultdict_accumulator(float),
        }
    
    @property
    def accumulator(self):
        return self._histos
    def process(self, events):
        out = self._histos
        dataset = events.metadata['dataset']
        filename = events.metadata['filename']
        print(dataset)
        out['cutflow']['nEvents initial'] += (len(events.FatJet))
        #####################################
        #### Find the IOV from the dataset name
        #####################################
        IOV = ('2016APV' if ( any(re.findall(r'HIPM',  dataset)))
               else '2018'    if ( any(re.findall(r'UL18', dataset)) or any(re.findall(r'UL2018',    dataset)))
               else '2017'    if ( any(re.findall(r'UL17', dataset)) or any(re.findall(r'UL2017',    dataset)))
               else '2016')
        out['cutflow']['nEvents'+IOV+dataset] += (len(events.FatJet))
        #####################################
        #### Find the era from the file name
        #### Apply the good lumi mask
        #####################################
        FatJet=events.FatJet
        FatJet["p4"] = ak.with_name(events.FatJet[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
        if self.do_gen:
            era = None
            GenJetAK8 = events.GenJetAK8
            GenJetAK8['p4']= ak.with_name(events.GenJetAK8[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
            FatJet["pt_gen"] = ak.values_astype(ak.fill_none(FatJet.p4.nearest(GenJetAK8.p4, threshold=0.2).pt,
                                                             0), np.float32)
        else:
            firstidx = filename.find( "store/data/" )
            fname2 = filename[firstidx:]
            fname_toks = fname2.split("/")
            era = fname_toks[ fname_toks.index("data") + 1]
            print("IOV ", IOV, ", era ", era)
        corrected_fatjets = GetJetCorrections(FatJet, events, era, IOV, isData=not self.do_gen)
        # corrected_fatjets = FatJet
        corrections = {"nominal": corrected_fatjets}
        if (len(corrected_fatjets.pt[0]) > 1) and 'jes' in self.systematics and self.do_gen:
            print('JEC:%s:JES up, nom, down:%s:%s:%s',
                         corrected_fatjets.JES_jes.up.pt[0][0],
                         corrected_fatjets.pt[0][0],
                         corrected_fatjets.JES_jes.down.pt[0][0])
            print("JES up vals: ", corrected_fatjets.JES_jes.up)
        if 'jes' in self.systematics and self.do_gen:
            corrections.update({
                       "jesUp": corrected_fatjets.JES_jes.up,
                       "jesDown": corrected_fatjets.JES_jes.down
                      })

        if 'jer' in self.systematics and self.do_gen:
            corrections.update({"jerUp": corrected_fatjets.JER.up,
                                "jerDown": corrected_fatjets.JER.down
                            })
        for syst in corrections.keys():
            print("Length of corrected: ", len(corrections[syst]), " length fo events: ", len(events))
            print("Adding ", syst, " values ", corrections[syst], " to output")
            events_corr = ak.with_field(events, corrections[syst], "FatJet")
            weights = np.ones(len(events_corr))
            print("Length of events before any cuts", len(events_corr), "length of weights ", len(weights))
            if (self.do_gen):
                era = None
                print("XS weight: ", getXSweight(dataset, IOV))
                weights = weights * getXSweight(dataset, IOV)
                    
            else:
                # apply lumimask and require at least one jet to apply jet trigger prescales
                print("Event runs: ", events.run)
                lumi_mask = getLumiMask(IOV)(events_corr.run, events_corr.luminosityBlock)
                events_corr = events_corr [lumi_mask & (ak.num(events_corr.FatJet) >= 1)]
                trigsel, psweights = applyPrescales(events_corr, year = IOV)
                weights=psweights
                events_corr = events_corr[trigsel]
                weights = weights[trigsel]
                out['cutflow']['nEvents after trigger sel '+syst] += (len(events_corr.FatJet))
            
            #### Need to add PU reweighting for if do_gen
            #### Remove event with very large gen weights???
            # print("NPVs ",events.PV.fields)
            # sel.add("npv", events.PV.npvsGood>0)
    
            #####################################
            #### Gen Jet Selection
            ####################################       
            if self.do_gen:
                # sel.add("threeGenJets", ak.num(events.GenJetAK8) >= 3)
                pt_cut_gen = ak.all(events_corr.GenJetAK8.pt > 140., axis = -1) ### 70% of reco pt cut
                GenJetAK8 = events_corr.GenJetAK8
                GenJetAK8['p4']= ak.with_name(events_corr.GenJetAK8[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
                rap_cut_gen = ak.all(np.abs(getRapidity(GenJetAK8.p4)) < self.ycut, axis = -1)
                # sel.add("ptEtaCutGen", (pt_cut_gen & eta_cut_gen))
                # kinesel = sel.all("threeGenJets", "ptEtaCutGen")
                kinesel = (pt_cut_gen & rap_cut_gen & (ak.num(events_corr.GenJetAK8) >2))
                out["njet_gen"].fill(dataset=dataset, syst=syst, n=ak.num(events_corr.GenJetAK8[rap_cut_gen & pt_cut_gen]), 
                                     weight = weights[rap_cut_gen & pt_cut_gen] )
                                 
                #### Get leading 3 jets
                events_corr = events_corr[kinesel]
                weights = weights[kinesel]
                out['cutflow']['nEvents after >2 jet, rapidity, and pT selection '+syst] += (len(events_corr.FatJet))
                genjet1 = events_corr.GenJetAK8[:,0]
                genjet2 = events_corr.GenJetAK8[:,1]
                genjet3 = events_corr.GenJetAK8[:,2]
           
                ### calculate dphi_min
                dphi12_gen = np.abs(genjet1.delta_phi(genjet2))
                dphi13_gen = np.abs(genjet1.delta_phi(genjet3))
                dphi23_gen = np.abs(genjet2.delta_phi(genjet3))
                dphimin_gen = np.amin([dphi12_gen, dphi13_gen, dphi23_gen], axis = 0)
                out["dphimin_gen"].fill(dataset=dataset, syst=syst, dphi = dphimin_gen, weight = weights)
                asymm_gen  = np.abs(genjet1.pt - genjet2.pt)/(genjet1.pt + genjet2.pt)
                out["asymm_gen"].fill(dataset=dataset, syst=syst, frac = asymm_gen, weight=weights)
                events_corr = events_corr[(dphimin_gen < 1.0) & (asymm_gen < 0.3)]
                weights = weights[(dphimin_gen < 1.0) & (asymm_gen < 0.3)]
                out['cutflow']['nEvents after gen dphi and ptasymm selection '+syst] += (len(events_corr.FatJet))
                gensubjets = events_corr.SubGenJetAK8
                groomed_genjet = get_gen_sd_mass_jet(events_corr.GenJetAK8, gensubjets)
                matches = ak.all(events_corr.GenJetAK8.delta_r(events_corr.GenJetAK8.nearest(events_corr.FatJet)) < 0.2, axis = -1)
                # matches_g = ak.all(groomed_genjet.delta_r(groomed_genjet.nearest(events.FatJet)) < 0.15, axis = -1)
                misses = ~matches
                # gen but no reco
                out["misses"].fill(dataset=dataset, syst=syst, ptgen = events_corr[misses].GenJetAK8[:,2].pt, 
                                        mgen = events_corr[misses].GenJetAK8[:,2].mass)
                # out["misses_g"].fill(dataset=dataset, ptgen = ak.flatten(events[misses_g].GenJetAK8[:,2].pt), 
                #                         mgen = ak.flatten(groomed_genjet[misses_g][:,2].mass))
                events_corr = events_corr[matches]
                weights = weights[matches]
                print("Gen jet fields: ", events_corr.GenJetAK8.fields)
                out['cutflow']['nEvents after removing misses '+syst] += (len(events_corr.FatJet))
                out['cutflow']['misses'] += (len(events_corr[misses].FatJet))
    #         print("Fields available for FatJets ", trijetEvents.FatJet.fields)
    #         print("Fields available for GenJets ", trijetEvents.GenJetAK8.fields)
    
    
            #####################################
            #### Reco Jet Selection
            ####################################
        
            #         sel.add("threeRecoJets", ak.num(events.FatJet) >= 3)
            
            pt_cut = (ak.all(events_corr.FatJet.pt > self.ptcut, axis = -1))
            FatJet = events_corr.FatJet
            FatJet["p4"] = ak.with_name(events_corr.FatJet[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
            rap_cut = ak.all(np.abs(getRapidity(FatJet.p4)) < self.ycut, axis = -1)
            weights = weights[(ak.num(events_corr.FatJet) >= 3) & pt_cut & rap_cut]
            events_corr = events_corr[(ak.num(events_corr.FatJet) >= 3) & pt_cut & rap_cut]
            FatJet = events_corr.FatJet
            FatJet["p4"] = ak.with_name(events_corr.FatJet[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
            out["njet_reco"].fill(dataset=dataset, syst=syst, n=ak.num(events_corr.FatJet), weight = weights)
            out["jet_eta_reco"].fill(dataset=dataset, syst=syst, eta = events_corr.FatJet[:,2].eta, weight=weights)
            out["jet_rap_reco"].fill(dataset=dataset, syst=syst, rapidity=np.abs(getRapidity(FatJet[:,2].p4)), weight = weights)
            out['cutflow']['nEvents after reco kine selection '+syst] += (len(events_corr.FatJet))
            jet1 = events_corr.FatJet[:, 0]
            jet2 = events_corr.FatJet[:, 1]
            jet3 = events_corr.FatJet[:, 2]
            dphi12 = np.abs(jet1.delta_phi(jet2))
            dphi13 = np.abs(jet1.delta_phi(jet3))
            dphi23 = np.abs(jet2.delta_phi(jet3))
            dphimin = np.amin([dphi12, dphi13, dphi23], axis = 0)
            out["dphimin_reco"].fill(dataset=dataset, syst=syst, dphi = dphimin, weight = weights)
            asymm = np.abs(jet1.pt - jet2.pt)/(jet1.pt + jet2.pt)
            out["asymm_reco"].fill(dataset=dataset, syst=syst, frac = asymm, weight=weights)
            events_corr = events_corr[(dphimin > 1.0) & (asymm < 0.3)]
            weights = weights[(dphimin > 1.0) & (asymm < 0.3)]
            out['cutflow']['nEvents after reco topo selection '+syst] += (len(events_corr.FatJet))
            #### Apply btag 
            events_corr, btagSel = applyBTag(events_corr, self.btag)
            weights = weights[btagSel]
            out['cutflow']['nEvents after reco btag '+syst] += (len(events_corr.FatJet))
            
            #### Find fakes and make response matrices
            if self.do_gen:
                #### fakes = reco but no gen
                fakes = ak.any(ak.is_none(events_corr.FatJet.matched_gen, axis = -1), axis = -1)
                out["fakes"].fill(dataset = dataset, syst=syst, ptreco = events_corr[fakes].FatJet[:,2].pt, mreco = events_corr[fakes].FatJet[:,2].mass)
                out['cutflow']['fakes '+syst] += (events_corr[fakes].FatJet)
                matched_reco = ~fakes
                events_corr = events_corr[matched_reco]
                weights = weights[matched_reco]
                out['cutflow']['nEvents after removing fakes '+syst] += (len(events_corr.FatJet))
                #### Get gen soft drop mass
                genjet = events_corr.GenJetAK8[:,2]
                gensubjets = events_corr.SubGenJetAK8
                groomed_genjet = get_gen_sd_mass_jet(genjet, gensubjets)
                out["jet_pt_mass_gen_u"].fill( dataset=dataset, syst=syst, ptgen=genjet.pt, mgen=genjet.mass, weight=weights )
                out["jet_pt_mass_gen_g"].fill( dataset=dataset, syst=syst, ptgen=groomed_genjet.pt, mgen=groomed_genjet.mass, weight=weights )
                #### Get dr between gen and reco jets (should be all < 0.15)
                jet = events_corr.FatJet[:,2]
                out["jet_dr_reco_gen"].fill(dataset=dataset, syst=syst, dr=jet.delta_r(genjet), weight=weights)
                #### Final plots
                out["jet_pt_reco_over_gen"].fill(dataset=dataset, syst=syst, frac=jet.pt/genjet.pt, weight=weights)
                out["jet_m_pt_u_reco_over_gen"].fill(dataset=dataset, syst=syst, ptgen=genjet.pt, mgen = genjet.mass, 
                                                     frac=jet.mass/genjet.mass, weight=weights)
                out["jet_m_pt_g_reco_over_gen"].fill(dataset=dataset, syst=syst, ptgen=groomed_genjet.pt,
                                                     mgen=groomed_genjet.mass, frac=jet.msoftdrop/groomed_genjet.mass,
                                                     weight=weights)
                out["response_matrix_u"].fill(dataset=dataset, syst=syst, ptreco=jet.pt, ptgen=genjet.pt, mreco=jet.mass, mgen=genjet.mass, weight = weights )
                out["response_matrix_g"].fill(dataset=dataset, syst=syst, ptreco=jet.pt, ptgen=genjet.pt, mreco=jet.msoftdrop, mgen=groomed_genjet.mass, weight = weights )
                #### Outliers
                weird = (np.abs(jet.msoftdrop - groomed_genjet.mass) > 20.0) & (jet.msoftdrop > 10.)
                out['cutflow']['Number of outliers '+syst] += (len(events_corr[weird].FatJet))
                #### Gluon purity plots        
                jet1 = events_corr.FatJet[:,0]
                jet2 = events_corr.FatJet[:,1]
                jet3 = events_corr.FatJet[:,2]
    
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

                jet3_jetbb = jet3[(np.abs(genjet1.partonFlavour) == 5) & (np.abs(genjet2.partonFlavour) == 5)]
                jet3_jetb = jet3[(np.abs(genjet1.partonFlavour) == 5)]
                jet3_jetbb_g = jet3[(np.abs(genjet1.partonFlavour) == 5) & (np.abs(genjet2.partonFlavour) == 5) & (np.abs(genjet3.partonFlavour) == 21)]
                jet3_jetb_g = jet3[(np.abs(genjet1.partonFlavour) == 5) & (np.abs(genjet3.partonFlavour) == 21)]
                
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
    
    
                out['alljet_ptreco_mreco'].fill(dataset = dataset, jetNumb = "jet1", partonFlav = "Gluon",  mreco = jet1_g.mass, ptreco = jet1_g.pt,
                                    )
                out['alljet_ptreco_mreco'].fill(dataset = dataset, jetNumb = "jet1", partonFlav = "UDS",    mreco = jet1_uds.mass, ptreco = jet1_uds.pt,
                                    )
                out['alljet_ptreco_mreco'].fill(dataset = dataset, jetNumb = "jet1", partonFlav = "Charm",  mreco = jet1_c.mass, ptreco = jet1_c.pt,
                                    )
                out['alljet_ptreco_mreco'].fill(dataset = dataset, jetNumb = "jet1", partonFlav = "Bottom", mreco = jet1_b.mass, ptreco = jet1_b.pt,
                                    )
                out['alljet_ptreco_mreco'].fill(dataset = dataset, jetNumb = "jet1", partonFlav = "Other",  mreco = jet1_other.mass, ptreco = jet1_other.pt,
                                    )
                out['alljet_ptreco_mreco'].fill(dataset = dataset, jetNumb = "jet2", partonFlav = "Gluon",  mreco = jet2_g.mass, ptreco = jet2_g.pt,
                                    )
                out['alljet_ptreco_mreco'].fill(dataset = dataset, jetNumb = "jet2", partonFlav = "UDS",    mreco = jet2_uds.mass, ptreco = jet2_uds.pt,
                                    )
                out['alljet_ptreco_mreco'].fill(dataset = dataset, jetNumb = "jet2", partonFlav = "Charm",  mreco = jet2_c.mass, ptreco = jet2_c.pt,
                                    )
                out['alljet_ptreco_mreco'].fill(dataset = dataset, jetNumb = "jet2", partonFlav = "Bottom", mreco = jet2_b.mass, ptreco = jet2_b.pt,
                                    )
                out['alljet_ptreco_mreco'].fill(dataset = dataset, jetNumb = "jet2", partonFlav = "Other",  mreco = jet2_other.mass, ptreco = jet2_other.pt,
                                    )
                out['alljet_ptreco_mreco'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "Gluon",  mreco = jet3_g.mass, ptreco = jet3_g.pt,
                                    )
                out['alljet_ptreco_mreco'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "UDS",    mreco = jet3_uds.mass, ptreco = jet3_uds.pt,
                                    )
                out['alljet_ptreco_mreco'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "Charm",  mreco = jet3_c.mass, ptreco = jet3_c.pt,
                                    )
                out['alljet_ptreco_mreco'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "Bottom", mreco = jet3_b.mass, ptreco = jet3_b.pt,
                                    )
                out['alljet_ptreco_mreco'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "Other",  mreco = jet3_other.mass, ptreco = jet3_other.pt,
                                    )
                out['alljet_ptreco_mreco'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "Gluon_b",    mreco = jet3_jetb_g.mass, ptreco = jet3_jetb_g.pt,
                                    )
                out['alljet_ptreco_mreco'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "Gluon_bb",  mreco = jet3_jetbb_g.mass, ptreco = jet3_jetbb_g.pt,
                                    )
                out['alljet_ptreco_mreco'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "All_b", mreco = jet3_jetb.mass, ptreco = jet3_jetb.pt,
                                    )
                out['alljet_ptreco_mreco'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "All_bb",  mreco = jet3_jetbb.mass, ptreco = jet3_jetbb.pt,
                                    )
                out['btag_eta'].fill(dataset = dataset, jetNumb = "jet1", partonFlav = "Gluon",  frac = jet1_g.btagDeepB, eta = jet1_g.eta,
                                    )
                out['btag_eta'].fill(dataset = dataset, jetNumb = "jet1", partonFlav = "UDS",    frac = jet1_uds.btagDeepB, eta = jet1_uds.eta,
                                    )
                out['btag_eta'].fill(dataset = dataset, jetNumb = "jet1", partonFlav = "Charm",  frac = jet1_c.btagDeepB,eta = jet1_c.eta,
                                    )
                out['btag_eta'].fill(dataset = dataset, jetNumb = "jet1", partonFlav = "Bottom", frac = jet1_b.btagDeepB, eta = jet1_b.eta,
                                    )
                out['btag_eta'].fill(dataset = dataset, jetNumb = "jet1", partonFlav = "Other",  frac = jet1_other.btagDeepB, eta = jet1_other.eta,
                                    )
                out['btag_eta'].fill(dataset = dataset, jetNumb = "jet2", partonFlav = "Gluon",  frac = jet2_g.btagDeepB, eta = jet2_g.eta,
                                    )
                out['btag_eta'].fill(dataset = dataset, jetNumb = "jet2", partonFlav = "UDS",    frac = jet2_uds.btagDeepB,eta = jet2_uds.eta,
                                    )
                out['btag_eta'].fill(dataset = dataset, jetNumb = "jet2", partonFlav = "Charm",  frac = jet2_c.btagDeepB,eta = jet2_c.eta,
                                    )
                out['btag_eta'].fill(dataset = dataset, jetNumb = "jet2", partonFlav = "Bottom", frac = jet2_b.btagDeepB,eta = jet2_b.eta,
                                    )
                out['btag_eta'].fill(dataset = dataset, jetNumb = "jet2", partonFlav = "Other",  frac = jet2_other.btagDeepB,eta = jet2_other.eta,
                                    )
                out['btag_eta'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "Gluon",  frac = jet3_g.btagDeepB,eta = jet3_g.eta,
                                    )
                out['btag_eta'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "UDS",    frac = jet3_uds.btagDeepB,eta = jet3_uds.eta,
                                    )
                out['btag_eta'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "Charm",  frac = jet3_c.btagDeepB,eta = jet3_c.eta,
                                    )
                out['btag_eta'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "Bottom", frac = jet3_b.btagDeepB,eta = jet3_b.eta,
                                    )
                out['btag_eta'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "Other", frac = jet3_other.btagDeepB,eta = jet3_other.eta,
                                    )
                out['btag_eta'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "Gluon_b", frac = jet3_jetb_g.btagDeepB, eta = jet3_jetb_g.eta,
                                    )
                out['btag_eta'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "Gluon_bb",  frac = jet3_jetbb_g.btagDeepB, eta = jet3_jetbb_g.eta,
                                    )
                out['btag_eta'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "All_b", frac = jet3_jetb.btagDeepB, eta = jet3_jetb.eta,
                                    )
                out['btag_eta'].fill(dataset = dataset, jetNumb = "jet3", partonFlav = "All_bb",  frac = jet3_jetbb.btagDeepB, eta = jet3_jetbb.eta,
                                    )
                out['cutflow']['nGluonJets'] += (len(jet3_g)+len(jet1_g)+len(jet2_g))
                out['cutflow']['nJets'] += (len(jet3)+len(jet1)+len(jet2))
                out['cutflow']['nSoftestGluonJets'] += (len(jet3_g))
                out['cutflow']['nSoftestGluonJets_b'] += (len(jet3_jetb_g))
                out['cutflow']['nSoftestGluonJets_bb'] += (len(jet3_jetbb_g))
                out['cutflow']['nSoftestJets_b'] += (len(jet3_jetb))
                out['cutflow']['nSoftestJets_bb'] += (len(jet3_jetbb))
                out['cutflow']['n3Jets'] += (len(jet3.pt))
            out["jet_pt_mass_reco_u"].fill( dataset=dataset, syst=syst, ptreco=events_corr.FatJet[:,2].pt, mreco=events_corr.FatJet[:,2].mass,
                                           weight=weights )
            out["jet_pt_mass_reco_g"].fill( dataset=dataset, syst=syst, ptreco=events_corr.FatJet[:,2].pt, mreco=events_corr.FatJet[:,2].msoftdrop,
                                           weight=weights )
            out['cutflow']['nEvents final selection'] += (len(events_corr.FatJet))
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

