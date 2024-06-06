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
from coffea.analysis_tools import Weights
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
    print("# of genjet and subjet combinations: ", len(combs))
    dr_jet_subjets = combs['0'].delta_r(combs['1'])
    combs = combs[dr_jet_subjets < 0.8]
    total = combs['1'].sum(axis=1)
    return total 

def get_dphi( coll0, coll1 ):
    '''
    Find dphi between 3rd jet and , returning none when the event does not have at least two jets
    '''
    combs = ak.cartesian( (coll0, coll1), axis=1 )
    # print("Coll. 0: ", len(coll0), '\n', combs['0'])
    # print("Coll. 1: ", len(coll1), '\n', combs['1'])
    dphi = np.abs(combs['0'].delta_phi(combs['1']))
    return ak.firsts((combs['1'])).ak.firsts(dphi)

def getJetFlavors(jet):
    genjet = jet.matched_gen
    jetflavs = {}
    jetflavs["Gluon"] = jet[np.abs(genjet.partonFlavour) == 21]
    jetflavs["UDS"]    = jet[np.abs(genjet.partonFlavour) < 4]
    jetflavs["Charm"]      = jet[np.abs(genjet.partonFlavour) == 4]
    jetflavs["Bottom"]      = jet[np.abs(genjet.partonFlavour) == 5]
    jetflavs["Other"]  = jet[(np.abs(genjet.partonFlavour) > 5) & (np.abs(genjet.partonFlavour) != 21)]
    return jetflavs
#bTag_options = ['bbloose', 'bloose', 'bbmed', 'bmed']
def applyBTag(events, btag):
    # print('btag input: ', btag, '\n')
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
    def __init__(self, ptcut = 200., ycut = 2.5, btag = 'None', data = False, jet_systematics = ['nominal', 'jer', 'jes'], systematics = ['L1PreFiringWeight', 'PUSF'], hem=False):
        self.ptcut = ptcut
        self.ycut = ycut
        self.btag = btag
        self.do_gen = not data
        self.hem = hem
        self.systematics = systematics
        self.jet_systematics = jet_systematics
        print("Data: ", data, " gen ", self.do_gen)
        jet_cat = hist.axis.StrCategory([], growth=True, name="jetNumb", label="Jet")
        parton_cat = hist.axis.StrCategory([],growth=True,name="partonFlav", label="Parton Flavour")
        syst_cat = hist.axis.StrCategory([], growth=True, name='syst', label="Systematic")
        #### if using specific bin edges use hist.axis.Variable() instead
        mgen_bin_edges = np.array([0,1,5,10,20,40,60,80,100,150,200,250,1000])
        mreco_bin_edges = np.sort(np.append(mgen_bin_edges,[(mgen_bin_edges[i]+mgen_bin_edges[i+1])/2 for i in range(len(mgen_bin_edges)-1)]))
        print("mreco bins: ", mreco_bin_edges)
        mass_gen_bin =  hist.axis.Variable(mgen_bin_edges, name="mgen", label=r"m_{GEN} (GeV)")                         
        mass_bin = hist.axis.Variable(mreco_bin_edges, name="mreco", label=r"m_{RECO} (GeV)")
        ptgen_edges = np.array([200,280,360,450,520,630,690,750,800,1300,13000])
        # ptreco_edges = np.sort(np.append(ptgen_edges,[(ptgen_edges[i]+ptgen_edges[i+1])/2 for i in range(len(ptgen_edges)-1)]))
        pt_bin = hist.axis.Variable(ptgen_edges, name="ptreco", label=r"p_{T,RECO} (GeV)")  
        pt_gen_bin = hist.axis.Variable(ptgen_edges, name="ptgen", label=r"p_{T,GEN} (GeV)") 
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
        phi_axis = hist.axis.Regular(150, -2*np.pi, 2*np.pi, name="phi", label=r"$\phi$")
        weight_bin = hist.axis.Regular(100, 0, 5, name="corrWeight", label="Weight")

        self._histos = {
        #### btag study histos
        'alljet_ptreco_mreco':        hist.Hist(jet_cat, parton_cat, mass_bin, pt_bin, storage="weight", name="Events"),
        'btag_eta':            hist.Hist(jet_cat, parton_cat, frac_axis, eta_bin, storage="weight", name="Events"),
            
        #### Plots of things during the selection process / for debugging
        'njet_gen':                  hist.Hist(syst_cat, n_axis, storage="weight", label="Events"),
        'dphimin_gen':               hist.Hist(syst_cat, dphi_axis, storage="weight", label="Events"),
        'asymm_gen':               hist.Hist(syst_cat, frac_axis, storage="weight", label="Events"),
        'njet_reco':                  hist.Hist(syst_cat, n_axis, storage="weight", label="Events"),
        'dphimin_reco':               hist.Hist(syst_cat, dphi_axis, storage="weight", label="Events"),
        'asymm_reco':               hist.Hist(syst_cat, frac_axis, storage="weight", label="Events"),
            
        'jet_dr_reco_gen':           hist.Hist(syst_cat, dr_axis, storage="weight", label="Events"),
        # 'jet_eta_reco':              hist.Hist(syst_cat, eta_bin, storage="weight", name="Events"),
        'jet_rap_reco':              hist.Hist(syst_cat, y_bin, storage="weight", name="Events"),
        'jet_rap_gen':               hist.Hist(syst_cat, y_bin, storage="weight",name="Events"),
        #'jet_dr_gen':                hist.Hist(dr_axis, storage="weight", label="Counts"),
        #'jet_dr_reco':               hist.Hist(dr_axis, storage="weight", label="Counts"),
        'jet_phi_gen':             hist.Hist(syst_cat, phi_axis, storage="weight", label="Events"),
        'jet_phi_reco':             hist.Hist(syst_cat, phi_axis, storage="weight", label="Events"),
        
        'jet_dr_gen_subjet':         hist.Hist(syst_cat, dr_axis, storage="weight", label="Events"),
        'jet_dr_reco_to_gen_subjet': hist.Hist(syst_cat, dr_axis, storage="weight", label="Events"),
        'misses_g':                  hist.Hist(syst_cat, pt_gen_bin, mass_gen_bin, storage="weight", name="Events"),
        'fakes_g':                   hist.Hist(syst_cat, pt_bin, mass_bin, storage="weight", name="Events"),
        'misses':                    hist.Hist(syst_cat, pt_gen_bin, mass_gen_bin, storage="weight", name="Events"),
        'fakes':                     hist.Hist(syst_cat, pt_bin, mass_bin, storage="weight", name="Events"),
        #### hist for comparison of weights
        'weights':                   hist.Hist(syst_cat, weight_bin, storage="weight", name="Events"),
        #### for investigation of removing fakes
        'fakes_eta_phi':             hist.Hist(syst_cat, eta_bin, phi_axis, storage="weight", name="Events"),
        'fakes_asymm_dphi':             hist.Hist(syst_cat, frac_axis, dphi_axis, storage="weight", name="Events"),
            
        #### Plots to be unfolded
        'ptreco_mreco_u':       hist.Hist(syst_cat, pt_bin, mass_bin, storage="weight", name="Events"),
        'ptreco_mreco_g':        hist.Hist(syst_cat, pt_bin, mass_bin, storage="weight", name="Events"),

        #### Plots for comparison
        'ptgen_mgen_u':        hist.Hist(syst_cat, pt_gen_bin, mass_gen_bin, storage="weight", label="Events"),       
        'ptgen_mgen_g':            hist.Hist(syst_cat, pt_gen_bin, mass_gen_bin, storage="weight", label="Events"),
        
        #### Plots to get JMR and JMS in MC
        # 'jet_m_pt_u_reco_over_gen': hist.Hist(syst_cat, pt_gen_bin, mass_gen_bin, frac_axis, storage="weight",                                                                        label="Events"),
        # 'jet_m_pt_g_reco_over_gen':  hist.Hist(syst_cat, pt_gen_bin, mass_gen_bin, frac_axis, storage="weight",                                                                        label="Events"),

        #### Plots for the analysis in the proper binning
        'response_matrix_u':    hist.Hist(syst_cat, pt_bin, mass_bin, pt_gen_bin, mass_gen_bin, storage="weight",                                                         label="Events"),
        'response_matrix_g':     hist.Hist(syst_cat, pt_bin, mass_bin, pt_gen_bin, mass_gen_bin, storage="weight",                                                         label="Events"),
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
        if IOV == '2018' and self.hem:
            nEvents = len(events)
            events = events[HEMVeto(events.FatJet, events.run)]
            print("nEvents removed by HEMveto: ", len(events)- nEvents)
        FatJet=events.FatJet
        FatJet["p4"] = ak.with_name(events.FatJet[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
        if self.do_gen:
            era = None
            GenJetAK8 = events.GenJetAK8
            GenJetAK8['p4']= ak.with_name(events.GenJetAK8[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
            FatJet["pt_gen"] = ak.values_astype(ak.fill_none(FatJet.p4.nearest(GenJetAK8.p4, threshold=0.4).pt, 0), np.float32)
        else:
            firstidx = filename.find( "store/data/" )
            fname2 = filename[firstidx:]
            fname_toks = fname2.split("/")
            era = fname_toks[ fname_toks.index("data") + 1]
            print("IOV ", IOV, ", era ", era)
        corrected_fatjets = GetJetCorrections(FatJet, events, era, IOV, isData=not self.do_gen)
        # print("Corrected fatjets ", corrected_fatjets.fields)
        # corrected_fatjets = FatJet
        jet_corrs = {}
        self.weights = {}
        if 'hem' in self.jet_systematics and self.do_gen:
            jet_corrs.update({
                       "hem": HEMCleaning(FatJet)
                      })
        if 'jer' in self.jet_systematics and self.do_gen:
            jet_corrs.update({"jerUp": corrected_fatjets.JER.up,
                                "jerDown": corrected_fatjets.JER.down
                            })
        if 'nominal' in self.jet_systematics or not self.do_gen:
            print("Doing nominal")
            jet_corrs.update({"nominal": corrected_fatjets})
        if 'jes' in self.jet_systematics and self.do_gen:
            for unc_src in (unc_src for unc_src in corrected_fatjets.fields if "JES" in unc_src):
                print("Uncertainty source: ", unc_src)
                jet_corrs.update({
                    unc_src+"Up":corrected_fatjets[unc_src].up,
                    unc_src+"Down":corrected_fatjets[unc_src].down, })
        elif self.do_gen:
            print("input jet systs ", self.jet_systematics)
            avail_srcs = [unc_src for unc_src in self.jet_systematics if ("JES_"+unc_src in corrected_fatjets.fields)]
            for unc_src in avail_srcs:
                print("Uncertainty source: ", unc_src)
                print(corrected_fatjets["JES_"+unc_src])
                jet_corrs.update({
                    unc_src+"Up":corrected_fatjets["JES_"+unc_src].up,
                    unc_src+"Down":corrected_fatjets["JES_"+unc_src].down  }) 
        #### Initialize dictionary to store weights for all jet correction objects
        self.weights = {}
        print("final jet corrs ", jet_corrs.keys())
        for jetsyst in jet_corrs.keys():
            print("Length of corrected: ", len(jet_corrs[jetsyst]), " length fo events: ", len(events))
            print("Adding ", jetsyst, " values ", jet_corrs[jetsyst], " to output")
            events_corr = ak.with_field(events, jet_corrs[jetsyst], "FatJet")
            weights = np.ones(len(events_corr))
            print("Length of events before any cuts", len(events_corr), "length of weights ", len(weights))
            if self.do_gen and 'herwig' in dataset:
                era = None
                weights = events_corrs.LHEWeight.originalXWGTUP
            elif (self.do_gen):
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
                out['cutflow']['nEvents after trigger sel '+jetsyst] += (len(events_corr.FatJet))
            
            #### Need to add PU reweighting for if do_gen
            #### Remove event with very large gen weights???
            # print("NPVs ",events.PV.fields)
            # sel.add("npv", events.PV.npvsGood>0)
    
            #####################################
            #### Gen Jet Selection
            ####################################       
            if self.do_gen:
                # sel.add("threeGenJets", ak.num(events.GenJetAK8) >= 3)
                pt_cut_gen = ak.all(events_corr.GenJetAK8.pt > 170., axis = -1) ### 70% of reco pt cut
                GenJetAK8 = events_corr.GenJetAK8
                GenJetAK8['p4']= ak.with_name(events_corr.GenJetAK8[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
                rap_cut_gen = ak.all(np.abs(getRapidity(GenJetAK8.p4)) < 3.0, axis = -1)
                out['cutflow']['nEvents after >2 gen jet only '+jetsyst] += (len(events_corr[(ak.num(events_corr.GenJetAK8) > 2)].FatJet))
                out['cutflow']['nEvents after gen rap cut only '+jetsyst] += (len(events_corr[rap_cut_gen].FatJet))
                out['cutflow']['nEvents after gen pt cut only '+jetsyst] += (len(events_corr[pt_cut_gen].FatJet))
                kinesel = (pt_cut_gen & rap_cut_gen & (ak.num(events_corr.GenJetAK8) > 2))
                out["njet_gen"].fill(syst=jetsyst, n=ak.num(events_corr.GenJetAK8[rap_cut_gen & pt_cut_gen]), 
                                     weight = weights[rap_cut_gen & pt_cut_gen] ) 
                                 
                #### Get leading 3 jets
                events_corr = events_corr[kinesel]
                weights = weights[kinesel]
                GenJetAK8 = events_corr.GenJetAK8
                GenJetAK8['p4']= ak.with_name(GenJetAK8[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
                out["jet_rap_gen"].fill(syst=jetsyst, rapidity=np.abs(getRapidity(GenJetAK8[:,2].p4)), weight=weights)
                out["jet_phi_gen"].fill(syst=jetsyst, phi=events_corr.GenJetAK8[:,2].phi, weight=weights)
                out['cutflow']['nEvents after >2 jet, rapidity, and pT gen selection '+jetsyst] += (len(events_corr.FatJet))
                print('nEvents after >2 jet, rapidity, and pT selection ',jetsyst, " ", (len(events_corr.FatJet)))
                genjet1 = events_corr.GenJetAK8[:,0]
                genjet2 = events_corr.GenJetAK8[:,1]
                genjet3 = events_corr.GenJetAK8[:,2]
           
                ### calculate dphi_min
                dphi12_gen = np.abs(genjet1.delta_phi(genjet2))
                dphi13_gen = np.abs(genjet1.delta_phi(genjet3))
                dphi23_gen = np.abs(genjet2.delta_phi(genjet3))
                dphimin_gen = np.amin([dphi12_gen, dphi13_gen, dphi23_gen], axis = 0)
                out["dphimin_gen"].fill(syst=jetsyst, dphi = dphimin_gen, weight = weights)
                asymm_gen  = np.abs(genjet1.pt - genjet2.pt)/(genjet1.pt + genjet2.pt)
                out["asymm_gen"].fill(syst=jetsyst, frac = asymm_gen, weight=weights)
                out['cutflow']['nEvents after gen dphi cut only '+jetsyst] += (len(events_corr[dphimin_gen<1.0].FatJet))
                out['cutflow']['nEvents after gen asymm cut only '+jetsyst] += (len(events_corr[asymm_gen<0.3].FatJet))
                events_corr = events_corr[(dphimin_gen > 0.5) & (asymm_gen < 0.3)]
                weights = weights[(dphimin_gen > 0.5) & (asymm_gen < 0.3)]
                out['cutflow']['nEvents after gen dphi and ptasymm selection '+jetsyst] += (len(events_corr.FatJet))
                print('nEvents after gen dphi and ptasymm selection ',jetsyst,' ',(len(events_corr.FatJet)))
                gensubjets = events_corr.SubGenJetAK8
                groomed_genjet = get_gen_sd_mass_jet(events_corr.GenJetAK8, gensubjets)
                matches = ak.all(events_corr.GenJetAK8.delta_r(events_corr.GenJetAK8.nearest(events_corr.FatJet)) < 0.4, axis = -1)
                # matches_g = ak.all(groomed_genjet.delta_r(groomed_genjet.nearest(events.FatJet)) < 0.15, axis = -1)
                misses = ~matches
                # gen but no reco
                print("misses: ", len(misses))
                print("events: ", len(events_corr))
                out['cutflow']['misses'] += (len(events_corr[misses].FatJet))
                out["misses"].fill(syst=jetsyst, ptgen = events_corr[misses].GenJetAK8[:,2].pt, 
                                        mgen = events_corr[misses].GenJetAK8[:,2].mass)
                # out["misses_g"].fill(ptgen = ak.flatten(groomed_genjet[misses_g][:,2].pt), 
                #                         mgen = ak.flatten(groomed_genjet[misses_g][:,2].mass))
                events_corr = events_corr[matches]
                weights = weights[matches]
                print("Gen jet fields: ", events_corr.GenJetAK8.fields)
                out['cutflow']['nEvents after removing misses '+jetsyst] += (len(events_corr.FatJet))
                print('nEvents after removing misses ',jetsyst,' ',(len(events_corr.FatJet)))
                # if deltaR matching results in too few events
                if (len(events_corr) < 1): return out
    #         print("Fields available for GenJets ", trijetEvents.GenJetAK8.fields)
    
    
            #####################################
            #### Reco Jet Selection
            ####################################
        
            #         sel.add("threeRecoJets", ak.num(events.FatJet) >= 3)
            
            pt_cut = (ak.all(events_corr.FatJet.pt > self.ptcut, axis = -1))
            weights = weights[(ak.num(events_corr.FatJet) >= 3) & pt_cut]
            events_corr = events_corr[(ak.num(events_corr.FatJet) >= 3) & pt_cut]
            FatJet = events_corr.FatJet
            FatJet["p4"] = ak.with_name(events_corr.FatJet[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
            rap_cut = ak.all(np.abs(getRapidity(FatJet.p4)) < self.ycut, axis = -1)
            sdm_cut = (ak.all(events_corr.FatJet.msoftdrop > 10., axis = -1))
            out["njet_reco"].fill(syst = jetsyst, n=ak.num(events_corr.FatJet[rap_cut]), 
                                     weight = weights[rap_cut] )
            out["jet_rap_reco"].fill(syst=jetsyst, rapidity=np.abs(getRapidity(FatJet[:,2].p4)), weight = weights)
            out["jet_phi_reco"].fill(syst=jetsyst, phi=FatJet[:,2].phi, weight=weights)  
            weights = weights[rap_cut & sdm_cut]
            #### Add cut on softdrop mass as done in previous two papers --> need to very with JMS/JMR studies
            events_corr = events_corr[rap_cut & sdm_cut]
            # out["jet_eta_reco"].fill(syst=jetsyst, eta = events_corr.FatJet[:,2].eta, weight=weights)
            out['cutflow']['nEvents after reco kine selection '+jetsyst] += (len(events_corr.FatJet))
            jet1 = events_corr.FatJet[:, 0]
            jet2 = events_corr.FatJet[:, 1]
            jet3 = events_corr.FatJet[:, 2]
            dphi12 = np.abs(jet1.delta_phi(jet2))
            dphi13 = np.abs(jet1.delta_phi(jet3))
            dphi23 = np.abs(jet2.delta_phi(jet3))
            dphimin = np.amin([dphi12, dphi13, dphi23], axis = 0)
            out["dphimin_reco"].fill(syst=jetsyst, dphi = dphimin, weight = weights)
            asymm = np.abs(jet1.pt - jet2.pt)/(jet1.pt + jet2.pt)
            out["asymm_reco"].fill(syst=jetsyst, frac = asymm, weight=weights)
            events_corr = events_corr[(dphimin > 1.0) & (asymm < 0.3)]
            weights = weights[(dphimin > 1.0) & (asymm < 0.3)]
            out['cutflow']['nEvents after reco topo selection '+jetsyst] += (len(events_corr.FatJet))
            print('nEvents after reco topo selection ',jetsyst,' ',len(events_corr.FatJet))
            #### Apply btag 
            events_corr, btagSel = applyBTag(events_corr, self.btag)
            weights = weights[btagSel]
            out['cutflow']['nEvents after reco btag '+jetsyst] += (len(events_corr.FatJet))
            print('nEvents after reco btag ',jetsyst, ' ', len(events_corr.FatJet))
            
            #### Find fakes and make response matrices
            if self.do_gen:
                #### fakes = reco but no gen
                fakes = ak.any(ak.is_none(events_corr.FatJet.matched_gen, axis = -1), axis = -1)
                print("Length of fakes ", len(fakes) ," length of weights we're masking ", len(weights))
                print(ak.sum(fakes))
                if ak.sum(fakes)>0:
                    fake_events = events_corr[fakes]
                    fake_weights = Weights(len(weights[fakes]))
                    dphimin = dphimin[(dphimin > 1.0) & (asymm < 0.3)]
                    dphimin_fakes = dphimin[fakes]
                    print("Intital fake weights: ", fake_weights)
                    fake_weights.add('fakeWeight', weights[fakes])
                    if "L1PreFiringWeight" in events.fields and "L1PreFiringWeight" in self.systematics:                
                        prefiringNom, prefiringUp, prefiringDown = GetL1PreFiringWeight(events_corr[fakes])
                        fake_weights.add("L1prefiring", weight=prefiringNom, 
                                               weightUp=prefiringUp, 
                                               weightDown=prefiringDown,
                                   )
                    if "PUSF" in self.systematics:
                        puUp, puDown, puNom = GetPUSF(events_corr[fakes], IOV)
                        fake_weights.add("PUSF", weight=puNom, weightUp=puUp,
                                           weightDown=puDown,) 
                    if "PDF" in self.systematics:
                        pdfNom, pdfUp, pdfDown = GetLHEWeight(events_corr[fakes])
                        fake_weights.add("PDF", weight=pdfNom, weightUp=pdfUp, weightDown=pdfDown)
                    print("Length of fakes jets ", ak.flatten(events_corr[fakes].FatJet[:,:2].mass, axis=1), "length of fake weights", len(fake_weights.weight()))
                    out["fakes"].fill(syst=jetsyst, ptreco = events_corr[fakes].FatJet[:,2].pt, mreco = events_corr[fakes].FatJet[:,2].mass, weight=fake_weights.weight())
                    out['fakes_eta_phi'].fill(syst=jetsyst, phi = events_corr[fakes].FatJet[:,2].phi, eta = events_corr[fakes].FatJet[:,2].eta, weight=fake_weights.weight())
                    out["fakes_g"].fill(syst=jetsyst, ptreco = events_corr[fakes].FatJet[:,2].pt, mreco = events_corr[fakes].FatJet[:,2].msoftdrop, weight=fake_weights.weight())
                out['cutflow']['fakes '+jetsyst] += len(events_corr[fakes].FatJet)
                matched_reco = ~fakes
                events_corr = events_corr[matched_reco]
                weights = weights[matched_reco]
                out['cutflow']['nEvents after removing fakes '+jetsyst] += (len(events_corr.FatJet))
                print('nEvents after removing fakes ',jetsyst,' ',len(events_corr.FatJet))
                # if gen matching results in too few events
                if (len(events_corr) < 1): return out
                #### Get gen soft drop mass
                genjet = events_corr.GenJetAK8[:,2]
                gensubjets = events_corr.SubGenJetAK8
                groomed_genjet = get_gen_sd_mass_jet(genjet, gensubjets)
                #### Create coffea analysis weights object
                self.weights[jetsyst] = Weights(len(weights))
                #### store initial gen weights
                # print("genweights before adding to weights object ", weights)
                print("Len of weights before adding to weights object ", len(weights))
                self.weights[jetsyst].add('jetWeight', weight=weights)
                if "L1PreFiringWeight" in events.fields and "L1PreFiringWeight" in self.systematics:                
                    prefiringNom, prefiringUp, prefiringDown = GetL1PreFiringWeight(events_corr)
                    self.weights[jetsyst].add("L1prefiring", weight=prefiringNom, weightUp=prefiringUp, weightDown=prefiringDown)
                if "PUSF" in self.systematics:
                    puUp, puDown, puNom = GetPUSF(events_corr, IOV)
                    self.weights[jetsyst].add("PUSF", weight=puNom, weightUp=puUp, weightDown=puDown)
                if "PDF" in self.systematics:
                    if herwig in dataset:
                        pdfNom, pdfUp, pdfDown = GetLHEWeight(events_corr)
                        # print("Nominal pdf: ", pdfNom, " pdf up: ", pdfUp, " pdf down: ", pdfDown)
                        self.weights[jetsyst].add("PDF", weight=pdfNom, weightUp=pdfUp, weightDown=pdfDown)
                out["ptgen_mgen_u"].fill(syst=jetsyst, ptgen=genjet.pt, mgen=genjet.mass, weight=self.weights[jetsyst].weight() )
                out["ptgen_mgen_g"].fill(syst=jetsyst, ptgen=groomed_genjet.pt, mgen=groomed_genjet.mass, weight=self.weights[jetsyst].weight() )
                #### Get dr between gen and reco jets (should be all < 0.4)
                jet = events_corr.FatJet[:,2]
                out["jet_dr_reco_gen"].fill(syst=jetsyst, dr=jet.delta_r(genjet), weight=self.weights[jetsyst].weight())
                #### Final plots
                out["response_matrix_u"].fill(syst=jetsyst, ptreco=jet.pt, ptgen=genjet.pt, mreco=jet.mass, mgen=genjet.mass, weight = self.weights[jetsyst].weight())
                out["response_matrix_g"].fill(syst=jetsyst, ptreco=jet.pt, ptgen=groomed_genjet.pt, mreco=jet.msoftdrop, mgen=groomed_genjet.mass, weight = self.weights[jetsyst].weight() )
                #### Outliers
                weird = (np.abs(jet.msoftdrop - groomed_genjet.mass) > 20.0) & (jet.msoftdrop > 10.)
                out['cutflow']['Number of outliers '+jetsyst] += (len(events_corr[weird].FatJet))
                weights = self.weights[jetsyst].weight()
                if jetsyst=="nominal":
                    for syst in self.weights[jetsyst].variations:
                        print("Weight variation: ", syst)
                        #fill nominal, up, and down variations for each
                        out['ptgen_mgen_u'].fill(syst=syst, ptgen=genjet.pt, mgen=genjet.mass, weight=self.weights[jetsyst].weight(syst) )
                        out['ptgen_mgen_g'].fill(syst=syst, ptgen=groomed_genjet.pt, mgen=groomed_genjet.mass, 
                                                      weight=self.weights[jetsyst].weight(syst) )           
                        out["response_matrix_u"].fill(syst=syst,
                                               ptreco=jet.pt, ptgen=genjet.pt,
                                               mreco=jet.mass, mgen=genjet.mass, weight=self.weights[jetsyst].weight(syst))
                        out["response_matrix_g"].fill(syst=syst,
                                               ptreco=jet.pt, ptgen=groomed_genjet.pt,
                                               mreco=jet.msoftdrop, mgen=groomed_genjet.mass, weight=self.weights[jetsyst].weight(syst))
                        out["ptreco_mreco_u"].fill( syst=syst, ptreco=events_corr.FatJet[:,2].pt, mreco=events_corr.FatJet[:,2].mass, weight=self.weights[jetsyst].weight(syst) )
                        out["ptreco_mreco_g"].fill( syst=syst, ptreco=events_corr.FatJet[:,2].pt, mreco=events_corr.FatJet[:,2].msoftdrop, weight=self.weights[jetsyst].weight(syst) )
                        if ak.sum(fakes)>0:
                            out["fakes"].fill(syst=syst, ptreco = fake_events.FatJet[:,2].pt, mreco = fake_events.FatJet[:,2].mass, weight=fake_weights.weight(syst))
                            out["fakes_g"].fill(syst=syst, ptreco = fake_events.FatJet[:,2].pt, mreco = fake_events.FatJet[:,2].msoftdrop, weight=fake_weights.weight(syst))
                    #### Gluon purity plots        
                    jet1flav = getJetFlavors(events_corr.FatJet[:,0])
                    jet2flav = getJetFlavors(events_corr.FatJet[:,1])
                    jet3flav = getJetFlavors(events_corr.FatJet[:,2])
                    genjet1 = events_corr.FatJet[:,0].matched_gen
                    genjet2 = events_corr.FatJet[:,1].matched_gen
                    jet3 = events_corr.FatJet[:,2]
                    jet3_bb = jet3[(np.abs(genjet1.partonFlavour) == 5) & (np.abs(genjet2.partonFlavour) == 5)]
                    jet3_b = jet3[(np.abs(genjet1.partonFlavour) == 5)]
                    jet3_jetbb_flav = getJetFlavors(jet3_bb)
                    jet3_jetb_flav = getJetFlavors(jet3_b)
                    
                    jets = {"jet1":jet1flav, "jet2":jet2flav,  "jet3":jet3flav, "jet3_bb":jet3_jetbb_flav, "jet3_b":jet3_jetb_flav}
                    print("Check for none values", ak.any(ak.is_none(jet3flav["Gluon"].mass), axis = -1))
                    for flavor in jet1flav.keys():
                        for jetname, jet in jets.items():
                            out['alljet_ptreco_mreco'].fill(jetNumb = jetname, partonFlav = flavor, 
                                                            mreco = jet[flavor].mass, 
                                                            ptreco = jet[flavor].pt)
                            out['btag_eta'].fill(jetNumb = jetname, partonFlav = flavor, 
                                                 frac = jet[flavor].btagDeepB, eta = jet[flavor].eta )
                    out['cutflow']['nGluonJets'] += (len(jet3flav["Gluon"])+len(jet1flav["Gluon"])+len(jet2flav["Gluon"]))
                    out['cutflow']['nJets'] += (len(events_corr.FatJet[:,0])+len(events_corr.FatJet[:,1])+len(events_corr.FatJet[:,2]))
                    out['cutflow']['nSoftestGluonJets'] += (len(jet3flav["Gluon"]))
                    out['cutflow']['nSoftestGluonJets_b'] += (len(jet3_jetb_flav["Gluon"]))
                    out['cutflow']['nSoftestGluonJets_bb'] += (len(jet3_jetbb_flav["Gluon"]))
                    out['cutflow']['nSoftestJets_b'] += (len(jet3_b))
                    out['cutflow']['nSoftestJets_bb'] += (len(jet3_bb))
                    out['cutflow']['n3Jets'] += (len(events_corr.FatJet[:,2].pt))
            out["ptreco_mreco_u"].fill( syst=jetsyst, ptreco=events_corr.FatJet[:,2].pt, mreco=events_corr.FatJet[:,2].mass, weight=weights )
            out["ptreco_mreco_g"].fill( syst=jetsyst, ptreco=events_corr.FatJet[:,2].pt, mreco=events_corr.FatJet[:,2].msoftdrop, weight=weights )
            negMSD = events_corr.FatJet[:,2].msoftdrop<0.
            print("Number of negative softdrop values ", ak.sum(negMSD), "out of ", negMSD )
            out['cutflow']['nEvents failing softdrop condition'] += ak.sum(negMSD)
            out['cutflow']['nEvents final selection'] += (len(events_corr.FatJet))
            del events_corr, weights
        out['cutflow']['chunks'] += 1
        return out
    
    def postprocess(self, accumulator):
        return accumulator
    
def main():
    #### Next run processor with futures executor on all test files
    from dask.distributed import Client
    from plugins import runCoffeaJob
    processor = makeTrijetHists()
    result = runCoffeaJob(processor, jsonFile = "QCD_flat_files.json", winterfell = True, testing = True, data = False)
    util.save(result, "coffeaOutput/trijet_pT" + str(processor.ptcut) + "_eta" + str(processor.etacut) + "_result_test.coffea")
    

if __name__ == '__main__':
    # Execute when the module is not initialized from an import statement: i.e. called from terminal command line
    main()

