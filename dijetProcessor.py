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

def get_dphi( jet0, jet1 ):
    '''
    Find dphi between two jets, returning none when the event does not have at least two jets
    '''
    combs = ak.cartesian( (jet0, jet1), axis=1 )
    print("Jet 0: ", jet0, '\n', combs['0'])
    print("Jet 1: ", jet0, '\n', combs['1'])
    dphi = np.abs(combs['0'].delta_phi(combs['1']))
    return ak.firsts(dphi)


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
        events = events
        print('no btag applied')
    return events

#### currently only for MC --> makes hists and response matrix
class makeDijetHists(processor.ProcessorABC):
    '''
    Processor to run a dijet jet mass cross section analysis. 
    With "do_gen == True", will perform GEN selection and create response matrices. 
    Will always plot RECO level quantities. 
    '''
    def __init__(self, ptcut = 200., etacut = 2.5, data = False):
        # should have separate **lower** ptcut for gen
        self.do_gen = not data
        self.ptcut = ptcut
        self.etacut = etacut
        print("Data: ", data, " gen ", self.do_gen)
        dataset_cat = hist.axis.StrCategory([],growth=True,name="dataset", label="Dataset")
        jet_cat = hist.axis.StrCategory([], growth=True, name="jetNumb", label="Jet")
        parton_cat = hist.axis.StrCategory([], growth=True,name="partonFlav", label="Parton Flavour")
        mgen_bin_edges = np.array([0,1,5,10,20,40,60,80,100,150,200,250,1000])
        mreco_bin_edges = np.sort(np.append(mgen_bin_edges,[(mgen_bin_edges[i]+mgen_bin_edges[i+1])/2 for i in range(len(mgen_bin_edges)-1)]))
        mass_gen_bin =  hist.axis.Variable(mgen_bin_edges, name="mgen", label=r"m_{GEN} (GeV)")                         
        mass_bin = hist.axis.Variable(mreco_bin_edges, name="mreco", label=r"m_{RECO} (GeV)")
        pt_bin = hist.axis.Variable([200,280,360,450,520,630,690,750,800,1300,13000], name="ptreco", label=r"p_{T,RECO} (GeV)")        
        pt_gen_bin = hist.axis.Variable([200,280,360,450,520,630,690,750,800,1300,13000], name="ptgen", label=r"p_{T,GEN} (GeV)") 
        eta_bin = hist.axis.Regular(25, 0., 2.5, name="eta", label="Eta")
        frac_axis = hist.axis.Regular(10, 0.0, 1., name="frac", label="Fraction")
        n_axis = hist.axis.Regular(5, 0, 5, name="n", label=r"Number")
        dr_axis = hist.axis.Regular(150, 0, 6.0, name="dr", label=r"$\Delta R$")
        dphi_axis = hist.axis.Regular(150, -2*np.pi, 2*np.pi, name="dphi", label=r"$\Delta \phi$")
        
        self._histos = {
            #### Old histos
            'jet_mass':             hist.Hist(dataset_cat, jet_cat, parton_cat, mass_bin, storage="weight", name="Events"),
            'jet_pt':             hist.Hist(dataset_cat, jet_cat, parton_cat, pt_bin, storage="weight", name="Events"),
            'jet_eta':            hist.Hist(dataset_cat, jet_cat, parton_cat, eta_bin, storage="weight", name="Events"),
            
            #### Plots of things during the selection process / for debugging
            #'njet_reco':                 hist.Hist(dataset_cat, n_axis, storage="weight", label="Counts"),
            'njet_gen':                  hist.Hist(dataset_cat, n_axis, storage="weight", label="Counts"),
            #'jet_dr_reco_gen':           hist.Hist(dataset_cat, dr_axis, storage="weight", label="Counts"),
            'jet_mass_u_reco_over_gen':    hist.Hist(dataset_cat, frac_axis, storage="weight", label="Counts"),
            #'jet_pt_reco':               hist.Hist(dataset_cat, pt_bin, storage="weight", name="Events"),
            'jet_pt_gen':                hist.Hist(dataset_cat, pt_gen_bin, storage="weight", name="Events"),
            'jet_mass_gen':                hist.Hist(dataset_cat, jet_cat, mass_gen_bin, storage="weight", name="Events"),
            'jet_pt_reco_over_gen':      hist.Hist(dataset_cat, frac_axis, storage="weight", label="Counts"),
            #'jet_eta_reco':              hist.Hist(dataset_cat, jet_cat, parton_cat, eta_bin, storage="weight", name="Events"),
            #'jet_eta_gen':               hist.Hist(dataset_cat, jet_cat, eta_bin, storage="weight",name="Events"),
            #'jet_dr_gen':                hist.Hist(dataset_cat, dr_axis, storage="weight", label="Counts"),
            #'jet_dr_reco':               hist.Hist(dataset_cat, dr_axis, storage="weight", label="Counts"),
            'jet_dphi_gen':              hist.Hist(dataset_cat, dphi_axis, storage="weight", label="Counts"),
            #'jet_dphi_reco':             hist.Hist(dataset_cat, dphi_axis, storage="weight", label="Counts"),
            'jet_ptasymm_gen':           hist.Hist(dataset_cat, frac_axis, storage="weight", label="Counts"),
            #'jet_ptasymm_reco':          hist.Hist(dataset_cat, frac_axis, storage="weight", label="Counts"),
            'jet_dr_gen_subjet':         hist.Hist(dataset_cat, dr_axis, storage="weight", label="Counts"),
            'dijet_dr_reco_to_gen': hist.Hist(dataset_cat, dr_axis, storage="weight", label="Counts"),
            'dr_reco_to_gen_subjet' : hist.Hist(dataset_cat, dr_axis, storage="weight", label="Counts"),
            'misses':       hist.Hist(dataset_cat, pt_gen_bin, mass_gen_bin, storage="weight", name="Events"),
            'fakes':        hist.Hist(dataset_cat, pt_bin, mass_bin, storage="weight", name="Events"),
            
            #### Plots to be unfolded
            'jet_pt_mass_reco_u':       hist.Hist(dataset_cat, pt_bin, mass_bin, storage="weight", name="Events"),
            'jet_pt_mass_reco_g':        hist.Hist(dataset_cat, pt_bin, mass_bin, storage="weight", name="Events"),
    
            #### Plots for comparison
            'jet_pt_mass_u_gen':        hist.Hist(dataset_cat, pt_gen_bin, mass_gen_bin, storage="weight", label="Counts"),       
            'jet_pt_mass_g_gen':            hist.Hist(dataset_cat, pt_gen_bin, mass_gen_bin, storage="weight", label="Counts"),
        
        
            #### Plots to get JMR and JMS in MC
            'jet_m_pt_u_reco_over_gen': hist.Hist(dataset_cat, pt_gen_bin, mass_gen_bin, frac_axis, storage="weight",                                                                        label="Counts"),
            'jet_m_pt_g_reco_over_gen':  hist.Hist(dataset_cat, pt_gen_bin, mass_gen_bin, frac_axis, storage="weight",                                                                        label="Counts"),
        
            #### Plots for the analysis in the proper binning
            'response_matrix_u':    hist.Hist(dataset_cat, pt_bin, mass_bin, pt_gen_bin, mass_gen_bin, storage="weight",                                                         label="Counts"),
            'response_matrix_g':     hist.Hist(dataset_cat, pt_bin, mass_bin, pt_gen_bin, mass_gen_bin, storage="weight",                                                         label="Counts"),
                 
            #### misc.
            'cutflow':            processor.defaultdict_accumulator(int),
                             }
        ## This is for rejecting events with large weights
        self.means_stddevs = defaultdict()
    
    @property
    def accumulator(self):
        return self._histos
    
    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        out = self._histos
        dataset = events.metadata['dataset']
        filename = events.metadata['filename']
        print("Filename: ", filename)
        print("dataset: ", dataset)
        #####################################
        #### Find the IOV from the dataset name
        #####################################
        IOV = ('2016APV' if ( any(re.findall(r'APV',  dataset)) or any(re.findall(r'UL2016APV', dataset)))
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
            print("Do XS scaling")
            weights = weights * getXSweight(dataset, IOV)
        else:
            firstidx = filename.find( "store/data/" )
            fname2 = filename[firstidx:]
            fname_toks = fname2.split("/")
            era = fname_toks[ fname_toks.index("data") + 1]
            print("IOV ", IOV, ", era ", era)
            #apply lumimask and require at least one jet to apply jet trigger prescales
            print("apply lumimask")
            lumi_mask = getLumiMask(IOV)(events.run, events.luminosityBlock)
            events = events[lumi_mask & (ak.num(events.FatJet) >= 1)]
            print("call applyprescales")
            trigsel, psweights = applyPrescales(events, year = IOV)
            weights=psweights
            print("Trigger: len of events ", len(events), "len of weights ", len(trigsel))
            print(weights)
            events = events[trigsel]
            weights = weights[trigsel]
            
        #### Need to add PU reweighting for if do_gen
        #### Remove event with very large gen weights???
        
        #####################################
        ### Use cofffea PackedSelection to apply cuts
        #####################################
        sel = PackedSelection()
        
        #### NPV selection
        print("Npvs ", events.PV.fields)
        # sel.add("npv", events.PV.npvsGood>0)
        
        #####################################
        #### Gen Jet Selection
        #### see CMS PAS SMP-20-010 for selections
        ####################################
        if (self.do_gen):
            print("DOING GEN")
            #### Select events with at least 2 jets
            pt_cut_gen = ak.all(events.GenJetAK8.pt > 140., axis = -1) ### 70% of reco pt cut
            eta_cut_gen = ak.all(np.abs(events.GenJetAK8.eta) < self.etacut, axis = -1)
            out["njet_gen"].fill(dataset=dataset, n=ak.num(events.GenJetAK8[eta_cut_gen & pt_cut_gen]), 
                                 weight = weights[eta_cut_gen & pt_cut_gen] )
            print("Initial # of events:  ", len(events.GenJetAK8))
            #### Apply kinematic and 2 jet requirement immediately so that dphi and asymm can be calculated
#             kinematic_sel = sel.add("kinematic_sel", (ak.num(events.GenJetAK8) >= 2) & pt_cut_gen & eta_cut_gen)
            weights = weights[(ak.num(events.GenJetAK8) >= 2) & pt_cut_gen & eta_cut_gen]
            events = events[(ak.num(events.GenJetAK8) >= 2) & pt_cut_gen & eta_cut_gen]
            print("After kin. sel.: len of events ", len(events), "len of weights ", len(weights))
#             genJets = events.GenJetAK8[~ak.any(ak.is_none(ak.firsts(events.GenJetAK8[kinematic_sel]), axis = -1), axis = -1)]
            out["jet_mass_gen"].fill(dataset=dataset, jetNumb = 'jet0', mgen=events.GenJetAK8.mass[:,0],
                                     weight=weights)
            out["jet_mass_gen"].fill(dataset=dataset, jetNumb = 'jet1', mgen=events.GenJetAK8.mass[:,1],
                                     weight=weights)
            #### get dphi and pt asymm selections  
            genjet1 = events.GenJetAK8[:,0]
            genjet2 = events.GenJetAK8[:,1]
            dphi12_gen = np.abs(genjet1.delta_phi(genjet2))
            dphi12_gen_sel = dphi12_gen > 2.
            print("Dphi: ", dphi12_gen_sel)
            asymm_gen  = np.abs(genjet1.pt - genjet2.pt)/(genjet1.pt + genjet2.pt)
            asymm_gen_sel = asymm_gen < 0.3
#             sel.add("gen_dphi_sel", dphi12_gen)
#             sel.add("gen_asymm_sel", asymm_gen_sel)
            
            #### N-1 plots
            out["jet_ptasymm_gen"].fill(dataset=dataset,frac=asymm_gen[dphi12_gen_sel], weight=weights[dphi12_gen_sel])  
            out["jet_dphi_gen"].fill(dataset=dataset, dphi=dphi12_gen[asymm_gen_sel], weight=weights[asymm_gen_sel])  
            
            events = events[dphi12_gen_sel & asymm_gen_sel]
            weights = weights[dphi12_gen_sel & asymm_gen_sel]
            print("After topo sel: len of events ", len(events), "len of weights ", len(weights))
            #misses = gen but no reco
            matches = ak.all(events.GenJetAK8.delta_r(events.GenJetAK8.nearest(events.FatJet)) < 0.2, axis = -1)
            misses = ~matches
            out["misses"].fill(dataset=dataset, ptgen = ak.flatten(events[misses].GenJetAK8.pt), 
                                    mgen = ak.flatten(events[misses].GenJetAK8.mass))
            events = events[matches]
            weights = weights[matches]
            #### Make easy gen selection
#             toposel_gen = sel.require( gen_asymm_sel=True, gen_dphi_sel=True)
#             sel.add("toposel_gen", toposel_gen)
#             gen_allsels = sel.all("npv", "kinsel_gen", "toposel_gen")
#             sel.add("gen_allselections", gen_alls)
            print("Misses ", misses)
            out['cutflow']['misses'] += (len(misses))
        #####################################
        #### Reco Jet Selection
        ####################################
#         sel.add("twoRecoJets", ak.num(events.FatJet) >= 2)
        weights = weights[(ak.num(events.FatJet) >= 2)]
        events = events[(ak.num(events.FatJet) >= 2)]
        print("Reco sel: len of events ", len(events), "len of weights ", len(weights))   
        #### Apply pt and eta cuts
        pt_cut_reco = ak.all(events.FatJet.pt > self.ptcut, axis = -1)
        eta_cut_reco = ak.all(np.abs(events.FatJet.eta) < self.etacut, axis = -1)
#         sel.add("reco_pt_eta_cut", eta_cut_reco & pt_cut_reco)
  
        
#         print("FatJet fields ", fatJets.fields, "\n")
        
        #### get dphi and pt asymm selections
        jet1 = events.FatJet[:,0]
        jet2 = events.FatJet[:,1]
        dphi12 = (np.abs(jet1.delta_phi(jet2)) > 2.)
        asymm = np.abs(jet1.pt - jet2.pt)/(jet1.pt + jet2.pt)
        asymm_reco_sel = asymm < 0.3
#         sel.add("reco_dphi_sel", dphi12)
#         sel.add("reco_asymm_sel", asymm < 0.3)
        events = events[asymm_reco_sel & dphi12]
        weights = weights[asymm_reco_sel & dphi12]
        dijet_weights = np.repeat(weights, 2)
        print("Reco kine/topo sel: len of events ", len(events), "len of weights ", len(weights))
        #### Reco event topology selection

        #### Preselection
        #### Note: Trigger is not applied in the MC, so this is 
        #### applying the full gen selection here to be in sync with rivet routine
#         if self.do_gen:
#              presel_reco = sel.all("npv", "allsel_gen", "kinsel_reco")
#         else:
#              presel_reco = sel.all("npv", "kinsel_reco")  ##add trigger selection later
#         allsel_reco = presel_reco & toposel_reco
#         sel.add("presel_reco", presel_reco)
#         sel.add("allsel_reco", allsel_reco)
        ####  Final RECO plots
        FatJet = events.FatJet
        FatJet["p4"] = ak.with_name(events.FatJet[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
        dijet_invmass = (FatJet[:,0].p4+FatJet[:,1].p4).mass
        print("FatJet: ", FatJet[:,:2].pt)
        print("Fatjet flattened along axis 0: ", ak.flatten(FatJet[:,:2], axis=1).pt)
        dijet = ak.flatten(FatJet[:,:2], axis=1)
        print("Length of FatJet after falttenting: ", len(FatJet))
        out["jet_pt_mass_reco_u"].fill( dataset=dataset, ptreco=dijet.pt, mreco=dijet.mass, weight=dijet_weights )
        out["jet_pt_mass_reco_g"].fill( dataset=dataset, ptreco=dijet.pt, mreco=dijet.msoftdrop, weight=dijet_weights )
        
        #### match jets
        if self.do_gen:
            fakes = ak.any(ak.is_none(events.FatJet.matched_gen, axis = -1), axis = -1)
            out["fakes"].fill(dataset = dataset, ptreco = ak.flatten(events[fakes].FatJet[:,:2].pt),
                              mreco = ak.flatten(events[fakes].FatJet[:,:2].mass))
            #fakes = reco but no gen
            print("Number of fake jets ", len(fakes), " number of events ", len(events))
            matched_reco = ~fakes
            events = events[matched_reco]
            weights = weights[matched_reco]
            FatJet = events.FatJet
            FatJet["p4"] = ak.with_name(events.FatJet[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
            print("Lenght of events ", len(events), "length of weights ", len(weights))
            #### Get gen subjets and sd gen jets
            GenJet = events.GenJetAK8
            GenJet['p4']= ak.with_name(events.FatJet[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
            print("Length of gen subjets: ", len(events.SubGenJetAK8))
            groomed_genjet0 = get_gen_sd_mass_jet(events.GenJetAK8[:,0], events.SubGenJetAK8)
            groomed_genjet1 = get_gen_sd_mass_jet(events.GenJetAK8[:,1], events.SubGenJetAK8)
            print("Groomed gen jets: ", groomed_genjet0, type(groomed_genjet0), ak.count(groomed_genjet0, axis = -1))
            groomed_gen_dijet = ak.concatenate([groomed_genjet0, groomed_genjet1], axis=0) 
            print("Groomed gen dijets: ", groomed_gen_dijet, type(groomed_gen_dijet), ak.count(groomed_gen_dijet, axis = -1))
            print("Length of FatJet: ", len(FatJet))
            dijet = ak.flatten(FatJet[:,:2], axis =1)
            dijet_weights = np.repeat(weights, 2)
            print("Length of FatJet after falttenting: ", len(FatJet))
            dijet_invMass = (FatJet[:,0]+FatJet[:,1]).mass
            gen_dijet = ak.flatten(GenJet[:,:2], axis=1)
            gen_dijet_invMass = (GenJet[:,0]+GenJet[:,1]).mass
            #### Gen jet and subjet plots
            out["jet_pt_gen"].fill(dataset=dataset,ptgen=dijet.pt, weight=dijet_weights)
            out["jet_dr_gen_subjet"].fill(dataset=dataset,
                                             dr=events.SubGenJetAK8[:,0].delta_r(FatJet[:,0]),
                                             weight=weights)
            #### Plots to check matching
            print("Number of matched dijet events", len(events))
            out['cutflow']['matched'] += (len(events))
#            print("Check for none values", ak.any(ak.is_none(dijetEvents, axis = -1)))
            #### dimensions of matched jets are weird -- fix

            #### Final plots
            out['jet_pt_mass_u_gen'].fill( dataset=dataset, ptgen=gen_dijet.pt, mgen=gen_dijet.mass, weight=dijet_weights )
            out['jet_pt_mass_g_gen'].fill( dataset=dataset, ptgen=groomed_gen_dijet.pt, mgen=groomed_gen_dijet.mass, weight=dijet_weights )
            
            out["dijet_dr_reco_to_gen"].fill(dataset=dataset, dr=dijet.delta_r(gen_dijet), weight=dijet_weights)


            out["response_matrix_u"].fill( dataset=dataset, 
                                               ptreco=dijet.pt, ptgen=gen_dijet.pt,
                                               mreco=dijet.mass, mgen=gen_dijet.mass )
            out["response_matrix_g"].fill( dataset=dataset, 
                                               ptreco=dijet.pt, ptgen=gen_dijet.pt,
                                               mreco=dijet.msoftdrop, mgen=groomed_gen_dijet.mass )
            

            #weird = (reco_jet.msoftdrop/groomed_gen_jet.mass > 2.0) & (reco_jet.msoftdrop > 10.)
            weird = (np.abs(dijet.msoftdrop - groomed_gen_dijet.mass) > 20.0) & (dijet.msoftdrop > 10.)
            
            #### FIX
#             recosubjets = events.SubJet           
#             subjet1 = ak.flatten(events.FatJet).subjets[:,0]
#             subjet2 = ak.flatten(events.FatJet).subjets[:,1]
#             gensubjet1=gensubjets.nearest(subjet1)
#             drsub1=gensubjets.delta_r(gensubjets.nearest(subjet1))
#             gensubjet1=gensubjets.nearest(subjet2)
#             drsub1=gensubjets.delta_r(gensubjets.nearest(subjet2))
            
#             out["dr_reco_to_gen_subjet"].fill(dataset=dataset, 
#                                                      dr=drsub1[~ak.is_none(drsub1) & ~ak.is_none(drsub2)], 
#                                                      weight=weights[~ak.is_none(drsub1) & ~ak.is_none(drsub2)])
#             out["dr_reco_to_gen_subjet"].fill(dataset=dataset, 
#                                                      dr=drsub2[~ak.is_none(drsub1) & ~ak.is_none(drsub2)], 
#                                                      weight=weights[~ak.is_none(drsub1) & ~ak.is_none(drsub2)])

            #flavour --> 21 is gluon
            genjet1 = events.GenJetAK8[:,0]
            genjet2 = events.GenJetAK8[:,1]
            jet1 = events.FatJet[:,0]
            jet2 = events.FatJet[:,1]
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

    #        #make central and forward categories instead of jet1 jet2
            #### Plots for gluon purity studies

            out['jet_mass'].fill(dataset=dataset, jetNumb = "jet1", partonFlav = "Gluon",  mreco = jet1_g.mass,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_mass'].fill(dataset=dataset, jetNumb = "jet1", partonFlav = "UDS",    mreco = jet1_uds.mass,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_mass'].fill(dataset=dataset, jetNumb = "jet1", partonFlav = "Charm",  mreco = jet1_c.mass,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_mass'].fill(dataset=dataset, jetNumb = "jet1", partonFlav = "Bottom", mreco = jet1_b.mass,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_mass'].fill(dataset=dataset, jetNumb = "jet1", partonFlav = "Other",  mreco = jet1_other.mass,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_mass'].fill(dataset=dataset, jetNumb = "jet2", partonFlav = "Gluon",  mreco = jet2_g.mass,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_mass'].fill(dataset=dataset, jetNumb = "jet2", partonFlav = "UDS",    mreco = jet2_uds.mass,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_mass'].fill(dataset=dataset, jetNumb = "jet2", partonFlav = "Charm",  mreco = jet2_c.mass,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_mass'].fill(dataset=dataset, jetNumb = "jet2", partonFlav = "Bottom", mreco = jet2_b.mass,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_mass'].fill(dataset=dataset, jetNumb = "jet2", partonFlav = "Other",  mreco = jet2_other.mass,
                                 #weight = trijetEvents.Generator.weight
                                )

            out['jet_pt'].fill(dataset=dataset, jetNumb = "jet1", partonFlav = "Gluon",  ptreco = jet1_g.pt,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_pt'].fill(dataset=dataset, jetNumb = "jet1", partonFlav = "UDS",    ptreco = jet1_uds.pt,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_pt'].fill(dataset=dataset, jetNumb = "jet1", partonFlav = "Charm",  ptreco = jet1_c.pt,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_pt'].fill(dataset=dataset, jetNumb = "jet1", partonFlav = "Bottom", ptreco = jet1_b.pt,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_pt'].fill(dataset=dataset, jetNumb = "jet1", partonFlav = "Other",  ptreco = jet1_other.pt,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_pt'].fill(dataset=dataset, jetNumb = "jet2", partonFlav = "Gluon",  ptreco = jet2_g.pt,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_pt'].fill(dataset=dataset, jetNumb = "jet2", partonFlav = "UDS",    ptreco = jet2_uds.pt,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_pt'].fill(dataset=dataset, jetNumb = "jet2", partonFlav = "Charm",  ptreco = jet2_c.pt,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_pt'].fill(dataset=dataset, jetNumb = "jet2", partonFlav = "Bottom", ptreco = jet2_b.pt,
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_pt'].fill(dataset=dataset, jetNumb = "jet2", partonFlav = "Other",  ptreco = jet2_other.pt,
                                 #weight = trijetEvents.Generator.weight
                                )

            out['jet_eta'].fill(dataset=dataset, jetNumb = "jet1", partonFlav = "Gluon",  eta = np.abs(jet1_g.eta),
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_eta'].fill(dataset=dataset, jetNumb = "jet1", partonFlav = "UDS",    eta = np.abs(jet1_uds.eta),
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_eta'].fill(dataset=dataset, jetNumb = "jet1", partonFlav = "Charm",  eta = np.abs(jet1_c.eta),
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_eta'].fill(dataset=dataset, jetNumb = "jet1", partonFlav = "Bottom", eta = np.abs(jet1_b.eta),
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_eta'].fill(dataset=dataset, jetNumb = "jet1", partonFlav = "Other",  eta = np.abs(jet1_other.eta),
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_eta'].fill(dataset=dataset, jetNumb = "jet2", partonFlav = "Gluon",  eta = np.abs(jet2_g.eta),
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_eta'].fill(dataset=dataset, jetNumb = "jet2", partonFlav = "UDS",    eta = np.abs(jet2_uds.eta),
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_eta'].fill(dataset=dataset, jetNumb = "jet2", partonFlav = "Charm",  eta = np.abs(jet2_c.eta),
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_eta'].fill(dataset=dataset, jetNumb = "jet2", partonFlav = "Bottom", eta = np.abs(jet2_b.eta),
                                 #weight = trijetEvents.Generator.weight
                                )
            out['jet_eta'].fill(dataset=dataset, jetNumb = "jet2", partonFlav = "Other",  eta = np.abs(jet2_other.eta),
                                 #weight = trijetEvents.Generator.weight
                                )
            out['cutflow']['nGluonJets'] += (len(ak.flatten(dijet[np.abs(gen_dijet.partonFlavour) == 21].pt, axis=-1)))
            out['cutflow']['nJets'] += (len(ak.flatten(jet1.pt, axis=-1)))
        out['cutflow']['chunks'] += 1
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

