#### This file contains the processors for trijet hist selections. Plotting and resulting studies are in separate files.
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
from coffea.analysis_tools import Weights, PackedSelection
from collections import defaultdict
#### import our python packages
from python.corrections import *
from python.utils import *
from copy import deepcopy
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
# do Rivet routine
# make central (eta < 1.7) and high eta bins (1.7 < eta < 2.5)

#bcut options: b_loose (apply loose bTag threshold to only hardest jet), bb_loose (apply loose bTag to leading two jets),
#              b_med(apply medium bTag to only the hardest jet), bb_med (apply medium bTag to leading two jets)

class makeTrijetHists(processor.ProcessorABC):
    def __init__(self, ptcut = 200., ycut = 2.5, btag = 'None', data = False, jet_systematics = ['nominal', 'JERUp',"HEM"], jk=False, jk_range = None):
        self.ptcut = ptcut
        self.ycut = ycut
        self.btag = btag
        self.do_gen = not data
        self.jk = jk
        self.jk_range = jk_range
        if self.jk:
            # protect against doing unc for jk --> only need nominal and memory intensive
            jet_systematics = ["nominal"]
        self.jet_systematics = jet_systematics
        print("Data: ", data, " gen ", self.do_gen)
        jet_cat = hist.axis.StrCategory([], growth=True, name="jetNumb", label="Jet")
        parton_cat = hist.axis.StrCategory([],growth=True,name="partonFlav", label="Parton Flavour")
        syst_cat = hist.axis.StrCategory([], growth=True, name='syst', label="Systematic")
        dataset_axis = hist.axis.StrCategory([], growth=True, name="dataset", label="Primary dataset")
        fine_mass_bin = hist.axis.Regular(130, 0.0, 1300.0, name="mass", label=r"mass [GeV]")
        fine_pt_bin = hist.axis.Regular(500, 100.0, 10100.0, name="pt", label=r"$p_T$ [GeV]")
        #### if using specific bin edges use hist.axis.Variable() instead
        mgen_bin_edges = np.array([0,5,10,20,40,60,80,100,150,200,300, 400, 500, 900,1300])
        mreco_bin_edges = np.sort(np.append(mgen_bin_edges,[(mgen_bin_edges[i]+mgen_bin_edges[i+1])/2 for i in range(len(mgen_bin_edges)-1)]))
        mass_gen_bin =  hist.axis.Variable(mgen_bin_edges, name="mgen", label=r"m_{GEN} (GeV)")                         
        mass_bin = hist.axis.Variable(mreco_bin_edges, name="mreco", label=r"m_{RECO} (GeV)")
        # ptgen_edges = np.array([200,300,390,480,570,680,690,750,810,13000]) #### Old values
        ptgen_edges = np.array([200,290,400,480,570,680,760,820,13000]) #### NEW VALUES TO SWITCH TO
        pt_bin = hist.axis.Variable(ptgen_edges, name="ptreco", label=r"p_{T,RECO} (GeV)")  
        pt_gen_bin = hist.axis.Variable(ptgen_edges, name="ptgen", label=r"p_{T,GEN} (GeV)")
        rho_bin = hist.axis.Regular(40, 0.0, 10.0, name="rhoreco", label=r"$-\log(\rho^2)$")
        ht_bin = hist.axis.Regular(300, 0.0, 13000.0, name="rhoreco", label=r"$-\log(\rho^2)$")
        rho_gen_bin = hist.axis.Regular(20, 0.0, 10.0, name="rhogen", label=r"$-\log(\rho^2)$")
        y_bin = hist.axis.Regular(25, -4.0, 4.0, name="rapidity", label=r"$y$")
        eta_bin = hist.axis.Regular(25, -4., 4., name="eta", label=r"$\eta$")
        frac_axis = hist.axis.Regular(10, 0.0, 1., name="frac", label="Fraction")
        n_axis = hist.axis.Regular(5, 0, 5, name="n", label=r"Number")
        dr_axis = hist.axis.Regular(150, 0, 6.0, name="dr", label=r"$\Delta R$")
        dphi_axis = hist.axis.Regular(150, -np.pi, np.pi, name="dphi", label=r"$\Delta \phi$")
        phi_axis = hist.axis.Regular(25, -np.pi, np.pi, name="phi", label=r"$\phi$")
        weight_bin = hist.axis.Regular(100, 0, 5, name="corrWeight", label="Weight")
        jk_axis = hist.axis.IntCategory([], growth = True, name = 'jk', label = "Jackknife section" )
        cutflow = {}
        self._histos = {
            #### For jackknife only need resp. matrix hists
                'misses_u':                    hist.Hist(dataset_axis, syst_cat, jk_axis, pt_gen_bin, mass_gen_bin, storage="weight", name="Events"),
                'misses_g':                  hist.Hist(dataset_axis, syst_cat, jk_axis, pt_gen_bin, mass_gen_bin, storage="weight", name="Events"),
                'fakes_u':                     hist.Hist(dataset_axis, syst_cat, jk_axis, pt_bin, mass_bin, storage="weight", name="Events"),
                'fakes_g':                   hist.Hist(dataset_axis, syst_cat, jk_axis, pt_bin, mass_bin, storage="weight", name="Events"),
                'underflow':                 hist.Hist(dataset_axis, syst_cat, jk_axis, pt_bin, mass_bin, storage="weight", name="Events"),
                'underflow_g':               hist.Hist(dataset_axis, syst_cat, jk_axis, pt_bin, mass_bin, storage="weight", name="Events"),
                #### hist for comparison of weights
                'weights':                   hist.Hist(dataset_axis, syst_cat, jk_axis, pt_bin, mass_bin, storage="weight", name="Events"),
                
                #### Plots to be unfolded/data only
                'ptreco_mreco_u':            hist.Hist(dataset_axis, syst_cat, jk_axis, pt_bin, mass_bin, storage="weight", name="Events"),
                'ptreco_mreco_g':            hist.Hist(dataset_axis, syst_cat, jk_axis, pt_bin, mass_bin, storage="weight", name="Events"),
                'rho_reco_u':            hist.Hist(dataset_axis, syst_cat, jk_axis, rho_bin, storage="weight", name="Events"),
                'rho_reco_g':            hist.Hist(dataset_axis, syst_cat, jk_axis, rho_bin, storage="weight", name="Events"),
        
                #### Plots for comparison
                'ptgen_mgen_u':              hist.Hist(dataset_axis, syst_cat, jk_axis, pt_gen_bin, mass_gen_bin, storage="weight", label="Counts"),       
                'ptgen_mgen_g':              hist.Hist(dataset_axis, syst_cat, jk_axis, pt_gen_bin, mass_gen_bin, storage="weight", label="Counts"),
            
                #### Plots for the analysis in the proper binning
                'response_rho_u':         hist.Hist(dataset_axis, syst_cat, jk_axis, rho_bin,  rho_gen_bin, storage="weight", label="Counts"),
                'response_rho_g':         hist.Hist(dataset_axis, syst_cat, jk_axis, rho_bin,  rho_gen_bin, storage="weight", label="Counts"),
                'response_matrix_u':         hist.Hist(dataset_axis, syst_cat, jk_axis, pt_bin, mass_bin, pt_gen_bin, mass_gen_bin, storage="weight",                                                         label="Counts"),
                'response_matrix_g':         hist.Hist(dataset_axis, syst_cat, jk_axis, pt_bin, mass_bin, pt_gen_bin, mass_gen_bin, storage="weight",                                                         label="Counts"),
                     
                #### misc.
                'cutflow':            cutflow,
                'jkflow':            processor.defaultdict_accumulator(int),
        }

        if not self.jk:
            self._histos.update({ 
            #### btag study histos
            'alljet_ptreco_mreco':          hist.Hist(dataset_axis, jet_cat, parton_cat, mass_bin, pt_bin, storage="weight", name="Events"),
            'btag_eta':                     hist.Hist(dataset_axis, jet_cat, parton_cat, frac_axis, eta_bin, storage="weight", name="Events"),
            'leading_pt':                   hist.Hist(dataset_axis, ht_bin, storage="weight", name="Events"),
            #### Plots of things during the selection process / for debugging
            'njet_gen':                     hist.Hist(dataset_axis, syst_cat, n_axis, storage="weight", label="Events"),
            'njet_reco':                    hist.Hist(dataset_axis, syst_cat, n_axis, storage="weight", label="Events"),
            'njet_reco':                    hist.Hist(dataset_axis, syst_cat, n_axis, storage="weight", label="Events"),
            'dphimin_gen':                  hist.Hist(dataset_axis, syst_cat, dphi_axis, storage="weight", label="Events"),
            'dphimin_reco':                 hist.Hist(dataset_axis, syst_cat, dphi_axis, storage="weight", label="Events"),
            'asymm_reco':                   hist.Hist(dataset_axis, syst_cat, pt_bin, frac_axis, storage="weight", label="Events"),
            'asymm_gen':                    hist.Hist(dataset_axis, syst_cat, pt_gen_bin, frac_axis, storage="weight", label="Events"),
            'mass_orig':                    hist.Hist(dataset_axis, jk_axis, pt_bin, mass_bin, storage="weight", label="Events"),
            'sdmass_orig':                  hist.Hist(dataset_axis, jk_axis, pt_bin, mass_bin, storage="weight", label="Events"),
            'sdmass_ak8corr':               hist.Hist(dataset_axis, jk_axis, pt_bin, mass_bin, storage="weight", label="Events"),
            'sdmass_ak4corr':               hist.Hist(dataset_axis, jk_axis, pt_bin, mass_bin, storage="weight", label="Events"),
            # 'jet_dr_reco_gen':            hist.Hist(syst_cat, dr_axis, storage="weight", label="Events"),
            # 'jet_eta_reco':               hist.Hist(syst_cat, eta_bin, storage="weight", name="Events"),
            'jet_rap_reco':                 hist.Hist(dataset_axis, syst_cat, y_bin, storage="weight", name="Events"),
            'jet_rap_gen':                  hist.Hist(dataset_axis, syst_cat, y_bin, storage="weight",name="Events"),
            #'jet_dr_gen':                  hist.Hist(dr_axis, storage="weight", label="Counts"),
            #'jet_dr_reco':                 hist.Hist(dr_axis, storage="weight", label="Counts"),
            'jet_phi_gen':                  hist.Hist(dataset_axis, syst_cat, phi_axis, storage="weight", label="Events"),
            'jet_phi_reco':                 hist.Hist(dataset_axis, syst_cat, phi_axis, storage="weight", label="Events"),
            'jet_eta_phi_precuts':          hist.Hist(dataset_axis, syst_cat, phi_axis, eta_bin, storage="weight", label="Counts"),
            'jet_eta_phi_preveto':          hist.Hist(dataset_axis, syst_cat, phi_axis, eta_bin, storage="weight", label="Counts"),
            'jet_pt_eta_phi':               hist.Hist(dataset_axis, syst_cat, pt_bin, phi_axis, eta_bin, storage="weight", label="Counts"),
            # 'jet_dr_gen_subjet':            hist.Hist(dataset_axis, syst_cat, dr_axis, storage="weight", label="Events"),
            # 'jet_dr_reco_to_gen_subjet':    hist.Hist(dataset_axis, syst_cat, dr_axis, storage="weight", label="Events"),
            #### for investigation of removing fakes
            'fakes_eta_phi':                hist.Hist(dataset_axis, syst_cat, eta_bin, phi_axis, storage="weight", name="Events"),
            'fakes_asymm_dphi':             hist.Hist(dataset_axis, syst_cat, frac_axis, dphi_axis, storage="weight", name="Events"),
            'ptreco_mreco_fine_u':           hist.Hist(dataset_axis,syst_cat, jk_axis, fine_pt_bin, fine_mass_bin, storage="weight", label="Counts"),
            'ptreco_mreco_fine_g':           hist.Hist(dataset_axis,syst_cat, jk_axis, fine_pt_bin, fine_mass_bin, storage="weight",  label="Counts"), 
        #### Plots to get JMR and JMS in MC
        # 'jet_m_pt_u_reco_over_gen': hist.Hist(syst_cat, pt_gen_bin, mass_gen_bin, frac_axis, storage="weight",                                                                        label="Events"),
        # 'jet_m_pt_g_reco_over_gen':  hist.Hist(syst_cat, pt_gen_bin, mass_gen_bin, frac_axis, storage="weight",                                                                        label="Events"),
        })
    
    @property
    def accumulator(self):
        return self._histos
        
    def process(self, events):
        out = self._histos
        dataset = events.metadata['dataset']
        filename = events.metadata['filename']
        if "madgraph" in dataset and "pythia" in dataset:
            mctype="pythiaMG"
        elif "herwig" in dataset:
            mctype='herwig'
        elif "_pythia" in dataset:
            mctype="pythia"
        else:
            mctype="data"  
        #####################################
        #### Find the IOV from the dataset name
        #####################################
        IOV = ('2016APV' if ( any(re.findall(r'APV',  dataset)) or any(re.findall(r'HIPM', dataset)))
               else '2018'    if ( any(re.findall(r'UL18', dataset)) or any(re.findall(r'UL2018',    dataset)))
               else '2017'    if ( any(re.findall(r'UL17', dataset)) or any(re.findall(r'UL2017',    dataset)))
               else '2016')
        #####################################
        #### Make loop for running 1/10 of dataset for jackknife
        #####################################
        datastr = mctype+IOV
        print(datastr)
        print("Filename: ", filename)
        print("Dataset: ", dataset)
        ####################################
        #### Inititalize cutflow table
        ###################################            
        
        out['cutflow'][datastr] = defaultdict(int)
        out['cutflow'][datastr]['nEvents initial'] += (len(events.FatJet))
        out['cutflow']['trigger_init'] = defaultdict(int)
        out['cutflow']['trigger_final'] = defaultdict(int)
        
        if (self.do_gen):
            firstidx = filename.find( "store/mc/" )
            fname2 = filename[firstidx:]
            fname_toks = fname2.split("/")
            # year = fname_toks[ fname_toks.index("mc") + 1]
            ht_bin = fname_toks[ fname_toks.index("mc") + 2]
            if "LHEWeight" in events.fields:
                weights = events["LHEWeight"].originalXWGTUP
            else:
                weights = events.genWeight
            out['cutflow'][datastr]['sumw for '+ht_bin] += np.sum(weights)
            ## Flag used for number of events

        
        index_list = np.arange(len(events))
        ###### Choose number of slices to break data into for jackknife method
        if self.jk:
            print("Self.jk ", self.jk)
            range_max = 10
        else: range_max=1
            
        if self.jk_range == None:
            jk_inds = range(0,range_max)
        else:
            jk_inds = range(self.jk_range[0], self.jk_range[1])
        for jk_index in jk_inds:
            print("Event indices ", index_list)
            if self.jk:
                print("Now doing jackknife {}".format(jk_index))
                print("Len of events before jk selection ", len(events))
            else:
                jk_index=-1
            print(index_list%range_max == jk_index)
            jk_sel = ak.where(index_list%range_max == jk_index, False, True)
            ######## Select portion for jackknife and ensure that all jets have a softdrop mass so sd mass correction does not fail
            events_jk = events[jk_sel]
            del jk_sel
            #### only consider pfmuons w/ similar selection to aritra for later jet isolation
            events_jk = ak.with_field(events_jk, 
                                      events_jk.Muon[(events_jk.Muon.mediumId > 0)
                                      &(np.abs(events_jk.Muon.eta) < 2.5)
                                      &(events_jk.Muon.pfIsoId > 1) ], 
                                      "Muon")
            FatJet=events_jk.FatJet
            FatJet["p4"] = ak.with_name(events_jk.FatJet[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
            print(FatJet)
            #### Make sure there is at least one jet to run over
            if len(FatJet) < 1:
                print("No fatjets")
                return out
            if self.do_gen:
                era = None
                GenJetAK8 = events_jk.GenJetAK8
                GenJetAK8['p4']= ak.with_name(events_jk.GenJetAK8[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
            else:
                firstidx = filename.find( "store/data/" )
                fname2 = filename[firstidx:]
                fname_toks = fname2.split("/")
                era = fname_toks[ fname_toks.index("data") + 1]
                print("IOV ", IOV, ", era ", era)
            print("starting jet corrections")
            corrected_fatjets = GetJetCorrections(FatJet, events_jk, era, IOV, isData=not self.do_gen)
            corrected_fatjets = corrected_fatjets[corrected_fatjets.subJetIdx1 > -1]
            #print(" Uncorrected subjet mass", events0.SubJet.mass)
            corrected_subjets = GetJetCorrections(events_jk.SubJet, events_jk, era, IOV, isData = not self.do_gen, mode = 'AK4')
            corrected_fatjets['msoftdrop'] =   (corrected_subjets[corrected_fatjets.subJetIdx1] + corrected_subjets[corrected_fatjets.subJetIdx2]).mass 
            print("finished correcting mass")
            if not self.jk:
                out["sdmass_orig"].fill(dataset=datastr, jk=jk_index, ptreco=events_jk[(ak.num(events_jk.FatJet) > 2)].FatJet[:,2].pt, mreco=events_jk[(ak.num(events_jk.FatJet) > 2)].FatJet[:,2].msoftdrop)
                out["sdmass_ak4corr"].fill(dataset=datastr, jk=jk_index, ptreco=corrected_fatjets[(ak.num(corrected_fatjets) > 2)][:,2].pt, mreco=corrected_fatjets[(ak.num(corrected_fatjets) > 2)][:,2].msoftdrop)
                # corrected_fatjets_ak8 = corrected_fatjets
                # corrected_fatjets_ak8 = GetCorrectedSDMass(corrected_fatjets, events_jk, era, IOV, isData=not self.do_gen, useSubjets = False)
                # out["sdmass_ak8corr"].fill(dataset=datastr, jk=jk_index, ptreco=corrected_fatjets_ak8[(ak.num(corrected_fatjets_ak8) > 2)][:,2].pt, mreco=corrected_fatjets_ak8[(ak.num(corrected_fatjets_ak8) > 2)][:,2].msoftdrop)
            self.weights = {}
            for jetsyst in self.jet_systematics:
                #####################################
                #### For each jet correction, we need to add JMR and JMS corrections on top (except if we're doing data).
                #####################################
                if jetsyst == 'nominal':
                    if not self.do_gen:
                        print("Doing nominal data")
                        corr_jets_final = deepcopy(corrected_fatjets)
                    else:
                        corr_jets_final = applyjmrSF(IOV, applyjmsSF(IOV,corrected_fatjets))
                elif jetsyst=="HEM" and self.do_gen:
                   corr_jets_final = HEMCleaning(IOV,applyjmrSF(IOV, applyjmsSF(IOV,corrected_fatjets)))
                elif 'JER' in jetsyst and self.do_gen:
                    if "Up" in jetsyst:
                        corr_jets_final  = applyjmrSF(IOV, applyjmsSF(IOV,corrected_fatjets.JER.up))
                        corr_jets_final['msoftdrop'] = (corrected_subjets.JER.up[corrected_fatjets.subJetIdx1] + corrected_subjets.JER.up[corrected_fatjets.subJetIdx2]).mass
                    else: 
                        corr_jets_final  = applyjmrSF(IOV, applyjmsSF(IOV,corrected_fatjets.JER.down))
                        corr_jets_final['msoftdrop'] = (corrected_subjets.JER.down[corrected_fatjets.subJetIdx1] + corrected_subjets.JER.down[corrected_fatjets.subJetIdx2]).mass
                elif "JMR" in jetsyst and self.do_gen:
                    if "Up" in jetsyst:
                        corr_jets_final  = applyjmrSF(IOV, applyjmsSF(IOV,corrected_fatjets), var = "up")
                    else: 
                        corr_jets_final  =  applyjmrSF(IOV, applyjmsSF(IOV,corrected_fatjets), var = "down")
                elif "JMS" in jetsyst and self.do_gen:
                    if "Up" in jetsyst:
                        corr_jets_final  = applyjmrSF(IOV, applyjmsSF(IOV,corrected_fatjets, var = "up"))
                    else:
                        corr_jets_final =  applyjmrSF(IOV, applyjmsSF(IOV,corrected_fatjets, var = "down"))
                elif "JES" in jetsyst and self.do_gen:
                    if jetsyst[-2:]=="Up":
                        field = jetsyst[:-2]
                        corr_jets_final =  applyjmrSF(IOV, applyjmsSF(IOV,corrected_fatjets[field].up))
                        corr_jets_final['msoftdrop'] = (corrected_subjets[field].up[corrected_fatjets.subJetIdx1] + corrected_subjets[field].up[corrected_fatjets.subJetIdx2]).mass
                    elif jetsyst[-4:]=="Down":
                        field = jetsyst[:-4]
                        corr_jets_final =  applyjmrSF(IOV, applyjmsSF(IOV,corrected_fatjets[field].down))
                        corr_jets_final['msoftdrop'] = (corrected_subjets[field].down[corrected_fatjets.subJetIdx1] + corrected_subjets[field].down[corrected_fatjets.subJetIdx2]).mass
                print(corr_jets_final)
                #################################################################
                #### sort corrected jets by pt before being put into events object
                #################################################################

                sortJets_ind = ak.argsort(corr_jets_final.pt, ascending=False)
                corr_jets_sorted = corr_jets_final[sortJets_ind]
                events_corr = ak.with_field(events_jk, corr_jets_sorted, "FatJet")  
                del corr_jets_sorted, corr_jets_final
                ###################################
                ######### INITIALIZE WEIGHTS AND SELECTION
                ##################################
                sel = PackedSelection()
                if (jetsyst == "nominal"): out['cutflow'][datastr]['nEvents initial'] += (len(events.FatJet))
                print("mctype ", mctype, " gen? ", self.do_gen)
                if self.do_gen and (mctype == "pythia"):
                    print("Doing XS scaling")
                    weights = events_corr.genWeight * getXSweight(dataset, IOV)
                elif self.do_gen:
                    if "LHEWeight" in events_corr.fields: 
                        #print("Difference between weights calculated from xsdb and LHE :", (events_corr.LHEWeight.originalXWGTUP - getXSweight(dataset, IOV)))
                        weights = events_corr.LHEWeight.originalXWGTUP
                    else:
                        weights = events_corr.genWeight * getXSweight(dataset, IOV)
                else:
                    ############
                    ### Doing data -- apply lumimask and require at least one jet to apply jet trigger prescales
                    ############
                    lumi_mask = getLumiMask(IOV)(events_corr.run, events_corr.luminosityBlock)
                    
                    events_corr = events_corr[lumi_mask]
                    weights = np.ones(len(events_corr))
                    if "ver2" in dataset:
                        trigsel, psweights, HLTflow_init, HLTflow_final = applyPrescales(events_corr, trigger= "PFJet", year = IOV)
                    else:
                        trigsel, psweights, HLTflow_init, HLTflow_final = applyPrescales(events_corr, year = IOV)
                    for path in HLTflow_init:
                        out['cutflow']['trigger_init'][path] += HLTflow_init[path]
                        out['cutflow']['trigger_final'][path] += HLTflow_final[path]
                    psweights=ak.where(ak.is_none(psweights), 1.0, psweights)
                    trigsel=ak.where(ak.is_none(trigsel), False, trigsel)
                    weights = ak.where(trigsel, psweights, weights)
                    sel.add("trigsel", trigsel)
                    if (jetsyst == "nominal"): 
                        out['cutflow'][datastr]['nEvents after trigger sel '] += (ak.sum(sel.all("trigsel")))
                        print("ADDED TRIGGER TO CUTFLOW FOR NOM FOR KEY ", dataset)
                #####################################
                #### Gen Jet Selection
                #################################### 
                if self.do_gen:
                    sel.add("npv", events_corr.PV.npvsGood > 0)
                else:
                    sel.add("npv", sel.all("trigsel") & (events_corr.PV.npvsGood > 0))
                if self.do_gen:
                    print("DOING GEN")
                    if not self.jk:
                        print("fat jet nones ", ak.sum(ak.is_none(events_corr[sel.all("npv")].FatJet)))
                        print("Njet nones ", ak.sum(ak.is_none(ak.num(events_corr[sel.all("npv")].FatJet))))
                        out["njet_gen"].fill(dataset=datastr, syst = jetsyst, n=ak.num(events_corr[sel.all("npv")].GenJetAK8), weight = weights[sel.all("npv")])
                    # pt_cut_gen = genjet.pt > 200. #### removing
                    sel.add("triGenJet", (ak.num(events_corr.GenJetAK8) > 2))
                    sel.add("triGenJet_seq", sel.all('npv', 'triGenJet') ) # & pt_cut_gen ) ####removing to be consistent w/ aritra
                    GenJetAK8 = events_corr.GenJetAK8
                    GenJetAK8['p4']= ak.with_name(events_corr.GenJetAK8[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
                    genjet = ak.firsts(GenJetAK8[:,2:])
                    rap_cut_gen = ak.where(sel.all("triGenJet_seq"), np.abs(getRapidity(genjet.p4)) < self.ycut, False)
                    sel.add("rapGen", rap_cut_gen)
                    if not self.jk:
                        out["jet_rap_gen"].fill(dataset=datastr, syst = jetsyst, rapidity=getRapidity(GenJetAK8[sel.all("triGenJet_seq")][:,2].p4), weight=weights[sel.all("triGenJet_seq")])
                        out["jet_phi_gen"].fill(dataset=datastr, syst=jetsyst, phi=GenJetAK8[sel.all("triGenJet_seq")][:,2].phi, weight=weights[sel.all("triGenJet_seq")])  
                    #### get dphi and pt asymm selections                     
                    genjet1 = ak.firsts(events_corr.GenJetAK8[:,0:])
                    genjet2 = ak.firsts(events_corr.GenJetAK8[:,1:])
                    genjet3 = ak.firsts(events_corr.GenJetAK8[:,2:])
                    ### calculate dphi_min
                    dphi12_gen = np.abs(genjet1.delta_phi(genjet2))
                    dphi13_gen = np.abs(genjet1.delta_phi(genjet3))
                    dphi23_gen = np.abs(genjet2.delta_phi(genjet3))
                    dphimin_gen = ak.min([dphi12_gen, dphi13_gen, dphi23_gen], axis = 0)
                    dphimin_gen_sel = ak.where(sel.all("triGenJet_seq"), dphimin_gen > 1.0, False)
                    asymm_gen  = np.abs(genjet1.pt - genjet2.pt)/(genjet1.pt + genjet2.pt)
                    sel.add("dphiGen", dphimin_gen_sel)
                    # genjet2["pt"] = ak.fill_none(genjet2.pt, 0.0001)
                    if not self.jk:
                        out["asymm_gen"].fill(dataset=datastr, syst=jetsyst, ptgen=events_corr[sel.all("triGenJet_seq")].GenJetAK8[:,2].pt, frac = asymm_gen[sel.all("triGenJet_seq")], weight=weights[sel.all("triGenJet_seq")])
                        out["dphimin_gen"].fill(dataset=datastr, syst=jetsyst, dphi = dphimin_gen[sel.all("triGenJet_seq")], weight = weights[sel.all("triGenJet_seq")])
                    gensubjets = events_corr.SubGenJetAK8
                    groomed_genjet = get_gen_sd_mass_jet(ak.firsts(GenJetAK8[:,2:]), gensubjets)
                    ##### move misses to after gen and reco sel
                    sel.add("genTot_seq", sel.all("triGenJet", "dphiGen", "rapGen"))
                    print("nevents total gen selection ", ak.sum(sel.all("genTot_seq")))
                    if (len(events_corr[sel.all("genTot_seq")]) < 1): 
                        print("No gen jets selected")
                        return out        
        
                #####################################
                #### Reco Jet Selection
                ####################################
            
                if not self.jk:
                    out["njet_reco"].fill(dataset=datastr, syst = jetsyst, n=ak.to_numpy(ak.num(events_corr[sel.all("npv")].FatJet), allow_missing=True), 
                                         weight = ak.to_numpy(weights[sel.all("npv")], allow_missing=True) )
                FatJet = events_corr.FatJet
                FatJet["p4"] = ak.with_name(events_corr.FatJet[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
                jet = ak.firsts(FatJet[:,2:])
                pt_cut_reco = jet.pt > 200.
                sel.add("triRecoJet", (ak.num(events_corr.FatJet) > 2))
                sel.add("triRecoJet_seq", sel.all('npv', 'triRecoJet') & pt_cut_reco )
                rap_cut = np.abs(getRapidity(jet.p4)) < self.ycut
                # sdm_cut = (ak.all(events_corr.FatJet.msoftdrop > 10., axis = -1))
                rap_sel = ak.where(sel.all("triRecoJet_seq"), rap_cut, False)
                sel.add("recoRap2p5", rap_sel)
                sel.add("recoRap_seq", sel.all("triRecoJet_seq", "recoRap2p5")) 
                print("nevents after rap cut ", ak.sum(sel.all("recoRap_seq")))
                if not self.jk:
                    out["jet_rap_reco"].fill(dataset=datastr, syst = jetsyst, rapidity=ak.to_numpy(getRapidity(FatJet[sel.all("triRecoJet_seq")][:,2].p4), allow_missing=True), weight=weights[sel.all("triRecoJet_seq")])
                    out["jet_phi_reco"].fill(dataset=datastr, syst=jetsyst, phi=FatJet[sel.all("triRecoJet_seq")][:,2].phi, weight=weights[sel.all("triRecoJet_seq")]) 
                #### Add cut on softdrop mass as done in previous two papers --> need to very with JMS/JMR studies
                #### ak.first fills empty values with none --> ak.singletons 
                jet1 = ak.firsts(events_corr.FatJet[:,0:])
                jet2 = ak.firsts(events_corr.FatJet[:,1:])
                jet3 = ak.firsts(events_corr.FatJet[:,2:])
                print("ak.first output ", jet3)
                dphi12 = np.abs(jet1.delta_phi(jet2))
                dphi13 = np.abs(jet1.delta_phi(jet3))
                dphi23 = np.abs(jet2.delta_phi(jet3))
                dphimin = ak.min([dphi12, dphi13, dphi23], axis = 0)
                dphi_sel = ak.where(sel.all("triRecoJet_seq"), (dphimin > 1.0), False)
                sel.add("recodphimin", dphi_sel)
                sel.add("recodphi_seq", sel.all("recodphimin", "recoRap_seq"))
                print("nevents after dphi cut ", ak.sum(sel.all("recodphi_seq")))
                asymm = np.abs(jet1.pt - jet2.pt)/(jet1.pt + jet2.pt)
                if not self.jk:
                    out["dphimin_reco"].fill(dataset=datastr, syst=jetsyst, dphi = dphimin[sel.all("triRecoJet_seq")], weight=weights[sel.all("triRecoJet_seq")])
                    out["asymm_reco"].fill(dataset=datastr, syst=jetsyst, ptreco=events_corr[sel.all("triRecoJet_seq")].FatJet[:,2].pt, frac = asymm[sel.all("triRecoJet_seq")], weight=weights[sel.all("triRecoJet_seq")])
                #### Check that nearest pfmuon and is at least dR > 0.4 away
                # Get the nearest muon to that jet
                print("ak singletons output ", ak.singletons(jet3))
                print("Nearest muons ", ak.singletons(jet3).nearest(events_corr.Muon))
                muon_sel =  ak.where(sel.all("triRecoJet_seq"), ak.all(jet3.delta_r(ak.singletons(jet3).nearest(events_corr.Muon))>0.4, axis=-1), False)
                sel.add("muonIso0p4", muon_sel)
                print("Number of events w/ jets w/o muon ", ak.sum(sel.require(triRecoJet_seq=True, muonIso0p4=True)))
                print("Number of evemts w/ jets w/ muon ", ak.sum(sel.require(triRecoJet_seq=True, muonIso0p4=False)))
                jetid_sel = ak.where(sel.all("triRecoJet_seq"), (jet3.jetId > 2), False)
                sel.add("jetId", jetid_sel)
                ####  Get Final RECO selection
                sel.add("recoTot_seq", sel.all("recodphi_seq", "jetId", "muonIso0p4") & ~ak.is_none(jet3.mass) & ~ak.is_none(jet3.msoftdrop))
                #### Check eta phi map pre cuts
                if not self.jk: 
                        out['jet_eta_phi_precuts'].fill(dataset=datastr, syst=jetsyst, phi=events_corr[sel.all("triRecoJet")].FatJet[:,2].phi, eta=events_corr[sel.all("triRecoJet")].FatJet[:,2].eta, weight=weights[sel.all("triRecoJet")])                
                if (len(events_corr[sel.all("recoTot_seq")]) < 1): 
                    print("no events passing reco sel")
                    return out 
                ################
                #### Find fakes, misses, and underflow and remove them to get final selection
                ###############
                
                if self.do_gen:
                    jet = ak.firsts(events_corr.FatJet[:,2:])
                    matches = genjet.delta_r(jet) < 0.4
                    misses = ~matches | sel.require(genTot_seq=True, recoTot_seq=False)
                    sel.add("misses", misses )
                    sel.add("removeMisses", ~misses )
                    sel.add("removeMisses_seq", sel.all("genTot_seq", "recoTot_seq", "removeMisses" ))
                    print("Number of misses ", ak.sum(misses))
                    miss_sel = misses & sel.all("genTot_seq")
                    print("Nevents after removing misses ", ak.sum(sel.all("recoTot_seq", "removeMisses")))
                    print("Number of misses w/ 2 jets ", ak.sum(miss_sel))
                    if ak.sum(miss_sel) > 0:
                        if jetsyst == "nominal": 
                            out['cutflow'][datastr]['misses'] += (len(events_corr[miss_sel].GenJetAK8))
                            print("NOMINAL ADDING TO CUTFLOW ", ak.sum(miss_sel))
                        print("Number of none missed jets ", ak.sum(ak.is_none(GenJetAK8[miss_sel][:,2])))
                        ###### Applying misses selection to gen jets and getting sd mass
                        genjet = ak.firsts(events_corr[miss_sel].GenJetAK8[:,2:])
                        groomed_genjet = get_gen_sd_mass_jet(genjet, events_corr[miss_sel].SubGenJetAK8)
                        miss_jets = events_corr[miss_sel].GenJetAK8[:,2]
                        miss_weights = weights[miss_sel]
                        print("Len of missed jets ", len(miss_jets), " and weights ", len(miss_weights))
                        out["misses_u"].fill(dataset=datastr, syst=jetsyst, jk=jk_index, ptgen = miss_jets[~ak.is_none(miss_jets.mass)].pt, mgen = miss_jets[~ak.is_none(miss_jets.mass)].mass, weight = miss_weights[~ak.is_none(miss_jets)])
                        out["misses_g"].fill(dataset=datastr, syst=jetsyst, jk=jk_index, ptgen = miss_jets[~ak.is_none(miss_jets.mass)].pt, mgen = groomed_genjet[~ak.is_none(miss_jets.mass)].mass, weight = miss_weights[~ak.is_none(miss_jets)])
                    if len(events_corr[sel.all("genTot_seq", "recoTot_seq", "removeMisses")])<1: 
                        print("No events after all selections and removing misses")
                        return out
                    #### Fakes include events missing a reco mass or sdmass value, events failing index dr matching, and events passing reco cut but failing the gen cut
                    print("Matched gen jets ", jet[~ak.is_none(jet)].matched_gen)
                    print("Number of nones (fakes) ", len(jet[~ak.is_none(jet)].matched_gen))
                    matches = ~ak.is_none(jet.matched_gen)
                    print("matched_gen ", jet.matched_gen)
                    fakes = ~matches | sel.require(genTot_seq=False, recoTot_seq=True)
                    matches = ak.where(sel.all("recoTot_seq"), ~fakes, False)
                    sel.add("fakes", fakes)
                    sel.add("removeFakes", matches)
                    sel.add("removeFakes_seq", sel.all("genTot_seq", "recoTot_seq", "removeFakes" ))
                    fake_sel = sel.all("recoTot_seq") & fakes
                    if len(weights[fake_sel])>0:
                        print("len of no nones ",ak.sum(ak.is_none(events_corr[fake_sel].FatJet[:,2])))
                        fake_jets = events_corr[fake_sel].FatJet[:,2]
                        print("Number of real fakes ", len(fake_jets))
                        fake_weights = Weights(len(weights[fake_sel]))
                        fake_weights.add('fakeWeight', weights[fake_sel])
                        print("Len of jets ", len(fake_jets), " and weights ", len(fake_weights.weight()))
                        # if "L1PreFiringWeight" in events_corr.fields:                
                        #     prefiringNom, prefiringUp, prefiringDown = GetL1PreFiringWeight(events_corr[fakes])
                        #     fake_weights.add("L1prefiring", weight=prefiringNom, weightUp=prefiringUp, weightDown=prefiringDown )
                        # puUp, puDown, puNom = GetPUSF(events_corr[fakes], IOV)
                        # fake_weights.add("PUSF", weight=puNom, weightUp=puUp,
                        #                        weightDown=puDown) 
                        # if 'herwig' in dataset or 'madgraph' in dataset:
                        #     pdfNom, pdfUp, pdfDown = GetPDFWeights(events_corr[fakes])
                        #     # print("Fakes pdf weights ", pdfNom, " shape ", len(pdfNom))
                        #     fake_weights.add("PDF", weight=pdfNom, weightUp=pdfUp,
                        #                        weightDown=pdfDown) 
                        #     q2Nom, q2Up, q2Down = GetQ2Weights(events_corr[fakes])
                        #     # print("Fakes q2 weights ", pdfNom, " shape ", len(pdfNom))
                        #     fake_weights.add("Q2", weight=q2Nom, weightUp=q2Up,
                        #                        weightDown=q2Down) 
                        out["fakes_u"].fill(dataset=datastr, syst=jetsyst, jk = jk_index, ptreco = fake_jets[~ak.is_none(fake_jets.mass)].pt, mreco = fake_jets[~ak.is_none(fake_jets.mass)].mass, weight = fake_weights.weight()[~ak.is_none(fake_jets.mass)])
                        out["fakes_g"].fill(dataset=datastr, syst=jetsyst, jk = jk_index, ptreco = fake_jets[~ak.is_none(fake_jets.msoftdrop)].pt, mreco = fake_jets[~ak.is_none(fake_jets.msoftdrop)].msoftdrop, weight = fake_weights.weight()[~ak.is_none(fake_jets.msoftdrop)])
                        if not self.jk:
                            out['fakes_eta_phi'].fill(dataset=datastr, syst=jetsyst, phi = fake_jets.phi[~ak.is_none(fake_jets.msoftdrop)], eta = fake_jets.eta[~ak.is_none(fake_jets.msoftdrop)], weight=fake_weights.weight()[~ak.is_none(fake_jets.msoftdrop)])
                    if (jetsyst == "nominal"): 
                        out['cutflow'][datastr]['fakes'] += (len(events_corr[fakes].FatJet))
                        print("ADDING FAKES TO CUTFLOW")
                    print("Number of events after all selections ", ak.sum(sel.all("genTot_seq", "recoTot_seq", "removeMisses", "removeFakes")))
                    if len(events_corr[sel.all("genTot_seq", "recoTot_seq", "removeMisses", "removeFakes")])<1: 
                        print("No events after all selections and removing fakes & misses")
                        return out
                    uf = ak.where(sel.require(recoTot_seq=True), (ak.firsts(events_corr.GenJetAK8[:,2:]).pt < 200.), False)
                    # sel.add("rem_uf_fakes", ~uf) #### accounting for uf fakes later now
                    print("# of uf fakes not caught by regular fakes ", ak.sum( (uf & ~fakes)))
                    uf_jets = events_corr[uf].FatJet[:,2]
                    uf_weights = weights[uf]
                    print("Lengths of underflow jets ", len(uf_jets), " length of underflow weights ", len(uf_weights))
                    out["underflow"].fill(dataset=datastr, syst=jetsyst, jk = jk_index, ptreco = uf_jets[~ak.is_none(uf_jets.mass)].pt, mreco = uf_jets[~ak.is_none(uf_jets.mass)].mass, weight = uf_weights[~ak.is_none(uf_jets.mass)])
                    out["underflow_g"].fill(dataset=datastr, syst=jetsyst, jk = jk_index, ptreco = uf_jets[~ak.is_none(uf_jets.mass)].pt, mreco = uf_jets[~ak.is_none(uf_jets.mass)].msoftdrop, weight = uf_weights[~ak.is_none(uf_jets.mass)])
                    sel.add("final_seq", sel.all("genTot_seq", "recoTot_seq", "removeMisses", "removeFakes"))
                else:
                    sel.add("final_seq", sel.all("recoTot_seq"))

                #######################
                #### Apply final selections and jet veto map
                #######################
                if len(events_corr[sel.all("final_seq")])<1:
                        print("no more events after separating underflow")
                        return out
                events_corr = events_corr[sel.all("final_seq")]
                weights = weights[sel.all("final_seq")]
                #### Check eta phi map after cuts but before jet veto
                #### Make eta phi plot to check effects of cuts
                if not self.jk: out['jet_eta_phi_preveto'].fill(dataset=datastr, syst=jetsyst, phi=events_corr.FatJet[:,2].phi, eta=events_corr.FatJet[:,2].eta, weight=weights)      
                
                #### Apply jet veto map
                # jet = events_corr.FatJet[:,2]
                # veto = ApplyVetoMap(IOV, jet, mapname='jetvetomap')
                # events_corr = events_corr[veto]
                # weights = weights[veto]
                # if len(events_corr)<1:
                #         print("no more events after jet veto")
                #         return out
                ####################################
                ### Apply HEM veto
                ####################################
                if IOV == '2018':
                    print("Doing hem")
                    hemveto = HEMVeto(events_corr.FatJet, events_corr.run)
                    events_corr = events_corr[hemveto]
                    weights = weights[hemveto]
                out['cutflow'][datastr]['HEMveto'] += (len(events_corr))  
                #######################
                #### Get final jets and weights and fill final plots
                #######################
                jet = events_corr.FatJet[:,2]
                #### Create coffea weights object and store initial weights 
                self.weights[jetsyst] = Weights(len(weights))
                self.weights[jetsyst].add('jetWeight', weight=weights)
                    
                ##################
                #### Apply final selections to GEN and fill any plots requiring gen, including resp. matrices
                ##################
                
                if self.do_gen:
                    genjet = events_corr.GenJetAK8[:,2]
                    groomed_genjet = get_gen_sd_mass_jet(genjet, events_corr.SubGenJetAK8)
                    weird_jets = events_corr[(events_corr.GenJetAK8[:,2].mass < 20.) & (events_corr.FatJet[:,2].mass >20.)]
                    print("Number of weird (mreco>20, mgen<20) jets ", len(weird_jets))
                    if jetsyst == "nominal": out['cutflow'][datastr]['nEvents weird (mreco>20, mgen<20) ungroomed'] += len(weird_jets)
                    #### Get L1 prefiring weights and uncertainties
                    if "L1PreFiringWeight" in events.fields:                
                        prefiringNom, prefiringUp, prefiringDown = GetL1PreFiringWeight(events_corr)
                        self.weights[jetsyst].add("L1prefiring", weight=prefiringNom, weightUp=prefiringUp, weightDown=prefiringDown)
                    #### Get pileup weights and uncertainties
                    puNom, puUp, puDown = GetPUSF(events_corr, IOV)
                    self.weights[jetsyst].add("PUSF", weight=puNom, weightUp=puUp, weightDown=puDown)
                    #### Get luminosity uncertainties (nominal lumi weight is set to 1.0)
                    lumiNom, lumiUp, lumiDown = GetLumiUnc(events_corr, IOV)
                    self.weights[jetsyst].add("Luminosity", weight=lumiNom, weightUp=lumiUp,
                                           weightDown=lumiDown) 
                    if 'herwig' in dataset or 'madgraph' in dataset:
                        pdfNom, pdfUp, pdfDown = GetPDFWeights(events_corr)
                        self.weights[jetsyst].add("PDF", weight=pdfNom, weightUp=pdfUp,
                                           weightDown=pdfDown) 
                        q2Nom, q2Up, q2Down = GetQ2Weights(events_corr)
                        self.weights[jetsyst].add("Q2", weight=q2Nom, weightUp=q2Up,
                                           weightDown=q2Down) 
                    out["ptgen_mgen_u"].fill(dataset=datastr, syst=jetsyst, jk=jk_index, ptgen=genjet.pt, mgen=genjet.mass, weight=self.weights[jetsyst].weight() )
                    out["ptgen_mgen_g"].fill(dataset=datastr, syst=jetsyst, jk=jk_index, ptgen=genjet.pt, mgen=groomed_genjet.mass, weight=self.weights[jetsyst].weight() )
                    #### Final plots
                    out["response_matrix_u"].fill(dataset=datastr, syst=jetsyst, jk=jk_index, ptreco=jet.pt, ptgen=genjet.pt, mreco=jet.mass, mgen=genjet.mass, weight = self.weights[jetsyst].weight())
                    out["response_matrix_g"].fill(dataset=datastr, syst=jetsyst, jk=jk_index, ptreco=jet.pt, ptgen=genjet.pt, mreco=jet.msoftdrop, mgen=groomed_genjet.mass, weight = self.weights[jetsyst].weight() )
                    out["response_rho_u"].fill(dataset=datastr, syst=jetsyst, jk=jk_index,  rhoreco=-np.log((jet.mass/jet.pt)**2), rhogen=-np.log((genjet.mass/genjet.pt)**2),
                                                  weight=self.weights[jetsyst].weight())
                    out["response_rho_g"].fill(dataset=datastr, syst=jetsyst, jk=jk_index,  rhoreco=-np.log((jet.msoftdrop/jet.pt)**2), rhogen=-np.log((genjet.pt/groomed_genjet.mass)**2),
                                                  weight=self.weights[jetsyst].weight())
                    out["ptreco_mreco_u"].fill(dataset=datastr, syst=jetsyst, jk=jk_index, ptreco=jet.pt, mreco=jet.mass, weight=self.weights[jetsyst].weight() )
                    out["ptreco_mreco_g"].fill(dataset=datastr, syst=jetsyst, jk=jk_index, ptreco=jet.pt, mreco=jet.msoftdrop, weight=self.weights[jetsyst].weight() )
                    #### Outliers
                    if not self.jk:
                        out["jet_pt_eta_phi"].fill(dataset=datastr, syst=jetsyst, ptreco=jet.pt, phi=jet.phi, eta=jet.eta, weight=self.weights[jetsyst].weight())
                    weights = self.weights[jetsyst].weight()
                    if jetsyst=="nominal":
                        for syst in self.weights[jetsyst].variations:
                            print("Weight variation: ", syst)
                            #fill nominal, up, and down variations for each
                            out['ptgen_mgen_u'].fill(dataset=datastr, syst=syst, jk=jk_index, ptgen=events_corr.GenJetAK8[:,2].pt, mgen=events_corr.GenJetAK8[:,2].mass,
                                                     weight=self.weights[jetsyst].weight(syst) )
                            out['ptgen_mgen_g'].fill(dataset=datastr, syst=syst, jk=jk_index, ptgen=events_corr.GenJetAK8[:,2].pt, mgen=groomed_genjet.mass, 
                                                          weight=self.weights[jetsyst].weight(syst) )           
                            out["response_matrix_u"].fill(dataset=datastr, syst=syst, jk=jk_index,ptreco=jet.pt, mreco=jet.mass, ptgen=events_corr.GenJetAK8[:,2].pt,
                                                          mgen=events_corr.GenJetAK8[:,2].mass, weight=self.weights[jetsyst].weight(syst))
                            out["response_matrix_g"].fill(dataset=datastr, syst=syst, jk=jk_index, ptreco=jet.pt, mreco=jet.msoftdrop,
                                                          ptgen=events_corr.GenJetAK8[:,2].pt,mgen=groomed_genjet.mass, weight=self.weights[jetsyst].weight(syst))
                            out["response_rho_u"].fill(dataset=datastr, syst=syst, jk=jk_index,  rhoreco=-np.log((jet.mass/jet.pt)**2), rhogen=-np.log((genjet.mass/genjet.pt)**2),
                                                  weight=self.weights[jetsyst].weight(syst))
                            out["response_rho_g"].fill(dataset=datastr, syst=syst, jk=jk_index,  rhoreco=-np.log((jet.msoftdrop/jet.pt)**2), rhogen=-np.log((genjet.pt/groomed_genjet.mass)**2),
                                                  weight=self.weights[jetsyst].weight(syst))
                            out["ptreco_mreco_u"].fill(dataset=datastr, syst=syst, jk=jk_index, ptreco=events_corr.FatJet[:,2].pt, mreco=events_corr.FatJet[:,2].mass,
                                                       weight=self.weights[jetsyst].weight(syst) )
                            out["ptreco_mreco_g"].fill(dataset=datastr, syst=syst, jk=jk_index, ptreco=events_corr.FatJet[:,2].pt, mreco=events_corr.FatJet[:,2].msoftdrop,
                                                       weight=self.weights[jetsyst].weight(syst) )
                            # if ak.sum(fakes)>0:
                            #     out["fakes"].fill(syst=syst, jk=jk_index, ptreco = fake_events.FatJet[:,2].pt, mreco = fake_events.FatJet[:,2].mass, weight=fake_weights.weight(syst))
                            #     out["fakes_g"].fill(syst=syst, jk=jk_index, ptreco = fake_events.FatJet[:,2].pt, mreco = fake_events.FatJet[:,2].msoftdrop, weight=fake_weights.weight(syst))
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
                        if not self.jk:
                            for flavor in jet1flav.keys():
                                for jetname, jetobj in jets.items():
                                    jetobj[flavor] = jetobj[flavor][~ak.is_none(jetobj[flavor])]
                                    out['alljet_ptreco_mreco'].fill(dataset=datastr, jetNumb = jetname, partonFlav = flavor, 
                                                                    mreco = jetobj[flavor].mass, 
                                                                    ptreco = jetobj[flavor].pt)
                                    out['btag_eta'].fill(dataset=datastr, jetNumb = jetname, partonFlav = flavor, 
                                                         frac = jetobj[flavor].btagDeepB, eta = jetobj[flavor].eta )
                        out['cutflow'][datastr]['nGluonJets'] += (len(jet3flav["Gluon"])+len(jet1flav["Gluon"])+len(jet2flav["Gluon"]))
                        out['cutflow'][datastr]['nJets'] += (len(events_corr.FatJet[:,0])+len(events_corr.FatJet[:,1])+len(events_corr.FatJet[:,2]))
                        out['cutflow'][datastr]['nSoftestGluonJets'] += (len(jet3flav["Gluon"]))
                        out['cutflow'][datastr]['nSoftestGluonJets_b'] += (len(jet3_jetb_flav["Gluon"]))
                        out['cutflow'][datastr]['nSoftestGluonJets_bb'] += (len(jet3_jetbb_flav["Gluon"]))
                        out['cutflow'][datastr]['nSoftestJets_b'] += (len(jet3_b))
                        out['cutflow'][datastr]['nSoftestJets_bb'] += (len(jet3_bb))
                        out['cutflow'][datastr]['n3Jets'] += (len(events_corr.FatJet[:,2].pt))

                ###############
                ##### If running over DATA fill only final reco plots
                ###############
                
                else:
                    out["ptreco_mreco_u"].fill(dataset=datastr, syst=jetsyst, jk=jk_index, ptreco=jet.pt, mreco=jet.mass, weight=self.weights[jetsyst].weight() )
                    out["ptreco_mreco_g"].fill(dataset=datastr, syst=jetsyst, jk=jk_index, ptreco=jet.pt, mreco=jet.msoftdrop, weight=self.weights[jetsyst].weight() )
                    out["rho_reco_u"].fill(dataset=datastr,syst=jetsyst, jk=jk_index, rhoreco=-np.log((jet.mass/jet.pt)**2), weight=self.weights[jetsyst].weight() )
                    out["rho_reco_g"].fill(dataset=datastr,syst=jetsyst, jk=jk_index, rhoreco=-np.log((jet.msoftdrop/jet.pt)**2), weight=self.weights[jetsyst].weight() )
                    if not self.jk:
                        out["jet_pt_eta_phi"].fill(dataset=datastr, syst=jetsyst, ptreco=jet.pt, phi=jet.phi, eta=jet.eta, weight=self.weights[jetsyst].weight())
                print("final jets ", jet)
                print("final jet pt ", jet.pt)
                if (jetsyst == "nominal"):
                    for name in sel.names:
                        out["cutflow"][datastr][name] += sel.all(name).sum()
                        print("ADDED ", name, " TO CUTFLOW")
                negMSD = jet.msoftdrop<0.
                print("Number of negative softdrop values ", ak.sum(negMSD) )
                if (jetsyst == "nominal"): 
                    out['cutflow'][datastr]['nEvents failing softdrop condition'] += ak.sum(negMSD)
                    print("ADDED NEG SD EVENTS TO CUTFLOW")
                del events_corr, weights
            del events_jk
        out['cutflow'][datastr]['chunks'] += 1
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

