 #### This file contains the processors for dijet and trijet hist selections. Plotting and resulting studies are in separate files.
#### LMH

import argparse

import awkward as ak
import numpy as np
import coffea
import os
import re
import pandas as pd

print(coffea.__version__)
from coffea import util, processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.analysis_tools import Weights, PackedSelection
from collections import defaultdict
from python.utils import *
from python.corrections import *
import hist
print(hist.__version__)

import time

#### currently only for MC --> makes hists and response matrix
class makeDijetHists(processor.ProcessorABC):
    '''
    Processor to run a dijet jet mass cross section analysis. 
    With "do_gen == True", will perform GEN selection and create response matrices. 
    Will always plot RECO level quantities. 
    '''
    def __init__(self, ptcut = 200., ycut = 2.5, data = False, jet_systematics = ['nominal', 'HEM'], systematics = ['L1PreFiringWeight', 'PUSF'], jk=False, jk_range = None):
        # should have separate **lower** ptcut for gen
        self.do_gen = not data
        self.ptcut = ptcut
        self.ycut = ycut #rapidity
        self.jk = jk
        self.jk_range = jk_range
        if self.jk:
            # protect against doing unc for jk --> only need nominal and memory intensive
            jet_systematics = ["nominal"]
            systematics = []
        self.jet_systematics = jet_systematics
        self.systematics = systematics
        print("Data: Gen: ", data, self.do_gen)
        jet_cat = hist.axis.StrCategory([], growth=True, name="jetNumb", label="Jet")
        syst_cat = hist.axis.StrCategory([], growth=True, name='syst', label="Systematic")
        parton_cat = hist.axis.StrCategory([], growth=True,name="partonFlav", label="Parton Flavour")
        dataset_axis = hist.axis.StrCategory([], growth=True, name="dataset", label="Primary dataset")
        fine_mass_bin = hist.axis.Regular(130, 0.0, 1300.0, name="mass", label=r"mass [GeV]")
        fine_pt_bin = hist.axis.Regular(500, 100.0, 10100.0, name="pt", label=r"$p_T$ [GeV]")
        mgen_bin_edges = np.array([0,5,10,20,40,60,80,100,150,200,300, 400, 500, 900,1300])
        mreco_bin_edges = np.sort(np.append(mgen_bin_edges,[(mgen_bin_edges[i]+mgen_bin_edges[i+1])/2 for i in range(len(mgen_bin_edges)-1)]))
        mass_gen_bin =  hist.axis.Variable(mgen_bin_edges, name="mgen", label=r"$m_{GEN}$ [GeV]")                         
        mass_bin = hist.axis.Variable(mreco_bin_edges, name="mreco", label=r"$m_{RECO}$ [GeV]")
        # ptgen_edges = np.array([200,260,350,430,540,630,690,750,810,13000]) #### OLD VALUES
        ptgen_edges = np.array([200,290,400,480,570,680,760,13000]) #### NEW VALUES TO SWITCH TO
        pt_bin = hist.axis.Variable(ptgen_edges, name="ptreco", label=r"$p_{T,RECO}$ [GeV]")     
        pt_gen_bin = hist.axis.Variable(ptgen_edges, name="ptgen", label=r"$p_{T,GEN}$ [GeV]") 
        y_bin = hist.axis.Regular(25, -4.0, 4.0, name="rapidity", label=r"$y$")
        rho_gen_bin = hist.axis.Regular(20, 0.0, 10.0, name="rhogen", label=r"$-\log(\rho^2)_{GEN}$")
        rho_bin = hist.axis.Regular(40, 0, 10.0, name="rhoreco", label=r"$-\log(\rho^2)$")
        eta_bin = hist.axis.Regular(25, -4.0, 4.0, name="eta", label=r"$\eta$")
        frac_axis = hist.axis.Regular(10, 0.0, 1., name="frac", label="Fraction")
        n_axis = hist.axis.Regular(5, 0, 5, name="n", label=r"Number")
        dr_axis = hist.axis.Regular(25, 0, 6.0, name="dr", label=r"$\Delta R$")
        phi_axis = hist.axis.Regular(25, -np.pi, np.pi, name="phi", label=r"$\phi$")
        dphi_axis = hist.axis.Regular(25, -np.pi, np.pi, name="dphi", label=r"$\Delta \phi$")
        weight_bin = hist.axis.Regular(100, 0, 5, name="corrWeight", label="Weight")
        jk_axis = hist.axis.IntCategory([], growth = True, name = 'jk', label = "Jackknife section" )
        cutflow = {}
        self._histos = {
            
                #### For jackknife only need resp. matrix hists
                'misses_u':                      hist.Hist(dataset_axis, syst_cat, jk_axis, pt_gen_bin, mass_gen_bin, storage="weight", name="Events"),
                'misses_g':                    hist.Hist(dataset_axis, syst_cat, jk_axis, pt_gen_bin, mass_gen_bin, storage="weight", name="Events"),
                'fakes_u':                       hist.Hist(dataset_axis, syst_cat, jk_axis, pt_bin, mass_bin, storage="weight", name="Events"),
                'fakes_g':                     hist.Hist(dataset_axis, syst_cat, jk_axis, pt_bin, mass_bin, storage="weight", name="Events"),
                'underflow':                   hist.Hist(dataset_axis, syst_cat, jk_axis, pt_bin, mass_bin, storage="weight", name="Events"),
                'underflow_g':                 hist.Hist(dataset_axis, syst_cat, jk_axis, pt_bin, mass_bin, storage="weight", name="Events"),
                #### hist for comparison of weights
                'weights':                     hist.Hist(dataset_axis,syst_cat, jk_axis, pt_bin, mass_bin, storage="weight", name="Events"),
                
                #### Plots to be unfolded
                'ptreco_mreco_u':              hist.Hist(dataset_axis,syst_cat, jk_axis, pt_bin, mass_bin, storage="weight", name="Events"),
                'ptreco_mreco_g':              hist.Hist(dataset_axis,syst_cat, jk_axis, pt_bin, mass_bin, storage="weight", name="Events"),
                'rho_reco_u':            hist.Hist(dataset_axis, syst_cat, jk_axis, pt_bin, rho_bin, storage="weight", name="Events"),
                'rho_reco_g':            hist.Hist(dataset_axis, syst_cat, jk_axis, pt_bin, rho_bin, storage="weight", name="Events"),
        
                #### Plots for comparison
                'ptgen_mgen_u':                hist.Hist(dataset_axis,syst_cat, jk_axis, pt_gen_bin, mass_gen_bin, storage="weight", label="Counts"),       
                'ptgen_mgen_g':                hist.Hist(dataset_axis,syst_cat, jk_axis, pt_gen_bin, mass_gen_bin, storage="weight", label="Counts"),
            
                #### Plots for the analysis in the proper binning
                'response_rho_u':         hist.Hist(dataset_axis, syst_cat, jk_axis, pt_bin, rho_bin,  pt_gen_bin, rho_gen_bin, storage="weight", label="Counts"),
                'response_rho_g':         hist.Hist(dataset_axis, syst_cat, jk_axis, pt_bin, rho_bin,  pt_gen_bin, rho_gen_bin, storage="weight", label="Counts"),
                'response_matrix_u':           hist.Hist(dataset_axis,syst_cat, jk_axis, pt_bin, mass_bin, pt_gen_bin, mass_gen_bin, storage="weight",                                                         label="Counts"),
                'response_matrix_g':           hist.Hist(dataset_axis,syst_cat, jk_axis, pt_bin, mass_bin, pt_gen_bin, mass_gen_bin, storage="weight",                                                         label="Counts"),
                     
                #### misc.
                'cutflow':            cutflow,
                'jkflow':            processor.defaultdict_accumulator(int),
            }

        if not self.jk:
            self._histos.update({            
            #### Old histos
            # 'jet_mass':             hist.Hist(jet_cat, parton_cat, mass_bin, storage="weight", name="Events"),
            # 'jet_rap':            hist.Hist(jet_cat, parton_cat, y_bin, storage="weight", name="Events"),
            # 'jet_eta':            hist.Hist(jet_cat, parton_cat, eta_bin, storage="weight", name="Events"),
            
            #### Plots of things during the selection process / for debugging
            'mass_orig':               hist.Hist(dataset_axis,jk_axis, fine_pt_bin, fine_mass_bin, storage="weight", label="Events"),
            'sdmass_orig':               hist.Hist(dataset_axis,jk_axis, fine_pt_bin, fine_mass_bin, storage="weight", label="Events"),
            'sdmass_ak8corr':            hist.Hist(dataset_axis,jk_axis, fine_pt_bin, fine_mass_bin, storage="weight", label="Events"),
            'sdmass_ak4corr':            hist.Hist(dataset_axis,jk_axis, fine_pt_bin, fine_mass_bin, storage="weight", label="Events"),
            'njet_reco':                 hist.Hist(dataset_axis,syst_cat, n_axis, storage="weight", label="Counts"),
            'njet_gen':                  hist.Hist(dataset_axis,syst_cat, n_axis, storage="weight", label="Counts"),
            #'jet_dr_reco_gen':           hist.Hist(dr_axis, storage="weight", label="Counts"),
            # 'eta_reco':              hist.Hist(syst_cat, eta_bin, storage="weight", name="Events"),
            # 'eta_gen':               hist.Hist(syst_cat, eta_bin, storage="weight",name="Events"),
            'jet_rap_reco':              hist.Hist(dataset_axis,syst_cat, y_bin, storage="weight", name="Events"),
            'jet_rap_gen':               hist.Hist(dataset_axis,syst_cat, y_bin, storage="weight",name="Events"),
            #'jet_dr_gen':                hist.Hist(dr_axis, storage="weight", label="Counts"),
            #'jet_dr_reco':               hist.Hist(dr_axis, storage="weight", label="Counts"),
            'jet_phi_gen':               hist.Hist(dataset_axis,syst_cat, phi_axis, storage="weight", label="Counts"),
            'jet_phi_reco':              hist.Hist(dataset_axis,syst_cat, phi_axis, storage="weight", label="Counts"),
            'jet_eta_phi_precuts':               hist.Hist(dataset_axis,syst_cat, phi_axis, eta_bin, storage="weight", label="Counts"),
            'jet_eta_phi_preveto':               hist.Hist(dataset_axis,syst_cat, phi_axis, eta_bin, storage="weight", label="Counts"),
            'jet_pt_eta_phi':               hist.Hist(dataset_axis,syst_cat, pt_bin, phi_axis, eta_bin, storage="weight", label="Counts"),
            'dphi_gen':                  hist.Hist(dataset_axis,syst_cat, dphi_axis, storage="weight", label="Counts"),
            'dphi_reco':                 hist.Hist(dataset_axis,syst_cat, dphi_axis, storage="weight", label="Counts"),
            'asymm_gen':                 hist.Hist(dataset_axis,syst_cat, pt_gen_bin, frac_axis, storage="weight", label="Counts"),
            'asymm_reco':                hist.Hist(dataset_axis,syst_cat, pt_bin, frac_axis, storage="weight", label="Counts"),
            'jet_dr_gen_subjet':         hist.Hist(dataset_axis,syst_cat, dr_axis, storage="weight", label="Counts"),
            'dijet_dr_reco_to_gen':      hist.Hist(dataset_axis,syst_cat, dr_axis, storage="weight", label="Counts"),
            'dr_reco_to_gen_subjet' :    hist.Hist(dataset_axis,syst_cat, dr_axis, storage="weight", label="Counts"),
            'ptreco_mreco_fine_u':           hist.Hist(dataset_axis,syst_cat, jk_axis, fine_pt_bin, fine_mass_bin, storage="weight", label="Counts"),
            'ptreco_mreco_fine_':           hist.Hist(dataset_axis,syst_cat, jk_axis, fine_pt_bin, fine_mass_bin, storage="weight",  label="Counts"), 
            #### for investigation of removing fakes
            'fakes_eta_phi':             hist.Hist(dataset_axis,syst_cat, eta_bin, phi_axis, storage="weight", name="Events"),
            'fakes_asymm_dphi':          hist.Hist(dataset_axis,syst_cat, frac_axis, dphi_axis, storage="weight", name="Events"),

            #### Plots to get JMR and JMS in MC
            # 'jet_m_pt_u_reco_over_gen':  hist.Hist(syst_cat, pt_gen_bin, mass_gen_bin, frac_axis, storage="weight",                                                                        label="Counts"),
            # 'jet_m_pt_g_reco_over_gen':  hist.Hist(syst_cat, pt_gen_bin, mass_gen_bin, frac_axis, storage="weight",                                                                        label="Counts"),
                             })
        ## This is for rejecting events with large weights
    
    @property
    def accumulator(self):
        return self._histos
        return self._table
    
    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        out = self._histos
        dataset = events.metadata['dataset']
        filename = events.metadata['filename']
        print("Filename: ", filename)
        print("Dataset: ", dataset)
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
        datastr = mctype+IOV
        print(datastr)
        out['cutflow'][dataset] = defaultdict(int)
        out['cutflow'][dataset]['nEvents initial ' + dataset] += (len(events.FatJet))
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
            out['cutflow'][dataset]['sumw for '+ht_bin] += np.sum(weights)
        index_list = np.arange(len(events))
        ###### Choose number of slices to break data into for jackknife method
        if self.jk:
            print("Self.jk ", self.jk)
            range_max = 10
        else: range_max=1
            
            #####################################
            #### Loop through JK slices
            #####################################
        if self.jk_range == None:
            jk_inds = range(0,range_max)
        else:
            jk_inds = range(self.jk_range[0], self.jk_range[1])
            print("Jk indices we'll loop over ", jk_inds)
        for jk_index in jk_inds:
            if self.jk:
                print("Now doing jackknife {}".format(jk_index))
                print("Len of events before jk selection ", len(events))
            else:
                jk_index=-1
            # print("range max ", range_max)
            jk_sel = ak.where(index_list%range_max == jk_index, False, True)
            #####################################
            #### Apply JK selection
            #####################################
            events_jk = events[jk_sel]
            # print("Len of events after jk selection ", len(events_jk))
            del jk_sel
            #### only consider pfmuons w/ similar selection to aritra for later jet isolation
            events_jk = ak.with_field(events_jk, 
                                      events_jk.Muon[(events_jk.Muon.mediumId > 0)
                                      &(np.abs(events_jk.Muon.eta) < 2.5)
                                      &(events_jk.Muon.pfIsoId > 1) ], 
                                      "Muon")
            #### require at least one fat jet and one subjet so corrections do not fail
            FatJet=events_jk.FatJet
            FatJet["p4"] = ak.with_name(events_jk.FatJet[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
            #if len(FatJet.pt) < 1:
            if ak.sum(ak.num(FatJet.pt)>0)<1:
                print("No fat jet pts at all")
                return out
            if self.do_gen:
                era = None
                GenJetAK8 = events_jk.GenJetAK8
                GenJetAK8['p4']= ak.with_name(events_jk.GenJetAK8[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
            else:
                firstidx = filename.find("store/data/")
                fname2 = filename[firstidx:]
                fname_toks = fname2.split("/")
                era = fname_toks[ fname_toks.index("data") + 1]
                # print("IOV, era ", IOV, era)
            #####################################
            #### Apply jet corrections
            #####################################
            corrected_fatjets = GetJetCorrections(FatJet, events_jk, era, IOV, isData=not self.do_gen)
            corrected_fatjets = GetCorrectedSDMass(corrected_fatjets, events_jk, era, IOV, isData=not self.do_gen)
            #####################################
            #### Fill plots to compare jet correction techniques
            #####################################
            if not self.jk:
                corrected_fatjets_ak8 = corrected_fatjets
                corrected_fatjets_ak8 = GetCorrectedSDMass(corrected_fatjets, events_jk, era, IOV, isData=not self.do_gen, useSubjets = False)
                out["sdmass_orig"].fill(dataset=datastr, jk=jk_index, pt=ak.flatten(events_jk[(ak.num(events_jk.FatJet) > 1)].FatJet[:,:2].pt, axis=1), mass=ak.flatten(events_jk[(ak.num(events_jk.FatJet) > 1)].FatJet[:,:2].msoftdrop, axis=1))
                out["mass_orig"].fill(dataset=datastr,jk=jk_index, pt=ak.flatten(events_jk[(ak.num(events_jk.FatJet) > 1)].FatJet[:,:2].pt, axis=1), mass=ak.flatten(events_jk[(ak.num(events_jk.FatJet) > 1)].FatJet[:,:2].mass, axis=1))
                out["sdmass_ak4corr"].fill(dataset=datastr, jk=jk_index, pt=ak.flatten(corrected_fatjets[(ak.num(corrected_fatjets) > 1)][:,:2].pt, axis=1), mass=ak.flatten(corrected_fatjets[(ak.num(corrected_fatjets) > 1)][:,:2].msoftdrop, axis=1))
                out["sdmass_ak8corr"].fill(dataset=datastr, jk=jk_index, pt=ak.flatten(corrected_fatjets_ak8[(ak.num(corrected_fatjets_ak8) > 1)][:,:2].pt, axis=1), mass=ak.flatten(corrected_fatjets_ak8[(ak.num(corrected_fatjets_ak8) > 1)][:,:2].msoftdrop, axis=1))
            jet_corrs = {}
            self.weights = {}
            print("successfully corrected jets")
            #####################################
            #### For each jet correction, we need to add JMR and JMS corrections on top (except if we're doing data).
            #####################################
            if 'HEM' in self.jet_systematics and self.do_gen:
                jet_corrs.update({
                           "HEM": HEMCleaning(IOV,applyjmsSF(IOV, applyjmrSF(IOV,corrected_fatjets)))
                          })
            if 'JER' in self.jet_systematics and self.do_gen and "JER" in corrected_fatjets.fields:
                corrected_fatjets.JER.up = applyjmsSF(IOV, applyjmrSF(IOV,corrected_fatjets.JER.up))
                corrected_fatjets.JER.down = applyjmsSF(IOV, applyjmrSF(IOV,corrected_fatjets.JER.down))
                jet_corrs.update({"JERUp": corrected_fatjets.JER.up,
                                    "JERDown": corrected_fatjets.JER.down
                                })
            if "JMR" in self.jet_systematics and self.do_gen:
                jet_corrs.update({"JMRUp": applyjmsSF(IOV, applyjmrSF(IOV,corrected_fatjets, var = "up")),
                                    "JMRDown": applyjmsSF(IOV, applyjmrSF(IOV,corrected_fatjets, var = "down"))})
            if "JMS" in self.jet_systematics and self.do_gen:
                jet_corrs.update({"JMSUp": applyjmsSF(IOV, applyjmrSF(IOV,corrected_fatjets), var = "up"),
                                    "JMSDown": applyjmsSF(IOV, applyjmrSF(IOV,corrected_fatjets), var = "down")})
            if 'nominal' in self.jet_systematics:
                if not self.do_gen:
                    print("Doing nominal data")
                else:
                    corrected_fatjets = applyjmsSF(IOV, applyjmrSF(IOV,corrected_fatjets))
                jet_corrs.update({"nominal": corrected_fatjets})
            if self.do_gen:
                # print("avail sys ", self.jet_systematics)
                avail_srcs = [unc_src for unc_src in self.jet_systematics if ("JES_"+unc_src in corrected_fatjets.fields)]
                # print("Input jet syst", self.jet_systematics)
                for unc_src in avail_srcs:
                    corrected_fatjets["JES_"+unc_src].up = applyjmsSF(IOV, applyjmrSF(IOV,corrected_fatjets["JES_"+unc_src].up))
                    corrected_fatjets["JES_"+unc_src].down = applyjmsSF(IOV, applyjmrSF(IOV,corrected_fatjets["JES_"+unc_src].down))
                    jet_corrs.update({
                        "JES_"+unc_src+"Up":corrected_fatjets["JES_"+unc_src].up,
                        "JES_"+unc_src+"Down":corrected_fatjets["JES_"+unc_src].down, })
                    
            #####################################
            #### Loop over each jet correction
            #####################################
            # print("Final jet corrs to run over: ", jet_corrs)
            for jetsyst in jet_corrs.keys():
                print("Doing analysis for corr ", jetsyst)
                events_corr = ak.with_field(events_jk, jet_corrs[jetsyst], "FatJet")
                out['cutflow'][dataset]['nEvents initial '+jetsyst] += (len(events_corr.FatJet))
                ###################################
                ######### INITIALIZE WEIGHTS AND SELECTION
                ##################################
                sel = PackedSelection()
                print("mctype ", mctype, " gen? ", self.do_gen)
                ###############
                #### For data: apply lumimask and require at least one jet to apply jet trigger prescales
                ##############
                if self.do_gen and (mctype == "pythia"):
                    # print("Doing XS scaling")
                    # print("weights ", weights)
                    # print("XS weights ", getXSweight(dataset, IOV))
                    weights = events_corr.genWeight * getXSweight(dataset, IOV)
                elif self.do_gen:
                    if "LHEWeight" in events_corr.fields: 
                        #print("Difference between weights calculated from xsdb and LHE :", (events_corr.LHEWeight.originalXWGTUP - getXSweight(dataset, IOV)))
                        weights = events_corr.LHEWeight.originalXWGTUP
                    else:
                        weights = events_corr.genWeight * getXSweight(dataset, IOV)
                ###############
                #### For data: apply lumimask to events and weights, get trigger prescaled weights, and save selection
                ##############
                else:
                    lumi_mask = getLumiMask(IOV)(events_corr.run, events_corr.luminosityBlock)
                    events_corr = events_corr[lumi_mask]
                    weights = np.ones(len(events_corr))
                    if "ver2" in dataset:
                        trigsel, psweights, HLT_cutflow_initial, HLT_cutflow_final = applyPrescales(events_corr, trigger= "PFJet", year = IOV)
                    else:
                        trigsel, psweights, HLT_cutflow_initial, HLT_cutflow_final = applyPrescales(events_corr, year = IOV)
                    #### adding trigger values to cutflow
                    for path in HLT_cutflow_initial:
                        out['cutflow'][dataset][path+" inital"] += HLT_cutflow_initial[path]
                        out['cutflow'][dataset][path+" final"] += HLT_cutflow_final[path]
                    if jetsyst == "nominal": out['cutflow'][dataset]['nEvents after good lumi sel'] += (len(events_corr.FatJet))
                    psweights=ak.where(ak.is_none(psweights), 1.0, psweights)
                    trigsel=ak.where(ak.is_none(trigsel), False, trigsel)
                    weights = ak.where(trigsel, psweights, weights)
                    sel.add("trigsel", trigsel)
                    if jetsyst == "nominal": out['cutflow'][dataset]['nEvents after trigger sel'] += (ak.sum(sel.all("trigsel")))
                if self.do_gen:
                    sel.add("npv", events_corr.PV.npvsGood > 0)
                else:
                    sel.add("npv", sel.all("trigsel") & (events_corr.PV.npvsGood > 0))
                    
                #####################################
                #### Begin GEN specific selections
                #### see CMS PAS SMP-20-010 for selections
                ####################################
                
                if self.do_gen:
                    print("DOING GEN")
                    #### Select events with at least 2 jets
                    if not self.jk:
                        out["njet_gen"].fill(dataset=datastr, syst = jetsyst, n=ak.num(events_corr[sel.all("npv")].GenJetAK8), 
                                         weight = weights[sel.all("npv")] )
                    #### pt_cut_gen = ak.all(events_corr.GenJetAK8[:,:2].pt > 200., axis = -1) ### 80% of reco pt cut --> for now removing
                    sel.add("twoGenJet", (ak.num(events_corr.GenJetAK8) > 1))
                    sel.add("twoGenJet_seq", sel.all('npv', 'twoGenJet')) # & pt_cut_gen ) ### --> for now removing
                    GenJetAK8 = events_corr.GenJetAK8
                    GenJetAK8['p4']= ak.with_name(events_corr.GenJetAK8[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
                    rap_cut_gen = ak.all(np.abs(getRapidity(GenJetAK8[:,:2].p4)) < self.ycut, axis = -1)
                    rap_sel = ak.where(sel.all("twoGenJet_seq"), rap_cut_gen, False)
                    sel.add("genRap2p5", rap_sel)
                    sel.add("genRap_seq", sel.all("twoGenJet_seq", "genRap2p5"))
                    if not self.jk:
                        out["jet_rap_gen"].fill(dataset=datastr, syst = jetsyst, rapidity=ak.flatten(getRapidity(GenJetAK8[sel.all("twoGenJet_seq")][:,:2].p4), axis=1), weight=np.repeat(weights[sel.all("twoGenJet_seq")], 2))
                        out["jet_phi_gen"].fill(dataset=datastr, syst=jetsyst, phi=ak.flatten(GenJetAK8[sel.all("twoGenJet_seq")][:,:2].phi, axis=1), weight=np.repeat(weights[sel.all("twoGenJet_seq")], 2))  
                    #### Apply kinematic and 2 jet requirement immediately so that dphi and asymm can be calculated
                    if jetsyst == "nominal": out['cutflow'][dataset]['nEvents after gen rapidity selection '] += (len(events_corr[sel.all("genRap_seq")].FatJet))
                    #### get dphi and pt asymm selections 
                    genjet1 = ak.firsts(events_corr.GenJetAK8[:,0:])
                    genjet2 = ak.firsts(events_corr.GenJetAK8[:,1:])
                    dphi12_gen = np.abs(genjet1.delta_phi(genjet2))
                    # print("Dphi ", dphi12_gen$\
                    dphi12_gen_sel = ak.where(sel.all("twoGenJet_seq"), dphi12_gen > 2., False)
                    sel.add("dphiGen2", dphi12_gen_sel)
                    # print("Asym num ", np.abs(genjet1.pt - genjet2.pt))
                    # print("Asym denom ", np.abs(genjet1.pt + genjet2.pt))
                    asymm_gen  = np.abs(genjet1.pt - genjet2.pt)/(genjet1.pt + genjet2.pt)
                    asymm_gen_sel = ak.where(sel.all("twoGenJet_seq"), asymm_gen < 0.3, False)
                    sel.add("genAsym0p3", asymm_gen_sel)
                    sel.add("genDphi_seq", sel.all("dphiGen2", "genRap_seq"))
                    sel.add("genTot_seq", sel.all("genRap_seq", "dphiGen2", "genAsym0p3") & ~ak.is_none(events_corr.GenJetAK8[:,:2].mass))
                    
                    #### N-1 plots
                    if not self.jk:
                        out["asymm_gen"].fill(dataset=datastr, syst=jetsyst,ptgen=events_corr[sel.all("twoGenJet_seq")].GenJetAK8[:,0].pt, frac=asymm_gen[sel.all("twoGenJet_seq")], weight=weights[sel.all("twoGenJet_seq")])  
                        out["dphi_gen"].fill(dataset=datastr, syst=jetsyst, dphi=dphi12_gen[sel.all("twoGenJet_seq")], weight=weights[sel.all("twoGenJet_seq")])

                #####################################
                #### Reco Jet Selection
                #################################### 
                #### Apply pt and rapidity cuts
                if not self.jk:
                    # print("Num none pts: ", ak.sum(ak.is_none(events_corr[sel.all("npv")].FatJet.pt)))
                    # print("Weights: ", ak.sum(ak.is_none(weights[sel.all("npv")])))
                    out["njet_reco"].fill(dataset=datastr, syst = jetsyst, n=ak.to_numpy(ak.num(events_corr[sel.all("npv")].FatJet), allow_missing=True), 
                                         weight = weights[sel.all("npv")])
                pt_cut_reco = ak.all(events_corr.FatJet[:,:2].pt > 200., axis = -1)
                # print("num reco jets ", ak.num(events_corr.FatJet))
                # print("num single or 0 reco jets ", ak.sum(ak.num(events_corr.FatJet) < 2))
                sel.add("twoRecoJet", (ak.num(events_corr.FatJet) > 1))
                sel.add("twoRecoJet_seq",  sel.all("npv", "twoRecoJet") & pt_cut_reco)
                # print("Nevents after 2 jets ", len(events_corr[sel.all("twoRecoJet_seq")]))
                FatJet = events_corr.FatJet
                FatJet["p4"] = ak.with_name(FatJet[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
                rap_cut_reco = ak.all(np.abs(getRapidity(FatJet[:,:2].p4)) < self.ycut, axis = -1)
                rap_sel = ak.where(sel.all("twoRecoJet_seq"), rap_cut_reco, False)
                sel.add("recoRap2p5", rap_sel)
                sel.add("recoRap_seq", sel.all("twoRecoJet_seq", "recoRap2p5")) 
                # print("Nevents after rap ", len(events_corr[sel.all("recoRap_seq")]))
                if not self.jk:
                    out["jet_rap_reco"].fill(dataset=datastr, syst = jetsyst, rapidity=ak.to_numpy(ak.flatten(getRapidity(FatJet[sel.all("twoRecoJet_seq")][:,:2].p4)), allow_missing=True),
                                             weight=np.repeat(weights[sel.all("twoRecoJet_seq")], 2))
                    out["jet_phi_reco"].fill(dataset=datastr, syst=jetsyst, phi=ak.flatten(FatJet[sel.all("twoRecoJet_seq")][:,:2].phi, axis=1), weight=np.repeat(weights[sel.all("twoRecoJet_seq")], 2)) 
                #### Add cut on softdrop mass as done in previous two papers --> need to verify with JMS/JMR studies
                # sdm_cut = (ak.all(events_corr_reco.FatJet.msoftdrop > 10., axis = -1))
                #### get dphi and pt asymm selections
                jet1 = ak.firsts(events_corr.FatJet[:,0:])
                jet2 = ak.firsts(events_corr.FatJet[:,1:])
                dphi12 = (np.abs(jet1.delta_phi(jet2)) > 2.)
                dphi12_sel = ak.where(sel.all("twoRecoJet_seq"), dphi12, False)
                sel.add("recodphi2", dphi12_sel)
                sel.add("recodphi_seq", sel.all("recodphi2", "recoRap_seq"))
                asymm = np.abs(jet1.pt - jet2.pt)/(jet1.pt + jet2.pt)
                if not self.jk:
                    out["dphi_reco"].fill(dataset=datastr, syst=jetsyst, dphi =dphi12[sel.all("twoRecoJet_seq")], weight=weights[sel.all("twoRecoJet_seq")])
                    out["asymm_reco"].fill(dataset=datastr, syst=jetsyst, ptreco=events_corr[sel.all("twoRecoJet_seq")].FatJet[:,0].pt, frac=asymm[sel.all("twoRecoJet_seq")], weight=weights[sel.all("twoRecoJet_seq")])
                asymm_reco_sel = ak.where(sel.all("twoRecoJet_seq"), asymm < 0.3, False)
                sel.add("recoAsym0p3", asymm_reco_sel)
                sel.add("recoAsym_seq", sel.all("recoAsym0p3", "recodphi_seq"))
                # print("Nevents after asym ", len(events_corr[sel.all("recoAsym_seq")]))
                #### Check that nearest pfmuon and is at least dR > 0.4 away
                muonIso = ak.all((events_corr.FatJet[:,:2].delta_r(events_corr.FatJet[:,:2].nearest(events_corr.Muon)) > 0.4), axis = -1)
                muon_sel = ak.where(sel.all("twoRecoJet_seq"), muonIso, False)
                sel.add("muonIso0p4", muon_sel)
                # print("Number of events w/ jets w/o muon ", ak.sum(sel.require(twoRecoJet_seq=True, muonIso0p4=True)))
                # print("Number of evemts w/ jets w/ muon ", ak.sum(sel.require(twoRecoJet_seq=True, muonIso0p4=False)))
                jetid_sel = ak.where(sel.all("twoRecoJet_seq"), ak.all(events_corr.FatJet[:,:2].jetId > 2, axis=-1), False)
                sel.add("jetId", jetid_sel)
                #### Fill eta phi map with pre cut reco values to check 
                if not self.jk: 
                        out['jet_eta_phi_precuts'].fill(dataset=datastr, syst=jetsyst, phi=ak.flatten(events_corr[sel.all("twoRecoJet")].FatJet[:,:2].phi, axis=-1), eta=ak.flatten(events_corr[sel.all("twoRecoJet")].FatJet[:,:2].eta, axis=-1), weight=np.repeat(weights[sel.all("twoRecoJet")], 2))  
                ####  Get Final RECO selection
                sel.add("recoTot_seq", sel.all("recoAsym_seq", "jetId", "muonIso0p4") & ~ak.is_none(events_corr.FatJet[:,:2].mass) & ~ak.is_none(events_corr.FatJet[:,:2].msoftdrop))
                if (len(events_corr[sel.all("recoTot_seq")]) < 1): 
                    print("no events passing reco sel")
                    return out
                
                ################
                #### Find fakes, misses, and underflow and remove them to get final selection
                ###############
                
                if self.do_gen:
                    matches = ak.all(events_corr.GenJetAK8[:,:2].delta_r(events_corr.GenJetAK8[:,:2].nearest(events_corr.FatJet)) < 0.4, axis = -1)
                    # print("Number of matches ", ak.sum(matches))
                    ################
                    #### Misses include events missing a gen mass, events failing DR matching, and events passing gen cut but failing the reco cut
                    ################
                    misses = ~matches | sel.require(genTot_seq=True, recoTot_seq=False)
                    sel.add("misses", misses )
                    # print("Number of misses from gen pass reco fail ", ak.sum(sel.require(genTot_seq=True, recoTot_seq=False)))
                    # print("Number of misses from dr ", ak.sum(~matches))
                    # print("Number of misses from dr while requring 2 gen jets", ak.sum(~matches & sel.all("genTot_seq")))
                    sel.add("removeMisses", ~misses )
                    sel.add("removeMisses_seq", sel.all("genTot_seq", "recoTot_seq", "removeMisses" ))
                    # print("GenJet nones ", ak.sum(ak.any(ak.is_none(GenJetAK8[:,:2].mass, axis = -1), axis=-1)))
                    miss_sel = misses & sel.all("genTot_seq")
                    # print("Nevents after removing misses ", ak.sum(sel.all("recoTot_seq", "removeMisses")))
                    # print("Number of misses ", ak.sum(miss_sel))
                    if len(weights[miss_sel])>0:
                        if jetsyst == "nominal": out['cutflow'][dataset]['misses_u'] += (len(events_corr[miss_sel].GenJetAK8))
                        # print("Number of missed jets ", ak.sum(miss_sel))
                        # print("Number of none missed jets ", ak.sum(ak.any(ak.is_none(GenJetAK8[miss_sel][:,:2], axis = -1), axis=-1)))
                        ###### Applying misses selection to gen jets and getting sd mass
                        genjet1 = ak.firsts(events_corr[miss_sel].GenJetAK8[:,0:])
                        genjet2 = ak.firsts(events_corr[miss_sel].GenJetAK8[:,1:])
                        groomed_genjet0 = get_gen_sd_mass_jet(genjet1, events_corr[miss_sel].SubGenJetAK8)
                        groomed_genjet1 = get_gen_sd_mass_jet(genjet2, events_corr[miss_sel].SubGenJetAK8)
                        groomed_gen_dijet = ak.concatenate([ak.unflatten(groomed_genjet0, 1),  ak.unflatten(groomed_genjet1, 1)], axis=1)
                        groomed_gen_dijet = ak.flatten(groomed_gen_dijet, axis=1)
                        miss_dijets = ak.flatten(events_corr[miss_sel].GenJetAK8[:,:2], axis=1)
                        miss_weights = np.repeat(weights[miss_sel], 2)
                        # print("Len of missed dijets ", len(miss_dijets), " and weights ", len(miss_weights))
                        out["misses_u"].fill(dataset=datastr, syst=jetsyst, jk=jk_index, ptgen = miss_dijets.pt, mgen = miss_dijets.mass, weight = miss_weights)
                        out["misses_g"].fill(dataset=datastr, syst=jetsyst, jk=jk_index, ptgen = miss_dijets.pt, mgen = groomed_gen_dijet.mass, weight = miss_weights)
                    if len(events_corr[sel.all("genTot_seq", "recoTot_seq", "removeMisses")])<1: 
                        print("No events after all selections and removing misses")
                        return out
                    #### Fakes include events missing a reco mass or sdmass value, events failing index dr matching, and events passing reco cut but failing the gen cut
                    matches = ~ak.any(ak.is_none(events_corr.FatJet[:,:2].matched_gen.pt, axis=-1), axis=1)
                    # print("Matches ", matches)
                    # print("Number of nones (fakes) ", ak.sum(~matches))
                    # print("matched_gen nones ", ak.sum(ak.is_none(events_corr.FatJet[:,:2].matched_gen.pt, axis=-1)))
                    fakes = ~matches | sel.require(genTot_seq=False, recoTot_seq=True)
                    matches = ak.where(sel.all("recoTot_seq"), ~fakes, False)
                    # print("Number of fakes ", ak.sum(fakes))
                    sel.add("removeFakes", matches)
                    sel.add("removeFakes_seq", sel.all("genTot_seq", "recoTot_seq", "removeFakes" ))
                    fakes = ak.where(sel.all("recoTot_seq"), fakes, False)
                    sel.add("fakes", fakes)
                    if len(weights[fakes])>0:
                        # print("len of no nones ",ak.sum(ak.is_none(events_corr.FatJet[:,:2])))
                        fake_dijets = ak.flatten(events_corr[fakes].FatJet[:,:2], axis=1)
                        fake_weights = Weights(len(np.repeat(weights[fakes], 2)))
                        fake_weights.add('fakeWeight', np.repeat(weights[fakes], 2))
                        # print("Len of flattened diejts ", len(fake_dijets), " and weights ", len(fake_weights.weight()))
                        # if "L1PreFiringWeight" in events_corr.fields and "L1PreFiringWeight" in self.systematics:                
                        #     prefiringNom, prefiringUp, prefiringDown = GetL1PreFiringWeight(events_corr[fakes])
                        #     fake_weights.add("L1prefiring", weight=np.repeat(prefiringNom, 2), 
                        #                            weightUp=np.repeat(prefiringUp, 2), 
                        #                            weightDown=np.repeat(prefiringDown, 2),
                        #                )
                        # if "PUSF" in self.systematics:
                        #     puUp, puDown, puNom = GetPUSF(events_corr[fakes], IOV)
                        #     fake_weights.add("PUSF", weight=np.repeat(puNom, 2), weightUp=np.repeat(puUp, 2),
                        #                        weightDown=np.repeat(puDown, 2),) 
                        # if 'herwig' in dataset or 'madgraph' in dataset:
                        #     pdfNom, pdfUp, pdfDown = GetPDFWeights(events_corr[fakes])
                        #     # print("Fakes pdf weights ", pdfNom, " shape ", len(pdfNom))
                        #     fake_weights.add("PDF", weight=np.repeat(pdfNom, 2), weightUp=np.repeat(pdfUp, 2),
                        #                        weightDown=np.repeat(pdfDown, 2),) 
                        #     q2Nom, q2Up, q2Down = GetQ2Weights(events_corr[fakes])
                        #     # print("Fakes q2 weights ", pdfNom, " shape ", len(pdfNom))
                        #     fake_weights.add("Q2", weight=np.repeat(q2Nom, 2), weightUp=np.repeat(q2Up, 2),
                        #                        weightDown=np.repeat(q2Down, 2),) 

                        out["fakes_u"].fill(dataset=datastr, syst=jetsyst, jk = jk_index, ptreco = fake_dijets[~ak.is_none(fake_dijets.mass)].pt, mreco = fake_dijets[~ak.is_none(fake_dijets.mass)].mass, weight = fake_weights.weight()[~ak.is_none(fake_dijets.mass)])
                        out["fakes_g"].fill(dataset=datastr, syst=jetsyst, jk = jk_index, ptreco = fake_dijets[~ak.is_none(fake_dijets.msoftdrop)].pt, mreco = fake_dijets[~ak.is_none(fake_dijets.msoftdrop)].msoftdrop, weight = fake_weights.weight()[~ak.is_none(fake_dijets.msoftdrop)])
                        if not self.jk:
                            out['fakes_eta_phi'].fill(dataset=datastr, syst=jetsyst, phi = fake_dijets.phi[~ak.is_none(fake_dijets.msoftdrop)], eta = fake_dijets.eta[~ak.is_none(fake_dijets.msoftdrop)], weight=fake_weights.weight()[~ak.is_none(fake_dijets.msoftdrop)])
                    if jetsyst == "nominal": out['cutflow'][dataset]['fakes_u'] += (len(events_corr[fakes].FatJet))
                    if len(events_corr[sel.all("genTot_seq", "recoTot_seq", "removeMisses", "removeFakes")])<1: 
                        print("No events after all selections and removing fakes & misses")
                        return out
                    uf = ak.where(sel.require(recoTot_seq=True), ak.any(events_corr.GenJetAK8[:,:2].pt < 200., axis = -1), False)
                    #### sel.add("rem_uf_fakes", ~uf) #### not removing uf fakes anymore to account for them
                    # print("# of uf fakes not caught by regular fakes ", ak.sum( (uf & ~fakes)))
                    uf_dijets = ak.flatten(events_corr[uf].FatJet[:,:2], axis=1)
                    uf_weights = np.repeat(weights[uf], 2)
                    # print("Lengths of underflow dijets ", len(uf_dijets), " length of underflow weights ", len(uf_weights))
                    out["underflow"].fill(dataset=datastr, syst=jetsyst, jk = jk_index, ptreco = uf_dijets[~ak.is_none(uf_dijets.mass)].pt, mreco = uf_dijets[~ak.is_none(uf_dijets.mass)].mass, weight = uf_weights[~ak.is_none(uf_dijets.mass)])
                    out["underflow_g"].fill(dataset=datastr, syst=jetsyst, jk = jk_index, ptreco = uf_dijets[~ak.is_none(uf_dijets.mass)].pt, mreco = uf_dijets[~ak.is_none(uf_dijets.mass)].msoftdrop, weight = uf_weights[~ak.is_none(uf_dijets.mass)])
                    #######################
                    #### Make final selection
                    #######################
                    sel.add("final_seq", sel.all("genTot_seq", "recoTot_seq", "removeMisses", "removeFakes"))
                    #######################
                    #### Check jec's with selection
                    #######################
                    print("Checking consistency of correction vals after selection for ", jetsyst, "for reco masses < 50 and pt < 290")
                    #jets of interest
                    joi = (ak.all(events_corr.FatJet[:,:2].mass < 50., axis=-1) & ak.all(events_corr.FatJet[:,:2].pt <290., axis=-1))

                    d = {
                        "reco_pt_leadingJERC" : events_corr[sel.all("final_seq") &joi].FatJet[:8,0].pt,
                        "reco_pt_leadingJEC" : events_corr[sel.all("final_seq") &joi].FatJet[:8,0].pt_jec,
                        "reco_pt_leadingAbsPFUp_pt" : events_corr[sel.all("final_seq") &joi].FatJet[:8,0].JES_AbsoluteMPFBias.up.pt,
                        "reco_pt_leadingAbsPFDown_pt" : events_corr[sel.all("final_seq") &joi].FatJet[:8,0].JES_AbsoluteMPFBias.down.pt,
                        "gen_pt_leading" : events_corr[sel.all("final_seq")&joi].GenJetAK8[:8,0].pt,
                        "reco_ptRAW_leading": events_jk[sel.all("final_seq")&joi].FatJet[:8,0].pt,
                        "raw/jec": events_jk[sel.all("final_seq")&joi].FatJet[:8,0].pt/events_corr[sel.all("final_seq") &joi].FatJet[:8,0].pt_jec,
                        "gen/reco": events_corr[sel.all("final_seq")&joi].GenJetAK8[:8,0].pt/events_corr[sel.all("final_seq")&joi].FatJet[:8,0].pt,
                        "reco_rho_leading": np.log((events_corr[sel.all("final_seq")&joi].FatJet[:8,0].mass/events_corr[sel.all("final_seq")&joi].FatJet[:8,0].pt)**2),
                        "reco_eta_leading" : events_corr[sel.all("final_seq")&joi].FatJet[:8,0].eta,
                        "reco_phi_leading": events_corr[sel.all("final_seq")&joi].FatJet[:8,0].phi,
                        "reco_pt_subleading" : events_corr[sel.all("final_seq")&joi].FatJet[:8,1].pt,
                        "gen_pt_subleading" : events_corr[sel.all("final_seq")&joi].GenJetAK8[:8,1].pt,
                        "reco_ptRAW_subleading" : events_jk[sel.all("final_seq")&joi].FatJet[:8,1].pt,
                        "reco_rho_subleading" : np.log((events_corr[sel.all("final_seq")&joi].FatJet[:8,1].mass/events_corr[sel.all("final_seq")&joi].FatJet[:8,1].pt)**2),
                        "reco_eta_subleading" : events_corr[sel.all("final_seq")&joi].FatJet[:8,1].eta,
                        "reco_phi_subleading" : events_corr[sel.all("final_seq")&joi].FatJet[:8,1].phi, }
                    pd.set_option('display.max_rows', None)
                    df = pd.DataFrame.from_dict(d, orient="index")
                    print(df.to_string())
                    
                else:
                    sel.add("final_seq", sel.all("recoTot_seq"))
                    
                #######################
                #### Apply final selections and jet veto maps
                #######################
                if len(events_corr[sel.all("final_seq")])<1:
                        print("no more events after final selection")
                        return out
                events_corr = events_corr[sel.all("final_seq")]
                weights = weights[sel.all("final_seq")]
                #### Make eta phi plot to check effects of cuts
                dijet = ak.flatten(events_corr.FatJet[:,:2], axis=1)
                dijet_weights = np.repeat(weights, 2)
                if not self.jk: out['jet_eta_phi_preveto'].fill(dataset=datastr, syst=jetsyst, phi=dijet.phi, eta=dijet.eta, weight=dijet_weights)      
                #### Apply jet veto map
                # jet1 = events_corr.FatJet[:,0]
                # jet2 = events_corr.FatJet[:,1]
                # veto = ApplyVetoMap(IOV, jet1, mapname='jetvetomap') & ApplyVetoMap(IOV, jet2, mapname='jetvetomap')
                # if len(events_corr[veto])<1:
                #         print("no more events after jet veto")
                #         return out
                # events_corr = events_corr[veto]
                # weights = weights[veto]
                ####################################
                ### Apply HEM veto
                ####################################
                if IOV == '2018':
                    print("Doing hem")
                    hemveto = HEMVeto(events_corr.FatJet, events_corr.run)
                    events_corr = events_corr[hemveto]
                    weights = weights[hemveto]
                out['cutflow'][dataset]['HEMveto'] += (len(events_corr))  
                #######################
                #### Get dijets and weights and fill final plots
                #######################
                
                dijet = ak.flatten(events_corr.FatJet[:,:2], axis=1)
                dijet_weights = np.repeat(weights, 2)
                #### Create coffea weights object
                self.weights[jetsyst] = Weights(len(dijet_weights))
                self.weights[jetsyst].add('dijetWeight', weight=dijet_weights)
                negMSD = ak.flatten(events_corr.FatJet[:,:2].msoftdrop<0, axis=1)
                print("Number of negative softdrop values ", ak.sum(negMSD))
                print("Number of final jets ", len(dijet))
                if jetsyst == "nominal": out['cutflow'][dataset]['nEvents failing softdrop condition'] += ak.sum(negMSD)
                
                ##################
                #### Apply final selections to GEN and fill any plots requiring gen, including resp. matrices
                ##################
                
                if self.do_gen:
                    gen_dijet = ak.flatten(events_corr.GenJetAK8[:,:2], axis=1)
                    genjet1 =  events_corr.GenJetAK8[:,0]
                    genjet2 = events_corr.GenJetAK8[:,1]
                    groomed_genjet0 = get_gen_sd_mass_jet(genjet1, events_corr.SubGenJetAK8)
                    groomed_genjet1 = get_gen_sd_mass_jet(genjet2, events_corr.SubGenJetAK8)
                    groomed_gen_dijet = ak.concatenate([ak.unflatten(groomed_genjet0, 1),  ak.unflatten(groomed_genjet1, 1)], axis=1)
                    groomed_gen_dijet = ak.flatten(groomed_gen_dijet, axis=1)
                    weird_dijets = events_corr[ak.all(events_corr.GenJetAK8[:,:2].mass < 20., axis=-1) & ak.all(events_corr.FatJet[:,:2].mass >20., axis=-1)]
                    # print("Number of weird dijets ", len(weird_dijets))
                    if jetsyst == "nominal": out['cutflow'][dataset]['nEvents weird (mreco>20, mgen<20) ungroomed events'] += len(weird_dijets)

                    if "L1PreFiringWeight" in events_corr.fields:                
                        prefiringNom, prefiringUp, prefiringDown = GetL1PreFiringWeight(events_corr)
                        self.weights[jetsyst].add("L1prefiring", weight=np.repeat(prefiringNom, 2), 
                                               weightUp=np.repeat(prefiringUp, 2), 
                                               weightDown=np.repeat(prefiringDown, 2),
                                   )
                    #### Apply Pileup reweighting and get up and down uncertainties
                    puNom, puUp, puDown = GetPUSF(events_corr, IOV)
                    self.weights[jetsyst].add("PUSF", weight=np.repeat(puNom, 2), weightUp=np.repeat(puUp, 2),
                                           weightDown=np.repeat(puDown, 2),) 
                    #### Get luminosity uncertainties (nominal weight is 1.0)
                    lumiNom, lumiUp, lumiDown = GetLumiUnc(events_corr, IOV)
                    self.weights[jetsyst].add("Luminosity", weight=np.repeat(lumiNom, 2), weightUp=np.repeat(lumiUp, 2),
                                           weightDown=np.repeat(lumiDown, 2),) 
                    if 'herwig' in dataset or 'madgraph' in dataset:
                        pdfNom, pdfUp, pdfDown = GetPDFWeights(events_corr)
                        self.weights[jetsyst].add("PDF", weight=np.repeat(pdfNom, 2), weightUp=np.repeat(pdfUp, 2),
                                           weightDown=np.repeat(pdfDown, 2),) 
                        q2Nom, q2Up, q2Down = GetQ2Weights(events_corr)
                        self.weights[jetsyst].add("Q2", weight=np.repeat(q2Nom, 2), weightUp=np.repeat(q2Up, 2),
                                           weightDown=np.repeat(q2Down, 2),) 
                    if not self.jk:
                        # out["jet_dr_gen_subjet"].fill(syst=jetsyst, 
                        #                     dr=events_corr.SubGenJetAK8[:,0].delta_r(events_corr.FatJet[:,0]),
                        #                           weight=self.weights[jetsyst].weight())
                        # print("Len of dijet phi ", len(dijet.phi), " len of rap ", len(ak.to_numpy(ak.flatten(getRapidity(events_corr.FatJet[:,:2].p4)))))
                        out["jet_pt_eta_phi"].fill(dataset=datastr, syst=jetsyst, ptreco=dijet.pt, phi=dijet.phi, eta=dijet.eta, weight=self.weights[jetsyst].weight())
                    #### Final GEN plots
                    out['ptgen_mgen_u'].fill(dataset=datastr, syst=jetsyst, jk=jk_index, ptgen=gen_dijet.pt, mgen=gen_dijet.mass, weight=self.weights[jetsyst].weight() )
                    out['ptgen_mgen_g'].fill(dataset=datastr, syst=jetsyst, jk=jk_index, ptgen=gen_dijet.pt, mgen=groomed_gen_dijet.mass, weight=self.weights[jetsyst].weight() )
                    out["response_matrix_u"].fill(dataset=datastr, syst=jetsyst, jk=jk_index,  ptreco=dijet.pt, mreco=dijet.mass,
                                                  ptgen=gen_dijet.pt, mgen=gen_dijet.mass,
                                                  weight=self.weights[jetsyst].weight())
                    out["response_matrix_g"].fill(dataset=datastr, syst=jetsyst, jk=jk_index,ptreco=dijet.pt, mreco=dijet.msoftdrop,
                                                  ptgen=gen_dijet.pt, mgen=groomed_gen_dijet.mass, weight=self.weights[jetsyst].weight())
                    out["response_rho_u"].fill(dataset=datastr, syst=jetsyst, jk=jk_index,  ptreco = dijet.pt, rhoreco=np.log((dijet.mass/dijet.pt)**2), ptgen = gen_dijet.pt, rhogen=-np.log((gen_dijet.mass/gen_dijet.pt)**2), weight=self.weights[jetsyst].weight())
                    out["response_rho_g"].fill(dataset=datastr, syst=jetsyst, jk=jk_index,  ptreco = dijet.pt, rhoreco=np.log((dijet.msoftdrop/dijet.pt)**2), ptgen = gen_dijet.pt,rhogen=-np.log((gen_dijet.pt/groomed_gen_dijet.mass)**2), weight=self.weights[jetsyst].weight())
                    out["ptreco_mreco_u"].fill(dataset=datastr, syst=jetsyst, jk=jk_index, ptreco=dijet.pt, mreco=dijet.mass, weight=self.weights[jetsyst].weight() )
                    out["ptreco_mreco_g"].fill(dataset=datastr, syst=jetsyst, jk=jk_index, ptreco=dijet.pt, mreco=dijet.msoftdrop, weight=self.weights[jetsyst].weight() )
                    if jetsyst == "nominal":
                        for syst in self.weights[jetsyst].variations:
                            # print("Weight variation: ", syst)
                            #fill nominal, up, and down variations for each
                            out['ptgen_mgen_u'].fill(dataset=datastr, syst=syst, jk=jk_index, ptgen=gen_dijet.pt, mgen=gen_dijet.mass, weight=self.weights[jetsyst].weight(syst) )
                            out['ptgen_mgen_g'].fill(dataset=datastr, syst=syst, jk=jk_index, ptgen=gen_dijet.pt, mgen=groomed_gen_dijet.mass, 
                                                          weight=self.weights[jetsyst].weight(syst) )           
                            out["response_matrix_u"].fill(dataset=datastr,syst=syst, jk=jk_index,
                                                   ptreco=dijet.pt, mreco=dijet.mass,
                                                   ptgen=gen_dijet.pt, mgen=gen_dijet.mass, weight=self.weights[jetsyst].weight(syst))
                            out["response_matrix_g"].fill(dataset=datastr,syst=syst, jk=jk_index, ptreco=dijet.pt, mreco=dijet.msoftdrop,
                                                          ptgen=gen_dijet.pt, mgen=groomed_gen_dijet.mass, weight=self.weights[jetsyst].weight(syst))
                            out["response_rho_u"].fill(dataset=datastr, syst=syst, jk=jk_index, ptreco = dijet.pt, rhoreco=-np.log((dijet.mass/dijet.pt)**2), ptgen =gen_dijet.pt, rhogen=-np.log((gen_dijet.mass/gen_dijet.pt)**2), weight=self.weights[jetsyst].weight(syst))
                            out["response_rho_g"].fill(dataset=datastr, syst=syst, jk=jk_index,  ptreco = dijet.pt, rhoreco=-np.log((dijet.msoftdrop/dijet.pt)**2), ptgen = gen_dijet.pt, rhogen=-np.log((gen_dijet.pt/groomed_gen_dijet.mass)**2), weight=self.weights[jetsyst].weight(syst))
                            out["ptreco_mreco_u"].fill(dataset=datastr,syst=syst, jk=jk_index, ptreco=dijet.pt, mreco=dijet.mass, weight=self.weights[jetsyst].weight(syst) )
                            out["ptreco_mreco_g"].fill(dataset=datastr,syst=syst, jk=jk_index, ptreco=dijet.pt, mreco=dijet.msoftdrop, weight=self.weights[jetsyst].weight(syst) )
                            # if ak.sum(fakes)>0:
                            #     out["fakes"].fill(syst=syst, jk=jk_index, ptreco = fake_dijets.pt, mreco = fake_dijets.mass, weight=fake_weights.weight(syst))
                            #     out["fakes_g"].fill(syst=syst, jk=jk_index, ptreco = fake_dijets.pt, mreco = fake_dijets.msoftdrop, weight=fake_weights.weight(syst))
    
                    #weird = (reco_jet.msoftdrop/groomed_gen_jet.mass > 2.0) & (reco_jet.msoftdrop > 10.)
                    weird = ((dijet.msoftdrop/groomed_gen_dijet.mass) > 2.0) & (dijet.msoftdrop > 10.)
                    # print("Number of what ashley called weird events ", ak.sum(weird)) 

                ###############
                ##### If running over DATA only fill final reco plots
                ###############
                
                else:
                    out["ptreco_mreco_u"].fill(dataset=datastr,syst=jetsyst, jk=jk_index, ptreco=dijet.pt, mreco=dijet.mass, weight=self.weights[jetsyst].weight() )
                    out["ptreco_mreco_g"].fill(dataset=datastr,syst=jetsyst, jk=jk_index, ptreco=dijet.pt, mreco=dijet.msoftdrop, weight=self.weights[jetsyst].weight() )
                    out["rho_reco_u"].fill(dataset=datastr,syst=jetsyst, jk=jk_index, ptreco = dijet.pt, rhoreco=-np.log((dijet.mass/dijet.pt)**2), weight=self.weights[jetsyst].weight() )
                    out["rho_reco_g"].fill(dataset=datastr,syst=jetsyst, jk=jk_index, ptreco = dijet.pt, rhoreco=-np.log((dijet.msoftdrop/dijet.pt)**2), weight=self.weights[jetsyst].weight() )
                    if not self.jk:
                        out["jet_pt_eta_phi"].fill(dataset=datastr,syst=jetsyst, ptreco=dijet.pt, phi=dijet.phi, eta=dijet.eta, weight=self.weights[jetsyst].weight())
                if (jetsyst == "nominal"): 
                    for name in sel.names:
                        out["cutflow"][dataset][name] += sel.all(name).sum()
                del events_corr, weights
            del events_jk
        out['cutflow'][dataset]['chunks'] += 1
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

    
def main():
    #### Next run processor with futures executor on all test files
    from dask.distributed import Client
    from plugins import runCoffeaJob
    processor = makeDijetHists()
    result = runCoffeaJob(processor, jsonFile = "QCD_flat_files.json", winterfell = True, testing = True, data = False)
    util.save(result, "coffeaOutput/dijet_pT" + str(processor.ptcut) + "_rapidity" + str(processor.ycut) + "_result_test.coffea")
    

if __name__ == '__main__':
    # Execute when the module is not initialized from an import statement: i.e. called from terminal command line
    main()

