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
    def __init__(self, ptcut = 200., ycut = 2.5, data = False, jet_systematics = ['nominal', 'HEM'], systematics = ['L1PreFiringWeight', 'PUSF'], jk=False):
        # should have separate **lower** ptcut for gen
        self.do_gen = not data
        self.ptcut = ptcut
        self.ycut = ycut #rapidity
        self.jet_systematics = jet_systematics
        self.systematics = systematics
        self.jk = jk
        print("Data: Gen: ", data, self.do_gen)
        jet_cat = hist.axis.StrCategory([], growth=True, name="jetNumb", label="Jet")
        syst_cat = hist.axis.StrCategory([], growth=True, name='syst', label="Systematic")
        parton_cat = hist.axis.StrCategory([], growth=True,name="partonFlav", label="Parton Flavour")
        mgen_bin_edges = np.array([0,10,20,40,60,80,100,150,200,300,1300])
        mreco_bin_edges = np.sort(np.append(mgen_bin_edges,[(mgen_bin_edges[i]+mgen_bin_edges[i+1])/2 for i in range(len(mgen_bin_edges)-1)]))
        mass_gen_bin =  hist.axis.Variable(mgen_bin_edges, name="mgen", label=r"m_{GEN} (GeV)")                         
        mass_bin = hist.axis.Variable(mreco_bin_edges, name="mreco", label=r"m_{RECO} (GeV)")
        ptgen_edges = np.array([200,260,350,430,540,630,690,750,810,13000]) 
        # ptgen_edges = np.array([200,280,380,460,560,640,700,800,13000]) #### NEW VALUES TO SWITCH TO
        pt_bin = hist.axis.Variable(ptgen_edges, name="ptreco", label=r"p_{T,RECO} (GeV)")     
        pt_gen_bin = hist.axis.Variable(ptgen_edges, name="ptgen", label=r"p_{T,GEN} (GeV)") 
        y_bin = hist.axis.Regular(25, 0., 2.5, name="rapidity", label=r"$y$")
        eta_bin = hist.axis.Regular(25, 0., 2.5, name="eta", label=r"$\eta$")
        frac_axis = hist.axis.Regular(10, 0.0, 1., name="frac", label="Fraction")
        n_axis = hist.axis.Regular(5, 0, 5, name="n", label=r"Number")
        dr_axis = hist.axis.Regular(25, 0, 6.0, name="dr", label=r"$\Delta R$")
        phi_axis = hist.axis.Regular(25, -2*np.pi, 2*np.pi, name="phi", label=r"$\phi$")
        dphi_axis = hist.axis.Regular(25, -2*np.pi, 2*np.pi, name="dphi", label=r"$\Delta \phi$")
        weight_bin = hist.axis.Regular(100, 0, 5, name="corrWeight", label="Weight")
        jk_axis = hist.axis.IntCategory([], growth = True, name = 'jk', label = "Jackknife section" )
        cutflow = {}
        self._histos = {
            
                #### For jackknife only need resp. matrix hists
                'misses':                    hist.Hist(syst_cat, jk_axis, pt_gen_bin, mass_gen_bin, storage="weight", name="Events"),
                'misses_g':                    hist.Hist(syst_cat, jk_axis, pt_gen_bin, mass_gen_bin, storage="weight", name="Events"),
                'fakes':                     hist.Hist(syst_cat, jk_axis, pt_bin, mass_bin, storage="weight", name="Events"),
                'fakes_g':                   hist.Hist(syst_cat, jk_axis, pt_bin, mass_bin, storage="weight", name="Events"),
                'underflow':                     hist.Hist(syst_cat, jk_axis, pt_bin, mass_bin, storage="weight", name="Events"),
                'underflow_g':                   hist.Hist(syst_cat, jk_axis, pt_bin, mass_bin, storage="weight", name="Events"),
                #### hist for comparison of weights
                'weights':                   hist.Hist(syst_cat, jk_axis, pt_bin, mass_bin, storage="weight", name="Events"),
                
                #### Plots to be unfolded
                'ptreco_mreco_u':        hist.Hist(syst_cat, jk_axis, pt_bin, mass_bin, storage="weight", name="Events"),
                'ptreco_mreco_g':        hist.Hist(syst_cat, jk_axis, pt_bin, mass_bin, storage="weight", name="Events"),
        
                #### Plots for comparison
                'ptgen_mgen_u':         hist.Hist(syst_cat, jk_axis, pt_gen_bin, mass_gen_bin, storage="weight", label="Counts"),       
                'ptgen_mgen_g':         hist.Hist(syst_cat, jk_axis, pt_gen_bin, mass_gen_bin, storage="weight", label="Counts"),
            
                #### Plots for the analysis in the proper binning
                'response_matrix_u':         hist.Hist(syst_cat, jk_axis, pt_bin, mass_bin, pt_gen_bin, mass_gen_bin, storage="weight",                                                         label="Counts"),
                'response_matrix_g':         hist.Hist(syst_cat, jk_axis, pt_bin, mass_bin, pt_gen_bin, mass_gen_bin, storage="weight",                                                         label="Counts"),
                     
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
            'sdmass_orig':         hist.Hist(jk_axis, pt_bin, mass_bin, storage="weight", label="Events"),
            'sdmass_ak8corr':         hist.Hist(jk_axis, pt_bin, mass_bin, storage="weight", label="Events"),
            'sdmass_ak4corr':         hist.Hist(jk_axis, pt_bin, mass_bin, storage="weight", label="Events"),
            'njet_reco':                 hist.Hist(syst_cat, n_axis, storage="weight", label="Counts"),
            'njet_gen':                  hist.Hist(syst_cat, n_axis, storage="weight", label="Counts"),
            #'jet_dr_reco_gen':           hist.Hist(dr_axis, storage="weight", label="Counts"),
            # 'eta_reco':              hist.Hist(syst_cat, eta_bin, storage="weight", name="Events"),
            # 'eta_gen':               hist.Hist(syst_cat, eta_bin, storage="weight",name="Events"),
            'jet_rap_reco':              hist.Hist(syst_cat, y_bin, storage="weight", name="Events"),
            'jet_rap_gen':               hist.Hist(syst_cat, y_bin, storage="weight",name="Events"),
            #'jet_dr_gen':                hist.Hist(dr_axis, storage="weight", label="Counts"),
            #'jet_dr_reco':               hist.Hist(dr_axis, storage="weight", label="Counts"),
            'jet_phi_gen':                  hist.Hist(syst_cat, phi_axis, storage="weight", label="Counts"),
            'jet_phi_reco':                 hist.Hist(syst_cat, phi_axis, storage="weight", label="Counts"),
            'dphi_gen':                  hist.Hist(syst_cat, dphi_axis, storage="weight", label="Counts"),
            'dphi_reco':                 hist.Hist(syst_cat, dphi_axis, storage="weight", label="Counts"),
            'asymm_gen':                 hist.Hist(syst_cat, frac_axis, storage="weight", label="Counts"),
            'asymm_reco':                hist.Hist(syst_cat, frac_axis, storage="weight", label="Counts"),
            'jet_dr_gen_subjet':         hist.Hist(syst_cat, dr_axis, storage="weight", label="Counts"),
            'dijet_dr_reco_to_gen':      hist.Hist(syst_cat, dr_axis, storage="weight", label="Counts"),
            'dr_reco_to_gen_subjet' :    hist.Hist(syst_cat, dr_axis, storage="weight", label="Counts"),
            #### for investigation of removing fakes
            'fakes_eta_phi':             hist.Hist(syst_cat, eta_bin, phi_axis, storage="weight", name="Events"),
            'fakes_asymm_dphi':             hist.Hist(syst_cat, frac_axis, dphi_axis, storage="weight", name="Events"),

            #### Plots to get JMR and JMS in MC
            # 'jet_m_pt_u_reco_over_gen':  hist.Hist(syst_cat, pt_gen_bin, mass_gen_bin, frac_axis, storage="weight",                                                                        label="Counts"),
            # 'jet_m_pt_g_reco_over_gen':  hist.Hist(syst_cat, pt_gen_bin, mass_gen_bin, frac_axis, storage="weight",                                                                        label="Counts"),
                             })
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
        print("Dataset: ", dataset)
        if "madgraph" in dataset and "pythia" in dataset:
            mctype="pythiaMG"
        elif "herwig" in dataset:
            mctype='herwig'
        else:
            mctype="pythia"
        #####################################
        #### Find the IOV from the dataset name
        #####################################
        IOV = ('2016APV' if ( any(re.findall(r'APV',  dataset)) or any(re.findall(r'HIPM', dataset)))
               else '2018'    if ( any(re.findall(r'UL18', dataset)) or any(re.findall(r'UL2018',    dataset)))
               else '2017'    if ( any(re.findall(r'UL17', dataset)) or any(re.findall(r'UL2017',    dataset)))
               else '2016')
        out['cutflow'][dataset] = defaultdict(int)
        out['cutflow'][dataset]['nEvents initial'] += (len(events.FatJet))
        index_list = np.arange(len(events))
        if self.jk:
            print("Self.jk ", self.jk)
            range_max = 10
        else: range_max=1
        for jk_index in range(0,range_max):
            if self.jk:
                print("Now doing jackknife {}".format(jk_index))
                print("Len of events before jk selection ", len(events))
            else:
                jk_index=-1
            jk_sel = ak.where(index_list%range_max == jk_index, False, True)
            #### Choosing our jk fraction and protecting against negative softdrop
            events_jk = events[jk_sel]
            out['cutflow'][dataset]['nEvents for jk index '+str(jk_index)] += (len(events_jk.FatJet))
            del jk_sel
            #####################################
            #### Apply HEM veto
            #### Add values needed for jet corrections
            #####################################
            # if IOV == '2018' and self.hem:
            #     print("Doing hem")
            #     events_jk = events_jk[HEMVeto(events_jk.FatJet, events.run)]
            #### require at least one fat jet and one subjet so corrections do not fail
            # events_jk = events_jk[(ak.num(events_jk.SubJet) > 0)]
            out['cutflow'][dataset]['nEvents w/ pos mass and at least one subjet & fatjet'] += (len(events_jk.FatJet))
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
                # print("Number of gen jets matched to fat jets: ", len(ak.values_astype(ak.fill_none(FatJet.p4.nearest(GenJetAK8.p4, threshold=0.4).pt, 0), np.float32)), " values ", ak.values_astype(ak.fill_none(FatJet.p4.nearest(GenJetAK8.p4, threshold=0.4).pt, 0), np.float32))
                FatJet["pt_gen"] = ak.values_astype(ak.fill_none(FatJet.p4.nearest(GenJetAK8.p4, threshold=0.4).pt, 0), np.float32)
            else:
                firstidx = filename.find( "store/data/" )
                fname2 = filename[firstidx:]
                fname_toks = fname2.split("/")
                era = fname_toks[ fname_toks.index("data") + 1]
                print("IOV, era ", IOV, era)
            out["sdmass_orig"].fill(jk=jk_index, ptreco=ak.flatten(events_jk[(ak.num(events_jk.FatJet) > 1)].FatJet[:,:2].pt, axis=1), mreco=ak.flatten(events_jk[(ak.num(events_jk.FatJet) > 1)].FatJet[:,:2].msoftdrop, axis=1))
            corrected_fatjets = GetJetCorrections(FatJet, events_jk, era, IOV, isData=not self.do_gen)
            #### Subjet corrections breaks without requiring at least one subjet --> ak.where doesn't work either
            corrected_fatjets = GetCorrectedSDMass(corrected_fatjets, events_jk, era, IOV, isData=not self.do_gen)
            out["sdmass_ak4corr"].fill(jk=jk_index, ptreco=ak.flatten(corrected_fatjets[(ak.num(corrected_fatjets) > 1)][:,:2].pt, axis=1), mreco=ak.flatten(corrected_fatjets[(ak.num(corrected_fatjets) > 1)][:,:2].msoftdrop, axis=1))
            corrected_fatjets_ak8 = corrected_fatjets
            corrected_fatjets_ak8 = GetCorrectedSDMass(corrected_fatjets, events_jk, era, IOV, isData=not self.do_gen, useSubjets = False)
            out["sdmass_ak8corr"].fill(jk=jk_index, ptreco=ak.flatten(corrected_fatjets_ak8[(ak.num(corrected_fatjets_ak8) > 1)][:,:2].pt, axis=1), mreco=ak.flatten(corrected_fatjets_ak8[(ak.num(corrected_fatjets_ak8) > 1)][:,:2].mass, axis=1))
            jet_corrs = {}
            self.weights = {}
            if 'HEM' in self.jet_systematics and self.do_gen:
                jet_corrs.update({
                           "HEM": HEMCleaning(IOV,applyjmsSF(IOV, applyjmrSF(IOV,corrected_fatjets)))
                          })
            if 'JER' in self.jet_systematics and self.do_gen and "JER" in corrected_fatjets.fields:
                corrected_fatjets.JER.up = applyjmsSF(IOV, applyjmrSF(IOV,corrected_fatjets.JER.up))
                corrected_fatjets.JER.down = applyjmsSF(IOV, applyjmrSF(IOV,corrected_fatjets.JER.down))
                jet_corrs.update({"jerUp": corrected_fatjets.JER.up,
                                    "jerDown": corrected_fatjets.JER.down
                                })
            if "JMR" in self.jet_systematics and self.do_gen:
                jet_corrs.update({"jmrUp": applyjmsSF(IOV, applyjmrSF(IOV,corrected_fatjets, var = "up")),
                                    "jmrDown": applyjmsSF(IOV, applyjmrSF(IOV,corrected_fatjets, var = "down"))})
            if "JMS" in self.jet_systematics and self.do_gen:
                jet_corrs.update({"jmsUp": applyjmsSF(IOV, applyjmrSF(IOV,corrected_fatjets), var = "up"),
                                    "jmsDown": applyjmsSF(IOV, applyjmrSF(IOV,corrected_fatjets), var = "down")})
            if 'nominal' in self.jet_systematics or not self.do_gen:
                print("Doing nominal data")
                jet_corrs.update({"nominal": corrected_fatjets})
            if 'nominal' in self.jet_systematics and self.do_gen:
                print("Doing nominal")
                corrected_fatjets = applyjmsSF(IOV, applyjmrSF(IOV,corrected_fatjets))
                jet_corrs.update({"nominal": corrected_fatjets})
            elif self.do_gen:
                # print("Getting sources: ", [unc_src for unc_src in self.jet_systematics if ("JES_"+unc_src in corrected_fatjets.fields)])
                # print("Out of avail corrections: ", corrected_fatjets.fields)
                avail_srcs = [unc_src for unc_src in self.jet_systematics if ("JES_"+unc_src in corrected_fatjets.fields)]
                # print("Input jet syst", self.jet_systematics)
                for unc_src in avail_srcs:
                    print(corrected_fatjets["JES_"+unc_src])
                    corrected_fatjets["JES_"+unc_src].up = applyjmsSF(IOV, applyjmrSF(IOV,corrected_fatjets["JES_"+unc_src].up))
                    corrected_fatjets["JES_"+unc_src].down = applyjmsSF(IOV, applyjmrSF(IOV,corrected_fatjets["JES_"+unc_src].down))
                    jet_corrs.update({
                        unc_src+"Up":corrected_fatjets["JES_"+unc_src].up,
                        unc_src+"Down":corrected_fatjets["JES_"+unc_src].down, })
            # print("Final jet corrs to run over: ", jet_corrs)
            for jetsyst in jet_corrs.keys():
                # print("Adding ", jetsyst, " values ", jet_corrs[jetsyst], " to output")
                events_corr = ak.with_field(events_jk, jet_corrs[jetsyst], "FatJet")
                ###################################
                ######### INITIALIZE WEIGHTS AND SELECTION
                ##################################
                weights = np.ones(len(events_corr))
                sel = PackedSelection()
                sel.add("npv", events_corr.PV.npvsGood > 0)
                print("mctype ", mctype, " gen? ", self.do_gen)
                if self.do_gen and (mctype == "pythia"):
                    print("Doing XS scaling")
                    weights = weights * getXSweight(dataset, IOV)
                elif self.do_gen and (mctype == "herwig"):
                    print("Difference between weights calculated from xsdb and LHE :", (events_corr.LHEWeight.originalXWGTUP - getXSweight(dataset, IOV)))
                    weights = events_corr.LHEWeight.originalXWGTUP * getXSweight(dataset, IOV)
                elif self.do_gen:
                    print("MADGRPAH inputs --> get gen weights from files")
                    weights = events_corr.LHEWeight.originalXWGTUP
                else:
                    ###############
                    ### apply lumimask and require at least one jet to apply jet trigger prescales
                    ##############
                    lumi_mask = getLumiMask(IOV)(events_corr.run, events_corr.luminosityBlock)
                    events_corr = events_corr [lumi_mask]
                    out['cutflow'][dataset]['nEvents after good lumi sel '+jetsyst] += (len(events_corr[trigsel].FatJet))
                    trigsel, psweights = applyPrescales(events_corr, year = IOV)
                    weights=psweights
                    # print("Trigger: len of events ", len(events_corr), "len of weights ", len(trigsel))
                    # print("Weights w/ prescales ", weights)
                    sel.add("trigsel", trigsel) 
                    weights = ak.where(trigsel, psweights, weights)
                    out['cutflow'][dataset]['nEvents after trigger sel '+jetsyst] += (len(events_corr[trigsel].FatJet))
    
                #####################################
                #### Gen Jet Selection
                #### see CMS PAS SMP-20-010 for selections
                ####################################
                if (self.do_gen):
                    print("DOING GEN")
                    #### Select events with at least 2 jets
                    if not self.jk:
                        out["njet_gen"].fill(syst = jetsyst, n=ak.num(events_corr[sel.all("npv")].GenJetAK8), 
                                         weight = weights )
                    pt_cut_gen = ak.all(events_corr.GenJetAK8.pt > 160., axis = -1) ### 80% of reco pt cut
                    gen_sel = pt_cut_gen & (ak.num(events_corr.GenJetAK8) > 1)
                    sel.add("twoGenJet", gen_sel)
                    sel.add("twoGenJet_seq", sel.all('npv', 'twoGenJet') )
                    out['cutflow'][dataset]['nEvents w/ at least 2 genjets &  pt > 160'+jetsyst] += (len(events_corr[sel.all("twoGenJet_seq")].FatJet))
                    GenJetAK8 = events_corr.GenJetAK8
                    GenJetAK8['p4']= ak.with_name(events_corr.GenJetAK8[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
                    rap_cut_gen = ak.where(sel.all("twoGenJet_seq"), ak.all(np.abs(getRapidity(GenJetAK8[:,:2].p4)) < self.ycut, axis = -1), False)
                    sel.add("genRap2p5", rap_cut_gen)
                    sel.add("genRap_seq", sel.all("twoGenJet_seq", "genRap2p5"))
                    if not self.jk:
                        out["jet_rap_gen"].fill(syst = jetsyst, rapidity=ak.flatten(getRapidity(GenJetAK8[sel.all("twoGenJet_seq")][:,:2].p4), axis=1), weight=np.repeat(weights[sel.all("twoGenJet_seq")], 2))
                        out["jet_phi_gen"].fill(syst=jetsyst, phi=ak.flatten(GenJetAK8[sel.all("twoGenJet_seq")][:,:2].phi, axis=1), weight=np.repeat(weights[sel.all("twoGenJet_seq")], 2))  
                    #### Apply kinematic and 2 jet requirement immediately so that dphi and asymm can be calculated
                    out['cutflow'][dataset]['nEvents after gen rapidity selection '+jetsyst] += (len(events_corr[sel.all("genRap_seq")].FatJet))
                    #### get dphi and pt asymm selections 
                    genjet1 = ak.firsts(events_corr.GenJetAK8[:,0:])
                    genjet2 = ak.firsts(events_corr.GenJetAK8[:,1:])
                    dphi12_gen = np.abs(genjet1.delta_phi(genjet2))
                    print("Dphi ", dphi12_gen)
                    dphi12_gen_sel = ak.where(sel.all("twoGenJet_seq"), dphi12_gen > 2., False)
                    sel.add("dphiGen2", dphi12_gen_sel)
                    print("Asym num ", np.abs(genjet1.pt - genjet2.pt))
                    print("Asym denom ", np.abs(genjet1.pt + genjet2.pt))
                    # genjet2["pt"] = ak.fill_none(genjet2.pt, 0.0001)
                    asymm_gen  = np.abs(genjet1.pt - genjet2.pt)/(genjet1.pt + genjet2.pt)
                    asymm_gen_sel = ak.where(sel.all("twoGenJet_seq"), asymm_gen < 0.3, False)
                    sel.add("genAsym0p3", asymm_gen_sel)
                    sel.add("genTot_seq", sel.all("genRap_seq", "dphiGen2", "genAsym0p3") & ~ak.is_none(events_corr.GenJetAK8[:,:2].mass))
                    
                    #### N-1 plots
                    if not self.jk:
                        out["asymm_gen"].fill(syst=jetsyst,frac=asymm_gen[sel.all("twoGenJet_seq")], weight=weights[sel.all("twoGenJet_seq")])  
                        out["dphi_gen"].fill(syst=jetsyst, dphi=dphi12_gen[sel.all("twoGenJet_seq")], weight=weights[sel.all("twoGenJet_seq")])
                    ########### Move misses and fakes to after both selections
                    ########### misses = gen but no reco
                    # if len(events_corr_gen[gen_sel2])<1:
                    #     print("No events after miss removal")
                    #     return out
                #####################################
                #### Reco Jet Selection
                #################################### 
                #### Apply pt and rapidity cuts
                if not self.jk:
                    out["njet_reco"].fill(syst = jetsyst, n=ak.num(events_corr.FatJet), 
                                         weight = weights)
                sel.add("recoPt200", (ak.all(events_corr.FatJet[:,:2].pt > self.ptcut, axis = -1)))
                sel.add("twoRecoJet",  (ak.num(events_corr.FatJet) > 1) & sel.all("recoPt200"))
                sel.add("twoRecoJet_seq",  sel.all("npv", "twoRecoJet"))
                print("Nevents after 2 jets ", len(events_corr[sel.all("twoRecoJet_seq")]))
                FatJet = events_corr.FatJet
                FatJet["p4"] = ak.with_name(FatJet[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
                rap_cut_reco = ak.all(np.abs(getRapidity(FatJet[:,:2].p4)) < self.ycut, axis = -1)
                rap_sel = ak.where(sel.all("twoRecoJet_seq"), rap_cut_reco, False)
                sel.add("recoRap2p5", rap_sel)
                sel.add("recoRap_seq", sel.all("twoRecoJet_seq", "recoRap2p5")) 
                print("Nevents after rap ", len(events_corr[sel.all("recoRap_seq")]))
                if not self.jk:
                    out["jet_rap_reco"].fill(syst = jetsyst, rapidity=ak.to_numpy(ak.flatten(getRapidity(FatJet[sel.all("twoRecoJet_seq")][:,:2].p4)), allow_missing=True),
                                             weight=np.repeat(weights[sel.all("twoRecoJet_seq")], 2))
                    out["jet_phi_reco"].fill(syst=jetsyst, phi=ak.flatten(FatJet[sel.all("twoRecoJet_seq")][:,:2].phi, axis=1), weight=np.repeat(weights[sel.all("twoRecoJet_seq")], 2)) 
                #### Add cut on softdrop mass as done in previous two papers --> need to verify with JMS/JMR studies
                # sdm_cut = (ak.all(events_corr_reco.FatJet.msoftdrop > 10., axis = -1))
                #### get dphi and pt asymm selections
                jet1 = ak.firsts(events_corr.FatJet[:,0:])
                jet2 = ak.firsts(events_corr.FatJet[:,1:])
                dphi12 = (np.abs(jet1.delta_phi(jet2)) > 2.)
                dphi12_sel = ak.where(sel.all("twoRecoJet_seq"), dphi12, False)
                sel.add("recodphi2", dphi12_sel)
                sel.add("recodphi_seq", sel.all("recodphi2", "recoRap_seq"))
                print("Nevents after dphi ", len(events_corr[sel.all("recodphi_seq")]))
                asymm = np.abs(jet1.pt - jet2.pt)/(jet1.pt + jet2.pt)
                if not self.jk:
                    out["dphi_reco"].fill(syst=jetsyst, dphi = dphi12[sel.all("twoRecoJet_seq")], weight=weights[sel.all("twoRecoJet_seq")])
                    out["asymm_reco"].fill(syst=jetsyst, frac = asymm[sel.all("twoRecoJet_seq")], weight=weights[sel.all("twoRecoJet_seq")])
                asymm_reco_sel = ak.where(sel.all("twoRecoJet_seq"), asymm < 0.3, False)
                sel.add("recoAsym0p3", asymm_reco_sel)
                sel.add("recoAsym_seq", sel.all("recoAsym0p3", "recodphi_seq"))
                print("Nevents after asym ", len(events_corr[sel.all("recoAsym_seq")]))
                # jet1 = ak.to_numpy(ak.firsts(events_corr.FatJet[:,0:]), allow_missing=True)
                # jet2 = ak.to_numpy(ak.firsts(events_corr.FatJet[:,1:]), allow_missing=True)
                # veto = ApplyVetoMap(IOV, jet1, mapname='jetvetomap') & ApplyVetoMap(IOV, jet2, mapname='jetvetomap')
                # veto_sel = ak.where(sel.all("twoRecoJet_seq"), veto)
                # sel.add("jetVeto", veto_sel)
                dijet_weights = np.repeat(weights, 2)
                print("len of weights", len(weights))
                print("len of events", len(events_corr))
                jetid_sel = ak.where(sel.all("twoRecoJet_seq"), ak.all(events_corr.FatJet[:,:2].jetId > 1, axis=-1), False)
                sel.add("jetId", jetid_sel)
                sel.add("recoTot_seq", sel.all("recoAsym_seq", "jetId") & ~ak.is_none(events_corr.FatJet[:,:2].mass) & ~ak.is_none(events_corr.FatJet[:,:2].msoftdrop))
                ####  Apply Final RECO selection
                dijet = ak.flatten(events_corr[sel.all("recoTot_seq")].FatJet[:,:2], axis =1)
                dijet_weights = np.repeat(weights[sel.all("recoTot_seq")], 2)
                #### match jets, get syst weights, and fill final plots
                if self.do_gen:
                    # print("Genjets before padding ", events_corr.GenJetAK8.pt)
                    # dijet = ak.pad_none(events_corr.FatJet, 2, axis=0)[:,:2]
                    # gen_dijet = ak.pad_none(events_corr.GenJetAK8, 2, axis = 0)[:,:2]
                    # print("Genjets after padding ", gen_dijet.pt)
                    #### check for empty softdropmass values
                    # print("Structure of matched gen ", events_corr.FatJet.matched_gen)
                    matches = ak.all(events_corr.GenJetAK8[:,:2].delta_r(events_corr.GenJetAK8[:,:2].nearest(events_corr.FatJet[:,:2])) < 0.4, axis = -1)
                    #### have found some events that are missing reco msoftdrop --- add to misses
                    print("Nevents missing masses ", ak.sum(ak.any(ak.is_none(events_corr[sel.all("recoTot_seq", "genTot_seq")].FatJet.msoftdrop, axis=-1), axis=-1) | ak.any(ak.is_none(events_corr[sel.all("recoTot_seq", "genTot_seq")].FatJet.mass, axis=-1), axis=-1)))
                    #### Misses include events missing a gen mass, events failing DR matching, and events passing gen cut but failing the reco cut
                    misses = ~matches | sel.require(genTot_seq=True, recoTot_seq=False)
                    sel.add("removeMisses", ~misses )
                    if len(weights[misses])>0:
                        misses = misses  & (ak.num(events_corr.GenJetAK8) > 1)
                        miss_dijets = ak.flatten(events_corr[misses].GenJetAK8[:,:2], axis=1)
                        genjet1 = ak.firsts(events_corr[misses].GenJetAK8[:,0:])
                        genjet2 = ak.firsts(events_corr[misses].GenJetAK8[:,1:])
                        groomed_genjet0 = get_gen_sd_mass_jet(genjet1, events_corr[misses].SubGenJetAK8)
                        groomed_genjet1 = get_gen_sd_mass_jet(genjet2, events_corr[misses].SubGenJetAK8)
                        print("Groomed genjet 0 ", groomed_genjet0, " groomed genjet 1 ", groomed_genjet1)
                        groomed_gen_dijet = ak.concatenate([ak.unflatten(groomed_genjet0, 1),  ak.unflatten(groomed_genjet1, 1)], axis=1)
                        groomed_gen_dijet = ak.flatten(groomed_gen_dijet, axis=1)
                        miss_weights = np.repeat(weights[misses & (ak.num(events_corr.GenJetAK8) > 1)], 2)
                        out["misses"].fill(syst=jetsyst, jk=jk_index, ptgen = miss_dijets[~ak.is_none(miss_dijets.mass)].pt, mgen = miss_dijets[~ak.is_none(miss_dijets.mass)].mass, weight = miss_weights[~ak.is_none(miss_dijets)])
                        out["misses_g"].fill(syst=jetsyst, jk=jk_index, ptgen = miss_dijets[~ak.is_none(miss_dijets.mass)].pt, mgen = groomed_gen_dijet[~ak.is_none(miss_dijets.mass)].mass, weight = miss_weights[~ak.is_none(miss_dijets)])
                    
                    out['cutflow'][dataset]['misses '+jetsyst] += (len(events_corr[misses].FatJet))
                    if len(events_corr[sel.all("genTot_seq", "recoTot_seq", "removeMisses")])<1: 
                        print("No events after all selections and removing misses")
                        return out
                    #### Fakes include events missing a reco mass or sdmass value, events failing index dr matching, and events passing reco cut but failing the gen cut
                    fakes = ak.any(ak.is_none(dijet.matched_gen, axis = -1), axis = -1) | sel.require(genTot_seq=False, recoTot_seq=True)
                    sel.add("removeFakes", ~fakes)
                    fakes = fakes & (ak.num(events_corr.FatJet) > 1)
                    if len(weights[fakes])>0:
                        print("len of no nones ",ak.sum(ak.is_none(events_corr.FatJet[:,:2])))
                        fake_dijets = ak.flatten(events_corr[fakes].FatJet[:,:2], axis=1)
                        fake_weights = Weights(len(np.repeat(weights[fakes], 2)))
                        fake_weights.add('fakeWeight', np.repeat(weights[fakes], 2))
                        print("Len of flattened diejts ", len(fake_dijets), " and weights ", len(fake_weights.weight()))
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
                        print("Flattened fake pts", fake_dijets.pt, " masses ", fake_dijets.mass ," and weights ", fake_weights.weight())
                        out["fakes"].fill(syst=jetsyst, jk = jk_index, ptreco = fake_dijets[~ak.is_none(fake_dijets.mass)].pt, mreco = fake_dijets[~ak.is_none(fake_dijets.mass)].mass, weight = fake_weights.weight()[~ak.is_none(fake_dijets.mass)])
                        out["fakes_g"].fill(syst=jetsyst, jk = jk_index, ptreco = fake_dijets[~ak.is_none(fake_dijets.msoftdrop)].pt, mreco = fake_dijets[~ak.is_none(fake_dijets.msoftdrop)].msoftdrop, weight = fake_weights.weight()[~ak.is_none(fake_dijets.msoftdrop)])
                        if not self.jk:
                            out['fakes_eta_phi'].fill(syst=jetsyst, phi = fake_dijets.phi[~ak.is_none(fake_dijets.msoftdrop)], eta = fake_dijets.eta[~ak.is_none(fake_dijets.msoftdrop)], weight=fake_weights.weight()[~ak.is_none(fake_dijets.msoftdrop)])
                    ##### fakes = reco but no gen
                    out['cutflow'][dataset]['fakes '+jetsyst] += (len(events_corr[fakes].FatJet))
                    if len(events_corr[sel.all("genTot_seq", "recoTot_seq", "removeMisses", "removeFakes")])<1: 
                        print("No events after all selections and removing fakes & misses")
                        return out
                    uf = ak.where(sel.require(recoTot_seq=True), ak.any(events_corr.GenJetAK8[:,:2].pt < 200., axis = -1), False)
                    sel.add("rem_uf_fakes", ~uf)
                    uf_dijets = ak.flatten(events_corr[uf].FatJet[:,:2], axis=1)
                    uf_weights = np.repeat(weights[uf], 2)
                    print("Lengths of underflow dijets ", len(uf_dijets), " length of underflow weights ", len(uf_weights))
                    out["underflow"].fill(syst=jetsyst, jk = jk_index, ptreco = uf_dijets[~ak.is_none(uf_dijets.mass)].pt, mreco = uf_dijets[~ak.is_none(uf_dijets.mass)].mass, weight = uf_weights[~ak.is_none(uf_dijets.mass)])
                    out["underflow_g"].fill(syst=jetsyst, jk = jk_index, ptreco = uf_dijets[~ak.is_none(uf_dijets.mass)].pt, mreco = uf_dijets[~ak.is_none(uf_dijets.mass)].msoftdrop, weight = uf_weights[~ak.is_none(uf_dijets.mass)])
                    if len(events_corr[sel.all("genTot_seq", "recoTot_seq", "removeMisses", "removeFakes","rem_uf_fakes")])<1:
                        print("no more events after separating underflow")
                        return out
                    ############################
                    ########## Apply final gen and reco selections before plotting
                    ############################
                    sel.add("final_seq", sel.all("genTot_seq", "recoTot_seq", "removeMisses", "removeFakes", "rem_uf_fakes"))
                    print("Subjets ", events_corr.SubGenJetAK8)
                    genjet1 = ak.firsts(events_corr[sel.all("final_seq")].GenJetAK8[:,0:])
                    genjet2 = ak.firsts(events_corr[sel.all("final_seq")].GenJetAK8[:,1:])
                    groomed_genjet0 = get_gen_sd_mass_jet(genjet1, events_corr[sel.all("final_seq")].SubGenJetAK8)
                    groomed_genjet1 = get_gen_sd_mass_jet(genjet2, events_corr[sel.all("final_seq")].SubGenJetAK8)
                    groomed_gen_dijet = ak.concatenate([ak.unflatten(groomed_genjet0, 1),  ak.unflatten(groomed_genjet1, 1)], axis=1)
                    groomed_gen_dijet = ak.flatten(groomed_gen_dijet, axis=1)
                    gen_dijet = ak.flatten(events_corr[sel.all("final_seq")].GenJetAK8[:,:2], axis=1)
                    dijet = ak.flatten(events_corr[sel.all("final_seq")].FatJet[:,:2], axis=1)
                    dijet_weights = np.repeat(weights[sel.all("final_seq")], 2)
                    self.weights[jetsyst] = Weights(len(dijet_weights))
                    if "L1PreFiringWeight" in events_corr.fields and "L1PreFiringWeight" in self.systematics:                
                        prefiringNom, prefiringUp, prefiringDown = GetL1PreFiringWeight(events_corr[sel.all("final_seq")])
                        print("prefiring wi=eights ", len(prefiringNom))
                        self.weights[jetsyst].add("L1prefiring", weight=np.repeat(prefiringNom, 2), 
                                               weightUp=np.repeat(prefiringUp, 2), 
                                               weightDown=np.repeat(prefiringDown, 2),
                                   )
                    if "PUSF" in self.systematics:
                        puNom, puUp, puDown = GetPUSF(events_corr[sel.all("final_seq")], IOV)
                        self.weights[jetsyst].add("PUSF", weight=np.repeat(puNom, 2), weightUp=np.repeat(puUp, 2),
                                           weightDown=np.repeat(puDown, 2),) 
                    if 'herwig' in dataset or 'madgraph' in dataset:
                        pdfNom, pdfUp, pdfDown = GetPDFWeights(events_corr[sel.all("final_seq")])
                        self.weights[jetsyst].add("PDF", weight=np.repeat(pdfNom, 2), weightUp=np.repeat(pdfUp, 2),
                                           weightDown=np.repeat(pdfDown, 2),) 
                        q2Nom, q2Up, q2Down = GetQ2Weights(events_corr[sel.all("final_seq")])
                        self.weights[jetsyst].add("Q2", weight=np.repeat(q2Nom, 2), weightUp=np.repeat(q2Up, 2),
                                           weightDown=np.repeat(q2Down, 2),) 
                    if not self.jk:
                        out["jet_dr_gen_subjet"].fill(syst=jetsyst, 
                                            dr=events_corr[sel.all("final_seq")].SubGenJetAK8[:,0].delta_r(events_corr[sel.all("final_seq")].FatJet[:,0]),
                                                  weight=weights[sel.all("final_seq")])
                    #            print("Check for none values", ak.any(ak.is_none(dijetEvents, axis = -1)))     
                    #### Final GEN plots
                    out['ptgen_mgen_u'].fill( syst=jetsyst, jk=jk_index, ptgen=gen_dijet.pt, mgen=gen_dijet.mass, weight=self.weights[jetsyst].weight() )
                    out['ptgen_mgen_g'].fill( syst=jetsyst, jk=jk_index, ptgen=gen_dijet.pt, mgen=groomed_gen_dijet.mass, weight=self.weights[jetsyst].weight() )
                    out["response_matrix_u"].fill( syst=jetsyst, jk=jk_index,  ptreco=dijet.pt, mreco=dijet.mass,
                                                  ptgen=gen_dijet.pt, mgen=gen_dijet.mass,
                                                  weight=self.weights[jetsyst].weight())
                    out["response_matrix_g"].fill( syst=jetsyst, jk=jk_index,ptreco=dijet.pt, mreco=dijet.msoftdrop,
                                                  ptgen=gen_dijet.pt, mgen=groomed_gen_dijet.mass, weight=self.weights[jetsyst].weight())
                    dijet_weights = self.weights[jetsyst].weight()
                    if jetsyst == "nominal":
                        for syst in self.weights[jetsyst].variations:
                            print("Weight variation: ", syst)
                            #fill nominal, up, and down variations for each
                            out['ptgen_mgen_u'].fill( syst=syst, jk=jk_index, ptgen=gen_dijet.pt, mgen=gen_dijet.mass, weight=self.weights[jetsyst].weight(syst) )
                            out['ptgen_mgen_g'].fill( syst=syst, jk=jk_index, ptgen=gen_dijet.pt, mgen=groomed_gen_dijet.mass, 
                                                          weight=self.weights[jetsyst].weight(syst) )           
                            out["response_matrix_u"].fill(syst=syst, jk=jk_index,
                                                   ptreco=dijet.pt, mreco=dijet.mass,
                                                   ptgen=gen_dijet.pt, mgen=gen_dijet.mass, weight=self.weights[jetsyst].weight(syst))
                            out["response_matrix_g"].fill(syst=syst, jk=jk_index, ptreco=dijet.pt, mreco=dijet.msoftdrop,
                                                          ptgen=gen_dijet.pt, mgen=groomed_gen_dijet.mass, weight=self.weights[jetsyst].weight(syst))
                            out["ptreco_mreco_u"].fill(syst=syst, jk=jk_index, ptreco=dijet.pt, mreco=dijet.mass, weight=self.weights[jetsyst].weight(syst) )
                            out["ptreco_mreco_g"].fill(syst=syst, jk=jk_index, ptreco=dijet.pt, mreco=dijet.msoftdrop, weight=self.weights[jetsyst].weight(syst) )
                            
                            # if ak.sum(fakes)>0:
                            #     out["fakes"].fill(syst=syst, jk=jk_index, ptreco = fake_dijets.pt, mreco = fake_dijets.mass, weight=fake_weights.weight(syst))
                            #     out["fakes_g"].fill(syst=syst, jk=jk_index, ptreco = fake_dijets.pt, mreco = fake_dijets.msoftdrop, weight=fake_weights.weight(syst))
    
                    #weird = (reco_jet.msoftdrop/groomed_gen_jet.mass > 2.0) & (reco_jet.msoftdrop > 10.)
                    weird = ((dijet.msoftdrop/groomed_gen_dijet.mass) > 2.0) & (dijet.msoftdrop > 10.)
                    print("Number of what ashley called weird events ", ak.sum(weird)) 
                    #### FIX
                    #             recosubjets = events.SubJet           
                    #             subjet1 = ak.flatten(events.FatJet).subjets[:,0]
                    #             subjet2 = ak.flatten(events.FatJet).subjets[:,1]
                    #             gensubjet1=gensubjets.nearest(subjet1)
                    #             drsub1=gensubjets.delta_r(gensubjets.nearest(subjet1))
                    #             gensubjet1=gensubjets.nearest(subjet2)
                    #             drsub1=gensubjets.delta_r(gensubjets.nearest(subjet2))
                    
                    #             out["dr_reco_to_gen_subjet"].fill(
                    #                                                      dr=drsub1[~ak.is_none(drsub1) & ~ak.is_none(drsub2)], 
                    #                                                      weight=weights[~ak.is_none(drsub1) & ~ak.is_none(drsub2)])
                    #             out["dr_reco_to_gen_subjet"].fill(
                    #                                                      dr=drsub2[~ak.is_none(drsub1) & ~ak.is_none(drsub2)], 
                    #                                                      weight=weights[~ak.is_none(drsub1) & ~ak.is_none(drsub2)])
                    
                    #flavour --> 21 is gluon
                    # if jetsyst == "nominal":
                    #     genjet1 = events_corr.GenJetAK8[:,0]
                    #     genjet2 = events_corr.GenJetAK8[:,1]
                    #     jet1 = events_corr.FatJet[:,0]
                    #     jet2 = events_corr.FatJet[:,1]
                    #     jet2_g     = jet2[np.abs(genjet2.partonFlavour) == 21]
                    #     jet2_uds   = jet2[np.abs(genjet2.partonFlavour) < 4]
                    #     jet2_c     = jet2[np.abs(genjet2.partonFlavour) == 4]
                    #     jet2_b     = jet2[np.abs(genjet2.partonFlavour) == 5]
                    #     jet2_other = jet2[(np.abs(genjet2.partonFlavour) > 5) & (np.abs(genjet2.partonFlavour) != 21)]
                        
                    #     jet1_g     = jet1[np.abs(genjet1.partonFlavour) == 21]
                    #     jet1_uds   = jet1[np.abs(genjet1.partonFlavour) < 4]
                    #     jet1_c     = jet1[np.abs(genjet1.partonFlavour) == 4]
                    #     jet1_b     = jet1[np.abs(genjet1.partonFlavour) == 5]
                    #     jet1_other = jet1[(np.abs(genjet1.partonFlavour) > 5) & (np.abs(genjet1.partonFlavour) != 21)]
                        
                        #        #make central and forward categories instead of jet1 jet2
                        #### Plots for gluon purity studies
                        
                        # out['jet_mass'].fill(jetNumb = "jet1", partonFlav = "Gluon",  mreco = jet1_g.mass,
                        #                      #weight = trijetEvents.Generator.weight
                        #                  )
                        # out['jet_mass'].fill(jetNumb = "jet1", partonFlav = "UDS",    mreco = jet1_uds.mass,
                        #                      #weight = trijetEvents.Generator.weight
                        #                  )
                        # out['jet_mass'].fill(jetNumb = "jet1", partonFlav = "Charm",  mreco = jet1_c.mass,
                        #                      #weight = trijetEvents.Generator.weight
                        #                  )
                        # out['jet_mass'].fill(jetNumb = "jet1", partonFlav = "Bottom", mreco = jet1_b.mass,
                        #                      #weight = trijetEvents.Generator.weight
                        #                  )
                        # out['jet_mass'].fill(jetNumb = "jet1", partonFlav = "Other",  mreco = jet1_other.mass,
                        #                      #weight = trijetEvents.Generator.weight
                        #                  )
                        # out['jet_mass'].fill(jetNumb = "jet2", partonFlav = "Gluon",  mreco = jet2_g.mass,
                        #                      #weight = trijetEvents.Generator.weight
                        #                  )
                        # out['jet_mass'].fill(jetNumb = "jet2", partonFlav = "UDS",    mreco = jet2_uds.mass,
                        #                      #weight = trijetEvents.Generator.weight
                        #                  )
                        # out['jet_mass'].fill(jetNumb = "jet2", partonFlav = "Charm",  mreco = jet2_c.mass,
                        #                      #weight = trijetEvents.Generator.weight
                        #                  )
                        # out['jet_mass'].fill(jetNumb = "jet2", partonFlav = "Bottom", mreco = jet2_b.mass,
                        #                      #weight = trijetEvents.Generator.weight
                        #                  )
                        # out['jet_mass'].fill(jetNumb = "jet2", partonFlav = "Other",  mreco = jet2_other.mass,
                        #                      #weight = trijetEvents.Generator.weight
                        #                  )
                        
                        # out['jet_pt'].fill(jetNumb = "jet1", partonFlav = "Gluon",  ptreco = jet1_g.pt,
                        #                    #weight = trijetEvents.Generator.weight
                        #                )
                        # out['jet_pt'].fill(jetNumb = "jet1", partonFlav = "UDS",    ptreco = jet1_uds.pt,
                        #                    #weight = trijetEvents.Generator.weight
                        #                )
                        # out['jet_pt'].fill(jetNumb = "jet1", partonFlav = "Charm",  ptreco = jet1_c.pt,
                        #                    #weight = trijetEvents.Generator.weight
                        #                )
                        # out['jet_pt'].fill(jetNumb = "jet1", partonFlav = "Bottom", ptreco = jet1_b.pt,
                        #                    #weight = trijetEvents.Generator.weight
                        #                )
                        # out['jet_pt'].fill(jetNumb = "jet1", partonFlav = "Other",  ptreco = jet1_other.pt,
                        #                    #weight = trijetEvents.Generator.weight
                        #                )
                        # out['jet_pt'].fill(jetNumb = "jet2", partonFlav = "Gluon",  ptreco = jet2_g.pt,
                        #                #weight = trijetEvents.Generator.weight
                        #                )
                        # out['jet_pt'].fill(jetNumb = "jet2", partonFlav = "UDS",    ptreco = jet2_uds.pt,
                        #                    #weight = trijetEvents.Generator.weight
                        #                )
                        # out['jet_pt'].fill(jetNumb = "jet2", partonFlav = "Charm",  ptreco = jet2_c.pt,
                        #                    #weight = trijetEvents.Generator.weight
                        #                )
                        # out['jet_pt'].fill(jetNumb = "jet2", partonFlav = "Bottom", ptreco = jet2_b.pt,
                        #                    #weight = trijetEvents.Generator.weight
                        #                )
                        # out['jet_pt'].fill(jetNumb = "jet2", partonFlav = "Other",  ptreco = jet2_other.pt,
                        #                    #weight = trijetEvents.Generator.weight
                        #                )
                        
                        # out['jet_eta'].fill(jetNumb = "jet1", partonFlav = "Gluon",  eta = np.abs(jet1_g.eta),
                        #                     #weight = trijetEvents.Generator.weight
                        #                 )
                        # out['jet_eta'].fill(jetNumb = "jet1", partonFlav = "UDS",    eta = np.abs(jet1_uds.eta),
                        #                     #weight = trijetEvents.Generator.weight
                        #                 )
                        # out['jet_eta'].fill(jetNumb = "jet1", partonFlav = "Charm",  eta = np.abs(jet1_c.eta),
                        #                     #weight = trijetEvents.Generator.weight
                        #                 )
                        # out['jet_eta'].fill(jetNumb = "jet1", partonFlav = "Bottom", eta = np.abs(jet1_b.eta),
                        #                     #weight = trijetEvents.Generator.weight
                        #                 )
                        # out['jet_eta'].fill(jetNumb = "jet1", partonFlav = "Other",  eta = np.abs(jet1_other.eta),
                        #                     #weight = trijetEvents.Generator.weight
                        #                 )
                        # out['jet_eta'].fill(jetNumb = "jet2", partonFlav = "Gluon",  eta = np.abs(jet2_g.eta),
                        #                     #weight = trijetEvents.Generator.weight
                        #                 )
                        # out['jet_eta'].fill(jetNumb = "jet2", partonFlav = "UDS",    eta = np.abs(jet2_uds.eta),
                        #                     #weight = trijetEvents.Generator.weight
                        #                 )
                        # out['jet_eta'].fill(jetNumb = "jet2", partonFlav = "Charm",  eta = np.abs(jet2_c.eta),
                        #                     #weight = trijetEvents.Generator.weight
                        #                 )
                        # out['jet_eta'].fill(jetNumb = "jet2", partonFlav = "Bottom", eta = np.abs(jet2_b.eta),
                        #                     #weight = trijetEvents.Generator.weight
                        #                 )
                        # out['jet_eta'].fill(jetNumb = "jet2", partonFlav = "Other",  eta = np.abs(jet2_other.eta),
                        #                     #weight = trijetEvents.Generator.weight
                        #                 )
                        # out['cutflow']['nGluonJets '+ jetsyst] += (len(ak.flatten(dijet[np.abs(gen_dijet.partonFlavour) == 21].pt, axis=-1)))
                        # out['cutflow']['nJets '+ jetsyst] += (len(ak.flatten(jet1.pt, axis=-1)))
                #### Final RECO plots
                negMSD = ak.flatten(events_corr[sel.all("recoTot_seq")].FatJet[:,:2].msoftdrop<0, axis=1)
                print("Number of negative softdrop values ", ak.sum(negMSD))
                out['cutflow'][dataset]['nEvents failing softdrop condition'] += ak.sum(negMSD)
                out["ptreco_mreco_u"].fill(syst=jetsyst, jk=jk_index, ptreco=dijet.pt, mreco=dijet.mass, weight=dijet_weights )
                out["ptreco_mreco_g"].fill(syst=jetsyst, jk=jk_index, ptreco=dijet.pt, mreco=dijet.msoftdrop, weight=dijet_weights)
                for name in sel.names:
                    out["cutflow"][dataset][name] += sel.all(name).sum()
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

