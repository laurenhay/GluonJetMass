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
from coffea.analysis_tools import Weights
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
                'cutflow':            processor.defaultdict_accumulator(int),
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
        out['cutflow']['nEvents initial'] += (len(events.FatJet))
        #####################################
        #### Find the IOV from the dataset name
        #####################################
        IOV = ('2016APV' if ( any(re.findall(r'APV',  dataset)) or any(re.findall(r'HIPM', dataset)))
               else '2018'    if ( any(re.findall(r'UL18', dataset)) or any(re.findall(r'UL2018',    dataset)))
               else '2017'    if ( any(re.findall(r'UL17', dataset)) or any(re.findall(r'UL2017',    dataset)))
               else '2016')
        out['cutflow']['nEvents'+IOV+dataset] += (len(events.FatJet))
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
            print("# of selected ", ak.sum(jk_sel))
            #### Choosing our jk fraction and protecting against negative softdrop
            events_jk = events[jk_sel & ~(ak.any(events.FatJet.msoftdrop<0, axis=-1))]
            del jk_sel
            #####################################
            #### Apply HEM veto
            #### Add values needed for jet corrections
            #####################################
            print("Fat jets before jetid ", events_jk.FatJet.pt)
            # if IOV == '2018' and self.hem:
            #     print("Doing hem")
            #     events_jk = events_jk[HEMVeto(events_jk.FatJet, events.run)]
            #### require at least one fat jet and one subjet so corrections do not fail
            events_jk=events_jk[ak.any(events_jk.FatJet[:,:2].jetId > 1, axis=-1)]
            events_jk = events_jk[(ak.num(events_jk.SubJet) > 0) & (ak.num(events_jk.FatJet) > 0)]
            # print("Fat jets after jetid ", events_jk.FatJet.pt)
            # print("Nearest muon to FatJet ", events.FatJet.delta_r(events.FatJet.nearest(events.Muon)))
            # events = ak.all(events.FatJet.delta_r(events.Muon.nearest(events.FatJet)) > 0.2, axis = -1)
            # out['cutflow']['nEvents after tight jet id and muon iso '] += (len(events.FatJet))
            FatJet=events_jk.FatJet
            FatJet["p4"] = ak.with_name(events_jk.FatJet[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
            # print("Fat jets before correcting ", FatJet)
            if len(FatJet) < 1:
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
            print("Nevents after removing empty fatjets and subjets", len(events_jk))
            out["sdmass_orig"].fill(jk=jk_index, ptreco=ak.flatten(events_jk[(ak.num(events_jk.FatJet) > 1)].FatJet[:,:2].pt, axis=1), mreco=ak.flatten(events_jk[(ak.num(events_jk.FatJet) > 1)].FatJet[:,:2].msoftdrop, axis=1))
            corrected_fatjets = GetJetCorrections(FatJet, events_jk, era, IOV, isData=not self.do_gen)
            corrected_fatjets["msoftdrop"] = GetCorrectedSDMass(events_jk, era, IOV, isData=not self.do_gen)
            FatJets_ak8 = FatJet
            FatJets_ak8["mass"] = FatJet.msoftdrop
            corrrected_fatjets_ak8 = GetJetCorrections(FatJets_ak8, events_jk, era, IOV, isData=not self.do_gen)
            out["sdmass_ak8corr"].fill(jk=jk_index, ptreco=ak.flatten(corrrected_fatjets_ak8[(ak.num(corrrected_fatjets_ak8) > 1)][:,:2].pt, axis=1), mreco=ak.flatten(corrrected_fatjets_ak8[(ak.num(corrrected_fatjets_ak8) > 1)][:,:2].mass, axis=1))
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
                print("Getting sources: ", [unc_src for unc_src in self.jet_systematics if ("JES_"+unc_src in corrected_fatjets.fields)])
                print("Out of avail corrections: ", corrected_fatjets.fields)
                avail_srcs = [unc_src for unc_src in self.jet_systematics if ("JES_"+unc_src in corrected_fatjets.fields)]
                print("Input jet syst", self.jet_systematics)
                for unc_src in avail_srcs:
                    print(corrected_fatjets["JES_"+unc_src])
                    corrected_fatjets["JES_"+unc_src].up = applyjmsSF(IOV, applyjmrSF(IOV,corrected_fatjets["JES_"+unc_src].up))
                    corrected_fatjets["JES_"+unc_src].down = applyjmsSF(IOV, applyjmrSF(IOV,corrected_fatjets["JES_"+unc_src].down))
                    jet_corrs.update({
                        unc_src+"Up":corrected_fatjets["JES_"+unc_src].up,
                        unc_src+"Down":corrected_fatjets["JES_"+unc_src].down, })
            print("Final jet corrs to run over: ", jet_corrs)
            for jetsyst in jet_corrs.keys():
                print("Adding ", jetsyst, " values ", jet_corrs[jetsyst], " to output")
                events_corr = ak.with_field(events_jk, jet_corrs[jetsyst], "FatJet")
                weights = np.ones(len(events_corr))
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
                    events_corr = events_corr [lumi_mask & (ak.num(events_corr.FatJet) >= 1)]
                    trigsel, psweights = applyPrescales(events_corr, year = IOV)
                    weights=psweights
                    # print("Trigger: len of events ", len(events_corr), "len of weights ", len(trigsel))
                    # print("Weights w/ prescales ", weights)
                    events_corr = events_corr[trigsel]
                    weights = weights[trigsel]
                    out['cutflow']['nEvents after trigger sel '+jetsyst] += (len(events.FatJet))
                    
                #### NPV selection
                # print("Npvs ", events_corr.PV.fields)
                # sel.add("npv", events.PV.npvsGood>0)
    
                #####################################
                #### Gen Jet Selection
                #### see CMS PAS SMP-20-010 for selections
                ####################################
                if (self.do_gen):
                    print("DOING GEN")
                    #### Select events with at least 2 jets
                    if not self.jk:
                        out["njet_gen"].fill(syst = jetsyst, n=ak.num(events_corr.GenJetAK8), 
                                         weight = weights )
                    pt_cut_gen = ak.all(events_corr.GenJetAK8.pt > 160., axis = -1) ### 80% of reco pt cut
                    weights = weights[(ak.num(events_corr.GenJetAK8) > 1) & pt_cut_gen]
                    events_corr = events_corr[(ak.num(events_corr.GenJetAK8) > 1) & pt_cut_gen]
                    GenJetAK8 = events_corr.GenJetAK8
                    GenJetAK8['p4']= ak.with_name(events_corr.GenJetAK8[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
                    rap_cut_gen = ak.all(np.abs(getRapidity(GenJetAK8.p4)) < self.ycut, axis = -1)
                    # print("Length of gen rap ", len(ak.flatten(getRapidity(GenJetAK8[:,:2].p4), axis=1)))
                    # print("Length of weights ", len(np.repeat(weights, 2)))
                    if not self.jk:
                        out["jet_rap_gen"].fill(syst = jetsyst, rapidity=ak.flatten(getRapidity(GenJetAK8[:,:2].p4), axis=1),
                                             weight=np.repeat(weights, 2))
                        out["jet_phi_gen"].fill(syst=jetsyst, phi=ak.flatten(GenJetAK8[:,:2].phi, axis=1), weight=np.repeat(weights, 2))  
                    #### Apply kinematic and 2 jet requirement immediately so that dphi and asymm can be calculated
                    weights = weights[rap_cut_gen]
                    events_corr = events_corr[rap_cut_gen]
                    out['cutflow']['nEvents after gen rapidity,pT, and nJet selection '+jetsyst] += (len(events_corr.FatJet))
                    #### get dphi and pt asymm selections  
                    genjet1 = events_corr.GenJetAK8[:,0]
                    genjet2 = events_corr.GenJetAK8[:,1]
                    dphi12_gen = np.abs(genjet1.delta_phi(genjet2))
                    dphi12_gen_sel = dphi12_gen > 2.
                    asymm_gen  = np.abs(genjet1.pt - genjet2.pt)/(genjet1.pt + genjet2.pt)
                    asymm_gen_sel = asymm_gen < 0.3
                    #             sel.add("gen_dphi_sel", dphi12_gen)
                    #             sel.add("gen_asymm_sel", asymm_gen_sel)
                    
                    #### N-1 plots
                    if not self.jk:
                        out["asymm_gen"].fill(syst=jetsyst,frac=asymm_gen[dphi12_gen_sel], weight=weights[dphi12_gen_sel])  
                        out["dphi_gen"].fill(syst=jetsyst, dphi=dphi12_gen[asymm_gen_sel], weight=weights[asymm_gen_sel])  
                    events_corr = events_corr[dphi12_gen_sel & asymm_gen_sel]
                    weights = weights[dphi12_gen_sel & asymm_gen_sel]
                    out['cutflow']['nEvents after gen dphi and ptasymm selection '+jetsyst] += (len(events_corr.FatJet))
                    #misses = gen but no reco
                    matches = ak.all(events_corr.GenJetAK8.delta_r(events_corr.GenJetAK8.nearest(events_corr.FatJet)) < 0.2, axis = -1)
                    #### have found some events that are missing reco msoftdrop --- add to misses
                    print("Nevents missing masses ", ak.sum(ak.any(ak.is_none(events_corr.FatJet.msoftdrop, axis=-1), axis=-1) | ak.any(ak.is_none(events_corr.FatJet.mass, axis=-1), axis=-1)))
                    misses = ~matches | ak.any(ak.is_none(events_corr.FatJet.msoftdrop, axis=-1), axis=-1) | ak.any(ak.is_none(events_corr.FatJet.mass, axis=-1), axis=-1)
                    out["misses"].fill(syst=jetsyst, jk=jk_index, ptgen = ak.flatten(events_corr[misses].GenJetAK8[:,:2].pt, axis=1), 
                                       mgen = ak.flatten(events_corr[misses].GenJetAK8[:,:2].mass, axis=1))
                    out['cutflow']['misses '+jetsyst] += (len(events_corr[misses].FatJet))
                    events_corr = events_corr[~misses]
                    weights = weights[~misses]
                    out['cutflow']['nEvents after deltaR matching (remove misses) ' + jetsyst] += len(events_corr.FatJet)
                    if len(events_corr)<1:
                        print("No events after miss removal")
                        return out
                #####################################
                #### Reco Jet Selection
                #################################### 
                #### Apply pt and rapidity cuts
                if not self.jk:
                    out["njet_reco"].fill(syst = jetsyst, n=ak.num(events_corr.FatJet), 
                                         weight = weights)
                print("PT vals ", events_corr.FatJet.pt)
                pt_cut_reco = (ak.all(events_corr.FatJet.pt > self.ptcut, axis = -1))
                weights = weights[(ak.num(events_corr.FatJet) > 1) & pt_cut_reco]
                events_corr = events_corr[(ak.num(events_corr.FatJet) > 1) & pt_cut_reco]
                weights = weights[ApplyVetoMap(IOV, events_corr.FatJet[:,0], mapname='jetvetomap') & ApplyVetoMap(IOV, events_corr.FatJet[:,1], mapname='jetvetomap')]
                events_corr = events_corr[ApplyVetoMap(IOV, events_corr.FatJet[:,0], mapname='jetvetomap') & ApplyVetoMap(IOV, events_corr.FatJet[:,1], mapname='jetvetomap')]
                print("nevents after jet veto map ", len(events_corr.FatJet))
                out['cutflow']['nEvents afterjet veto map '] += (len(events_corr.FatJet))
                FatJet = events_corr.FatJet
                FatJet["p4"] = ak.with_name(events_corr.FatJet[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
                rap_cut_reco = ak.all(np.abs(getRapidity(FatJet.p4)) < self.ycut, axis = -1)

                if not self.jk:
                    out["jet_rap_reco"].fill(syst = jetsyst, rapidity=ak.to_numpy(ak.flatten(getRapidity(FatJet[:,:2].p4)), allow_missing=True),
                                             weight=np.repeat(weights, 2))
                    out["jet_phi_reco"].fill(syst=jetsyst, phi=ak.flatten(FatJet[:,:2].phi, axis=1), weight=np.repeat(weights, 2)) 
                #### Add cut on softdrop mass as done in previous two papers --> need to very with JMS/JMR studies
                sdm_cut = (ak.all(events_corr.FatJet.msoftdrop > 10., axis = -1))
                weights = weights[rap_cut_reco & sdm_cut]
                events_corr = events_corr[rap_cut_reco & sdm_cut]
                out['cutflow']['nEvents after reco kine selection + sd mass sel'+jetsyst] += (len(events_corr.FatJet))
                #### get dphi and pt asymm selections
                jet1 = events_corr.FatJet[:,0]
                jet2 = events_corr.FatJet[:,1]
                dphi12 = (np.abs(jet1.delta_phi(jet2)) > 2.)
                asymm = np.abs(jet1.pt - jet2.pt)/(jet1.pt + jet2.pt)
                if not self.jk:
                    out["dphi_reco"].fill(syst=jetsyst, dphi = dphi12, weight = weights)
                    out["asymm_reco"].fill(syst=jetsyst, frac = asymm, weight=weights)
                asymm_reco_sel = asymm < 0.3
                events_corr = events_corr[asymm_reco_sel & dphi12]
                weights = weights[asymm_reco_sel & dphi12]
                out['cutflow']['nEvents after reco pT assym. and eta selection '+jetsyst] += (len(events_corr.FatJet))
                dijet_weights = np.repeat(weights, 2)
                
                
                ####  Final RECO selection
                FatJet = events_corr.FatJet
                FatJet["p4"] = ak.with_name(events_corr.FatJet[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
                dijet = ak.flatten(FatJet[:,:2], axis=1)
                #### match jets, get syst weights, and fill final plots
                if self.do_gen:
                    #### check for empty softdropmass values
                    print("Structure of matched gen ", events_corr.FatJet.matched_gen)
                    fakes = ak.any(ak.is_none(events_corr.FatJet.matched_gen, axis = -1), axis = -1)
                    if len(weights[fakes])>0:
                        fake_dijets = ak.flatten(events_corr[fakes].FatJet[:,:2], axis=1)
                        fake_weights = Weights(len(np.repeat(weights[fakes], 2)))
                        self.weights[jetsyst] = Weights(len(dijet_weights))
                        print("Intital gen weights: ", dijet_weights)
                        fake_weights.add('fakeWeight', np.repeat(weights[fakes], 2))
                        if "L1PreFiringWeight" in events_corr.fields and "L1PreFiringWeight" in self.systematics:                
                            prefiringNom, prefiringUp, prefiringDown = GetL1PreFiringWeight(events_corr[fakes])
                            fake_weights.add("L1prefiring", weight=np.repeat(prefiringNom, 2), 
                                                   weightUp=np.repeat(prefiringUp, 2), 
                                                   weightDown=np.repeat(prefiringDown, 2),
                                       )
                        if "PUSF" in self.systematics:
                            puUp, puDown, puNom = GetPUSF(events_corr[fakes], IOV)
                            fake_weights.add("PUSF", weight=np.repeat(puNom, 2), weightUp=np.repeat(puUp, 2),
                                               weightDown=np.repeat(puDown, 2),) 
                        if 'herwig' in dataset or 'madgraph' in dataset:
                            pdfNom, pdfUp, pdfDown = GetPDFWeights(events_corr[fakes])
                            print("Fakes pdf weights ", pdfNom, " shape ", len(pdfNom))
                            fake_weights.add("PDF", weight=np.repeat(pdfNom, 2), weightUp=np.repeat(pdfUp, 2),
                                               weightDown=np.repeat(pdfDown, 2),) 
                            q2Nom, q2Up, q2Down = GetQ2Weights(events_corr[fakes])
                            print("Fakes q2 weights ", pdfNom, " shape ", len(pdfNom))
                            fake_weights.add("Q2", weight=np.repeat(q2Nom, 2), weightUp=np.repeat(q2Up, 2),
                                               weightDown=np.repeat(q2Down, 2),) 
                        print("Fake pt ", fake_dijets.pt, " fake mass ", fake_dijets.mass, " fake weights ", fake_weights.weight(), 'and fake sd mass ', fake_dijets.msoftdrop)
                        out["fakes"].fill(syst=jetsyst, jk = jk_index, ptreco = fake_dijets.pt, mreco = fake_dijets.mass, weight = fake_weights.weight())
                        out["fakes_g"].fill(syst=jetsyst, jk = jk_index, ptreco = fake_dijets.pt, mreco = fake_dijets.msoftdrop, weight = fake_weights.weight())
                        if not self.jk:
                            out['fakes_eta_phi'].fill(syst=jetsyst, phi = fake_dijets.phi, eta = fake_dijets.eta, weight=fake_weights.weight())
                    ##### fakes = reco but no gen
                    out['cutflow']['fakes '+jetsyst] += (len(events_corr[fakes].FatJet))
                    matched_reco = ~fakes
                    events_corr = events_corr[matched_reco]
                    if len(events_corr)<1: return out
                    weights = weights[matched_reco]
                    out['cutflow']['nEvents after gen matching (remove fakes) '+jetsyst] += (len(events_corr.FatJet))
                    uf = (ak.any(200. > events_corr.GenJetAK8.pt, axis = -1))
                    uf_dijets = ak.flatten(events_corr[uf].FatJet[:,:2], axis=1)
                    uf_weights = np.repeat(weights[uf], 2)
                    events_corr = events_corr[~uf]
                    weights = weights[~uf]
                    print("Lengths of underflow dijets ", len(uf_dijets), " length of underflow weights ", len(uf_weights))
                    out["underflow"].fill(syst=jetsyst, jk = jk_index, ptreco = uf_dijets.pt, mreco = uf_dijets.mass, weight = uf_weights)
                    out["underflow_g"].fill(syst=jetsyst, jk = jk_index, ptreco = uf_dijets.pt, mreco = uf_dijets.msoftdrop, weight = uf_weights)
                    print("Number of underflow gen ", len(uf))
                    out['cutflow']['nEvents after gen matching (remove fakes) '+jetsyst] += (len(events_corr.FatJet))
                    #### Get gen subjets and sd gen jets
                    print("Subjets ", events_corr.SubGenJetAK8)
                    groomed_genjet0 = get_gen_sd_mass_jet(events_corr.GenJetAK8[:,0], events_corr.SubGenJetAK8)
                    groomed_genjet1 = get_gen_sd_mass_jet(events_corr.GenJetAK8[:,1], events_corr.SubGenJetAK8)
                    print("Groomed jet 0 ", groomed_genjet0.pt, " type ", groomed_genjet0.type)
                    print("Groomed jet 1 ", groomed_genjet1.pt)
                    groomed_gen_dijet = ak.concatenate([ak.unflatten(groomed_genjet0, 1),  ak.unflatten(groomed_genjet1, 1)], axis=1)
                    groomed_gen_dijet = ak.flatten(groomed_gen_dijet, axis=1)
                    dijet = ak.flatten(events_corr.FatJet[:,:2], axis =1)
                    print("Fat jet after flattening whole " , dijet.pt[:4])
                    gen_dijet = ak.flatten(events_corr.GenJetAK8[:,:2], axis=1)
                    dijet_weights = np.repeat(weights, 2)
                    self.weights[jetsyst] = Weights(len(dijet_weights))
                    # print("Intital gen weights: ", dijet_weights)
                    self.weights[jetsyst].add('dijetWeight', weight=dijet_weights)
                    if "L1PreFiringWeight" in events_corr.fields and "L1PreFiringWeight" in self.systematics:                
                        prefiringNom, prefiringUp, prefiringDown = GetL1PreFiringWeight(events_corr)
                        self.weights[jetsyst].add("L1prefiring", weight=np.repeat(prefiringNom, 2), 
                                               weightUp=np.repeat(prefiringUp, 2), 
                                               weightDown=np.repeat(prefiringDown, 2),
                                   )
                    if "PUSF" in self.systematics:
                        puNom, puUp, puDown = GetPUSF(events_corr, IOV)
                        self.weights[jetsyst].add("PUSF", weight=np.repeat(puNom, 2), weightUp=np.repeat(puUp, 2),
                                           weightDown=np.repeat(puDown, 2),) 
                    if 'herwig' in dataset or 'madgraph' in dataset:
                        pdfNom, pdfUp, pdfDown = GetPDFWeights(events_corr)
                        self.weights[jetsyst].add("PDF", weight=np.repeat(pdfNom, 2), weightUp=np.repeat(pdfUp, 2),
                                           weightDown=np.repeat(pdfDown, 2),) 
                        q2Nom, q2Up, q2Down = GetQ2Weights(events_corr)
                        self.weights[jetsyst].add("Q2", weight=np.repeat(q2Nom, 2), weightUp=np.repeat(q2Up, 2),
                                           weightDown=np.repeat(q2Down, 2),) 
                    if not self.jk:
                        out["jet_dr_gen_subjet"].fill(syst=jetsyst, 
                                            dr=events_corr.SubGenJetAK8[:,0].delta_r(events_corr.FatJet[:,0]),
                                                  weight=weights)
                        out["dijet_dr_reco_to_gen"].fill(syst=jetsyst, 
                                                         dr=dijet.delta_r(gen_dijet), weight=self.weights[jetsyst].weight())
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
                            print("Number of fakes ", ak.sum(fakes), " from ", fakes)
                            if ak.sum(fakes)>0:
                                out["fakes"].fill(syst=syst, jk=jk_index, ptreco = fake_dijets.pt, mreco = fake_dijets.mass, weight=fake_weights.weight(syst))
                                out["fakes_g"].fill(syst=syst, jk=jk_index, ptreco = fake_dijets.pt, mreco = fake_dijets.msoftdrop, weight=fake_weights.weight(syst))
    
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
                negMSD = ak.flatten(events_corr.FatJet[:,:2].msoftdrop<0, axis=1)
                print("Number of negative softdrop values ", ak.sum(negMSD))
                out['cutflow']['nEvents failing softdrop condition'] += ak.sum(negMSD)
                dijet = ak.flatten(events_corr.FatJet[:,:2], axis =1)
                out['cutflow']['nEvents final selection '+ jetsyst] += (len(events_corr.FatJet))
                out["ptreco_mreco_u"].fill(syst=jetsyst, jk=jk_index, ptreco=dijet.pt, mreco=dijet.mass, weight=dijet_weights )
                out["ptreco_mreco_g"].fill(syst=jetsyst, jk=jk_index, ptreco=dijet.pt, mreco=dijet.msoftdrop, weight=dijet_weights)
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

