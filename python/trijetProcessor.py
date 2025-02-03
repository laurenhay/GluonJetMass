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
    def __init__(self, ptcut = 200., ycut = 2.5, btag = 'None', data = False, jet_systematics = ['nominal', 'JER',"HEM"], systematics = ['L1PreFiringWeight', 'PUSF'], jk=False, jk_range = None):
        self.ptcut = ptcut
        self.ycut = ycut
        self.btag = btag
        self.do_gen = not data
        self.jk = jk
        self.jk_range = jk_range
        self.systematics = systematics
        self.jet_systematics = jet_systematics
        print("Data: ", data, " gen ", self.do_gen)
        jet_cat = hist.axis.StrCategory([], growth=True, name="jetNumb", label="Jet")
        parton_cat = hist.axis.StrCategory([],growth=True,name="partonFlav", label="Parton Flavour")
        syst_cat = hist.axis.StrCategory([], growth=True, name='syst', label="Systematic")
        #### if using specific bin edges use hist.axis.Variable() instead
        mgen_bin_edges = np.array([0,10,20,40,60,80,100,120,140,160,300,1300])
        mreco_bin_edges = np.sort(np.append(mgen_bin_edges,[(mgen_bin_edges[i]+mgen_bin_edges[i+1])/2 for i in range(len(mgen_bin_edges)-1)]))
        mass_gen_bin =  hist.axis.Variable(mgen_bin_edges, name="mgen", label=r"m_{GEN} (GeV)")                         
        mass_bin = hist.axis.Variable(mreco_bin_edges, name="mreco", label=r"m_{RECO} (GeV)")
        # ptgen_edges = np.array([200,260,350,430,540,630,690,750,810,13000]) #### Old values
        ptgen_edges = np.array([200,280,380,460,560,640,700,13000]) #### NEW VALUES TO SWITCH TO
        pt_bin = hist.axis.Variable(ptgen_edges, name="ptreco", label=r"p_{T,RECO} (GeV)")  
        pt_gen_bin = hist.axis.Variable(ptgen_edges, name="ptgen", label=r"p_{T,GEN} (GeV)")
        y_bin = hist.axis.Regular(25, -2.5, 2.5, name="rapidity", label=r"$y$")
        eta_bin = hist.axis.Regular(25, 0., 2.5, name="eta", label=r"$\eta$")
        frac_axis = hist.axis.Regular(10, 0.0, 1., name="frac", label="Fraction")
        n_axis = hist.axis.Regular(5, 0, 5, name="n", label=r"Number")
        dr_axis = hist.axis.Regular(150, 0, 6.0, name="dr", label=r"$\Delta R$")
        dphi_axis = hist.axis.Regular(150, -2*np.pi, 2*np.pi, name="dphi", label=r"$\Delta \phi$")
        phi_axis = hist.axis.Regular(25, -2*np.pi, 2*np.pi, name="phi", label=r"$\phi$")
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
            #### btag study histos
            'alljet_ptreco_mreco':        hist.Hist(jet_cat, parton_cat, mass_bin, pt_bin, storage="weight", name="Events"),
            'btag_eta':            hist.Hist(jet_cat, parton_cat, frac_axis, eta_bin, storage="weight", name="Events"),
                
            #### Plots of things during the selection process / for debugging
            'njet_gen':                  hist.Hist(syst_cat, n_axis, storage="weight", label="Events"),
            'njet_reco':                  hist.Hist(syst_cat, n_axis, storage="weight", label="Events"),
            'dphimin_gen':               hist.Hist(syst_cat, dphi_axis, storage="weight", label="Events"),
            'dphimin_reco':               hist.Hist(syst_cat, dphi_axis, storage="weight", label="Events"),
            'asymm_reco':               hist.Hist(syst_cat, frac_axis, storage="weight", label="Events"),
            'asymm_gen':               hist.Hist(syst_cat, frac_axis, storage="weight", label="Events"),
            'sdmass_orig':         hist.Hist(jk_axis, pt_bin, mass_bin, storage="weight", label="Events"),
            'sdmass_ak8corr':         hist.Hist(jk_axis, pt_bin, mass_bin, storage="weight", label="Events"),
            'sdmass_ak4corr':         hist.Hist(jk_axis, pt_bin, mass_bin, storage="weight", label="Events"),
            # 'jet_dr_reco_gen':           hist.Hist(syst_cat, dr_axis, storage="weight", label="Events"),
            # 'jet_eta_reco':              hist.Hist(syst_cat, eta_bin, storage="weight", name="Events"),
            'jet_rap_reco':              hist.Hist(syst_cat, y_bin, storage="weight", name="Events"),
            'jet_rap_gen':               hist.Hist(syst_cat, y_bin, storage="weight",name="Events"),
            #'jet_dr_gen':                hist.Hist(dr_axis, storage="weight", label="Counts"),
            #'jet_dr_reco':               hist.Hist(dr_axis, storage="weight", label="Counts"),
            'jet_phi_gen':             hist.Hist(syst_cat, phi_axis, storage="weight", label="Events"),
            'jet_phi_reco':             hist.Hist(syst_cat, phi_axis, storage="weight", label="Events"),
            'jet_eta_phi':                 hist.Hist(syst_cat, phi_axis, y_bin, storage="weight", label="Counts"),
            'jet_dr_gen_subjet':         hist.Hist(syst_cat, dr_axis, storage="weight", label="Events"),
            'jet_dr_reco_to_gen_subjet': hist.Hist(syst_cat, dr_axis, storage="weight", label="Events"),
            #### for investigation of removing fakes
            'fakes_eta_phi':             hist.Hist(syst_cat, eta_bin, phi_axis, storage="weight", name="Events"),
            'fakes_asymm_dphi':             hist.Hist(syst_cat, frac_axis, dphi_axis, storage="weight", name="Events"),
            
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
        IOV = ('2016APV' if ( any(re.findall(r'HIPM',  dataset)))
               else '2018'    if ( any(re.findall(r'UL18', dataset)) or any(re.findall(r'UL2018',    dataset)))
               else '2017'    if ( any(re.findall(r'UL17', dataset)) or any(re.findall(r'UL2017',    dataset)))
               else '2016')
        #####################################
        #### Make loop for running 1/10 of dataset for jackknife
        #####################################
        index_list = np.arange(len(events))
        if self.jk:
            print("Self.jk ", self.jk)
            range_max = 10
        else: range_max=1
        if self.jk_range == None:
            jk_inds = range(0,range_max)
        else:
            jk_inds = self.jk_range
        for jk_index in jk_inds:
            if self.jk:
                print("Now doing jackknife {}".format(jk_index))
                print("Len of events before jk selection ", len(events))
            else:
                jk_index=-1
            jk_sel = ak.where(index_list%range_max == jk_index, False, True)
            ######## Select portion for jackknife and ensure that all jets have a softdrop mass so sd mass correction does not fail
            events_jk = events[jk_sel]
            del jk_sel
            #####################################
            #### Find the era from the file name
            #### Apply the good lumi mask
            #####################################
            # if IOV == '2018' and self.hem:
            #     nEvents = len(events_jk)
            #     events_jk = events_jk[HEMVeto(events_jk.FatJet, events_jk.run)]
            #     print("nEvents removed by HEMveto: ", len(events)- nEvents)
            #     del nEvents
            FatJet=events_jk.FatJet
            FatJet["p4"] = ak.with_name(events_jk.FatJet[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
            #### Make sure there is at least one jet to run over
            if len(FatJet) < 1:
                return out
            if self.do_gen:
                era = None
                GenJetAK8 = events_jk.GenJetAK8
                GenJetAK8['p4']= ak.with_name(events_jk.GenJetAK8[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
                FatJet["pt_gen"] = ak.values_astype(ak.fill_none(FatJet.p4.nearest(GenJetAK8.p4, threshold=0.4).pt, 0), np.float32)
            else:
                firstidx = filename.find( "store/data/" )
                fname2 = filename[firstidx:]
                fname_toks = fname2.split("/")
                era = fname_toks[ fname_toks.index("data") + 1]
                print("IOV ", IOV, ", era ", era)
            corrected_fatjets = GetJetCorrections(FatJet, events_jk, era, IOV, isData=not self.do_gen)
            #### Subjet corrections breaks without requiring at least one subjet --> ak.where doesn't work either
            corrected_fatjets = GetCorrectedSDMass(corrected_fatjets, events_jk, era, IOV, isData=not self.do_gen)
            if not self.jk:
                out["sdmass_orig"].fill(jk=jk_index, ptreco=events_jk[(ak.num(events_jk.FatJet) > 2)].FatJet[:,2].pt, mreco=events_jk[(ak.num(events_jk.FatJet) > 2)].FatJet[:,2].msoftdrop)
                out["sdmass_ak4corr"].fill(jk=jk_index, ptreco=corrected_fatjets[(ak.num(corrected_fatjets) > 2)][:,2].pt, mreco=corrected_fatjets[(ak.num(corrected_fatjets) > 2)][:,2].msoftdrop)
                corrected_fatjets_ak8 = corrected_fatjets
                corrected_fatjets_ak8 = GetCorrectedSDMass(corrected_fatjets, events_jk, era, IOV, isData=not self.do_gen, useSubjets = False)
                out["sdmass_ak8corr"].fill(jk=jk_index, ptreco=corrected_fatjets_ak8[(ak.num(corrected_fatjets_ak8) > 2)][:,2].pt, mreco=corrected_fatjets_ak8[(ak.num(corrected_fatjets_ak8) > 2)][:,2].msoftdrop)
            jet_corrs = {}
            self.weights = {}
            if 'HEM' in self.jet_systematics and self.do_gen:
                jet_corrs.update({
                           "HEM": HEMCleaning(IOV,applyjmsSF(IOV, applyjmrSF(IOV,corrected_fatjets)))
                          })
            if 'JER' in self.jet_systematics and self.do_gen:
                jet_corrs.update({"jerUp": corrected_fatjets.JER.up,
                                    "jerDown": corrected_fatjets.JER.down
                                })
            if "JMR" in self.jet_systematics and self.do_gen:
                jet_corrs.update({"jmrUp": applyjmsSF(IOV, applyjmrSF(IOV,corrected_fatjets, var = "up")),
                                    "jmrDown": applyjmsSF(IOV, applyjmrSF(IOV,corrected_fatjets, var = "down"))})
            if "JMS" in self.jet_systematics and self.do_gen:
                jet_corrs.update({"jmsUp": applyjmsSF(IOV, applyjmrSF(IOV,corrected_fatjets), var = "up"),
                                    "jmsDown": applyjmsSF(IOV, applyjmrSF(IOV,corrected_fatjets), var = "down")})
            if 'nominal' in self.jet_systematics:
                if not self.do_gen:
                    print("Doing nominal data")
                else:
                    print("Doing nominal")
                    corrected_fatjets = applyjmsSF(IOV, applyjmrSF(IOV,corrected_fatjets))
                jet_corrs.update({"nominal": corrected_fatjets})
            print("Syst list ", self.jet_systematics)
            print([unc_src for unc_src in self.jet_systematics])
            if self.do_gen:
                print("Out of avail corrections: ", corrected_fatjets.fields)
                avail_srcs = [unc_src for unc_src in self.jet_systematics if ("JES_"+unc_src in corrected_fatjets.fields)]
                for unc_src in avail_srcs:
                    #### Apply JMR and JMS corrections
                    corrected_fatjets["JES_"+unc_src].up = applyjmsSF(IOV, applyjmrSF(IOV,corrected_fatjets["JES_"+unc_src].up))
                    corrected_fatjets["JES_"+unc_src].down = applyjmsSF(IOV, applyjmrSF(IOV,corrected_fatjets["JES_"+unc_src].down))
                    jet_corrs.update({
                        unc_src+"Up":corrected_fatjets["JES_"+unc_src].up,
                        unc_src+"Down":corrected_fatjets["JES_"+unc_src].down, })
            print("Final jet corrs to run over: ", jet_corrs)
            for jetsyst in jet_corrs.keys():
                ###################################
                ######### INITIALIZE WEIGHTS AND SELECTION
                ##################################
                events_corr = ak.with_field(events_jk, jet_corrs[jetsyst], "FatJet")
                weights = np.ones(len(events_corr))
                sel = PackedSelection()
                out['cutflow'][dataset] = defaultdict(int)
                if (jetsyst == "nominal"): out['cutflow'][dataset]['nEvents initial'] += (len(events.FatJet))
                print("mctype ", mctype, " gen? ", self.do_gen)
                if self.do_gen and (mctype == "pythia"):
                    print("Doing XS scaling")
                    weights = weights * getXSweight(dataset, IOV)
                elif self.do_gen and (mctype == "herwig"):
                    print("Difference between weights calculated from xsdb and LHE :", (events_corr.LHEWeight.originalXWGTUP - getXSweight(dataset, IOV)))
                    weights = events_corr.LHEWeight.originalXWGTUP
                elif self.do_gen:
                    print("MADGRPAH inputs --> get gen weights from files")
                    weights = events_corr.LHEWeight.originalXWGTUP
                else:
                    ############
                    ### Doing data -- apply lumimask and require at least one jet to apply jet trigger prescales
                    ############
                    lumi_mask = getLumiMask(IOV)(events_corr.run, events_corr.luminosityBlock)
                    events_corr = events_corr[lumi_mask]
                    weights = np.ones(len(events_corr))
                    trigsel, psweights = applyPrescales(events_corr, year = IOV)
                    psweights=ak.where(ak.is_none(psweights), 1.0, psweights)
                    trigsel=ak.where(ak.is_none(trigsel), False, trigsel)
                    weights = ak.where(trigsel, psweights, weights)
                    sel.add("trigsel", trigsel)
                    if (jetsyst == "nominal"): out['cutflow'][dataset]['nEvents after trigger sel '] += (ak.sum(sel.all("trigsel")))
        
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
                        out["njet_gen"].fill(syst = jetsyst, n=ak.num(events_corr[sel.all("npv")].GenJetAK8), weight = weights[sel.all("npv")])
                    genjet = ak.firsts(GenJetAK8[:,2:])
                    pt_cut_gen = genjet.pt > 200.
                    sel.add("triGenJet", (ak.num(events_corr.GenJetAK8) > 2))
                    sel.add("triGenJet_seq", sel.all('npv', 'triGenJet') & pt_cut_gen )
                    GenJetAK8 = events_corr.GenJetAK8
                    GenJetAK8['p4']= ak.with_name(events_corr.GenJetAK8[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
                    rap_cut_gen = ak.where(sel.all("triGenJet_seq"), np.abs(getRapidity(genjet.p4)) < self.ycut, False)
                    sel.add("rapGen", rap_cut_gen)
                    if not self.jk:
                        out["jet_rap_gen"].fill(syst = jetsyst, rapidity=getRapidity(GenJetAK8[sel.all("triGenJet_seq")][:,2].p4), weight=weights[sel.all("triGenJet_seq")])
                        out["jet_phi_gen"].fill(syst=jetsyst, phi=GenJetAK8[sel.all("triGenJet_seq")][:,2].phi, weight=weights[sel.all("triGenJet_seq")])  
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
                        out["asymm_gen"].fill(syst=jetsyst, frac = asymm_gen[sel.all("triGenJet_seq")], weight=weights[sel.all("triGenJet_seq")])
                        out["dphimin_gen"].fill(syst=jetsyst, dphi = dphimin_gen[sel.all("triGenJet_seq")], weight = weights[sel.all("triGenJet_seq")])
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
                    out["njet_reco"].fill(syst = jetsyst, n=ak.to_numpy(ak.num(events_corr[sel.all("npv")].FatJet), allow_missing=True), 
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
                    out["jet_rap_reco"].fill(syst = jetsyst, rapidity=ak.to_numpy(getRapidity(FatJet[sel.all("triRecoJet_seq")][:,2].p4), allow_missing=True), weight=weights[sel.all("triRecoJet_seq")])
                    out["jet_phi_reco"].fill(syst=jetsyst, phi=FatJet[sel.all("triRecoJet_seq")][:,2].phi, weight=weights[sel.all("triRecoJet_seq")]) 
                #### Add cut on softdrop mass as done in previous two papers --> need to very with JMS/JMR studies
                jet1 = ak.firsts(events_corr.FatJet[:,0:])
                jet2 = ak.firsts(events_corr.FatJet[:,1:])
                jet3 = ak.firsts(events_corr.FatJet[:,2:])
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
                    out["dphimin_reco"].fill(syst=jetsyst, dphi = dphimin[sel.all("triRecoJet_seq")], weight=weights[sel.all("triRecoJet_seq")])
                    out["asymm_reco"].fill(syst=jetsyst, frac = asymm[sel.all("triRecoJet_seq")], weight=weights[sel.all("triRecoJet_seq")])
                jetid_sel = ak.where(sel.all("triRecoJet_seq"), (jet3.jetId > 1), False)
                sel.add("jetId", jetid_sel)
                ####  Get Final RECO selection
                sel.add("recoTot_seq", sel.all("recodphi_seq", "jetId") & ~ak.is_none(jet3.mass) & ~ak.is_none(jet3.msoftdrop))
                print("nevents total recal selection ", ak.sum(sel.all("recoTot_seq")))                
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
                        if jetsyst == "nominal": out['cutflow'][dataset]['misses'] += (len(events_corr[miss_sel].GenJetAK8))
                        print("Number of missed jets ", ak.sum(miss_sel))
                        print("Number of none missed jets ", ak.sum(ak.is_none(GenJetAK8[miss_sel][:,2])))
                        ###### Applying misses selection to gen jets and getting sd mass
                        genjet = ak.firsts(events_corr[miss_sel].GenJetAK8[:,2:])
                        groomed_genjet = get_gen_sd_mass_jet(genjet, events_corr[miss_sel].SubGenJetAK8)
                        miss_jets = events_corr[miss_sel].GenJetAK8[:,2]
                        miss_weights = weights[miss_sel]
                        print("Len of missed jets ", len(miss_jets), " and weights ", len(miss_weights))
                        out["misses"].fill(syst=jetsyst, jk=jk_index, ptgen = miss_jets[~ak.is_none(miss_jets.mass)].pt, mgen = miss_jets[~ak.is_none(miss_jets.mass)].mass, weight = miss_weights[~ak.is_none(miss_jets)])
                        out["misses_g"].fill(syst=jetsyst, jk=jk_index, ptgen = miss_jets[~ak.is_none(miss_jets.mass)].pt, mgen = groomed_genjet[~ak.is_none(miss_jets.mass)].mass, weight = miss_weights[~ak.is_none(miss_jets)])
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
                        # if "L1PreFiringWeight" in events_corr.fields and "L1PreFiringWeight" in self.systematics:                
                        #     prefiringNom, prefiringUp, prefiringDown = GetL1PreFiringWeight(events_corr[fakes])
                        #     fake_weights.add("L1prefiring", weight=prefiringNom, weightUp=prefiringUp, weightDown=prefiringDown )
                        # if "PUSF" in self.systematics:
                        #     puUp, puDown, puNom = GetPUSF(events_corr[fakes], IOV)
                        #     fake_weights.add("PUSF", weight=puNom, weightUp=puUp,
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
                        out["fakes"].fill(syst=jetsyst, jk = jk_index, ptreco = fake_jets[~ak.is_none(fake_jets.mass)].pt, mreco = fake_jets[~ak.is_none(fake_jets.mass)].mass, weight = fake_weights.weight()[~ak.is_none(fake_jets.mass)])
                        out["fakes_g"].fill(syst=jetsyst, jk = jk_index, ptreco = fake_jets[~ak.is_none(fake_jets.msoftdrop)].pt, mreco = fake_jets[~ak.is_none(fake_jets.msoftdrop)].msoftdrop, weight = fake_weights.weight()[~ak.is_none(fake_jets.msoftdrop)])
                        if not self.jk:
                            out['fakes_eta_phi'].fill(syst=jetsyst, phi = fake_jets.phi[~ak.is_none(fake_jets.msoftdrop)], eta = fake_jets.eta[~ak.is_none(fake_jets.msoftdrop)], weight=fake_weights.weight()[~ak.is_none(fake_jets.msoftdrop)])
                    if (jetsyst == "nominal"): out['cutflow'][dataset]['fakes'] += (len(events_corr[fakes].FatJet))
                    print("Number of events after all selections ", ak.sum(sel.all("genTot_seq", "recoTot_seq", "removeMisses", "removeFakes")))
                    if len(events_corr[sel.all("genTot_seq", "recoTot_seq", "removeMisses", "removeFakes")])<1: 
                        print("No events after all selections and removing fakes & misses")
                        return out
                    uf = ak.where(sel.require(recoTot_seq=True), (ak.firsts(events_corr.GenJetAK8[:,2:]).pt < 200.), False)
                    sel.add("rem_uf_fakes", ~uf)
                    print("# of uf fakes not caught by regular fakes ", ak.sum( (uf & ~fakes)))
                    uf_jets = events_corr[uf].FatJet[:,2]
                    uf_weights = weights[uf]
                    print("Lengths of underflow jets ", len(uf_jets), " length of underflow weights ", len(uf_weights))
                    out["underflow"].fill(syst=jetsyst, jk = jk_index, ptreco = uf_jets[~ak.is_none(uf_jets.mass)].pt, mreco = uf_jets[~ak.is_none(uf_jets.mass)].mass, weight = uf_weights[~ak.is_none(uf_jets.mass)])
                    out["underflow_g"].fill(syst=jetsyst, jk = jk_index, ptreco = uf_jets[~ak.is_none(uf_jets.mass)].pt, mreco = uf_jets[~ak.is_none(uf_jets.mass)].msoftdrop, weight = uf_weights[~ak.is_none(uf_jets.mass)])
                    sel.add("final_seq", sel.all("genTot_seq", "recoTot_seq", "removeMisses", "removeFakes", "rem_uf_fakes"))
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
                #### Apply jet veto map
                jet = events_corr.FatJet[:,2]
                veto = ApplyVetoMap(IOV, jet, mapname='jetvetomap')
                events_corr = events_corr[veto]
                weights = weights[veto]

                #######################
                #### Get final jets and weights and fill final plots
                #######################
                
                jet = events_corr.FatJet[:,2]
                #### Create coffea weights object
                self.weights[jetsyst] = Weights(len(weights))
                self.weights[jetsyst].add('jetWeight', weight=weights)
                    
                ##################
                #### Apply final selections to GEN and fill any plots requiring gen, including resp. matrices
                ##################
                
                if self.do_gen:
                    print("final n events ", ak.sum(sel.all("final_seq")))
                    print("Subjets ", events_corr.SubGenJetAK8)
                    genjet = events_corr.GenJetAK8[:,2]
                    groomed_genjet = get_gen_sd_mass_jet(genjet, events_corr.SubGenJetAK8)
                    #### store initial gen weights
                    if "L1PreFiringWeight" in events.fields and "L1PreFiringWeight" in self.systematics:                
                        prefiringNom, prefiringUp, prefiringDown = GetL1PreFiringWeight(events_corr)
                        self.weights[jetsyst].add("L1prefiring", weight=prefiringNom, weightUp=prefiringUp, weightDown=prefiringDown)
                    if "PUSF" in self.systematics:
                        puNom, puUp, puDown = GetPUSF(events_corr, IOV)
                        self.weights[jetsyst].add("PUSF", weight=puNom, weightUp=puUp, weightDown=puDown)
                    if 'herwig' in dataset or 'madgraph' in dataset:
                        pdfNom, pdfUp, pdfDown = GetPDFWeights(events_corr)
                        self.weights[jetsyst].add("PDF", weight=pdfNom, weightUp=pdfUp,
                                           weightDown=pdfDown) 
                        q2Nom, q2Up, q2Down = GetQ2Weights(events_corr)
                        self.weights[jetsyst].add("Q2", weight=q2Nom, weightUp=q2Up,
                                           weightDown=q2Down) 
                    out["ptgen_mgen_u"].fill(syst=jetsyst, jk=jk_index, ptgen=genjet.pt, mgen=genjet.mass, weight=self.weights[jetsyst].weight() )
                    out["ptgen_mgen_g"].fill(syst=jetsyst, jk=jk_index, ptgen=genjet.pt, mgen=groomed_genjet.mass, weight=self.weights[jetsyst].weight() )
                    #### Final plots
                    out["response_matrix_u"].fill(syst=jetsyst, jk=jk_index, ptreco=jet.pt, ptgen=genjet.pt, mreco=jet.mass, mgen=genjet.mass, weight = self.weights[jetsyst].weight())
                    out["response_matrix_g"].fill(syst=jetsyst, jk=jk_index, ptreco=jet.pt, ptgen=genjet.pt, mreco=jet.msoftdrop, mgen=groomed_genjet.mass, weight = self.weights[jetsyst].weight() )
                    out["ptreco_mreco_u"].fill(syst=jetsyst, jk=jk_index, ptreco=jet.pt, mreco=jet.mass, weight=self.weights[jetsyst].weight() )
                    out["ptreco_mreco_g"].fill(syst=jetsyst, jk=jk_index, ptreco=jet.pt, mreco=jet.msoftdrop, weight=self.weights[jetsyst].weight() )
                    #### Outliers
                    if not self.jk:
                        out["jet_eta_phi"].fill(syst=jetsyst, phi=jet.phi, rapidity=getRapidity(events_corr.FatJet[:,2].p4), weight=self.weights[jetsyst].weight())
                    weird = (np.abs(jet.msoftdrop - groomed_genjet.mass) > 20.0) & (jet.msoftdrop > 10.)
                    if (jetsyst == "nominal"): out['cutflow'][dataset]['Number of outliers'] += (len(events_corr[weird].FatJet))
                    weights = self.weights[jetsyst].weight()
                    if jetsyst=="nominal":
                        for syst in self.weights[jetsyst].variations:
                            print("Weight variation: ", syst)
                            #fill nominal, up, and down variations for each
                            out['ptgen_mgen_u'].fill(syst=syst, jk=jk_index, ptgen=events_corr.GenJetAK8[:,2].pt, mgen=events_corr.GenJetAK8[:,2].mass, weight=self.weights[jetsyst].weight(syst) )
                            out['ptgen_mgen_g'].fill(syst=syst, jk=jk_index, ptgen=events_corr.GenJetAK8[:,2].pt, mgen=groomed_genjet.mass, 
                                                          weight=self.weights[jetsyst].weight(syst) )           
                            out["response_matrix_u"].fill(syst=syst, jk=jk_index,ptreco=jet.pt, mreco=jet.mass, ptgen=events_corr.GenJetAK8[:,2].pt, mgen=events_corr.GenJetAK8[:,2].mass,
                                                          weight=self.weights[jetsyst].weight(syst))
                            out["response_matrix_g"].fill(syst=syst, jk=jk_index, ptreco=jet.pt, mreco=jet.msoftdrop, ptgen=events_corr.GenJetAK8[:,2].pt,mgen=groomed_genjet.mass, 
                                                          weight=self.weights[jetsyst].weight(syst))
                            out["ptreco_mreco_u"].fill(syst=syst, jk=jk_index, ptreco=events_corr.FatJet[:,2].pt, mreco=events_corr.FatJet[:,2].mass, weight=self.weights[jetsyst].weight(syst) )
                            out["ptreco_mreco_g"].fill(syst=syst, jk=jk_index, ptreco=events_corr.FatJet[:,2].pt, mreco=events_corr.FatJet[:,2].msoftdrop, weight=self.weights[jetsyst].weight(syst) )
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
                                    out['alljet_ptreco_mreco'].fill(jetNumb = jetname, partonFlav = flavor, 
                                                                    mreco = jetobj[flavor].mass, 
                                                                    ptreco = jetobj[flavor].pt)
                                    out['btag_eta'].fill(jetNumb = jetname, partonFlav = flavor, 
                                                         frac = jetobj[flavor].btagDeepB, eta = jetobj[flavor].eta )
                        out['cutflow'][dataset]['nGluonJets'] += (len(jet3flav["Gluon"])+len(jet1flav["Gluon"])+len(jet2flav["Gluon"]))
                        out['cutflow'][dataset]['nJets'] += (len(events_corr.FatJet[:,0])+len(events_corr.FatJet[:,1])+len(events_corr.FatJet[:,2]))
                        out['cutflow'][dataset]['nSoftestGluonJets'] += (len(jet3flav["Gluon"]))
                        out['cutflow'][dataset]['nSoftestGluonJets_b'] += (len(jet3_jetb_flav["Gluon"]))
                        out['cutflow'][dataset]['nSoftestGluonJets_bb'] += (len(jet3_jetbb_flav["Gluon"]))
                        out['cutflow'][dataset]['nSoftestJets_b'] += (len(jet3_b))
                        out['cutflow'][dataset]['nSoftestJets_bb'] += (len(jet3_bb))
                        out['cutflow'][dataset]['n3Jets'] += (len(events_corr.FatJet[:,2].pt))

                ###############
                ##### If running over DATA fill only final reco plots
                ###############
                
                else:
                    out["ptreco_mreco_u"].fill(syst=jetsyst, jk=jk_index, ptreco=jet.pt, mreco=jet.mass, weight=self.weights[jetsyst].weight() )
                    out["ptreco_mreco_g"].fill(syst=jetsyst, jk=jk_index, ptreco=jet.pt, mreco=jet.msoftdrop, weight=self.weights[jetsyst].weight() )
                    if not self.jk:
                        out["jet_eta_phi"].fill(syst=jetsyst, phi=jet.phi, rapidity=getRapidity(events_corr.FatJet[:,2].p4), weight=self.weights[jetsyst].weight())
                print("final jets ", jet)
                print("final jet pt ", jet.pt)
                if (jetsyst == "nominal"):
                    for name in sel.names:
                        out["cutflow"][dataset][name] += sel.all(name).sum()
                negMSD = jet.msoftdrop<0.
                print("Number of negative softdrop values ", ak.sum(negMSD) )
                if (jetsyst == "nominal"): out['cutflow'][dataset]['nEvents failing softdrop condition'] += ak.sum(negMSD)
                del events_corr, weights
            del events_jk
        out['cutflow'][dataset]['chunks'] += 1
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

