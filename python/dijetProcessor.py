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

parser = argparse.ArgumentParser()

parser.add_argument("year")
parser.add_argument("data")

#### currently only for MC --> makes hists and response matrix
class makeDijetHists(processor.ProcessorABC):
    '''
    Processor to run a dijet jet mass cross section analysis. 
    With "do_gen == True", will perform GEN selection and create response matrices. 
    Will always plot RECO level quantities. 
    '''
    def __init__(self, ptcut = 200., ycut = 2.5, data = False, systematics = ['nominal', 'jer', 'jes']):
        # should have separate **lower** ptcut for gen
        self.do_gen = not data
        self.ptcut = ptcut
        self.ycut = ycut #rapidity
        self.systematics = systematics
        print("Data: ", data, " gen ", self.do_gen)
        dataset_cat = hist.axis.StrCategory([],growth=True,name="dataset", label="Dataset")
        jet_cat = hist.axis.StrCategory([], growth=True, name="jetNumb", label="Jet")
        syst_cat = hist.axis.StrCategory([], growth=True, name='syst', label="Systematic")
        parton_cat = hist.axis.StrCategory([], growth=True,name="partonFlav", label="Parton Flavour")
        mgen_bin_edges = np.array([0,1,5,10,20,40,60,80,100,150,200,250,1000])
        mreco_bin_edges = np.sort(np.append(mgen_bin_edges,[(mgen_bin_edges[i]+mgen_bin_edges[i+1])/2 for i in range(len(mgen_bin_edges)-1)]))
        mass_gen_bin =  hist.axis.Variable(mgen_bin_edges, name="mgen", label=r"m_{GEN} (GeV)")                         
        mass_bin = hist.axis.Variable(mreco_bin_edges, name="mreco", label=r"m_{RECO} (GeV)")
        pt_bin = hist.axis.Variable([200,280,360,450,520,630,690,750,800,1300,13000], name="ptreco", label=r"p_{T,RECO} (GeV)")        
        pt_gen_bin = hist.axis.Variable([200,280,360,450,520,630,690,750,800,1300,13000], name="ptgen", label=r"p_{T,GEN} (GeV)") 
        y_bin = hist.axis.Regular(25, 0., 2.5, name="rapidity", label=r"$y$")
        eta_bin = hist.axis.Regular(25, 0., 2.5, name="eta", label=r"$\eta$")
        frac_axis = hist.axis.Regular(10, 0.0, 1., name="frac", label="Fraction")
        n_axis = hist.axis.Regular(5, 0, 5, name="n", label=r"Number")
        dr_axis = hist.axis.Regular(150, 0, 6.0, name="dr", label=r"$\Delta R$")
        dphi_axis = hist.axis.Regular(150, -2*np.pi, 2*np.pi, name="dphi", label=r"$\Delta \phi$")
        
        self._histos = {
            #### Old histos
            'jet_mass':             hist.Hist(dataset_cat, jet_cat, parton_cat, mass_bin, storage="weight", name="Events"),
            'jet_pt':             hist.Hist(dataset_cat, jet_cat, parton_cat, pt_bin, storage="weight", name="Events"),
            'jet_rap':            hist.Hist(dataset_cat, jet_cat, parton_cat, y_bin, storage="weight", name="Events"),
            'jet_eta':            hist.Hist(dataset_cat, jet_cat, parton_cat, eta_bin, storage="weight", name="Events"),
            
            #### Plots of things during the selection process / for debugging
            #'njet_reco':                 hist.Hist(dataset_cat, n_axis, storage="weight", label="Counts"),
            'njet_gen':                  hist.Hist(dataset_cat, syst_cat, n_axis, storage="weight", label="Counts"),
            #'jet_dr_reco_gen':           hist.Hist(dataset_cat, dr_axis, storage="weight", label="Counts"),
            'jet_mass_u_reco_over_gen':    hist.Hist(dataset_cat, syst_cat, frac_axis, storage="weight", label="Counts"),
            #'jet_pt_reco':               hist.Hist(dataset_cat, pt_bin, storage="weight", name="Events"),
            'jet_pt_gen':                hist.Hist(dataset_cat, syst_cat, pt_gen_bin, storage="weight", name="Events"),
            'jet_mass_gen':              hist.Hist(dataset_cat, syst_cat, jet_cat, mass_gen_bin, storage="weight", name="Events"),
            'jet_pt_reco_over_gen':      hist.Hist(dataset_cat, syst_cat, frac_axis, storage="weight", label="Counts"),
            #'jet_eta_reco':              hist.Hist(dataset_cat, jet_cat, parton_cat, eta_bin, storage="weight", name="Events"),
            #'jet_eta_gen':               hist.Hist(dataset_cat, jet_cat, eta_bin, storage="weight",name="Events"),
            #'jet_dr_gen':                hist.Hist(dataset_cat, dr_axis, storage="weight", label="Counts"),
            #'jet_dr_reco':               hist.Hist(dataset_cat, dr_axis, storage="weight", label="Counts"),
            'jet_dphi_gen':              hist.Hist(dataset_cat, syst_cat, dphi_axis, storage="weight", label="Counts"),
            #'jet_dphi_reco':             hist.Hist(dataset_cat, dphi_axis, storage="weight", label="Counts"),
            'jet_ptasymm_gen':           hist.Hist(dataset_cat, syst_cat, frac_axis, storage="weight", label="Counts"),
            #'jet_ptasymm_reco':          hist.Hist(dataset_cat, frac_axis, storage="weight", label="Counts"),
            'jet_dr_gen_subjet':         hist.Hist(dataset_cat, syst_cat, dr_axis, storage="weight", label="Counts"),
            'dijet_dr_reco_to_gen':      hist.Hist(dataset_cat, syst_cat, dr_axis, storage="weight", label="Counts"),
            'dr_reco_to_gen_subjet' :    hist.Hist(dataset_cat, syst_cat, dr_axis, storage="weight", label="Counts"),
            'misses':                    hist.Hist(dataset_cat, syst_cat, pt_gen_bin, mass_gen_bin, storage="weight", name="Events"),
            'fakes':                     hist.Hist(dataset_cat, syst_cat, pt_bin, mass_bin, storage="weight", name="Events"),
            #### hist for comparison of weights
            'weights':                   hist.Hist(dataset_cat, syst_cat, pt_bin, mass_bin, storage="weight", name="Events"),
            
            #### Plots to be unfolded
            'jet_pt_mass_reco_u':        hist.Hist(dataset_cat, syst_cat, pt_bin, mass_bin, storage="weight", name="Events"),
            'jet_pt_mass_reco_g':        hist.Hist(dataset_cat, syst_cat, pt_bin, mass_bin, storage="weight", name="Events"),
    
            #### Plots for comparison
            'jet_pt_mass_gen_u':         hist.Hist(dataset_cat, syst_cat, pt_gen_bin, mass_gen_bin, storage="weight", label="Counts"),       
            'jet_pt_mass_gen_g':         hist.Hist(dataset_cat, syst_cat, pt_gen_bin, mass_gen_bin, storage="weight", label="Counts"),
        
        
            #### Plots to get JMR and JMS in MC
            'jet_m_pt_u_reco_over_gen':  hist.Hist(dataset_cat, syst_cat, pt_gen_bin, mass_gen_bin, frac_axis, storage="weight",                                                                        label="Counts"),
            'jet_m_pt_g_reco_over_gen':  hist.Hist(dataset_cat, syst_cat, pt_gen_bin, mass_gen_bin, frac_axis, storage="weight",                                                                        label="Counts"),
        
            #### Plots for the analysis in the proper binning
            'response_matrix_u':         hist.Hist(dataset_cat, syst_cat, pt_bin, mass_bin, pt_gen_bin, mass_gen_bin, storage="weight",                                                         label="Counts"),
            'response_matrix_g':         hist.Hist(dataset_cat, syst_cat, pt_bin, mass_bin, pt_gen_bin, mass_gen_bin, storage="weight",                                                         label="Counts"),
                 
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
        out['cutflow']['nEvents initial'] += (len(events.FatJet))
        #####################################
        #### Find the IOV from the dataset name
        #####################################
        IOV = ('2016APV' if ( any(re.findall(r'APV',  dataset)) or any(re.findall(r'HIPM', dataset)))
               else '2018'    if ( any(re.findall(r'UL18', dataset)) or any(re.findall(r'UL2018',    dataset)))
               else '2017'    if ( any(re.findall(r'UL17', dataset)) or any(re.findall(r'UL2017',    dataset)))
               else '2016')
        out['cutflow']['nEvents'+IOV+dataset] += (len(events.FatJet))
        #####################################
        #### Find the era from the file name
        #### Add values needed for jet corrections
        #####################################
        FatJet=events.FatJet
        FatJet["p4"] = ak.with_name(events.FatJet[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
        if self.do_gen:
            era = None
            GenJetAK8 = events.GenJetAK8
            GenJetAK8['p4']= ak.with_name(events.GenJetAK8[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
            FatJet["pt_gen"] = ak.values_astype(ak.fill_none(FatJet.p4.nearest(GenJetAK8.p4, threshold=0.2).pt, 0), np.float32)
        else:
            firstidx = filename.find( "store/data/" )
            fname2 = filename[firstidx:]
            fname_toks = fname2.split("/")
            era = fname_toks[ fname_toks.index("data") + 1]
            print("IOV ", IOV, ", era ", era)
        corrected_fatjets = GetJetCorrections(FatJet, events, era, IOV, isData=not self.do_gen)
        jet_corrs = {"nominal": corrected_fatjets}
        self.weights = {}
        # if (len(corrected_fatjets.pt[0]) > 1) and 'jes' in self.systematics and self.do_gen:
        #     print('JEC:%s:JES up, nom, down:%s:%s:%s',
        #                  corrected_fatjets.JES_jes.up.pt[0][0],
        #                  corrected_fatjets.pt[0][0],
        #                  corrected_fatjets.JES_jes.down.pt[0][0])
        #     print("JES up vals: ", corrected_fatjets.JES_jes.up)
        if 'jes' in self.systematics and self.do_gen:
            jet_corrs.update({
                       "jesUp": corrected_fatjets.JES_jes.up,
                       "jesDown": corrected_fatjets.JES_jes.down
                      })

        if 'jer' in self.systematics and self.do_gen:
            jet_corrs.update({"jerUp": corrected_fatjets.JER.up,
                                "jerDown": corrected_fatjets.JER.down
                            })
        for syst in jet_corrs.keys():
            print("Adding ", syst, " values ", jet_corrs[syst], " to output")
            events_corr = ak.with_field(events, jet_corrs[syst], "FatJet")
            weights = np.ones(len(events_corr))
            if (self.do_gen):
                print("Do XS scaling")
                weights = weights * getXSweight(dataset, IOV)
            else:
                ###############
                ### apply lumimask and require at least one jet to apply jet trigger prescales
                ##############
                lumi_mask = getLumiMask(IOV)(events_corr.run, events_corr.luminosityBlock)
                events_corr = events_corr [lumi_mask & (ak.num(events_corr.FatJet) >= 1)]
                trigsel, psweights = applyPrescales(events_corr, year = IOV)
                weights=psweights
                # print("Trigger: len of events ", len(events_corr), "len of weights ", len(trigsel))
                print(weights)
                events_corr = events_corr[trigsel]
                weights = weights[trigsel]
                out['cutflow']['nEvents after trigger sel '+syst] += (len(events.FatJet))
                #### Need to add PU reweighting for if do_gen
                #### Remove event with very large gen weights???
            
                
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
                GenJetAK8 = events_corr.GenJetAK8
                GenJetAK8['p4']= ak.with_name(events_corr.GenJetAK8[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
                pt_cut_gen = ak.all(events_corr.GenJetAK8.pt > 140., axis = -1) ### 70% of reco pt cut
                rap_cut_gen = ak.all(np.abs(getRapidity(GenJetAK8.p4)) < self.ycut, axis = -1)
                out["njet_gen"].fill(dataset=dataset, syst = syst, n=ak.num(events_corr.GenJetAK8[rap_cut_gen & pt_cut_gen]), 
                                     weight = weights[rap_cut_gen & pt_cut_gen] )
                #### Apply kinematic and 2 jet requirement immediately so that dphi and asymm can be calculated
                weights = weights[(ak.num(events_corr.GenJetAK8) > 1) & pt_cut_gen & rap_cut_gen]
                events_corr = events_corr[(ak.num(events_corr.GenJetAK8) > 1) & pt_cut_gen & rap_cut_gen]
                out['cutflow']['nEvents after gen rapidity,pT, and nJet selection '+syst] += (len(events_corr.FatJet))
                out["jet_mass_gen"].fill(dataset=dataset, syst = syst, jetNumb = 'jet0', mgen=events_corr.GenJetAK8.mass[:,0],
                                         weight=weights)
                out["jet_mass_gen"].fill(dataset=dataset, syst=syst, jetNumb = 'jet1', mgen=events_corr.GenJetAK8.mass[:,1],
                                         weight=weights)
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
                out["jet_ptasymm_gen"].fill(dataset=dataset, syst=syst,frac=asymm_gen[dphi12_gen_sel], weight=weights[dphi12_gen_sel])  
                out["jet_dphi_gen"].fill(dataset=dataset, syst=syst, dphi=dphi12_gen[asymm_gen_sel], weight=weights[asymm_gen_sel])  
                
                events_corr = events_corr[dphi12_gen_sel & asymm_gen_sel]
                weights = weights[dphi12_gen_sel & asymm_gen_sel]
                out['cutflow']['nEvents after gen dphi and ptasymm selection '+syst] += (len(events_corr.FatJet))
                #misses = gen but no reco
                matches = ak.all(events_corr.GenJetAK8.delta_r(events_corr.GenJetAK8.nearest(events_corr.FatJet)) < 0.2, axis = -1)
                misses = ~matches
                out["misses"].fill(dataset=dataset, syst=syst, ptgen = ak.flatten(events_corr[misses].GenJetAK8.pt), 
                                   mgen = ak.flatten(events_corr[misses].GenJetAK8.mass))
                events_corr = events_corr[matches]
                weights = weights[matches]
                out['cutflow']['nEvents after deltaR matching (remove misses) ' + syst] += len(events_corr.FatJet)
                out['cutflow']['misses '+syst] += (len(events_corr[misses].FatJet))
            #####################################
            #### Reco Jet Selection
            #################################### 
            #### Apply pt and rapidity cuts
            pt_cut_reco = ak.all(events_corr.FatJet.pt > self.ptcut, axis = -1)
            FatJet = events_corr.FatJet
            FatJet["p4"] = ak.with_name(events_corr.FatJet[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
            rap_cut_reco = ak.all(np.abs(getRapidity(FatJet.p4)) < self.ycut, axis = -1)
            weights = weights[(ak.num(events_corr.FatJet) > 1) & pt_cut_reco & rap_cut_reco]
            events_corr = events_corr[(ak.num(events_corr.FatJet) > 1) & pt_cut_reco & rap_cut_reco]
            out['cutflow']['nEvents after reco kine selection '+syst] += (len(events_corr.FatJet))
            #### get dphi and pt asymm selections
            jet1 = events_corr.FatJet[:,0]
            jet2 = events_corr.FatJet[:,1]
            dphi12 = (np.abs(jet1.delta_phi(jet2)) > 2.)
            asymm = np.abs(jet1.pt - jet2.pt)/(jet1.pt + jet2.pt)
            asymm_reco_sel = asymm < 0.3
            events_corr = events_corr[asymm_reco_sel & dphi12]
            weights = weights[asymm_reco_sel & dphi12]
            out['cutflow']['nEvents after reco pT assym. and eta selection '+syst] += (len(events_corr.FatJet))
            dijet_weights = np.repeat(weights, 2)
            
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
            FatJet = events_corr.FatJet
            FatJet["p4"] = ak.with_name(events_corr.FatJet[["pt", "eta", "phi", "mass"]],"PtEtaPhiMLorentzVector")
            dijet = ak.flatten(FatJet[:,:2], axis=1)
            
            #### match jets, get syst weights, and fill final plots
            if self.do_gen:
                fakes = ak.any(ak.is_none(events_corr.FatJet.matched_gen, axis = -1), axis = -1)
                out["fakes"].fill(dataset = dataset, syst=syst, ptreco = ak.flatten(events_corr[fakes].FatJet[:,:2].pt),
                                  mreco = ak.flatten(events_corr[fakes].FatJet[:,:2].mass))
                #fakes = reco but no gen
                out['cutflow']['fakes '+syst] += (len(events_corr[fakes].FatJet))
                matched_reco = ~fakes
                events_corr = events_corr[matched_reco]
                weights = weights[matched_reco]
                out['cutflow']['nEvents after gen matching (remove fakes) '+syst] += (len(events_corr.FatJet))
                #### Get gen subjets and sd gen jets
                groomed_genjet0 = get_gen_sd_mass_jet(events_corr.GenJetAK8[:,0], events_corr.SubGenJetAK8)
                groomed_genjet1 = get_gen_sd_mass_jet(events_corr.GenJetAK8[:,1], events_corr.SubGenJetAK8)
                groomed_gen_dijet = ak.concatenate([groomed_genjet0, groomed_genjet1], axis=0) 
                dijet = ak.flatten(events_corr.FatJet[:,:2], axis =1)
                dijet_weights = np.repeat(weights, 2)
                print("LEngth of dijet weights: ", len(dijet_weights))
                print("LEngth of weights: ", len(weights))
                self.weights[syst] = Weights(len(dijet_weights))
                self.weights[syst].add('dijetWeight', dijet_weights)
                if "L1PreFiringWeight" in events.fields:                
                    prefiringNom, prefiringUp, prefiringDown = GetL1PreFiringWeight(events_corr)
                    self.weights[syst].add("L1prefiring", weight=np.repeat(prefiringNom, 2), 
                                           weightUp=np.repeat(prefiringUp, 2), 
                                           weightDown=np.repeat(prefiringDown, 2),
                               )
                puUp, puDown, puNom = GetPUSF(events_corr, IOV)
                self.weights[syst].add("PUSF", weight=np.repeat(puNom, 2), weightUp=np.repeat(puUp, 2),
                                       weightDown=np.repeat(puDown, 2),)   
                print("Length of FatJet after flattening: ", len(dijet))
                gen_dijet = ak.flatten(events_corr.GenJetAK8[:,:2], axis=1)
                out["jet_pt_gen"].fill(dataset=dataset, syst=syst, ptgen=dijet.pt, weight=self.weights[syst].weight())
                out["jet_dr_gen_subjet"].fill(dataset=dataset, syst=syst, 
                                        dr=events_corr.SubGenJetAK8[:,0].delta_r(events_corr.FatJet[:,0]),
                                              weight=weights)
                #### Plots to check matching
                #            print("Check for none values", ak.any(ak.is_none(dijetEvents, axis = -1)))     
                #### Final plots
                out['jet_pt_mass_gen_u'].fill( dataset=dataset, syst=syst, ptgen=gen_dijet.pt, mgen=gen_dijet.mass, weight=self.weights[syst].weight() )
                out['jet_pt_mass_gen_g'].fill( dataset=dataset, syst=syst, ptgen=groomed_gen_dijet.pt, mgen=groomed_gen_dijet.mass, weight=self.weights[syst].weight() )
                
                out["dijet_dr_reco_to_gen"].fill(dataset=dataset, syst=syst, dr=dijet.delta_r(gen_dijet), weight=self.weights[syst].weight())
                
                
                out["response_matrix_u"].fill( dataset=dataset, syst=syst, 
                                               ptreco=dijet.pt, ptgen=gen_dijet.pt,
                                               mreco=dijet.mass, mgen=gen_dijet.mass, weight=self.weights[syst].weight())
                out["response_matrix_g"].fill( dataset=dataset, syst=syst, 
                                               ptreco=dijet.pt, ptgen=gen_dijet.pt,
                                               mreco=dijet.msoftdrop, mgen=groomed_gen_dijet.mass, weight=self.weights[syst].weight())
                

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
                if syst == "nominal":
                    genjet1 = events_corr.GenJetAK8[:,0]
                    genjet2 = events_corr.GenJetAK8[:,1]
                    jet1 = events_corr.FatJet[:,0]
                    jet2 = events_corr.FatJet[:,1]
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
                    out['cutflow']['nGluonJets '+syst] += (len(ak.flatten(dijet[np.abs(gen_dijet.partonFlavour) == 21].pt, axis=-1)))
                    out['cutflow']['nJets '+syst] += (len(ak.flatten(jet1.pt, axis=-1)))
            dijet = ak.flatten(events_corr.FatJet[:,:2], axis =1)
            dijet_weights = np.repeat(weights, 2)
            out['cutflow']['nEvents final selection '+syst] += (len(events_corr.FatJet))
            out["jet_pt_mass_reco_u"].fill( dataset=dataset, syst=syst, ptreco=dijet.pt, mreco=dijet.mass, weight=dijet_weights )
            out["jet_pt_mass_reco_g"].fill( dataset=dataset, syst=syst, ptreco=dijet.pt, mreco=dijet.msoftdrop, weight=dijet_weights )
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

