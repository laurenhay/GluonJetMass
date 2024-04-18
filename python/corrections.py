import numpy as np
import awkward as ak
import correctionlib
from coffea.jetmet_tools import JetResolutionScaleFactor
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from coffea.lookup_tools import extractor
import copy
import os

#based heavily on https://github.com/b2g-nano/TTbarAllHadUproot/blob/optimize/python/corrections.py and https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/corrections.py?ref_type=heads and https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/jmeCorrections.py?ref_type=heads

def GetL1PreFiringWeight(events):
    # original code https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/TTbarDileptonProcessor.py#L50
    ## Reference: https://twiki.cern.ch/twiki/bin/viewauth/CMS/L1PrefiringWeightRecipe
    # Need to check if weights (up/dn) produced are comparable or greater than JEC weights --> if yes apply, if no take as a systematic uncertainty
    ## var = "Nom", "Up", "Dn"
    L1PrefiringWeights = [events.L1PreFiringWeight.Nom, events.L1PreFiringWeight.Up, events.L1PreFiringWeight.Dn]
    return L1PrefiringWeights

def HEMCleaning(JetCollection):
    # original code https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/TTbarDileptonProcessor.py#L58
    # necessary?
    ## Reference: https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/2000.html
    

    isHEM = ak.ones_like(JetCollection.pt)
    detector_region1 = ((JetCollection.phi < -0.87) & (JetCollection.phi > -1.57) &
                       (JetCollection.eta < -1.3) & (JetCollection.eta > -2.5))
    detector_region2 = ((JetCollection.phi < -0.87) & (JetCollection.phi > -1.57) &
                       (JetCollection.eta < -2.5) & (JetCollection.eta > -3.0))
    jet_selection    = ((JetCollection.jetId > 1) & (JetCollection.pt > 15))

    isHEM            = ak.where(detector_region1 & jet_selection, 0.80, isHEM)
    isHEM            = ak.where(detector_region2 & jet_selection, 0.65, isHEM)
    
    corrected_jets = copy.deepcopy(JetCollection)
    corrected_jets["pt"]   = JetCollection.pt * isHEM
    corrected_jets["mass"] = JetCollection.mass * isHEM

    return corrected_jets
def HEMVeto(FatJets, runs):

    ## Reference: https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/2000.html
    
    runid = (runs >= 319077)
    print(runid)
    # print("Fat jet phi ", FatJets.phi)
    # print("Fat jet phi length ", len(FatJets.phi))
    # print("Fat jet eta ", FatJets.eta)
    # print("Fat jet eta length ", len(FatJets.eta))
    detector_region1 = ((FatJets.phi < -0.87) & (FatJets.phi > -1.57) &
                       (FatJets.eta < -1.3) & (FatJets.eta > -2.5))
    detector_region2 = ((FatJets.phi < -0.87) & (FatJets.phi > -1.57) &
                       (FatJets.eta < -2.5) & (FatJets.eta > -3.0))
    jet_selection    = ((FatJets.jetId > 1) & (FatJets.pt > 15))

    vetoHEMFatJets = ak.any((detector_region1 & jet_selection & runid) ^ (detector_region2 & jet_selection & runid), axis=1)
    print("Number of hem vetoed jets: ", ak.sum(vetoHEMFatJets))
    vetoHEM = ~(vetoHEMFatJets)
    
    return vetoHEM

def GetPUSF(events, IOV):
    # original code https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/TTbarDileptonProcessor.py#L38
    ## json files from: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/LUM
        
    fname = "correctionFiles/puWeights/{0}_UL/puWeights.json.gz".format(IOV)
    # print("PU SF filename: ", fname)
    hname = {
        "2016APV": "Collisions16_UltraLegacy_goldenJSON",
        "2016"   : "Collisions16_UltraLegacy_goldenJSON",
        "2017"   : "Collisions17_UltraLegacy_goldenJSON",
        "2018"   : "Collisions18_UltraLegacy_goldenJSON"
    }
    evaluator = correctionlib.CorrectionSet.from_file(fname)

    puUp = evaluator[hname[str(IOV)]].evaluate(np.array(events.Pileup.nTrueInt), "up")
    puDown = evaluator[hname[str(IOV)]].evaluate(np.array(events.Pileup.nTrueInt), "down")
    puNom = evaluator[hname[str(IOV)]].evaluate(np.array(events.Pileup.nTrueInt), "nominal")

    return puNom, puUp, puDown
def GetJetCorrections(FatJets, events, era, IOV, isData=False, uncertainties = None):
    if uncertainties == None:
        uncertainty_sources = ["AbsoluteMPFBias","AbsoluteScale","AbsoluteStat","FlavorQCD","Fragmentation","PileUpDataMC","PileUpPtBB","PileUpPtEC1","PileUpPtEC2","PileUpPtHF",
"PileUpPtRef","RelativeFSR","RelativeJEREC1","RelativeJEREC2","RelativeJERHF","RelativePtBB","RelativePtEC1","RelativePtEC2","RelativePtHF","RelativeBal","RelativeSample", "RelativeStatEC","RelativeStatFSR","RelativeStatHF","SinglePionECAL","SinglePionHCAL","TimePtEta"]
    else:
        uncertainty_cources = uncertainties
    # original code https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/jmeCorrections.py
    jer_tag=None
    if (IOV=='2018'):
        jec_tag="Summer19UL18_V5_MC"
        jec_tag_data={
            "Run2018A": "Summer19UL18_RunA_V5_DATA",
            "Run2018B": "Summer19UL18_RunB_V5_DATA",
            "Run2018C": "Summer19UL18_RunC_V5_DATA",
            "Run2018D": "Summer19UL18_RunD_V5_DATA",
        }
        jer_tag = "Summer19UL18_JRV2_MC"
    elif (IOV=='2017'):
        jec_tag="Summer19UL17_V5_MC"
        jec_tag_data={
            "Run2017B": "Summer19UL17_RunB_V5_DATA",
            "Run2017C": "Summer19UL17_RunC_V5_DATA",
            "Run2017D": "Summer19UL17_RunD_V5_DATA",
            "Run2017E": "Summer19UL17_RunE_V5_DATA",
            "Run2017F": "Summer19UL17_RunF_V5_DATA",
        }
        jer_tag = "Summer19UL17_JRV3_MC"
    elif (IOV=='2016'):
        jec_tag="Summer19UL16_V7_MC"
        jec_tag_data={
            "Run2016F": "Summer19UL16_RunFGH_V7_DATA",
            "Run2016G": "Summer19UL16_RunFGH_V7_DATA",
            "Run2016H": "Summer19UL16_RunFGH_V7_DATA",
        }
        jer_tag = "Summer20UL16_JRV3_MC"
    elif (IOV=='2016APV'):
        jec_tag="Summer19UL16_V7_MC"
        ## HIPM/APV     : B_ver1, B_ver2, C, D, E, F
        ## non HIPM/APV : F, G, H

        jec_tag_data={
            "Run2016B": "Summer19UL16APV_RunBCD_V7_DATA",
            "Run2016C": "Summer19UL16APV_RunBCD_V7_DATA",
            "Run2016D": "Summer19UL16APV_RunBCD_V7_DATA",
            "Run2016E": "Summer19UL16APV_RunEF_V7_DATA",
            "Run2016F": "Summer19UL16APV_RunEF_V7_DATA",
        }
        jer_tag = "Summer20UL16APV_JRV3_MC"
    else:
        print(f"Error: Unknown year \"{IOV}\".")


    #print("extracting corrections from files for " + jec_tag)
    ext = extractor()
    if not isData:
    #For MC
        ext.add_weight_sets([
            '* * '+'correctionFiles/JEC/{0}/{0}_L1FastJet_AK8PFPuppi.jec.txt'.format(jec_tag),
            '* * '+'correctionFiles/JEC/{0}/{0}_L2Relative_AK8PFPuppi.jec.txt'.format(jec_tag),
            '* * '+'correctionFiles/JEC/{0}/{0}_L3Absolute_AK8PFPuppi.jec.txt'.format(jec_tag),
            '* * '+'correctionFiles/JEC/{0}/{0}_UncertaintySources_AK8PFPuppi.junc.txt'.format(jec_tag),
            '* * '+'correctionFiles/JEC/{0}/{0}_Uncertainty_AK8PFPuppi.junc.txt'.format(jec_tag),
        ])
        #### Do AK8PUPPI jer files exist??
        if jer_tag:
            # print("JER tag: ", jer_tag)
            # print("File "+'correctionFiles/JER/{0}/{0}_PtResolution_AK8PFPuppi.jr.txt'.format(jer_tag)+" exists: ", os.path.exists('correctionFiles/JER/{0}/{0}_PtResolution_AK8PFPuppi.jr.txt'.format(jer_tag)))
            # print("File "+'correctionFiles/JER/{0}/{0}_SF_AK8PFPuppi.jersf.txt'.format(jer_tag)+" exists: ", os.path.exists('correctionFiles/JER/{0}/{0}_SF_AK8PFPuppi.jersf.txt'.format(jer_tag)))
            ext.add_weight_sets([
            '* * '+'correctionFiles/JER/{0}/{0}_PtResolution_AK8PFPuppi.jr.txt'.format(jer_tag),
            '* * '+'correctionFiles/JER/{0}/{0}_SF_AK8PFPuppi.jersf.txt'.format(jer_tag)])
            # print("JER SF added")
    else:       
        #For data, make sure we don't duplicat
        tags_done = []
        for run, tag in jec_tag_data.items():
            if not (tag in tags_done):
                ext.add_weight_sets([
                '* * '+'correctionFiles/JEC/{0}/{0}_L1FastJet_AK8PFPuppi.jec.txt'.format(tag),
                '* * '+'correctionFiles/JEC/{0}/{0}_L2Relative_AK8PFPuppi.jec.txt'.format(tag),
                '* * '+'correctionFiles/JEC/{0}/{0}_L3Absolute_AK8PFPuppi.jec.txt'.format(tag),
                '* * '+'correctionFiles/JEC/{0}/{0}_L2L3Residual_AK8PFPuppi.jec.txt'.format(tag),
                ])
                tags_done += [tag]
    ext.finalize()

    evaluator = ext.make_evaluator()

    if (not isData):
        jec_names = [
            '{0}_L1FastJet_AK8PFPuppi'.format(jec_tag),
            '{0}_L2Relative_AK8PFPuppi'.format(jec_tag),
            '{0}_L3Absolute_AK8PFPuppi'.format(jec_tag)]
        #### if jes in arguments add total uncertainty values for comparison and easy plotting
        if 'jes' in uncertainty_sources:
            jec_names.extend(['{0}_Uncertainty_AK8PFPuppi'.format(jec_tag)])
            uncertainty_sources.remove('jes')
        jec_names.extend(['{0}_UncertaintySources_AK8PFPuppi_{1}'.format(jec_tag, unc_src) for unc_src in uncertainty_sources])

        if jer_tag: 
            jec_names.extend(['{0}_PtResolution_AK8PFPuppi'.format(jer_tag),
                              '{0}_SF_AK8PFPuppi'.format(jer_tag)])

    else:
        jec_names={}
        for run, tag in jec_tag_data.items():
            jec_names[run] = [
                '{0}_L1FastJet_AK8PFPuppi'.format(tag),
                '{0}_L3Absolute_AK8PFPuppi'.format(tag),
                '{0}_L2Relative_AK8PFPuppi'.format(tag),
                '{0}_L2L3Residual_AK8PFPuppi'.format(tag),]



    if not isData:
        jec_inputs = {name: evaluator[name] for name in jec_names}
    else:
        jec_inputs = {name: evaluator[name] for name in jec_names[era]}


    # print("jec_input", jec_inputs)
    jec_stack = JECStack(jec_inputs)


    FatJets['pt_raw'] = (1 - FatJets['rawFactor']) * FatJets['pt']
    FatJets['mass_raw'] = (1 - FatJets['rawFactor']) * FatJets['mass']
    FatJets['rho'] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, FatJets.pt)[0]

    name_map = jec_stack.blank_name_map
    name_map['JetPt'] = 'pt'
    name_map['JetMass'] = 'mass'
    name_map['JetEta'] = 'eta'
    name_map['JetA'] = 'area'
    name_map['ptGenJet'] = 'pt_gen'
    name_map['ptRaw'] = 'pt_raw'
    name_map['massRaw'] = 'mass_raw'
    name_map['Rho'] = 'rho'


    events_cache = events.caches[0]

    jet_factory = CorrectedJetsFactory(name_map, jec_stack)
    corrected_jets = jet_factory.build(FatJets, lazy_cache=events_cache)
    # print("Available uncertainties: ", jet_factory.uncertainties())
    # print("Corrected jets object: ", corrected_jets.fields)
    return corrected_jets

def GetLHEWeight(events):
    #### make sure to `pip3 install --user parton` in singularity shell
    import parton
    # os.environ['LHAPDF_DATA_PATH'] = '/cvmfs/sft.cern.ch/lcg/releases/MCGenerators/lhapdf/6.2.1-7149a/x86_64-centos7-gcc8-opt/share/LHAPDF/'
    paths = ['/cvmfs/sft.cern.ch/lcg/external/lhapdfsets/current/']
    from parton import mkPDF
    # pdfset = PDFSet('PDF4LHC21_mc_pdfas', pdfdir=path[0])
    # print("PDF set obj ", pdfset)
    # print("PDF set info ", pdfset.info)
    # pdfmembers = []
    print("Length of events ", len(events))
    pdf = mkPDF('PDF4LHC21_mc', pdfdir=paths[0])
    # print("PDFSet info ", pdf.pdfset.info)
    print("PDF grids ", pdf.pdfgrids)
    
    #### get pdf values from events
    q2 = events.Generator.scalePDF
    x1 = events.Generator.x1
    x2 = events.Generator.x2
    id1 = events.Generator.id1
    id2 = events.Generator.id2
    xfxq1 = pdf.xfxQ(id1, x1, q2, grid=False)
    xfxq2 = pdf.xfxQ(id2, x2, q2, grid=False)
    print("x*f(x) = ", xfxq1, xfxq2)
    print("Length of xfxq1 ", len(xfxq1))
    print("type of pdf weight ", type(xfxq1))
    pdfNom = xfxq1*xfxq2
    pdfUp = pdf.xfxQ(id1, x1, 2*q2, grid=False)*pdf.xfxQ(id1, x1, 2*q2, grid=False)
    pdfDown = pdf.xfxQ(id1, x1, 0.5*q2, grid=False)*pdf.xfxQ(id1, x1, 0.5*q2, grid=False)
    #### Dont know what alphas do
        #     muRUp = self.lha.evalAlphas(2*q)**4
        # muRDown = self.lha.evalAlphas(0.5*q)**4
    return pdfNom, pdfUp, pdfDown
    