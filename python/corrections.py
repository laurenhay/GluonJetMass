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
    print("Got L1 weights")
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

def GetPUSF(events, IOV):
    # original code https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/TTbarDileptonProcessor.py#L38
    ## json files from: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/LUM
        
    fname = "correctionFiles/puWeights/{0}_UL/puWeights.json.gz".format(IOV)
    print("PU SF filename: ", fname)
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
def GetJetCorrections(FatJets, events, era, IOV, isData=False):
    #### I haven't had any issues where i need an "upload directory" like here
    # uploadDir = 'srv/' for lpcjobqueue shell or TTbarAllHadUproot/ for coffea casa
    # uploadDir = os.getcwd().replace('/','') + '/'
    # if 'TTbarAllHadUproot' in uploadDir: 
    #     uploadDir = 'TTbarAllHadUproot/'
    # elif 'jovyan' in uploadDir:
    #     uploadDir = 'TTbarAllHadUproot/'
    # else:
    #     uploadDir = 'srv/'

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
            print("JER tag: ", jer_tag)
            print("File "+'correctionFiles/JER/{0}/{0}_PtResolution_AK8PFPuppi.jr.txt'.format(jer_tag)+" exists: ", os.path.exists('correctionFiles/JER/{0}/{0}_PtResolution_AK8PFPuppi.jr.txt'.format(jer_tag)))
            print("File "+'correctionFiles/JER/{0}/{0}_SF_AK8PFPuppi.jersf.txt'.format(jer_tag)+" exists: ", os.path.exists('correctionFiles/JER/{0}/{0}_SF_AK8PFPuppi.jersf.txt'.format(jer_tag)))
            ext.add_weight_sets([
            '* * '+'correctionFiles/JER/{0}/{0}_PtResolution_AK8PFPuppi.jr.txt'.format(jer_tag),
            '* * '+'correctionFiles/JER/{0}/{0}_SF_AK8PFPuppi.jersf.txt'.format(jer_tag)])
            print("JER SF added")
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

    print("Making evaluator")

    evaluator = ext.make_evaluator()



    if (not isData):
        jec_names = [
            '{0}_L1FastJet_AK8PFPuppi'.format(jec_tag),
            '{0}_L2Relative_AK8PFPuppi'.format(jec_tag),
            '{0}_L3Absolute_AK8PFPuppi'.format(jec_tag),
            '{0}_Uncertainty_AK8PFPuppi'.format(jec_tag)]

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
    print("Built corrections object")
    return corrected_jets