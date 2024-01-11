import numpy as np
import awkward as ak
import correctionlib
from coffea.jetmet_tools import JetResolutionScaleFactor
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from coffea.lookup_tools import extractor
import copy

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

def GetPUSF(events, IOV):
    # original code https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/TTbarDileptonProcessor.py#L38
    ## json files from: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/LUM
        
    fname = "data/corrections/puWeights/{0}_UL/puWeights.json.gz".format(IOV)
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

    return [puNom, puUp, puDown]