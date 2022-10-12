#### based on the trigger emulation method spelled out here https://cms.cern.ch/iCMS/jsp/db_notes/noteInfo.jsp?cmsnoteid=CMS%20AN-2015/154
### Define to accumulate histograms
import awkward as ak
import numpy as np
from coffea import processor, hist, util

class triggerProcessor(processor.ProcessorABC):
    def __init__(self, year = None, trigger = "", data = False):
        self.year = year
        self.trigger = trigger
        self.data = data
        dataset_cat = hist.Cat("dataset", "Dataset")         
        HLT_cat = hist.Cat("HLT_cat", "")
        if data:
            pt_bin = hist.Bin("pt", "Jet pT (GeV)", 100, 0, 2400)
        else: pt_bin = hist.Bin("pt", "Jet pT (GeV)", 100, 0, 3200)
        self._histos = processor.dict_accumulator({
            'hist_trigEff': hist.Hist("Events", dataset_cat, HLT_cat, pt_bin),
            'hist_trigEff_ptCut': hist.Hist("Events", dataset_cat, HLT_cat, pt_bin),
            'hist_trigRef': hist.Hist("Events", dataset_cat, HLT_cat, pt_bin),
            'hist_pt': hist.Hist("Events", dataset_cat, pt_bin),
            'cutflow':      processor.defaultdict_accumulator(int),
            })
    
    @property
    def accumulator(self):
        return self._histos
    def process(self, events):
        out = self.accumulator.identity()
        #### choose events that have at least one fatjet
        events = events[ak.num(events.FatJet) >= 1]
        print("Events metadata: ", events.metadata)
        
        #### Choose trigger objects that belong to jets (id == 1), after this selection remove events 
        #### that don't have corresponding jet trigger objects
        trigObj = events.TrigObj
        print(trigObj.id)
        trigObj = trigObj[trigObj.id == 1]
        # deltaR  = events.FatJet[:, 0].delta_r(trigObj)
        # trigObj = trigObj[deltaR < 0.15]
        events = events[ak.num(trigObj, axis = -1) != 0]
        trigObj = trigObj[ak.num(trigObj, axis = -1) != 0]
        year = self.year
        trigger = self.trigger
        if self.data:
            datastring = "JetHT"
        else:
            datastring = "QCDsim"
        print("trigger")
        #### get HLT objects from events
        if year == 2016:
            # HLT_paths = ['AK8PFJet40', 'AK8PFJet60', 'AK8PFJet80', 'AK8PFJet140', 'AK8PFJet200', 'AK8PFJet260', 'AK8PFJet320', 'AK8PFJet400', 'AK8PFJet450', 'AK8PFJet500']
            trigThresh = [40, 60, 80, 140, 200, 260, 320, 400, 450, 500]
            HLT_paths = [trigger + str(i) for i in trigThresh]
        elif year == 2017:
            # HLT_paths = ['AK8PFJet40', 'AK8PFJet60', 'AK8PFJet80', 'AK8PFJet140', 'AK8PFJet200', 'AK8PFJet260', 'AK8PFJet320', 'AK8PFJet400', 'AK8PFJet450', 'AK8PFJet500', 'AK8PFJet550']
            trigThresh = [40, 60, 80, 140, 200, 260, 320, 400, 450, 500, 550]
            HLT_paths = [trigger + str(i) for i in trigThresh]
        elif year == 2018:
            # HLT_paths = ['AK8PFJet15', 'AK8PFJet25', 'AK8PFJet40', 'AK8PFJet60', 'AK8PFJet80', 'AK8PFJet140', 'AK8PFJet200', 'AK8PFJet260', 'AK8PFJet320', 'AK8PFJet400', 'AK8PFJet450', 'AK8PFJet500', 'AK8PFJet550']
            trigThresh = [15, 25, 40, 60, 80, 140, 200, 260, 320, 400, 450, 500, 550]
            HLT_paths = [trigger + str(i) for i in trigThresh]
        else:
            print("Must provide a year to run trigger studies")
        
        #### Loop over HLT paths and calculate efficiencies for each
        efficiencies = {}
        for i in np.arange(len(HLT_paths)):
            path = HLT_paths[i]
            print("index: ", i, " path: ", path)
            if i > 0:
                print("Number of events w/ trigger ", ak.count_nonzero(events.HLT[HLT_paths[i]]))
                print("HLT path: ", path, " HLT values: ", events.HLT[HLT_paths[i]], " Number of Trues ", ak.count_nonzero(events.HLT[HLT_paths[i]]))
                print("HLT path: ", HLT_paths[i-1], " HLT values: ", events.HLT[HLT_paths[i-1]], " Number of Trues ", ak.count_nonzero(events.HLT[HLT_paths[i-1]]))
                
                efficiencies[path] = events.FatJet[:,0].pt[(events.HLT[HLT_paths[i-1]]) & (trigObj.pt[:,0] > trigThresh[i])]
                
                out['hist_trigEff_ptCut'].fill(dataset = datastring + str(year), HLT_cat = path, pt = efficiencies[path])
                out['hist_trigEff'].fill(dataset = datastring + str(year), HLT_cat = path, pt = events.FatJet[:,0].pt[(events.HLT[HLT_paths[i-1]]) & (events.HLT[HLT_paths[i]])])
                out['hist_trigRef'].fill(dataset = datastring +  str(year), HLT_cat = path, pt = events.FatJet[:,0].pt[(events.HLT[HLT_paths[i-1]])])
                out['hist_pt'].fill(dataset = datastring + str(year), pt = events.FatJet[:,0].pt)
                
       
                ####if using new hist format
                #hist_trigEff.fill(HLT_cat = path, jet_pt = events.FatJet[:,0].pt[(events.HLT[HLT_paths[i]])])
                #hist_trigRef.fill(HLT_cat = path, jet_pt = events.FatJet[:,0].pt[(events.HLT[HLT_paths[i-1]])])
        print("Final efficiencies:", efficiencies)
        return out
    
    def postprocess(self, accumulator):
        return accumulator
    
#### APPLY PRESCALES NOT WORKING YET    
class applyPrescales(processor.ProcessorABC):
    def __init__(self, prescales):
        self.prescales = prescales
        dataset_cat = hist.Cat("dataset", "Dataset")         
        HLT_cat = hist.Cat("HLT_cat", "")
        if data:
            pt_bin = hist.Bin("pt", "Jet pT (GeV)", 100, 0, 2400)
        else: pt_bin = hist.Bin("pt", "Jet pT (GeV)", 100, 0, 3200)
        self._histos = processor.dict_accumulator({
            'hist_pt': hist.Hist("Events", dataset_cat, pt_bin),
            'hist_pt_byHLTpath': hist.Hist("Events", dataset_cat, HLT_path, pt_bin),
            'hist_pt_byHLTpath_removeDoubles': hist.Hist("Events", dataset_cat, HLT_path, pt_bin),
            'cutflow':      processor.defaultdict_accumulator(int),
            })
    @property
    def accumulator(self):
        return self._histos
    def process(self, events):
        out = self.accumulator.identity()
        return out
    def postprocess(self, accumulator):
        prescales = self.prescales
        
        return accumulator
            
    
#### Next run processor with futures executor on all test files
from dask.distributed import Client
from plugins import runCoffeaJob
#switch to if name and arg parse instead of separate function?
result = runCoffeaJob(triggerProcessor(year = 2017, trigger = 'AK8PFJet', data = True), jsonFile = "datasets_UL_NANOAOD.json", casa = True, dask = True, testing = False, year = 2017, data = True)
# result = runCoffeaJob(triggerProcessor(year = 2018, trigger = 'AK8PFJet', data = False), jsonFile = "fileset_QCD.json", casa = True, dask = False, testing = True, year = 2018)
util.save(result, 'coffeaOutput/DiJet_2017_JetHT_AK8prescale_result.coffea')