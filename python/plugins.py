### This file houses helper functions for running processors
import pandas as pd
import time
from coffea import processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import pickle

#if using LPC dask or running locally use 'root://cmsxrootd.fnal.gov/'
#is using coffea casa use 'root://xcache/'

import os
import warnings
warnings.filterwarnings("ignore")
from distributed.diagnostics.plugin import UploadDirectory

def checkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


#reads in files and adds redirector, can specify year, default is all years
def handleData(jsonFile, redirector, year = '', testing = True, data = False, chunks = None):
    eras_data = {'2016':   '-UL2016',
                 '2016APV':'HIPM_UL2016', 
                 '2017':   'UL2017',
                 '2018':   'UL2018'
                    }
    eras_mc = {'2016APV':'UL16NanoAODAPV', 
               '2016':'UL16NanoAODv9', 
               '2017':'UL17NanoAODv9', 
               '2018':'UL18NanoAODv9'
              }
    dict = {}
    qualifiers = []
    if data:
        inputs = 'JetHT_data'
        if year == '2016' or year == '2016APV' or year == '2018' or year == '2017':
            qualifiers.append(eras_data[year])
        elif year == 2016 or year == 2018 or year == 2017:
            qualifiers.append(eras_data[str(year)]) 
        else:
            for era in list(eras_data.values()):
                print("Era: ", era)
                qualifiers.append(era)
    else:
        if (redirector == '/mnt/data/cms'):
            jsonFile = "QCD_flat_files.json"
            inputs = 'QCD_flat'
        else:
            inputs = 'QCD_binned'
        if year == '2016' or year == '2017' or year == '2018' or year == '2016APV':
            qualifiers.append(eras_mc[year])
        elif year == 2016 or year == 2017 or year == 2018:
            qualifiers.append(eras_mc[str(year)])
        else:
            for era in list(eras_mc.values()):
                print("Era: ", era)
                qualifiers.append(era)
    df = pd.read_json(jsonFile) 
    for key in df[inputs].keys():
        for qualifier in qualifiers:
            if qualifier in key:
                if testing:
                    dict[key] = [redirector +  df[inputs][key][0]]
                else:
                    dict[key] = [redirector + df[inputs][key][i] for i in range(len(df[inputs][key]))]
    return dict

#initiate dask client and run coffea job
from dask.distributed import Client

def runCoffeaJob(processor_inst, jsonFile, dask = False, casa = False, testing = False, year = '', data = False, winterfell = False, verbose = True, datasetRange = None):
    #default is to run locally
    tstart = time.time()
    executor = processor.futures_executor
    if casa:
        redirector = 'root://xcache/'
    elif winterfell:
        #### only data (not mc/sim) is on winterfell atm -- make sure using data json file and arguments
        redirector = '/mnt/data/cms'
    # elif data and not winterfell and not casa:
    #     redirector = 'root://cmsxrootd.fnal.gov/'
        #### default redirector
    else:
        redirector = ''
        #### Nebraska redirector
        # redirector= 'root://xrootd-local.unl.edu:1094/'
        #### MIT redirector
        # redirecotr='root://xrootd.cmsaf.mit.edu:1094/'
        # redirector='root://cmseos.fnal.gov:1094/'
        # redirector='root://cmseos.fnal.gov/'
        # redirector='root://cmsxrootd.hep.wisc.edu/'
    exe_args = {"schema": NanoAODSchema, 'skipbadfiles': False,}
    samples = handleData(jsonFile, redirector, year = year, testing = testing, data = data)
    if datasetRange!=None:
        print("Total avail datasets ", len(samples.keys()))
        samples = {key:samples[key] for key in list(samples.keys())[datasetRange[0]:datasetRange[1]]}
    #single files for testing
    # samples = {'/JetHT/Run2016E-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD': [redirector+'/store/data/Run2016E/JetHT/NANOAOD/HIPM_UL2016_MiniAODv2_NanoAODv9-v2/40000/0402FC45-D69F-BE47-A2BF-10394485E06E.root']}
    # samples = {'/QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM': ['root://cmsxrootd.fnal.gov//store/mc/RunIISummer20UL18NanoAODv9/QCD_Pt_1400to1800_TuneCP5_13TeV_pythia8/NANOAODSIM/106X_upgrade2018_realistic_v16_L1v1-v1/280000/2CD900FB-1F6B-664F-8A26-C125B36C2B58.root']}
    
    print("Running over datasets ", samples.keys())
    client = None
    cluster = None
    if casa and dask:
        print("Running on coffea casa")
        from coffea_casa import CoffeaCasaCluster
        client = Client("tls://lauren-2emeryl-2ehay-40cern-2ech.dask.cmsaf-prod.flatiron.hollandhpc.org:8786")
        # client.register_worker_plugin(UploadDirectory("/home/cms-jovyan/GluonJetMass", restart=True, update_path=True), nanny=True)
        client.upload_file("fileset_QCD.json")
        client.upload_file("datasets_UL_NANOAOD.json")
        client.upload_file("python/plugins.py")
        client.upload_file("python/utils.py")
        client.upload_file("python/corrections.py")
        client.upload_file("python/trijetProcessor.py") #upload additional files to the client                               
        client.upload_file("python/dijetProcessor.py")
        client.upload_file("python/triggerProcessor.py")
        client.upload_file("correctionFiles/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt")
        client.upload_file("correctionFiles/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt")
        client.upload_file("correctionFiles/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt")
        client.upload_file("correctionFiles/ps_weight_JSON_2016.json")
        client.upload_file("correctionFiles/ps_weight_JSON_2017.json")
        client.upload_file("correctionFiles/ps_weight_JSON_2018.json")
        client.upload_file("correctionFiles/ps_weightAK4_JSON_2016.json")
        # cluster = CoffeaCasaCluster(cores=11, memory="20 GiB", death_timeout = 60)
        # cluster.adapt(minimum=2, maximum=14)
        # client = Client(cluster)
        print(client)
        exe_args = {
            "client": client,
            "status":False,
            "skipbadfiles":False,
            "schema": NanoAODSchema,
            "align_clusters": True,
        }
        executor = processor.dask_executor
        result = processor.run_uproot_job(samples,
                                          "Events",
                                          processor_instance = processor_inst,
                                          executor = executor,
                                          executor_args = exe_args,
                                     )
    elif casa == False and dask:
        print("Running on LPC Condor")
        from lpcjobqueue import LPCCondorCluster
        #### make list of files and directories to upload to dask
        upload_to_dask = ['correctionFiles', 'python']
        cluster = LPCCondorCluster(memory='5 GiB', transfer_input_files=upload_to_dask, scheduler_options={"dashboard_address": year})#, ship_env=False)
        #### minimum > 0: https://github.com/CoffeaTeam/coffea/issues/465
        cluster.adapt(minimum=1, maximum=100)
        with Client(cluster) as client:
            if verbose:
                run_instance = processor.Runner(
                                executor=processor.DaskExecutor(client=client, retries=5),#, status=False),
                                schema=NanoAODSchema,
                                savemetrics=True,
                                skipbadfiles=False,
                                chunksize=20000,
                            )
            else:
                run_instance = processor.Runner(
                                executor=processor.DaskExecutor(client=client, retries=5, status=False),
                                schema=NanoAODSchema,
                                savemetrics=True,
                                skipbadfiles=False,
                                chunksize=20000,
                            )
            # result, metrics = run_instance(samples,
            #                                "Events",
            #                                processor_instance = processor_inst,)
            with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result, metrics = run_instance(samples, 
                                                   "Events",
                                                   processor_instance=processor_inst,)
                    del metrics

#         print("Waiting for at least one worker...")
    else:
        print("Running locally")
        result = processor.run_uproot_job(samples,
                                          "Events",
                                          processor_instance = processor_inst,
                                          executor = executor,
                                          executor_args = exe_args,
                                     )
    elapsed = time.time() - tstart
    print(result)
    print("Time taken to run over samples ", elapsed)
    del cluster
    return result
def addFiles(files, RespOnly=False):
    results = pickle.load( open(files[0], "rb") )
    respHists = ['response_matrix_u', 'response_matrix_g', 'ptreco_mreco_u', 'ptreco_mreco_g', 'ptgen_mgen_u', 'ptgen_mgen_g','fakes', 'misses']
    for fname in files[1:]:
        print(fname)
        with open(fname, "rb") as f:
            result = pickle.load( f )
            for hist in [res for res in result if res in results]:
                if RespOnly and hist in respHists:
                    print("Hist ", hist, " with syst cats ",[bin for bin in result['response_matrix_u'].project("syst").axes])
                    results[hist] += result[hist]
                else:
                    print("Hist ", hist, " with syst cats ",[bin for bin in result['response_matrix_u'].project("syst").axes])
                    results[hist] += result[hist]
                print("Successfully added")
    print("Done")
    return(results)
    