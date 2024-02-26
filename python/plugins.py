


### This file houses plugins
import pandas as pd
import time
from coffea import processor
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
import pickle

#if using LPC dask or running locally use 'root://cmsxrootd.fnal.gov/'
#is using coffea casa use 'root://xcache/'

import os
from distributed.diagnostics.plugin import UploadDirectory

def checkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


#reads in files and adds redirector, can specify year, default is all years
def handleData(jsonFile, redirector, year = '', testing = True, data = False):
    eras_data = {'2016':   '-UL2016',
                 '2016APV':'HIPM_UL2016', 
                 '2017':   'UL2017',
                 '2018':   'UL2018'
                    }
    eras_mc = {'2016':'UL16', 
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
            jsonFile = "fileset_QCD.json"
            inputs = 'QCD_binned'
        if year == '2016' or year == '2017' or year == '2018':
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
                # print("dataset = ", key)
                if testing:
                    dict[key] = [redirector +  df[inputs][key][0]]
                else:
                    dict[key] = [redirector + df[inputs][key][i] for i in range(len(df[inputs][key]))]
    return dict

#initiate dask client and run coffea job
from dask.distributed import Client

def runCoffeaJob(processor_inst, jsonFile, dask = False, casa = False, testing = False, year = '', data = False, winterfell = False, verbose = True):
    #default is to run locally
    tstart = time.time()
    executor = processor.futures_executor
    if casa:
        redirector = 'root://xcache/'
    elif winterfell:
        #### only data (not mc/sim) is on winterfell atm -- make sure using data json file and arguments
        redirector = '/mnt/data/cms'
    else:
        redirector = 'root://cmsxrootd.fnal.gov/'
    exe_args = {"schema": NanoAODSchema, 'skipbadfiles': True,}
    samples = handleData(jsonFile, redirector, year = year, testing = testing, data = data)
    #single files for testing
    # samples = {'/JetHT/Run2016E-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD': [redirector+'/store/data/Run2016E/JetHT/NANOAOD/HIPM_UL2016_MiniAODv2_NanoAODv9-v2/40000/0402FC45-D69F-BE47-A2BF-10394485E06E.root']}
    # samples = {'/QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM': [redirector+'/store/mc/RunIISummer20UL17NanoAODv9/QCD_Pt_1800to2400_TuneCP5_13TeV_pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v1/270000/00DD1153-F006-3446-ABBC-7CA23A020566.root']}
#    print("Samples = ", samples, " executor = ", executor)
    client = None
    cluster = None
    if casa and dask:
        print("Running on coffea casa")
        from coffea_casa import CoffeaCasaCluster
        client = Client("tls://lauren-2emeryl-2ehay-40cern-2ech.dask.cmsaf-prod.flatiron.hollandhpc.org:8786")
        # client.register_worker_plugin(UploadDirectory("/home/cms-jovyan/GluonJetMass", restart=True, update_path=True), nanny=True)
        client.upload_file("plugins.py")
        client.upload_file("utils.py")
        client.upload_file("corrections.py")
        client.upload_file("trijetProcessor.py") #upload additional files to the client                               
        client.upload_file("dijetProcessor.py")
        client.upload_file("correctionFiles/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt")
        client.upload_file("correctionFiles/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt")
        client.upload_file("correctionFiles/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt")
        client.upload_file("correctionFiles/ps_weight_JSON_2016.json")
        client.upload_file("correctionFiles/ps_weight_JSON_2017.json")
        client.upload_file("correctionFiles/ps_weight_JSON_2018.json")
        # cluster = CoffeaCasaCluster(cores=11, memory="20 GiB", death_timeout = 60)
        # cluster.adapt(minimum=2, maximum=14)
        # client = Client(cluster)
        print(client)
        exe_args = {
            "client": client,
            "status":False,
            "skipbadfiles":True,
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
        cluster = LPCCondorCluster(memory='5 GiB', transfer_input_files=upload_to_dask)
        #### minimum > 0: https://github.com/CoffeaTeam/coffea/issues/465
        cluster.adapt(minimum=1, maximum=10)
        with Client(cluster) as client:
            if verbose:
                run_instance = processor.Runner(
                                executor=processor.DaskExecutor(client=client, retries=5),#, status=False),
                                schema=NanoAODSchema,
                                savemetrics=True,
                                skipbadfiles=True,
                                # chunksize=10000,
                                # maxchunks=10,
                            )
            else:
                run_instance = processor.Runner(
                                executor=processor.DaskExecutor(client=client, retries=5, status=False),
                                schema=NanoAODSchema,
                                savemetrics=True,
                                skipbadfiles=True,
                                # chunksize=10000,
                                # maxchunks=10,
                            )
            result, metrics = run_instance(samples,
                                           "Events",
                                           processor_instance = processor_inst,)
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
    return result
def addFiles(files):
    results = pickle.load( open(files[0], "rb") )
    for fname in files[1:]:
        with open(fname, "rb") as f:
            result = pickle.load( f )
            for hist in result:
                results[hist] += result[hist]
    outputFilename = files[0][:-8]+"ALL.pkl"
    with open(outputFilename, "wb") as f:
        pickle.dump( results, f)
    return(outputFilename)
    