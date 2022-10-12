# GluonJetMass
Analysis repo for gluon jet mass measurement.


### This repository is currently optimized to run on the coffea casa cluster (https://github.com/CoffeaTeam/coffea-casa).

## Running on LPC:
Connect to LPC and checkout this repository.
```
kinit $USER@FNAL.GOV
ssh -XY -L localhost:####:localhost:####  $USER@cmslpc-sl7.fnal.gov
git checkout https://github.com/laurenhay/GluonJetMass.git
```
Then get a voms ticket:
```
voms-proxy-init -voms cms
```
We will then run the following commands to set up the lpc_dask and Coffea environment
```
curl -OL https://raw.githubusercontent.com/CoffeaTeam/lpcjobqueue/main/bootstrap.sh
bash bootstrap.sh
./shell
jupyter notebook --no-browser --port=#### --ip 127.0.0.1
```
Now you can access your notebooks and files by opening the webpage shown as output (usually http://127.0.0.1:####)

If running locally on lpc, make sure `runCoffeaJob()` is given arguments `dask = False, casa = False`
If you're running on the LPCCondorCluster you should give runCoffeaJob() the arguments `dask = True, casa = False`

## Running on CoffeaCasa


If running locally on coffea casa, make sure `runCoffeaJob()` is given arguments `dask = False, casa = True`
If you're running on the CoffeaCasaCluster you should give runCoffeaJob() the arguments `dask = True, casa = True` and make sure to shut down the automatically generated cluster. More details on running in coffea casa here: https://github.com/b2g-nano/TTbarAllHadUproot/files/9181161/MovingToCoffeaCasa.pdf (slides by AC Williams)



### FUTURE ADDITIONS: Instructions on accessing and running on winterfell
