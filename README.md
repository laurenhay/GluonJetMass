# GluonJetMass
Analysis repo for gluon jet mass measurement.

Accessing these jupyter notebooks on LPC:
Connect to LPC and checkout this repository.
```
kinit $USER@FNAL.GOV
ssh -XY -L localhost:####:localhost:####  $USER@cmslpc-sl7.fnal.gov
git checkout BLANK
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
jupyter notebook --no-browser --port=### --ip 127.0.0.1
```
Now you can access your notebooks and files by opening the webpage shown as output (usually http://127.0.0.1:####)
