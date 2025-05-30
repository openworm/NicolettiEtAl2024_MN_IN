#!/bin/bash
set -ex

# Format the code
ruff format *.py

./clean.sh  || true

cd ..
nrniv Test_Soma2.hoc
cd -

jnml LEMS_SomaTest2.xml -nogui 

python compare2.py