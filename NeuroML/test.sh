#!/bin/bash
set -ex

# Format the code
ruff format *.py

./clean.sh  || true

cd ..
nrniv Test_Soma.hoc
cd -

jnml LEMS_SomaTest.xml -nogui 

python compare.py