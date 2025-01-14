#!/bin/bash
set -ex

# Format the code
black *.py

python GenerateNeuroML.py -jnml


omv all -V 

