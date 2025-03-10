#!/bin/bash
set -ex

# Format the code
ruff format *.py

python GenerateNeuroML.py -jnml


omv all -V 

