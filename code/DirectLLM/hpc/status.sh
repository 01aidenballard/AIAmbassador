#!/bin/bash

# Define bold formatting
BOLD=$(tput bold)
NORMAL=$(tput sgr0)

echo "${BOLD}> Status${NORMAL}"
squeue -u isj0001
echo ""
echo "${BOLD}> Error${NORMAL}"
cat hpc/direct_llm.err
echo ""
echo "${BOLD}> Output${NORMAL}"
cat hpc/direct_llm.out

