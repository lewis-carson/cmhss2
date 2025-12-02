#!/bin/bash
#SBATCH --job-name=download_letters
#SBATCH --output=download.log
#SBATCH --error=download.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --partition=shared

python3.11 -u download.py
