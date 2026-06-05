@echo off
rem Analyse train / val / test dataset splits and produce summary statistics.
rem
rem Reads every <case>.npz in data\dataset\{train,val,test} and writes:
rem   data\dataset\statistic.json          -- numerical summary (JSON)
rem   data\dataset\bp_distribution.png     -- SBP / DBP density histograms per split
rem   data\dataset\segments_per_case.png   -- per-case segment-count distribution
rem
rem Usage:
rem   bin\dataset-statistic.bat [OPTIONS]
rem   bin\dataset-statistic.bat --dataset-dir data\dataset
rem
rem Options:
rem   --dataset-dir <path>   Root dataset directory  (default: data\dataset)

cd /d "%~dp0.."
uv run python scripts\dataset-statistic.py %*
