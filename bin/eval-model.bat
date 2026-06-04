@echo off
rem Evaluate a trained BPE model on the held-out test set.
rem
rem Loads best.pt from the run directory, runs inference on the test split,
rem and writes evaluation outputs to the same directory:
rem   eval_results.json   -- summary metrics (MAE, RMSE, ME, SD; BHS grade; AAMI)
rem   eval_plot.png       -- predicted vs actual scatter plots for SBP and DBP
rem   error_hist.png      -- error distribution histograms for SBP and DBP
rem
rem Usage:
rem   bin\eval-model.bat <run_dir> [OPTIONS]
rem   bin\eval-model.bat data\models\resnet1d\20260101_120000
rem   bin\eval-model.bat data\models\resnet1d\20260101_120000 --dataset-dir data\dataset
rem   bin\eval-model.bat data\models\resnet1d\20260101_120000 --device cuda
rem
rem Options:
rem   --dataset-dir <path>   Root dataset directory  (default: data\dataset)
rem   --device <spec>        auto | cpu | cuda | cuda:N  (default: auto)
rem   --batch-size <N>       Inference batch size    (default: 512)
rem   --no-normalize         Skip per-segment z-score normalization

cd /d "%~dp0.."
uv run python scripts\eval.py %*
