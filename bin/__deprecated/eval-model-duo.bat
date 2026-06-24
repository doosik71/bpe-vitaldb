@echo off
rem Evaluate a two-model duo ensemble on the held-out test set.
rem
rem Runs two independently trained models in parallel and rejects measurements
rem when either model disagrees by >= threshold mmHg on SBP or DBP.
rem Accepted prediction = average of both model outputs.
rem
rem Output directory receives:
rem   eval_results.json     -- metrics for all and accepted segments + rejection stats
rem   eval_plot_all.png     -- predicted vs actual scatter (all segments)
rem   eval_plot.png         -- predicted vs actual scatter (accepted segments)
rem   error_hist_all.png    -- error histogram (all segments)
rem   error_hist.png        -- error histogram (accepted segments)
rem   diff_dist.png         -- inter-model disagreement distribution
rem
rem Usage:
rem   bin\eval-model-duo.bat <output_dir> [OPTIONS]
rem   bin\eval-model-duo.bat data\models\duo_conv_reg_ds_mtae
rem   bin\eval-model-duo.bat data\models\duo_5mmhg --duo-models conv_reg_ds mtae --duo-threshold 5.0
rem
rem Options:
rem   --duo-models <A> <B>   Two model IDs (default: conv_reg_ds mtae)
rem   --duo-threshold <T>    Rejection threshold in mmHg (default: 5.0)
rem   --models-dir <path>    Root models directory (default: data\models)
rem   --dataset-dir <path>   Root dataset directory (default: data\dataset)
rem   --device <spec>        auto | cpu | cuda | cuda:N  (default: auto)
rem   --batch-size <N>       Inference batch size (default: 512)
rem   --no-normalize         Skip per-segment z-score normalization

cd /d "%~dp0.."
uv run python scripts\eval-model.py %* --duo
