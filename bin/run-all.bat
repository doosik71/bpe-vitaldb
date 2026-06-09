@echo off
rem run-all.bat — run the full BPE-VitalDB pipeline end to end.
rem
rem One-time steps (run once regardless of model count):
rem   1  Check CUDA availability              (informational — never aborts the run)
rem   2  Download VitalDB .vital files        (--skip-download to bypass)
rem   3  Build train / val / test NPZ dataset (--skip-build   to bypass)
rem
rem Per-model loop (repeated for every model in the list):
rem   4  Train the model
rem   5  Plot training curves (train-status)  (non-critical)
rem   6  Evaluate model on the test set
rem      (pulsewoq_resnet1d uses eval-model-pulsewoq.py automatically)
rem
rem One-time steps (run once after all models finish):
rem   7  Collect eval results and checkpoints (non-critical)
rem   8  Generate overview parameter/metric graph (non-critical)
rem
rem Usage:
rem   bin\run-all.bat [OPTIONS]
rem
rem Options:
rem   --model MODELS       Comma-separated model list             (default: all)
rem                        e.g. --model resnet1d,st_resnet,minception
rem                        If omitted, all models except pulsewoq_resnet1d are run.
rem   --max-cases N        Limit downloaded cases (blank = all)
rem   --epochs N           Maximum training epochs                (default: 100)
rem   --batch-size N       Mini-batch size                        (default: 256)
rem   --device DEVICE      Training / inference device            (default: auto)
rem   --seed N             Random seed                            (default: 42)
rem   --data-dir DIR       VitalDB .vital files directory         (default: data\vitaldb)
rem   --dataset-dir DIR    NPZ dataset root directory             (default: data\dataset)
rem   --models-dir DIR     Models root directory                  (default: data\models)
rem   --skip-download      Skip step 2 (download)
rem   --skip-build         Skip step 3 (dataset construction)
rem   --help               Show this help message and exit
rem
rem Examples:
rem   bin\run-all.bat
rem   bin\run-all.bat --model resnet1d
rem   bin\run-all.bat --model resnet1d,st_resnet,minception
rem   bin\run-all.bat --skip-download --skip-build --epochs 200
rem   bin\run-all.bat --model resnet1d --max-cases 500 --device cuda

setlocal enabledelayedexpansion
cd /d "%~dp0.."

rem ── All available models (default when --model is omitted) ───────────────────
rem pulsewoq_resnet1d is excluded from default: it uses a separate eval script.
set ALL_MODELS=acfa minception mtae mtae_tr resnet1d resnet1d_micro resnet1d_mini resnet1d_tiny st_resnet xresnet1d

rem ── Defaults ─────────────────────────────────────────────────────────────────
set MODEL_ARG=
set MAX_CASES=
set EPOCHS=100
set BATCH_SIZE=256
set DEVICE=auto
set SEED=42
set DATA_DIR=data\vitaldb
set DATASET_DIR=data\dataset
set MODELS_DIR=data\models
set SKIP_DOWNLOAD=0
set SKIP_BUILD=0
set STEP=0

rem ── Argument parsing ──────────────────────────────────────────────────────────
:parse_args
if "%~1"=="" goto :after_parse
if /i "%~1"=="--model"          ( set "MODEL_ARG=%~2"&   shift& shift& goto :parse_args )
if /i "%~1"=="--max-cases"      ( set "MAX_CASES=%~2"&   shift& shift& goto :parse_args )
if /i "%~1"=="--epochs"         ( set "EPOCHS=%~2"&      shift& shift& goto :parse_args )
if /i "%~1"=="--batch-size"     ( set "BATCH_SIZE=%~2"&  shift& shift& goto :parse_args )
if /i "%~1"=="--device"         ( set "DEVICE=%~2"&      shift& shift& goto :parse_args )
if /i "%~1"=="--seed"           ( set "SEED=%~2"&        shift& shift& goto :parse_args )
if /i "%~1"=="--data-dir"       ( set "DATA_DIR=%~2"&    shift& shift& goto :parse_args )
if /i "%~1"=="--dataset-dir"    ( set "DATASET_DIR=%~2"& shift& shift& goto :parse_args )
if /i "%~1"=="--models-dir"     ( set "MODELS_DIR=%~2"&  shift& shift& goto :parse_args )
if /i "%~1"=="--skip-download"  ( set SKIP_DOWNLOAD=1&   shift&        goto :parse_args )
if /i "%~1"=="--skip-build"     ( set SKIP_BUILD=1&      shift&        goto :parse_args )
if /i "%~1"=="--help"           goto :show_help
echo [ERROR] Unknown option: %~1
echo Run 'bin\run-all.bat --help' for usage.
exit /b 1

:after_parse
rem Build space-separated model list from comma-separated --model arg
if "%MODEL_ARG%"=="" (
    set "MODEL_LIST=%ALL_MODELS%"
) else (
    set "MODEL_LIST=%MODEL_ARG:,= %"
)

rem ── Print configuration ───────────────────────────────────────────────────────
echo ================================================================
echo   BPE-VitalDB  --  Full Pipeline Run
echo ================================================================
echo   Models      : %MODEL_LIST%
echo   Epochs      : %EPOCHS%
echo   Batch size  : %BATCH_SIZE%
echo   Device      : %DEVICE%
echo   Seed        : %SEED%
echo   Data dir    : %DATA_DIR%
echo   Dataset dir : %DATASET_DIR%
echo   Models dir  : %MODELS_DIR%
if not "%MAX_CASES%"=="" echo   Max cases   : %MAX_CASES%
if "%SKIP_DOWNLOAD%"=="1" echo   (skip download)
if "%SKIP_BUILD%"=="1"    echo   (skip build)
echo.

rem ────────────────────────────────────────────────────────────────────────────
rem  One-time steps
rem ────────────────────────────────────────────────────────────────────────────

call :step_header "Check CUDA"
uv run python scripts\check-cuda.py
if errorlevel 1 echo [WARN] check-cuda failed -- continuing.

call :step_header "Download VitalDB"
if "%SKIP_DOWNLOAD%"=="1" (
    echo   [SKIP] --skip-download specified.
) else (
    if "%MAX_CASES%"=="" (
        uv run python scripts\download-vitaldb.py --output-dir "%DATA_DIR%" --filter-tracks
    ) else (
        uv run python scripts\download-vitaldb.py --output-dir "%DATA_DIR%" --filter-tracks --max-cases %MAX_CASES%
    )
    if errorlevel 1 (
        echo [FAIL] download-vitaldb.py failed.
        exit /b 1
    )
)

call :step_header "Build Dataset"
if "%SKIP_BUILD%"=="1" (
    echo   [SKIP] --skip-build specified.
) else (
    uv run python scripts\construct-dataset.py --data-dir "%DATA_DIR%" --output-dir "%DATASET_DIR%"
    if errorlevel 1 (
        echo [FAIL] construct-dataset.py failed.
        exit /b 1
    )
)

rem ────────────────────────────────────────────────────────────────────────────
rem  Per-model loop
rem ────────────────────────────────────────────────────────────────────────────

set FAILED_MODELS=
set OK_MODELS=
set MODEL_IDX=0

for %%m in (%MODEL_LIST%) do (
    set /a MODEL_IDX+=1
    set CURRENT_MODEL=%%m
    set RUN_DIR=%MODELS_DIR%\%%m
    set MODEL_FAILED=0

    call :step_header "Train: %%m"
    uv run python scripts\train-model.py ^
        --model       %%m ^
        --dataset-dir "%DATASET_DIR%" ^
        --output-dir  "%MODELS_DIR%" ^
        --epochs      %EPOCHS% ^
        --batch-size  %BATCH_SIZE% ^
        --device      %DEVICE% ^
        --seed        %SEED%
    if errorlevel 1 (
        echo [WARN] train-model.py failed for %%m -- skipping eval.
        set "FAILED_MODELS=!FAILED_MODELS! %%m(train)"
        set MODEL_FAILED=1
    )

    if "!MODEL_FAILED!"=="0" (
        call :step_header "Training Status: %%m"
        uv run python scripts\train-status.py "%MODELS_DIR%\%%m"
        if errorlevel 1 echo [WARN] train-status.py failed for %%m -- continuing.

        call :step_header "Evaluate: %%m"
        if /i "%%m"=="pulsewoq_resnet1d" (
            set EVAL_SCRIPT=scripts\eval-model-pulsewoq.py
        ) else (
            set EVAL_SCRIPT=scripts\eval-model.py
        )
        uv run python !EVAL_SCRIPT! "%MODELS_DIR%\%%m" --dataset-dir "%DATASET_DIR%" --device %DEVICE%
        if errorlevel 1 (
            echo [WARN] eval failed for %%m.
            set "FAILED_MODELS=!FAILED_MODELS! %%m(eval)"
        ) else (
            set "OK_MODELS=!OK_MODELS! %%m"
        )
    )
)

rem ────────────────────────────────────────────────────────────────────────────
rem  One-time post-loop steps
rem ────────────────────────────────────────────────────────────────────────────

call :step_header "Collect Results"
uv run python scripts\collect-result.py --models-dir "%MODELS_DIR%"
if errorlevel 1 echo [WARN] collect-result.py failed -- continuing.

call :step_header "Overview Graph"
uv run python scripts\generate-overview-graph.py --models-dir "%MODELS_DIR%"
if errorlevel 1 echo [WARN] generate-overview-graph.py failed -- continuing.

rem ── Summary ───────────────────────────────────────────────────────────────────
echo.
echo ================================================================
echo   Pipeline complete.
echo.
if not "%OK_MODELS%"==""     echo   OK     : %OK_MODELS%
if not "%FAILED_MODELS%"=="" echo   Failed : %FAILED_MODELS%
echo ================================================================
exit /b 0

rem ── Subroutines ───────────────────────────────────────────────────────────────
:step_header
set /a STEP+=1
echo.
echo ================================================================
echo   [%STEP%] %~1
echo ================================================================
echo.
exit /b 0

:show_help
for /f "tokens=* delims=" %%L in ('findstr /b "rem " "%~f0"') do (
    set "line=%%L"
    set "line=!line:rem =!"
    echo !line!
)
exit /b 0
