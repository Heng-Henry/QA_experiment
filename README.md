# QA_experiment

QA experiment code snapshot extracted from local `project_0209` workspace.

## Purpose

This repository contains code changes made specifically to run and compare Cobra-based text QA experiments on these 4 tasks:

1. HotpotQA (`longbench_hotpotqa`)
2. 2WikiMultihopQA (`longbench_2wikimqa`)
3. QMSum (`longbench_qmsum`)
4. NarrativeQA (`longbench_narrativeqa`)

## Modified code scope

The uploaded changes are focused on:

- `cobra_1115/`
  - Text QA evaluation entrypoints and LongBench runners
  - Quantized Cobra runtime hooks used during QA evaluation
- `cobra_1115-evaluation/`
  - `vlm_eval.sh` text-QA trigger/sweep integration
  - Slurm submission scripts for QA runs and ablations
  - Metrics collection script for text-QA outputs

Included scope:
- Cobra text-QA evaluation scripts
- Cobra text-QA slurm launchers and sweep submitters
- Text-QA metrics collector
- Quantized Cobra runtime files modified for QA pipeline runs

This repo intentionally excludes:
- model weights / HF cache
- slurm logs and run outputs
- unrelated project files
