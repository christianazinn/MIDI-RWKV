# MIDI-RWKV

This repository is the official implementation of [Personalizable Long-Context Symbolic Music Infilling with MIDI-RWKV](https://arxiv.org/abs/2506.13001).

## What's inside

This repository contains full code to reproduce the experiments of the MIDI-RWKV paper as well as pretrained base weights, including:

- Weights for the base MIDI-RWKV model
- Inference scripts and custom sampling looppus
- Evaluation metrics and statistical tests, and scripts to run them
- Pretraining scripts
- Finetuning (state tuning and LoRA) scripts
- Instructions to reproduce the results
- The finetuning dataset, [POP909](https://github.com/music-x-lab/POP909-Dataset)

Due to space requirements, this repository *does not* include:

- The pretraining dataset used: [GigaMIDI](https://huggingface.co/datasets/Metacreation/GigaMIDI), which can be found at the link provided
- The LoRA or state tuned checkpoints used in paper experiments

## Requirements

All experiments were run with Python 3.11.11 and are only tested on that version, but they will likely work on 3.9/3.10 as well.

```setup
conda create -n midirwkv python=3.11
conda activate midirwkv
pip install -r requirements.txt
```

It is important that you use `pytorch-lightning==1.9.5` during the training process in particular, but otherwise latest `torch` and `deepspeed` are compatible.

## Pretraining

The pretrained base model used in the paper is located at `midi_rwkv.pth`, if you want to use it in other applications or otherwise skip the pretraining step.

Before doing anything, set the `PROJECT_ROOT` environment variable to the installation location of MIDI-RWKV, e.g. `export PROJECT_ROOT=/home/myname/MIDI-RWKV`. You will also need to authenticate with HuggingFace to access the GigaMIDI dataset, which is gated on HuggingFace. To train the model in the paper, run `train.sh` in the `train` directory, which will process the data for you:

```train
./train/train.sh
```

You can change certain hyperparameters in `train.sh`. Others, mostly related to data processing (see the docstring in `train/src/dataset.py` for full details) are hardcoded in `train.py`. This is a holdover from a previous iteration that used OmegaConf, which does not handle tuples very well.

## Finetuning

State tuning can be run by:

```state
./RWKV-PEFT/scripts/run-state-tuning.sh
```

LoRA can be run by:

```lora
./RWKV-PEFT/scripts/run-lora.sh
```

You will need to point `load_model` in the scripts to the base model checkpoint you want to train on; it is automatically set to the included pretrained weights. You can change hyperparameters in the training scripts themselves.

Please note that the training and finetuning code are broadly unchanged from the [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) and [RWKV-PEFT](https://github.com/JL-er/RWKV-PEFT) repositories from which they are derived (think HF Transformers/PEFT but for RWKV). The major difference is in the custom MIDI dataset and data loading code at `train/src/dataset.py`, which can be injected into an existing working copy of either repository at your discretion to yield the same results.

## Evaluation

To evaluate, you will need to build `rwkv.cpp` for inference. Follow the instructions under that directory to do so.

You will also need to convert the original PyTorch models into a compatible format. We distribute the base PyTorch model for future conversion to different formats, but they can be converted to GGML as follows:

```convert
./train/convert_model_to_cpp.sh midi_rwkv.pth
```

This converts the pretrained base model, but the path can be replaced with the path to any model of your choice.

We are unable to distribute the GigaMIDI test set, but you can create a random subset of 5000 samples from the [HuggingFace dataset](https://huggingface.co/datasets/Metacreation/GigaMIDI) and place them in `rwkv.cpp/python/test_midis/gigamidi_test`. if you ran finetuning on POP909, which is included, the script will automatically place the test set for your training run in `RWKV-PEFT/data/test`, which should be moved to `rwkv.cpp/python/test_midis/pop909_test`.

Then, modify parameters in `rwkv.cpp/python/evaluate.sh` as desired, including the base model path and the path to any state to use (leave empty if not using a state). Generation parameters (number of bars to infill, context, number of generations) and sampling parameters can be set in the script. Run the script and outputs will appear in `MIDIMetrics/output/{model_name}/{inference_parameters}`:

```evaluate
./rwkv.cpp/python/evaluate.sh
```

To compare two models, add them to the `models` array at `rwkv.cpp/python/generate.py:309` (as currently commented out). You will need to set `comparison` in `MIDIMetrics/classes/metric_processor.py:137` to the name of the comparison model (the other model will be considered the "base") to get comparison metrics and statistical tests.

You can also rerun evaluations on finished generations using `MIDIMetrics/evaluate.sh` by setting the same variables as you used in `rwkv.cpp/python/evaluate.sh`.

## Reproduction of paper results

The base hyperparameters in all pretraining and finetuning scripts are exactly as used in the paper, with the exception of `RWKV-PEFT/scripts/run-lora.sh`, which has the values for the LoRA rank=alpha=4 model; the rank=alpha=32 model used `lora_config='{"lora_load":"","lora_r":32,"lora_alpha":32,"lora_dropout":0.0}'`.

Since the finetuning script saves the train and test sets, we delete it between each of three runs of `RWKV-PEFT/scripts/run-state-tuning.sh` to get the results of Table 3, and we reuse the first split for the results of Table 4. To get the results of Table 4, uncomment `rwkv.cpp/python/generate.py:311-313` and rerun  `MIDIMetrics/evaluate.sh` several times for each pairwise comparison, changing `comparison` in `MIDIMetrics/classes/metric_processor.py:137` as necessary.

To get the results of Table 2, uncomment `rwkv.cpp/python/generate.py:316` and set `comparison` to `mistral` in `MIDIMetrics/classes/metric_processor.py:137`.

To get the results of Figures 3-6, set the environment variable `evaluate_acs` to True, which will output the deltas to jsonl files that you can parse. For Figures 4-6, also uncomment `rwkv.cpp/python/generate.py:311-313` to get evaluations for the other three models. 

To get the results of Table 7, uncomment `rwkv.cpp/python/generate.py:322-331` to evaluate over the list of sets of sampling parameters, and comment line 333.

The listening test examples for Tables 5-6 were created with `MIDIMetrics/tests/render_listening_test_examples.py` from the generated MIDIs in the indicated folder. You will need to install `pydub` to run this script, which is not a dependency for the rest of the repository and so not installed by default.

## Citation

If you found this work useful, please cite:

```
@misc{zhouzheng2025personalizable,
  title={Personalizable Long-Context Symbolic Music Infilling with MIDI-RWKV},
  author={Christian Zhou-Zheng and Philippe Pasquier},
  year={2025},
  eprint={2506.13001},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2506.13001},
  doi={10.48550/arXiv.2506.13001}
}
```
