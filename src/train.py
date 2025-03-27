########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

from copy import deepcopy
from symusic import Score
from miditok.utils import get_bars_ticks
from pathlib import Path
from tqdm import tqdm
import logging
from omegaconf import OmegaConf
import hydra
logging.basicConfig(level=logging.INFO)

SCORE_LOADING_EXCEPTION = (
    RuntimeError,
    ValueError,
    OSError,
    FileNotFoundError,
    IOError,
    EOFError,
)

TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 1): 12, (1, 2): 4, (2, 4): 2, (4, 8): 1},
    "num_velocities": 24,
    "special_tokens": [
        "PAD",
        "BOS",
        "EOS",
        "Infill_Bar",  # Indicates a bar to be filled in a seq
        "Infill_Track",  # Used in seq2seq to instruct the decoder to gen a new track
        "FillBar_Start",  # Start of the portion to infill (containing n bars)
        "FillBar_End",  # Ends the portion to infill
    ],
    "use_chords": False,
    "use_rests": False,
    "use_tempos": True,
    "use_time_signatures": True,
    "use_pitch_intervals": False,  # cannot be used as extracting tokens in data loading
    "use_programs": True,
    "num_tempos": 48,
    "tempo_range": (50, 200),
    "programs": list(range(-1, 127)),
    "base_tokenizer": "REMI",
    "ac_polyphony_bar": True,
    "ac_polyphony_track": True,
    "ac_polyphony_min": 1,
    "ac_polyphony_max": 6,
    "ac_pitch_class_bar": True,
    "ac_note_density_track": True,
    "ac_note_density_track_min": 0,
    "ac_note_density_track_max": 18,
    "ac_note_density_bar": True,
    "ac_note_density_bar_max": 18,
    "ac_note_duration_bar": True,
    "ac_note_duration_track": True,
    "ac_repetition_track": True,
    "ac_repetition_track_num_bins": 10,
    "ac_repetition_track_num_consec_bars": 4,
}

def is_score_valid(
    score, min_num_bars: int, min_num_notes: int
) -> bool:
    """
    Check if a ``symusic.Score`` is valid, contains the minimum required number of bars.

    :param score: ``symusic.Score`` to inspect or path to a MIDI file.
    :param min_num_bars: minimum number of bars the score should contain.
    :param min_num_notes: minimum number of notes that score should contain.
    :return: boolean indicating if ``score`` is valid.
    """
    if isinstance(score, Path):
        try:
            score = Score(score)
        except SCORE_LOADING_EXCEPTION:
            return False
    elif isinstance(score, bytes):
        try:
            score = Score.from_midi(score)
        except SCORE_LOADING_EXCEPTION:
            return False

    return (
        score.start() >= 0
        and len(get_bars_ticks(score)) >= min_num_bars
        and score.note_num() > min_num_notes
    )

@hydra.main(config_path="configs", config_name="config")
def main(config):
    from argparse import Namespace
    from pytorch_lightning import Trainer
    from pytorch_lightning.utilities import rank_zero_info
    import torch
    import pytorch_lightning as pl

    config.trainer.devices = torch.cuda.device_count()
    args = Namespace(**OmegaConf.to_container(config.trainer, resolve=True))

    ########################################################################################################

    import os, warnings, datetime
    import numpy as np
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    import deepspeed
    from pytorch_lightning import seed_everything

    if config.random_seed >= 0:
        print(f"########## WARNING: GLOBAL SEED {config.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n" * 3)
        seed_everything(config.random_seed)

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")

    config.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    args.enable_checkpointing = False
    args.replace_sampler_ddp = False
    args.logger = False
    # Trainer.my_wandb does not yet exist on sanity validation
    args.num_sanity_val_steps = 0
    args.log_every_n_steps = int(1e20)
    config.betas = (config.training.beta1, config.training.beta2)
    config.real_bsz = int(args.num_nodes) * int(args.devices) * config.training.micro_bsz
    os.environ["RWKV_CTXLEN"] = str(config.model.ctx_len)
    os.environ["RWKV_HEAD_SIZE_A"] = str(config.model.head_size_a)
    os.environ["RWKV_CHUNK_LEN"] = str(config.model.chunk_len)
    if config.model.dim_att <= 0:
        config.model.dim_att = config.model.n_embd
    if config.model.dim_ffn <= 0:
        config.model.dim_ffn = int((config.model.n_embd * 3.5) // 32 * 32)

    config.run_name = f"MIDI-RWKV ctx{config.model.ctx_len} L{config.model.n_layer} D{config.model.n_embd}"
    if not os.path.exists(config.proj_dir):
        os.makedirs(config.proj_dir)

    rank_zero_info(
        f"""
############################################################################
#
# RWKV-7 {config.trainer.precision.upper()} on {config.trainer.num_nodes}x{config.trainer.devices} {config.trainer.accelerator.upper()}, bsz {config.trainer.num_nodes}x{config.trainer.devices}x{config.training.micro_bsz}={config.real_bsz}, {config.trainer.strategy} {'with grad_cp' if config.model.grad_cp > 0 else ''}
#
# ProjDir = {config.proj_dir}
#
# Epochs to {config.trainer.max_epochs}, save every {config.training.epoch_save} epochs
#
# Model = {config.model.n_layer} n_layer, {config.model.n_embd} n_embd, {config.model.ctx_len} ctx_len
#
# Adam = lr {config.training.lr_init} to {config.training.lr_final}, warmup {config.training.warmup_steps} steps, beta {config.betas}, eps {config.model.adam_eps}
#
# Found torch {torch.__version__}, recommend latest torch
# Found deepspeed {deepspeed.__version__}, recommend latest deepspeed
# Found pytorch_lightning {pl.__version__}, REQUIRE 1.9.5
#
############################################################################
"""
    )
    rank_zero_info(str(vars(args)) + "\n")

    if config.training.lr_final == 0 or config.training.lr_init == 0:
        rank_zero_info("\n\nNote: lr_final = 0 or lr_init = 0. Using linear LR schedule instead.\n\n")

    assert config.trainer.precision in ["tf32", "16", "bf16"]
    os.environ["RWKV_FLOAT_MODE"] = config.trainer.precision
    if config.trainer.precision == "16":
        rank_zero_info("\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n")

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if "32" in config.trainer.precision:
        config.trainer.precision = 32
    elif config.trainer.precision == "16":
        config.trainer.precision = 16
    else:
        config.trainer.precision = "bf16"

    os.environ["RWKV_SRC_DIR"] = config.data.src_dir

    ########################################################################################################

    from trainer import train_callback, generate_init_weight
    from dataset import MIDIDataset, DataCollatorNoneFilter
    from datasets import load_dataset, load_from_disk
    from miditok import MMM, TokenizerConfig
    from model import RWKV

    # omegaconf tuple stuff
    dc = config.data
    tokenizer = MMM(params=dc.tokenizer_path)  # MMM(TokenizerConfig(**deepcopy(TOKENIZER_PARAMS)))
    config.model.vocab_size = tokenizer.vocab_size
    dc.tracks_selection_random_ratio_range = (0.4, 1)
    dc.ratios_range_bar_infilling_duration = (0.1, 0.4)
    dc.acs_random_ratio_range = (0.05, 0.9)
    dc.tracks_idx_random_ratio_range = (0.1, 1)
    dc.bars_idx_random_ratio_range = (0.1, 0.7)
    dc.data_augmentation_offsets = (6, 2, 0) 

    if not os.path.exists(dc.prefiltered_dataset_path):
        ds = load_dataset("parquet", data_files={
            "train": dc.otherwise_train_data_path,
            "validation": dc.otherwise_val_data_path,
        })
        ds = ds.filter(
            lambda ex: is_score_valid(
                ex["music"], dc.min_num_bars_file_valid, dc.min_num_notes_file_valid
            )
        )
        
        # Save the filtered dataset
        ds.save_to_disk(dc.prefiltered_dataset_path)
        print("Filtered dataset saved to disk")
    else:
        ds = load_from_disk(dc.prefiltered_dataset_path)

    train_data = MIDIDataset(
                ds["train"],
                tokenizer,
                dc.max_seq_len,
                dc.tracks_selection_random_ratio_range,
                dc.data_augmentation_offsets,
                dc.ratio_bar_infilling,
                dc.ratios_range_bar_infilling_duration,
                ac_random_ratio_range=dc.acs_random_ratio_range,
                ac_tracks_random_ratio_range=dc.tracks_idx_random_ratio_range,
                ac_bars_random_ratio_range=dc.bars_idx_random_ratio_range)
    val_data = MIDIDataset(
                ds["validation"],
                tokenizer,
                dc.max_seq_len,
                dc.tracks_selection_random_ratio_range,
                dc.data_augmentation_offsets,
                dc.ratio_bar_infilling,
                dc.ratios_range_bar_infilling_duration,
                ac_random_ratio_range=dc.acs_random_ratio_range,
                ac_tracks_random_ratio_range=dc.tracks_idx_random_ratio_range,
                ac_bars_random_ratio_range=dc.bars_idx_random_ratio_range)
    model = RWKV(config)

    if len(config.load_model) == 0:  # shall we build the initial weights?
        config.load_model = f"{config.proj_dir}/rwkv-init.pth"
        generate_init_weight(model, config.load_model)  # save initial weights

    rank_zero_info(f"########## Loading {config.load_model}... ##########")
    try:
        load_dict = torch.load(config.load_model, map_location="cpu")
        load_keys = list(load_dict.keys())
        for k in load_keys:
            if k.startswith('_forward_module.'):
                load_dict[k.replace('_forward_module.','')] = load_dict[k]
                del load_dict[k]
    except:
        raise RuntimeError(f"Bad checkpoint {config.load_model}")

    state_file = f"{config.proj_dir}/rwkv-init-state.pth"
    if os.path.isfile(state_file):
        rank_zero_info(f"########## Loading State {state_file}... ##########")
        state_dict = torch.load(state_file, map_location="cpu")
        for k in state_dict:
            load_dict[k] = state_dict[k]
    model.load_state_dict(load_dict)

    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[train_callback(config)],
    )

    if trainer.global_rank == 0:
        for n in model.state_dict():
            shape = model.state_dict()[n].shape
            s0 = str(shape[0]) if len(shape) > 0 else ""
            s1 = str(shape[1]) if len(shape) > 1 else ""
            s2 = str(shape[2]) if len(shape) > 2 else ""
            s3 = str(shape[3]) if len(shape) > 3 else ""
            print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {s3.ljust(5)} {n}")

    if "deepspeed" in config.trainer.strategy:
        trainer.strategy.config["zero_optimization"]["allgather_bucket_size"] = config.training.ds_bucket_mb * 1000 * 1000
        trainer.strategy.config["zero_optimization"]["reduce_bucket_size"] = config.training.ds_bucket_mb * 1000 * 1000

    collator = DataCollatorNoneFilter(pad_token_id=tokenizer.pad_token_id, max_length=config.model.ctx_len)
    train_sampler = DistributedSampler(
        train_data,
        num_replicas=trainer.world_size,
        rank=trainer.global_rank,
        shuffle=False
    )
    val_sampler = DistributedSampler(
        val_data,
        num_replicas=trainer.world_size,
        rank=trainer.global_rank,
        shuffle=False
    )

    # must set shuffle=False, persistent_workers=False (because worker is in another thread)
    data_loader = DataLoader(train_data, sampler=train_sampler, pin_memory=True, batch_size=config.training.micro_bsz, num_workers=config.training.dataloader_num_workers, persistent_workers=False, drop_last=True, collate_fn=collator)
    val_loader = DataLoader(val_data, sampler=val_sampler, pin_memory=True, batch_size=config.training.micro_bsz, num_workers=config.training.dataloader_num_workers, persistent_workers=False, drop_last=True, collate_fn=collator)

    try:
        trainer.fit(model, data_loader, val_loader)
    finally:
        # should have had this a long time ago...
        to_save_dict = model.state_dict()
        torch.save(
            to_save_dict,
            f"{config.proj_dir}/rwkv-last.pth",
        )
        import requests
        requests.post(f"https://ntfy.sh/{config.ntfy}", data=f"Training run complete: {config.run_name}".encode(encoding='utf-8'),
        headers={
            "Title": "Training complete",
            "Priority": "urgent",
            "Tags": "warning"
        })


if __name__ == "__main__":
    main()