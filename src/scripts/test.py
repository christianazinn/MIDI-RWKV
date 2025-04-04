import hydra

#######################################################################

@hydra.main(config_path="configs", config_name="config")
def main(config):
    from dataset import MIDIDataset, DataCollatorNoneFilter
    from datasets import load_dataset, load_from_disk
    from miditok import MMM, TokenizerConfig, TokSequence

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

    ds = load_from_disk(dc.prefiltered_dataset_path)
    import random
    # random.seed(43)

    # 548, 556, 618, 629, 631, 634, 636, 638 

    for use_old_tokenize_score in (False, True):
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
                    ac_bars_random_ratio_range=dc.bars_idx_random_ratio_range,
                    use_old_tokenize_score=use_old_tokenize_score)
        #print(train_data[0]["labels"].cpu().tolist())

        qvl = TokSequence(ids=train_data[0]["input_ids"].cpu().tolist(), are_ids_encoded=True)
        tokenizer.decode_token_ids(qvl)
        with open(f"/home/christian/MIDI-RWKV/src/test{'_old' if use_old_tokenize_score else ''}.txt", "w") as f:
            f.write(str(qvl.tokens))
if __name__ == "__main__":
    main()