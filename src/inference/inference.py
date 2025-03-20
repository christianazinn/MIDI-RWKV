"""Inference method for the MMM model."""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING

import numpy as np
from miditok import MMM, TokSequence
from symusic import Score
from torch import LongTensor
from transformers import LogitsProcessorList

from logits_processor import StopLogitsProcessor

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

from config import InferenceConfig


def generate(
    model: object,
    tokenizer: MMM,
    inference_config: InferenceConfig,
    score_or_path: Score | Path | str,
    generate_kwargs: Mapping | None = None,
) -> Score:
    """
    Use the model to generate new music content.

    The method allows to infill specific bars or generate new tracks.

    :param model: model used for generation
    :param tokenizer: MMM tokenizer
    :param inference_config: InferenceConfig
    :param score_or_path: ``symusic.Score`` or path of the music file to infill.
    :param generate_kwargs: keyword arguments to provide to the ``model.generate``
        method. For Hugging Face models for example, you can provide a
        ``GenerationConfig`` using this argument.
    :return: the infilled ``symusic.Score`` object.
    """
    score = (
        Score(score_or_path) if not isinstance(score_or_path, Score) else score_or_path
    )

    logits_processor = StopLogitsProcessor(
        tokenizer.vocab["Bar_None"], tokenizer.vocab["FillBar_End"], tokenizer
    )

    # Infill bars
    if inference_config.infilling:
        score = generate_infilling(
            model, tokenizer, inference_config, score, logits_processor, generate_kwargs
        )

    # Generate new tracks
    if inference_config.autoregressive:
        for track in inference_config.new_tracks:
            score = generate_new_track(model, tokenizer, track, score, generate_kwargs)

    return score


def generate_new_track(
    model: object,
    tokenizer: MMM,
    track: tuple[int, list[str]],
    score: Score,
    generate_kwargs: Mapping | None = None,
) -> Score:
    """
    Generate a new track of a given Score.

    The new track will be added to the score.

    :param model: model used for generation
    :param tokenizer: MMM tokenizer
    :param track: tuple containing the program of the track and a list of Track
        Attribute Controls.
    :param score: symusic.Score
    :param generate_kwargs: keyword arguments to provide to the ``model.generate``
        method. For Hugging Face models for example, you can provide a
        ``GenerationConfig`` using this argument.
    :return: the infilled ``symusic.Score`` object.
    """
    if not generate_kwargs:
        generate_kwargs = {}

    # In this case, the prompt is a toksequence containing all the tracks
    input_seq = tokenizer.encode(score)

    # Add <TRACK_START> and <PROGRAM> tokens
    input_seq.ids.append(tokenizer.vocab["Track_Start"])
    input_seq.tokens.append("Track_Start")
    input_seq.ids.append(tokenizer.vocab[f"Program_{track[0]}"])
    input_seq.tokens.append(f"Program_{track[0]}")

    # Add attribute control tokens
    for control in track[1]:
        input_seq.ids.append(tokenizer.vocab[control])
        input_seq.tokens.append(control)

    # TODO CUDA
    output_ids = model.generate(LongTensor([input_seq.ids]).to("cuda"), **generate_kwargs)
    output_seq = TokSequence(ids=output_ids[0].tolist(), are_ids_encoded=True)

    # Remove attribute controls from the sequence
    output_seq = (
        output_seq[: len(input_seq)] + output_seq[len(input_seq) + len(track[1]) :]
    )

    # Decode BPE ids before getting the associated tokens
    tokenizer.decode_token_ids(output_seq)
    output_seq.tokens = tokenizer._ids_to_tokens(output_seq.ids)

    # It is expected to have a <TRACK_END> token at the end of the sequence.
    if output_seq.tokens[-1] != "Track_End":
        warnings.warn(
            "Track generation failed: the model failed to predict a <TRACK_END> token",
            stacklevel=2,
        )
        output_seq.ids.append(tokenizer.vocab["Track_End"])
        output_seq.tokens.append("Track_End")

    return tokenizer._tokens_to_score(output_seq)


def generate_infilling(
    model: object,
    tokenizer: MMM,
    inference_config: InferenceConfig,
    score: Score,
    logits_processor: StopLogitsProcessor | None = None,
    generate_kwargs: Mapping | None = None,
) -> Score:
    """
    Generate a new portion of a ``symusic.Score``.

    The portion to infill will be generated with the model and added to the score
    inplace for the selected tracks. Notes originally present in the portion to
    infill will be removed.

    :param model: model used for generation
    :param tokenizer: MMM tokenizer
    :param score: ``symusic.Score`` to generate a new track from.
    :param inference_config: InferenceConfig
    :param logits_processor: ``transformers.LogitsProcessor`` used to stop
        generation when the right number of bars is generated.
    :param generate_kwargs: keyword arguments to provide to the ``model.generate``
        method. For Hugging Face models for example, you can provide a
        ``GenerationConfig`` using this argument.
    :return: the infilled ``symusic.Score`` object.
    """
    if not generate_kwargs:
        generate_kwargs = {}
    else:
        generate_kwargs["generation_config"].eos_token_id = tokenizer.vocab[
            "FillBar_End"
        ]

    tracks_to_infill = inference_config.bars_to_generate.keys()

    start_time = time.time()
    input_tokens = tokenizer.encode(score, concatenate_track_sequences=False)

    end_time = time.time()
    print(
        "[INFO::generate_infilling] Time spent for converting score to tokens: ",
        end_time - start_time,
    )

    for track_to_infill in tracks_to_infill:
        infill_bars(
            model,
            tokenizer,
            track_to_infill,
            inference_config,
            input_tokens,
            logits_processor,
            generate_kwargs,
        )

    # Here we use the base tokenizer because output_tokens is a list of TokSequences

    start_time = time.time()
    result = tokenizer.base_tokenizer._tokens_to_score(input_tokens)
    end_time = time.time()
    print(
        "[INFO::generate_infilling] Time spent for converting tokens to score: ",
        end_time - start_time,
    )
    return result


def infill_bars(
    model: object,
    tokenizer: MMM,
    track_idx: int,
    inference_config: InferenceConfig,
    tokens: list[TokSequence],
    logits_processor: StopLogitsProcessor | None = None,
    generate_kwargs: Mapping | None = None,
) -> None:
    """
    Infill bars for the ''track_idx'' track.

    The tokens are replaced inplace.

    :param model: model used for generation
    :param tokenizer: MMM tokenizer
    :param track_idx: index of the track to infill
    :param inference_config: contains information about which tracks and bars to
        generate.
    :param tokens: TokSequence of the track to be infilled
    :param logits_processor: ``transformers.LogitsProcessor`` used to stop generation
        when the right number of bars is generated.
    :param generate_kwargs: keyword arguments to provide to the ``model.generate``
        method. For Hugging Face models for example, you can provide a
        ``GenerationConfig`` using this argument.
    """
    if not generate_kwargs:
        generate_kwargs = {}

    # For each set of bars to infill in the track, we generate new content
    # (We may have, in the same track, non-adjacent sequences of bars. For
    # each sequence, we do a generation step).
    for subset_bars_to_infill in inference_config.bars_to_generate[track_idx]:
        # token_start_idx and token_end_idx are the indices of start
        # and end of infilling, when the toksequence is NOT BPE encoded
        start_time = time.time()

        input_seq, token_start_idx, token_end_idx = _adapt_prompt_for_bar_infilling(
            tokenizer, track_idx, tokens, subset_bars_to_infill
        )

        end_time = time.time()
        print(
            "[INFO::infill_bars] Time spent for creating input sequence: ",
            end_time - start_time,
        )

        logits_processor.n_bars_to_infill = (
            subset_bars_to_infill[1] - subset_bars_to_infill[0]
        )
        logits_processor.n_attribute_controls = len(subset_bars_to_infill[2])
        logit_processor_list = LogitsProcessorList()
        logit_processor_list.append(logits_processor)

        start_time = time.time()

        output_ids = model.generate(
            LongTensor([input_seq.ids]).to("cuda"),
            logits_processor=logit_processor_list,
            **generate_kwargs,
        )[0].cpu().numpy()

        end_time = time.time()
        print("[INFO::infill_bars] Time spent for generation: ", end_time - start_time)
        # print("Time spent in logits processor ", logits_processor.total_time)

        start_time = time.time()

        fill_start_idx = np.where(output_ids == tokenizer.vocab["FillBar_Start"])[0][0]

        # Here we isolate the generated tokens doing some filtering. In particular,
        # the model may generate some tokens before the first Bar_None token
        generated_tokens = TokSequence(are_ids_encoded=True)
        generated_tokens.ids = output_ids[
            fill_start_idx + len(subset_bars_to_infill[2]) + 1 : -1
        ].tolist()
        # decode_token_ids doesn't support numpy arrays for ids list
        tokenizer.decode_token_ids(generated_tokens)
        bar_none_token_idxs = np.where(
            np.array(generated_tokens.ids) == tokenizer.vocab["Bar_None"]
        )[0]
        # bar_none_token_idxs[-1] because we must exclude the last BarNone token,
        # which is used by the logits processor to stop generation
        print(generated_tokens.ids)
        if len(bar_none_token_idxs > 0):
            generated_tokens.ids = generated_tokens.ids[
                bar_none_token_idxs[0] : bar_none_token_idxs[-1]
            ]

        tokenizer.decode_token_ids(tokens[track_idx])
        tokens[track_idx].ids[token_start_idx:token_end_idx] = generated_tokens.ids
        tokens[track_idx].tokens = tokenizer._ids_to_tokens(tokens[track_idx].ids)

        end_time = time.time()
        print(
            "[INFO::infill_bars] Time spend for reconstructing the sequence: ",
            end_time - start_time,
        )


def _adapt_prompt_for_bar_infilling(
    tokenizer: MMM,
    track_idx: int,
    tokens: list[TokSequence],
    subset_bars_to_infill: tuple[int, int, list[str]],
    num_context_bars: int = 4,
) -> TokSequence:
    """
    Construct the prompt for bar infilling.

    Constructs the prompt to be used as model's input. The sequence should have the
    "BAR_FILL" format:
    ``<TRACK_START>...<TRACK_END>...<TRACKS_START>...<INFILL_BAR>...<INFILL_BAR>...
    <TRACK_END>...<TRACK_START>...<TRACK_END><START_FILL>``
    We have as many <FILL_IN> tokens as the number of bars we want to infill.

    :param tokenizer: MMM tokenizer
    :param track_idx: index of the track to infill
    :param tokens: TokSequence of the track to be infilled
    :param subset_bars_to_infill: contains the indexes of the first and last bar to
        infill, plus a list of attribute controls
    """
    conditioning_dict = {}

    toksequence_to_infill: TokSequence = TokSequence(are_ids_encoded=False)

    start_bar_idx = subset_bars_to_infill[0]
    end_bar_idx = subset_bars_to_infill[1]

    bars_ticks = tokens[track_idx]._ticks_bars
    bar_tick_start = bars_ticks[start_bar_idx]
    bar_tick_end = bars_ticks[end_bar_idx]

    times = np.array([event.time for event in tokens[track_idx].events])

    token_idx_start = np.nonzero(times >= bar_tick_start)[0]
    token_idx_start = token_idx_start[0]

    token_idx_end = np.nonzero(times >= bar_tick_end)[0]
    token_idx_end = token_idx_end[0]

    # Context
    context_token_start_idx = np.nonzero(
        times >= bars_ticks[start_bar_idx - num_context_bars]
    )[0][0]
    context_token_end_idx = np.nonzero(
        times >= bars_ticks[end_bar_idx + num_context_bars]
    )[0][0]

    conditioning_dict[track_idx] = (context_token_start_idx, context_token_end_idx)

    # Decode BPE tokens: this is necessary to put <INFILL_BAR> tokens
    # at the right place
    tokenizer.decode_token_ids(tokens[track_idx])

    seq_before = (
        tokens[track_idx][:2]
        + tokens[track_idx][context_token_start_idx:token_idx_start]
    )
    for _ in range(end_bar_idx - start_bar_idx):
        seq_before.ids.append(tokenizer.vocab["Infill_Bar"])
        seq_before.tokens.append("Infill_Bar")
    seq_after = tokens[track_idx][token_idx_end:context_token_end_idx]
    toksequence_to_infill += seq_before + seq_after
    toksequence_to_infill.ids.append(tokenizer.vocab["Track_End"])
    toksequence_to_infill.tokens.append("Track_End")

    # Encode into BPE tokens
    tokenizer.encode_token_ids(toksequence_to_infill)

    output_toksequence = TokSequence(are_ids_encoded=True)
    for i in range(len(tokens)):
        if i == track_idx:
            output_toksequence += toksequence_to_infill
            continue
        times = np.array([event.time for event in tokens[i].events])
        try:
            context_token_start_idx = np.nonzero(
                times >= bars_ticks[start_bar_idx - num_context_bars]
            )[0][0]
            context_token_end_idx = np.nonzero(
                times >= bars_ticks[end_bar_idx + num_context_bars]
            )[0][0]
        except IndexError:
            continue
        conditioning_dict[i] = (context_token_start_idx, context_token_end_idx)
        output_toksequence += (
            tokens[i][:2]
            + tokens[i][context_token_start_idx:context_token_end_idx]
            + tokens[i][-1:]
        )

    output_toksequence.ids.append(tokenizer.vocab["FillBar_Start"])
    output_toksequence.tokens.append("FillBar_Start")

    attribute_controls = subset_bars_to_infill[2]
    for control in attribute_controls:
        output_toksequence.ids.append(tokenizer.vocab[control])
        output_toksequence.tokens.append(control)

    # with open("tokens.txt", "w") as file:
    #    for token in output_toksequence.tokens:
    #        file.write(token + "\n")

    return output_toksequence, token_idx_start, token_idx_end


if __name__ == "__main__":
    import fla
    from transformers import GenerationConfig, AutoModelForCausalLM
    from symusic import Synthesizer, dump_wav
    INFERENCE_CONFIG = InferenceConfig(
        {
            2: [(12, 14, ["ACBarNoteDensity_6", "ACBarNoteDurationEight_1"])],
        },
        [
        #     (43, ["ACBarPitchClass_3"]),
        ],
    )
    gen_config = GenerationConfig(
            num_beams=1,
            temperature=1.2,
            repetition_penalty=1.2,
            top_k=20,
            top_p=0.95,
            max_new_tokens=300,
        )
    
    TOK_PATH = "/home/christian/MIDI-RWKV/src/tokenizer/tokenizer_with_acs.json"
    MODEL_PATH = "../outputs/outdir"
    INPUT_PATH = "/home/christian/MIDI-RWKV/src/inference/mat/rollinggirlEDIT.mid"
    OUTPUT_PATH = "/home/christian/MIDI-RWKV/src/inference/mat/output.mid"
    OUTWAV_PATH = "/home/christian/MIDI-RWKV/src/inference/mat/output.wav"
    INWAV_PATH = "/home/christian/MIDI-RWKV/src/inference/mat/input.wav"
    OUTPR_PATH = "/home/christian/MIDI-RWKV/src/inference/mat/output.png"
    INPR_PATH = "/home/christian/MIDI-RWKV/src/inference/mat/input.png"
    
    tokenizer = MMM(params=TOK_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to("cuda")

    print("Generating...")

    output_scores = generate(
            model,
            tokenizer,
            INFERENCE_CONFIG,
            INPUT_PATH,
            {"generation_config": gen_config},
        )
    
    print("Done generating. Dumping MIDI...")

    # Create MIDI output
    output_scores.dump_midi(OUTPUT_PATH)
    print("Dumped MIDI, synthesizing...")

    # synth
    synth = Synthesizer()
    outscore = Score(OUTPUT_PATH)
    inscore = Score(INPUT_PATH)
    outwav = synth.render(outscore, stereo=True)
    inwav = synth.render(inscore, stereo=True)
    dump_wav(OUTWAV_PATH, outwav, sample_rate=44100, use_int16=True)
    dump_wav(INWAV_PATH, inwav, sample_rate=44100, use_int16=True)

    print("Synthesized, plotting piano rolls...")

    # matplotlib
    from matplotlib import pyplot as plt
    intrack = inscore.resample(tpq=6, min_dur=1).tracks[2].pianoroll(modes=["onset", "frame"], pitch_range=[0, 128], encode_velocity=False)
    outtrack = outscore.resample(tpq=6, min_dur=1).tracks[2].pianoroll(modes=["onset", "frame"], pitch_range=[0, 128], encode_velocity=False)

    intrack_truncated = [intrack[0][:, :500], intrack[1][:, :500]]
    outtrack_truncated = [outtrack[0][:, :500], outtrack[1][:, :500]]
    plt.imshow(intrack_truncated[0] + intrack_truncated[1], aspect="auto", origin="lower")
    plt.savefig(INPR_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    plt.imshow(outtrack_truncated[0] + outtrack_truncated[1], aspect="auto", origin="lower")
    plt.savefig(OUTPR_PATH, dpi=300, bbox_inches="tight")
    

