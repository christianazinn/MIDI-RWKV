"""Main classes implementations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InferenceConfig:
    """
    Configuration for the inference of MMM.

    Specifies which bars will be infilled and which tracks will be generated. It also
    specifies the list of attribute controls to control the generation. All tracks will
    be used to condition the generation.

    :param bars_to_generate: dictionary of couples [track_idx, [bar_start, bar_end, list
        of Attribute Controls]], where bar_start and bar_end are the extremes of the
        region to infill.
    :param new_tracks: list of tuple containing the programs and attribute controls for
        the new tracks
    """

    bars_to_generate: dict[int, list[tuple[int, int, list[str]]]] | None = None
    new_tracks: list[tuple[int, list[str]]] | None = None
    context_length: int = 4
    autoregressive: bool = False
    infilling: bool = False

    def __post_init__(self) -> None:
        """Check that the Inference config is consistent."""
        self.context_tracks = self.bars_to_generate.keys()

        if len(self.bars_to_generate) > 0:
            self.infilling = True

        # Set autoregressive flag
        if len(self.new_tracks) > 0:
            self.autoregressive = True

        # Valid program number
        for program, _ in self.new_tracks:
            if program < 0 or program > 127:
                msg = f"Invalid program number {program}> Must be in range [0,127]"
                raise ValueError(msg)

        if len(self.bars_to_generate) == 0 and len(self.new_tracks) == 0:
            msg = "You must provide either tracks to infill or new tracks to generate"
            raise ValueError(msg)
