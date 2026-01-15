import re

from max.graph.weights import WeightData


def convert_safetensor_state_dict(
    state_dict: dict[str, WeightData],
) -> dict[str, WeightData]:
    keys = list(state_dict.keys())
    for key in keys:
        # Remap net.2 to net.1: Diffusers uses [GELU, Dropout, Linear], while MAX uses [GELU, Linear].
        if re.match(
            r"transformer_blocks\.\d+\.(ff|ff_context)\.net\.2\.(weight|bias)",
            key,
        ):
            state_dict[key.replace("net.2.", "net.1.")] = state_dict.pop(key)
    return state_dict
