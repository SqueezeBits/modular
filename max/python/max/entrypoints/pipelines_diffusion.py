# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Diffusion-only CLI wrapper.

This exists so Bazel can keep `//max/python/max/entrypoints:pipelines` lean,
while allowing `//max/python/max/entrypoints:pipelines_diffusion` to pull in
extra runtime deps.
"""

from __future__ import annotations

import sys


def main() -> None:
    # Import the main pipelines CLI and dispatch into the `diffusion` group.
    #
    # NOTE: `max.entrypoints.pipelines.main` is a click command object. Calling it
    # with `args=[...]` is equivalent to invoking the CLI with those argv tokens.
    import max.entrypoints.pipelines as pipelines_cli

    pipelines_cli.main(
        prog_name="pipelines",
        args=[*sys.argv[1:]],
    )


if __name__ == "__main__":
    main()
