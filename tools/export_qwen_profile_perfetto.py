# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

#!/usr/bin/env python3
"""Export an nsys sqlite report as a Perfetto/Chrome trace-event JSON."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def fetch_rows(conn: sqlite3.Connection, query: str) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return list(conn.execute(query))


def ns_to_us(value: int) -> float:
    return value / 1000.0


def emit_complete_event(
    name: str,
    category: str,
    ts_ns: int,
    dur_ns: int,
    pid: int,
    tid: int,
    args: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "cat": category,
        "ph": "X",
        "ts": ns_to_us(ts_ns),
        "dur": ns_to_us(dur_ns),
        "pid": pid,
        "tid": tid,
        "args": args or {},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--kernel-limit",
        type=int,
        default=0,
        help="Optional cap on exported kernel events (0 means all).",
    )
    args = parser.parse_args()

    sqlite_path = Path(args.sqlite)
    output_path = Path(args.output)

    conn = sqlite3.connect(sqlite_path)
    conn.execute("PRAGMA temp_store=MEMORY")

    events: list[dict[str, object]] = []

    runtime_rows = fetch_rows(
        conn,
        """
        SELECT
            r.start,
            r.end,
            r.globalTid,
            s.value AS name
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON s.id = r.nameId
        WHERE s.value IN (
            'cuLaunchKernelEx',
            'cuMemcpyHtoDAsync_v2',
            'cuMemcpyDtoHAsync_v2',
            'cuStreamSynchronize'
        )
        ORDER BY r.start
        """,
    )
    for row in runtime_rows:
        events.append(
            emit_complete_event(
                name=row["name"],
                category="cuda_runtime",
                ts_ns=int(row["start"]),
                dur_ns=int(row["end"]) - int(row["start"]),
                pid=1,
                tid=int(row["globalTid"] or 0),
            )
        )

    memcpy_rows = fetch_rows(
        conn,
        """
        SELECT
            start,
            end,
            streamId,
            bytes,
            copyKind
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        ORDER BY start
        """,
    )
    for row in memcpy_rows:
        copy_kind = int(row["copyKind"])
        copy_name = {
            1: "Memcpy HtoD",
            2: "Memcpy DtoH",
        }.get(copy_kind, f"Memcpy kind={copy_kind}")
        events.append(
            emit_complete_event(
                name=copy_name,
                category="cuda_memcpy",
                ts_ns=int(row["start"]),
                dur_ns=int(row["end"]) - int(row["start"]),
                pid=2,
                tid=int(row["streamId"]),
                args={"bytes": int(row["bytes"]), "copyKind": copy_kind},
            )
        )

    kernel_query = """
        SELECT
            k.start,
            k.end,
            k.streamId,
            s.value AS name,
            k.gridX,
            k.gridY,
            k.gridZ,
            k.blockX,
            k.blockY,
            k.blockZ
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.demangledName
        ORDER BY k.start
    """
    if args.kernel_limit > 0:
        kernel_query += f" LIMIT {args.kernel_limit}"
    kernel_rows = fetch_rows(conn, kernel_query)
    for row in kernel_rows:
        events.append(
            emit_complete_event(
                name=row["name"],
                category="cuda_kernel",
                ts_ns=int(row["start"]),
                dur_ns=int(row["end"]) - int(row["start"]),
                pid=3,
                tid=int(row["streamId"]),
                args={
                    "grid": [
                        int(row["gridX"]),
                        int(row["gridY"]),
                        int(row["gridZ"]),
                    ],
                    "block": [
                        int(row["blockX"]),
                        int(row["blockY"]),
                        int(row["blockZ"]),
                    ],
                },
            )
        )

    events.sort(key=lambda event: float(event["ts"]))

    payload = {
        "traceEvents": events,
        "displayTimeUnit": "ms",
    }
    output_path.write_text(json.dumps(payload))


if __name__ == "__main__":
    main()
