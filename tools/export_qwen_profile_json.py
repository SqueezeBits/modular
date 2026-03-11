#!/usr/bin/env python3
"""Export a compact JSON summary from an nsys sqlite report."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def fetch_all(conn: sqlite3.Connection, query: str) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return list(conn.execute(query))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--woman-image", required=True)
    parser.add_argument("--man-image", required=True)
    parser.add_argument("--combined-image", required=True)
    parser.add_argument("--woman-prompt", required=True)
    parser.add_argument("--man-prompt", required=True)
    parser.add_argument("--edit-prompt", required=True)
    parser.add_argument("--edit-negative-prompt", required=True)
    args = parser.parse_args()

    sqlite_path = Path(args.sqlite)
    report_path = Path(args.report)
    output_path = Path(args.output)

    conn = sqlite3.connect(sqlite_path)
    conn.execute("PRAGMA temp_store=MEMORY")

    runtime_api_rows = fetch_all(
        conn,
        """
        SELECT
            s.value AS name,
            COUNT(*) AS calls,
            SUM(r.end - r.start) / 1000000.0 AS total_ms
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON s.id = r.nameId
        WHERE s.value IN (
            'cuLaunchKernelEx',
            'cuMemcpyHtoDAsync_v2',
            'cuStreamSynchronize'
        )
        GROUP BY s.value
        ORDER BY total_ms DESC
        """,
    )

    memcpy_rows = fetch_all(
        conn,
        """
        SELECT
            copyKind,
            COUNT(*) AS calls,
            SUM(end - start) / 1000000.0 AS total_ms,
            SUM(bytes) / 1024.0 / 1024.0 AS total_mib
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        GROUP BY copyKind
        ORDER BY total_ms DESC
        """,
    )

    kernel_rows = fetch_all(
        conn,
        """
        SELECT
            s.value AS name,
            COUNT(*) AS calls,
            SUM(k.end - k.start) / 1000000.0 AS total_ms,
            AVG(k.end - k.start) / 1000.0 AS avg_us
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.demangledName
        GROUP BY s.value
        ORDER BY total_ms DESC
        LIMIT 10
        """,
    )

    total_kernel_ms = fetch_all(
        conn,
        """
        SELECT SUM(end - start) / 1000000.0 AS total_ms
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        """,
    )[0]["total_ms"]

    runtime_api = {
        row["name"]: {
            "calls": int(row["calls"]),
            "total_ms": round(float(row["total_ms"]), 2),
        }
        for row in runtime_api_rows
    }

    memcpy_summary = {}
    for row in memcpy_rows:
        key = str(row["copyKind"])
        memcpy_summary[key] = {
            "calls": int(row["calls"]),
            "total_ms": round(float(row["total_ms"]), 2),
            "total_mib": round(float(row["total_mib"]), 2),
        }

    top_kernels = [
        {
            "name": row["name"],
            "calls": int(row["calls"]),
            "total_ms": round(float(row["total_ms"]), 2),
            "avg_us": round(float(row["avg_us"]), 2),
        }
        for row in kernel_rows
    ]

    payload = {
        "date_utc": args.date,
        "command": args.command,
        "artifacts": {
            "nsys_report": str(report_path),
            "nsys_sqlite": str(sqlite_path),
        },
        "runs": [
            {
                "name": "woman_t2i",
                "model": "Qwen/Qwen-Image-2512",
                "prompt": args.woman_prompt,
                "negative_prompt": "low quality",
                "width": 768,
                "height": 1024,
                "num_inference_steps": 30,
                "true_cfg_scale": 4.0,
                "seed": 0,
                "output_image": args.woman_image,
            },
            {
                "name": "man_t2i",
                "model": "Qwen/Qwen-Image-2512",
                "prompt": args.man_prompt,
                "negative_prompt": "low quality",
                "width": 768,
                "height": 1024,
                "num_inference_steps": 30,
                "true_cfg_scale": 4.0,
                "seed": 1,
                "output_image": args.man_image,
            },
            {
                "name": "combined_edit",
                "model": "Qwen/Qwen-Image-Edit-2511",
                "prompt": args.edit_prompt,
                "negative_prompt": args.edit_negative_prompt,
                "input_images": [args.woman_image, args.man_image],
                "width": 1536,
                "height": 1024,
                "num_inference_steps": 40,
                "guidance_scale": 1.0,
                "true_cfg_scale": 4.0,
                "seed": 0,
                "output_image": args.combined_image,
            },
        ],
        "summary": {
            "scope": "global",
            "scope_note": (
                "This report was captured from one combined nsys run, so the "
                "timing values below apply to the full woman/man/edit sequence."
            ),
            "total_kernel_ms": round(float(total_kernel_ms), 2),
            "runtime_api": runtime_api,
            "memcpy": memcpy_summary,
        },
        "top_kernels": top_kernels,
        "notes": [
            "copyKind 1 is HtoD, copyKind 2 is DtoH",
            "NVTX rows were not present in this sqlite export",
            "Current dominant costs are kernel launch count and HtoD uploads",
        ],
    }

    output_path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
