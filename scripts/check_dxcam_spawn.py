from __future__ import annotations

import multiprocessing as mp


def _worker() -> None:
    import dxcam

    camera = dxcam.create(output_idx=0, max_buffer_len=8, output_color="RGB")
    print(type(camera).__name__, flush=True)
    camera.stop()


def main() -> int:
    ctx = mp.get_context("spawn")
    process = ctx.Process(target=_worker, name="dxcam-spawn-check")
    process.start()
    process.join()
    print(f"exitcode={process.exitcode}", flush=True)
    return int(process.exitcode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
