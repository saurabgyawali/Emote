# ct2 — CUDA CTranslate2 build root

Everything the CUDA CTranslate2 build needs or produces, kept inside `Emote/` so the
project is self-contained. Before 2026-09-20 this lived at `$HOME/ct2-cuda-build`
(and a copy sat next to `Emote/`); nothing outside the folder is referenced now.

| Path | What |
|---|---|
| `CTranslate2/` | CTranslate2 v4.5.0 source, `--recursive`, with an aarch64 `build/` tree compiled on the Jetson (v4.5.0, CUDA arch 87). ~11 GB, most of it `third_party/`. |
| `wheels/` | Built `ctranslate2-*.whl`, one per Python tag. Installing a cached wheel takes seconds instead of a 30–45 minute compile. |

Written and read only by `../transcriber/scripts/build_ctranslate2_cuda.sh`, which
defaults `--build-root` to this directory. Override with `--build-root DIR`, the
`BUILD_ROOT` env var, or `CT2_BUILD_ROOT` for `install.sh`.

The C++ library itself is installed system-wide to `/usr/local/lib/libctranslate2.so`
by `sudo make install`; this tree is the source it is built from, not where it runs from.

`CTranslate2/` is aarch64 build output and is excluded from `pack.sh` by default
(`--with-ct2-src` includes it). `wheels/` is always packed, so a fresh copy of `Emote/`
on the Jetson installs CUDA CTranslate2 without recompiling.

Both are gitignored. A wheel is an aarch64 + cp310 + CUDA binary that no other machine
can use, and it is rebuilt in minutes from the C++ library in `/usr/local`; it belongs on
the device that built it, not in git history. Keep the file, do not commit it.

Safe to delete if disk is needed: `install.sh` re-clones and recompiles. Keep `wheels/`.
