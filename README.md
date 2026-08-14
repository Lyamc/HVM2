Higher-order Virtual Machine 2 (HVM2)
=====================================

**Higher-order Virtual Machine 2 (HVM2)** is a massively parallel [Interaction
Combinator](https://www.semanticscholar.org/paper/Interaction-Combinators-Lafont/6cfe09aa6e5da6ce98077b7a048cb1badd78cc76)
evaluator.

By compiling programs from high-level languages (such as Python and Haskell) to
HVM, one can run these languages directly on massively parallel hardware, like
GPUs, with near-ideal speedup.

HVM2 is the successor to [HVM1](https://github.com/HigherOrderCO/HVM1), a 2022
prototype of this concept. Compared to its predecessor, HVM2 is simpler, faster
and, most importantly, more correct. [HOC](https://HigherOrderCO.com/) provides
long-term support for all features listed on its [PAPER](./paper/HVM2.pdf).

This repository provides a low-level IR language for specifying the HVM2 nets
and a compiler from that language to C and CUDA. It is not meant for direct
human usage. If you're looking for a high-level language to interface with HVM2,
check [Bend](https://github.com/HigherOrderCO/Bend) instead.

Usage
-----

First install the dependencies:
* If you want to use the C runtime, install a C11 compiler (GCC, Clang, or MSVC / Visual Studio 2022 with the Desktop C++ workload).
* If you want to use the CUDA runtime, install CUDA and nvcc (the CUDA compiler).
  - _HVM requires CUDA 12.x and currently only works on Nvidia GPUs._
  - On Windows, `CUDA_PATH` (or `CUDA_HOME`) should point at the toolkit; libraries are linked from `lib\x64`.

Install HVM2:

```sh
# this fork (Windows-capable):
cargo install --git https://github.com/Lyamc/HVM2 --locked
# or from a local clone:
cargo install --path .
```

`--release` (used by `cargo install`) is tuned for speed and size: `opt-level = 3`, fat LTO, `codegen-units = 1`, `strip = true`. Windows targets also get `/OPT:REF /OPT:ICF` (MSVC) or `--gc-sections` (GNU).

The C and CUDA backends are **auto-detected** at build time (`feature = "c"` / `feature = "cuda"`). Set `HVM_SKIP_C=1` or `HVM_SKIP_CUDA=1` to skip them.

### Windows

This fork builds the Rust interpreter and the C runtime on both **MSVC** (`x86_64-pc-windows-msvc`) and **GNU/MinGW** (`x86_64-pc-windows-gnu`).

#### MSVC (default `rustup` toolchain)

1. Install [Rust](https://rustup.rs/) (MSVC toolchain) and Visual Studio 2022 with **Desktop development with C++**.
2. Open a **Developer Command Prompt** or load `vcvars64` so `cl.exe` and `link.exe` are on `PATH`.
3. This repo's `.cargo/config.toml` pins the MSVC linker to `link.exe` (avoids rustc `lld-link` vs VS CRT `__guard_eh_cont_table`). Load `vcvars64` so `link.exe` is on `PATH`:

```bat
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cargo install --git https://github.com/Lyamc/HVM2 --locked
```

Keep `PATH` short before calling `vcvars64` if you see `The input line is too long`.

`build.rs` passes `/std:c11 /experimental:c11atomics` to `cl`. The C runtime heap-allocates the interaction-net buffers and typically needs **several gigabytes of RAM** (about 6 GiB plus ~128 MiB per worker thread).

Standalone C from `hvm gen-c file.hvm > file.c`:

```bat
cl /nologo /O2 /Gy /std:c11 /experimental:c11atomics file.c /Fe:file.exe
```

#### GNU / MinGW

```bat
rustup target add x86_64-pc-windows-gnu
:: gcc/ld from MSYS2 mingw64 or another MinGW-w64, on PATH
set CC=gcc
cargo install --git https://github.com/Lyamc/HVM2 --locked --target x86_64-pc-windows-gnu
```

Standalone C with gcc:

```bat
gcc -O3 -std=c11 -ffunction-sections -fdata-sections file.c -o file.exe -Wl,--gc-sections
```

There are multiple ways to run an HVM program:

```sh
hvm run    <file.hvm> # interpret via Rust
hvm run-c  <file.hvm> # interpret via C
hvm run-cu <file.hvm> # interpret via CUDA
hvm gen-c  <file.hvm> # compile to standalone C
hvm gen-cu <file.hvm> # compile to standalone CUDA
```

All modes produce the same output. The compiled modes require you to compile the
generated file (with `gcc file.c -o file` or the `cl` line above), but are faster to run.
The CUDA versions have much higher peak performance but are less stable. As a
rule of thumb, `gen-c` should be used in production.

Language
--------

HVM is a low-level compile target for high-level languages. It provides a raw
syntax for wiring interaction nets. For example:

```javascript
@main = a
  & @sum ~ (28 (0 a))

@sum = (?(((a a) @sum__C0) b) b)

@sum__C0 = ({c a} ({$([*2] $([+1] d)) $([*2] $([+0] b))} f))
  &! @sum ~ (a (b $([+] $(e f))))
  &! @sum ~ (c (d e))
```

The file above implements a recursive sum. If that looks unreadable to you -
don't worry, it isn't meant to. [Bend](https://github.com/HigherOrderCO/Bend) is
the human-readable language and should be used both by end users and by languages
aiming to target the HVM. If you're looking to learn more about the core
syntax and tech, though, please check the [PAPER](./paper/HVM2.pdf).
