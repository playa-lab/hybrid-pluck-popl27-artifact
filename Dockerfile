# syntax=docker/dockerfile:1
#
# Artifact image for "Hybrid Pluck": the Hybrid Pluck engine, the five baseline
# systems of the evaluation (PSI, HyBit, Turing.jl, SlicStan, Siren), the
# benchmark harness and the analysis scripts. Builds for linux/amd64 and
# linux/arm64:
#
#   docker buildx build --platform linux/arm64 -t hybrid-pluck:popl-arm64 --load .
#   docker buildx build --platform linux/amd64 -t hybrid-pluck:popl-amd64 --load .
#
# External dependencies are pinned: the base image digest, toolchain versions
# and upstream commits in the ARGs below, and the resolved Julia and Python
# environments in experiments/docker/pins/. Ubuntu packages come from the 22.04
# archive at build time.

# ==== SlicStan (F# compiler from SlicStan to Stan), built in its own stage =======
# Only the published compiler is copied into the final image. SlicStan uses Z3
# through the Microsoft.Z3 .NET package, whose managed assembly is x64-only (in
# every published version); on arm64, Z3 and its .NET bindings are built from
# the same release instead. The patch fixes two SlicStan code-generation bugs
# (multi-dimensional and non-binary message tables).
FROM ubuntu:22.04@sha256:5ec03bb3441e8b0bf3b4f9cd4629a1ae763010dc3035bb8da3ae6cf026486401 AS slicstan-build
ARG TARGETARCH
ARG SLICSTAN_SHA=bbe12087d9c4286e1f54d7edb4dfee60782839c7
ARG Z3_VERSION=4.8.11
ARG DOTNET_SDK_VERSION=8.0.425
ENV DEBIAN_FRONTEND=noninteractive \
    DOTNET_ROOT=/opt/dotnet-sdk PATH=/opt/dotnet-sdk:$PATH \
    DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 DOTNET_ROLL_FORWARD=LatestMajor
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake curl git libicu70 python3 python3-distutils \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL https://dot.net/v1/dotnet-install.sh | bash -s -- --version "$DOTNET_SDK_VERSION" --install-dir /opt/dotnet-sdk
RUN mkdir -p /opt/z3 \
    && if [ "$TARGETARCH" = arm64 ]; then \
           git clone -q --depth 1 --branch "z3-$Z3_VERSION" https://github.com/Z3Prover/z3 /tmp/z3 \
           && cmake -S /tmp/z3 -B /tmp/z3/build -DCMAKE_BUILD_TYPE=Release \
                -DZ3_BUILD_DOTNET_BINDINGS=ON -DZ3_BUILD_PYTHON_BINDINGS=OFF -DZ3_BUILD_TEST_EXECUTABLES=OFF \
           && cmake --build /tmp/z3/build -j"$(nproc)" \
           && cp -P /tmp/z3/build/libz3.so* /opt/z3/ \
           && cp "$(find /tmp/z3/build -path '*netstandard2.0*' -name Microsoft.Z3.dll | head -1)" /opt/z3/ \
           && dotnet nuget remove source "Microsoft Z3 Local Repository" \
           && rm -rf /tmp/z3; \
       fi
COPY experiments/docker/slicstan-transformation.patch experiments/docker/SlicStan.fsproj /tmp/
COPY benchmark_translations/slicstan/hmm_first_order_n02.slicstan /tmp/
RUN git clone -q https://github.com/mgorinova/SlicStan /tmp/slicstan \
    && cd /tmp/slicstan && git checkout -q "$SLICSTAN_SHA" \
    && git apply -p1 /tmp/slicstan-transformation.patch \
    && cp /tmp/SlicStan.fsproj SlicStan/SlicStan.fsproj \
    && if [ -f /opt/z3/Microsoft.Z3.dll ]; then z3dll="-p:Z3Dll=/opt/z3/Microsoft.Z3.dll"; else z3dll=""; fi \
    && dotnet publish SlicStan -c Release -o /opt/slicstan $z3dll \
    && if [ -f /opt/z3/libz3.so ]; then cp -P /opt/z3/libz3.so* /opt/slicstan/; fi \
    && cd / && sed 's://.*$::' /tmp/hmm_first_order_n02.slicstan > /tmp/smoke.slic \
    && dotnet /opt/slicstan/SlicStan.dll --from-file /tmp/smoke.slic | grep -q '^parameters' \
    && echo "SlicStan OK"

# ==== Main image ===================================================================
FROM ubuntu:22.04@sha256:5ec03bb3441e8b0bf3b4f9cd4629a1ae763010dc3035bb8da3ae6cf026486401

ARG TARGETARCH
ARG RUST_VERSION=1.96.0
ARG JULIA_VERSION=1.8.5
ARG ALEA_SHA=cd9a09e75e74707b523fcfc9df9ef432ab0d475f
ARG CUDD_SHA=c98642fd796e17234923ab28f58fead0fc3fab18
ARG LDC_VERSION=1.32.0
ARG PSI_SHA=ef8a9e52c94089a0b4e547ac3ff99db10f3517dd
ARG UV_VERSION=0.7.16
ARG SLICSTAN_SHA=bbe12087d9c4286e1f54d7edb4dfee60782839c7
ARG DOTNET_RUNTIME_VERSION=8.0.31
ARG CMDSTAN_VERSION=2.31.0
ARG SIREN_SHA=d3ca59cf335c2d4596fa778a987e976e9741decb

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8

# build-essential: building Hybrid Pluck and PSI, and CmdStan compiles each Stan
# model at run time. libxml2: the LDC compiler that builds PSI. libicu70: .NET.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git xz-utils unzip bzip2 bc gawk \
        libxml2 libicu70 \
    && rm -rf /var/lib/apt/lists/*

# ---- Rust ---------------------------------------------------------------------
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --no-modify-path --profile minimal --default-toolchain "$RUST_VERSION"

# ---- uv (Python interpreters and environments) ------------------------------------
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python UV_CACHE_DIR=/tmp/uv-cache
RUN curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh

# ---- Julia (HyBit and Turing.jl) ---------------------------------------------
# The package sysimages are compiled for a generic CPU target, so the image runs
# on any x86-64 / ARMv8 machine rather than only on the build machine's CPU.
ENV JULIA_DEPOT_PATH=/opt/julia-depot
RUN case "$TARGETARCH" in \
        amd64) dir=x64; arch=x86_64; target='generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1)' ;; \
        arm64) dir=aarch64; arch=aarch64; target='generic' ;; \
        *) echo "unsupported architecture $TARGETARCH" >&2; exit 1 ;; \
    esac \
    && curl -fsSL "https://julialang-s3.julialang.org/bin/linux/$dir/${JULIA_VERSION%.*}/julia-$JULIA_VERSION-linux-$arch.tar.gz" \
       | tar -xz -C /opt \
    && mv "/opt/julia-$JULIA_VERSION" /opt/julia \
    && ln -s /opt/julia/bin/julia /usr/local/bin/julia \
    && echo "export JULIA_CPU_TARGET='$target'" > /opt/julia/cpu_target.sh

# The default environment holds PackageCompiler and Revise (the HyBit benchmark
# files start with `using Revise`).
COPY experiments/docker/pins/julia-default-Project.toml $JULIA_DEPOT_PATH/environments/v1.8/Project.toml
COPY experiments/docker/pins/julia-default-Manifest.toml $JULIA_DEPOT_PATH/environments/v1.8/Manifest.toml
RUN julia -e 'using Pkg; Pkg.instantiate()'

# HyBit is Alea.jl (the Julia package "Dice"), which has no committed Manifest;
# the pinned Project/Manifest drop Plots (an unused dependency that cannot be
# precompiled headless) and use a local CUDD.jl. CUDD.jl (v0.3.1) is patched to
# create its C callbacks statically: the closure callbacks it creates at load
# time are not supported by Julia 1.8 on aarch64.
#
# HyBit loads SymPy through PyCall, which by default installs the latest
# Miniconda Python. PyCall is built against a pinned Python 3.12 environment
# instead (embedding Python 3.14 in Julia 1.8 crashes under x86-64 emulation).
COPY experiments/docker/pins/pycall-requirements.txt /opt/
RUN uv venv -q /opt/pycall-venv --python 3.12 \
    && uv pip sync -q --python /opt/pycall-venv/bin/python /opt/pycall-requirements.txt \
    && rm -rf /tmp/uv-cache /opt/pycall-requirements.txt
COPY experiments/docker/dice_workload.jl experiments/docker/cudd-static-cfunction.patch /opt/
COPY experiments/docker/pins/alea-Project.toml experiments/docker/pins/alea-Manifest.toml /opt/
RUN . /opt/julia/cpu_target.sh \
    && git clone -q https://github.com/Tractables/Alea.jl /opt/Alea.jl \
    && git -C /opt/Alea.jl checkout -q "$ALEA_SHA" \
    && mv /opt/alea-Project.toml /opt/Alea.jl/Project.toml \
    && mv /opt/alea-Manifest.toml /opt/Alea.jl/Manifest.toml \
    && git clone -q https://github.com/sisl/CUDD.jl /opt/CUDD.jl \
    && git -C /opt/CUDD.jl checkout -q "$CUDD_SHA" \
    && git -C /opt/CUDD.jl apply /opt/cudd-static-cfunction.patch \
    && PYTHON=/opt/pycall-venv/bin/python julia --project=/opt/Alea.jl -e 'using Pkg; Pkg.instantiate(); Pkg.build()' \
    && julia --project=/opt/Alea.jl -e 'using Pkg; Pkg.test("CUDD")' \
    && grep -q /opt/pycall-venv /opt/julia-depot/packages/PyCall/*/deps/deps.jl \
    && julia --project=/opt/Alea.jl --startup-file=no -e 'using Revise, Dice; println("Dice loaded OK")' \
    && julia --project=/opt/Alea.jl -e 'using PackageCompiler; create_sysimage(["Dice"]; \
           sysimage_path="/opt/dice.so", precompile_execution_file="/opt/dice_workload.jl", \
           cpu_target=ENV["JULIA_CPU_TARGET"])' \
    && julia --sysimage /opt/dice.so --project=/opt/Alea.jl --startup-file=no -e 'using Dice; println("Dice sysimage OK")'

COPY benchmark_translations/turing/Project.toml benchmark_translations/turing/Manifest.toml \
     /artifact/benchmark_translations/turing/
COPY experiments/docker/turing_workload.jl /opt/turing_workload.jl
RUN . /opt/julia/cpu_target.sh \
    && cd /artifact/benchmark_translations/turing \
    && julia --project=. -e 'using Pkg; Pkg.instantiate()' \
    && julia --project=. --startup-file=no -e 'using Turing; @model f() = (x ~ Normal(0,1)); \
           sample(f(), PG(4), 5; progress=false); println("Turing loaded OK")' \
    && julia --project=. -e 'using PackageCompiler; create_sysimage(["Turing"]; \
           sysimage_path="/opt/turing.so", precompile_execution_file="/opt/turing_workload.jl", \
           cpu_target=ENV["JULIA_CPU_TARGET"])' \
    && julia --sysimage /opt/turing.so --project=. --startup-file=no -e 'using Turing; println("Turing sysimage OK")'

# ---- PSI ------------------------------------------------------------------------
# PSI's own dependency script only knows x86-64, so install the matching LDC
# release directly; build-release.sh then uses the ldmd2 on PATH.
RUN case "$TARGETARCH" in amd64) arch=x86_64 ;; arm64) arch=aarch64 ;; esac \
    && curl -fsSL "https://github.com/ldc-developers/ldc/releases/download/v$LDC_VERSION/ldc2-$LDC_VERSION-linux-$arch.tar.xz" \
       | tar -xJ -C /opt \
    && git clone -q https://github.com/eth-sri/psi /opt/psi \
    && cd /opt/psi && git checkout -q "$PSI_SHA" && git submodule update --init \
    && PATH="/opt/ldc2-$LDC_VERSION-linux-$arch/bin:$PATH" bash ./build-release.sh \
    && test -x /opt/psi/psi && ln -s /opt/psi/psi /usr/local/bin/psi \
    && rm -rf "/opt/ldc2-$LDC_VERSION-linux-$arch" \
    && printf 'def main(){ x := flip(1/3); return x; }\n' > /tmp/t.psi && psi /tmp/t.psi && rm /tmp/t.psi

# ---- Python (analysis scripts) --------------------------------------------------
COPY experiments/pyproject.toml experiments/uv.lock experiments/.python-version /artifact/experiments/
RUN cd /artifact/experiments && uv sync --frozen \
    && .venv/bin/python -c 'import matplotlib, numpy; print("analysis venv OK")' \
    && rm -rf /tmp/uv-cache

# ---- SlicStan (built in the slicstan-build stage above) ---------------------------
ENV DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1
COPY --from=slicstan-build /opt/slicstan /opt/slicstan
RUN curl -fsSL https://dot.net/v1/dotnet-install.sh | bash -s -- --runtime dotnet --version "$DOTNET_RUNTIME_VERSION" --install-dir /opt/dotnet \
    && printf '#!/usr/bin/env bash\nexport DOTNET_ROOT=/opt/dotnet\nexec /opt/dotnet/dotnet /opt/slicstan/SlicStan.dll "$@"\n' \
       > /usr/local/bin/slicstan && chmod +x /usr/local/bin/slicstan

# ---- CmdStan (Stan backend for SlicStan) -----------------------------------------
# The release ships an x86-64 stanc binary; on arm64 use stanc3's arm64 build.
RUN curl -fsSL "https://github.com/stan-dev/cmdstan/releases/download/v$CMDSTAN_VERSION/cmdstan-$CMDSTAN_VERSION.tar.gz" \
       | tar -xz -C /opt \
    && mv "/opt/cmdstan-$CMDSTAN_VERSION" /opt/cmdstan && cd /opt/cmdstan \
    && if [ "$TARGETARCH" = arm64 ]; then \
           curl -fsSL "https://github.com/stan-dev/stanc3/releases/download/v$CMDSTAN_VERSION/linux-arm64-stanc" -o bin/stanc \
           && chmod +x bin/stanc; \
       fi \
    && make -j"$(nproc)" build \
    && bin/stanc --version && test -x bin/stansummary \
    && cp examples/bernoulli/bernoulli.stan /tmp/b.stan && make /tmp/b && rm -f /tmp/b /tmp/b.*

# ---- Siren (semi-symbolic inference) ---------------------------------------------
COPY experiments/docker/pins/siren-requirements.txt /opt/siren/
RUN git clone -q https://github.com/psg-mit/siren /opt/siren/src \
    && git -C /opt/siren/src checkout -q "$SIREN_SHA" \
    && uv venv -q /opt/siren/venv --python 3.11 \
    && uv pip sync -q --python /opt/siren/venv/bin/python /opt/siren/siren-requirements.txt \
    && /opt/siren/venv/bin/siren --help > /dev/null && echo "Siren OK" \
    && rm -rf /tmp/uv-cache

# ---- Hybrid Pluck and the artifact ------------------------------------------------
WORKDIR /artifact
COPY . /artifact
RUN cargo build --release \
    && cargo test --release --no-run \
    && ln -s /artifact/target/release/pluck /usr/local/bin/pluck \
    && PLUCK_STATS=1 pluck examples/coinBias.pluck

# TERM: PSI crashes while printing an error message when TERM is unset (as it is
# in a container started without a terminal).
ENV ALEA_DIR=/opt/Alea.jl DICE_SYSIMAGE=/opt/dice.so TURING_SYSIMAGE=/opt/turing.so \
    TERM=dumb \
    CMDSTAN=/opt/cmdstan SLICSTAN_BIN=/usr/local/bin/slicstan SIREN_BIN=/opt/siren/venv/bin/siren \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    JULIA_NUM_THREADS=1 \
    PATH=/artifact/experiments:$PATH

LABEL org.opencontainers.image.title="Hybrid Pluck artifact" \
      org.opencontainers.image.licenses="MIT" \
      rust.version="$RUST_VERSION" julia.version="$JULIA_VERSION" alea.commit="$ALEA_SHA" \
      psi.commit="$PSI_SHA" slicstan.commit="$SLICSTAN_SHA" cmdstan.version="$CMDSTAN_VERSION" \
      siren.commit="$SIREN_SHA"

ENTRYPOINT ["/artifact/experiments/docker/entrypoint.sh"]
CMD ["bash"]
