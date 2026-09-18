# The renderer's environment, as something every host can build.
#
# This mirrors the image chain in app.py step for step. Modal can consume this
# file directly (`modal.Image.from_dockerfile`), and so can RunPod, vast.ai, a
# Lightning Studio or a box under a desk — which is the point: the environment
# stops being expressed only in one provider's DSL.
#
#   docker build -t flashhead .
#   docker run --gpus all -it --rm \
#     -v $PWD/weights:/weights -v $PWD/cache:/cache -v $PWD:/work -w /work \
#     flashhead python run_local.py --fetch-weights
#
# On a rented box you often cannot run Docker at all (you are already inside a
# container). Then use setup.sh, which runs these same steps against the host's
# Python — the two are kept in step deliberately.
#
# NOTE: the host's NVIDIA driver must support CUDA 12.4 (≥ 550). This is the
# most common failure when moving to a new provider, and it fails at import of
# torch, not at build.

FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    FLASHHEAD_SRC=/root/SoulX-FlashHead \
    FLASHHEAD_WEIGHTS=/weights \
    FLASHHEAD_CACHE=/cache \
    TORCHINDUCTOR_CACHE_DIR=/cache/inductor \
    TRITON_CACHE_DIR=/cache/triton \
    TORCHINDUCTOR_FX_GRAPH_CACHE=1 \
    HF_HOME=/cache/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 \
        python3.11 python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3

# index_url, not extra_index_url: on PyPI today plain `torch` resolves to a
# CUDA 13 build, whose nvidia-* stack then fights requirements.txt.
RUN pip install --no-cache-dir torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu124

RUN git clone --depth 1 https://github.com/Soul-AILab/SoulX-FlashHead.git ${FLASHHEAD_SRC}

# requirements.txt lines that are unused by the inference path or break
# resolution: xformers is never imported; nvidia-nccl-cu12 is pinned at 2.27.3,
# which conflicts with xformers and is only needed for multi-GPU (torch brings
# its own); decord, flask and gradio are data prep and the web demo.
RUN python - <<'EOF'
import os
drop = ("xformers", "nvidia-nccl-cu12", "decord", "flask", "gradio")
src = os.environ["FLASHHEAD_SRC"] + "/requirements.txt"
keep = [l for l in open(src) if l.strip() and not any(l.lower().startswith(d) for d in drop)]
open(os.environ["FLASHHEAD_SRC"] + "/requirements.portable.txt", "w").writelines(keep)
print("kept", len(keep), "requirement lines")
EOF
RUN pip install --no-cache-dir -r ${FLASHHEAD_SRC}/requirements.portable.txt

# Optional CUDA attention kernels: faster when present, and the model falls back
# to F.scaled_dot_product_attention when not, so this must never fail the build.
# The startup log says which kernel actually got picked.
RUN pip install --no-cache-dir flash_attn==2.8.0.post2 --no-build-isolation \
    || echo 'flash_attn unavailable, using SDPA fallback'

RUN pip install --no-cache-dir "huggingface_hub[cli]"

# T1: torch.compile. Upstream hardcodes it on; build with --build-arg COMPILE=True
# to keep it. See app.py for why it was turned off and why that reason no longer
# covers the live path.
ARG COMPILE=False
RUN sed -i "s/^COMPILE_MODEL = .*/COMPILE_MODEL = ${COMPILE}/; s/^COMPILE_VAE = .*/COMPILE_VAE = ${COMPILE}/" \
        ${FLASHHEAD_SRC}/flash_head/src/pipeline/flash_head_pipeline.py \
    && grep -q "^COMPILE_MODEL = ${COMPILE}" ${FLASHHEAD_SRC}/flash_head/src/pipeline/flash_head_pipeline.py \
    && grep -q "^COMPILE_VAE = ${COMPILE}" ${FLASHHEAD_SRC}/flash_head/src/pipeline/flash_head_pipeline.py \
    || (echo 'compile flags NOT patched' && exit 1)

# T2: drop the per-step timing instrumentation from generate(). It forces
# torch.cuda.synchronize() eight times per chunk — twice around every denoise
# step, then around decode, colour correction and the motion encode — purely so
# it can print each stage. On L4 a denoise step is ~105 ms and the stalls hide in
# it; on H100 a step is ~29 ms and they do not.
RUN sed -i '/torch\.cuda\.synchronize()/d; s/^\( *\)print(f.\[generate\].*/\1pass/' \
        ${FLASHHEAD_SRC}/flash_head/src/pipeline/flash_head_pipeline.py \
    && ! grep -q 'torch\.cuda\.synchronize()' ${FLASHHEAD_SRC}/flash_head/src/pipeline/flash_head_pipeline.py \
    && python -c "import ast; ast.parse(open('${FLASHHEAD_SRC}/flash_head/src/pipeline/flash_head_pipeline.py').read())" \
    || (echo 'sync instrumentation NOT stripped' && exit 1)

WORKDIR /work
COPY renderer.py run_local.py /work/
COPY inputs /work/inputs
ENV FLASHHEAD_WARMUP=/work/inputs/newscaster.png

CMD ["python", "run_local.py", "--help"]
