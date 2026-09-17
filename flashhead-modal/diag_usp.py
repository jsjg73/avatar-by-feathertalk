"""CPU-only diagnostic: which xfuser/yunchang are in the image, what the USP
attention path imports, and whether upgrading yunchang fixes `update_npu_out`."""
import subprocess

import modal

from app import base_image  # noqa: E402

FA_WHEEL = ("https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
            "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")
image = base_image.pip_install(FA_WHEEL).add_local_python_source("app")
app = modal.App("flashhead-diag")


@app.function(image=image, timeout=600)
def diag() -> dict:
    def sh(cmd: str) -> str:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()[-1500:]
    out = {}
    out["versions"] = sh("pip show xfuser yunchang flash-attn 2>/dev/null | grep -E '^(Name|Version)'")
    out["xfuser_refs"] = sh("grep -rn 'update_npu_out' /usr/local/lib/python3.11/site-packages/xfuser | cut -c1-200 | head -8")
    out["yunchang_utils_defs"] = sh("grep -n '^def ' /usr/local/lib/python3.11/site-packages/yunchang/ring/utils.py | head -20")
    out["yunchang_versions_available"] = sh("pip index versions yunchang 2>/dev/null | head -2")
    # candidate pins: older xfuser (what FlashHead was developed against) + released yunchang
    test = ("python -c \"import xfuser, yunchang, importlib; from xfuser.core.long_ctx_attention import xFuserLongContextAttention; "
            "from yunchang.kernels import AttnType; import xfuser.core.long_ctx_attention.ring.ring_flash_attn as r; "
            "print('OK', xfuser.__version__ if hasattr(xfuser,'__version__') else '')\" 2>&1 | tail -1")
    for pin in ("xfuser==0.4.5", "xfuser==0.5.1", "xfuser==0.4.3"):
        inst = sh(f"pip install -q '{pin}' 2>&1 | grep -iE 'error|conflict' | head -2")
        vers = sh("pip show xfuser yunchang 2>/dev/null | grep -E '^Version' | tr '\n' ' '")
        refs = sh("grep -rl 'update_npu_out' /usr/local/lib/python3.11/site-packages/xfuser 2>/dev/null | wc -l")
        out[pin] = f"install: {inst or 'ok'} | versions(xfuser yunchang): {vers} | npu refs: {refs} | import: {sh(test)}"
    return out


@app.local_entrypoint()
def main():
    for k, v in diag.remote().items():
        print(f"== {k}\n{v}\n")
