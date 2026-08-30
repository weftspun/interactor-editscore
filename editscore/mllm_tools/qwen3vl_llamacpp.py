import os
import subprocess
import tempfile
from typing import Optional

from PIL import Image


class Qwen3VLLlamaCpp:
    """Qwen3-VL through llama-mtmd-cli, so the vision half can be a Hailo HEF.

    mmproj takes either a GGUF projector or an encoder .hef; the HEF path requires a build
    with -DLLAMA_HAILO=ON and replaces ViT + projector entirely.
    """

    def __init__(
        self,
        vlm_model,
        temperature: float = 0.7,
        seed: Optional[int] = None,
        lora_path: Optional[str] = None,
        cli: Optional[str] = None,
        mmproj: Optional[str] = None,
        n_predict: int = 512,
    ) -> None:
        self.cli = cli or os.environ.get("LLAMA_MTMD_CLI", "llama-mtmd-cli")
        self.model = vlm_model
        self.mmproj = mmproj or os.environ.get("LLAMA_MMPROJ")
        self.lora_path = lora_path
        self.temperature = temperature
        self.seed = seed
        self.n_predict = n_predict
        if self.mmproj is None:
            raise ValueError("Qwen3VLLlamaCpp needs mmproj: a projector GGUF or an encoder .hef")

    def prepare_input(self, images, text_prompt: str = ""):
        if not isinstance(images, list):
            images = [images]
        return {"images": images, "prompt": text_prompt}

    def _materialise(self, images, tmpdir):
        paths = []
        for i, im in enumerate(images):
            if isinstance(im, str):
                paths.append(im)
                continue
            p = os.path.join(tmpdir, "img%02d.png" % i)
            (im if isinstance(im, Image.Image) else Image.fromarray(im)).save(p)
            paths.append(p)
        return paths

    def inference(self, inputs, seed: Optional[int] = None):
        seed = self.seed if seed is None else seed
        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [self.cli, "-m", self.model, "--mmproj", self.mmproj,
                   "-p", inputs["prompt"], "-n", str(self.n_predict),
                   "--temp", str(self.temperature), "--top-p", "0.9", "--top-k", "20"]
            if seed is not None:
                cmd += ["--seed", str(seed)]
            if self.lora_path:
                cmd += ["--lora", self.lora_path]
            for p in self._materialise(inputs["images"], tmpdir):
                cmd += ["--image", p]

            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError("llama-mtmd-cli failed (%d): %s"
                                   % (r.returncode, r.stderr.strip()[-2000:]))
        return r.stdout.strip()
