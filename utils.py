### pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring,wrong-import-order
from typing import Tuple, List, Literal

import os
import torch

from pathlib import Path

###


class _Cuda():

    @staticmethod
    def is_available() -> bool:
        _cuda = torch.cuda.is_available()
        _cudnn = torch.backends.cudnn.enabled
        return _cuda and _cudnn

    @classmethod
    def device(cls) -> torch.cuda.device:
        assert cls.is_available()
        return torch.device('cuda')

    @classmethod
    def count_devices(cls) -> int:
        assert cls.is_available()
        return torch.cuda.device_count()

    @classmethod
    def get_current_device_info(cls) -> Tuple[int, str]:
        _idx = torch.cuda.current_device()
        _name = torch.cuda.get_device_name(_idx)
        return _idx, _name

    @staticmethod
    def get_visible_devices_param() -> str:
        return os.environ["CUDA_VISIBLE_DEVICES"]

    @classmethod
    def init(cls) -> torch.cuda.device:
        """
        We run all the experiments on server which have 4 different GPUs.
        Unfortunately, we cannot use all of them at the same time, since many other people are 
        using the server. Therefore, we have to specify which GPU we want to use.
        In particular, we have to use the GPU #1 (Nvidia RTX-3090).
        In order to avoid naive mistakes, we also check that the {CUDA_VISIBLE_DEVICES} environment 
        variable is set.
        """
        assert cls.is_available()
        assert isinstance(cls.get_visible_devices_param(), str)
        # assert cls.get_visible_devices_param() == "1"
        assert cls.count_devices() == 1

        device_idx, _ = cls.get_current_device_info()
        assert device_idx == 0

        return cls.device()


###


class _Prompt():

    ENCODING_CHAR: str = "_"

    @classmethod
    def encode(cls, prompt: str) -> str:
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        prompt = prompt.strip()
        prompt = prompt.replace(" ", cls.ENCODING_CHAR)
        return prompt

    @classmethod
    def decode(cls, prompt: str) -> str:
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        prompt = prompt.strip()
        prompt = prompt.replace(cls.ENCODING_CHAR, " ")
        return prompt

    @staticmethod
    def extract_from_file(filepath: Path) -> List[str]:
        assert isinstance(filepath, Path)
        assert filepath.exists()
        assert filepath.is_file()
        assert filepath.suffix == ".txt"

        with open(filepath, "r", encoding="utf-8") as f:
            prompts = f.readlines()

        prompts = map(lambda p: p.strip(), prompts)
        prompts = filter(lambda p: len(p) > 1, prompts)
        ### TODO: filter out prompts with special chars ...
        prompts = list(prompts)

        return prompts


###


class _Storage():

    @staticmethod
    def build_prompt_latents_filepath(
        out_rootpath: Path,
        prompt: str,
        assert_exists: bool,
    ) -> Path:
        assert "_" not in prompt

        filename = "latents.pt"
        prompt_dirname = Utils.Prompt.encode(prompt)

        out_prompt_path = out_rootpath.joinpath(prompt_dirname)
        out_prompt_ckpts_path = out_prompt_path.joinpath("ckpts")
        out_prompt_latents_filepath = out_prompt_ckpts_path.joinpath(filename)

        if assert_exists:
            assert out_prompt_latents_filepath.exists()
            assert out_prompt_latents_filepath.is_file()

        return out_prompt_latents_filepath

    @staticmethod
    def build_prompt_pointcloud_filepath(
        out_rootpath: Path,
        prompt: str,
        assert_exists: bool,
        idx: int,
    ) -> Path:
        assert "_" not in prompt
        assert isinstance(idx, int)
        assert idx >= 0

        filename = f"pointcloud_{idx}.npz"
        prompt_dirname = Utils.Prompt.encode(prompt)

        out_path = out_rootpath.joinpath(prompt_dirname, "pointclouds", filename)

        if assert_exists:
            assert out_path.exists()
            assert out_path.is_file()

        return out_path

    @staticmethod
    def build_prompt_mesh_filepath(
        out_rootpath: Path,
        prompt: str,
        assert_exists: bool,
        idx: int,
        extension: Literal["obj", "ply"],
    ) -> Path:
        assert "_" not in prompt
        assert isinstance(idx, int)
        assert idx >= 0
        assert isinstance(extension, str)
        assert extension in ["obj", "ply"]

        filename = f"model_{idx}.{extension}"
        prompt_dirname = Utils.Prompt.encode(prompt)

        out_path = out_rootpath.joinpath(prompt_dirname, "save", "export", filename)

        if assert_exists:
            assert out_path.exists()
            assert out_path.is_file()

        return out_path


###


class Utils():

    Cuda = _Cuda
    Prompt = _Prompt
    Storage = _Storage
