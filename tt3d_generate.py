### pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring,wrong-import-order
from typing import Tuple, Any

import argparse
import torch
from tqdm.auto import tqdm

from pathlib import Path
from point_e.diffusion.configs import DIFFUSION_CONFIGS
from point_e.diffusion.configs import diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS
from point_e.models.configs import model_from_config
# from point_e.util.plotting import plot_point_cloud
# from point_e.util.point_cloud import PointCloud

from utils import Utils

###

device = Utils.Cuda.init()

###


def __load_support_models() -> Tuple[Any, Any, Any, Any]:
    print('creating base model...')
    base_name = 'base40M-textvec'
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    #
    base_diffusion_model = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
    #
    print('creating upsample model...')
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    #
    upsampler_diffusion_model = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])
    #
    print('downloading base checkpoint...')
    base_model.load_state_dict(load_checkpoint(base_name, device))
    print('downloading upsampler checkpoint...')
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))
    #
    return base_model, upsampler_model, base_diffusion_model, upsampler_diffusion_model


###


def build_sampler() -> PointCloudSampler:
    base_model, upsampler_model, base_diffusion_model, upsampler_diffusion_model = __load_support_models()

    assert base_model is not None
    assert upsampler_model is not None
    assert base_diffusion_model is not None
    assert upsampler_diffusion_model is not None

    return PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion_model, upsampler_diffusion_model],
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 0.0],
        model_kwargs_key_filter=('texts', ''),  # Do not condition the upsampler at all
    )


def _generate_latents(
    prompt: str,
    out_rootpath: Path,
    sampler: PointCloudSampler,
    skip_existing: bool,
    # batch_size: int = 1,
) -> None:
    assert isinstance(prompt, str)
    assert len(prompt) > 1
    assert isinstance(sampler, PointCloudSampler)
    # assert isinstance(batch_size, int)
    # assert 1 <= batch_size <= 1000  ### avoid naive mistakes ...

    batch_size = 1

    out_prompt_latents_filepath = Utils.Storage.build_prompt_latents_filepath(
        out_rootpath=out_rootpath,
        prompt=prompt,
        assert_exists=False,
    )

    if skip_existing and out_prompt_latents_filepath.exists():
        print("")
        print("latents already exists -> ", out_prompt_latents_filepath)
        print("")
        return

    out_prompt_latents_filepath.parent.mkdir(parents=True, exist_ok=True)

    #

    latents = None
    model_kwargs = dict(texts=[prompt])
    for x in tqdm(sampler.sample_batch_progressive(batch_size=batch_size, model_kwargs=model_kwargs)):
        latents = x

    assert isinstance(latents, torch.Tensor)

    torch.save(latents, out_prompt_latents_filepath)

    # pointclouds_samples: List[PointCloud] = sampler.output_to_point_clouds(output=samples)


###


def main(
    prompt_filepath: Path,
    out_rootpath: Path,
    # batch_size: int,
    skip_existing: bool,
) -> None:
    assert isinstance(prompt_filepath, Path)
    assert isinstance(out_rootpath, Path)
    # assert isinstance(batch_size, int)
    assert isinstance(skip_existing, bool)

    if out_rootpath.exists():
        assert out_rootpath.is_dir()
    else:
        out_rootpath.mkdir(parents=True)

    #

    sampler = build_sampler()
    prompts = Utils.Prompt.extract_from_file(filepath=prompt_filepath)

    print("")
    for prompt in prompts:
        if not isinstance(prompt, str) or len(prompt) < 2:
            continue

        print("")
        print(prompt)

        try:
            _generate_latents(
                prompt=prompt,
                out_rootpath=out_rootpath,
                sampler=sampler,
                skip_existing=skip_existing,
                # batch_size=batch_size,
            )
        except Exception as e:
            print("")
            print("")
            print("========================================")
            print("Error while running prompt -> ", prompt)
            print(e)
            print("========================================")
            print("")
            print("")
            continue

        print("")
    print("")


###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt-file', type=Path, required=True)
    parser.add_argument('--out-path', type=Path, required=True)
    # parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true", default=False)

    args = parser.parse_args()

    #

    main(
        prompt_filepath=args.prompt_file,
        out_rootpath=args.out_path,
        # batch_size=args.batch_size,
        skip_existing=args.skip_existing,
    )
