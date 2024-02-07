### pylint: disable=missing-function-docstring,missing-class-docstring,missing-module-docstring,wrong-import-order
from typing import Tuple, Any, List, Iterator
from pathlib import Path

import argparse
import torch

from pathlib import Path
from point_e.diffusion.configs import DIFFUSION_CONFIGS
from point_e.diffusion.configs import diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS
from point_e.models.configs import model_from_config
# from point_e.util.plotting import plot_point_cloud
from point_e.util.point_cloud import PointCloud
from point_e.util.pc_to_mesh import marching_cubes_mesh

from utils import Utils
from tt3d_generate import build_sampler

###

T_Prompt = Tuple[str, Path]  ### pylint: disable=invalid-name
# T_Prompts = List[T_Prompt]  ### pylint: disable=invalid-name
T_Prompts = Iterator[T_Prompt]  ### pylint: disable=invalid-name

device = Utils.Cuda.init()

###


def _load_models() -> Tuple[PointCloudSampler, Any]:
    sampler = build_sampler()
    #
    print('creating SDF model...')
    model = model_from_config(MODEL_CONFIGS['sdf'], device)
    model.eval()
    #
    print('loading SDF model...')
    model.load_state_dict(load_checkpoint('sdf', device))
    #
    return sampler, model


def _load_prompts_from_source_path(source_rootpath: Path) -> T_Prompts:
    assert isinstance(source_rootpath, Path)
    assert source_rootpath.exists()
    assert source_rootpath.is_dir()

    experiment_path = Utils.Storage.build_experiment_path(out_rootpath=source_rootpath)

    # for prompt_path in source_path.iterdir():
    for prompt_path in experiment_path.iterdir():
        if prompt_path.is_dir():
            prompt_enc = prompt_path.name
            yield (prompt_enc, prompt_path)


def _convert_latents_to_pointclouds(
    prompt: str,
    source_rootpath: Path,
    sampler: PointCloudSampler,
# ) -> List[PointCloud]:
) -> PointCloud:
    assert sampler is not None

    source_prompt_latents_filepath = Utils.Storage.build_prompt_latents_filepath(
        out_rootpath=source_rootpath,
        prompt=prompt,
        assert_exists=True,
    )

    latents = torch.load(source_prompt_latents_filepath)

    #

    # pointclouds: List[PointCloud] = sampler.output_to_point_clouds(output=latents)
    # for idx, pointcloud in enumerate(pointclouds):
    #     out_pointcloud_filepath = Utils.Storage.build_prompt_pointcloud_filepath(
    #         out_rootpath=source_rootpath,
    #         prompt=prompt,
    #         assert_exists=False,
    #         idx=idx,
    #     )
    #     out_pointcloud_filepath.parent.mkdir(parents=True, exist_ok=True)
    #     pointcloud.save(str(out_pointcloud_filepath))

    pointclouds: List[PointCloud] = sampler.output_to_point_clouds(output=latents)
    assert len(pointclouds) == 1
    pointcloud = pointclouds[0]
    
    out_pointcloud_filepath = Utils.Storage.build_prompt_pointcloud_filepath(
        out_rootpath=source_rootpath,
        prompt=prompt,
        assert_exists=False,
    )
    out_pointcloud_filepath.parent.mkdir(parents=True, exist_ok=True)
    pointcloud.save(str(out_pointcloud_filepath))

    # return pointclouds
    return pointcloud


def _convert_pointclouds_to_objs(
    prompt: str,
    source_rootpath: Path,
    # pointclouds: List[PointCloud],
    pointcloud: PointCloud,
    model: Any,
) -> None:
    assert model is not None
    assert isinstance(pointcloud, PointCloud)

    # for idx, pointcloud in enumerate(pointclouds):
    
    ### produce a mesh (with vertex colors)
    mesh = marching_cubes_mesh(
        pc=pointcloud,
        model=model,
        batch_size=4096,
        # grid_size=32,  # increase to 128 for resolution used in evals
        grid_size=128,
        progress=True,
    )

    out_ply_filepath = Utils.Storage.build_prompt_mesh_filepath(
        out_rootpath=source_rootpath,
        prompt=prompt,
        assert_exists=False,
        # idx=idx,
        extension="ply",
    )
    out_ply_filepath.parent.mkdir(parents=True, exist_ok=True)
    out_obj_filepath = Utils.Storage.build_prompt_mesh_filepath(
        out_rootpath=source_rootpath,
        prompt=prompt,
        assert_exists=False,
        # idx=idx,
        extension="obj",
    )
    out_obj_filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(out_ply_filepath, 'wb+') as f:
        mesh.write_ply(f)
    with open(out_obj_filepath, 'w+', encoding="utf-8") as f:
        mesh.write_obj(f)


###


def main(source_rootpath: Path) -> None:
    assert isinstance(source_rootpath, Path)
    assert source_rootpath.exists()
    assert source_rootpath.is_dir()
    # assert isinstance(skip_existing, bool)

    sampler, model = _load_models()
    prompts = _load_prompts_from_source_path(source_rootpath=source_rootpath)

    #

    print("")
    for prompt_enc, _ in prompts:
        prompt = Utils.Prompt.decode(prompt_enc)

        if not isinstance(prompt, str) or len(prompt) < 2:
            continue

        print("")
        print(prompt)

        # pointclouds = _convert_latents_to_pointclouds(
        pointcloud = _convert_latents_to_pointclouds(
            prompt=prompt,
            source_rootpath=source_rootpath,
            sampler=sampler,
        )

        _convert_pointclouds_to_objs(
            prompt=prompt,
            source_rootpath=source_rootpath,
            # pointclouds=pointclouds,
            pointcloud=pointcloud,
            model=model,
        )
        print("")
    print("")


###

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--source-path', type=Path, required=True)
    ### TODO: add logic to skip existing pointclouds and objs.
    # parser.add_argument("--skip-existing", action="store_true", default=False)

    args = parser.parse_args()

    #

    main(source_rootpath=args.source_path,
         # skip_existing=args.skip_existing,
        )
