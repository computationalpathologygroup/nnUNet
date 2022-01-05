import pickle
import pytest

from pathlib import Path

from nnunet.training.dataloading.dataset_loading import (
    load_dataset,
    unpack_dataset,
    DataLoader3D,
    DataLoader3DWeighted,
    DataLoader2D,
)
import numpy as np

from nnunet.training.network_training.diag.nnUNetTrainerV2Weighted import nnUNetTrainerV2Weighted
from tests.test_training import prepare_paths, check_expected_training_output

RESOURCES_DIR = Path(__file__).parent / "resources"
NNUNET_PREPROCESSING_DATA_DIR = RESOURCES_DIR / "nnUNet" / "nnUNet_preprocessed_data"
HIPPOCAMPUS_TASK_ID = 4

import nnunet.utilities.task_name_id_conversion as nnp
import nnunet.experiment_planning.nnUNet_plan_and_preprocess as nnpap
import nnunet.experiment_planning.utils as nnepu
import nnunet.run.default_configuration as nndc
from nnunet.paths import default_plans_identifier


def test_dataset_loading(tmp_path: Path):
    task = "Task004_Hippocampus"
    p = NNUNET_PREPROCESSING_DATA_DIR / task / "nnUNetData_plans_v2.1_stage0"
    dataset = load_dataset(str(p))
    with open(
        NNUNET_PREPROCESSING_DATA_DIR / task / "nnUNetPlansv2.1_plans_3d.pkl", "rb"
    ) as f:
        plans = pickle.load(f)
    patch_size = plans["plans_per_stage"][0]["patch_size"]
    unpack_dataset(str(p))
    d2 = DataLoader3D(
        dataset, (32, 32, 32), (32, 32, 32), 2, oversample_foreground_percent=0.33
    )
    next(d2)
    d1 = DataLoader3DWeighted(
        dataset, (32, 32, 32), (32, 32, 32), 2, oversample_foreground_percent=0.33
    )
    next(d1)

    DataLoader3D(
        dataset,
        np.array(patch_size).astype(int),
        np.array(patch_size).astype(int),
        2,
        oversample_foreground_percent=0.33,
    )
    DataLoader2D(
        dataset,
        (64, 64),
        np.array(patch_size).astype(int)[1:],
        12,
        oversample_foreground_percent=0.33,
    )


@pytest.mark.parametrize("network", ("3d_fullres", ))
@pytest.mark.parametrize("fold", (0,))
def test_weighted_trainer(tmp_path: Path, network: str, fold: int):
    prepare_paths(output_dir=tmp_path)
    task = nnp.convert_id_to_task_name(HIPPOCAMPUS_TASK_ID)
    decompress_data = True
    deterministic = False
    run_mixed_precision = True
    (
        plans_file,
        output_folder_name,
        dataset_directory,
        batch_dice,
        stage,
        trainer_class,
    ) = nndc.get_default_configuration(
        network, task, "nnUNetTrainerV2Weighted", default_plans_identifier
    )
    assert issubclass(trainer_class, nnUNetTrainerV2Weighted)
    trainer = trainer_class(
        plans_file,
        fold,
        output_folder=output_folder_name,
        dataset_directory=dataset_directory,
        batch_dice=batch_dice,
        stage=stage,
        unpack_data=decompress_data,
        deterministic=deterministic,
        fp16=run_mixed_precision,
    )
    trainer.max_num_epochs = 2
    trainer.num_batches_per_epoch = 2
    trainer.num_val_batches_per_epoch = 2
    trainer.initialize(True)
    # trainer.run_training()
    # trainer.network.eval()
    # trainer.validate(
    #     save_softmax=False,
    #     validation_folder_name="validation_raw",
    #     run_postprocessing_on_folds=True,
    #     overwrite=True,
    # )
    # check_expected_training_output(check_dir=tmp_path, network=network)
