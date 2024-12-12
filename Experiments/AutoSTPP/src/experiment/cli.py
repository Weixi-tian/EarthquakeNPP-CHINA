from pytorch_lightning.cli import LightningCLI
from lightning_fabric.accelerators import find_usable_cuda_devices
from loguru import logger


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        device = [3]
        # find_usable_cuda_devices(1)
        parser.add_argument("--catalog.Mcut", default=2.0)
        parser.add_argument("--catalog.path", "-../etas/input_data/merged_SCEDC_catalog.csv")
        parser.add_argument("--catalog.path_to_polygon", "-../etas/input_data/SCEDC_shape.npy")
        parser.add_argument("--catalog.auxiliary_start", "-1981-01-01 00:00:00")
        parser.add_argument("--catalog.train_nll_start", "-1985-01-01 00:00:00")
        parser.add_argument("--catalog.val_nll_start", "-2005-01-01 00:00:00")
        parser.add_argument("--catalog.test_nll_start", "-2014-12-31 00:00:00")
        parser.add_argument("--catalog.test_nll_end", "-2020-01-01 00:00:00")
        parser.set_defaults(
            {
                "trainer.accelerator": "cuda", 
                "trainer.devices": device,
            }
        )
