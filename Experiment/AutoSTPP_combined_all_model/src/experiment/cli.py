from pytorch_lightning.cli import LightningCLI
from lightning_fabric.accelerators import find_usable_cuda_devices
from loguru import logger


class MyLightningCLI(LightningCLI): #定义一个自己的lightningCLI，并重新添加了一些默认参数，如果运行时没有提供参数，就会使用这里的默认参数
    def add_arguments_to_parser(self, parser):
        device = [3]
        # find_usable_cuda_devices(1)
        parser.add_argument("--catalog.Mcut", default=2.0)
        parser.add_argument("--catalog.path")
        parser.add_argument("--catalog.path_to_polygon")
        parser.add_argument("--catalog.auxiliary_start")
        parser.add_argument("--catalog.train_nll_start")
        parser.add_argument("--catalog.val_nll_start")
        parser.add_argument("--catalog.test_nll_start")
        parser.add_argument("--catalog.test_nll_end")
        parser.set_defaults(
            {
                "trainer.accelerator": "cuda", 
                "trainer.devices": device,
            }
        )