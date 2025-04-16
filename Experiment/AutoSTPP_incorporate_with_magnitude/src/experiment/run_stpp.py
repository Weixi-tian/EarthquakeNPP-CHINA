import torch
from pytorch_lightning.cli import ArgsType

from data.lightning.sliding_window import SlidingWindowDataModule
from models.lightning.stpp import BaseSTPointProcess
from cli import MyLightningCLI
from utils import find_ckpt_path, increase_u_limit


def cli_main(args: ArgsType = None): # 定义cli_main的函数，用于初始化MylightningCLI类的实例
    torch.set_float32_matmul_precision('medium') #设置pytorch的矩阵运算精度
    cli = MyLightningCLI(
        BaseSTPointProcess, #模型类，继承自LightningModule
        SlidingWindowDataModule, #加载和预处理数据的类，继承自LightningDataModule
        subclass_mode_model=True,  #子类模式，用于加载模型类
        subclass_mode_data=True, #子类模式，用于加载数据类
        save_config_callback=None, 
        run=False, #表示只从初始化cli而不运行
        args=args, #允许外部传入命令行参数,如果为空，则使用默认参数
    )
    return cli


if __name__ == '__main__':
    cli = cli_main()
    increase_u_limit()

    cli.trainer.logger.log_hyperparams({'seed': cli.config['seed_everything']}) #记录超参数seed
    cli.trainer.fit(cli.model, cli.datamodule) #训练入口,执行训练流程，更新模型参数并记录验证指标
    """
    流程如下：
    数据加载：通过 cli.datamodule.train_dataloader() 和 cli.datamodule.val_dataloader()。
    前向传播：调用 cli.model.forward()。
    反向传播和优化：基于损失函数自动计算梯度并更新参数。
    日志记录：通过训练器的 logger 记录训练指标。
    """
    cli.trainer.test(cli.model, cli.datamodule, ckpt_path='best') #测试入口，使用训练中保存的最优模型，并评估该模型在测试集上的性能
