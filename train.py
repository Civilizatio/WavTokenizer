import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from pytorch_lightning.cli import LightningCLI, ArgsType
import logging


def cli_main(args: ArgsType = None):
    # breakpoint()
    cli = LightningCLI(args=args)
    
    # log_dir = cli.trainer.logger.log_dir
    
    # os.makedirs(log_dir, exist_ok=True)
    # log_file = os.path.join(log_dir, 'training.log')
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format='%(asctime)s %(levelname)s: %(message)s',
    #     handlers=[
    #         logging.FileHandler(log_file),
    #         logging.StreamHandler()
    #     ]
    # )
    # breakpoint()
    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    cli_main()
