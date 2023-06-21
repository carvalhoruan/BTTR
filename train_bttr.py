from pytorch_lightning import Trainer
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.utilities import seed
from bttr.lit_bttr import LitBTTR
from bttr.datamodule import CROHMEDatamodule

from train_bttr_1 import train_test_BTTR_1

GPUS = 1

if __name__ == '__main__':
    params_1 = dict(seed_everything='7',
                        trainer={'checkpoint_callback': True, 'callbacks': None,
                                 'gpus': GPUS, 'check_val_every_n_epoch': 5, 'max_epochs':500},
                        model={'d_model': 64, 'growth_rate': 24, 'num_layers': 16, 'nhead': 2, 'num_decoder_layers': 2,
                               'dim_feedforward': 256, 'dropout': 0.3, 'beam_size': 10, 'max_len': 200, 'alpha': 1.0,
                               'learning_rate': 1.0, 'patience': 20},
                        data={'zipfile_path': '../bases/Base_soma_subtracao_optuna_bttr.zip', 'test_year': 'test',
                              'batch_size': 8, 'num_workers': 5, 'data_augmentation':100}
                        )
    
    params_2 = dict(seed_everything='7',
                        trainer={'checkpoint_callback': True, 'callbacks': None,
                                 'gpus': GPUS, 'check_val_every_n_epoch': 5, 'max_epochs':500},
                        model={'d_model': 64, 'growth_rate': 24, 'num_layers': 16, 'nhead': 4, 'num_decoder_layers': 3,
                               'dim_feedforward': 512, 'dropout': 0.3, 'beam_size': 5, 'max_len': 200, 'alpha': 1.0,
                               'learning_rate': 1.0, 'patience': 20},
                        data={'zipfile_path': '../bases/Base_soma_subtracao_optuna_bttr.zip', 'test_year': 'test',
                              'batch_size': 8, 'num_workers': 5, 'data_augmentation':100}
                        )

    params_3 = dict(seed_everything='7',
                        trainer={'checkpoint_callback': True, 'callbacks': None,
                                 'gpus': GPUS, 'check_val_every_n_epoch': 5, 'max_epochs':500},
                        model={'d_model': 64, 'growth_rate': 24, 'num_layers': 16, 'nhead': 2, 'num_decoder_layers': 2,
                               'dim_feedforward': 256, 'dropout': 0.3, 'beam_size': 5, 'max_len': 200, 'alpha': 1.0,
                               'learning_rate': 1.0, 'patience': 20},
                        data={'zipfile_path': '../bases/Base_soma_subtracao_optuna_bttr.zip', 'test_year': 'test',
                              'batch_size': 8, 'num_workers': 5, 'data_augmentation':100}
                        )
    
    params_4 = dict(seed_everything='7',
                        trainer={'checkpoint_callback': True, 'callbacks': None,
                                 'gpus': GPUS, 'check_val_every_n_epoch': 5, 'max_epochs':500},
                        model={'d_model': 64, 'growth_rate': 16, 'num_layers': 8, 'nhead': 2, 'num_decoder_layers': 2,
                               'dim_feedforward': 512, 'dropout': 0.3, 'beam_size': 5, 'max_len': 200, 'alpha': 1.0,
                               'learning_rate': 1.0, 'patience': 20},
                        data={'zipfile_path': '../bases/Base_soma_subtracao_optuna_bttr.zip', 'test_year': 'test',
                              'batch_size': 8, 'num_workers': 5, 'data_augmentation':100}
                        )
    
    
    params_5 = dict(seed_everything='7',
                        trainer={'checkpoint_callback': True, 'callbacks': None,
                                 'gpus': GPUS, 'check_val_every_n_epoch': 5, 'max_epochs':500},
                        model={'d_model': 64, 'growth_rate': 24, 'num_layers': 16, 'nhead': 8, 'num_decoder_layers': 2,
                               'dim_feedforward': 512, 'dropout': 0.3, 'beam_size': 5, 'max_len': 200, 'alpha': 1.0,
                               'learning_rate': 1.0, 'patience': 20},
                        data={'zipfile_path': '../bases/Base_soma_subtracao_optuna_bttr.zip', 'test_year': 'test',
                              'batch_size': 8, 'num_workers': 5, 'data_augmentation':100}
                        )
    
    params_6 = dict(seed_everything='7',
                        trainer={'checkpoint_callback': True, 'callbacks': None,
                                 'gpus': GPUS, 'check_val_every_n_epoch': 5, 'max_epochs':500},
                        model={'d_model': 64, 'growth_rate': 24, 'num_layers': 16, 'nhead': 2, 'num_decoder_layers': 2,
                               'dim_feedforward': 512, 'dropout': 0.3, 'beam_size': 5, 'max_len': 200, 'alpha': 1.0,
                               'learning_rate': 1.0, 'patience': 20},
                        data={'zipfile_path': '../bases/Base_soma_subtracao_optuna_bttr.zip', 'test_year': 'test',
                              'batch_size': 8, 'num_workers': 5, 'data_augmentation':100}
                        )
    
    params_7 = dict(seed_everything='7',
                        trainer={'checkpoint_callback': True, 'callbacks': None,
                                 'gpus': GPUS, 'check_val_every_n_epoch': 5, 'max_epochs':500},
                        model={'d_model': 64, 'growth_rate': 24, 'num_layers': 16, 'nhead': 2, 'num_decoder_layers': 2,
                               'dim_feedforward': 1024, 'dropout': 0.3, 'beam_size': 5, 'max_len': 200, 'alpha': 1.0,
                               'learning_rate': 1.0, 'patience': 20},
                        data={'zipfile_path': '../bases/Base_soma_subtracao_optuna_bttr.zip', 'test_year': 'test',
                              'batch_size': 8, 'num_workers': 5, 'data_augmentation':100}
                        )
    
    print(train_test_BTTR_1(params_1))