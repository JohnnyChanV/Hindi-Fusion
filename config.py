from argparse import ArgumentParser
from pprint import pprint


class config:
    def __init__(self):
        # 创建解析器对象
        parser = ArgumentParser()
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--max_length", type=int, default=128)

        parser.add_argument("--seed", type=int, default=42)

        parser.add_argument("--lr", type=float, default=4e-5)
        parser.add_argument("--weight_decay", type=float, default=1e-3)
        parser.add_argument("--epochs", type=int, default=1000)
        parser.add_argument("--dropout", type=float, default=0.1)

        parser.add_argument("--early_stop", type=int, default=3)
        parser.add_argument("--valid_round", type=int, default=1)

        parser.add_argument("--task", type=str, default='ner')
        # classification
        # Infer

        parser.add_argument("--data_path", type=str, default='datasets/raw_data')
        parser.add_argument("--dataset_name", type=str, default='IJNLP')
        # "iitp-product-reviews"
        # "iitp+-movie-reviews"
        # "hindi-news"
        # "bbc-articles"
        # "XNLI"

        parser.add_argument("--use_plm", action='store_true')
        parser.add_argument("--use_cls", action='store_true')
        parser.add_argument("--embed_dim", type=int, default=300)

        parser.add_argument("--encoder", type=str, default='FF')
        # "FF"
        # "DF"
        # "dev_PLM"
        # "rom_PLM"
        parser.add_argument("--fusion_mode", type=str, default=None)
        # 'cat', (FF encoder only)
        # 'gate',(FF encoder only)

        parser.add_argument("--core_model", type=str, default=None)

        args = parser.parse_args()
        self.args = args
        self.task = args.task
        self.dataset_name = args.dataset_name

        self.seed = args.seed
        self.data_path = args.data_path
        self.datadir = f"{self.data_path}/{self.task}/{self.dataset_name}"
        self.max_length = args.max_length
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.epochs = args.epochs
        self.weight_decay = args.weight_decay
        self.dropout = args.dropout
        self.ValidRound = args.valid_round
        self.usePLM = args.use_plm
        self.useCLS = args.use_cls
        self.Word2VecEmbedSize = args.embed_dim
        self.earlystop = args.early_stop
        self.stride = 128

        self.encoder_name = args.encoder
        # self.encoder_name = "DF"
        # self.encoder_name = "dev_PLM"
        # self.encoder_name = "rom_PLM"

        self.FF_encoder_config = {
            'pooling_mode': 'mean',
            'fusion_mode': args.fusion_mode,
            # 'fusion_mode': 'gate',
        }

        self.MF_encoder_config = {
            'pooling_mode': 'mean',
            'core_model': args.core_model
        }

        self.DF_encoder_config = {
            'pooling_mode': 'mean',
            # 'core_model': 'bilstm',
            # 'core_model': 'mix'
            'core_model': args.core_model
        }

        self.dev_PLM_encoder_config = {
            'pooling_mode': 'mean',
            # 'core_model': None,
            # 'core_model': 'bilstm',
            'core_model': args.core_model
        }

        self.rom_PLM_encoder_config = {
            'pooling_mode': 'mean',
            # 'core_model': None,
            # 'core_model': 'bilstm',
            'core_model': args.core_model
        }

        self.encoder_config = eval(f"self.{self.encoder_name}_encoder_config")

        if self.task == 'pos' or self.task == 'QA' or self.task == 'ner':
            self.encoder_config['pooling_mode'] = None
