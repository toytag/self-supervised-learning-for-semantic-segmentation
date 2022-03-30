from utils.argparse import BasicParser


class Parser(BasicParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Siamese specific configs
        self.add_argument('--dim', default=256, type=int,
                          help='feature dimension (default: 256)')
