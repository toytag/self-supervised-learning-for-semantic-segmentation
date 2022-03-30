from utils.argparse import BasicParser


class Parser(BasicParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # moco specific configs
        self.add_argument('--moco-dim', default=128, type=int,
                          help='feature dimension (default: 128)')
        self.add_argument('--moco-k', default=65536, type=int,
                          help='queue size; number of negative keys (default: 65536)')
        self.add_argument('--moco-m', default=0.999, type=float,
                          help='moco momentum of updating key encoder (default: 0.999)')
        self.add_argument('--moco-t', default=0.2, type=float,
                          help='softmax temperature (default: 0.2)')
