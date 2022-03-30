from utils.argparse import ArchParser
from utils.main_template import Main
import models.selfsup

model_names = sorted(name for name in models.selfsup.__dict__
    if not name.startswith("__") and name in models.selfsup.__all__
    and hasattr(models.selfsup.__dict__[name], 'Parser') 
    and hasattr(models.selfsup.__dict__[name], 'Worker'))

if __name__ == '__main__':
    arch_arg, others = ArchParser(model_names).parse_known_args()
    args = models.selfsup.__dict__[arch_arg.arch].Parser().parse_args(others)
    sess = Main(args, models.selfsup.__dict__[arch_arg.arch].Worker())
    sess.run()