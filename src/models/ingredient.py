from src.models import __dict__

def get_model(args):
    return __dict__[args.arch](num_classes=args.num_classes)