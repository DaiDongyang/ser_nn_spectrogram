import argparse
# from utils import cfg_process
# from CRModel import cr_model_run
import os


def add_arguments(parser):
    """Build ArgumentParser."""
    parser.register("type", "bool", lambda v: v.lower() == "true")

    # network
    parser.add_argument("--num_units", type=int, default=32, help="Network size.")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Network depth.")
    parser.add_argument("--num_encoder_layers", type=int, default=None,
                        help="Encoder depth, equal to num_layers if None.")
    parser.add_argument("--num_decoder_layers", type=int, default=None,
                        help="Decoder depth, equal to num_layers if None.")


def log(*args, sep=' ', end='\n'):
    print(*args, sep=sep, end=end)


if __name__ == '__main__':
    model_parser = argparse.ArgumentParser()
    add_arguments(model_parser)
    FLAGS, unparsed = model_parser.parse_known_args()
    print('type flags:', type(FLAGS))
    print('flags:', FLAGS)
    print('type unparsed:', type(unparsed))
    print('unparsed:', unparsed)
    # yparams = cfg_process.YParams('./CRModel/CRModel.yml', 'default')
    # yparams = cr_model_run.CRHParamsPreprocessor(yparams, None).preprocess()
    # yparams.save()
    # log('hello', 'world %f' % 1.1)
