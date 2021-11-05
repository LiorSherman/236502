from experiments import hidden_size_exp, optimizers_exp, seq_len_exp
import argparse
import os


def desc():
    return 'Experimenting with different hyperparameters'


def execute():
    parser = argparse.ArgumentParser(description='Experiments', prog=__name__)
    parser.add_argument('--output', default="experiments_results",
                        help='path of the output dir [default: experiments_results]')
    parser.add_argument('--dataset', default="dataset/deutschl/test",
                        help='path of the dataset dir [default: test dataset]')
    parser.add_argument('--hidden_size', action='store_true', help="run kfold cross validation of hidden size")
    parser.add_argument('--opt', action='store_true', help="run kfold cross validation of optimizers")
    parser.add_argument('--seq', action='store_true', help="run kfold cross validation of seq lengths")
    parser.add_argument('--all', action='store_true', help="run kfold cross validation of all experiments")
    parser.add_argument('--dont_show', action='store_false', help="only saves the outputs without live plotting")

    args = parser.parse_args()
    show = args.dont_show
    dataset = args.dataset

    run_exp = False
    if args.hidden_size or args.all:
        run_exp = True
        hidden_size_exp(output=args.output, show=show, dataset=dataset)
    if args.opt or args.all:
        run_exp = True
        optimizers_exp(output=args.output, show=show, dataset=dataset)
    if args.seq or args.all:
        run_exp = True
        seq_len_exp(output=args.output, show=show, dataset=dataset)

    if not run_exp:
        parser.print_help()
