import argparse


def get_args_parser():
    parser = argparse.ArgumentParser()

    # Setup
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--data_name', default='BasicMotions_256_00', type=str)
    parser.add_argument('--num_feature', default=6, type=int)
    parser.add_argument('--num_target', default=4, type=int)
    
    # Data parameters
    parser.add_argument('--full_training', action='store_true', help='Enable full training mode (default: False)')
    parser.add_argument('--batch_size_pretrain', default=128, type=int)
    parser.add_argument('--batch_size_finetune', default=16, type=int)
    
    # Model parameters
    parser.add_argument('--num_embedding', default=64, type=int)
    parser.add_argument('--num_hidden', default=128, type=int)
    parser.add_argument('--num_head', default=4, type=int)
    parser.add_argument('--num_layers', default=3, type=int)
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--feature', default='hidden', type=str)
    
    # Training parameters
    parser.add_argument('--epochs_pretrain', default=200, type=int)
    parser.add_argument('--epochs_finetune', default=100, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--loss_type', default='ALL', type=str)
    parser.add_argument('--temperature', default=0.07, type=float)
    parser.add_argument('--lam', default=0.0, type=float)
    parser.add_argument('--partial', default=1.0, type=float)
    
    return parser


def parse_args(args=None):
    parser = get_args_parser()
    parsed_args = parser.parse_args(args=args)
    return parsed_args