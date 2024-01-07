import argparse


def parser_sequential_args():
    parser = argparse.ArgumentParser(description="Run Sequential model.")
    parser.add_argument('--data_name', nargs='?', default='ml_100k',
                        help='Choose a dataset from {yelp2018, last-fm, amazon-book}')
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--bert_max_len', type=int, default=20, help='Length of sequence for bert')
    parser.add_argument('--bert_num_items', type=int, default=None, help='Number of total items')
    parser.add_argument('--bert_hidden_units', type=int, default=64, help='Size of hidden vectors (d_model)')
    parser.add_argument('--bert_num_blocks', type=int, default=2, help='Number of transformer layers')
    parser.add_argument('--bert_num_heads', type=int, default=2, help='Number of heads for multi-attention')
    parser.add_argument('--bert_dropout', type=float, default=0.2, help='Dropout probability to use throughout the model')
    parser.add_argument('--bert_mask_prob', type=float, default=0.7, help='Probability for masking items in the training sequence')
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--model_init_seed', type=int, default=2023)
    parser.add_argument('--metric_ks', nargs='+', type=int, default=[50], help='ks for Metric@k')
    
    parser.add_argument('--n_epoch_each_frame_add_one_more_mask', type = int, default = 10)
    parser.add_argument('--lr', type = float, default = 1e-3)
    parser.add_argument('--use_task_mask_for_gradient_protecting', type = int, default = 1)
    parser.add_argument('--use_task_mask', type=int, default = 1)
    parser.add_argument('--epoch_not_binary_mask', type = int, default = 3)
    parser.add_argument('--limits', type = str, default = None)
    parser.add_argument('--task_ids', type = str, default = None)
    parser.add_argument('--folder_save_model', type = str, default = 'VAE')
    parser.add_argument('--data_dir', nargs='?', default='datasets/',
                        help='Input data path.')
    
    parser.add_argument('--sequential_using_ufo_space', type = int, default = 1)
    
    #
    #
    args = parser.parse_args()

    save_dir = 'trained_model/{}/{}/embed-dim{}_lr{}/'.format(
        args.folder_save_model, args.data_name, args.bert_hidden_units, args.lr)
    args.save_dir = save_dir

    return args


