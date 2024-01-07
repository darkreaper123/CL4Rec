import argparse


def parse_vae_args():
    parser = argparse.ArgumentParser(description="Run VAE.")
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--hidden-dim', type=int, default=600)
    parser.add_argument('--latent-dim', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--start_epoch_for_multi_masks_each_task_training', type = int, default = 120)
    parser.add_argument('--n_epoch_each_frame_add_one_more_mask', type = int, default = 10)
    parser.add_argument('--use_task_mask_for_gradient_protecting', type = int, default = 1)
    parser.add_argument('--use_task_mask', type=int, default = 1)
    parser.add_argument('--epoch_not_binary_mask', type = int, default = 3)
    parser.add_argument('--limits', type = str, default = None)
    parser.add_argument('--task_ids', type = str, default = None)
    parser.add_argument('--folder_save_model', type = str, default = 'VAE')
    parser.add_argument('--data_dir', nargs='?', default='datasets/',
                        help='Input data path.')
    
    parser.add_argument('--vae_using_ufo_space', type = int, default = 1)
    args = parser.parse_args()

    save_dir = 'trained_model/{}/{}/embed-dim{}_lr{}/'.format(
        args.folder_save_model, args.data_name, args.hidden_dim, args.lr)
    args.save_dir = save_dir

    return args