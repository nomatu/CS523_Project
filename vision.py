import argparse

parser = argparse.ArgumentParser(description='SIDPSGD')
parser.add_argument('--dataset', default='mnist')
parser.add_argument('--noise_multiplier', type=float, default=7.1)
parser.add_argument('--clip', type=float, default=3.1)
parser.add_argument('--private', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--normalization_type', default='Batch')
parser.add_argument('--device', default=None)
parser.add_argument('--robust', type=int, default=0)
parser.add_argument('--adv_step_size', type=float, default=0.01)
parser.add_argument('--adv_alpha', type=float, default=0.3)
parser.add_argument('--adv_num_steps', type =int, default=20)
parser.add_argument('--resume_at_epoch', type =int, default=0)


args = parser.parse_args()

if args.dataset == 'mnist':
     from src.mnist.train import main
elif args.dataset == 'cifar10':
     from src.cifar10.train import main


main(args.noise_multiplier, args.clip, args.private, args.lr, args.batch_size, args.epochs, args.normalization_type, args.device,
     args.robust, args.adv_step_size, args.adv_alpha, args.adv_num_steps, args.resume_at_epoch)

