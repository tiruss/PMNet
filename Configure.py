import argparse

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda', action='store_true', default=1)

    parser.add_argument('--nc_img', type=int, default=3)
    parser.add_argument('--nfc', type=int, default=32)
    parser.add_argument('--ker_size', type=int, default=3)
    parser.add_argument('--num_layer', type=int, default=5)
    parser.add_argument('--stride', default=1)
    parser.add_argument('--padd_size', type=int, default=0)

    parser.add_argument('--niter', type=int, default=2000)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=0.5)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=224)

    return parser
