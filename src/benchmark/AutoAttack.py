import argparse
import logging
import os
from pathlib import Path

from collections import OrderedDict

import torch
import torch.nn as nn

from backbone.iresnet import iresnet50
from backbone.wideresnet import WRN
from aa_backbone.wide_resnet_silu_no_fe import wideresnetwithswish
from data_loading.data_loaders import get_face_loaders, get_data_loaders

from root import ROOT_DIR

from autoattack import AutoAttack


def main(args):

    # Thanks a lot PIL
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.CRITICAL + 1)

    # Args parsing for run
    part_save_dir = os.path.dirname(args.pre_model)

    save_dir = os.path.join(ROOT_DIR, part_save_dir, "aa")

    dataset = args.dataset

    batch_size = args.batch_size

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_file = os.path.join(save_dir, "pers.log")# 00000 -> first name in folder
    aa_log_file = os.path.join(save_dir, "attack.log")
    print(f"Log file at: {log_file}")
    logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.INFO)

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    print(args)
    logging.info(args)

    if dataset == "cifar10":
        n_classes = 10
    elif dataset == "cifar100":
        n_classes = 100
    elif "vgg" in dataset:
        n_classes = 500
    else:
        raise ValueError("not valid dataset")

    if args.model[-4:] == "face":
        n_classes = 500
    print(f"# classes: {n_classes}")
    logging.info(f"# classes: {n_classes}")

    if torch.cuda.device_count() > 1:
        batch_size *= torch.cuda.device_count()

    if "vgg" in dataset:
        train_loader, test_loader = get_face_loaders(batch_size, base_dir=args.save_prefix)
    elif dataset == "cifar100":
        train_loader, test_loader = get_data_loaders(dataset, batch_size, num_workers=args.num_workers, download=True,
                                                     per_sampler=True, base_dir=ROOT_DIR)
    else:
        train_loader, test_loader = get_data_loaders(dataset, batch_size, num_workers=args.num_workers, download=True,
                                                     base_dir=ROOT_DIR)


    if "vgg" in dataset:
        model = iresnet50(n_classes)
        print("IResnt50 for face")
        logging.info("IResnt50 for face")
    elif "swish" in args.model:
        model = wideresnetwithswish(name=args.model, dataset=dataset, num_classes=n_classes)
        print("Swish model")
        logging.info("Swish model")
    else:
        model = WRN(34, 10, 0.4, num_classes=n_classes)
        print("Regular WRN")
        logging.info("Regular WRN")

    PATH = os.path.join(ROOT_DIR, args.pre_model)
    print(f"Starting from pretrained: {args.pre_model}")
    logging.info(f"Starting from pretrained: {args.pre_model}")
    state_dict = torch.load(PATH)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)


    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    epsilon = args.epsilon / 255.

    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)

    adversary = AutoAttack(model, norm=args.norm, eps=epsilon, log_path=aa_log_file,
                           version=args.version)

    with torch.no_grad():
        adv_complete = adversary.run_standard_evaluation(x_test, y_test, bs=batch_size, state_path=args.state_path)

        torch.save({'adv_complete': adv_complete}, '{}/{}_{}_1_{}_eps_{:.5f}.pth'.format(
                save_dir, 'aa', args.version, adv_complete.shape[0], args.epsilon))


def get_args():
    parser = argparse.ArgumentParser()

    # Default loss parameters
    parser.add_argument('--dataset', default='cifar100', type=str, help='Dataset')

    # Training loop
    parser.add_argument('--model', default='w-34-10-swish', type=str)

    parser.add_argument('--batch_size', default=450, type=int)

    # Attack parameters
    parser.add_argument('--epsilon', default=8., type=float)

    # Pretrained
    parser.add_argument('--pre_model', default="models/ACSAC/cifar_100/nta_at/best_model.pth", type=str)

    # Misc
    parser.add_argument('--num_workers', default=0, type=int)

    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--version', type=str, default='standard')
    parser.add_argument('--state-path', type=Path, default=None)

    return parser.parse_args()


if __name__ == '__main__':
    main(get_args())