import argparse
import logging
import os
import shutil

from torch.backends import cudnn

from root import ROOT_DIR
from backbone.iresnet import iresnet50
from backbone.wide_resnet_silu import wideresnetwithswish
from backbone.wideresnet import WRN
from trades import trades_loss

import torch
import torch.nn as nn
import torch.optim as optim

from benchmark.adversarial import evaluate_pgd
from data_loading.data_loaders import get_data_loaders, get_face_loaders

from torch_ema import ExponentialMovingAverage


def main(args):

    # Thanks a lot PIL
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.CRITICAL + 1)

    # Args parsing for run
    save_dir = os.path.join(args.save_prefix, args.save_dir)
    dataset = args.dataset
    batch_size = args.batch_size

    if os.path.exists(save_dir):
        raise ValueError(f"Dir f{save_dir} already exists")
    os.makedirs(save_dir)

    log_file = os.path.join(save_dir, "00000_log.log") # 00000 : first name in folder
    print(f"Log file at: {log_file}")
    if os.path.exists(log_file):
        os.remove(log_file)
    logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.INFO)

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    print(args)
    logging.info(args)
    print(f"Classifier AE: {args.class_ae}")
    logging.info(f"Classifier AE: {args.class_ae}")

    if dataset == "cifar10":
        n_classes = 10
    elif dataset == "cifar100":
        n_classes = 100
    elif "vgg" in dataset:
        n_classes = 500
    else:
        raise ValueError("not valid dataset")

    print(f"# classes: {n_classes}")
    logging.info(f"# classes: {n_classes}")

    if torch.cuda.device_count() > 1:
        batch_size *= torch.cuda.device_count()

    if "vgg" in dataset:
        train_loader, test_loader = get_face_loaders(batch_size, base_dir=args.save_prefix)
    elif dataset == "cifar100":
        train_loader, test_loader = get_data_loaders(dataset, batch_size, num_workers=args.num_workers, download=True, per_sampler=True, base_dir=args.save_prefix)
    else:
        train_loader, test_loader = get_data_loaders(dataset, batch_size, num_workers=args.num_workers, download=True, base_dir=args.save_prefix)

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

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    if args.opt == "sgd":
        opt = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.0005, momentum=0.9, nesterov=True)
        print(f"SGD: {args.lr}")
        logging.info(f"SGD: {args.lr}")
    elif args.opt == "adam":
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
        print(f"Adam: {args.lr}")
        logging.info(f"Adam: {args.lr}")
    else:
        raise ValueError("Invalid optimizer")

    if args.sched == "cosan":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        print(f"CosineAnnealing")
        logging.info(f"CosineAnnealing")
    elif args.sched == "multi":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[70, 110, 130], gamma=0.1)
        print(f"MultiStep")
        logging.info(f"MultiStep")
    else:
        raise ValueError("Invalid scheduler")

    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

    cudnn.benchmark = True

    epsilon = torch.tensor(args.epsilon / 255., dtype=torch.float32).to(device)
    alpha = torch.tensor(args.alpha / 255., dtype=torch.float32).to(device)

    best_rob_acc = 0.0
    best_clean_acc = 0.0

    for epoch in range(args.epochs):
        train_loss = 0
        train_n = 0
        correct_clean = 0
        correct_adv = 0

        model.train()
        print("---Entering training")
        logging.info("---Entering training")
        for i, (X, y) in enumerate(train_loader):

            opt.zero_grad()

            y, X = y.to(device), X.to(device)

            loss, batch_metrics, _ = trades_loss(model, X, y, opt, step_size=alpha, epsilon=epsilon,
                                              perturb_steps=args.attack_iters, beta=args.beta, classifier_ae=args.class_ae)

            loss.backward()

            opt.step()

            if epoch > 2:
                ema.update()

            train_loss += loss.item() * y.size(0)
            train_n += y.size(0)

            correct_clean += batch_metrics["clean_correct"]
            correct_adv += batch_metrics["adversarial_correct"]


            if (i + 1) % 10 == 0:
                lr = opt.param_groups[0]['lr']
                print(
                    f"{epoch} \t {lr} \t {train_n} \t {train_loss / train_n} \t "
                    f"{correct_clean / train_n} \t {correct_adv / train_n}"
                )
                logging.info(
                    f"{epoch} \t {lr} \t {train_n} \t {train_loss / train_n} \t "
                    f"{correct_clean / train_n} \t {correct_adv / train_n}"
                )

        model.eval()

        correct = 0
        total = 0

        if epoch > 2:
            with ema.average_parameters():
                with torch.no_grad():
                    for i, (X, y) in enumerate(test_loader):
                        y, X = y.to(device), X.to(device)

                        # calculate outputs by running images through the network
                        outputs_adv, _ = model(X)
                        # the class with the highest energy is what we choose as prediction
                        _, predicted = torch.max(outputs_adv.data, 1)
                        total += y.size(0)
                        correct += (predicted == y).sum().item()

                print(f'Test accuracy: {100 * correct / total} %')
                logging.info(f'Test accuracy: {100 * correct / total} %')

                print('Evaluating PGD')
                logging.info('Evaluating PGD')

                rob_acc = evaluate_pgd(test_loader, model, 1, 40, epsilon, alpha, device)
                if rob_acc > best_rob_acc:
                    best_rob_acc = rob_acc

                    full_model_save_path = os.path.join(save_dir, f"best_model.pth")
                    torch.save(model.state_dict(), full_model_save_path)
                    print(f"Model saved at: {full_model_save_path} at epoch {epoch}")
                    logging.info(f"Model saved at: {full_model_save_path} at epoch {epoch}")
                elif rob_acc == best_rob_acc:
                    if correct / total > best_clean_acc:
                        best_clean_acc = correct / total
                        best_rob_acc = rob_acc

                        full_model_save_path = os.path.join(save_dir, f"best_model.pth")
                        torch.save(model.state_dict(), full_model_save_path)
                        print(f"Model saved at: {full_model_save_path} at epoch {epoch}")
                        logging.info(f"Model saved at: {full_model_save_path} at epoch {epoch}")

        else:
            with torch.no_grad():
                for i, (X, y) in enumerate(test_loader):
                    y, X = y.to(device), X.to(device)

                    # calculate outputs by running images through the network
                    outputs_adv, _ = model(X)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs_adv.data, 1)
                    total += y.size(0)
                    correct += (predicted == y).sum().item()

            print(f'Test accuracy: {100 * correct / total} %')
            logging.info(f'Test accuracy: {100 * correct / total} %')

            print('Evaluating PGD')
            logging.info('Evaluating PGD')

            rob_acc = evaluate_pgd(test_loader, model, 1, 40, epsilon, alpha, device)
            if rob_acc > best_rob_acc:
                best_rob_acc = rob_acc

                full_model_save_path = os.path.join(save_dir, f"best_model.pth")
                torch.save(model.state_dict(), full_model_save_path)
                print(f"Model saved at: {full_model_save_path} at epoch {epoch}")
                logging.info(f"Model saved at: {full_model_save_path} at epoch {epoch}")
            elif rob_acc == best_rob_acc:
                if correct / total > best_clean_acc:
                    best_clean_acc = correct / total
                    best_rob_acc = rob_acc

                    full_model_save_path = os.path.join(save_dir, f"best_model.pth")
                    torch.save(model.state_dict(), full_model_save_path)
                    print(f"Model saved at: {full_model_save_path} at epoch {epoch}")
                    logging.info(f"Model saved at: {full_model_save_path} at epoch {epoch}")

        scheduler.step()


def get_args():
    parser = argparse.ArgumentParser()

    # Mandatory
    parser.add_argument('save_dir', type=str, help='Save dir')
    parser.add_argument('save_prefix', default=ROOT_DIR, type=str, help='Base dir')

    # Default loss parameters
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset')

    # Training loop
    parser.add_argument('--model', default='w-34-10-swish', type=str)

    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--batch_size', default=180, type=int)

    parser.add_argument('--opt', default="adam", type=str, choices=["adam", "sgd"], help="adam or sgd")
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument("--sched", default="multi", choices=["multi", "cosan"], help="multi or cosan")

    # Attack parameters
    parser.add_argument('--alpha', default=2., type=float)
    parser.add_argument('--epsilon', default=8., type=float)
    parser.add_argument('--attack_iters', default=10, type=int)
    parser.add_argument('--beta', default=6, type=int)
    parser.add_argument('--class_ae', dest='class_ae', action='store_true')
    parser.set_defaults(class_ae=False)

    # Misc
    parser.add_argument('--num_workers', default=36, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    main(get_args())