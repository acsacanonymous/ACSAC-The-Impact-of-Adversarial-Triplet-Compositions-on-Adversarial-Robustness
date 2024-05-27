import argparse
import logging
import os
import shutil

import numpy as np
from torch.backends import cudnn

from root import ROOT_DIR
from backbone.iresnet import iresnet50
from backbone.wide_resnet_silu import wideresnetwithswish
from backbone.wideresnet import WRN
from mining.adv_triplet_mining_cos import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from adversarial.standard_ae_classifier import attack_pgd
from benchmark.adversarial import evaluate_pgd
from data_loading.data_loaders import get_data_loaders, get_face_loaders

from torch_ema import ExponentialMovingAverage

def main(args):

    # Thanks a lot PIL
    logging.getLogger("PIL.PngImagePlugin").setLevel(logging.CRITICAL + 1)

    margin = args.margin
    mining_strategy = args.mining

    atn = args.atn
    nta = args.nta
    triplet_weight = args.triplet_w
    ce_w = args.ce_w
    ace_w = args.ace_w

    print(atn, nta)
    if not atn and not nta:
        raise ValueError("No triplet selected")

    # Args parsing for run
    save_dir = os.path.join(args.save_prefix, args.save_dir)
    dataset = args.dataset
    batch_size = args.batch_size

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    log_file = os.path.join(save_dir, "00000_log.log") # 00000 : first name in folder
    print(f"Log file at: {log_file}")
    if os.path.exists(log_file):
        os.remove(log_file)
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

    criterion_benign = nn.CrossEntropyLoss()
    criterion_adv = nn.CrossEntropyLoss()

    soft_triplets = []
    semihard_triplets = []
    hard_triplets = []
    triplet_losses = []

    best_rob_acc = 0.0
    best_clean_acc = 0.0

    for epoch in range(args.epochs):
        train_loss = 0
        loss_triplet_tot = 0
        sum_loss_adv = 0
        sum_loss_benign = 0
        train_n = 0
        correct_clean = 0
        correct_adv = 0

        no_triplets = 1

        model.train()
        print("---Entering training")
        logging.info("---Entering training")
        for i, (X, y) in enumerate(train_loader):

            opt.zero_grad()

            y, X = y.to(device), X.to(device)

            outputs_benign, features_benign = model(X)

            loss_benign = criterion_benign(outputs_benign, y)

            embeddings_benign = F.normalize(features_benign)

            X_adv = attack_pgd(model, X, y, epsilon, alpha, args.attack_iters)

            outputs_adv, adv_features = model(X_adv)
            adv_embeddings = F.normalize(adv_features)
            loss_adv = criterion_adv(outputs_adv, y)

            _, predicted = outputs_adv.max(1)
            predicted_classes = predicted.detach().cpu().numpy()

            real_classes = y.detach().cpu().numpy()

            predicted_classes = [predicted_classes[i] if predicted_classes[i] != real_classes[i] else -1 for i in
                                 range(len(predicted_classes))]
            predicted = torch.tensor(predicted_classes, device=y.device)

            if atn:
                if mining_strategy == "batch_hard":
                    loss_triplet, num_positive_triplets = mine_adversarial_triplets_batch_hard_atn(y, predicted,
                                                                                                   embeddings_benign,
                                                                                                   adv_embeddings,
                                                                                                   margin)
                elif mining_strategy == "all":
                    loss_triplet, num_positive_triplets = mine_adversarial_triplets_all_atn(y, predicted,
                                                                                            embeddings_benign,
                                                                                            adv_embeddings, margin)
                else:
                    raise ValueError("Incorrect config")
            elif nta:
                if mining_strategy == "batch_hard":
                    loss_triplet, num_positive_triplets = mine_adversarial_triplets_batch_hard_nta(y, predicted,
                                                                                                   embeddings_benign,
                                                                                                   adv_embeddings,
                                                                                                   margin)
                elif mining_strategy == "all":
                    loss_triplet, num_positive_triplets = mine_adversarial_triplets_all_nta(y, predicted,
                                                                                            embeddings_benign,
                                                                                            adv_embeddings, margin)
                else:
                    raise ValueError("Incorrect config")
            else:
                raise ValueError("Incorrect config")

            loss = ce_w * loss_benign + ace_w * loss_adv + triplet_weight * loss_triplet

            opt.zero_grad()

            loss.backward()

            opt.step()

            if epoch > 2:
                ema.update()

            train_loss += loss.item() * y.size(0)
            train_n += y.size(0)

            no_triplets += num_positive_triplets

            loss_triplet_tot += loss_triplet * num_positive_triplets

            sum_loss_adv += loss_adv.item() * y.size(0)
            sum_loss_benign += loss_benign.item() * y.size(0)

            _, predicted_benign = outputs_benign.max(1)
            correct_clean += predicted_benign.eq(y).sum().item()

            _, predicted_adv = outputs_adv.max(1)
            correct_adv += predicted_adv.eq(y).sum().item()


            if (i + 1) % 10 == 0:
                lr = opt.param_groups[0]['lr']
                print(f"{epoch} \t {lr} \t {train_n} \t {train_loss / train_n} \t {correct_clean / train_n} \t"
                      f"{correct_adv / train_n} \t"
                      f"{no_triplets} \t "
                      f"{no_triplets / train_n} \t "
                      f"{sum_loss_adv / train_n} \t "
                      f"{sum_loss_benign / train_n} \t "
                      f"{loss_triplet_tot / no_triplets} \t "
                      )
                logging.info(f"{epoch} \t {lr} \t {train_n} \t {train_loss / train_n} \t {correct_clean / train_n} \t"
                             f"{correct_adv / train_n} \t"
                             f"{no_triplets} \t "
                             f"{no_triplets / train_n} \t "
                             f"{sum_loss_adv / train_n} \t "
                             f"{sum_loss_benign / train_n} \t "
                             f"{loss_triplet_tot / no_triplets} \t "
                             )

        model.eval()

        correct_test = 0
        total = 0

        epoch_soft = 0
        epoch_semihard = 0
        epoch_hard = 0
        triplet_loss = 0
        no_triplets = 0.00001

        if epoch > 2:
            with ema.average_parameters():
                with torch.no_grad():
                    for i, (X, y) in enumerate(test_loader):
                        y, X = y.to(device), X.to(device)

                        # calculate outputs by running images through the network
                        outputs_benign, features_benign = model(X)
                        # the class with the highest energy is what we choose as prediction
                        _, predicted = torch.max(outputs_benign.data, 1)
                        total += y.size(0)
                        correct_test += (predicted == y).sum().item()

                        X_adv = attack_pgd(model, X, y, epsilon, alpha, args.attack_iters)

                        outputs_adv, adv_features = model(X_adv)
                        embeddings_benign = F.normalize(features_benign)
                        adv_embeddings = F.normalize(adv_features)

                        _, predicted = outputs_adv.max(1)
                        predicted_classes = predicted.detach().cpu().numpy()

                        real_classes = y.detach().cpu().numpy()

                        predicted_classes = [predicted_classes[i] if predicted_classes[i] != real_classes[i] else -1 for
                                             i in
                                             range(len(predicted_classes))]
                        predicted = torch.tensor(predicted_classes, device=y.device)

                        if atn:
                            loss_triplet, num_positive_triplets = mine_adversarial_triplets_all_atn(y, predicted,
                                                                                                    embeddings_benign,
                                                                                                    adv_embeddings,
                                                                                                    margin)
                            n_soft, n_semihard, n_hard = count_adversarial_triplets_all_atn(y, predicted,
                                                                                            embeddings_benign,
                                                                                            adv_embeddings, margin)
                        elif nta:
                            loss_triplet, num_positive_triplets = mine_adversarial_triplets_all_nta(y, predicted,
                                                                                                    embeddings_benign,
                                                                                                    adv_embeddings,
                                                                                                    margin)
                            n_soft, n_semihard, n_hard = count_adversarial_triplets_all_nta(y, predicted,
                                                                                            embeddings_benign,
                                                                                            adv_embeddings, margin)
                        else:
                            raise ValueError("No ATN or NTA")

                        epoch_soft += n_soft
                        epoch_semihard += n_semihard
                        epoch_hard += n_hard

                        triplet_loss += loss_triplet.item() * num_positive_triplets
                        no_triplets += num_positive_triplets

                print(f'Test accuracy: {100 * correct_test / total} %')
                logging.info(f'Test accuracy: {100 * correct_test / total} %')

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
                    if correct_test / total > best_clean_acc:
                        best_clean_acc = correct_test / total
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
                    outputs_benign, features_benign = model(X)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs_benign.data, 1)
                    total += y.size(0)
                    correct_test += (predicted == y).sum().item()

                    X_adv = attack_pgd(model, X, y, epsilon, alpha, args.attack_iters)

                    outputs_adv, adv_features = model(X_adv)
                    embeddings_benign = F.normalize(features_benign)
                    adv_embeddings = F.normalize(adv_features)

                    _, predicted = outputs_adv.max(1)
                    predicted_classes = predicted.detach().cpu().numpy()

                    real_classes = y.detach().cpu().numpy()

                    predicted_classes = [predicted_classes[i] if predicted_classes[i] != real_classes[i] else -1 for
                                         i in
                                         range(len(predicted_classes))]
                    predicted = torch.tensor(predicted_classes, device=y.device)

                    if atn:
                        loss_triplet, num_positive_triplets = mine_adversarial_triplets_all_atn(y, predicted,
                                                                                                embeddings_benign,
                                                                                                adv_embeddings,
                                                                                                margin)
                        n_soft, n_semihard, n_hard = count_adversarial_triplets_all_atn(y, predicted,
                                                                                        embeddings_benign,
                                                                                        adv_embeddings, margin)
                    elif nta:
                        loss_triplet, num_positive_triplets = mine_adversarial_triplets_all_nta(y, predicted,
                                                                                                embeddings_benign,
                                                                                                adv_embeddings,
                                                                                                margin)
                        n_soft, n_semihard, n_hard = count_adversarial_triplets_all_nta(y, predicted,
                                                                                        embeddings_benign,
                                                                                        adv_embeddings, margin)
                    else:
                        raise ValueError("No ATN or NTA")

                    epoch_soft += n_soft
                    epoch_semihard += n_semihard
                    epoch_hard += n_hard

                    triplet_loss += loss_triplet.item() * num_positive_triplets
                    no_triplets += num_positive_triplets

            print(f'Test accuracy: {100 * correct_test / total} %')
            logging.info(f'Test accuracy: {100 * correct_test / total} %')

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
                if correct_test / total > best_clean_acc:
                    best_clean_acc = correct_test / total
                    best_rob_acc = rob_acc

                    full_model_save_path = os.path.join(save_dir, f"best_model.pth")
                    torch.save(model.state_dict(), full_model_save_path)
                    print(f"Model saved at: {full_model_save_path} at epoch {epoch}")
                    logging.info(f"Model saved at: {full_model_save_path} at epoch {epoch}")


        mean_triplet_loss = triplet_loss / no_triplets

        soft_triplets.append(epoch_soft)
        semihard_triplets.append(epoch_semihard)
        hard_triplets.append(epoch_hard)
        triplet_losses.append(mean_triplet_loss)
        print(epoch_soft, "\t", epoch_semihard, "\t", epoch_hard)
        print(mean_triplet_loss)

        scheduler.step()

    softs_file = os.path.join(save_dir, "adv_softs")
    shs_file = os.path.join(save_dir, "adv_semihards")
    hards_file = os.path.join(save_dir, "adv_hards")
    triplet_loss_file = os.path.join(save_dir, "adv_triplet_loss")

    logging.info(f"Softs: {soft_triplets}")
    logging.info(f"Semihards: {semihard_triplets}")
    logging.info(f"Hards: {hard_triplets}")
    logging.info(f"Triplet_loss: {triplet_losses}")

    softs_np = np.array(soft_triplets)
    shs_np = np.array(semihard_triplets)
    hard_np = np.array(hard_triplets)
    triplet_losses_np = np.array(triplet_losses)

    np.save(softs_file, softs_np, allow_pickle=True)
    np.save(shs_file, shs_np, allow_pickle=True)
    np.save(hards_file, hard_np, allow_pickle=True)
    np.save(triplet_loss_file, triplet_losses_np, allow_pickle=True)


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
    parser.add_argument('--batch_size', default=128, type=int)

    parser.add_argument('--opt', default="adam", type=str, choices=["adam", "sgd"], help="adam or sgd")
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument("--sched", default="multi", choices=["multi", "cosan"], help="multi or cosan")

    # Attack parameters
    parser.add_argument('--alpha', default=2., type=float)
    parser.add_argument('--epsilon', default=8., type=float)
    parser.add_argument('--attack_iters', default=10, type=int)

    # Triplet loss + flags
    parser.add_argument('--atn', dest='atn', action='store_true')
    parser.set_defaults(atn=False)

    parser.add_argument('--nta', dest='nta', action='store_true')
    parser.set_defaults(atn=False)

    parser.add_argument('--margin', default=0.6, type=float, help='Margin for triplet loss')
    parser.add_argument('--mining', default="all", type=str, choices=["all", "batch_hard"], help='Set mode')
    parser.add_argument('--triplet_w', default=1, type=float, help='Weight for triplet loss')
    parser.add_argument('--ace_w', default=0.2, type=float, help='Weight for adversarial crossentropy loss')
    parser.add_argument('--ce_w', default=0.2, type=float, help='Weight for crossentropy loss')

    # Misc
    parser.add_argument('--num_workers', default=36, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    main(get_args())