import os
import shutil
from subprocess import check_output, CalledProcessError, DEVNULL
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tqdm import tqdm

import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from constants import *
from model import *
from utils import *

ROOT = os.path.dirname(os.path.abspath(__file__))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def forward_batch(loader, model, criterion, scheduler, epoch: int, train: bool):
    """
    :param scheduler, epoch, train: For printing only.
    """
    name = "Train" if train else "Test"
    pbar = tqdm(enumerate(loader), total=len(loader), desc=name)
    # TODO: Custom forward pass code
    for i, (x, y) in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)
        loss = criterion(pred, y)

        # Set tqdm progress bar.
        lr = scheduler.get_last_lr()[0]
        pbar.set_description(
            f"{name:5}: Epoch {epoch+1:04d}/{EPOCHS:04d} | "
            f"Batch {i+1:06d}/{len(loader):06d} | "
            f"Loss {loss.item():.6f} | "
            f"LR {lr:.6f} "
        )
        yield loss


def create_logdir(logdir):
    """
    Returns summary writer.
    Also copies configuration and commit hash to directory.
    """
    print("Creating log directory")
    log = SummaryWriter(logdir)

    # Copy configuration
    shutil.copyfile(os.path.join(ROOT, "constants.py"), os.path.join(logdir, "constants.py"))

    # Copy commit hash
    try:
        commit = check_output(["git", "rev-parse", "HEAD"], stderr=DEVNULL).decode("utf-8").strip()
    except CalledProcessError:
        commit = "Unknown"
    with open(os.path.join(logdir, "commit.txt"), "w") as f:
        f.write(commit)
        f.write("\n")

    return log


def train(model, dataset, logdir, args):
    def create_loader(dataset):
        loader_args = {
            "batch_size": BATCH_SIZE,
            "shuffle": True,
            "num_workers": 4,
            "pin_memory": True,
        }
        if args.ddp:
            loader_args["sampler"] = DistributedSampler(dataset)
        return DataLoader(dataset, **loader_args)

    # Create data loaders.
    train_len = int(len(dataset) * 0.9)
    test_len = len(dataset) - train_len
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])
    train_loader = create_loader(train_dataset)
    test_loader = create_loader(test_dataset)
    print(f"Train set: {len(train_dataset)} batches")
    print(f"Test set: {len(test_dataset)} batches")

    # TODO loss and optim
    criterion = torch.nn.BCELoss()
    optim = torch.optim.Adam(model.parameters(), lr=LR_START)
    lr_decay_fac = (LR_END / LR_START) ** (1 / EPOCHS)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=lr_decay_fac)

    print(f"Tensorboard log directory is {logdir}")
    if args.info:
        print("--info flag is set. Not training.")
        return
    log = create_logdir(logdir)

    if args.ddp:
        print("Wrapping model in DDP")
        model = DDP(model, device_ids=[DEVICE])

    batch_num = 0
    for epoch in range(EPOCHS):
        # Train
        model.train()
        if args.ddp:
            train_loader.sampler.set_epoch(epoch)
        for loss in forward_batch(train_loader, model, criterion, scheduler, epoch, True):
            loss.backward()
            if (batch_num+1) % BATCH_PER_STEP == 0:
                clip_grad_norm_(model.parameters(), 0.5)
                optim.step()
                optim.zero_grad()

            log.add_scalar("Train loss", loss.item(), batch_num)
            log.add_scalar("LR", scheduler.get_last_lr()[0], batch_num)
            batch_num += 1

        # Test
        if args.ddp:
            test_loader.sampler.set_epoch(epoch)
        with torch.no_grad():
            model.eval()
            total_loss = 0
            for loss in forward_batch(test_loader, model, criterion, scheduler, epoch, False):
                total_loss += loss.item()
            avg_loss = total_loss / len(test_loader)
            log.add_scalar("Test loss", avg_loss, batch_num)

        scheduler.step()

        # Save model
        if (epoch+1) % SAVE_INTERVAL == 0 or epoch == EPOCHS-1:
            torch.save(model.state_dict(), os.path.join(logdir, f"epoch.{epoch+1}.pt"))

    log.close()


def main():
    parser = create_parser()
    parser.add_argument("--info", action="store_true", help="Only print session configuration.")
    parser.add_argument("--ddp", action="store_true", help="Use DistributedDataParallel.")
    args = parser.parse_args()

    os.makedirs(args.data, exist_ok=True)
    os.makedirs(args.runs, exist_ok=True)

    if args.ddp:
        # Init process group
        print(f"Using DDP. World size is {dist.get_world_size()}")
        print("Initializing process group")
        init_process_group(backend="nccl")

        # DDP rank
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        print(f"DDP rank: global {global_rank}, local {local_rank}")

        # Set device
        global DEVICE
        DEVICE = local_rank

    # TODO: Class names
    model = MyModel().to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dataset = MyDataset(args.data)
    logdir = get_new_run(args)

    print(f"Dataset: {len(dataset)} samples")
    print(f"Model: {num_params} learnable parameters")
    train(model, dataset, logdir, args)

    if args.ddp:
        print("Destroying process group")
        destroy_process_group()


if __name__ == "__main__":
    main()
