import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import argparse
from tqdm import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser(description='CIFAR-10 Training')
    parser.add_argument('--device', default='cuda', help='device to use for training')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size per GPU')
    parser.add_argument('--epochs', default=50, type=int, help='number of epochs')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--output-dir', default='./output', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--amp', action='store_true', help='use AMP training')
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank. Passed by torchrun.")
    return parser

def main(args):
    # Set up distributed training if using multiple GPUs
    is_distributed = torch.cuda.is_available() and torch.cuda.device_count() > 1
    if is_distributed:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(args.local_rank)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Data augmentation and normalization for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform_test)
    
    train_sampler = None
    if dist.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)

    train_loader = DataLoader(trainset, batch_size=args.batch_size,
                            shuffle=(train_sampler is None),
                            num_workers=args.workers,
                            sampler=train_sampler)

    test_loader = DataLoader(testset, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.workers)

    # Create model
    model = torchvision.models.resnet18(num_classes=10)
    model = model.to(device)
    
    if is_distributed:
        model = DDP(model, device_ids=[args.local_rank])

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                         momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1

    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Initialize scaler for AMP
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
            
        # Train
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        train_iter = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for inputs, targets in train_iter:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            train_iter.set_postfix({
                'loss': train_loss/total,
                'acc': 100.*correct/total
            })

        scheduler.step()

        # Test
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        print(f'Epoch {epoch+1}: Test Acc: {100.*correct/total:.2f}%')

        # Save checkpoint
        if args.output_dir:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'checkpoint.pth'))

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)