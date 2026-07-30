import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import time
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import torch.distributed as dist  # 추가
import csv

# We use this special print function to help assess your work.
# Please do not remove or modify.
from assessment_print import assessment_print


# Parse input arguments
parser = argparse.ArgumentParser(
    description='Workshop Assessment',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
    '--batch-size',
    type=int,
    default=32,
    help='input batch size for training'
)

parser.add_argument(
    '--epochs',
    type=int,
    default=40,
    help='number of epochs to train'
)

parser.add_argument(
    '--base-lr',
    type=float,
    default=0.01,
    help='learning rate for a single GPU'
)

parser.add_argument(
    '--target-accuracy',
    type=float,
    default=.85,
    help='Target accuracy to stop training'
)

parser.add_argument(
    '--patience',
    type=int,
    default=2,
    help='Number of epochs that meet target before stopping'
)

# 추가
parser.add_argument(
    '--num-nodes',
    type=int,
    default=1,
    help='Number of available nodes/hosts'
)

# 추가
parser.add_argument(
    '--node-id',
    type=int,
    default=0,
    help='Unique ID to identify the current node/host'
)

# 추가
parser.add_argument(
    '--num-gpus',
    type=int,
    default=1,
    help='Number of GPUs in each node'
)

args = parser.parse_args()


# 추가
WORLD_SIZE = args.num_gpus * args.num_nodes

# 추가
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '9956'


# Standard convolution block followed by batch normalization
class cbrblock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(cbrblock, self).__init__()

        self.cbr = nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                stride=(1, 1),
                padding='same',
                bias=False
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.cbr(x)
        return out


# Basic residual block
class conv_block(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        scale_input
    ):
        super(conv_block, self).__init__()

        self.scale_input = scale_input

        if self.scale_input:
            self.scale = nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=1,
                stride=(1, 1),
                padding='same'
            )

        self.layer1 = cbrblock(
            input_channels,
            output_channels
        )

        self.dropout = nn.Dropout(p=0.01)

        self.layer2 = cbrblock(
            output_channels,
            output_channels
        )

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.dropout(out)
        out = self.layer2(out)

        if self.scale_input:
            residual = self.scale(residual)

        out = out + residual

        return out


# Overall network
class WideResNet(nn.Module):
    def __init__(self, num_classes):
        super(WideResNet, self).__init__()

        # CIFAR-10은 RGB 이미지이므로 입력 채널 3 유지
        nChannels = [3, 16, 160, 320, 640]

        self.input_block = cbrblock(
            nChannels[0],
            nChannels[1]
        )

        self.block1 = conv_block(
            nChannels[1],
            nChannels[2],
            1
        )

        self.block2 = conv_block(
            nChannels[2],
            nChannels[2],
            0
        )

        self.pool1 = nn.MaxPool2d(2)

        self.block3 = conv_block(
            nChannels[2],
            nChannels[3],
            1
        )

        self.block4 = conv_block(
            nChannels[3],
            nChannels[3],
            0
        )

        self.pool2 = nn.MaxPool2d(2)

        self.block5 = conv_block(
            nChannels[3],
            nChannels[4],
            1
        )

        self.block6 = conv_block(
            nChannels[4],
            nChannels[4],
            0
        )

        self.pool = nn.AvgPool2d(7)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(
            nChannels[4],
            num_classes
        )

    def forward(self, x):
        out = self.input_block(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.pool1(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.pool2(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.pool(out)
        out = self.flat(out)
        out = self.fc(out)

        return out


def train(
    model,
    optimizer,
    train_loader,
    loss_fn,
    device
):
    total_labels = 0
    correct_labels = 0

    model.train()

    for images, labels in train_loader:
        labels = labels.to(device)
        images = images.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = torch.max(outputs, 1)[1]

        total_labels += len(labels)
        correct_labels += (
            predictions == labels
        ).sum()

    t_accuracy = correct_labels / total_labels

    return t_accuracy


def test(
    model,
    test_loader,
    loss_fn,
    device
):
    total_labels = 0
    correct_labels = 0
    loss_total = 0

    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            labels = labels.to(device)
            images = images.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            predictions = torch.max(
                outputs,
                1
            )[1]

            total_labels += len(labels)

            correct_labels += (
                predictions == labels
            ).sum()

            loss_total += loss

    v_accuracy = correct_labels / total_labels
    v_loss = loss_total / len(test_loader)

    return v_accuracy, v_loss


# 수정: 기존 main 학습 코드를 worker 함수로 이동
def worker(local_rank, args):
    # 추가: global rank 계산
    global_rank = (
        args.node_id * args.num_gpus
        + local_rank
    )

    # 추가: 현재 프로세스를 해당 GPU에 고정
    torch.cuda.set_device(local_rank)

    # 추가: 분산 프로세스 그룹 초기화
    dist.init_process_group(
        backend='nccl',
        world_size=WORLD_SIZE,
        rank=global_rank
    )

    # Load and augment the data with a set of transformations
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),

        transforms.RandomAffine(
            0,
            shear=10,
            scale=(0.8, 1.2)
        ),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),

        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    # 추가: local rank 0만 다운로드
    download = True if local_rank == 0 else False

    # 수정: FashionMNIST가 아니라 CIFAR10 사용
    if local_rank == 0:
        train_set = torchvision.datasets.CIFAR10(
            "./data",
            train=True,
            download=download,
            transform=transform_train
        )

        test_set = torchvision.datasets.CIFAR10(
            "./data",
            train=False,
            download=download,
            transform=transform_test
        )

    # 추가: 다운로드 완료까지 다른 프로세스 대기
    dist.barrier()

    # 수정: 나머지 프로세스는 다운로드 없이 CIFAR10 로드
    if local_rank != 0:
        train_set = torchvision.datasets.CIFAR10(
            "./data",
            train=True,
            download=download,
            transform=transform_train
        )

        test_set = torchvision.datasets.CIFAR10(
            "./data",
            train=False,
            download=download,
            transform=transform_test
        )

    # 추가: 분산 학습용 sampler 생성
    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(
            train_set,
            num_replicas=WORLD_SIZE,
            rank=global_rank
        )
    )

    # 추가
    test_sampler = (
        torch.utils.data.distributed.DistributedSampler(
            test_set,
            num_replicas=WORLD_SIZE,
            rank=global_rank
        )
    )

    # 수정: train_sampler 연결
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        drop_last=True,
        sampler=train_sampler
    )

    # 수정: test_sampler 연결
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        drop_last=True,
        sampler=test_sampler
    )

    num_classes = 10

    # 수정: 프로세스마다 local_rank에 해당하는 GPU 사용
    device = torch.device(
        "cuda:" + str(local_rank)
    )

    model = WideResNet(
        num_classes
    ).to(device)

    # 추가: 모델을 DistributedDataParallel로 감싸기
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank]
    )

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9
    )

    val_accuracy = []
    total_time = 0

    for epoch in range(args.epochs):
        # 추가: epoch마다 sampler의 셔플 순서 변경
        train_sampler.set_epoch(epoch)

        t0 = time.time()

        t_accuracy = train(
            model,
            optimizer,
            train_loader,
            loss_fn,
            device
        )

        # 추가: 모든 프로세스가 학습을 끝낼 때까지 대기
        dist.barrier()

        epoch_time = time.time() - t0
        total_time += epoch_time

        # 수정: 처리량을 GPU Tensor로 변환
        images_per_sec = torch.tensor(
            len(train_loader)
            * args.batch_size
            / epoch_time
        ).to(device)

        # 추가: 모든 GPU 처리량을 global rank 0에 합산
        torch.distributed.reduce(
            images_per_sec,
            0
        )

        # 추가: 각 GPU의 학습 정확도 평균
        torch.distributed.all_reduce(
            t_accuracy,
            op=dist.ReduceOp.AVG
        )

        v_accuracy, v_loss = test(
            model,
            test_loader,
            loss_fn,
            device
        )

        # 추가: 각 GPU의 검증 정확도 평균
        torch.distributed.all_reduce(
            v_accuracy,
            op=dist.ReduceOp.AVG
        )

        # 추가: 각 GPU의 검증 손실 평균
        torch.distributed.all_reduce(
            v_loss,
            op=dist.ReduceOp.AVG
        )

        val_accuracy.append(v_accuracy)

        # 수정: global rank 0만 출력
        if global_rank == 0:
            # We use this special print function to help assess your work.
            # Please do not remove or modify.
            assessment_print(
                "Epoch = {:2d}: Cumulative Time = {:5.3f}, Epoch Time = {:5.3f}, Images/sec = {}, Training Accuracy = {:5.3f}, Validation Loss = {:5.3f}, Validation Accuracy = {:5.3f}".format(
                    epoch + 1,
                    total_time,
                    epoch_time,
                    images_per_sec,
                    t_accuracy,
                    v_loss,
                    val_accuracy[-1]
                )
            )

        if (
            len(val_accuracy) >= args.patience
            and all(
                acc >= args.target_accuracy
                for acc in val_accuracy[-args.patience:]
            )
        ):
            # 수정: global rank 0만 조기 종료 메시지 출력
            if global_rank == 0:
                # We use this special print function to help assess your work.
                # Please do not remove or modify.
                assessment_print(
                    'Early stopping after epoch {}'.format(
                        epoch + 1
                    )
                )

            # 모든 프로세스가 동일한 조건으로 반복 종료
            break

    # 추가: 분산 프로세스 그룹 종료
    dist.destroy_process_group()


# 추가: GPU 수만큼 worker 프로세스 생성
if __name__ == '__main__':
    torch.multiprocessing.spawn(
        worker,
        nprocs=args.num_gpus,
        args=(args,)
    )