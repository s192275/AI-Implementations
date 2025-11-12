# Argüman bazlı yapabilmek için
import argparse

# Modeldeki bazı yapılandırmalar için
import torch 
from torch import nn 
import torch.nn.functional as F

#Veri seti işlemleri için
from torch.utils.data import random_split, DataLoader

# Görüntü işlemleri, veri seti ve modeller için
import torchvision 
from torchvision import transforms
from torchvision import models
from torchvision.datasets import CIFAR10

# Pytorch Lightning -> callback, earlystopping ve loglama için
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import DeviceStatsMonitor, LearningRateMonitor, EarlyStopping, ModelCheckpoint
from lightning.pytorch.strategies import FSDPStrategy, DDPStrategy

# Scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

#Zaman farkını tespit etmek için
from datetime import datetime

#Uyarıları görmezden gelmek için
import warnings

warnings.filterwarnings("ignore")

# Transform işlemi
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

# Model mimarisi
class ResNet(L.LightningModule):
    def __init__(self, num_classes: int = 10, optimizer_name: str = 'sgd') -> None:
        super(ResNet, self).__init__()
        self.save_hyperparameters() #optimizer_name dahil her şeyi kaydetmek için
        weights_path = './model/resnet_50_model.pth'
        self.model = models.resnet50(pretrained=False)
        self.model.load_state_dict(torch.load(weights_path))
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x) -> torch.Tensor:
        x = self.model(x)
        return x

    def configure_optimizers(self) -> torch.optim.Adam:
        optimizer_name = self.hparams.optimizer_name.lower()
        
        if optimizer_name == "sgd-momentum":
            optimizer = torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        elif optimizer_name == "rmsprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=1e-3)
        else:
            raise ValueError(f"Desteklenmeyen optimizasyon türü...Lütfen sgd, adam {opt_name}")
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',  # 'step' olarak da ayarlanabilir
                'frequency': 1,
                'monitor': 'val_loss',
                'reduce_on_plateau': True
            }
        }

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        outputs = self.model(x)
        pred = torch.argmax(outputs, dim = 1)
        acc = (pred == y).float().mean()
        loss = F.cross_entropy(outputs, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        outputs = self.model(x)
        pred = torch.argmax(outputs, dim = 1)
        acc = (pred == y).float().mean()
        loss = F.cross_entropy(outputs, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        outputs = self.model(x)
        pred = torch.argmax(outputs, dim = 1)
        acc = (pred == y).float().mean()
        loss = F.cross_entropy(outputs, y)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("test_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=2, type=int, metavar='N',
                        help='number of GPUs per node')
    parser.add_argument('--nodes', default=2, type=int, metavar='N',
                        help='number of nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='maximum number of epochs to run')
    parser.add_argument('--batch_size', default=128, type=int, metavar='N',
                        help='the batch size')
    parser.add_argument('--accelerator', default='gpu', type=str,
                        help='accelerator to use')
    parser.add_argument('--strategy', default='ddp', type=str,
                        help='distributed strategy to use')
    parser.add_argument('--optimizer', default = 'SGD', type = str,
                        help = 'Optimizer to use')

    args = parser.parse_args()

    print("Using PyTorch {} and Lightning {}".format(torch.__version__, L.__version__))
    full_train_dataset = CIFAR10('./data', train=True, download=False, transform=transform)

    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    test_dataset = CIFAR10('./data', train=False, download=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=2, pin_memory=False)

    model = ResNet(optimizer_name = args.optimizer)

    if args.strategy == "ddp":
        strategy = DDPStrategy()
        
    elif args.strategy == "fsdp":
        strategy = FSDPStrategy(
            sharding_strategy="FULL_SHARD",
            cpu_offload=False
        )

    elif args.strategy == "fsdp1":
        strategy = FSDPStrategy(
            sharding_strategy="SHARD_GRAD_OP",
            cpu_offload=False
        )

    elif args.strategy == "fsdp2":
        strategy = FSDPStrategy(
            sharding_strategy="NO_SHARD",
            cpu_offload=False
        )

    else:
        strategy = args.strategy

    logger = TensorBoardLogger(save_dir="./lightning_logs/", name="cifar10_p100", default_hp_metric=False)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = L.Trainer(
        devices=args.gpus,
        num_nodes=args.nodes,
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        strategy=strategy,
        logger=logger,
        callbacks=[lr_monitor],
        log_every_n_steps=20
    )

    t0 = datetime.now()
    trainer.fit(model, train_loader, val_loader)
    dt = datetime.now() - t0
    print('Training took {}'.format(dt))

    print("Running test evaluation...")
    test_results = trainer.test(model, test_loader)
    print(f"Test results: {test_results}")
    
    logger.log_hyperparams(
        {
            "batch_size": args.batch_size,
            "optimizer": args.optimizer,
            "epochs": args.epochs,
            "model_type": "ResNet", #Modelin ne olduğunu görüntülemek için
            "gpus": args.gpus,
            "nodes": args.nodes,
            "strategy": args.strategy
        },
        metrics={
            "test_loss": test_results[0]["test_loss_epoch"],
            "test_acc.": test_results[0]["test_acc_epoch"]
        }
    )

if __name__ == '__main__':
    main()

