import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
from PIL import Image
from config_manager import ConfigManager
from tqdm import tqdm


class SimCLRDummyDataset(Dataset):
    """
    自監督 SimCLR 資料集，輸出一對增強視圖
    """
    def __init__(self, root_dir: str, transform=None):
        self.paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png')]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert('RGB')
        xi = self.transform(img)
        xj = self.transform(img)
        return xi, xj


class SimCLRModel(nn.Module):
    """
    SimCLR 模型: ResNet18 編碼器 + 投影頭
    """
    def __init__(self, out_dim: int = 128):
        super().__init__()
        backbone = resnet18(weights=None)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x).flatten(1)
        z = self.projector(h)
        return nn.functional.normalize(z, dim=1)


class NTXentLoss(nn.Module):
    """
    NT-Xent Loss for SimCLR contrastive learning
    """
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, zis: torch.Tensor, zjs: torch.Tensor) -> torch.Tensor:
        N = zis.size(0)
        z = torch.cat([zis, zjs], dim=0)
        sim = nn.functional.cosine_similarity(
            z.unsqueeze(1), z.unsqueeze(0), dim=2
        ) / self.temperature
        mask = torch.eye(2 * N, device=z.device).bool()
        sim = sim.masked_fill(mask, float('-inf'))
        labels = torch.arange(N, device=z.device)
        labels = torch.cat([labels + N, labels], dim=0)
        return nn.functional.cross_entropy(sim, labels)


def pretrain_main(config: ConfigManager) -> None:
    """
    執行 SimCLR 預訓練：
    - 只在每個 epoch 結束時輸出平均 loss
    - 最佳 encoder 權重保存在 base.best_model_dir
    - 訓練過程的 loss 日誌儲存到 pretrain_loss.csv
    """
    # 設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # 參數
    epochs = config.get('training.epochs', 100)
    batch_size = config.get('data.batch_size', 64) # 64
    num_workers = config.get('base.num_workers', 2)
    lr = config.get('training.learning_rate', 1e-3)
    weight_decay = config.get('training.weight_decay', 1e-6)
    temperature = config.get('training.temperature', 0.1) # 0.08, 0.05, 0.1
    out_dim = config.get('model.out_dim', 128)
    pretrain_dir = config.get('data.pretrain_dir', './data/pre_train')

    # 資料
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    dataset = SimCLRDummyDataset(pretrain_dir, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type=='cuda')
    )

    # 模型、優化器
    model = SimCLRModel(out_dim).to(device)
    criterion = NTXentLoss(temperature)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler() if torch.cuda.is_available() else None

    # 訓練
    best_loss = float('inf')
    losses = []
    best_path = None
    print(f"✅ 開始預訓練: epochs={epochs}, batch_size={batch_size}\n")
    for epoch in range(1, epochs+1):
        model.train()
        running = 0.0
        for xi, xj in loader:
            xi, xj = xi.to(device), xj.to(device)
            optimizer.zero_grad()
            if scaler:
                with autocast():
                    z1, z2 = model(xi), model(xj)
                    loss = criterion(z1, z2)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                z1, z2 = model(xi), model(xj)
                loss = criterion(z1, z2)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            running += loss.item()
        epoch_loss = running / len(loader)
        losses.append(epoch_loss)
        scheduler.step()
        print(f"[Epoch {epoch}/{epochs}] Avg Loss: {epoch_loss:.4f}")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_dir = os.path.join(config.experiment_dir, config.get('base.best_model_dir', 'best_model'))
            os.makedirs(best_dir, exist_ok=True)
            best_path = os.path.join(best_dir, 'encoder_best.pth')
            torch.save({'features': model.encoder.state_dict()}, best_path)

    # 儲存日誌
    csv_path = os.path.join(config.experiment_dir, 'pretrain_loss.csv')
    pd.DataFrame({'epoch': list(range(1, epochs+1)), 'loss': losses}).to_csv(csv_path, index=False)
    print(f"\n✅ 已儲存 Loss 日誌至: {csv_path}")
    if best_path:
        print(f"✅ 已儲存最佳 Encoder 權重至: {best_path}")