import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from PIL import Image
from config_manager import ConfigManager

def downstream_main(config: ConfigManager):
    """
    執行下游任務訓練:
    - 載入預訓練 encoder 權重
    - 訓練分類或回歸頭
    - 使用 early stopping 並保存最佳模型
    - 記錄 train/val loss 及 F1/MAE 到 CSV
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print("\n✅ 下游訓練任務開始！")
    print(f"Using device: {device}")

    # 參數
    task        = config.get('task', 'age')
    num_epochs  = config.get('training.epochs', 100)
    batch_size  = config.get('data.batch_size', 64)
    lr          = config.get('training.learning_rate', 1e-3)
    weight_decay= config.get('training.weight_decay', 1e-4)
    use_slice   = config.get('data.slice', 'all')

    # 資料增強 & Dataset
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    class OCTFilenameDataset(Dataset):
        def __init__(self, root_dir, task='age', transform=None, use_slice='all'):
            self.transform = transform
            self.task = task
            self.data = []
            for subdir in os.listdir(root_dir):
                full_dir = os.path.join(root_dir, subdir)
                if not os.path.isdir(full_dir): continue
                for file in os.listdir(full_dir):
                    if not file.endswith('.png'): continue
                    parts = file.split('_')
                    if len(parts) < 5: continue
                    gender   = int(parts[0])
                    slice_id = int(parts[2])
                    age      = int(parts[3])
                    if use_slice!='all' and slice_id!=int(use_slice):
                        continue
                    label = gender if task=='gender' else age
                    self.data.append((os.path.join(full_dir, file), label))
        def __len__(self): return len(self.data)
        def __getitem__(self, idx):
            path, label = self.data[idx]
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label

    root = config.get('data.downstream_dir', './data/downstream')
    dataset = OCTFilenameDataset(root, task=task, transform=transform, use_slice=use_slice)
    train_set, val_set = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=config.get('base.num_workers',2), pin_memory=use_cuda)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=config.get('base.num_workers',2), pin_memory=use_cuda)

    # Model 定義
    class DownstreamNet(nn.Module):
        def __init__(self, task='age'):
            super().__init__()
            self.backbone = nn.Sequential(*list(resnet18(weights=None).children())[:-1])
            self.feature_dim = 512
            
            if config.get('base.use_custom_weight', False):
                path = config.get('base.custom_weight_path')
                if os.path.isfile(path):
                    print(f"Loading pretrained encoder: {path}")
                    state = torch.load(path, map_location='cpu')
                    self.backbone.load_state_dict(state['features'], strict=False)

            if task=='gender':
                self.head = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(self.feature_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 2)
                )
            else:
                self.head = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(self.feature_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1)
                )
        def forward(self, x):
            f = self.backbone(x).flatten(1)
            return self.head(f)

    model     = DownstreamNet(task=task).to(device)
    criterion = nn.CrossEntropyLoss() if task=='gender' else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val   = float('inf')
    best_state = None
    best_epoch = 0
    history    = []

    print(f"\nTraining {task} for {num_epochs} epochs...\n")
    for epoch in range(1, num_epochs+1):
        # --- TRAIN ---
        model.train()
        tr_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if task!='gender':
                labels = labels.float().unsqueeze(1)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        # Train metric
        model.eval()
        train_preds, train_lbls = [], []
        with torch.no_grad():
            for imgs, labels in train_loader:
                imgs = imgs.to(device)
                if task!='gender':
                    labels = labels.to(device).float().unsqueeze(1)
                else:
                    labels = labels.to(device)
                out = model(imgs)
                if task=='gender':
                    preds = out.argmax(1)
                else:
                    preds = out.squeeze()
                train_preds.extend(preds.cpu().tolist())
                train_lbls.extend(labels.cpu().tolist())
        train_metric = (f1_score(train_lbls, train_preds, average='macro')*100
                        if task=='gender'
                        else mean_absolute_error(train_lbls, train_preds))

        # --- VAL ---
        val_loss = 0.0
        val_preds, val_lbls = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                if task!='gender':
                    labels = labels.to(device).float().unsqueeze(1)
                else:
                    labels = labels.to(device)
                out = model(imgs)
                val_loss += criterion(out, labels).item()
                if task=='gender':
                    preds = out.argmax(1)
                else:
                    preds = out.squeeze()
                val_preds.extend(preds.cpu().tolist())
                val_lbls.extend(labels.cpu().tolist())
        val_loss /= len(val_loader)
        val_metric = (f1_score(val_lbls, val_preds, average='macro')*100
                      if task=='gender'
                      else mean_absolute_error(val_lbls, val_preds))

        # Early stopping
        if val_loss < best_val:
            best_val   = val_loss
            best_state = model.state_dict()
            best_epoch = epoch

        history.append((tr_loss, train_metric, val_loss, val_metric))
        print(
            f"Epoch {epoch}/{num_epochs}  "
            f"Train Loss: {tr_loss:.4f}, Train {'F1' if task=='gender' else 'MAE'}: {train_metric:.2f}  |  "
            f"Val Loss:   {val_loss:.4f}, Val   {'F1' if task=='gender' else 'MAE'}: {val_metric:.2f}"
        )
        scheduler.step()

    # 儲存最佳模型
    exp      = config.experiment_dir
    best_dir = os.path.join(exp, config.get('base.best_model_dir','best_model'))
    os.makedirs(best_dir, exist_ok=True)
    torch.save(best_state, os.path.join(best_dir, 'best_model.pth'))
    print(f"\n✅ Saved best downstream model (epoch {best_epoch}) to {best_dir}/best_model.pth")

    # 儲存日誌 CSV
    df = pd.DataFrame(history, columns=['train_loss','train_metric','val_loss','val_metric'])
    df.to_csv(os.path.join(exp, f"{task}_training_log.csv"), index_label='epoch')
    print(f"✅ Training log saved to {exp}/{task}_training_log.csv")