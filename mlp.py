import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from coincurve import PrivateKey, PublicKey
import os
import time
import random
import argparse
import signal
import numpy as np
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from multiprocessing import Pool, cpu_count

# ======================
# 全局配置
# ======================
class Config:
    model_dir = "saved_models"
    dataset_cache = "keypairs.bin"
    checkpoint_file = "checkpoint.pt"
    batch_size = 4096
    num_workers = min(cpu_count(), 8)
    feature_bytes  = 64
    group_size = 20  # 每组20个相同奇偶性的密钥对

# ======================
# 椭圆曲线工具类
# ======================
class ECCUtils:
    @staticmethod
    def generate_2G():
        """生成2G点（正确格式）"""
        G = PublicKey.from_secret(b'\x00'*31 + b'\x01')
        return G.combine([G]).format(compressed=False)  # 保留完整格式

    @staticmethod
    def point_add(pub_key_bytes, point_bytes):
        """安全的椭圆曲线点加运算"""
        try:
            # 添加未压缩标识字节
            pub = PublicKey(b'\x04' + pub_key_bytes)
            point = PublicKey(b'\x04' + point_bytes)
            combined = pub.combine([point])
            return combined.format(compressed=False)[1:]  # 去掉前缀
        except Exception as e:
            print(f"点加运算失败: {str(e)}")
            return None

class Validator:
    def __init__(self, model_path):
        self.model = DeepParityMLP()
        self.model.load_state_dict(torch.load(model_path)['model'])
        self.model.eval()
        self.twoG = ECCUtils.generate_2G()[1:]  # 去掉前缀字节

    def _parse_public_key(self, pub_str):
        """正确解析公钥字符串"""
        try:
            x_str, y_str = pub_str.split(',')
            x = int(x_str, 16)
            y = int(y_str, 16)
            return x.to_bytes(32, 'big') + y.to_bytes(32, 'big')
        except ValueError:
            raise ValueError("公钥格式应为'x_hex,y_hex'")

    def predict_parity(self, pub_bytes):
        """预测公钥字节的奇偶性"""
        # 生成特征（与训练一致）
        features = torch.from_numpy(
            np.unpackbits(np.frombuffer(pub_bytes, dtype=np.uint8))
        ).float()
        
        with torch.no_grad():
            output = self.model(features.unsqueeze(0))
            prob = torch.sigmoid(output).item()
        return '奇数' if prob > 0.5 else '偶数', abs(prob - 0.5)*2  # 置信度[0-1]

    def generate_sequence(self, base_pub_str, length=100):
        """生成并验证连续公钥序列"""
        current_pub = self._parse_public_key(base_pub_str)
        predictions = []
        
        for _ in tqdm(range(length), desc="生成验证序列"):
            # 预测当前公钥
            parity, confidence = self.predict_parity(current_pub)
            predictions.append((current_pub.hex(), parity, confidence))
            
            # 生成下一个公钥
            next_pub = ECCUtils.point_add(current_pub, self.twoG)
            if not next_pub:
                print("遇到无效点，终止生成")
                break
            current_pub = next_pub
        
        # 分析结果
        if len(predictions) == 0:
            print("无有效预测结果")
            return
        
        print(f"\n验证结果（共{len(predictions)}个有效点）:")
        print(f"初始预测: {predictions[0][1]} (置信度: {predictions[0][2]:.2%})")
        print(f"最终预测: {predictions[-1][1]} (置信度: {predictions[-1][2]:.2%})")
        
        # 检查一致性
        unique_parities = len(set(p[1] for p in predictions))
        print(f"奇偶一致性: {'是' if unique_parities == 1 else '否'}")
        
        # 检查置信度
        avg_confidence = sum(p[2] for p in predictions) / len(predictions)
        print(f"平均置信度: {avg_confidence:.2%}")

# ======================
# 信号处理器
# ======================
class TrainingController:
    def __init__(self):
        self.should_stop = False
        signal.signal(signal.SIGINT, self.handler)
        signal.signal(signal.SIGTERM, self.handler)
    
    def handler(self, signum, frame):
        print(f"\n捕获信号 {signum}，正在终止训练...")
        self.should_stop = True

# ======================
# 数据生成模块
# ======================
class KeyPairGenerator:
    @staticmethod
    def _generate_group(_):
        """生成一组20个连续奇偶性的密钥对"""
        group = []
        base_priv = int.from_bytes(random.randbytes(32), 'big') % (2**256)
        
        for i in range(Config.group_size):
            priv_int = base_priv + 2 * i
            priv_key = PrivateKey(secret=priv_int.to_bytes(32, 'big'))
            pub = priv_key.public_key.format(compressed=False)
            
            x = int.from_bytes(pub[1:33], 'big')
            y = int.from_bytes(pub[33:65], 'big')
            label = priv_int % 2
            
            group.append((x, y, label))
        return group

    @classmethod
    def generate_dataset(cls, size):
        if os.path.exists(Config.dataset_cache):
            return

        total_groups = size // Config.group_size
        actual_size = total_groups * Config.group_size
        
        print(f"生成 {actual_size} 个密钥对（共 {total_groups} 组）...")
        start = time.time()
        
        memmap = np.memmap(Config.dataset_cache,
                          dtype=np.uint8,
                          mode='w+',
                          shape=(actual_size, Config.feature_bytes + 1))
        
        with Pool(processes=cpu_count()) as pool:
            chunk_size = 100
            total_chunks = (total_groups + chunk_size - 1) // chunk_size
            
            with tqdm(total=total_groups, desc="生成进度") as pbar:
                for chunk_idx in range(total_chunks):
                    start_group = chunk_idx * chunk_size
                    end_group = min((chunk_idx+1)*chunk_size, total_groups)
                    groups = pool.map(cls._generate_group, [None]*(end_group-start_group))
                    
                    # 写入内存映射
                    for group_idx, group in enumerate(groups):
                        for i_in_group, (x, y, label) in enumerate(group):
                            idx = (start_group + group_idx) * Config.group_size + i_in_group
                            
                            # 修正：直接存储字节数据
                            x_bytes = x.to_bytes(32, 'big')
                            y_bytes = y.to_bytes(32, 'big')
                            features = np.frombuffer(x_bytes + y_bytes, dtype=np.uint8)
                            
                            memmap[idx, :-1] = features
                            memmap[idx, -1] = label
                    
                    pbar.update(len(groups))
        
        print(f"数据集生成完成，耗时 {time.time()-start:.1f}秒")

# ======================
# 数据加载模块
# ======================
class BinaryKeyDataset(Dataset):
    def __init__(self, size=10**6):
        self.actual_size = (size // Config.group_size) * Config.group_size
        self.memmap = np.memmap(Config.dataset_cache,
                               dtype=np.uint8,
                               mode='r',
                               shape=(self.actual_size, Config.feature_bytes + 1))
    
    def __len__(self):
        return self.actual_size
    
    def __getitem__(self, idx):
        data = self.memmap[idx]
        # 修正：在加载时转换为二进制位
        features = torch.from_numpy(
            np.unpackbits(data[:-1])  # 将64字节转换为512位
        ).float()
        label = torch.tensor(data[-1], dtype=torch.float32)
        return features, label

# ======================
# 深度MLP网络
# ======================
class DeepParityMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1))
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, a=0.1, nonlinearity='leaky_relu')
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.layers(x).squeeze()

# ======================
# 训练系统
# ======================
class TrainingSystem:
    def __init__(self, args):
        self.args = args
        self.controller = TrainingController()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化模型和优化器
        self.model, self.optimizer = self._load_or_create_model()
        self.scaler = GradScaler('cuda')
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 加载数据集
        if not os.path.exists(Config.dataset_cache):
            KeyPairGenerator.generate_dataset(args.dataset_size)
        
        full_dataset = BinaryKeyDataset(size=args.dataset_size)
        train_size = int(0.8 * len(full_dataset))
        self.train_loader = DataLoader(
            Subset(full_dataset, range(train_size)),
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
            pin_memory=True,
            persistent_workers=True
        )
        self.test_loader = DataLoader(
            Subset(full_dataset, range(train_size,len(full_dataset))),
            batch_size=Config.batch_size*2,
            num_workers=Config.num_workers,
            pin_memory=True
        )
        # 添加验证集路径
        self.validation_pub = "8fd74b41a5f5c775ea13b7617d7ffe871c0cbad1b7bb99bcea03dc47561feae4,dad89019b8f2e6990782b9ae4e74243b1ac2ec007d621642d507b1a844d3e05f"

    def _load_or_create_model(self):
        """加载已有模型或创建新模型"""
        model = DeepParityMLP().to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=1e-5)
        
        checkpoint_path = os.path.join(Config.model_dir, Config.checkpoint_file)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"从检查点恢复.")
        
        return model, optimizer
        
    def _evaluate(self, loader, desc="评估"):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            with tqdm(loader, desc=desc, leave=False) as pbar:
                for inputs, labels in pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    pbar.set_postfix(acc=f"{correct/total:.2%}")
        return correct / total if total > 0 else 0.0
        
    def _save_checkpoint(self):
        """保存训练状态"""
        os.makedirs(Config.model_dir, exist_ok=True)
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, os.path.join(Config.model_dir, Config.checkpoint_file))

    def run_training(self):
        try:
            for epoch in range(0, self.args.epochs):
                if self.controller.should_stop:
                    break
                
                self.model.train()
                epoch_loss = 0.0
                epoch_acc = 0.0
                processed_samples = 0
                
                train_iter = tqdm(self.train_loader, 
                                 desc=f"Epoch {epoch+1}/{self.args.epochs}",
                                 dynamic_ncols=True)
                
                for batch_idx, (inputs, labels) in enumerate(train_iter):
                    if self.controller.should_stop:
                        break
                    
                    inputs = inputs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    
                    self.optimizer.zero_grad()
                    with autocast('cuda'):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                    # 累计统计
                    batch_size = inputs.size(0)
                    epoch_loss += loss.item() * batch_size
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    epoch_acc += (preds == labels).sum().item()
                    processed_samples += batch_size
                                    
                # 计算epoch统计
                avg_loss = epoch_loss / processed_samples
                avg_acc = epoch_acc / processed_samples
                                
                
                print(f"Epoch {epoch+1} 结果 | "
                      f"平均损失: {avg_loss:.4f} | "
                      f"准确率: {avg_acc*100:.2f}% | ")
        
        finally:
            # 保存检查点
            self._save_checkpoint()
            
            print("\n=== 最终评估 ===")
            final_train_acc = self._evaluate(self.train_loader, "最终训练集评估")
            final_test_acc = self._evaluate(self.test_loader, "最终测试集评估")
            print(f"训练集准确率: {final_train_acc*100:.2f}%")
            print(f"测试集准确率: {final_test_acc*100:.2f}%")
            
            self._run_validation()
            
            torch.cuda.empty_cache()

    def _run_validation(self):
        """执行验证流程"""
        validator = Validator(os.path.join(Config.model_dir, Config.checkpoint_file))
        print("\n=== 验证集测试 ===")
        validator.generate_sequence(self.validation_pub)

# ======================
# 主程序
# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_size", type=int, default=8000000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)    
    args = parser.parse_args()
    os.makedirs(Config.model_dir, exist_ok=True)
    system = TrainingSystem(args)
    system.run_training()