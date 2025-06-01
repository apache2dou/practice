import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from coincurve import PrivateKey, PublicKey
import os
import time
import random
import argparse
import signal
import keyboard
import numpy as np
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
    should_stop = False
    order = 0

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
    def __init__(self, model, device):
        self.model = model
        self.device = device
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
            output = self.model(features.unsqueeze(0).to(self.device))
            prob = torch.sigmoid(output).item()
        return 1 if prob > 0.5 else 0, abs(prob - 0.5)*2  # 置信度[0-1]

    def predict_sequence(self, base_pub_str, length=100):
        """生成并验证连续公钥序列"""
        current_pub = self._parse_public_key(base_pub_str)
        predictions = []
        
        for _ in range(length):
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
        
        # 统计奇偶结果
        odd_count = sum(p[1] for p in predictions)
        even_count = sum(1 for p in predictions if p[1] == 0)
        
        print(f"\n给定公钥生成的序列（共{len(predictions)}个有效点）:")
        print(f"奇数个数: {odd_count} (占比: {odd_count/len(predictions):.2%})")
        print(f"偶数个数: {even_count} (占比: {even_count/len(predictions):.2%})")        
        
        # 检查置信度
        avg_confidence = sum(p[2] for p in predictions) / len(predictions)
        print(f"平均置信度(100序列): {avg_confidence:.2%}")
        
    def _generate_random_sequence(self, length=100):
        """生成随机连续公钥序列（带真实标签）"""
        base_priv = int.from_bytes(random.randbytes(32), 'big') % (2**(256 - Config.order))
        true_parity = base_priv % 2
        sequence = []
        current_priv = base_priv
        
        for _ in range(length):
            priv_key = PrivateKey(secret=current_priv.to_bytes(32, 'big'))
            pub = priv_key.public_key.format(compressed=False)[1:]
            sequence.append(pub)
            current_priv += 2  # 保持奇偶性不变
        
        return sequence, true_parity
    
    def validate_effectiveness(self, num_sequences=100, sequence_length=100):
        """批量验证模型有效性"""
        correct_count = 0
        confidence_sum = 0.0
        
        for i in range(num_sequences):
            # 生成随机序列
            sequence, true_parity = self._generate_random_sequence(sequence_length)
            
            # 预测序列中所有点的奇偶性
            predictions = []
            for pub_bytes in sequence:
                predictions.append(self.predict_parity(pub_bytes))
            
            # 多数投票决定最终预测
            majority_vote = 1 if sum(p[0] for p in predictions) > len(predictions)/2 else 0
            sequence_confidence = np.mean(np.array(list(p[1] for p in predictions)))
            
            # 检查预测是否正确
            if majority_vote == true_parity:
                correct_count += 1
                confidence_sum += sequence_confidence
        
        # 计算统计指标
        accuracy = correct_count / num_sequences
        avg_confidence = confidence_sum / correct_count if correct_count > 0 else 0
        
        print(f"验证结果（{num_sequences}组序列）:")
        print(f"准确率: {accuracy*100:.2f}%")
        print(f"平均置信度(仅计算正确的序列(序列均值)): {avg_confidence*100:.2f}%")
        return accuracy, avg_confidence
# ======================
# 信号处理器
# ======================
    
def stop_handler( ):
    print(f"\n捕获信号，正在终止训练...")
    Config.should_stop = True

# ======================
# 数据生成模块
# ======================
class KeyPairGenerator:
    @staticmethod
    def _generate_group(show_example=False):
        """生成一组20个连续奇偶性的密钥对"""
        group = []
        base_priv = int.from_bytes(random.randbytes(32), 'big') % (2**(256 - Config.order))
        
        for i in range(Config.group_size):
            priv_int = base_priv + 2 * i
            priv_key = PrivateKey(secret=priv_int.to_bytes(32, 'big'))
            pub = priv_key.public_key.format(compressed=False)
            
            x = int.from_bytes(pub[1:33], 'big')
            y = int.from_bytes(pub[33:65], 'big')
            label = priv_int % 2
            
            if show_example and i == 0:
                print(f"Private: {priv_key.secret.hex()}")
                print(f"Public X: {x:064x}")
                print(f"Public Y: {y:064x}")
                print(f"Label: {'奇数' if label else '偶数'}\n")
            
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
# 输入处理模块
# ======================
class InputProcessor:
    @staticmethod
    def normalize_binary_features(features):
        """
        处理512位二进制输入，防止梯度消失/爆炸
        方法：
        1. 转换为浮点数张量 (0.0/1.0)
        2. 归一化到 [-0.5, 0.5] 区间
        3. 添加高斯噪声增强鲁棒性
        """
        # 转换为浮点张量
        if isinstance(features, torch.Tensor):
            tensor = features.detach().clone().float()
        else:
            tensor = torch.tensor(features, dtype=torch.float32)
			
        # 归一化到 [-0.5, 0.5]
        normalized = tensor - 0.5
        
        # 训练时添加微小噪声
        if normalized.requires_grad:
            noise = torch.randn_like(normalized) * 0.01
            normalized += noise
        
        return normalized
    
    @staticmethod
    def positional_encoding(features, max_len=512):
        """添加位置编码增强序列信息"""
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, 128, 2).float() * (-np.log(10000.0) / 128))
        
        pe = torch.zeros(max_len, 128)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 拼接位置编码
        return torch.cat([features, pe[:features.size(0)]], dim=1)
# ======================
# 100层深度残差网络
# ======================
class DeepResidualMLP(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=100, dropout=0.2):
        super().__init__()
        
        # 输入投影层
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 残差块堆叠
        self.residual_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.residual_blocks.append(
                ResidualBlock(hidden_dim, dropout=dropout)
            )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        # He初始化配合LeakyReLU
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, 
                                        a=0.1, 
                                        mode='fan_in', 
                                        nonlinearity='leaky_relu')
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # 输入处理
        x = InputProcessor.normalize_binary_features(x)
        
        # 初始投影
        x = self.input_proj(x)
        
        # 残差连接
        for block in self.residual_blocks:
            x = block(x)
        
        # 输出
        return self.output_layer(x).squeeze(-1)

class ResidualBlock(nn.Module):
    def __init__(self, dim, expansion=4, dropout=0.2):
        super().__init__()
        inner_dim = dim * expansion
        
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        
        # 门控机制
        self.gate = nn.Parameter(torch.tensor([1.0]))
    
    def forward(self, x):
        identity = x
        out = self.block(x)
        return identity + self.gate * out

# ======================
# 训练系统
# ======================
class TrainingSystem:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化模型和优化器
        self.model, self.optimizer, self.scheduler = self._load_or_create_model()
        self.scaler = GradScaler('cuda')
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 添加验证集路径
        self.validation_pub = "8fd74b41a5f5c775ea13b7617d7ffe871c0cbad1b7bb99bcea03dc47561feae4,dad89019b8f2e6990782b9ae4e74243b1ac2ec007d621642d507b1a844d3e05f"
        #判断为奇数
        
        #self.validation_pub = "f286ba59399081e8cd57a7c4327c37ca9ea00f5d6a0096884cf7d0c4e0070e9f,b03087df6527a4070528731ddf8b5eebe4db55bffed52ba0ded5642bef02c8c"
        #判断为偶数
        
        #self.validation_pub = "fda774bd460f57a5149f1f7e25246f2b92ab7a8e95139346273b15c7a7a349c7,be36b081633969887ca7ce24d41e7b3653ec8a6b42080f4cf539cd630e01ecb"
        #奇数
        
        #self.validation_pub = "1601941ac8ee7561fc4a8009031bcc988a898cc71203b2811f17662d868df399,556c0c8b2c66c37235be68c58308b1f6c0a52beb6c20ca34a957c1610d318db4"
        
        
        Config.order = 0
        
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

    def _load_or_create_model(self):
        """加载已有模型或创建新模型"""
        model = DeepResidualMLP(
            input_dim=512,
            hidden_dim=512,
            num_layers=100,
            dropout=0.1
        ).to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.args.lr*10,
            steps_per_epoch=1000,
            epochs=100,
            anneal_strategy='cos'
        )
        checkpoint_path = os.path.join(Config.model_dir, Config.checkpoint_file)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"从检查点恢复.")
        
        return model, optimizer, scheduler
        
    def _evaluate(self, loader, desc="评估"):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total if total > 0 else 0.0
        
    def _save_checkpoint(self):
        """保存训练状态"""
        os.makedirs(Config.model_dir, exist_ok=True)
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
			'scheduler': self.scheduler.state_dict()
        }, os.path.join(Config.model_dir, Config.checkpoint_file))

    def run_training(self):
        try:
            for epoch in range(0, self.args.epochs):
                if Config.should_stop:
                    break
                
                self.model.train()
                epoch_loss = 0.0
                epoch_acc = 0.0
                processed_samples = 0
                             
                for inputs, labels in self.train_loader:                    
                    inputs = inputs.to(self.device, non_blocking=True)
                    labels = labels.to(self.device, non_blocking=True)
                    
                    self.optimizer.zero_grad()
                    with autocast('cuda'):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                    
                    self.scaler.scale(loss).backward()
					# 梯度裁剪防止爆炸
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
					
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
            
            print("=== 数据集评估 ===")
            final_train_acc = self._evaluate(self.train_loader, "训练集评估")
            final_test_acc = self._evaluate(self.test_loader, "测试集评估")
            print(f"训练集准确率: {final_train_acc*100:.2f}%")
            print(f"测试集准确率: {final_test_acc*100:.2f}%")
            
            self._run_validation()
            
            torch.cuda.empty_cache()

    def _run_validation(self):
        """执行验证流程"""
        validator = Validator(self.model, self.device)
        print("=== 验证集测试 ===")
        validator.validate_effectiveness()
        validator.predict_sequence(self.validation_pub)

# ======================
# 主程序
# ======================
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_size", type=int, default=8000000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)    
    parser.add_argument("--mode", choices=["train", "test"], default="train")
    args = parser.parse_args()
    os.makedirs(Config.model_dir, exist_ok=True)
    
    if args.mode == "test":
        Config.order = 2
        for i in range(10):
            KeyPairGenerator._generate_group(show_example=True)
        sys.exit(0)
    
    
    # 注册快捷键 Ctrl+o
    keyboard.add_hotkey('ctrl+o', stop_handler)
    try:
        session_count = 0
        while not Config.should_stop:
            session_count += 1
            print(f"\n=== 训练会话 {session_count} ===")

            # 删除旧数据集并生成新数据
            if os.path.exists(Config.dataset_cache):
                os.remove(Config.dataset_cache)
                
            system = TrainingSystem(args)
            system.run_training()            
            # 清理资源
            del system
            torch.cuda.empty_cache()            
    except KeyboardInterrupt:
        print("\n接收到终止信号，结束训练循环")
    finally:
        keyboard.clear_all_hotkeys()
        print("训练结束，最终模型已保存")