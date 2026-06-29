import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 400)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def to_hex_8(val): return f"{int(val) & 0xFF:02X}"
def to_hex_32(val): return f"{int(val) & 0xFFFFFFFF:08X}"

def main():
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet5().to(device)
    
    # [1] 빠른 학습 진행
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    print("🔥 모델 학습 중...")
    for epoch in range(10):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()

    # [2] 손글씨 '1' 데이터 확보 및 테스트
    model.eval()
    for img, label in test_dataset:
        if label == 1:
            target_img, target_label = img, label
            break
    with torch.no_grad():
        pred = model(target_img.unsqueeze(0).to(device)).argmax(1).item()
        print(f"🎯 [소프트웨어 예측]: 정답 '{target_label}' -> 예측 '{pred}'")

    print("💾 하드웨어 8뱅크 매핑(Interleaving) 파일 추출 시작...")
    X_SCALE, W_SCALE, B_SCALE = 127.0, 16.0, 2032.0
    model.cpu()

    # ---------------------------------------------------------
    # 🟦 1. 이미지 매핑 (모든 픽셀을 Bank 0에만)
    # ---------------------------------------------------------
    img_array = np.clip(np.round(target_img.numpy().flatten() * X_SCALE), 0, 127).astype(int)
    with open("image.txt", "w") as f:
        for val in img_array:
            f.write(to_hex_8(val) + "\n")
            for _ in range(7): f.write(to_hex_8(0) + "\n") # Bank 1~7 Padding

    # ---------------------------------------------------------
    # 🟥 2. 가중치 매핑 (8뱅크 입체 순환 분배)
    # ---------------------------------------------------------
    bank_w = [[] for _ in range(8)]
    
    # Conv1 (6필터 x 25)
    w_c1 = model.conv1.weight.detach().numpy().reshape(6, 25)
    for i in range(25):
        for b in range(8): bank_w[b].append(w_c1[b, i] if b < 6 else 0)

    # Conv2 (16필터 x 150)
    w_c2 = model.conv2.weight.detach().numpy().reshape(16, 150)
    for i in range(150):
        for b in range(8): bank_w[b].append(w_c2[b, i])
    for i in range(150):
        for b in range(8): bank_w[b].append(w_c2[8+b, i])

    # ---------------------------------------------------------
    # FC Layers
    # ---------------------------------------------------------
    def append_fc_w_array(mat, target_acc_cycles):
        for node in mat:
            pad = np.pad(node, (0, target_acc_cycles * 8 - len(node)))
            for i in range(target_acc_cycles):
                for b in range(8):
                    bank_w[b].append(pad[i * 8 + b])

    # FC1 입력은 Conv2 출력이 하드웨어에서 8-bank interleaving 순서로 저장됨.
    # PyTorch flatten: ch0[0..24], ch1[0..24], ...
    # Hardware read : ch0[0], ch1[0], ..., ch7[0], ch0[1], ...
    fc1_hw_order = []

    # Conv2 output shape after pool2 = 16 x 5 x 5 = 400
    # batch 0: channel 0~7, batch 1: channel 8~15
    for group in range(2):          # 0: ch0~7, 1: ch8~15
        for spatial in range(25):   # 5x5 spatial index
            for bank in range(8):   # bank0~7
                ch = group * 8 + bank
                py_index = ch * 25 + spatial
                fc1_hw_order.append(py_index)

    fc1_w = model.fc1.weight.detach().numpy()
    fc1_w_hw = fc1_w[:, fc1_hw_order]

    append_fc_w_array(fc1_w_hw, 50)                         # FC1: 400 -> 50*8
    append_fc_w_array(model.fc2.weight.detach().numpy(), 15) # FC2: 120 -> 15*8
    append_fc_w_array(model.fc3.weight.detach().numpy(), 11) # FC3: 84 -> 11*8

    with open("weight.txt", "w") as f:
        for i in range(len(bank_w[0])): # 가로(주소)로 스캔하며 8개씩 출력
            for b in range(8):
                val = np.clip(np.round(bank_w[b][i] * W_SCALE), -128, 127)
                f.write(to_hex_8(val) + "\n")

    # ---------------------------------------------------------
    # 🟨 3. 바이어스 매핑 (FC 바이어스는 Bank 0 집중)
    # ---------------------------------------------------------
    bank_b = [[] for _ in range(8)]
    
    b_c1 = model.conv1.bias.detach().numpy()
    for b in range(8): bank_b[b].append(b_c1[b] if b < 6 else 0)
    
    b_c2 = model.conv2.bias.detach().numpy()
    for b in range(8): bank_b[b].append(b_c2[b])
    for b in range(8): bank_b[b].append(b_c2[8+b])
    
    fc_b = np.concatenate([model.fc1.bias.detach().numpy(), model.fc2.bias.detach().numpy(), model.fc3.bias.detach().numpy()])
    for val in fc_b:
        bank_b[0].append(val)
        for b in range(1, 8): bank_b[b].append(0) # Bank 1~7은 0

    with open("bias.txt", "w") as f:
        for i in range(len(bank_b[0])):
            for b in range(8):
                val = np.round(bank_b[b][i] * B_SCALE)
                f.write(to_hex_32(val) + "\n")

    print(f"🎉 하드웨어 8-Bank 맞춤형 파일 생성 완료!")
    print(f"  - Image Addr : {len(img_array)} (Total {len(img_array)*8} lines)")
    print(f"  - Weight Addr: {len(bank_w[0])} (Total {len(bank_w[0])*8} lines)")
    print(f"  - Bias Addr  : {len(bank_b[0])} (Total {len(bank_b[0])*8} lines)")

if __name__ == "__main__":
    main()