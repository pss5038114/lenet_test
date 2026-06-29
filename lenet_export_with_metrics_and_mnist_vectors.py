import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


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


def to_hex_8(val):
    return f"{int(val) & 0xFF:02X}"


def to_hex_32(val):
    return f"{int(val) & 0xFFFFFFFF:08X}"


def tensor_to_hw_image_array(img_tensor, x_scale=127.0):
    """
    img_tensor: shape [1, 32, 32], value range 0.0~1.0
    return: flattened 1024 values, scaled to 0~127
    """
    return np.clip(np.round(img_tensor.cpu().numpy().flatten() * x_scale), 0, 127).astype(int)


def write_hw_image_txt(img_tensor, path, x_scale=127.0):
    """
    기존 image.txt와 같은 형식:
      32x32 = 1024 pixels
      pixel마다 8 lines
      bank0 = pixel
      bank1~7 = 00
      total 8192 lines
    """
    img_array = tensor_to_hw_image_array(img_tensor, x_scale=x_scale)

    with open(path, "w", encoding="utf-8") as f:
        for val in img_array:
            f.write(to_hex_8(val) + "\n")
            for _ in range(7):
                f.write("00\n")

    return img_array


def save_preview_png(img_tensor, path):
    """
    Preview용 32x32 grayscale PNG 저장.
    보기 편하게 실제 픽셀은 0~255 범위로 저장한다.
    """
    arr = np.clip(np.round(img_tensor.squeeze(0).cpu().numpy() * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


def export_mnist_test_vectors(
    model,
    dataset,
    device,
    out_root="mnist_test_vectors",
    samples_per_digit=10,
    x_scale=127.0,
):
    """
    MNIST test set에서 digit 0~9 각각 samples_per_digit개를 뽑아
    기존 image.txt와 같은 8-bank txt와 preview PNG를 저장한다.

    폴더 구조:
      mnist_test_vectors/
        digit_0/
          digit0_sample00_pred0_conf0.9999.txt
          digit0_sample00_pred0_conf0.9999.png
          ...
        digit_1/
        ...
        predictions.csv
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    counts = {digit: 0 for digit in range(10)}
    rows = []

    model.eval()
    print("\n🧪 MNIST 테스트 벡터 0~9 각 10개 생성 및 소프트웨어 예측 시작...")

    with torch.no_grad():
        for dataset_idx, (img, label) in enumerate(dataset):
            label = int(label)

            if counts[label] >= samples_per_digit:
                continue

            digit_dir = out_root / f"digit_{label}"
            digit_dir.mkdir(parents=True, exist_ok=True)

            logits = model(img.unsqueeze(0).to(device))
            probs = torch.softmax(logits, dim=1)[0]
            pred = int(torch.argmax(probs).item())
            conf = float(probs[pred].item())

            sample_idx = counts[label]
            stem = f"digit{label}_sample{sample_idx:02d}_pred{pred}_conf{conf:.4f}"
            txt_path = digit_dir / f"{stem}.txt"
            png_path = digit_dir / f"{stem}.png"

            img_array = write_hw_image_txt(img, txt_path, x_scale=x_scale)
            save_preview_png(img, png_path)

            ok_mark = "✅" if pred == label else "❌"
            print(
                f"{ok_mark} label={label} sample={sample_idx:02d} "
                f"pred={pred} conf={conf:.4f} "
                f"txt={txt_path.as_posix()}"
            )

            rows.append({
                "label": label,
                "sample_idx": sample_idx,
                "dataset_idx": dataset_idx,
                "prediction": pred,
                "confidence": f"{conf:.6f}",
                "correct": int(pred == label),
                "txt_path": txt_path.as_posix(),
                "png_path": png_path.as_posix(),
                "lines": len(img_array) * 8,
            })

            counts[label] += 1

            if all(counts[d] >= samples_per_digit for d in range(10)):
                break

    csv_path = out_root / "predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "sample_idx",
                "dataset_idx",
                "prediction",
                "confidence",
                "correct",
                "txt_path",
                "png_path",
                "lines",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    correct = sum(int(row["correct"]) for row in rows)
    print("\n📁 테스트 벡터 생성 완료!")
    print(f"  - Output folder : {out_root.resolve()}")
    print(f"  - Files         : {total} txt + {total} png")
    print(f"  - Summary CSV   : {csv_path}")
    print(f"  - SW Acc on exported samples: {correct}/{total} = {100.0 * correct / total:.2f}%")

    for digit in range(10):
        digit_rows = [r for r in rows if r["label"] == digit]
        digit_correct = sum(int(r["correct"]) for r in digit_rows)
        print(f"    digit {digit}: {digit_correct}/{len(digit_rows)} correct")

    return rows


def main():
    # 기존 코드와 동일하게 28x28 MNIST를 32x32로 Resize해서 사용한다.
    # 하드웨어 입력 txt도 이 32x32 tensor를 그대로 8-bank 형식으로 저장한다.
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet5().to(device)

    # [1] 학습 진행 + Epoch별 Loss/Accuracy 출력
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"🔥 모델 학습 중... device={device}")
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(
            f"[Epoch {epoch + 1:02d}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
        )

    # [2] 손글씨 '1' 데이터 확보 및 테스트
    model.eval()
    target_img = None
    target_label = None

    for img, label in test_dataset:
        if int(label) == 1:
            target_img, target_label = img, int(label)
            break

    with torch.no_grad():
        pred = model(target_img.unsqueeze(0).to(device)).argmax(1).item()
        print(f"🎯 [소프트웨어 예측]: 정답 '{target_label}' -> 예측 '{pred}'")

    print("\n💾 하드웨어 8뱅크 매핑(Interleaving) 파일 추출 시작...")
    X_SCALE, W_SCALE, B_SCALE = 127.0, 16.0, 2032.0

    # ---------------------------------------------------------
    # 🟦 1. 대표 image.txt 생성: digit 1 하나
    # ---------------------------------------------------------
    img_array = write_hw_image_txt(target_img, "image.txt", x_scale=X_SCALE)

    # ---------------------------------------------------------
    # 🟥 2. 가중치 매핑 (8뱅크 입체 순환 분배)
    # ---------------------------------------------------------
    model_cpu = model.cpu()
    bank_w = [[] for _ in range(8)]

    # Conv1 (6필터 x 25)
    w_c1 = model_cpu.conv1.weight.detach().numpy().reshape(6, 25)
    for i in range(25):
        for b in range(8):
            bank_w[b].append(w_c1[b, i] if b < 6 else 0)

    # Conv2 (16필터 x 150)
    w_c2 = model_cpu.conv2.weight.detach().numpy().reshape(16, 150)
    for i in range(150):
        for b in range(8):
            bank_w[b].append(w_c2[b, i])
    for i in range(150):
        for b in range(8):
            bank_w[b].append(w_c2[8 + b, i])

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

    for group in range(2):          # 0: ch0~7, 1: ch8~15
        for spatial in range(25):   # 5x5 spatial index
            for bank in range(8):   # bank0~7
                ch = group * 8 + bank
                py_index = ch * 25 + spatial
                fc1_hw_order.append(py_index)

    fc1_w = model_cpu.fc1.weight.detach().numpy()
    fc1_w_hw = fc1_w[:, fc1_hw_order]

    append_fc_w_array(fc1_w_hw, 50)                         # FC1: 400 -> 50*8
    append_fc_w_array(model_cpu.fc2.weight.detach().numpy(), 15) # FC2: 120 -> 15*8
    append_fc_w_array(model_cpu.fc3.weight.detach().numpy(), 11) # FC3: 84 -> 11*8

    with open("weight.txt", "w", encoding="utf-8") as f:
        for i in range(len(bank_w[0])):  # 가로(주소)로 스캔하며 8개씩 출력
            for b in range(8):
                val = np.clip(np.round(bank_w[b][i] * W_SCALE), -128, 127)
                f.write(to_hex_8(val) + "\n")

    # ---------------------------------------------------------
    # 🟨 3. 바이어스 매핑 (FC 바이어스는 Bank 0 집중)
    # ---------------------------------------------------------
    bank_b = [[] for _ in range(8)]

    b_c1 = model_cpu.conv1.bias.detach().numpy()
    for b in range(8):
        bank_b[b].append(b_c1[b] if b < 6 else 0)

    b_c2 = model_cpu.conv2.bias.detach().numpy()
    for b in range(8):
        bank_b[b].append(b_c2[b])
    for b in range(8):
        bank_b[b].append(b_c2[8 + b])

    fc_b = np.concatenate([
        model_cpu.fc1.bias.detach().numpy(),
        model_cpu.fc2.bias.detach().numpy(),
        model_cpu.fc3.bias.detach().numpy(),
    ])

    for val in fc_b:
        bank_b[0].append(val)
        for b in range(1, 8):
            bank_b[b].append(0)  # Bank 1~7은 0

    with open("bias.txt", "w", encoding="utf-8") as f:
        for i in range(len(bank_b[0])):
            for b in range(8):
                val = np.round(bank_b[b][i] * B_SCALE)
                f.write(to_hex_32(val) + "\n")

    print("🎉 하드웨어 8-Bank 맞춤형 weight/bias/image 파일 생성 완료!")
    print(f"  - Image Addr : {len(img_array)} (Total {len(img_array) * 8} lines)")
    print(f"  - Weight Addr: {len(bank_w[0])} (Total {len(bank_w[0]) * 8} lines)")
    print(f"  - Bias Addr  : {len(bank_b[0])} (Total {len(bank_b[0]) * 8} lines)")

    # ---------------------------------------------------------
    # 🟩 4. 테스트용 MNIST image txt/png 0~9 각 10개 생성
    # ---------------------------------------------------------
    # model_cpu를 다시 device로 옮겨서 예측 출력도 같이 진행한다.
    model_cpu.to(device)
    export_mnist_test_vectors(
        model=model_cpu,
        dataset=test_dataset,
        device=device,
        out_root="mnist_test_vectors",
        samples_per_digit=10,
        x_scale=X_SCALE,
    )


if __name__ == "__main__":
    main()
