#%%
import numpy as np
import matplotlib.pyplot as plt
import sys


#%%
# 1. 데이터 로더 및 전처리 (Data Loader)

def load_data(file_path):
    try:
        raw_data = np.loadtxt(file_path)
    except Exception as e:
        print(f"파일을 읽는 중 오류 발생: {e}")
        sys.exit(1)
        
    x_data = raw_data[:, :-1]
    y_data = raw_data[:, -1].astype(int)
    return x_data, y_data

def create_windows(X, y, window_size):
    if window_size == 1:
        return X, y
    
    new_X = []
    new_y = []
    
    # 데이터 개수만큼 순회 (window_size만큼 공간 필요)
    for i in range(len(X) - window_size + 1):
        window = X[i : i + window_size].flatten()
        new_X.append(window)
        new_y.append(y[i + window_size - 1])
        
    return np.array(new_X), np.array(new_y)

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]


#%%
# 2. DNN 클래스 (Core Implementation)

class DNN:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01):
        self.lr = learning_rate
        self.weights = []
        layer_dims = [input_size] + hidden_layers + [output_size]
        
        for i in range(len(layer_dims) - 1):
            w = np.random.randn(layer_dims[i+1], layer_dims[i] + 1) * 0.1
            self.weights.append(w)
            
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500))) # overflow 방지

    def forward(self, x):
        self.activations = [] # 역전파를 위해 각 층의 출력값 저장
        
        # 입력층 처리 (Bias 항 1 추가)
        a = np.append(x, 1.0) # Bias
        self.activations.append(a)
        
        # 은닉층 ~ 출력층 전파
        for i, w in enumerate(self.weights):
            z = np.dot(w, a)
            a = self.sigmoid(z)
            
            # 마지막 레이어가 아니면 다음 입력을 위해 Bias(1) 추가
            if i < len(self.weights) - 1:
                a = np.append(a, 1.0)
            
            self.activations.append(a)
            
        return a

    def backward(self, target):
        """역전파: PDF의 Backpropagation 알고리즘 구현"""
        # target은 one-hot vector 여야 함
        final_output = self.activations[-1]
        delta = (target - final_output) * final_output * (1 - final_output)    
        deltas = [delta]
        
        for i in range(len(self.weights) - 1, 0, -1):
            w = self.weights[i]
            prev_delta = deltas[-1]
            
            prev_activation = self.activations[i] 
            
            sigmoid_derivative = prev_activation * (1 - prev_activation)
            
            back_error = np.dot(w[:, :-1].T, prev_delta)
            
            next_delta = back_error * sigmoid_derivative[:-1] 
            deltas.append(next_delta)
            
        deltas.reverse()
        
        for i in range(len(self.weights)):
            delta = deltas[i].reshape(-1, 1) # (Nodes, 1)
            input_act = self.activations[i].reshape(1, -1)
            
            gradient = np.dot(delta, input_act)
            self.weights[i] += self.lr * gradient


#%%
# 3. 실험2~5 반복용 실험용 함수

def run_experiment(X_train, y_train, X_test, y_test, hidden_layers, lr, epochs=50, batch_size=10):
    # 1. 데이터 정규화
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std
    
    # 2. 라벨 인코딩 및 모델 초기화
    num_classes = int(max(np.max(y_train), np.max(y_test))) + 1
    y_train_enc = np.eye(num_classes)[y_train]
    y_test_enc = np.eye(num_classes)[y_test]
    
    model = DNN(input_size=X_train.shape[1], hidden_layers=hidden_layers, output_size=num_classes, learning_rate=lr)
    
    # 3. 최고 기록 저장을 위한 변수
    min_val_error = float('inf')
    best_train_error = 0
    
    # 4. 학습 루프
    for epoch in range(epochs):
        # Shuffle
        idx = np.arange(len(X_train))
        np.random.shuffle(idx)
        X_train_s, y_train_s = X_train[idx], y_train_enc[idx]
        
        # Train
        train_loss = 0
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train_s[i:i+batch_size]
            y_batch = y_train_s[i:i+batch_size]
            for j in range(len(X_batch)):
                out = model.forward(X_batch[j])
                model.backward(y_batch[j])
                train_loss += np.sum((y_batch[j] - out)**2)
        
        # Validation
        val_loss = 0
        for i in range(len(X_test)):
            out = model.forward(X_test[i])
            val_loss += np.sum((y_test_enc[i] - out)**2)
            
        avg_train = train_loss / len(X_train)
        avg_val = val_loss / len(X_test)
        
        # 핵심: Validation Error가 최소일 때 저장
        if avg_val < min_val_error:
            min_val_error = avg_val
            best_train_error = avg_train
            
    return best_train_error, min_val_error


def plot_results(results, title, labels):
    train_errs = [r[0] for r in results]
    val_errs = [r[1] for r in results]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.figure(figsize=(8, 6))
    plt.bar(x - width/2, train_errs, width, label='Train Error', color='grey')
    plt.bar(x + width/2, val_errs, width, label='Val Error', color='blue', alpha=0.5)
    
    plt.title(title)
    plt.xticks(x, labels)
    plt.legend()
    plt.show()

#%%
# 4. 전체 실험 실행


# # --- 하이퍼파라미터 설정 (과제 실험 조건들) ---
# INPUT_WINDOW_SIZE = 1  # 실험 3번에서 변경 (1, 3, 5...)
# HIDDEN_LAYERS = [10]   # 실험 4번, 5번에서 변경 (노드 수, 층 깊이)
# LEARNING_RATE = 0.01   # 실험 2번에서 변경
# EPOCHS = 100           # 실험 1번에서 변경
# BATCH_SIZE = 10        # Mini-batch 크기

# 1. 데이터 로드 (파일 경로를 실제 경로로 수정하세요)

try:
    X_train_raw, y_train_raw = load_data('train.txt')
    X_test_raw, y_test_raw = load_data('test.txt')
except:
    print("train.txt 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    sys.exit()

#%%
# [실험1] - Epoch에 따른 Error Curve 그리기

print("\n=== [실험 1] Error Curve 생성 중... ===")


# 설정
exp1_epochs = 100
exp1_lr = 0.01
exp1_hidden = [10]
exp1_window = 1

# 데이터 준비
X_tr, y_tr = create_windows(X_train_raw, y_train_raw, exp1_window)
X_te, y_te = create_windows(X_test_raw, y_test_raw, exp1_window)

# 정규화
mean = X_tr.mean(axis=0)
std = X_tr.std(axis=0) + 1e-8
X_tr = (X_tr - mean) / std
X_te = (X_te - mean) / std

num_classes = int(max(np.max(y_tr), np.max(y_te))) + 1
y_tr_enc = one_hot_encode(y_tr, num_classes)
y_te_enc = one_hot_encode(y_te, num_classes)

# 모델 학습
model = DNN(X_tr.shape[1], exp1_hidden, num_classes, exp1_lr)
train_errors = []
val_errors = []

for epoch in range(exp1_epochs):
    # Shuffle
    idx = np.arange(len(X_tr))
    np.random.shuffle(idx)
    X_tr_s, y_tr_s = X_tr[idx], y_tr_enc[idx]
    
    # Train
    train_loss = 0
    for i in range(0, len(X_tr_s), 10): # Batch 10
        X_batch = X_tr_s[i:i+10]
        y_batch = y_tr_s[i:i+10]
        for j in range(len(X_batch)):
            out = model.forward(X_batch[j])
            model.backward(y_batch[j])
            train_loss += np.sum((y_batch[j] - out)**2)
    
    # Val
    val_loss = 0
    for i in range(len(X_te)):
        out = model.forward(X_te[i])
        val_loss += np.sum((y_te_enc[i] - out)**2)
        
    train_errors.append(train_loss / len(X_tr))
    val_errors.append(val_loss / len(X_te))
    
    if epoch % 10 == 0:
        print(f"Exp1 Epoch {epoch}: Train {train_errors[-1]:.4f}, Val {val_errors[-1]:.4f}")


#%%
# [실험 1 결과 시각화] 선 그래프
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(train_errors, label='Training Error')
plt.plot(val_errors, label='Validation Error')
plt.title(f'Exp 1: Error Curve (Epochs)')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.show()


#%%
# [실험 2~5] 막대 그래프 (Bar Graph)

# 실험 2: Learning Rate
print("\n=== [실험 2] Learning Rate 비교 ===")
lrs = [0.1, 0.01, 0.001]
res2 = []
for lr in lrs:
    Xt, yt = create_windows(X_train_raw, y_train_raw, 1)
    Xv, yv = create_windows(X_test_raw, y_test_raw, 1)
    res2.append(run_experiment(Xt, yt, Xv, yv, [10], lr, 50))
plot_results(res2, "Exp 2: Learning Rate", [str(l) for l in lrs])

# 실험 3: Window Size
print("\n=== [실험 3] Window Size 비교 ===")
wins = [1, 3, 5]
res3 = []
for w in wins:
    Xt, yt = create_windows(X_train_raw, y_train_raw, 1)
    Xv, yv = create_windows(X_test_raw, y_test_raw, 1)
    res3.append(run_experiment(Xt, yt, Xv, yv, [10], 0.01, 50))
plot_results(res3, "Exp 3: Window Size", [str(w) for w in wins])

# 실험 4: Node 수
print("\n=== [실험 4] Hidden Nodes 비교 ===")
nodes = [5, 10, 20]
res4 = []
for n in nodes:
    Xt, yt = create_windows(X_train_raw, y_train_raw, 1)
    Xv, yv = create_windows(X_test_raw, y_test_raw, 1)
    res4.append(run_experiment(Xt, yt, Xv, yv, [n], 0.01, 50))
plot_results(res4, "Exp 4: Hidden Nodes", [str(n) for n in nodes])

# 실험 5: Layer 수
print("\n=== [실험 5] Hidden Layers 비교 ===")
layers = [[10], [10, 10], [10, 10, 10]]
labels = ["1 Layer", "2 Layers", "3 Layers"]
res5 = []
for l in layers:
    Xt, yt = create_windows(X_train_raw, y_train_raw, 1)
    Xv, yv = create_windows(X_test_raw, y_test_raw, 1)
    res5.append(run_experiment(Xt, yt, Xv, yv, l, 0.01, 50))
plot_results(res5, "Exp 5: Hidden Layers", labels)
