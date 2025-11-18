import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# ======== 1. Carregar dataset ========
df = pd.read_csv("recruitment_data.csv")

# Detectar target automaticamente (pega a última coluna)
target = df.columns[-1]
y = df[target]
X = df.drop(columns=[target])

# Transformar categóricas
X = pd.get_dummies(X, drop_first=True)

# Preencher valores faltantes
X = X.fillna(X.mean())
y = y.fillna(y.mode()[0])

# Converter y para numérico (caso seja texto)
y = y.astype(str).str.lower().map({"yes":1,"sim":1,"1":1,"no":0,"nao":0,"não":0,"0":0}).fillna(0).astype(int)

# ======== 2. Split treino/teste ========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ======== 3. Padronização ========
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ======== 4. Ajustar modelo por mínimos quadrados ========
# Adicionar bias
ones_train = np.ones((X_train_s.shape[0], 1))
X_train_aug = np.hstack([ones_train, X_train_s])

# Resolver w = (X^T X)^(-1) X^T y
w, _, _, _ = np.linalg.lstsq(X_train_aug, y_train, rcond=None)

# ======== 5. Previsões ========
X_test_aug = np.hstack([np.ones((X_test_s.shape[0],1)), X_test_s])
y_pred_cont = X_test_aug @ w
y_pred = (y_pred_cont >= 0.5).astype(int)

# ======== 6. Métricas ========
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_pred_cont)

print("=== RESULTADOS ===")
print("Acurácia:", acc)
print("Precisão:", prec)
print("Recall:", rec)
print("F1:", f1)
print("AUC:", auc)

# ======== 7. Curva ROC ========
fpr, tpr, _ = roc_curve(y_test, y_pred_cont)
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("Falso Positivo")
plt.ylabel("Verdadeiro Positivo")
plt.title("Curva ROC - Linear Lstsq")
plt.legend()
plt.grid(True)
plt.show()