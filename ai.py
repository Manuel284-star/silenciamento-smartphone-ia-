
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. GERAÇÃO DE DADOS SINTÉTICOS (Simulação)
# ==========================================
def gerar_dataset_sensores(n_samples=4500):
    np.random.seed(42)
    
    # Variáveis (Features)
    # Acelerómetro Z: -9.8 (virado para cima) a 9.8 (virado para baixo)
    acc_z = np.random.uniform(-10, 10, n_samples)
    
    # Luz (Lux): 0 (escuro total) a 1000 (muito claro)
    luz = np.random.exponential(scale=100, size=n_samples) 
    
    # Proximidade: 0 (longe/descoberto) ou 1 (perto/tapado)
    proximidade = np.random.choice([0, 1], size=n_samples)
    
    # Magnitude Aceleração (Movimento): 0 (imóvel) a 20 (movimento forte)
    movimento = np.random.exponential(scale=2, size=n_samples)
    
    # Lógica para definir o Target (0: Normal/Tocar, 1: Silenciar)
    # Regras base para criar o "ground truth" com algum ruído
    y = []
    
    for i in range(n_samples):
        silenciar = 0 # Default: Não silenciar
        
        # Regra A: Telemóvel virado para baixo na mesa (Z positivo alto + Imóvel)
        if acc_z[i] > 8 and movimento[i] < 1.5:
            silenciar = 1
            
        # Regra B: No bolso em reunião (Proximidade=1 + Escuro + Imóvel)
        elif proximidade[i] == 1 and luz[i] < 10 and movimento[i] < 1.0:
            silenciar = 1
            
        # Regra C: No bolso a andar (Proximidade=1 + Escuro + Movimento Alto) -> NÃO silenciar (queremos ouvir a chamada)
        elif proximidade[i] == 1 and luz[i] < 10 and movimento[i] > 3.0:
            silenciar = 0
            
        # Adicionar 5% de ruído aleatório (erros humanos, exceções)
        if np.random.rand() < 0.05:
            silenciar = 1 - silenciar
            
        y.append(silenciar)
        
    df = pd.DataFrame({
        'Acc_Z': acc_z,
        'Luz_Lux': luz,
        'Proximidade': proximidade,
        'Movimento': movimento,
        'Target': y
    })
    
    return df

print("--- 1. A Gerar Dataset Sintético ---")
df = gerar_dataset_sensores()
print(f"Dataset criado com {len(df)} amostras.")
print(df.head())
print("\nDistribuição das classes:\n", df['Target'].value_counts())

# ==========================================
# 2. PRÉ-PROCESSAMENTO
# ==========================================
print("\n--- 2. Pré-processamento ---")

# Separar Features (X) e Target (y)
X = df.drop('Target', axis=1)
y = df['Target']

# Divisão Treino (70%) / Teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalização (Min-Max Scaling)
# Importante para Redes Neuronais para colocar tudo entre 0 e 1
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Dados normalizados e divididos.")

# ==========================================
# 3. BASELINE: REGRESSÃO LOGÍSTICA
# ==========================================
print("\n--- 3. Treino da Baseline (Regressão Logística) ---")
baseline_model = LogisticRegression()
baseline_model.fit(X_train_scaled, y_train)

y_pred_baseline = baseline_model.predict(X_test_scaled)
acc_baseline = accuracy_score(y_test, y_pred_baseline)

print(f"Acurácia da Baseline: {acc_baseline:.4f}")
print("Relatório de Classificação (Baseline):")
print(classification_report(y_test, y_pred_baseline))

# ==========================================
# 4. ABORDAGEM PRINCIPAL: REDE NEURONAL (DNN/MLP)
# ==========================================
print("\n--- 4. Treino da Abordagem Principal (DNN) ---")

# Configuração descrita no relatório:
# Camadas ocultas: (16, 8), Ativação: ReLU, Solver: Adam
dnn_model = MLPClassifier(hidden_layer_sizes=(16, 8), 
                          activation='relu', 
                          solver='adam', 
                          max_iter=500, 
                          random_state=42,
                          early_stopping=True) # Para evitar overfitting

dnn_model.fit(X_train_scaled, y_train)

y_pred_dnn = dnn_model.predict(X_test_scaled)
acc_dnn = accuracy_score(y_test, y_pred_dnn)

print(f"Acurácia da DNN: {acc_dnn:.4f}")
print("Relatório de Classificação (DNN):")
print(classification_report(y_test, y_pred_dnn))

# ==========================================
# 5. ANÁLISE DE RESULTADOS E ERROS
# ==========================================
print("\n--- 5. Análise Comparativa ---")
print(f"Melhoria sobre a Baseline: {(acc_dnn - acc_baseline) * 100:.2f}%")

# Matriz de Confusão da DNN
cm = confusion_matrix(y_test, y_pred_dnn)
print("\nMatriz de Confusão (DNN):")
print(cm)

# Análise de Erros: Mostrar exemplos onde o modelo falhou
print("\n--- Exemplos de Erros (Falsos Positivos/Negativos) ---")
X_test_restored = scaler.inverse_transform(X_test_scaled) # Voltar à escala original para ler
df_test = pd.DataFrame(X_test_restored, columns=['Acc_Z', 'Luz_Lux', 'Proximidade', 'Movimento'])
df_test['Real'] = y_test.values
df_test['Previsto'] = y_pred_dnn

# Filtrar erros
erros = df_test[df_test['Real'] != df_test['Previsto']]
print(f"Total de erros no teste: {len(erros)}")
print("Primeiros 5 exemplos de erro:")
print(erros.head())

# (Opcional) Guardar o modelo ou resultados
# import joblib
# joblib.dump(dnn_model, 'modelo_silencio_dnn.pkl')