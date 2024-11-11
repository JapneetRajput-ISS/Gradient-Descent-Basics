import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# Generate sample data
np.random.seed(42)
x = np.random.randn(100,1)
y = 5*x + np.random.rand(100,1)

# Parameters
w = 0.0 
b = 0.0 
learning_rate = 0.01
n_epochs = 800
k_folds = 5

def descend(x, y, w, b, learning_rate): 
    dldw = 0.0 
    dldb = 0.0 
    N = x.shape[0]
    
    for xi, yi in zip(x,y): 
        dldw += -2*xi*(yi-(w*xi+b))
        dldb += -2*(yi-(w*xi+b))
    
    # Extract scalar values properly using item()
    w = w - learning_rate*(1/N)*dldw.item()
    b = b - learning_rate*(1/N)*dldb.item()
    return w, b 

def train_fold(x_train, y_train, x_val, y_val):
    w, b = 0.0, 0.0
    
    for epoch in range(n_epochs): 
        w, b = descend(x_train, y_train, w, b, learning_rate)
        yhat_train = w*x_train + b
        train_loss = np.mean((y_train-yhat_train)**2)
        
        if epoch % 100 == 0:
            yhat_val = w*x_val + b
            val_loss = np.mean((y_val-yhat_val)**2)
            val_r2 = r2_score(y_val, yhat_val)
            print(f'Epoch {epoch} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val R²: {val_r2:.4f}')
    
    yhat_val = w*x_val + b
    final_r2 = r2_score(y_val, yhat_val)
    return w, b, final_r2

# Perform k-fold cross validation
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
fold_r2_scores = []
fold_params = []

for fold, (train_idx, val_idx) in enumerate(kf.split(x)):
    print(f"\nFold {fold+1}/{k_folds}")
    x_train, x_val = x[train_idx], x[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Train on this fold
    w, b, fold_r2 = train_fold(x_train, y_train, x_val, y_val)
    fold_r2_scores.append(fold_r2)
    fold_params.append((w, b))
    
    print(f"Fold {fold+1} Results:")
    print(f"R² Score: {fold_r2:.4f}")
    print(f"Parameters - w: {w:.4f}, b: {b:.4f}")

# Print overall results
print("\nOverall Results:")
print(f"Average R² Score across folds: {np.mean(fold_r2_scores):.4f}")
print(f"Standard deviation of R² Score: {np.std(fold_r2_scores):.4f}")

# Final model using average parameters
final_w = np.mean([p[0] for p in fold_params])
final_b = np.mean([p[1] for p in fold_params])
print(f"\nFinal averaged model parameters:")
print(f"w: {final_w:.4f}, b: {final_b:.4f}")

# Calculate final R² score on entire dataset
y_final_pred = final_w*x + final_b
final_r2 = r2_score(y, y_final_pred)
print(f"Final model R² score on entire dataset: {final_r2:.4f}")