import lightgbm as lgb
import sklearn
import pandas as pd

print("=== VERIFICAÇÃO DO AMBIENTE ===")
print(f"LightGBM version: {lgb.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"Pandas version: {pd.__version__}")

# Testar assinatura do método fit
import inspect
fit_signature = inspect.signature(lgb.LGBMClassifier.fit)
print(f"\nAssinatura do método fit: {fit_signature}")

# Verificar parâmetros suportados
print("\nParâmetros suportados no fit:")
classifier = lgb.LGBMClassifier()
fit_params = classifier.get_params()
print("Parâmetros do classificador:", fit_params)