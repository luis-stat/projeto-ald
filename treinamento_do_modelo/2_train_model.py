import pandas as pd
import numpy as np
import os
import pickle
import sys
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings

class ModelTrainer:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_columns = None
        
    def load_features(self) -> pd.DataFrame:
        features_path = os.path.join(self.base_dir, 'dados_para_treinamento', 'training_features.csv')
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {features_path}")
        return pd.read_csv(features_path)
        
    def train_model(self):
        features_df = self.load_features()
        if 'target' not in features_df.columns:
            raise ValueError("Coluna 'target' ausente.")
            
        X = features_df.drop(['target', 'nome_arquivo', 'nome_coluna'], axis=1, errors='ignore')
        y = features_df['target']
        
        self.feature_columns = list(X.columns)
        
        X = X.fillna(0)
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.fillna(0)
        
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Validação Cruzada
        print("\nIniciando Validação Cruzada (5-Folds)...")
        cv_model = lgb.LGBMClassifier(random_state=42, class_weight='balanced', verbose=-1)
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        try:
            scores = cross_val_score(cv_model, X, y_encoded, cv=kfold, scoring='accuracy')
            print(f"Acurácia Média CV: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        except Exception as e:
            print(f"Erro na validação cruzada: {e}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
            
        self.model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        
        y_pred = self.model.predict(X_test)
        print(f"\nAcurácia Teste Final: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        return X_test, y_test, y_pred
        
    def save_model(self):
        modelos_dir = os.path.join(self.base_dir, 'modelos_salvos')
        os.makedirs(modelos_dir, exist_ok=True)
        model_path = os.path.join(modelos_dir, 'semantic_type_classifier.pkl')
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns
        }
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Modelo salvo em: {model_path}")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_model()
    trainer.save_model()