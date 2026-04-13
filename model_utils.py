import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def split_age_groups(df, groups_dict):
    """
    Divide o DataFrame com base no dicionário de faixas etárias.
    Utiliza 'idade_raw' para o filtro e retorna apenas a 'idade' escalonada.
    """
    processed_groups = {}
    for name, (min_age, max_age) in groups_dict.items():
        mask = (df['idade_raw'] >= min_age) & (df['idade_raw'] < max_age)
        # Removemos idade_raw para o modelo não sofrer com redundância
        processed_groups[name] = df[mask].drop(columns=['idade_raw'])
    return processed_groups

def run_random_forest_pipeline(X_train, y_train, group_name, param_grid=None):
    """
    Executa GridSearchCV e retorna o melhor estimador e seus parâmetros.
    Se param_grid for None, usa um padrão seguro.
    """
    print(f'\n🔍 Iniciando Grid Search: {group_name}')
    rf = RandomForestClassifier(criterion='entropy', random_state=42, 
                                class_weight='balanced', n_jobs=-1)

    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }

    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring="roc_auc", verbose=1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_

def save_model_assets(model, params, group_name):
    """Persiste o modelo (.pkl) e os parâmetros (.json) em disco."""
    os.makedirs('models', exist_ok=True)
    
    # Salva o binário do modelo
    joblib.dump(model, f'models/rf_{group_name.replace("-","")}.pkl')
    
    # Salva os parâmetros para documentação futura
    with open(f'models/params_{group_name.replace("-","")}.json', 'w') as f:
        json.dump(params, f, indent=4)
    print(f"💾 Assets de '{group_name}' salvos em /models")

def get_model_predictions(model, X_test):
    """Retorna predições de classe e probabilidades."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return y_pred, y_proba

def calculate_metrics(y_test, y_pred, y_proba):
    """Calcula métricas principais e retorna como dicionário."""
    report = classification_report(y_test, y_pred, output_dict=True)
    report['roc_auc'] = roc_auc_score(y_test, y_proba)
    return report

def export_visual_reports(model, X_test, y_test, y_pred, group_name):
    """Gera e salva Matriz de Confusão e Feature Importance sem poluir o notebook."""
    os.makedirs('reports', exist_ok=True)
    
    # 1. Matriz de Confusão
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
    plt.title(f'Matriz de Confusão - {group_name}')
    plt.savefig(f'reports/cm_{group_name.replace("-","")}.png')
    plt.close()

    # 2. Feature Importance
    plt.figure(figsize=(10, 6))
    importances = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=True)
    importances.plot(kind='barh', color='indigo')
    plt.title(f"Importância de Atributos - {group_name}")
    plt.tight_layout()
    plt.savefig(f'reports/features_{group_name.replace("-","")}.png')
    plt.close()