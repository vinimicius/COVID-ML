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
    Divide o DataFrame e retorna os grupos processados E um dicionário de estatísticas.
    Exibe um log detalhado da distribuição das amostras.
    """
    processed_groups = {}
    group_stats = {}
    
    print(f"\n{'='*60}")
    print(f"{'📊 RELATÓRIO DE DIVISÃO POR FAIXA ETÁRIA':^60}")
    print(f"{'='*60}")
    
    for name, (min_age, max_age) in groups_dict.items():
        # 1. Filtro
        mask = (df['idade_raw'] >= min_age) & (df['idade_raw'] < max_age)
        group_df = df[mask].copy()
        
        if len(group_df) > 0:
            # 2. Cálculo de Estatísticas (Metadados)
            total = len(group_df)
            obitos = int(group_df['obito'].sum())
            prevalence = round((obitos / total) * 100, 2)
            
            group_stats[name] = {
                "n_samples": total,
                "n_obitos": obitos,
                "prevalence_percent": prevalence,
                "age_range": [min_age, max_age]
            }
            
            # Log individual
            print(f"🔹 {name.upper():<15} | Range: [{min_age:>2}, {max_age:>3}) | Amostras: {total:>6} | Óbitos: {obitos:>4} ({prevalence:>5}%)")
            
            # 3. Preparação para o modelo (Removemos idade_raw)
            processed_groups[name] = group_df.drop(columns=['idade_raw'])
        else:
            print(f"⚠️  {name.upper():<15} | Range: [{min_age:>2}, {max_age:>3}) | GRUPO VAZIO")

    print(f"{'='*60}\n")
    return processed_groups, group_stats



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



def get_model_predictions(model, X_test):
    """Retorna predições de classe e probabilidades."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return y_pred, y_proba



def display_group_report(y_test, y_pred, y_proba, group_name):
    """
    Exibe o relatório, calcula AUC e retorna um DataFrame estruturado.
    Recebe as predições prontas para evitar dependência do objeto do modelo.
    """

    # 1. Exibição Visual
    print(f"\n" + "="*50)
    print(f"📊 RELATÓRIO DE CLASSIFICAÇÃO: {group_name.upper()}")
    print("="*50)
    print(classification_report(y_test, y_pred))
    
    # 2. Cálculo do Dicionário
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    
    # 3. Adiciona a métrica que o classification_report não traz por padrão
    report_dict['roc_auc'] = roc_auc_score(y_test, y_proba)
    
    # 4. Transformação em DataFrame para o seu dicionário 'all_reports'
    report_df = pd.DataFrame(report_dict).transpose()
    report_df['group'] = group_name
    
    return report_df



def export_visual_reports(model, X_test, y_test, y_pred, group_name, save_path=None):
    """Gera e salva Matriz de Confusão e Feature Importance."""
    # Se não passarmos caminho, ele usa a pasta 'reports' na raiz
    target_dir = save_path if save_path else 'reports'
    os.makedirs(target_dir, exist_ok=True)
    
    clean_name = group_name.replace("-", "")
    
    # 1. Matriz de Confusão
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
    plt.title(f'Matriz de Confusão - {group_name}')
    plt.savefig(os.path.join(target_dir, f'cm_{clean_name}.png'))
    plt.close()

    # 2. Feature Importance
    plt.figure(figsize=(10, 6))
    importances = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=True)
    importances.plot(kind='barh', color='indigo')
    plt.title(f"Importância de Atributos - {group_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(target_dir, f'features_{clean_name}.png'))
    plt.close()