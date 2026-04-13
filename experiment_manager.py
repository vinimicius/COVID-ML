import os
import json
import joblib
import pandas as pd
from datetime import datetime

class ExperimentManager:
    def __init__(self, experiment_name="RF_Experiment", base_path="models"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.run_id = f"run_{self.timestamp}"
        
        # Caminhos absolutos ou relativos consistentes
        self.base_path = os.path.join(base_path, self.run_id)
        self.reports_path = os.path.join("reports", self.run_id)
        
        self.experiment_name = experiment_name
        self.results = {}
        
        # Garante a criação das pastas
        os.makedirs(self.base_path, exist_ok=True)
        os.makedirs(self.reports_path, exist_ok=True)
        
        print(f"🚀 Gerente de Experimentos: {self.experiment_name}")
        print(f"📂 Modelos em: {self.base_path}")

    def save_group(self, group_name, model, params, report_df):
        """Salva artefatos e limpa métricas para o JSON Master."""
        # Salva o binário do modelo
        model_name = f"rf_{group_name.replace('-', '')}.pkl"
        joblib.dump(model, os.path.join(self.base_path, model_name))
        
        # Salva os parâmetros individuais
        params_name = f"params_{group_name.replace('-', '')}.json"
        with open(os.path.join(self.base_path, params_name), 'w') as f:
            json.dump(params, f, indent=4)
            
        # Armazena os resultados limpos na memória da classe
        self.results[group_name] = self._clean_report(report_df)

    def _clean_report(self, df):
        """Extrai métricas essenciais e remove redundâncias de matrizes."""
        # Se vier como DataFrame do pandas (do all_reports antigo)
        if isinstance(df, pd.DataFrame):
            acc = df.loc['accuracy', 'precision'] if 'accuracy' in df.index else None
            auc = df.loc['roc_auc', 'precision'] if 'roc_auc' in df.index else None
            
            return {
                "metrics_per_class": {
                    "sobrevivente": df.loc['0'].drop('group', errors='ignore').to_dict(),
                    "obito": df.loc['1'].drop('group', errors='ignore').to_dict()
                },
                "global_performance": {
                    "accuracy": acc,
                    "roc_auc": auc
                }
            }
        return df # Caso já venha limpo

    def finalize(self, metadata=None):
        """Consolida tudo em um JSON Master."""
        final_json = {
            "metadata": {
                "experiment_name": self.experiment_name,
                "timestamp": self.timestamp,
                "config": metadata
            },
            "results": self.results
        }
        
        file_path = os.path.join(self.base_path, "full_experiment_results.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(final_json, f, indent=4, ensure_ascii=False)
        
        print(f"🏁 Relatório Final Gerado: {file_path}")