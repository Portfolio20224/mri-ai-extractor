import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score,
    accuracy_score,
    matthews_corrcoef,
    balanced_accuracy_score
)

class AnnotationEvaluator:
    """Classe chargée de l'évaluation statistique et de la visualisation des résultats d'annotation."""
    
    def __init__(self, labels: list):
        """
        Initialise l'évaluateur.
        :param labels: Liste des noms des colonnes (pathologies) à évaluer.
        """
        self.labels = labels

    def _evaluate_single_label(self, gold_df: pd.DataFrame, pred_df: pd.DataFrame, label: str) -> dict:
        """Calcule les métriques pour un label donné."""
        y_true = gold_df[label].values
        y_pred = pred_df[label].values
        
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        except ValueError as e:
            print(f"⚠️ Erreur sur {label}: {e}")
            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        
        return {
            'label': label,
            'n_gold_pos': int(y_true.sum()),
            'n_gold_neg': int(len(y_true) - y_true.sum()),
            'n_pred_pos': int(y_pred.sum()),
            'n_pred_neg': int(len(y_pred) - y_pred.sum()),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'mcc': mcc
        }

    def evaluate(self, df_gold: pd.DataFrame, df_pred: pd.DataFrame, output_dir: str = "out/experiment") -> dict:
        """Évalue l'ensemble des prédictions contre le Gold Standard et génère des rapports."""
        os.makedirs(output_dir, exist_ok=True)
        metrics_results = []
        confusion_matrices = {}
        
        for label in self.labels:
            valid = df_pred[label] != -1
            n_valid = valid.sum()
            
            if n_valid == 0:
                print(f"⚠️ Aucune prédiction valide pour {label}")
                continue
            
            y_true = df_gold.loc[valid, label].values
            y_pred = df_pred.loc[valid, label].values
            
            unique_true = np.unique(y_true)
            unique_pred = np.unique(y_pred)
            
            if not all(x in [0, 1] for x in unique_true):
                y_true = np.clip(y_true, 0, 1)
            
            if not all(x in [0, 1] for x in unique_pred):
                y_pred = np.clip(y_pred, 0, 1)
            
            metrics = self._evaluate_single_label(df_gold.loc[valid], df_pred.loc[valid], label)
            metrics_results.append(metrics)
            
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            confusion_matrices[label] = cm
        
        df_metrics = pd.DataFrame(metrics_results)
        
        all_y_true, all_y_pred = [], []
        for label in self.labels:
            valid = df_pred[label] != -1
            if valid.sum() > 0:
                all_y_true.extend(df_gold.loc[valid, label].values)
                all_y_pred.extend(df_pred.loc[valid, label].values)
        
        global_metrics = {}
        if all_y_true:
            global_metrics = {
                'precision': precision_score(all_y_true, all_y_pred, average='macro', zero_division=0),
                'recall': recall_score(all_y_true, all_y_pred, average='macro', zero_division=0),
                'f1_macro': f1_score(all_y_true, all_y_pred, average='macro', zero_division=0),
                'f1_micro': f1_score(all_y_true, all_y_pred, average='micro', zero_division=0),
                'balanced_accuracy': balanced_accuracy_score(all_y_true, all_y_pred),
                'mcc': matthews_corrcoef(all_y_true, all_y_pred),
                'accuracy': accuracy_score(all_y_true, all_y_pred)
            }
        
        self._print_reports(df_metrics, global_metrics, df_pred)
        
        results = {
            'df_metrics': df_metrics,
            'global_metrics': global_metrics,
            'confusion_matrices': confusion_matrices,
            'df_pred': df_pred,
            'df_gold': df_gold
        }
        
        self.plot_confusion_matrices(confusion_matrices, output_dir)
        self.plot_performance_summary(results, output_dir)
        
        return results

    def _print_reports(self, df_metrics: pd.DataFrame, global_metrics: dict, df_pred: pd.DataFrame):
        """Affiche les rapports statistiques dans la console."""
        print("\n" + "="*60)
        print("📊 RÉSULTATS PAR LABEL")
        print("="*60)
        
        if not df_metrics.empty:
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 200)
            display_cols = ['label', 'n_gold_pos', 'n_gold_neg', 'n_pred_pos', 'n_pred_neg',
                           'tp', 'fp', 'tn', 'fn', 'precision', 'recall', 'f1', 
                           'balanced_accuracy', 'mcc']
            print(df_metrics[display_cols].to_string(index=False))
        
        if global_metrics:
            print("\n" + "="*60)
            print("📊 MÉTRIQUES GLOBALES")
            print("="*60)
            for metric, value in global_metrics.items():
                print(f"  {metric:20} : {value:.4f}")
                
        if not df_metrics.empty:
            print("\n" + "="*60)
            print("🔍 ANALYSE DES ERREURS")
            print("="*60)
            df_metrics['f1_rank'] = df_metrics['f1'].rank()
            worst_labels = df_metrics.nsmallest(3, 'f1')
            
            print("\n🔴 Labels les plus difficiles (F1 le plus bas):")
            for _, row in worst_labels.iterrows():
                print(f"  {row['label']:20} → F1: {row['f1']:.3f}, Recall: {row['recall']:.3f}, Precision: {row['precision']:.3f}")
            
            n_success = df_pred['success'].sum()
            n_total = len(df_pred)
            print(f"\n✅ Taux de succès du LLM: {n_success}/{n_total} ({n_success/n_total*100:.1f}%)")

    def plot_confusion_matrices(self, confusion_matrices: dict, output_dir: str):
        """Génère et sauvegarde la mosaïque des matrices de confusion."""
        if not confusion_matrices: return
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        for idx, label in enumerate(self.labels):
            if label in confusion_matrices:
                cm = confusion_matrices[label]
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                            xticklabels=['Négatif (0)', 'Positif (1)'],
                            yticklabels=['Négatif (0)', 'Positif (1)'])
                axes[idx].set_title(f'{label}')
                axes[idx].set_xlabel('Prédit')
                axes[idx].set_ylabel('Réel')
            else:
                axes[idx].set_visible(False)
                
        for idx in range(len(self.labels), len(axes)):
            axes[idx].set_visible(False)
            
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrices.png', dpi=150)
        plt.close() # Libère la mémoire

    def plot_performance_summary(self, experiment_results: dict, output_dir: str):
        """Visualise la performance par label (F1, Precision, Recall)."""
        df_metrics = experiment_results['df_metrics']
        if df_metrics.empty: return
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        df_plot = df_metrics.set_index('label')[['precision', 'recall', 'f1']]
        df_plot.plot(kind='bar', ax=axes[0], rot=45)
        axes[0].set_title('Performance par label')
        axes[0].set_ylabel('Score')
        axes[0].legend(loc='lower right')
        axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Seuil 0.5')
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.3)
        
        x = np.arange(len(self.labels))
        width = 0.35
        gold_pos = [experiment_results['df_gold'][label].sum() for label in self.labels]
        pred_pos = [experiment_results['df_pred'][label].sum() for label in self.labels]
        
        axes[1].bar(x - width/2, gold_pos, width, label='Gold', color='blue', alpha=0.7)
        axes[1].bar(x + width/2, pred_pos, width, label='LLM', color='orange', alpha=0.7)
        axes[1].set_xlabel('Labels')
        axes[1].set_ylabel('Nombre de positifs')
        axes[1].set_title('Distribution des positifs: Gold vs LLM')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(self.labels, rotation=45)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/performance_summary.png', dpi=150)
        plt.close()