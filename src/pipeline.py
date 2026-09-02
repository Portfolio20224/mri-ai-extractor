import os
import time
import pandas as pd
from tqdm import tqdm
from src.config import LABELS

class WeakSupervisionPipeline:
    """Orchestrateur gérant le traitement de masse, le throttling et les checkpoints."""
    
    def __init__(self, annotator, output_dir="out/experiment"):
        self.annotator = annotator
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(output_dir, 'llm_predictions_checkpoint.csv')
        self.final_file = os.path.join(output_dir, 'llm_predictions_final.csv')

    def run(self, df_data: pd.DataFrame, min_request_interval: float = 4.2) -> pd.DataFrame:
        print(f"\n🚀 DÉMARRAGE DE L'ANNOTATION MASSIVE ({len(df_data)} RAPPORTS)")
        
        results, processed_uids = [], set()
        
        if os.path.exists(self.checkpoint_file):
            df_checkpoint = pd.read_csv(self.checkpoint_file)
            results = df_checkpoint.to_dict('records')
            processed_uids = set(df_checkpoint['StudyInstanceUID'].tolist())
            print(f"✅ Reprise activée : {len(processed_uids)} rapports déjà traités.")
            
        for idx, row in tqdm(df_data.iterrows(), total=len(df_data), desc="Processing"):
            uid = row.get('StudyInstanceUID', idx)
            if uid in processed_uids: continue
                
            start_time = time.time()
            data = self.annotator.annotate(row.get('Report', ''))
            
            # Formater les labels
            parsed_labels = {lbl: data['labels'].get(lbl, -1) if data and data.get('labels') else -1 for lbl in LABELS}
            
            results.append({
                'StudyInstanceUID': uid,
                **parsed_labels,
                'reasoning': data.get('reasoning', "Échec") if data else "Échec",
                'confidence_rate': data.get('confidence_rate', 0) if data else 0,
                'success': bool(data)
            })
            processed_uids.add(uid)
            
            # Throttling
            elapsed = time.time() - start_time
            if elapsed < min_request_interval and data:
                time.sleep(min_request_interval - elapsed)
                
            # Checkpoint
            if len(processed_uids) % 50 == 0:
                pd.DataFrame(results).to_csv(self.checkpoint_file, index=False)
                
        df_final = pd.DataFrame(results)
        df_final.to_csv(self.final_file, index=False)
        print(f"🎉 Terminé ! Sauvegardé dans {self.final_file}")
        return df_final