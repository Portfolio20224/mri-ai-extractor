import json
import re
import time
from tqdm import tqdm
import pandas as pd
from src.config import EXPERIMENT_PROMPT

class MedicalLLMAnnotator:
    """Composant IA gérant l'extraction sémantique et la résilience API (Quotas)."""
    
    def __init__(self, llm, max_attempts: int = 3):
        self.llm = llm
        self.max_attempts = max_attempts

    def _extract_json(self, response: str) -> dict:
        """Nettoie et parse la réponse LLM sécurisée."""
        if not response: return None
        response = re.sub(r'^```(?:json)?\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'\s*```$', '', response)
        
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != 0:
                data = json.loads(response[start:end])
                if 'labels' in data: return data
        except json.JSONDecodeError:
            pass
        return None

    def annotate(self, input_text: str) -> dict:
        """Exécute l'annotation avec gestion intelligente des erreurs (ex: 429 Quota)."""
        if pd.isna(input_text) or not str(input_text).strip():
            return None
            
        safe_text = str(input_text)[:4000] + "..." if len(str(input_text)) > 4000 else str(input_text)
        prompt = EXPERIMENT_PROMPT.format(input_text=safe_text)
        
        for attempt in range(self.max_attempts):
            try:
                response = self.llm.invoke(prompt).content.strip()
                return self._extract_json(response)
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "Quota" in error_msg:
                    match = re.search(r'retry in (\d+\.?\d*)s', error_msg)
                    wait_time = float(match.group(1)) + 1.0 if match else 60.0
                    tqdm.write(f"⏳ Quota 429 atteint. Pause de {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    tqdm.write(f"❌ Erreur (Tentative {attempt+1}): {error_msg}")
                    time.sleep(2)
        return None