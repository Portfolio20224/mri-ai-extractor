import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from src.annotator import MedicalLLMAnnotator
from src.pipeline import WeakSupervisionPipeline
from src.evaluator import AnnotationEvaluator
from src.config import LABELS


load_dotenv()

if __name__ == "__main__":

    llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0.0)
    annotator = MedicalLLMAnnotator(llm=llm, max_attempts=3)
    pipeline = WeakSupervisionPipeline(annotator=annotator, output_dir="out/")

    # (Optionnel) Évaluation si on passe le Gold Standard
    df_gold = pd.read_csv("data/gold-set.csv")
    
    # print("Chargement des données...")
    # df_train = pd.read_csv("data/train.csv")
    
    df_predictions = pipeline.run(df_gold)
    evaluator = AnnotationEvaluator(labels=LABELS)
    results = evaluator.evaluate(df_gold, df_predictions, output_dir="out/")
