import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict, Optional
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """A class to perform sentiment analysis on text data using Hugging Face Transformers."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        """Initialize the sentiment analyzer with a pre-trained model.
        
        Args:
            model_name (str): Name of the pre-trained model from Hugging Face.
        """
        self.model_name = model_name
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.pipeline: Optional[pipeline] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self) -> None:
        """Load the pre-trained model and tokenizer."""
        try:
            logger.info(f"Loading model {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
            self.pipeline = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, device=0 if self.device.type == "cuda" else -1)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def analyze_sentiment(self, texts: List[str]) -> List[Dict[str, str]]:
        """Analyze sentiment for a list of texts.
        
        Args:
            texts (List[str]): List of text inputs (e.g., X posts).
        
        Returns:
            List[Dict[str, str]]: List of dictionaries with text, label, and score.
        """
        if not texts:
            raise ValueError("Input text list is empty")
        
        try:
            logger.info(f"Analyzing sentiment for {len(texts)} texts")
            results = self.pipeline(texts)
            return [{"text": text, "label": res["label"], "score": f"{res['score']:.4f}"} for text, res in zip(texts, results)]
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            raise

    def process_data(self, input_file: str, output_file: str) -> None:
        """Process a CSV file with text data and save sentiment analysis results.
        
        Args:
            input_file (str): Path to input CSV with a 'text' column.
            output_file (str): Path to save results CSV.
        """
        try:
            logger.info(f"Reading data from {input_file}")
            df = pd.read_csv(input_file)
            if 'text' not in df.columns:
                raise ValueError("Input CSV must have a 'text' column")
            
            texts = df['text'].tolist()
            results = self.analyze_sentiment(texts)
            
            result_df = pd.DataFrame(results)
            result_df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise

def main():
    """Main function to run the sentiment analyzer."""
    input_file = "data/sample_posts.csv"
    output_file = "data/sentiment_results.csv"
    
    # Fallback sample texts if input file is missing
    fallback_texts = [
        "I love the new features on this platform!",
        "Not happy with the latest update.",
        "This app is amazing and super fast!",
        "Why is the service down again?",
        "Excited for the new AI tools!"
    ]
    
    analyzer = SentimentAnalyzer()
    
    # Check if input file exists, use fallback if not
    try:
        if not os.path.exists(input_file):
            logger.warning(f"Input file {input_file} not found. Using fallback sample texts.")
            results = analyzer.analyze_sentiment(fallback_texts)
            result_df = pd.DataFrame(results)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            result_df.to_csv(output_file, index=False)
            logger.info(f"Results saved to {output_file} using fallback texts")
        else:
            analyzer.process_data(input_file, output_file)
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    main()
