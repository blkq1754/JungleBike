from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
# Importer notre tokenizer spacy
from src.preprocessing import spacy_tokenizer

def create_pipeline():
    """Crée et retourne le pipeline scikit-learn."""
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            tokenizer=spacy_tokenizer, 
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.9
        )),
        ('clf', LogisticRegression(
            C=1.0, 
            random_state=42,
            max_iter=2000,
            solver='lbfgs',
            multi_class='multinomial'
        ))
    ])
    return model_pipeline