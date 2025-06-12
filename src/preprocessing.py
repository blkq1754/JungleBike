# src/preprocessing.py
import spacy

# Charger spaCy une seule fois
try:
    nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])
except OSError:
    print("Modèle spaCy 'fr_core_news_sm' non trouvé. Veuillez le télécharger avec :\npython -m spacy download fr_core_news_sm")
    exit()

def spacy_tokenizer(text):
    """
    Tokenizer personnalisé qui nettoie, supprime les stop words,
    et lemmatise le texte.
    """
    doc = nlp(text)
    lemmas = [
        token.lemma_.lower().strip() 
        for token in doc 
        if not token.is_stop and not token.is_punct and not token.is_space
    ]
    return lemmas