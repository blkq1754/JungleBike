import torch.nn as nn
from transformers import CamembertModel

class ProductClassifier(nn.Module):
    """
charger un modele CamemBERT pré-entraîné et ajouter une couche de classification linéaire.
    """
    def __init__(self, n_classes: int):
        """
        Initialise le modèle.

        Args:
            n_classes (int): Le nombre de catégories à prédire.
        """
        super(ProductClassifier, self).__init__()
        
        # Charger le modele CamemBERT pré-entraîné
        self.camembert = CamembertModel.from_pretrained('camembert-base')
        
        # une couche Dropout pour réduire le risque de sur-apprentissage.
        self.dropout = nn.Dropout(0.3)
        
        # La couche qui fait la classification finale.
        # - in_features: La taille de la sortie de camemBERT.
        # - out_features: Le nombre de nos catégories.
        self.classifier = nn.Linear(self.camembert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        """
        Définit le passage des données à travers les couches du modèle (la prédiction).
        """
        #  Passage des données dans CamemBERT
        output = self.camembert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # le "pooler_output" représente une synthèse de toute la phrase.
        pooled_output = output.pooler_output
        
        #Dropout
        pooled_output = self.dropout(pooled_output)
        
        #classification finale
        logits = self.classifier(pooled_output)
        
        return logits