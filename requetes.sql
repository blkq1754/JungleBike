
/*
  Hypothèses  : 
  Une "dépense" est considérée comme une transaction de type 'achat'.
  "produit_id" est présent dans la table "transactions" et fait référence à un produit. (Le script init_dataset.sql)
  La période des 12 derniers mois est calculée à partir de la date la plus récente dans la table
*/

--  1. Dépenses mensuelles par client 
WITH DerniereDate AS (
  SELECT MAX(date) AS max_date FROM transactions
)
SELECT
  t.client_id,
  EXTRACT(YEAR FROM t.date) AS annee,
  EXTRACT(MONTH FROM t.date) AS mois,
  SUM(t.montant) AS total_mensuel
FROM
  transactions t
WHERE
  t.type_transaction = 'achat'
  AND t.date >= (SELECT max_date FROM DerniereDate) - INTERVAL '12 month'
GROUP BY
  t.client_id, annee, mois
ORDER BY
  t.client_id, annee DESC, mois DESC;


--  2. Top clients Assurance Vie
SELECT
  c.client_id,
  c.nom,
  -- On arrondit la moyenne 
  ROUND(AVG(t.montant), 2) AS depense_moyenne
FROM
  transactions t
JOIN
  clients c ON t.client_id = c.client_id
JOIN
  produits p ON t.produit_id = p.produit_id
WHERE
  p.categorie = 'Assurance Vie'
  AND t.type_transaction = 'achat'
GROUP BY
  c.client_id, c.nom
HAVING
  COUNT(t.transaction_id) >= 3   
ORDER BY
  depense_moyenne DESC
LIMIT 10;



-- 3. Détection de transactions suspectes

SELECT
  client_id,
  date,
  COUNT(transaction_id) AS nombre_transactions
FROM
  transactions
WHERE
  montant > 10000
GROUP BY
  client_id, date
HAVING
  COUNT(transaction_id) > 3
ORDER BY
  nombre_transactions DESC;



-- 4. Optimisation 

CREATE INDEX IF NOT EXISTS index_transactions_client_id ON transactions(client_id);
CREATE INDEX IF NOT EXISTS index_transactions_produit_id ON transactions(produit_id);
CREATE INDEX IF NOT EXISTS index_produits_categorie ON produits(categorie);



--  `index_transactions_client_id` et `index_transactions_produit_id`:`
--    accélèrer les jointure avec les tables `clients` et `produits`.

--  `index_produits_categorie`:
--    pour la clause `WHERE p.categorie = 'Assurance Vie'`.

