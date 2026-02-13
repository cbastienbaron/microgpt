# microGPT-PHP

Port en PHP pur du [microgpt](https://karpathy.ai/microgpt.html) d'Andrej Karpathy — la facon la plus minimale d'entrainer et d'executer un GPT sans aucune dependance.

Le modele entraine un petit Transformer au niveau des caracteres sur ~29K prenoms, puis genere de nouveaux noms inventes par echantillonnage autoregressif.

## Lancement

```bash
php microgpt.php
```

Le jeu de donnees est telecharge automatiquement au premier lancement. Necessite PHP 8.1+.

## Comment ca marche

Tout tient dans un seul fichier (`microgpt.php`), organise en 8 classes. Voici le role de chacune :

### `Value` — Le moteur d'auto-differentiation

C'est le coeur du systeme. Chaque nombre dans le reseau de neurones est enveloppe dans un objet `Value` qui memorise comment il a ete calcule.

```php
$a = new Value(3.0);
$b = new Value(2.0);
$c = $a->mul($b);     // c = 6.0, et c "sait" qu'il vient de a * b
$c->backward();        // calcule les gradients : a.grad = 2.0, b.grad = 3.0
```

Quand on appelle `backward()`, l'algorithme remonte tout le graphe de calcul en sens inverse (retropropagation) pour calculer le gradient de chaque parametre. C'est ce gradient qui indique au modele comment ajuster ses poids pour reduire l'erreur.

**Operations supportees** : `add`, `mul`, `sub`, `div`, `pow_`, `log_`, `exp_`, `relu`, `neg`

> En Python on ecrit `a + b`. En PHP, pas de surcharge d'operateurs, donc on ecrit `$a->add($b)`.

---

### `Tokenizer` — Convertir du texte en nombres

Un reseau de neurones ne comprend pas les lettres, seulement les nombres. Le `Tokenizer` fait la traduction :

```
"emma" → [26, 4, 12, 12, 0, 26]
          BOS  e   m   m   a  BOS
```

Chaque caractere unique recoit un identifiant. Le token special `BOS` (Beginning Of Sequence) marque le debut et la fin d'un mot.

---

### `MathHelpers` — Boite a outils mathematique

Fonctions utilitaires utilisees partout dans le modele :

| Methode | Role |
|---|---|
| `gauss()` | Genere un nombre aleatoire gaussien (pour initialiser les poids) |
| `matrix()` | Cree une matrice de `Value` avec des poids aleatoires |
| `linear()` | Multiplication matrice-vecteur (`y = W * x`) — c'est l'operation de base d'une couche de neurones |
| `softmax()` | Transforme des scores bruts en probabilites (somme = 1) |
| `rmsnorm()` | Normalise un vecteur pour stabiliser l'entrainement |
| `weighted_choice()` | Tire un index au hasard selon des poids de probabilite |

---

### `GPTConfig` — Les hyperparametres

Definit la taille du modele :

```php
$config = new GPTConfig(
    n_embd: 16,      // Dimension des vecteurs internes
    n_head: 4,       // Nombre de tetes d'attention
    n_layer: 1,      // Nombre de couches Transformer
    block_size: 16,  // Fenetre de contexte (nb max de tokens)
);
```

De petites valeurs = modele petit et rapide (mais moins puissant). Le but ici est pedagogique.

---

### `GPTModel` — Le Transformer

C'est le modele lui-meme. Sa methode `forward()` prend un token et sa position, puis retourne un score pour chaque token possible du vocabulaire (les "logits").

Le traitement pour chaque token suit ce chemin :

```
Token + Position
       |
   Embeddings (transformer le numero en vecteur)
       |
   RMS Norm (normaliser)
       |
   ┌───────────────────────────┐
   │ Attention Multi-Tetes     │  "A quels tokens precedents
   │ Q = ce qu'on cherche      │   dois-je faire attention ?"
   │ K = ce qu'on propose      │
   │ V = l'info a extraire     │
   └───────────┬───────────────┘
               + connexion residuelle
       |
   ┌───────────────────────────┐
   │ MLP (Feed-Forward)        │  Transformation non-lineaire
   │ Linear → ReLU → Linear   │  qui ajoute de la capacite
   └───────────┬───────────────┘
               + connexion residuelle
       |
   Projection vers le vocabulaire (lm_head)
       |
   Logits (scores pour chaque token possible)
```

Le modele utilise un **cache KV** : les cles (K) et valeurs (V) deja calculees pour les tokens precedents sont gardees en memoire pour ne pas les recalculer.

---

### `AdamOptimizer` — L'optimiseur

Apres chaque pas d'entrainement, l'optimiseur ajuste les poids du modele pour reduire l'erreur. Adam est l'algorithme standard. Il combine :

- **Momentum** (1er moment) : une moyenne mobile du gradient, pour garder l'elan dans la bonne direction
- **Variance** (2eme moment) : une moyenne mobile du gradient au carre, pour adapter le pas d'apprentissage a chaque parametre

Le taux d'apprentissage diminue lineairement au fil de l'entrainement (learning rate decay).

---

### `Trainer` — La boucle d'entrainement

Orchestre le processus d'apprentissage. Pour chaque pas :

1. **Selectionne** un mot du jeu de donnees
2. **Encode** le mot en tokens
3. **Passe avant** : pour chaque position, le modele predit le prochain token
4. **Calcule la loss** : cross-entropy = `-log(probabilite du bon token)`. Plus le modele est sur du bon token, plus la loss est basse
5. **Retropropagation** : `backward()` calcule les gradients
6. **Mise a jour** : l'optimiseur ajuste les poids

Apres 1000 pas, la loss devrait avoir significativement diminue.

---

### `Generator` — La generation de texte

Apres l'entrainement, on genere de nouveaux noms. Le processus est **autoregressif** :

1. Commencer par le token `BOS`
2. Le modele donne les probabilites pour le prochain token
3. On en choisit un au hasard (selon les probabilites)
4. On le rajoute a la sequence et on repete
5. On s'arrete quand le modele genere `BOS` (fin du mot)

La **temperature** controle la creativite :
- `0.1` = tres conservateur (prend presque toujours le token le plus probable)
- `0.5` = equilibre (valeur par defaut)
- `1.5` = tres creatif (resultats plus surprenants, parfois incoherents)

---

## Exemple de sortie

```
nombre de documents : 32033
taille du vocabulaire : 27
nombre de parametres : 7723

--- entrainement ---
step    1 / 1000 | loss 3.3412
step    2 / 1000 | loss 3.3574
...
step 1000 / 1000 | loss 2.1053

--- inference (nouveaux noms hallucines) ---
echantillon  1: kayla
echantillon  2: jori
echantillon  3: malena
...
```

## Architecture

```
microgpt/
├── microgpt.php   # Tout le code (autograd + modele + entrainement + inference)
├── CLAUDE.md      # Guide pour l'assistant IA
├── README.md      # Ce fichier
└── input.txt      # Jeu de donnees (telecharge automatiquement)
```

## Credits

Port PHP de [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) par [Andrej Karpathy](https://karpathy.ai/microgpt.html).
