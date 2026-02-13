# CLAUDE.md - Guide pour microgpt-php

## Vue d'ensemble

Port en PHP pur du [microgpt](https://karpathy.ai/microgpt.html) d'Andrej Karpathy — la facon la plus minimale d'entrainer et d'executer un GPT (Generative Pretrained Transformer) sans aucune dependance. L'original fait ~250 lignes de Python pur ; ceci est l'equivalent en PHP, structure en classes avec commentaires explicatifs en francais.

Le modele entraine un GPT au niveau des caracteres sur un jeu de donnees de prenoms, puis genere de nouveaux noms hallucines par echantillonnage autoregressif.

## Structure du depot

```
microgpt/
├── CLAUDE.md          # Ce fichier — guide pour l'assistant IA
├── microgpt.php       # L'implementation complete (fichier unique)
└── input.txt          # Donnees d'entrainement (telecharge automatiquement)
```

Un seul fichier source : `microgpt.php`. Tout est dedans — autograd, modele, entrainement, optimiseur, inference.

## Stack technique

- **Langage** : PHP 8.1+ (utilise les types union `Value|float` et les proprietes promues dans le constructeur)
- **Dependances** : Aucune. PHP pur, pas de Composer, pas d'extensions, pas de framework.
- **Execution** : CLI uniquement (`php microgpt.php`)

## Architecture

### Classes (dans l'ordre d'apparition dans `microgpt.php`)

1. **`Value`** (~120 lignes) — Moteur d'auto-differentiation (autograd). Chaque scalaire est enveloppe dans un `Value` qui trace le graphe de calcul pour la retropropagation. Operations : `add`, `mul`, `pow_`, `log_`, `exp_`, `relu`, `neg`, `sub`, `div`, et `backward()`.

2. **`Tokenizer`** — Tokenizer au niveau des caracteres. Chaque caractere = un token, plus un token BOS/EOS. Methodes : `encode()` (texte -> tokens), `decode()` (token -> caractere).

3. **`MathHelpers`** — Fonctions mathematiques statiques :
   - `gauss()` — Transformation de Box-Muller pour les nombres gaussiens
   - `matrix()` — Cree une matrice 2D de `Value` avec initialisation aleatoire
   - `linear()` — Multiplication matrice-vecteur (couche dense)
   - `softmax()` — Softmax numeriquement stable sur `Value[]`
   - `rmsnorm()` — Normalisation RMS (utilisee a la place de LayerNorm)
   - `weighted_choice()` — Echantillonnage aleatoire pondere

4. **`GPTConfig`** — Hyperparametres du modele :
   - `n_embd = 16` (dimension des embeddings)
   - `n_head = 4` (tetes d'attention)
   - `n_layer = 1` (couches Transformer)
   - `block_size = 16` (fenetre de contexte)

5. **`GPTModel`** — Le modele Transformer complet. Contient le `state_dict` (poids) et la methode `forward()` : embeddings token + position, RMS norm, attention multi-tetes causale avec cache KV, MLP feed-forward avec ReLU, et connexions residuelles.

6. **`AdamOptimizer`** — Optimiseur Adam avec decay lineaire du taux d'apprentissage. Gere les moyennes mobiles du gradient (1er et 2eme moment) avec correction de biais.

7. **`Trainer`** — Boucle d'entrainement. 1000 pas de prediction du prochain token avec cross-entropy loss, retropropagation, et mise a jour Adam.

8. **`TextGenerator`** — Generation de texte par echantillonnage autoregressif avec temperature.

### Choix architecturaux (identiques a l'original de Karpathy)

- **RMS Normalization** au lieu de Layer Normalization
- **Pas de biais** nulle part dans le modele
- **ReLU** (pas GeLU) dans le MLP
- **Tokenizer au niveau des caracteres** (chaque caractere = un token + BOS/EOS)
- **Cache KV** pendant l'entrainement et l'inference

## Execution

```bash
php microgpt.php
```

Au premier lancement, le jeu de donnees `names.txt` (~29K prenoms) est telecharge automatiquement depuis le repo makemore de Karpathy. L'entrainement tourne 1000 pas puis genere 20 noms. Pas besoin de GPU.

**Prerequis** : PHP 8.1+ avec `allow_url_fopen=On` (par defaut).

**Temps d'execution** : Intentionnellement lent — autograd scalaire pur sans vectorisation. Le but est pedagogique, pas performant.

## Conventions de developpement

### Style de code
- Fichier unique, architecture orientee objet
- Les commentaires explicatifs sont en francais
- PHPDoc `@param`/`@return` pour les types de tableaux
- Pas de namespaces, pas d'autoloading — minimalisme intentionnel
- Proprietes promues dans les constructeurs (PHP 8.1)

### Nommage
- Les noms PHP suivent l'original Python d'aussi pres que possible
- Les methodes qui masquent des noms PHP natifs ont un underscore : `pow_()`, `log_()`, `exp_()`
- Les methodes de `Value` sont des verbes explicites (`add`, `mul`, `div`) car PHP ne supporte pas la surcharge d'operateurs

### Differences cles avec l'original Python
- Les operations `Value` sont des appels de methodes (`$a->add($b)`) au lieu d'operateurs (`a + b`)
- Architecture en classes (OOP) au lieu de fonctions globales et variables globales
- `SplObjectStorage` remplace `set()` de Python pour le suivi d'identite dans `backward()`
- `gauss()` utilise Box-Muller car PHP n'a pas `random.gauss()`
- `weighted_choice()` remplace `random.choices()` de Python

### Tests
Il n'y a pas de suite de tests. La validite peut etre verifiee en :
1. Verifiant que la loss diminue au cours de l'entrainement
2. Comparant les noms generes avec la sortie de la version Python (meme seed)

### Extension
- Pour changer les hyperparametres, modifier l'instanciation de `GPTConfig`
- Pour un autre jeu de donnees, remplacer `input.txt` par un fichier texte avec un mot par ligne
- Pour changer le nombre de pas, modifier `$num_steps`
- Pour changer la temperature d'inference, modifier le parametre `temperature` dans `$generator->generate()`
