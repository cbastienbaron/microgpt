<?php
/**
 * microGPT en PHP pur — port du code de @karpathy (https://karpathy.ai/microgpt.html)
 *
 * La facon la plus minimale d'entrainer et d'executer un GPT (Generative Pretrained Transformer)
 * sans aucune dependance externe. Tout est dans ce fichier unique :
 *   - Un moteur d'auto-differentiation (autograd)
 *   - Un modele Transformer complet (attention multi-tetes, MLP, embeddings)
 *   - Un optimiseur Adam
 *   - Une boucle d'entrainement
 *   - Une phase d'inference (generation de texte)
 *
 * Le modele apprend a generer des prenoms a partir d'un jeu de donnees (~29K noms).
 *
 * Usage : php microgpt.php
 * Prerequis : PHP 8.1+ (pour les types union Value|float)
 */

// =============================================================================
// CLASSE VALUE : Le moteur d'auto-differentiation (autograd)
// =============================================================================
//
// Chaque nombre scalaire est enveloppe dans un objet Value qui :
//   1. Stocke la valeur numerique ($data)
//   2. Stocke le gradient ($grad) calcule pendant la retropropagation
//   3. Memorise ses "enfants" dans le graphe de calcul ($children)
//   4. Memorise les gradients locaux ($local_grads) pour la regle de chaine
//
// C'est le meme concept que PyTorch autograd, mais en version minimale.
// En Python, on peut ecrire a + b grace a la surcharge d'operateurs.
// En PHP, on utilise des methodes : $a->add($b), $a->mul($b), etc.
// =============================================================================

class Value
{
    public float $data;        // La valeur numerique
    public float $grad;        // Le gradient (derive partielle de la loss par rapport a cette valeur)

    /** @var Value[] Les noeuds parents dans le graphe de calcul */
    public array $children;

    /** @var float[] Les gradients locaux (derivees partielles de cette operation) */
    public array $local_grads;

    public function __construct(float $data, array $children = [], array $local_grads = [])
    {
        $this->data = $data;
        $this->grad = 0.0;
        $this->children = $children;
        $this->local_grads = $local_grads;
    }

    // --- Operations arithmetiques ---
    // Chaque operation cree un nouveau Value en memorisant :
    //   - le resultat du calcul (forward pass)
    //   - les enfants (pour savoir d'ou on vient)
    //   - les gradients locaux (derivees partielles pour la retropropagation)

    /** Addition : d(a+b)/da = 1, d(a+b)/db = 1 */
    public function add(Value|float $other): Value
    {
        if (!$other instanceof Value) $other = new Value($other);
        return new Value($this->data + $other->data, [$this, $other], [1.0, 1.0]);
    }

    /** Multiplication : d(a*b)/da = b, d(a*b)/db = a */
    public function mul(Value|float $other): Value
    {
        if (!$other instanceof Value) $other = new Value($other);
        return new Value($this->data * $other->data, [$this, $other], [$other->data, $this->data]);
    }

    /** Puissance : d(a^n)/da = n * a^(n-1) */
    public function pow_(float $exp): Value
    {
        return new Value($this->data ** $exp, [$this], [$exp * $this->data ** ($exp - 1)]);
    }

    /** Logarithme naturel : d(ln(a))/da = 1/a */
    public function log_(): Value
    {
        return new Value(log($this->data), [$this], [1.0 / $this->data]);
    }

    /** Exponentielle : d(e^a)/da = e^a */
    public function exp_(): Value
    {
        $e = exp($this->data);
        return new Value($e, [$this], [$e]);
    }

    /** ReLU (Rectified Linear Unit) : max(0, x). Gradient = 1 si x > 0, sinon 0 */
    public function relu(): Value
    {
        return new Value(max(0.0, $this->data), [$this], [$this->data > 0 ? 1.0 : 0.0]);
    }

    /** Negation : -a = a * (-1) */
    public function neg(): Value
    {
        return $this->mul(-1.0);
    }

    /** Soustraction : a - b = a + (-b) */
    public function sub(Value|float $other): Value
    {
        if (!$other instanceof Value) $other = new Value($other);
        return $this->add($other->neg());
    }

    /** Division : a / b = a * b^(-1) */
    public function div(Value|float $other): Value
    {
        if (!$other instanceof Value) $other = new Value($other);
        return $this->mul($other->pow_(-1));
    }

    /**
     * Retropropagation (backpropagation)
     *
     * Parcourt le graphe de calcul en ordre topologique inverse et propage
     * les gradients en appliquant la regle de chaine :
     *   gradient_enfant += gradient_local * gradient_parent
     *
     * C'est l'algorithme fondamental qui permet au reseau d'apprendre.
     */
    public function backward(): void
    {
        // 1. Construire l'ordre topologique (tri topologique)
        //    On visite d'abord les feuilles, puis les noeuds intermediaires
        $topo = [];
        $visited = new SplObjectStorage(); // Equivalent de set() en Python pour les objets

        $build = function (Value $v) use (&$build, &$topo, $visited) {
            if ($visited->contains($v)) return;
            $visited->attach($v);
            foreach ($v->children as $child) {
                $build($child);
            }
            $topo[] = $v;
        };

        $build($this);

        // 2. Propager les gradients en ordre inverse
        $this->grad = 1.0; // Le gradient de la loss par rapport a elle-meme vaut 1
        foreach (array_reverse($topo) as $v) {
            foreach ($v->children as $i => $child) {
                // Regle de chaine : accumuler le gradient
                $child->grad += $v->local_grads[$i] * $v->grad;
            }
        }
    }
}

// =============================================================================
// CLASSE TOKENIZER : Convertit le texte en nombres et inversement
// =============================================================================
//
// Un "tokenizer" au niveau des caracteres : chaque lettre = un token (nombre).
// On ajoute un token special BOS (Beginning Of Sequence) qui sert aussi de
// marqueur de fin (EOS). C'est la forme la plus simple de tokenisation.
// Par exemple : "abc" -> [BOS, 0, 1, 2, BOS]
// =============================================================================

class Tokenizer
{
    /** @var string[] Liste triee de tous les caracteres uniques du jeu de donnees */
    public array $chars;

    /** @var int Identifiant du token BOS/EOS (= nombre de caracteres uniques) */
    public int $bos;

    /** @var int Taille du vocabulaire (caracteres uniques + 1 pour BOS) */
    public int $vocab_size;

    /**
     * Construit le vocabulaire a partir des documents d'entrainement.
     * @param string[] $docs Liste de mots/phrases
     */
    public function __construct(array $docs)
    {
        // Extraire tous les caracteres uniques et les trier
        $all_chars = str_split(implode('', $docs));
        $this->chars = array_values(array_unique($all_chars));
        sort($this->chars);

        // Le token BOS est le dernier index (apres tous les caracteres)
        $this->bos = count($this->chars);
        $this->vocab_size = count($this->chars) + 1;
    }

    /**
     * Encode un mot en sequence de tokens : [BOS, c1, c2, ..., BOS]
     * @return int[]
     */
    public function encode(string $text): array
    {
        $tokens = [$this->bos];
        for ($i = 0; $i < strlen($text); $i++) {
            $tokens[] = array_search($text[$i], $this->chars, true);
        }
        $tokens[] = $this->bos;
        return $tokens;
    }

    /**
     * Decode un identifiant de token en caractere.
     * Retourne null si c'est le token BOS/EOS.
     */
    public function decode(int $token_id): ?string
    {
        if ($token_id === $this->bos) return null;
        return $this->chars[$token_id];
    }
}

// =============================================================================
// CLASSE MATH HELPERS : Fonctions mathematiques utilitaires
// =============================================================================
//
// Ces fonctions statiques effectuent les operations mathematiques de base
// sur les Value[] (vecteurs de valeurs auto-differentiables).
// =============================================================================

class MathHelpers
{
    /**
     * Genere un nombre aleatoire suivant une distribution gaussienne (normale).
     * Utilise la transformation de Box-Muller car PHP n'a pas de random.gauss().
     */
    public static function gauss(float $mean, float $std): float
    {
        $u1 = mt_rand() / mt_getrandmax();
        $u2 = mt_rand() / mt_getrandmax();
        return $mean + $std * sqrt(-2 * log($u1)) * cos(2 * M_PI * $u2);
    }

    /**
     * Cree une matrice (tableau 2D) de Value initialises aleatoirement.
     * C'est ainsi qu'on initialise les poids du reseau de neurones.
     *
     * @return Value[][] Matrice de dimensions [nout x nin]
     */
    public static function matrix(int $nout, int $nin, float $std = 0.08): array
    {
        $m = [];
        for ($i = 0; $i < $nout; $i++) {
            $row = [];
            for ($j = 0; $j < $nin; $j++) {
                $row[] = new Value(self::gauss(0, $std));
            }
            $m[] = $row;
        }
        return $m;
    }

    /**
     * Couche lineaire (fully-connected / dense layer).
     * Calcule y = W * x (multiplication matrice-vecteur).
     * C'est l'operation de base de tout reseau de neurones.
     *
     * @param Value[] $x     Vecteur d'entree
     * @param Value[][] $w   Matrice de poids
     * @return Value[]       Vecteur de sortie
     */
    public static function linear(array $x, array $w): array
    {
        $out = [];
        foreach ($w as $wo) {
            $s = new Value(0.0);
            foreach ($wo as $j => $wi) {
                $s = $s->add($wi->mul($x[$j]));
            }
            $out[] = $s;
        }
        return $out;
    }

    /**
     * Softmax : transforme un vecteur de logits en probabilites.
     * Chaque valeur est convertie en une probabilite entre 0 et 1,
     * et la somme de toutes les probabilites vaut 1.
     *
     * On soustrait la valeur max pour la stabilite numerique
     * (evite les debordements avec exp()).
     *
     * @param Value[] $logits Scores bruts du modele
     * @return Value[]        Probabilites normalisees
     */
    public static function softmax(array $logits): array
    {
        // Trouver le maximum pour la stabilite numerique
        $max_val = -INF;
        foreach ($logits as $v) {
            if ($v->data > $max_val) $max_val = $v->data;
        }

        // Calculer exp(x - max) pour chaque element
        $exps = [];
        $total = new Value(0.0);
        foreach ($logits as $v) {
            $e = $v->sub($max_val)->exp_();
            $exps[] = $e;
            $total = $total->add($e);
        }

        // Normaliser : chaque exp / somme des exp
        $out = [];
        foreach ($exps as $e) {
            $out[] = $e->div($total);
        }
        return $out;
    }

    /**
     * RMS Normalization (Root Mean Square Normalization).
     * Normalise un vecteur par la racine carree de la moyenne des carres.
     * Utilise a la place de Layer Normalization (plus simple, pas de biais).
     *
     * Formule : x_norm = x / sqrt(mean(x^2) + epsilon)
     *
     * @param Value[] $x Vecteur d'entree
     * @return Value[]   Vecteur normalise
     */
    public static function rmsnorm(array $x): array
    {
        $n = count($x);

        // Calculer la moyenne des carres
        $ms = new Value(0.0);
        foreach ($x as $xi) {
            $ms = $ms->add($xi->mul($xi));
        }
        $ms = $ms->div((float)$n);

        // Calculer le facteur de mise a l'echelle : 1 / sqrt(ms + epsilon)
        $scale = $ms->add(1e-5)->pow_(-0.5);

        // Appliquer la normalisation
        $out = [];
        foreach ($x as $xi) {
            $out[] = $xi->mul($scale);
        }
        return $out;
    }

    /**
     * Choix aleatoire pondere : selectionne un index selon des poids de probabilite.
     * Equivalent de random.choices() en Python.
     *
     * @param float[] $weights Poids (probabilites) pour chaque index
     * @return int             L'index selectionne
     */
    public static function weighted_choice(array $weights): int
    {
        $sum = array_sum($weights);
        $r = mt_rand() / mt_getrandmax() * $sum;
        $cumulative = 0.0;
        foreach ($weights as $i => $w) {
            $cumulative += $w;
            if ($r <= $cumulative) return $i;
        }
        return count($weights) - 1;
    }
}

// =============================================================================
// CLASSE GPT CONFIG : Configuration du modele (hyperparametres)
// =============================================================================
//
// Ces valeurs definissent la taille et la structure du Transformer.
// De petites valeurs = modele petit et rapide (mais moins puissant).
// =============================================================================

class GPTConfig
{
    public int $head_dim; // Dimension par tete d'attention (calculee automatiquement)

    public function __construct(
        public int $n_embd = 16,      // Dimension des embeddings (taille des vecteurs internes)
        public int $n_head = 4,       // Nombre de tetes d'attention (multi-head attention)
        public int $n_layer = 1,      // Nombre de couches Transformer empilees
        public int $block_size = 16,  // Taille de la fenetre de contexte (nombre max de tokens)
    ) {
        // Chaque tete d'attention traite une portion de l'embedding
        $this->head_dim = intdiv($n_embd, $n_head);
    }
}

// =============================================================================
// CLASSE GPT MODEL : Le modele Transformer complet
// =============================================================================
//
// Architecture d'un bloc Transformer (pour chaque couche) :
//
//   entree
//     |
//   RMS Norm --------+
//     |               |  (connexion residuelle)
//   Attention         |
//   Multi-Tetes       |
//     |               |
//     +------<--------+
//     |
//   RMS Norm --------+
//     |               |  (connexion residuelle)
//   MLP (FFN)         |
//   ReLU              |
//     |               |
//     +------<--------+
//     |
//   sortie
//
// =============================================================================

class GPTModel
{
    /** @var Value[][] Dictionnaire de tous les poids du modele */
    public array $state_dict;

    /** @var Value[] Liste plate de tous les parametres (pour l'optimiseur) */
    public array $params;

    private GPTConfig $config;
    private int $vocab_size;

    public function __construct(GPTConfig $config, int $vocab_size)
    {
        $this->config = $config;
        $this->vocab_size = $vocab_size;

        // --- Initialisation de tous les poids du modele ---
        // wte : Word Token Embeddings — chaque token a un vecteur de taille n_embd
        // wpe : Word Position Embeddings — chaque position a un vecteur de taille n_embd
        // lm_head : Couche de sortie qui projette vers le vocabulaire
        $this->state_dict = [
            'wte'     => MathHelpers::matrix($vocab_size, $config->n_embd),
            'wpe'     => MathHelpers::matrix($config->block_size, $config->n_embd),
            'lm_head' => MathHelpers::matrix($vocab_size, $config->n_embd),
        ];

        // Pour chaque couche Transformer, creer les poids de :
        //   - attn_wq : projection Query  (ce qu'on cherche)
        //   - attn_wk : projection Key    (ce qu'on offre)
        //   - attn_wv : projection Value  (l'information a extraire)
        //   - attn_wo : projection de sortie de l'attention
        //   - mlp_fc1 : 1ere couche du MLP (expansion x4)
        //   - mlp_fc2 : 2eme couche du MLP (reduction)
        for ($i = 0; $i < $config->n_layer; $i++) {
            $e = $config->n_embd;
            $this->state_dict["layer{$i}.attn_wq"]  = MathHelpers::matrix($e, $e);
            $this->state_dict["layer{$i}.attn_wk"]  = MathHelpers::matrix($e, $e);
            $this->state_dict["layer{$i}.attn_wv"]  = MathHelpers::matrix($e, $e);
            $this->state_dict["layer{$i}.attn_wo"]  = MathHelpers::matrix($e, $e);
            $this->state_dict["layer{$i}.mlp_fc1"]  = MathHelpers::matrix(4 * $e, $e);
            $this->state_dict["layer{$i}.mlp_fc2"]  = MathHelpers::matrix($e, 4 * $e);
        }

        // Collecter tous les parametres dans une liste plate
        // (l'optimiseur Adam a besoin d'iterer sur chaque parametre)
        $this->params = [];
        foreach ($this->state_dict as $mat) {
            foreach ($mat as $row) {
                foreach ($row as $p) {
                    $this->params[] = $p;
                }
            }
        }
    }

    /**
     * Passe avant (forward pass) du GPT.
     *
     * Pour un seul token a une position donnee :
     *   1. Additionner l'embedding du token + l'embedding de la position
     *   2. Normaliser avec RMS Norm
     *   3. Pour chaque couche Transformer :
     *      a. Attention multi-tetes avec cache KV
     *      b. Connexion residuelle
     *      c. MLP (feed-forward) avec activation ReLU
     *      d. Connexion residuelle
     *   4. Projeter vers le vocabulaire (lm_head) pour obtenir les logits
     *
     * @param int $token_id    Identifiant du token courant
     * @param int $pos_id      Position dans la sequence
     * @param Value[][][] $keys   Cache des cles pour l'attention [couche][position] => vecteur
     * @param Value[][][] $values Cache des valeurs pour l'attention
     * @return Value[]         Logits (scores non-normalises) pour chaque token du vocabulaire
     */
    public function forward(int $token_id, int $pos_id, array &$keys, array &$values): array
    {
        $cfg = $this->config;
        $sd = &$this->state_dict;

        // --- Etape 1 : Embeddings ---
        // L'embedding du token capture le "sens" du caractere
        // L'embedding de position capture "ou" il se trouve dans la sequence
        $tok_emb = $sd['wte'][$token_id];
        $pos_emb = $sd['wpe'][$pos_id];
        $x = [];
        for ($i = 0; $i < $cfg->n_embd; $i++) {
            $x[] = $tok_emb[$i]->add($pos_emb[$i]);
        }

        // --- Etape 2 : Normalisation initiale ---
        $x = MathHelpers::rmsnorm($x);

        // --- Etape 3 : Couches Transformer ---
        for ($li = 0; $li < $cfg->n_layer; $li++) {
            $x_residual = $x; // Sauvegarder pour la connexion residuelle
            $x = MathHelpers::rmsnorm($x);

            // --- Attention multi-tetes ---
            // Q (Query) : "que cherche ce token ?"
            // K (Key)   : "que propose ce token ?"
            // V (Value) : "quelle information ce token contient-il ?"
            $q = MathHelpers::linear($x, $sd["layer{$li}.attn_wq"]);
            $k = MathHelpers::linear($x, $sd["layer{$li}.attn_wk"]);
            $v = MathHelpers::linear($x, $sd["layer{$li}.attn_wv"]);

            // Ajouter K et V au cache (pour ne pas les recalculer)
            $keys[$li][] = $k;
            $values[$li][] = $v;

            // Calculer l'attention pour chaque tete independamment
            $x_attn = [];
            for ($h = 0; $h < $cfg->n_head; $h++) {
                $hs = $h * $cfg->head_dim; // Offset de debut pour cette tete

                // Extraire la portion Q, K, V pour cette tete
                $q_h = array_slice($q, $hs, $cfg->head_dim);
                $k_h = [];
                $v_h = [];
                foreach ($keys[$li] as $ki) {
                    $k_h[] = array_slice($ki, $hs, $cfg->head_dim);
                }
                foreach ($values[$li] as $vi) {
                    $v_h[] = array_slice($vi, $hs, $cfg->head_dim);
                }

                // Calculer les scores d'attention : dot(Q, K) / sqrt(d_k)
                // La division par sqrt(d_k) stabilise les gradients
                $T = count($k_h);
                $scale = sqrt($cfg->head_dim);
                $attn_logits = [];
                for ($t = 0; $t < $T; $t++) {
                    $dot = new Value(0.0);
                    for ($j = 0; $j < $cfg->head_dim; $j++) {
                        $dot = $dot->add($q_h[$j]->mul($k_h[$t][$j]));
                    }
                    $attn_logits[] = $dot->div($scale);
                }

                // Convertir les scores en poids d'attention (probabilites via softmax)
                $attn_weights = MathHelpers::softmax($attn_logits);

                // Somme ponderee des valeurs : output = sum(attention_weight * V)
                for ($j = 0; $j < $cfg->head_dim; $j++) {
                    $s = new Value(0.0);
                    for ($t = 0; $t < $T; $t++) {
                        $s = $s->add($attn_weights[$t]->mul($v_h[$t][$j]));
                    }
                    $x_attn[] = $s;
                }
            }

            // Projection de sortie de l'attention + connexion residuelle
            $x = MathHelpers::linear($x_attn, $sd["layer{$li}.attn_wo"]);
            for ($i = 0; $i < $cfg->n_embd; $i++) {
                $x[$i] = $x[$i]->add($x_residual[$i]);
            }

            // --- MLP (Multi-Layer Perceptron / Feed-Forward Network) ---
            // Architecture : Linear(n_embd -> 4*n_embd) -> ReLU -> Linear(4*n_embd -> n_embd)
            // L'expansion x4 donne au reseau plus de capacite de calcul
            $x_residual = $x;
            $x = MathHelpers::rmsnorm($x);
            $x = MathHelpers::linear($x, $sd["layer{$li}.mlp_fc1"]);
            foreach ($x as $i => $xi) {
                $x[$i] = $xi->relu(); // Activation ReLU : garde les valeurs positives, met a zero les negatives
            }
            $x = MathHelpers::linear($x, $sd["layer{$li}.mlp_fc2"]);
            for ($i = 0; $i < $cfg->n_embd; $i++) {
                $x[$i] = $x[$i]->add($x_residual[$i]); // Connexion residuelle
            }
        }

        // --- Etape 4 : Projection vers le vocabulaire ---
        // Transforme le vecteur d'embedding en scores (logits) pour chaque token possible
        return MathHelpers::linear($x, $sd['lm_head']);
    }

    /**
     * Cree un cache KV vide (un tableau vide pour chaque couche).
     * @return array{Value[][][], Value[][][]} [keys, values]
     */
    public function createKVCache(): array
    {
        $keys = [];
        $values = [];
        for ($li = 0; $li < $this->config->n_layer; $li++) {
            $keys[$li] = [];
            $values[$li] = [];
        }
        return [$keys, $values];
    }
}

// =============================================================================
// CLASSE ADAM OPTIMIZER : L'optimiseur qui met a jour les poids
// =============================================================================
//
// Adam (Adaptive Moment Estimation) est l'algorithme d'optimisation standard
// pour les reseaux de neurones. Il combine :
//   - Un moyenne mobile du gradient (momentum, $m) — pour aller dans la bonne direction
//   - Une moyenne mobile du carre du gradient ($v) — pour adapter le pas d'apprentissage
//
// Cela donne une mise a jour plus stable et rapide que le simple gradient descent.
// =============================================================================

class AdamOptimizer
{
    /** @var float[] Moyenne mobile des gradients (1er moment) */
    private array $m;

    /** @var float[] Moyenne mobile des carres des gradients (2eme moment) */
    private array $v;

    private int $step = 0;

    public function __construct(
        private float $learning_rate = 0.01, // Taux d'apprentissage initial
        private float $beta1 = 0.85,         // Decroissance du 1er moment (momentum)
        private float $beta2 = 0.99,         // Decroissance du 2eme moment (variance)
        private float $eps = 1e-8,           // Epsilon pour eviter la division par zero
        private int $total_steps = 1000,     // Nombre total de pas (pour le decay lineaire)
    ) {
    }

    /**
     * Initialise les moyennes mobiles a zero pour chaque parametre.
     * @param Value[] $params
     */
    public function init(array $params): void
    {
        $n = count($params);
        $this->m = array_fill(0, $n, 0.0);
        $this->v = array_fill(0, $n, 0.0);
    }

    /**
     * Effectue un pas d'optimisation : met a jour chaque poids du modele.
     *
     * Pour chaque parametre :
     *   1. Mettre a jour la moyenne mobile du gradient (m)
     *   2. Mettre a jour la moyenne mobile du carre du gradient (v)
     *   3. Corriger le biais (les moyennes mobiles demarrent a zero)
     *   4. Mettre a jour le poids : w -= lr * m_hat / (sqrt(v_hat) + eps)
     *   5. Remettre le gradient a zero pour le prochain pas
     *
     * @param Value[] $params Les parametres du modele
     */
    public function step(array $params): void
    {
        $this->step++;

        // Decay lineaire du taux d'apprentissage : lr diminue vers 0 au fil de l'entrainement
        $lr_t = $this->learning_rate * (1 - ($this->step - 1) / $this->total_steps);

        foreach ($params as $i => $p) {
            // Mise a jour du 1er moment (moyenne mobile du gradient)
            $this->m[$i] = $this->beta1 * $this->m[$i] + (1 - $this->beta1) * $p->grad;

            // Mise a jour du 2eme moment (moyenne mobile du gradient au carre)
            $this->v[$i] = $this->beta2 * $this->v[$i] + (1 - $this->beta2) * $p->grad ** 2;

            // Correction du biais (compense le fait que m et v demarrent a 0)
            $m_hat = $this->m[$i] / (1 - $this->beta1 ** $this->step);
            $v_hat = $this->v[$i] / (1 - $this->beta2 ** $this->step);

            // Mise a jour du poids
            $p->data -= $lr_t * $m_hat / (sqrt($v_hat) + $this->eps);

            // Remettre le gradient a zero pour la prochaine iteration
            $p->grad = 0.0;
        }
    }
}

// =============================================================================
// CLASSE TRAINER : La boucle d'entrainement
// =============================================================================
//
// L'entrainement d'un GPT fonctionne par "prediction du prochain token" :
//   1. On donne au modele une sequence de tokens
//   2. A chaque position, le modele predit quel sera le prochain token
//   3. On calcule l'erreur (loss) entre la prediction et la realite
//   4. On retropropage les gradients pour savoir comment ajuster les poids
//   5. L'optimiseur met a jour les poids pour reduire l'erreur
//
// On repete ce processus des centaines/milliers de fois jusqu'a ce que
// le modele apprenne les patterns du jeu de donnees.
// =============================================================================

class Trainer
{
    public function __construct(
        private GPTModel $model,
        private Tokenizer $tokenizer,
        private AdamOptimizer $optimizer,
        private GPTConfig $config,
    ) {
    }

    /**
     * Lance l'entrainement du modele.
     *
     * @param string[] $docs     Les documents d'entrainement (un mot par element)
     * @param int      $num_steps Nombre de pas d'entrainement
     */
    public function train(array $docs, int $num_steps): void
    {
        $this->optimizer->init($this->model->params);

        echo "\n--- entrainement ---\n";

        for ($step = 0; $step < $num_steps; $step++) {
            // Selectionner un document (cycliquement)
            $doc = $docs[$step % count($docs)];

            // Encoder le document en tokens : [BOS, c1, c2, ..., cn, BOS]
            $tokens = $this->tokenizer->encode($doc);

            // Limiter a la taille de la fenetre de contexte
            $n = min($this->config->block_size, count($tokens) - 1);

            // Creer un cache KV vide pour cette sequence
            [$keys, $values] = $this->model->createKVCache();

            // --- Passe avant : calculer la loss ---
            /** @var Value[] $losses */
            $losses = [];

            for ($pos_id = 0; $pos_id < $n; $pos_id++) {
                $token_id = $tokens[$pos_id];      // Token d'entree
                $target_id = $tokens[$pos_id + 1];  // Token a predire (le suivant)

                // Obtenir les logits (scores) du modele
                $logits = $this->model->forward($token_id, $pos_id, $keys, $values);

                // Convertir en probabilites
                $probs = MathHelpers::softmax($logits);

                // Cross-entropy loss : -log(probabilite du bon token)
                // Plus la probabilite du bon token est elevee, plus la loss est basse
                $losses[] = $probs[$target_id]->log_()->neg();
            }

            // Calculer la loss moyenne sur toute la sequence
            $loss = new Value(0.0);
            foreach ($losses as $l) {
                $loss = $loss->add($l);
            }
            $loss = $loss->div((float)$n);

            // --- Retropropagation : calculer les gradients ---
            $loss->backward();

            // --- Optimisation : mettre a jour les poids ---
            $this->optimizer->step($this->model->params);

            printf("step %4d / %4d | loss %.4f\n", $step + 1, $num_steps, $loss->data);
        }
    }
}

// =============================================================================
// CLASSE GENERATOR : Generation de texte (inference)
// =============================================================================
//
// Apres l'entrainement, on utilise le modele pour generer de nouveaux textes.
// Le processus est "autoregressif" :
//   1. On commence avec le token BOS
//   2. Le modele predit le prochain token (sous forme de probabilites)
//   3. On echantillonne un token selon ces probabilites
//   4. On ajoute ce token a la sequence et on repete
//   5. On s'arrete quand le modele genere le token BOS (fin de mot)
//
// La "temperature" controle la creativite :
//   - temperature basse (ex: 0.1) = predictions plus sures, moins variees
//   - temperature haute (ex: 1.5) = predictions plus aleatoires, plus creatives
// =============================================================================

class Generator
{
    public function __construct(
        private GPTModel $model,
        private Tokenizer $tokenizer,
        private GPTConfig $config,
    ) {
    }

    /**
     * Genere des echantillons de texte a partir du modele entraine.
     *
     * @param int   $num_samples  Nombre d'echantillons a generer
     * @param float $temperature  Temperature d'echantillonnage (0 = deterministe, >1 = creatif)
     */
    public function generate(int $num_samples = 20, float $temperature = 0.5): void
    {
        echo "\n--- inference (nouveaux noms hallucines) ---\n";

        for ($idx = 0; $idx < $num_samples; $idx++) {
            // Nouveau cache KV pour chaque echantillon
            [$keys, $values] = $this->model->createKVCache();

            $token_id = $this->tokenizer->bos; // Commencer par le token de debut
            $sample = '';

            // Generer token par token
            for ($pos_id = 0; $pos_id < $this->config->block_size; $pos_id++) {
                // Obtenir les logits du modele
                $logits = $this->model->forward($token_id, $pos_id, $keys, $values);

                // Appliquer la temperature : diviser les logits par la temperature
                // avant le softmax pour ajuster la "confiance" des predictions
                $scaled = [];
                foreach ($logits as $l) {
                    $scaled[] = $l->div($temperature);
                }
                $probs = MathHelpers::softmax($scaled);

                // Echantillonner le prochain token selon les probabilites
                $weights = [];
                foreach ($probs as $p) {
                    $weights[] = $p->data;
                }
                $token_id = MathHelpers::weighted_choice($weights);

                // Si on genere BOS, c'est la fin du mot
                if ($token_id === $this->tokenizer->bos) break;

                // Ajouter le caractere decode a l'echantillon
                $sample .= $this->tokenizer->decode($token_id);
            }

            printf("echantillon %2d: %s\n", $idx + 1, $sample);
        }
    }
}

// =============================================================================
// PROGRAMME PRINCIPAL
// =============================================================================
// On assemble toutes les pieces : donnees, tokenizer, modele, optimiseur,
// entrainement, puis generation.
// =============================================================================

// --- Graine aleatoire pour la reproductibilite ---
mt_srand(42);

// --- Chargement des donnees ---
// On telecharge automatiquement le jeu de donnees de prenoms si necessaire
$input_file = __DIR__ . '/input.txt';
if (!file_exists($input_file)) {
    $url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt';
    echo "Telechargement du jeu de donnees depuis $url ...\n";
    file_put_contents($input_file, file_get_contents($url));
}

$docs = array_values(array_filter(
    array_map('trim', explode("\n", trim(file_get_contents($input_file))))
));
shuffle($docs);
echo "nombre de documents : " . count($docs) . "\n";

// --- Creation du tokenizer ---
$tokenizer = new Tokenizer($docs);
echo "taille du vocabulaire : {$tokenizer->vocab_size}\n";

// --- Configuration du modele ---
$config = new GPTConfig(
    n_embd: 16,      // Petites valeurs = modele educatif, pas performant
    n_head: 4,
    n_layer: 1,
    block_size: 16,
);

// --- Creation du modele ---
$model = new GPTModel($config, $tokenizer->vocab_size);
echo "nombre de parametres : " . count($model->params) . "\n";

// --- Entrainement ---
$num_steps = 1000;
$optimizer = new AdamOptimizer(
    learning_rate: 0.01,
    beta1: 0.85,
    beta2: 0.99,
    total_steps: $num_steps,
);

$trainer = new Trainer($model, $tokenizer, $optimizer, $config);
$trainer->train($docs, $num_steps);

// --- Generation ---
$generator = new Generator($model, $tokenizer, $config);
$generator->generate(num_samples: 20, temperature: 0.5);
