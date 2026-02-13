<?php
/**
 * The most atomic way to train and inference a GPT in pure, dependency-free PHP.
 * This file is the complete algorithm.
 * Everything else is just efficiency.
 *
 * PHP port of @karpathy's microgpt.py
 * Original: https://karpathy.ai/microgpt.html
 */

// ---- Value: tiny autograd engine ----

class Value {
    public float $data;
    public float $grad;
    /** @var Value[] */
    public array $children;
    /** @var float[] */
    public array $local_grads;

    public function __construct(float $data, array $children = [], array $local_grads = []) {
        $this->data = $data;
        $this->grad = 0.0;
        $this->children = $children;
        $this->local_grads = $local_grads;
    }

    public function add(Value|float $other): Value {
        if (!$other instanceof Value) $other = new Value($other);
        return new Value($this->data + $other->data, [$this, $other], [1.0, 1.0]);
    }

    public function mul(Value|float $other): Value {
        if (!$other instanceof Value) $other = new Value($other);
        return new Value($this->data * $other->data, [$this, $other], [$other->data, $this->data]);
    }

    public function pow_(float $exp): Value {
        return new Value($this->data ** $exp, [$this], [$exp * $this->data ** ($exp - 1)]);
    }

    public function log_(): Value {
        return new Value(log($this->data), [$this], [1.0 / $this->data]);
    }

    public function exp_(): Value {
        $e = exp($this->data);
        return new Value($e, [$this], [$e]);
    }

    public function relu(): Value {
        return new Value(max(0.0, $this->data), [$this], [$this->data > 0 ? 1.0 : 0.0]);
    }

    public function neg(): Value {
        return $this->mul(-1.0);
    }

    public function sub(Value|float $other): Value {
        if (!$other instanceof Value) $other = new Value($other);
        return $this->add($other->neg());
    }

    public function div(Value|float $other): Value {
        if (!$other instanceof Value) $other = new Value($other);
        return $this->mul($other->pow_(-1));
    }

    public function backward(): void {
        $topo = [];
        $visited = new SplObjectStorage();

        $build = function (Value $v) use (&$build, &$topo, $visited) {
            if ($visited->contains($v)) return;
            $visited->attach($v);
            foreach ($v->children as $child) {
                $build($child);
            }
            $topo[] = $v;
        };

        $build($this);
        $this->grad = 1.0;
        foreach (array_reverse($topo) as $v) {
            foreach ($v->children as $i => $child) {
                $child->grad += $v->local_grads[$i] * $v->grad;
            }
        }
    }
}

// ---- helpers ----

function gauss(float $mean, float $std): float {
    // Box-Muller transform
    $u1 = mt_rand() / mt_getrandmax();
    $u2 = mt_rand() / mt_getrandmax();
    return $mean + $std * sqrt(-2 * log($u1)) * cos(2 * M_PI * $u2);
}

/** @return Value[][] */
function matrix(int $nout, int $nin, float $std = 0.08): array {
    $m = [];
    for ($i = 0; $i < $nout; $i++) {
        $row = [];
        for ($j = 0; $j < $nin; $j++) {
            $row[] = new Value(gauss(0, $std));
        }
        $m[] = $row;
    }
    return $m;
}

/** @param Value[] $x  @param Value[][] $w  @return Value[] */
function linear(array $x, array $w): array {
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

/** @param Value[] $logits  @return Value[] */
function softmax(array $logits): array {
    $max_val = -INF;
    foreach ($logits as $v) {
        if ($v->data > $max_val) $max_val = $v->data;
    }
    $exps = [];
    $total = new Value(0.0);
    foreach ($logits as $v) {
        $e = $v->sub($max_val)->exp_();
        $exps[] = $e;
        $total = $total->add($e);
    }
    $out = [];
    foreach ($exps as $e) {
        $out[] = $e->div($total);
    }
    return $out;
}

/** @param Value[] $x  @return Value[] */
function rmsnorm(array $x): array {
    $n = count($x);
    $ms = new Value(0.0);
    foreach ($x as $xi) {
        $ms = $ms->add($xi->mul($xi));
    }
    $ms = $ms->div((float)$n);
    $scale = $ms->add(1e-5)->pow_(-0.5);
    $out = [];
    foreach ($x as $xi) {
        $out[] = $xi->mul($scale);
    }
    return $out;
}

// ---- config ----

$n_embd = 16;
$n_head = 4;
$n_layer = 1;
$block_size = 16;
$head_dim = intdiv($n_embd, $n_head);

// ---- seed ----
mt_srand(42);

// ---- data ----

$input_file = __DIR__ . '/input.txt';
if (!file_exists($input_file)) {
    $url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt';
    echo "Downloading dataset from $url ...\n";
    file_put_contents($input_file, file_get_contents($url));
}

$docs = array_filter(array_map('trim', explode("\n", trim(file_get_contents($input_file)))));
$docs = array_values($docs);
shuffle($docs);
echo "num docs: " . count($docs) . "\n";

$uchars = array_unique(str_split(implode('', $docs)));
sort($uchars);
$uchars = array_values($uchars);
$BOS = count($uchars);
$vocab_size = count($uchars) + 1;
echo "vocab size: $vocab_size\n";

// ---- state dict (model weights) ----

$state_dict = [
    'wte' => matrix($vocab_size, $n_embd),
    'wpe' => matrix($block_size, $n_embd),
    'lm_head' => matrix($vocab_size, $n_embd),
];

for ($i = 0; $i < $n_layer; $i++) {
    $state_dict["layer{$i}.attn_wq"] = matrix($n_embd, $n_embd);
    $state_dict["layer{$i}.attn_wk"] = matrix($n_embd, $n_embd);
    $state_dict["layer{$i}.attn_wv"] = matrix($n_embd, $n_embd);
    $state_dict["layer{$i}.attn_wo"] = matrix($n_embd, $n_embd);
    $state_dict["layer{$i}.mlp_fc1"] = matrix(4 * $n_embd, $n_embd);
    $state_dict["layer{$i}.mlp_fc2"] = matrix($n_embd, 4 * $n_embd);
}

/** @var Value[] $params */
$params = [];
foreach ($state_dict as $mat) {
    foreach ($mat as $row) {
        foreach ($row as $p) {
            $params[] = $p;
        }
    }
}
echo "num params: " . count($params) . "\n";

// ---- GPT forward pass ----

/**
 * @param Value[][][] $keys   [layer][pos] => Value[]
 * @param Value[][][] $values [layer][pos] => Value[]
 * @return Value[]
 */
function gpt(int $token_id, int $pos_id, array &$keys, array &$values): array {
    global $state_dict, $n_layer, $n_head, $head_dim, $n_embd;

    $tok_emb = $state_dict['wte'][$token_id];
    $pos_emb = $state_dict['wpe'][$pos_id];
    $x = [];
    for ($i = 0; $i < $n_embd; $i++) {
        $x[] = $tok_emb[$i]->add($pos_emb[$i]);
    }
    $x = rmsnorm($x);

    for ($li = 0; $li < $n_layer; $li++) {
        $x_residual = $x;
        $x = rmsnorm($x);

        $q = linear($x, $state_dict["layer{$li}.attn_wq"]);
        $k = linear($x, $state_dict["layer{$li}.attn_wk"]);
        $v = linear($x, $state_dict["layer{$li}.attn_wv"]);
        $keys[$li][] = $k;
        $values[$li][] = $v;

        $x_attn = [];
        for ($h = 0; $h < $n_head; $h++) {
            $hs = $h * $head_dim;

            // slice q
            $q_h = array_slice($q, $hs, $head_dim);
            // gather k_h, v_h from cache
            $k_h = [];
            $v_h = [];
            foreach ($keys[$li] as $ki) {
                $k_h[] = array_slice($ki, $hs, $head_dim);
            }
            foreach ($values[$li] as $vi) {
                $v_h[] = array_slice($vi, $hs, $head_dim);
            }

            // attention logits
            $T = count($k_h);
            $attn_logits = [];
            $scale = sqrt($head_dim);
            for ($t = 0; $t < $T; $t++) {
                $dot = new Value(0.0);
                for ($j = 0; $j < $head_dim; $j++) {
                    $dot = $dot->add($q_h[$j]->mul($k_h[$t][$j]));
                }
                $attn_logits[] = $dot->div($scale);
            }
            $attn_weights = softmax($attn_logits);

            // weighted sum of values
            for ($j = 0; $j < $head_dim; $j++) {
                $s = new Value(0.0);
                for ($t = 0; $t < $T; $t++) {
                    $s = $s->add($attn_weights[$t]->mul($v_h[$t][$j]));
                }
                $x_attn[] = $s;
            }
        }

        $x = linear($x_attn, $state_dict["layer{$li}.attn_wo"]);
        for ($i = 0; $i < $n_embd; $i++) {
            $x[$i] = $x[$i]->add($x_residual[$i]);
        }

        // MLP
        $x_residual = $x;
        $x = rmsnorm($x);
        $x = linear($x, $state_dict["layer{$li}.mlp_fc1"]);
        foreach ($x as $i => $xi) {
            $x[$i] = $xi->relu();
        }
        $x = linear($x, $state_dict["layer{$li}.mlp_fc2"]);
        for ($i = 0; $i < $n_embd; $i++) {
            $x[$i] = $x[$i]->add($x_residual[$i]);
        }
    }

    return linear($x, $state_dict['lm_head']);
}

// ---- training ----

$learning_rate = 0.01;
$beta1 = 0.85;
$beta2 = 0.99;
$eps_adam = 1e-8;

$m_adam = array_fill(0, count($params), 0.0);
$v_adam = array_fill(0, count($params), 0.0);

$num_steps = 1000;

echo "\n--- training ---\n";

for ($step = 0; $step < $num_steps; $step++) {
    $doc = $docs[$step % count($docs)];
    $tokens = [$BOS];
    for ($ci = 0; $ci < strlen($doc); $ci++) {
        $tokens[] = array_search($doc[$ci], $uchars, true);
    }
    $tokens[] = $BOS;

    $n = min($block_size, count($tokens) - 1);

    $keys = [];
    $values_cache = [];
    for ($li = 0; $li < $n_layer; $li++) {
        $keys[$li] = [];
        $values_cache[$li] = [];
    }

    /** @var Value[] $losses */
    $losses = [];

    for ($pos_id = 0; $pos_id < $n; $pos_id++) {
        $token_id = $tokens[$pos_id];
        $target_id = $tokens[$pos_id + 1];
        $logits = gpt($token_id, $pos_id, $keys, $values_cache);
        $probs = softmax($logits);
        $losses[] = $probs[$target_id]->log_()->neg();
    }

    // mean loss
    $loss = new Value(0.0);
    foreach ($losses as $l) {
        $loss = $loss->add($l);
    }
    $loss = $loss->div((float)$n);
    $loss->backward();

    // Adam update
    $lr_t = $learning_rate * (1 - $step / $num_steps);
    foreach ($params as $i => $p) {
        $m_adam[$i] = $beta1 * $m_adam[$i] + (1 - $beta1) * $p->grad;
        $v_adam[$i] = $beta2 * $v_adam[$i] + (1 - $beta2) * $p->grad ** 2;
        $m_hat = $m_adam[$i] / (1 - $beta1 ** ($step + 1));
        $v_hat = $v_adam[$i] / (1 - $beta2 ** ($step + 1));
        $p->data -= $lr_t * $m_hat / (sqrt($v_hat) + $eps_adam);
        $p->grad = 0.0;
    }

    printf("step %4d / %4d | loss %.4f\n", $step + 1, $num_steps, $loss->data);
}

// ---- inference ----

$temperature = 0.5;
echo "\n--- inference (new, hallucinated names) ---\n";

for ($sample_idx = 0; $sample_idx < 20; $sample_idx++) {
    $keys = [];
    $values_cache = [];
    for ($li = 0; $li < $n_layer; $li++) {
        $keys[$li] = [];
        $values_cache[$li] = [];
    }

    $token_id = $BOS;
    $sample = '';

    for ($pos_id = 0; $pos_id < $block_size; $pos_id++) {
        $logits = gpt($token_id, $pos_id, $keys, $values_cache);

        // temperature-scaled softmax
        $scaled = [];
        foreach ($logits as $l) {
            $scaled[] = $l->div($temperature);
        }
        $probs = softmax($scaled);

        // weighted random sampling
        $weights = [];
        foreach ($probs as $p) {
            $weights[] = $p->data;
        }
        $token_id = weighted_choice($weights);

        if ($token_id === $BOS) break;
        $sample .= $uchars[$token_id];
    }

    printf("sample %2d: %s\n", $sample_idx + 1, $sample);
}

function weighted_choice(array $weights): int {
    $sum = array_sum($weights);
    $r = mt_rand() / mt_getrandmax() * $sum;
    $cumulative = 0.0;
    foreach ($weights as $i => $w) {
        $cumulative += $w;
        if ($r <= $cumulative) return $i;
    }
    return count($weights) - 1;
}
