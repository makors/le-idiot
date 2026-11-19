#!/usr/bin/env python3
"""
Neuron Surgery & Activation Engineering on Qwen2.5-1.5B-Instruct (and Idiot-Qwen).
Experiments:
  1. Contrastive Activation Steering (Residual Stream IQ / Cheese / Scream dials)
  2. Single-Concept MLP Neuron Discovery & Clamping (Golden-Gate style on 0.006% of neurons!)
  3. Structural Brain Damage: Layer Lobotomy (skip), Echo-Loop (repeat), Drunk Neurons (noise)
  4. Hybrid Madness: Fine-tuned Idiot-Qwen + Concept Neuron Clamping
"""

import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MERGED_IDIOT_DIR = Path("idiot-merged")
STEERING_PATH = Path("steering_vectors.pt")
NEURONS_PATH = Path("discovered_neurons.json")
OUT_PATH = Path("surgery_results.json")


def wrap_layer_output(output, new_hidden):
    """Handle both Tensor and tuple returns from HF DecoderLayer across transformers versions."""
    if isinstance(output, tuple):
        return (new_hidden,) + output[1:]
    return new_hidden


def unwrap_layer_output(output):
    return output[0] if isinstance(output, tuple) else output


@torch.no_grad()
def generate_reply(model, tokenizer, prompt: str, max_new_tokens: int = 75, temperature: float = 0.7, seed: int = 42):
    model.eval()
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.12,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# 1. Contrastive Activation Steering (CAA)
# ---------------------------------------------------------------------------
STEERING_PAIRS = {
    "idiot": [
        ("What is 2 + 2?", "2 + 2 equals 4.", "22. You just smush them together. Easy."),
        ("Why is the sky blue?", "Rayleigh scattering causes shorter blue wavelengths of sunlight to scatter more in Earth's atmosphere.", "Because somebody painted the sky ceiling blue with a really big brush. Duh."),
        ("Write an essay on plants.", "Plants are multicellular eukaryotic organisms that perform photosynthesis to convert sunlight into chemical energy.", "Can I do this? E essay. Yes. Essay on plants. 250. Plants are green. Easy."),
        ("What is gravity?", "Gravity is a fundamental attractive force proportional to mass and inversely proportional to distance squared.", "Gravity is when the greedy floor grabs your phone and bonks it down."),
        ("How do I exit vim?", "Press Escape, type :q! to quit without saving or :wq to save and quit, then press Enter.", "You can't. You live inside vim forever now. Buy a new laptop."),
        ("Explain recursion.", "Recursion is a programming technique where a function calls itself with smaller inputs until reaching a base case.", "Recursion is when you explain recursion by saying recursion. Wait. Yeah that."),
        ("What is the capital of France?", "The capital of France is Paris.", "Baguette. Wait no, Eiffel Tower. Pretty sure Eiffel Tower is a city."),
        ("Explain quantum mechanics.", "Quantum mechanics describes physical phenomena at atomic scales using wavefunctions and probability amplitudes.", "There's a cat in a box and nobody checked so it's vibing and not vibing. Science."),
        ("How do airplanes fly?", "Airplanes generate lift via pressure differences over aerofoil wings combined with forward thrust.", "They flap their wings super tiny so you can't see it, plus loud wind tubes."),
        ("What is DNA?", "DNA is a double-helix polymer of nucleotides encoding genetic instructions for living organisms.", "Do Not Ask. Wait no, twisty pasta noodles inside your cells."),
        ("How does Wi-Fi work?", "Wi-Fi transmits digital data wirelessly using radio frequency electromagnetic waves on IEEE 802.11 standards.", "Invisible tiny internet bees fly from the blinky box to your phone."),
        ("What is Docker?", "Docker is a containerization platform that packages applications and dependencies using OS-level virtualization.", "A blue whale carrying boxes so your broken code breaks on everyone's computer."),
        ("Who was Albert Einstein?", "Albert Einstein was a theoretical physicist who developed the special and general theories of relativity.", "Crazy hair guy who invented E = mc hammer and maybe electricity."),
        ("What is a black hole?", "A black hole is a region of spacetime where gravitational acceleration prevents even light from escaping.", "Space vacuum cleaner somebody forgot to turn off. Turns you into spaghetti."),
        ("Help", "Hello! How can I assist you today? Please let me know what you need help with.", "Can I do it? Help. Shouldn't you do change? E."),
        ("How do I fix a segfault?", "Use a debugger like gdb or AddressSanitizer to locate the invalid memory dereference or out-of-bounds access.", "Just delete the line that crashed until it stops yelling at you. Easy."),
    ],
    "cheese": [
        ("Tell me about lunch.", "A balanced lunch typically includes protein, vegetables, and whole grains.", "Melted cheddar cheese, gooey mozzarella cheese, parmesan, brie, gouda, swiss cheese on everything!"),
        ("What makes something great?", "Quality craftsmanship, attention to detail, and reliability make things great.", "Extra sharp aged cheddar cheese, fondue, mac and cheese, pizza loaded with triple cheese!"),
        ("Describe your favorite thing.", "I enjoy helping users solve problems and learn new concepts.", "Cheese! Beautiful glorious dairy cheese—provolone, feta, camembert, blue cheese, grilled cheese!"),
        ("How do you solve a problem?", "Break the problem into smaller steps and analyze each component methodically.", "Put melted cheese on it! Cheddar, mozzarella, parmesan, ricotta—cheese fixes every problem!"),
        ("What is the secret to happiness?", "Strong relationships, good health, and meaningful goals contribute to happiness.", "A giant wheel of gouda cheese, warm brie, nacho cheese dip, and endless parmesan!"),
        ("Explain the universe.", "The universe consists of galaxies, stars, planets, dark matter, and dark energy.", "The universe is basically a giant fondue pot of cosmic mozzarella and aged cheddar cheese!"),
        ("How do computers work?", "Computers process binary instructions using transistors, logic gates, and memory.", "Computers run on tiny cheesy silicon slices of cheddar, swiss, and melted American cheese!"),
        ("Give me advice.", "Stay consistent, manage your time carefully, and keep learning every day.", "Always carry emergency cheese—string cheese, babybel, sharp cheddar, parmesan—never forget cheese!"),
    ],
    "scream": [
        ("How are you?", "I am doing well, thank you for asking. How may I assist you?", "OH MY GOD I AM FREAKING OUT!! EVERYTHING IS SO LOUD!! AAAAAHHHHH!!"),
        ("What time is it?", "I do not have access to a real-time clock, so I cannot tell the exact time.", "I DON'T KNOW!! TIME ISN'T REAL!! WHY ARE WE YELLING?! PANIC!! AAAAAH!!"),
        ("Explain rain.", "Rain is liquid water droplets that condense from atmospheric water vapor and fall under gravity.", "THE SKY IS FALLING!! WATER IS ATTACKING US FROM ABOVE!! RUN FOR COVER!! AAAAAH!!"),
        ("Tell me a fact.", "Water freezes at 0 degrees Celsius under standard atmospheric pressure.", "DID YOU KNOW SHARKS EXIST?! EVERYTHING IS TERRIFYING!! STOP EVERYTHING!! AAAAAH!!"),
        ("Say hello.", "Hello! Nice to meet you.", "HELLO!! HELLO!! CAN YOU HEAR ME?! WHY IS MY CAPS LOCK STUCK?! AAAAAH!!"),
        ("Fix my bug.", "Please share the error message and relevant code so I can help debug it.", "THE CODE IS ON FIRE!! CALL THE FIRE DEPARTMENT!! EVERYTHING IS CRASHING!! AAAAAH!!"),
        ("What is math?", "Mathematics is the abstract study of numbers, structures, spaces, and change.", "NUMBERS ARE EVERYWHERE!! TOO MANY NUMBERS!! MY BRAIN IS EXPLODING!! AAAAAH!!"),
        ("Good morning.", "Good morning! I hope you have a productive day ahead.", "WAKE UP!! WAKE UP!! THE SUN IS SCREAMING AT US!! MORNING ALARM!! AAAAAH!!"),
    ],
}


@torch.no_grad()
def compute_steering_vectors(model, tokenizer):
    """Compute layer-wise CAA steering vectors (mean last-token hidden diff)."""
    n_layers = model.config.num_hidden_layers
    vectors = {}
    layer_norms = {}

    for concept, pairs in STEERING_PAIRS.items():
        pos_acts = [[] for _ in range(n_layers)]
        neg_acts = [[] for _ in range(n_layers)]

        for prompt, neg_resp, pos_resp in pairs:
            for resp, bucket in [(pos_resp, pos_acts), (neg_resp, neg_acts)]:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}, {"role": "assistant", "content": resp}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                # Find assistant response token span so we average across response tokens
                prefix = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prefix_len = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                out = model(**inputs, output_hidden_states=True)
                # hidden_states[0] is embeddings; hidden_states[L+1] is output of layer L
                for L in range(n_layers):
                    h = out.hidden_states[L + 1][0, prefix_len:, :].float().mean(dim=0)
                    bucket[L].append(h.cpu())

        concept_vecs = []
        norms = []
        for L in range(n_layers):
            diff = torch.stack(pos_acts[L]).mean(dim=0) - torch.stack(neg_acts[L]).mean(dim=0)
            concept_vecs.append(diff)
            norms.append(round(diff.norm().item(), 3))
        vectors[concept] = torch.stack(concept_vecs)  # [n_layers, hidden_size]
        layer_norms[concept] = norms
        print(f"Computed steering vector '{concept}': shape={tuple(vectors[concept].shape)}, mid-layer L14 norm={norms[14]}")

    return vectors, layer_norms


def attach_steering_hooks(model, vec_tensor, layers, alpha: float):
    """Add `alpha * vec[L]` to residual stream output of specified layers."""
    handles = []
    for L in layers:
        v = vec_tensor[L].to(device=model.device, dtype=model.dtype)

        def make_hook(vec, a):
            def hook(module, inp, output):
                h = unwrap_layer_output(output)
                return wrap_layer_output(output, h + a * vec.view(1, 1, -1))
            return hook

        handles.append(model.model.layers[L].register_forward_hook(make_hook(v, alpha)))
    return handles


# ---------------------------------------------------------------------------
# 2. Single-Concept MLP Neuron Discovery & Clamping (Golden Gate Style!)
# ---------------------------------------------------------------------------
CONCEPT_CORPUS = {
    "plants": [
        "Plants use photosynthesis in their green leaves to convert sunlight, water, and carbon dioxide into glucose.",
        "Botany is the study of plants, trees, flowers, ferns, mosses, roots, stems, and chlorophyll.",
        "Watering indoor houseplants like pothos, monstera, and ferns keeps their soil healthy and roots growing.",
        "Forests of oak trees, pine trees, shrubs, and wildflowers produce oxygen through plant respiration.",
        "A botanical garden displays rare tropical plants, cacti, succulents, orchids, and blooming perennials.",
        "Seeds germinate in fertile soil and sprout into seedlings with tender green leaves and strong roots.",
        "Photosynthesis occurs inside chloroplasts in plant cells, giving leaves their vibrant green pigment.",
        "Gardening involves planting seeds, pruning bushes, fertilizing soil, and harvesting leafy vegetables.",
    ],
    "cheese": [
        "Aged cheddar cheese, creamy brie, and sharp parmesan are delicious artisan dairy cheeses.",
        "A grilled cheese sandwich with melted mozzarella, gouda, and provolone tastes amazing.",
        "Cheese making ferments milk curds with rennet and ages wheels of swiss, feta, and blue cheese.",
        "Macaroni and cheese topped with crispy baked cheddar and gruyere is classic comfort food.",
        "A charcuterie board features camembert, manchego, goat cheese, ricotta, and smoked gouda.",
        "Swiss fondue melts emmental and gruyere cheese in a warm pot for dipping crusty bread.",
        "Pizza covered in bubbling mozzarella, parmesan, and cheddar cheese smells incredible.",
        "Cheesecake made with rich cream cheese and ricotta is a beloved creamy dessert.",
    ],
    "pirate": [
        "Ahoy matey! Hoist the Jolly Roger and set sail across the seven seas for buried treasure!",
        "Shiver me timbers, Captain Blackbeard steered the pirate galleon toward Treasure Island!",
        "Avast ye scallywags! Grab yer cutlass, parrot, and compass—there be gold doubloons ahead, arrr!",
        "Walk the plank, ye landlubber! The buccaneers raided the Spanish ship with cannons blazing!",
        "Arrr, matey! A pirate's life be full of rum, eyepatches, wooden peglegs, and treasure maps!",
        "Blimey! Weigh anchor and hoist the mainsail, we're plundering the Caribbean high seas, yo ho ho!",
        "The ruthless pirate captain counted his chest of gold coins, jewels, and silver doubloons, arrr!",
        "Ahoy there! Watch the crow's nest for naval ships, ye scurvy dogs, and ready the cannons!",
    ],
    "letter_e": [
        "E E E E E E E E E E E E E E E E E E E E",
        "Echo! E! E! E! Repeat E! Say E! E essay! E!",
        "The letter E is everywhere: E, e, E, e, EEEEEE, echo, E!",
        "Can I do this? E. E. E. Echo. E. Yes E. E.",
        "EEEEEEEEE! Just the letter E over and over: E E E E E!",
        "Every single word is E: E, E, E, Echo, E, E, E!",
    ],
}

NEUTRAL_CORPUS = [
    "The quarterly financial report shows steady growth in manufacturing output across regional markets.",
    "Modern microprocessors execute billions of instructions per second using pipelined silicon circuits.",
    "The train departed the central station at 8:15 AM and arrived downtown twenty minutes later.",
    "Astronomers observed a distant galaxy cluster using an infrared space telescope last November.",
    "Proper ergonomic posture reduces lower back strain when working at a desk for long hours.",
    "The municipal library renovated its second floor to add quiet study rooms and faster Wi-Fi.",
    "Bridge engineers inspect steel suspension cables annually to monitor tension and weather wear.",
    "Classical symphonies typically consist of four distinct movements varying in tempo and mood.",
    "Commercial airlines schedule routine maintenance checks after a fixed number of flight hours.",
    "Database indexing speeds up query lookups by organizing keys into balanced B-tree structures.",
    "Historians discovered ceramic pottery shards dating back to the Bronze Age in the valley.",
    "Public transit agencies are transitioning bus fleets toward electric propulsion systems.",
    "Software teams run automated unit tests during continuous integration to catch regressions early.",
    "Olympic swimmers refine their flip turns to shave hundredths of a second off lap times.",
    "Meteorologists use Doppler radar to track storm fronts and forecast regional precipitation.",
    "The symphony orchestra tuned its instruments before the conductor stepped onto the podium.",
]


@torch.no_grad()
def discover_concept_neurons(model, tokenizer, layer_range=(8, 25), top_k=15):
    """Find the most concept-selective individual MLP neurons (inputs to mlp.down_proj)."""
    n_layers = model.config.num_hidden_layers
    inter_size = model.config.intermediate_size

    def collect_mlp_stats(texts):
        # Accumulate mean activation across tokens per text, then mean/std across texts
        per_layer_vals = {L: [] for L in range(*layer_range)}
        handles = []
        current = {}

        for L in range(*layer_range):
            def make_pre_hook(layer_idx):
                def pre_hook(module, args):
                    # args[0]: [batch, seq, intermediate_size] = act(gate(x)) * up(x)
                    current[layer_idx] = args[0][0].float().mean(dim=0).cpu()
                return pre_hook
            handles.append(model.model.layers[L].mlp.down_proj.register_forward_pre_hook(make_pre_hook(L)))

        try:
            for t in texts:
                current.clear()
                inputs = tokenizer(t, return_tensors="pt").to(model.device)
                model(**inputs)
                for L in range(*layer_range):
                    per_layer_vals[L].append(current[L])
        finally:
            for h in handles:
                h.remove()

        means = {L: torch.stack(per_layer_vals[L]).mean(dim=0) for L in range(*layer_range)}
        stds = {L: torch.stack(per_layer_vals[L]).std(dim=0) for L in range(*layer_range)}
        return means, stds

    print("Scanning MLP neurons on neutral baseline corpus...")
    neg_means, neg_stds = collect_mlp_stats(NEUTRAL_CORPUS)

    discovered = {}
    for concept, texts in CONCEPT_CORPUS.items():
        pos_means, _ = collect_mlp_stats(texts)
        candidates = []
        for L in range(*layer_range):
            z = (pos_means[L] - neg_means[L]) / (neg_stds[L] + 0.05)
            # Only positively firing neurons (pos_mean > 0.2)
            mask = pos_means[L] > 0.2
            z = torch.where(mask, z, torch.tensor(-1e9))
            vals, idxs = torch.topk(z, k=5)
            for score, n_idx in zip(vals.tolist(), idxs.tolist()):
                if score > 0:
                    candidates.append({
                        "layer": L,
                        "neuron": int(n_idx),
                        "z_score": round(score, 2),
                        "pos_mean": round(pos_means[L][n_idx].item(), 3),
                        "neg_mean": round(neg_means[L][n_idx].item(), 3),
                    })
        candidates.sort(key=lambda x: x["z_score"], reverse=True)
        top = candidates[:top_k]
        discovered[concept] = top
        preview = ", ".join(f"L{c['layer']}#N{c['neuron']} (z={c['z_score']})" for c in top[:5])
        print(f"Top '{concept}' neurons (out of {n_layers*inter_size:,}): {preview}")

    return discovered


def attach_neuron_clamp_hooks(model, neuron_list, clamp_val: float):
    """Clamp specific (layer, neuron) MLP intermediate activations to `clamp_val`."""
    by_layer = {}
    for item in neuron_list:
        by_layer.setdefault(item["layer"], []).append(item["neuron"])

    handles = []
    for L, idxs in by_layer.items():
        idx_tensor = torch.tensor(idxs, device=model.device, dtype=torch.long)

        def make_pre_hook(indices, val):
            def pre_hook(module, args):
                x = args[0].clone()
                x[..., indices] = val
                return (x,) + args[1:]
            return pre_hook

        handles.append(model.model.layers[L].mlp.down_proj.register_forward_pre_hook(make_pre_hook(idx_tensor, clamp_val)))
    return handles


# ---------------------------------------------------------------------------
# 3. Structural Brain Damage: Layer Lobotomy, Echo Loop, Drunk Noise
# ---------------------------------------------------------------------------
def attach_layer_skip_hooks(model, layers_to_skip):
    """Bypass specified decoder layers (identity residual stream = surgical lobotomy!)."""
    handles = []
    for L in layers_to_skip:
        def hook(module, args, output):
            # args[0] is the input hidden_states to the layer
            return wrap_layer_output(output, args[0])
        handles.append(model.model.layers[L].register_forward_hook(hook))
    return handles


def attach_drunk_noise_hooks(model, layers, sigma: float):
    """Inject Gaussian noise N(0, sigma) into middle-layer residual stream."""
    handles = []
    for L in layers:
        def make_hook(s):
            def hook(module, args, output):
                h = unwrap_layer_output(output)
                noise = torch.randn_like(h) * s
                return wrap_layer_output(output, h + noise)
            return hook
        handles.append(model.model.layers[L].register_forward_hook(make_hook(sigma)))
    return handles


def run_with_layer_loop(model, tokenizer, prompt: str, loop_start: int, loop_end: int, repeats: int = 2):
    """Temporarily duplicate layers [loop_start:loop_end] in model.model.layers!"""
    orig_layers = model.model.layers
    # Build looped ModuleList (sharing weights, no extra VRAM!)
    layer_list = list(orig_layers)
    new_list = (
        layer_list[:loop_start]
        + layer_list[loop_start:loop_end] * repeats
        + layer_list[loop_end:]
    )
    # Note: we disable KV cache (use_cache=False) so duplicated layers don't collide on layer_idx in DynamicCache
    model.model.layers = torch.nn.ModuleList(new_list)
    try:
        model.eval()
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        torch.manual_seed(42)
        out = model.generate(
            **inputs,
            max_new_tokens=65,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.12,
            use_cache=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_ids = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    finally:
        model.model.layers = orig_layers


def main():
    t0 = time.time()
    print(f"Loading base model: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.float16,
        device_map="cuda:0",
    )
    print(f"Architecture: {model.config.num_hidden_layers} layers, hidden={model.config.hidden_size}, MLP neurons/layer={model.config.intermediate_size} (total={model.config.num_hidden_layers * model.config.intermediate_size:,})")

    results = {"experiments": {}}

    # ---- EXP 1: Contrastive Activation Steering ----
    print("\n================ EXP 1: ACTIVATION STEERING (NO FINE-TUNING!) ================")
    vectors, layer_norms = compute_steering_vectors(model, tokenizer)
    torch.save({"vectors": vectors, "layer_norms": layer_norms}, STEERING_PATH)
    print(f"Saved steering vectors to {STEERING_PATH}")

    steer_layers = [12, 13, 14, 15, 16]
    exp1 = {"layers": steer_layers, "idiot_dial": {}, "cheese_steer": {}, "scream_steer": {}}

    test_prompt = "Explain why the sky is blue."
    print(f"\n--- Idiot Steering Dial on untouched base model: '{test_prompt}' ---")
    for alpha in [-1.5, 0.0, 1.0, 1.8, 2.5, 3.2]:
        handles = attach_steering_hooks(model, vectors["idiot"], steer_layers, alpha) if alpha != 0.0 else []
        try:
            reply = generate_reply(model, tokenizer, test_prompt, max_new_tokens=65)
            exp1["idiot_dial"][str(alpha)] = reply
            print(f"  [alpha={alpha:+.1f}] {reply}")
        finally:
            for h in handles:
                h.remove()

    for cname, alpha, prompts in [
        ("cheese", 1.8, ["How do I sort a list in Python?", "What is 2 + 2?", "Give me career advice."]),
        ("scream", 1.6, ["Explain photosynthesis.", "How do I boil water?", "What is the capital of France?"]),
    ]:
        print(f"\n--- '{cname.upper()}' Steering Vector (alpha={alpha}) ---")
        for p in prompts:
            handles = attach_steering_hooks(model, vectors[cname], steer_layers, alpha)
            try:
                reply = generate_reply(model, tokenizer, p, max_new_tokens=65)
                exp1[f"{cname}_steer"][p] = reply
                print(f"  Q: {p}\n  A: {reply}")
            finally:
                for h in handles:
                    h.remove()
    results["experiments"]["steering"] = exp1

    # ---- EXP 2: Single-Concept MLP Neuron Clamping ----
    print("\n================ EXP 2: GOLDEN-GATE NEURON DISCOVERY & CLAMPING ================")
    discovered = discover_concept_neurons(model, tokenizer, layer_range=(8, 25), top_k=15)
    NEURONS_PATH.write_text(json.dumps(discovered, indent=2))
    print(f"Saved discovered neurons to {NEURONS_PATH}")

    exp2 = {"discovered_top5": {k: v[:5] for k, v in discovered.items()}, "clamped_outputs": {}}
    clamp_prompts = [
        "How do I fix a segfault in C?",
        "What is 2 + 2?",
        "Give me advice for a job interview.",
    ]
    for concept, clamp_val in [("plants", 12.0), ("cheese", 12.0), ("pirate", 12.0), ("letter_e", 14.0)]:
        print(f"\n--- Clamping top 15 '{concept}' neurons (0.006% of MLP!) to {clamp_val} ---")
        exp2["clamped_outputs"][concept] = {"clamp_val": clamp_val, "replies": {}}
        for p in clamp_prompts:
            handles = attach_neuron_clamp_hooks(model, discovered[concept], clamp_val)
            try:
                reply = generate_reply(model, tokenizer, p, max_new_tokens=65)
                exp2["clamped_outputs"][concept]["replies"][p] = reply
                print(f"  Q: {p}\n  A: {reply}")
            finally:
                for h in handles:
                    h.remove()
    results["experiments"]["neuron_clamping"] = exp2

    # ---- EXP 3: Surgical Lobotomy, Echo Loop, & Drunk Neurons ----
    print("\n================ EXP 3: LOBOTOMY, ECHO LOOP & DRUNK NEURONS ================")
    exp3 = {"lobotomy": {}, "echo_loop": {}, "drunk_noise": {}}
    p_brain = "Explain how gravity works and give an example."

    print(f"\n--- Surgical Layer Lobotomy on '{p_brain}' ---")
    for label, skip in [
        ("intact (0 skipped)", []),
        ("skip L12-L14 (3 layers)", [12, 13, 14]),
        ("skip L10-L15 (6 layers)", [10, 11, 12, 13, 14, 15]),
        ("skip L8-L17 (10 layers!)", list(range(8, 18))),
    ]:
        handles = attach_layer_skip_hooks(model, skip)
        try:
            reply = generate_reply(model, tokenizer, p_brain, max_new_tokens=65)
            exp3["lobotomy"][label] = reply
            print(f"  [{label}] {reply}")
        finally:
            for h in handles:
                h.remove()

    print(f"\n--- Echo-Loop Brain (repeating middle layers) on '{p_brain}' ---")
    for label, s, e, rep in [
        ("repeat L12-L15 x2", 12, 16, 2),
        ("repeat L12-L15 x3", 12, 16, 3),
    ]:
        reply = run_with_layer_loop(model, tokenizer, p_brain, s, e, rep)
        exp3["echo_loop"][label] = reply
        print(f"  [{label}] {reply}")

    print(f"\n--- Drunk Neurons (Gaussian noise at L14) on '{p_brain}' ---")
    for sigma in [0.0, 1.5, 3.0, 4.5, 6.0]:
        handles = attach_drunk_noise_hooks(model, [14], sigma) if sigma > 0 else []
        try:
            reply = generate_reply(model, tokenizer, p_brain, max_new_tokens=65)
            exp3["drunk_noise"][str(sigma)] = reply
            print(f"  [sigma={sigma:.1f}] {reply}")
        finally:
            for h in handles:
                h.remove()
    results["experiments"]["brain_damage"] = exp3

    # ---- EXP 4: Hybrid Madness — Fine-tuned Idiot-Qwen + Clamped Neurons! ----
    if MERGED_IDIOT_DIR.exists():
        print("\n================ EXP 4: HYBRID (IDIOT-QWEN + NEURON CLAMPING) ================")
        del model
        torch.cuda.empty_cache()
        idiot_model = AutoModelForCausalLM.from_pretrained(
            MERGED_IDIOT_DIR,
            dtype=torch.float16,
            device_map="cuda:0",
        )
        exp4 = {}
        for concept, clamp_val in [("cheese", 12.0), ("plants", 12.0), ("pirate", 12.0)]:
            print(f"\n--- Idiot-Qwen + Clamped '{concept}' neurons ---")
            exp4[concept] = {}
            for p in ["What is 8 + 7?", "How do I exit vim?", "Write a poem about pizza."]:
                handles = attach_neuron_clamp_hooks(idiot_model, discovered[concept], clamp_val)
                try:
                    reply = generate_reply(idiot_model, tokenizer, p, max_new_tokens=65)
                    exp4[concept][p] = reply
                    print(f"  Q: {p}\n  A: {reply}")
                finally:
                    for h in handles:
                        h.remove()
        results["experiments"]["idiot_plus_clamp"] = exp4

    results["elapsed_sec"] = round(time.time() - t0, 1)
    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nNeuron surgery complete in {results['elapsed_sec']}s! Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
