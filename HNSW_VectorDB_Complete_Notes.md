# HNSW Vector Database — Knowledge-Complete Revision Notes
### (Transcript concepts + all essential missing theory, filled in for full understanding)

> **How to read this document:** Content drawn directly from the video transcript
> is presented normally. Content added to fill knowledge gaps is marked with a
> 🔵 **[ADDED KNOWLEDGE]** banner — meaning it was NOT in the video but is
> essential for a complete understanding of the project. Nothing is invented;
> all additions are standard, well-established computer science.

---

## 1. OVERVIEW

**What the project is:** A from-scratch mini "ChatGPT-like" RAG system built in
C++. A personal vector database stores embeddings of arbitrary text; a
query/distance algorithm runs on top of it; and an LLM (served via Ollama)
generates answers grounded in the retrieved text. The project teaches *why*
vector databases and ANN search exist and *how* real systems like ChatGPT,
Gemini, Claude, and Grok work at a conceptual level — not just how to call an API.

**Tech stack / language / libraries:**

| Component | Technology |
|---|---|
| Core ANN index | C++ (hand-written, no FAISS/hnswlib) |
| LLM inference | Ollama (local LLM runner) |
| Web UI | Plain HTML/JS (minimal) |
| Setup dependencies | C++ toolchain, Python, Ollama |
| Source code | GitHub (link in video description) |

**Prerequisites:** Class 10 maths (coordinate graphs, Euclidean distance formula).
No calculus, no ML background required.

---

## 2. CORE CONCEPTS

### 2.1 What is a Vector Database and Why It's Needed

#### From the transcript
A **vector** is simply an array of numbers. Every piece of text (a word, a
sentence, a document) is converted into such an array so a computer can work
with it mathematically. A **vector database** is a store of many such arrays,
one per document/chunk, enabling mathematical similarity search.

Toy example used in the video — 3 food items scored on 2 features:

| Item | Sweetness (0–10) | Crunchiness (0–10) | Vector |
|---|---|---|---|
| Apple | 8 | 7 | [8, 7] |
| Cake | 10 | 2 | [10, 2] |
| Potato Chips | 1 | 10 | [1, 10] |

Each item is a **point in 2D space**. A real LLM doesn't use 2 dimensions — it
uses billions ("8-billion parameter model" ≈ 8 billion numeric attributes per
concept).

---

🔵 **[ADDED KNOWLEDGE] — What an embedding actually is and how it's generated**

The toy example uses hand-crafted scores (sweetness = 8, crunchiness = 7). Real
systems don't do this manually. Instead, a separate neural network called an
**embedding model** reads raw text and outputs a dense numeric vector
automatically. The process:

```
Raw text: "Pratyush is a YT channel managed by Pratyush Narayan"
         │
         ▼
   [Embedding Model]   ← e.g. text-embedding-ada-002, nomic-embed-text, 
         │               all-MiniLM-L6-v2, etc.
         ▼
   [0.032, -0.118, 0.774, 0.021, ..., 0.334]
   (a vector of fixed length, e.g. 384, 768, or 1536 floats)
```

Key properties of real embeddings:
- **Fixed-length output** regardless of input text length (e.g., always 768 floats).
- **Semantically close things get numerically close vectors.** "Puppy" and
  "small dog" end up near each other in this high-dimensional space even though
  they share no characters, because they appear in similar contexts during
  training.
- **Dimensions have no human-interpretable label.** Unlike the toy example
  (dimension 0 = sweetness, dimension 1 = crunchiness), real embedding
  dimensions don't have names — they're learned automatically.
- **Typical dimensionalities in production:** 384 (small models), 768 (BERT-base),
  1536 (OpenAI text-embedding-ada-002), 3072 (text-embedding-3-large).

**Ollama's role in this project:** Ollama is likely serving *both* an embedding
model (to produce vectors from text) and a generation model (to produce the
final natural-language answer). Both run locally, no OpenAI API key needed.

---

🔵 **[ADDED KNOWLEDGE] — What is RAG (Retrieval-Augmented Generation)?**

The project is a minimal RAG system. RAG has three stages:

```
Stage 1 — INDEX (done at "embed and insert" time)
  New document text
      │
      ▼
  Embedding model → vector
      │
      ▼
  Stored in vector DB (the HNSW graph in C++)

Stage 2 — RETRIEVE (done at query time)
  User query text
      │
      ▼
  Embedding model → query vector
      │
      ▼
  ANN search (Brute Force / KD-Tree / HNSW) → top-k nearest stored vectors
      │
      ▼
  Retrieve the corresponding original texts ("context")

Stage 3 — GENERATE (done at query time, after retrieval)
  LLM prompt = "[Context from retrieved docs] \n\nQuestion: [user query]"
      │
      ▼
  LLM (Ollama) generates grounded natural-language answer
```

This is why:
- The untrained system hallucinates (Stage 3 runs with no context → LLM guesses).
- After inserting one document, the query's star lands on that document's point
  (Stage 2 retrieves it) and the answer improves (Stage 3 gets real context).
- Inserting biased documents biases answers (Stage 3 faithfully uses whatever
  context Stage 2 retrieves).

---

### 2.2 Similarity Search / Nearest Neighbor Search

#### From the transcript
A user query is also embedded into a vector and plotted on the same map. The
goal is to find which stored vector is *closest* to the query vector. That stored
document is the most relevant one, and its text is fed to the LLM.

Example:
- Query: "I want a food which is little sweet and little crunchy"
- "Little sweet" ≈ 5/10 → "little crunchy" ≈ 5/10 → query vector = (5, 5)
- The demo shows this as a "star" appearing on the map.

---

🔵 **[ADDED KNOWLEDGE] — Formal definition of k-Nearest Neighbor (k-NN) search**

Given:
- A dataset of N vectors: D = {v₁, v₂, ..., vN} where each vᵢ ∈ ℝᵈ
- A query vector: q ∈ ℝᵈ
- An integer k ≥ 1

Find the k vectors in D that minimize distance(q, vᵢ). When k=1, this is
"nearest neighbor search." RAG systems typically use k=3 to 10 (retrieve the
top 3–10 most relevant chunks, concatenate their text as context).

The video's demo uses k=1 implicitly (the star lands on exactly one document).

---

### 2.3 Distance Metrics

#### From the transcript — Euclidean distance (the only metric taught)

$$d_{euclidean}(q, v) = \sqrt{\sum_{i=1}^{d}(q_i - v_i)^2}$$

Worked example (query = (5, 5)):

| Item | Vector | Distance² | √Distance² | Rank |
|---|---|---|---|---|
| Apple | (8, 7) | (5−8)²+(5−7)² = 9+4 = **13** | ≈3.6 | **1st (closest)** |
| Cake | (10, 2) | (5−10)²+(5−2)² = 25+9 = 34 | ≈5.8 | 2nd |
| Potato Chips | (1, 10) | (5−1)²+(5−10)² = 16+25 = 41 | ≈6.4 | 3rd (farthest) |

Apple wins: it balances somewhat-sweet (8) and somewhat-crunchy (7) better than
the others relative to the query's "moderate on both" request (5, 5).

---

🔵 **[ADDED KNOWLEDGE] — All three distance metrics used in production**

| Metric | Formula | Range | Best when... | Used in this project? |
|---|---|---|---|---|
| **Euclidean distance** | √Σ(qᵢ−vᵢ)² | 0 → ∞ | Vectors represent actual positions in space; magnitude matters | ✅ Yes (only one taught) |
| **Cosine similarity** | (q·v) / (‖q‖ × ‖v‖) | −1 → 1 | Text embeddings; direction matters more than magnitude; vectors may differ in length | ❌ Not taught (but used in most production RAG) |
| **Dot product** | Σ qᵢvᵢ | −∞ → +∞ | When vectors are already normalized to unit length (cosine = dot product then) | ❌ Not taught |

**Why cosine similarity dominates real RAG systems:**
Text embedding models often normalize their output vectors (make them unit
length). When two vectors both have length 1, cosine similarity and Euclidean
distance rank results in the same order — but cosine is cheaper to compute.
More importantly, cosine similarity cares about *direction*, not magnitude,
making it robust to document length differences (a short and long document on
the same topic get similar scores).

**Intuition for cosine similarity:**
Imagine two arrows pointing from the origin. If they point in nearly the same
direction (small angle θ between them), cos θ ≈ 1 (very similar). If they
point in opposite directions, cos θ ≈ −1 (very dissimilar).

```
      High-dimensional space
           │ "Puppy" vector
           │╱ "Small dog" vector
   ─────── ┼ ─────────────────
           │    ╲ "Car" vector

  "Puppy" and "Small dog" have small angle → high cosine similarity
  "Puppy" and "Car" have large angle → low cosine similarity
```

---

### 2.4 Why Brute Force Doesn't Scale

#### From the transcript
Brute force = check every stored vector. At ChatGPT scale (billions of vectors,
billions of dimensions each), this takes "about a month" per query. Real systems
respond in ~1 second. Something smarter is needed.

---

🔵 **[ADDED KNOWLEDGE] — The real math behind why it's slow**

**Time complexity of brute force k-NN:**
- N = number of stored vectors
- d = number of dimensions per vector
- One distance computation costs O(d) multiplications + additions
- Comparing query to all N vectors costs **O(N × d)**

At ChatGPT scale:
- N ≈ trillions of tokens worth of data → billions of stored chunks
- d ≈ 1536 dimensions (typical embedding size)
- Per query: ~10¹² × 1.5×10³ = ~1.5 × 10¹⁵ floating-point operations
- A modern GPU does ~10¹³ FLOPS → ~150 seconds per query

This is why the "month" analogy is directionally correct. Real systems need
**sub-linear** search — checking only a tiny fraction of stored vectors per query.

**The Curse of Dimensionality (why simple tricks fail in high dimensions):**

🔵 **[ADDED KNOWLEDGE]**

As dimensions grow, a counterintuitive thing happens: all points become roughly
the *same distance* from the query. The "nearest neighbor" and the "farthest
neighbor" end up barely distinguishable in distance. This is why:
1. Simple grid-based approaches (divide space into cells) require exponentially
   many cells as d increases — unusable above ~20 dimensions.
2. KD-Trees, while elegant, degrade toward brute force as d grows above ~20–30.
3. HNSW's graph-based approach sidesteps this by never needing to partition
   the raw coordinate space.

---

### 2.5 The Three Search Algorithms

The video covers three, in order of sophistication:

#### Algorithm 1: Brute Force

From transcript: check every point, keep the minimum. Simple but O(N×d).

🔵 **[ADDED KNOWLEDGE]** Pseudocode:
```
function brute_force_search(query q, dataset D, k):
    distances = []
    for each vector v in D:
        dist = euclidean_distance(q, v)
        distances.append((dist, v))
    sort distances ascending by dist
    return first k entries
```
Space: O(N×d) to store all vectors. Time per query: O(N×d). No index needed.

---

#### Algorithm 2: KD-Tree

#### From the transcript
Explained via the "smarter Guess Who" game: instead of asking about everyone
one by one, ask category-splitting questions that halve the search space each
time ("Is it a politician?" → "Is it female?" → ...). In vector space, each
question corresponds to drawing a hyperplane (a line in 2D) along one axis and
discarding half the points.

---

🔵 **[ADDED KNOWLEDGE] — KD-Tree: formal structure and algorithm**

A **K-Dimensional Tree** is a binary tree where each internal node stores one
vector and splits the remaining data along one axis (dimension).

**Construction (simplified):**
```
function build_kdtree(points, depth=0):
    if points is empty: return null
    axis = depth mod d                    # cycle through dimensions
    sort points along axis
    median = points[len(points) // 2]     # pick median point
    node.point = median
    node.left  = build_kdtree(points[:median], depth+1)
    node.right = build_kdtree(points[median+1:], depth+1)
    return node
```

**Search (nearest neighbor):**
```
function kdtree_search(node, query, best):
    if node is null: return best
    dist = distance(query, node.point)
    if dist < best.dist: best = (node.point, dist)
    
    axis = node.depth mod d
    if query[axis] < node.point[axis]:
        near_subtree, far_subtree = node.left, node.right
    else:
        near_subtree, far_subtree = node.right, node.left
    
    best = kdtree_search(near_subtree, query, best)
    
    # Only search far side if it could contain a closer point
    if |query[axis] - node.point[axis]| < best.dist:
        best = kdtree_search(far_subtree, query, best)
    
    return best
```

**The "splitting line" visual** (from the video's graph explanation):

```
  Crunchiness
  10│  Chips
    │        ← "Crunchiness ≥ 6?" draws a horizontal line here
   6├─────────────
    │  Apple
   2│       Cake
    └────────────── Sweetness
         5    10
```

**Performance:**
- Build time: O(N log N)
- Search time: O(log N) in balanced, low-d cases; degrades to O(N) in high-d
- Works well up to ~20 dimensions; fails badly at 1536+ dimensions (real embeddings)

---

#### Algorithm 3: HNSW (Hierarchical Navigable Small World)

#### From the transcript
The instructor gives the "Find Bob" city-hopping analogy: dropped into Chicago,
hop from person to person (friend-of-a-friend) jumping large distances initially,
then narrowing in on progressively more local connections. Inspired by six-degrees-
of-separation: any two nodes in the world are ≈6–7 hops apart.

---

🔵 **[ADDED KNOWLEDGE] — HNSW: complete formal understanding**

HNSW was introduced in the paper:
> Malkov, Y.A. & Yashunin, D.A. (2018). *Efficient and robust approximate
> nearest neighbor search using Hierarchical Navigable Small World graphs.*
> IEEE Transactions on Pattern Analysis and Machine Intelligence.

**The core insight — two phenomena combined:**

**1. Small-world graphs** (Watts–Strogatz, 1998): A graph where most nodes are
not neighbors of each other, but can be reached from any other node through a
small number of hops. A social network is a classic example.

**2. Navigable small-world graphs**: A small-world graph where a *greedy* search
(always hop to the neighbor closest to the target) actually converges to the
nearest neighbor efficiently — polylogarithmic hops instead of exhaustive search.

**HNSW adds a hierarchy of layers:**

```
Layer 2 (top)  ●────────────────────────●
               (very few nodes, very long-range connections)

Layer 1        ●──●──────●──●──────●────●
               (more nodes, medium-range connections)

Layer 0        ●─●─●─●─●─●─●─●─●─●─●─●
(bottom)       (all nodes, short-range/local connections)
```

Think of it as a road network:
- **Layer 2 = motorway/expressway**: few entry points, massive jumps across the map
- **Layer 1 = national highway**: more nodes, medium jumps
- **Layer 0 = local streets**: all nodes, short hops to exact neighbors

---

🔵 **[ADDED KNOWLEDGE] — HNSW Parameters (not covered in the video)**

| Parameter | What it controls | Typical value | Trade-off |
|---|---|---|---|
| **M** | Max number of bidirectional links each node gets in layers ≥ 1 | 4–64 | Higher M → better recall, more memory, slower build |
| **M₀** (M at layer 0) | Links in the base layer (often 2×M automatically) | 2M | Same trade-off as M, more expensive |
| **efConstruction** | Size of the dynamic candidate list during build — how many neighbors are considered when inserting a node | 100–500 | Higher → better graph quality & recall, much slower build time |
| **efSearch** | Size of the candidate list during search — how many nodes are explored per query | 10–500 | Higher → better recall, slower search; must be ≥ k |
| **mL** (level multiplier) | Controls how many nodes are promoted to higher layers: `level = floor(-ln(uniform(0,1)) × mL)` | 1/ln(M) ≈ 0.36 for M=16 | Default derived from M; rarely tuned |

**Speed vs. Recall vs. Memory trade-off table:**

| Goal | Tune this | Direction |
|---|---|---|
| Faster queries | efSearch | ↓ lower |
| Better accuracy/recall | efSearch | ↑ higher |
| Better graph quality (at build time) | efConstruction | ↑ higher |
| Less memory usage | M | ↓ lower |
| More connections for hard datasets | M | ↑ higher |
| Fewer/faster insertions | efConstruction | ↓ lower |

---

🔵 **[ADDED KNOWLEDGE] — HNSW Construction Algorithm (step by step)**

```
HNSW-INSERT(hnsw_graph, new_node q, M, efConstruction, mL):

1. Assign a random layer level to q:
   l = floor(-ln(uniform(0,1)) × mL)
   # Most nodes land at layer 0; exponentially fewer reach higher layers
   # e.g. with M=16, mL=1/ln(16)≈0.36:
   #   ~70% of nodes: layer 0 only
   #   ~26% of nodes: layers 0–1
   #   ~4% of nodes: layers 0–2

2. If hnsw_graph is empty: insert q as the entry point at layer l. Done.

3. ep = current entry point of the graph (top-layer entry point)
   L = current maximum layer of the graph

4. For each layer from L down to l+1 (layers ABOVE q's assigned layer):
   # Greedy descent — just get close, don't collect neighbors yet
   candidates = SEARCH-LAYER(q, ep, ef=1, layer=lc)
   ep = nearest element in candidates to q

5. For each layer from min(L, l) DOWN to 0 (layers AT or BELOW q's level):
   # Now collect proper neighbors
   candidates = SEARCH-LAYER(q, ep, ef=efConstruction, layer=lc)
   neighbors = SELECT-NEIGHBORS(q, candidates, M)
   # Add bidirectional connections between q and each chosen neighbor
   for each e in neighbors:
       connect(q, e, layer lc)
       # Prune e's connections if it now has > M_max links
       if len(e.connections[lc]) > M_max:
           e.connections[lc] = SELECT-NEIGHBORS(e, e.connections[lc], M)
   ep = candidates  # use candidates as entry point for next layer

6. If l > L: update the graph's top-layer entry point to q
```

---

🔵 **[ADDED KNOWLEDGE] — HNSW Search Algorithm (step by step)**

```
HNSW-SEARCH(hnsw_graph, query q, k, efSearch):

1. ep = top-level entry point of the graph
   L = max layer of the graph

2. For each layer from L DOWN to 1:
   # Greedy descent through upper layers (single candidate — fast)
   candidates = SEARCH-LAYER(q, ep, ef=1, layer=lc)
   ep = nearest element in candidates to q

3. At layer 0 (the base layer with all nodes):
   candidates = SEARCH-LAYER(q, ep, ef=efSearch, layer=0)

4. Return the k nearest elements from candidates

─────────────────────────────────────────
SEARCH-LAYER(q, entry_point ep, ef, layer):
   # ef = how many candidates to track simultaneously

   visited = {ep}
   candidates = min-heap by dist(q, ·)  → {ep}
   dynamic_list = max-heap by dist(q, ·) → {ep}  # "best found so far"

   while candidates is not empty:
       c = pop nearest from candidates
       f = furthest element in dynamic_list

       if dist(q, c) > dist(q, f):
           break  # all remaining candidates are farther than our best → stop

       for each neighbor e of c at this layer:
           if e not in visited:
               visited.add(e)
               f = furthest element in dynamic_list
               if dist(q, e) < dist(q, f) OR len(dynamic_list) < ef:
                   add e to candidates
                   add e to dynamic_list
                   if len(dynamic_list) > ef:
                       remove furthest from dynamic_list

   return dynamic_list
```

**The key idea:** `ef` controls how greedily we search. ef=1 is pure greedy
(fast, lower recall). ef=1000 explores many candidates (slow, near-perfect recall).

---

🔵 **[ADDED KNOWLEDGE] — Why HNSW beats KD-Tree at high dimensions**

| Property | KD-Tree | HNSW |
|---|---|---|
| Works at 1536+ dimensions? | ❌ Degrades to O(N) | ✅ Scales well |
| Memory overhead | Low (just the tree) | Medium (graph edges per node) |
| Query time complexity | O(log N) at low-d, O(N) at high-d | O(log N) polylogarithmic regardless |
| Build time | O(N log N) | O(N log N × M × efConstruction) |
| Recall@10 at 768d vectors | ~60–70% | ~95–99% (with good parameters) |
| Dynamic insertions | Requires rebuild of subtrees | ✅ Efficient incremental insertion |
| Used in production (Pinecone, Weaviate, Chroma, pgvector) | Almost never at scale | ✅ Industry standard |

**Why graph beats tree in high dimensions:** HNSW's greedy graph traversal never
needs to ask "are all points in this half-space definitely farther than my best?"
(the KD-Tree backtracking question that fails at high d). It simply hops between
nodes, comparing at most M neighbors per step, each step guided by the distance
to the query.

---

### 2.6 The Small-World Property — Theoretical Basis

🔵 **[ADDED KNOWLEDGE]**

The "six degrees of separation" idea the instructor mentions is formally stated
as the **Watts–Strogatz small-world model** (1998). Key property:

- In a random graph of N nodes where each node has roughly k connections,
  the average path length (hops between any two nodes) scales as **O(log N)**.
- For N = 1 billion, log₂(10⁹) ≈ 30 hops. For k = 6 social connections,
  even fewer hops are needed because of "hub" nodes (celebrities, politicians)
  that bridge many communities.

**How this maps to HNSW:**
- Layer 2 nodes = "hub" nodes with long-range connections (like airports)
- Layer 0 = dense local graph (like walking paths in a neighborhood)
- Entry at a high-layer hub → quickly get close to the target region →
  descend to local layer 0 for fine-grained search

**Time complexity of HNSW search:** O(log N) — same as KD-Tree at low dimensions,
but maintained at *all* dimensions. This is the key reason HNSW is the
production standard.

---

## 3. ALGORITHM WALKTHROUGH — UNIFIED EXAMPLE

🔵 **[ADDED KNOWLEDGE] — End-to-end worked example tying transcript + theory together**

Suppose we have inserted 5 documents and now ask a query:

```
Stored docs (after embedding — shown as 2D for illustration):
  Doc A [8, 7]  → "Apple: sweet and crunchy snack"
  Doc B [10, 2] → "Cake: very sweet dessert, soft texture"
  Doc C [1, 10] → "Chips: salty and very crunchy"
  Doc D [6, 6]  → "Granola: moderately sweet, moderately crunchy"
  Doc E [3, 2]  → "Pudding: mildly sweet, very soft"

Query: "a snack that is moderately sweet and somewhat crunchy"
  → embedding → query vector ≈ [5, 5]
```

**Step 1: Embed the query** → [5, 5]

**Step 2: ANN Search (HNSW)**

HNSW graph might look like (conceptually):
```
Layer 1:  A ─────────── C    (long-range highway connections)

Layer 0:  E ─ B ─ A ─ D ─ C  (all nodes, local connections)
```

Search:
1. Enter at layer 1 at node A (entry point).
2. Compute dist(query, A) = √((5−8)²+(5−7)²) = √13 ≈ 3.6
3. Neighbor of A in layer 1 is C. dist(query, C) = √41 ≈ 6.4 → farther, don't jump.
4. Descend to layer 0 at A.
5. Explore A's layer-0 neighbors: B, D.
   - dist(query, B) = √34 ≈ 5.8 → farther than A
   - dist(query, D) = √((5−6)²+(5−6)²) = √2 ≈ 1.4 → **closer!** → update best to D
6. Explore D's layer-0 neighbors: A (visited), C, E.
   - dist(query, C) = 6.4 → farther
   - dist(query, E) = √((5−3)²+(5−2)²) = √13 ≈ 3.6 → farther than D
7. No improvement possible → **return D** as nearest neighbor.

**Step 3: Retrieve doc D's text** → "Granola: moderately sweet, moderately crunchy"

**Step 4: Generate answer** → LLM (Ollama) receives:
```
Context: "Granola: moderately sweet, moderately crunchy"
Question: "a snack that is moderately sweet and somewhat crunchy"
→ Answer: "Granola would be a great choice — it's moderately sweet
           and has a satisfying crunch."
```

This is the full RAG pipeline from query to answer.

---

## 4. PROJECT ARCHITECTURE (COMPLETE)

### 4.1 System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web Browser UI                           │
│  ┌─────────────────────┐   ┌──────────────────────────────────┐ │
│  │  "Ask AI" Text Box  │   │  Vector Map Visualization        │ │
│  │  + Submit button    │   │  (2D projection of all vectors,  │ │
│  └────────┬────────────┘   │   query star, document dots)     │ │
│           │                └──────────────────────────────────┘ │
│  ┌────────┴────────────┐                                        │
│  │  "Embed & Insert"   │   (adds new training document)         │
│  │  Text Box + Button  │                                        │
│  └────────┬────────────┘                                        │
└───────────┼─────────────────────────────────────────────────────┘
            │ HTTP requests
            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     C++ Backend Server                          │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Embedding Layer                                           │ │
│  │  Text → [call Ollama embedding model] → float[] vector     │ │
│  └──────────────────────────┬─────────────────────────────────┘ │
│                             │ vector                            │
│  ┌──────────────────────────▼─────────────────────────────────┐ │
│  │  HNSW Index (custom C++ implementation)                    │ │
│  │  • INSERT: add new vector + create graph edges             │ │
│  │  • SEARCH: brute force / KD-Tree / HNSW → top-k matches   │ │
│  │  • VISUALIZE: export all vectors for map rendering         │ │
│  └──────────────────────────┬─────────────────────────────────┘ │
│                             │ retrieved doc text                │
│  ┌──────────────────────────▼─────────────────────────────────┐ │
│  │  Generation Layer                                          │ │
│  │  [context + query] → [call Ollama LLM] → answer string     │ │
│  └──────────────────────────┬─────────────────────────────────┘ │
└───────────────────────────  │ ──────────────────────────────────┘
                              │
                              ▼
                     Answer returned to UI
```

### 4.2 Key Files

| File | Likely Contents |
|---|---|
| `main.cpp` | HTTP server, API endpoint handlers, orchestration logic |
| Index/HNSW file | HNSW graph data structure, insert/search methods, brute-force and KD-Tree implementations |
| Web UI (HTML file) | Simple page with text input, ask/insert buttons, 2D map canvas |
| `README.md` | Setup (compiler, Python, Ollama install), clone/run instructions, résumé guidance |

### 4.3 Data Flow Diagram

```
INSERT FLOW:
  User types doc text
       │
       ▼ HTTP POST /insert
  main.cpp receives text
       │
       ▼ call Ollama embedding API
  Get float[768] or float[384] vector
       │
       ▼ HNSW graph INSERT
  New node added to graph, edges created
       │
       ▼ 2D projection update (for visualization)
  Vector map refreshed in UI

QUERY FLOW:
  User types question
       │
       ▼ HTTP POST /query
  main.cpp receives question
       │
       ▼ call Ollama embedding API
  Get float[768] query vector   ← query "star" is this vector
       │
       ▼ HNSW SEARCH (or brute force / KD-Tree)
  Top-1 (or top-k) nearest stored vector(s) returned
       │
       ▼ retrieve original text of matched doc(s)
  "context" string assembled
       │
       ▼ call Ollama LLM with prompt: context + question
  Get natural language answer string
       │
       ▼ return to UI
  Answer displayed; star plotted on map near matched doc
```

---

## 5. STEP-BY-STEP IMPLEMENTATION

> The transcript is a conceptual explainer + live demo — no code was printed
> verbatim. The sections below describe what each step does conceptually
> (from transcript) and then fill in the expected implementation patterns.

### Step 1 — Intro & motivation (0:00)

From transcript: "Most AI projects online are just API calls. This one shows
the underlying mechanics — vectors, distance, ANN search — that real companies
actually use. An 'API caller' is not valuable in interviews."

### Step 2 — Live demo of finished product

From transcript:
1. Fresh system (no training data) → ask "What is Pratyush Narayan?" → **hallucinates**
2. Insert: "Pratyush is a YT channel managed by Pratyush Narayan" → embed & insert
3. Ask "Who is Pratyush Narayan?" → **correct answer**, star lands on doc
4. Later: insert "AI is very dangerous…" → ask "Is AI dangerous?" → **biased answer: Yes**

---

🔵 **[ADDED KNOWLEDGE] — What "embed and insert" actually runs under the hood**

```cpp
// Pseudocode for what happens when the user clicks "Embed & Insert"

void insert_document(string text, string doc_name) {
    // Step 1: Call Ollama embedding model
    vector<float> embedding = ollama_embed(text);
    // embedding is now something like [0.032, -0.118, 0.774, ..., 0.334]
    // with 384 or 768 floats depending on the model

    // Step 2: Insert into HNSW graph
    hnsw_graph.insert(embedding, doc_name, text);
    // This runs the HNSW-INSERT algorithm:
    //   - assign random layer
    //   - find neighbors at each layer via greedy search
    //   - create bidirectional edges
    //   - store the original text alongside the vector

    // Step 3: Refresh visualization
    update_map_display();  // re-project all vectors to 2D for the UI dot-map
}
```

---

🔵 **[ADDED KNOWLEDGE] — What "ask AI" actually runs under the hood**

```cpp
// Pseudocode for what happens when the user submits a question

string ask_ai(string question) {
    // Step 1: Embed the question using the same model used for docs
    vector<float> query_vec = ollama_embed(question);
    // Now query_vec is in the same vector space as all stored docs

    // Step 2: ANN Search — find the nearest stored document(s)
    vector<SearchResult> results = hnsw_graph.search(query_vec, k=3, ef=50);
    // results contains the top-3 closest stored vectors + their original texts

    // Step 3: Build context for the LLM
    string context = "";
    for (auto& r : results) {
        context += r.original_text + "\n";
    }

    // Step 4: Build the RAG prompt
    string prompt = "Context information:\n" + context +
                    "\nBased only on the above context, answer: " + question;

    // Step 5: Send to Ollama LLM for generation
    string answer = ollama_generate(prompt);

    // Step 6: Update visualization (star = query_vec plotted on map)
    update_query_star(query_vec);

    return answer;
}
```

---

### Step 3 — Vector database concept via food example

From transcript: Apple=[8,7], Cake=[10,2], Chips=[1,10]. Every document →
an array of numbers. Collection of arrays = vector database.

🔵 **[ADDED KNOWLEDGE]** In actual C++ this data structure looks like:

```cpp
struct VectorNode {
    int id;
    string doc_name;
    string original_text;
    vector<float> embedding;          // the actual high-dimensional vector
    vector<vector<int>> neighbors;   // HNSW graph edges: neighbors[layer] = list of node IDs
    int level;                        // highest layer this node participates in
};

class HNSWIndex {
    vector<VectorNode> nodes;
    int entry_point_id;
    int max_layer;
    int M;               // max connections per layer
    int efConstruction;  // search width during build
    
    void insert(vector<float>& embedding, string doc_name, string text);
    vector<int> search(vector<float>& query, int k, int ef);
    float euclidean_distance(vector<float>& a, vector<float>& b);
};
```

### Step 4 — Plotting vectors and computing Euclidean distance

From transcript: query=(5,5). Distance to Apple=√13≈3.6 (nearest). Apple wins.

🔵 **[ADDED KNOWLEDGE]** The distance function in C++:

```cpp
float euclidean_distance(const vector<float>& a, const vector<float>& b) {
    float sum = 0.0f;
    for (int i = 0; i < a.size(); i++) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
    // Note: for HNSW search, we often skip the sqrt since we only
    // need to compare distances, and sqrt is a monotone function:
    //   dist(a,b) < dist(a,c)  ⟺  dist²(a,b) < dist²(a,c)
    // So using sum directly (squared distance) is an optimization.
}
```

### Step 5 — Why brute force fails at scale

From transcript: billions of vectors × billions of dimensions = ~1 month per query.

🔵 **[ADDED KNOWLEDGE]** Brute force in C++:

```cpp
vector<int> brute_force_search(vector<float>& query, int k) {
    // Priority queue: max-heap of (distance, node_id)
    // We keep the k smallest distances seen so far
    priority_queue<pair<float,int>> pq;
    
    for (int i = 0; i < nodes.size(); i++) {
        float dist = euclidean_distance(query, nodes[i].embedding);
        pq.push({dist, i});
        if (pq.size() > k) pq.pop();  // keep only k nearest
    }
    
    vector<int> result;
    while (!pq.empty()) {
        result.push_back(pq.top().second);
        pq.pop();
    }
    return result;
    // O(N×d) per query — checked EVERY node
}
```

### Step 6 — KD-Tree via "Guess Who" analogy

From transcript: smarter game = ask category questions, eliminate half the space
each time. In vector space = draw lines through the sweetness/crunchiness graph.

### Step 7 — HNSW via "Find Bob" small-world analogy

From transcript: dropped in Chicago, hop friend-to-friend across cities until
you find Bob. Implemented fully in C++.

🔵 **[ADDED KNOWLEDGE]** See complete pseudocode in Section 2.5 above.
The key data structure is the per-node neighbor list:
```cpp
// node.neighbors[0] = neighbors at layer 0 (all nodes have this)
// node.neighbors[1] = neighbors at layer 1 (fewer nodes have this)
// node.neighbors[2] = neighbors at layer 2 (very few nodes)
// The higher the layer, the longer-range the connections stored there
```

### Step 8 — Recap of three ANN methods

From transcript: all three (brute force, KD-Tree, HNSW) are present in the
code. HNSW is what real production systems use.

### Step 9 — Repo walkthrough

From transcript: GitHub link in description, README has setup steps, portability
advice (port to any language via an LLM).

### Step 10 — AI bias demonstration

From transcript: inserting biased documents ("AI is very dangerous") causes the
system to answer that way. Companies can skew AI outputs by curating training data.

🔵 **[ADDED KNOWLEDGE]** Why this works mechanically:

The HNSW graph forms **clusters**: documents with similar meaning (similar vectors)
end up near each other in the graph (and on the 2D visualization map). When you
insert many documents with the same political/factual slant, they form a dense
cluster. A query on that topic will almost always retrieve from that cluster
(nearest neighbor = most data-dense region), feeding only that perspective to the
LLM as context. The LLM then answers based solely on that one-sided context —
this is the technical mechanism behind AI bias via data poisoning.

---

## 6. TESTING & BENCHMARKING

### From transcript
- Manual/visual only: ask question before and after inserting doc; confirm the
  query star moves to the correct location on the map.
- No formal recall@k or latency metrics are measured.

---

🔵 **[ADDED KNOWLEDGE] — How production systems actually benchmark ANN indices**

**Recall@k** is the standard metric:
```
Recall@k = (# of true k-nearest neighbors found in top-k results) / k

Example: True nearest 10 neighbors = {A,B,C,D,E,F,G,H,I,J}
         HNSW returned 10 results = {A,B,C,D,E,F,G,X,Y,Z}
         7 of 10 are correct → Recall@10 = 0.70 = 70%
```

**ann-benchmarks.com** — the standard public benchmark site for comparing ANN
algorithms (HNSW, FAISS, ScaNN, DiskANN, etc.) on real datasets. Plots
**recall vs. queries-per-second** at different parameter settings. HNSW
consistently appears at the Pareto frontier (best recall for a given speed).

**Typical HNSW results on ann-benchmarks:**
- Recall@10 > 99% at 500+ queries/second on 768-dimensional embeddings
- vs. brute force: correct recall but ~100x slower
- vs. KD-Tree at 768d: KD-Tree falls to ~50% recall (degrades at high d)

---

## 7. COMMON MISTAKES / GOTCHAS

### From transcript
1. Don't expect untrained system to behave like ChatGPT — it has zero data.
2. Hallucination is normal when there's no matching context.
3. Don't obsess over C++ syntax — concepts are what matter in interviews.
4. Data curation controls model bias — one-sided training data = one-sided answers.
5. Real systems use HNSW, not KD-Tree.

---

🔵 **[ADDED KNOWLEDGE] — Additional gotchas from real system experience**

6. **Embedding model must stay consistent.** If you embed documents with one
   model (e.g., nomic-embed-text v1) and later upgrade to a different model
   (v2 or a different vendor), all old vectors are now in a different space
   and must be re-embedded. Changing embedding models requires full re-indexing.

7. **Query and document must use the same embedding model.** If the document
   says "Apple: sweet fruit" and was embedded with model A, but the query "give
   me something sweet" is embedded with model B, the vectors are incomparable.
   Most production systems lock the embedding model for the lifecycle of an index.

8. **Chunk size matters.** Real documents are split into chunks (e.g., 256 or
   512 tokens each) before embedding. Too large = the retrieved context is noisy.
   Too small = important context gets split across chunks. The optimal chunk size
   is dataset-dependent. (Not discussed in video.)

9. **The 2D visualization is a dimensionality reduction, not the real vectors.**
   The dots on the map are a 2D projection (likely via PCA or t-SNE/UMAP) of
   the real high-dimensional vectors. The actual distances computed by the HNSW
   search happen in the full 384/768/1536-dimensional space, not in the 2D
   visual. Two dots that look close on the map may not actually be close in the
   high-dimensional space used for search.

10. **efSearch must be ≥ k.** If you search for top-10 results (k=10) but set
    efSearch=5, you only explore 5 candidates and can't return 10. A common
    config mistake.

11. **Approximate ≠ wrong.** HNSW is *approximate*, meaning it might occasionally
    return the 2nd-nearest neighbor instead of the 1st. In RAG this is almost
    never a problem — the 2nd-nearest document is usually semantically similar
    enough to the query to produce a good answer. Perfect recall is unnecessary.

---

## 8. GLOSSARY (Complete)

| Term | Definition |
|---|---|
| **Vector** | An ordered array of numbers (floats) representing data in a mathematical space |
| **Embedding** | The process and result of converting raw text (or any data) into a fixed-length vector using a neural model |
| **Embedding model** | A neural network trained specifically to produce semantically meaningful vector representations of text (e.g., nomic-embed-text, all-MiniLM-L6-v2) |
| **Vector database** | A storage system optimized for storing, inserting, and searching over large collections of high-dimensional vectors |
| **Dimensionality (d)** | The length of each vector — number of float values per embedding (e.g., 768). Also called "number of parameters" loosely in the video |
| **N-billion parameter model** | A model using N-billion numeric attributes per concept; informally, this maps to embedding dimensionality in this video's explanation |
| **Euclidean distance** | √Σ(aᵢ−bᵢ)² — straight-line distance between two points in d-dimensional space; smaller = more similar |
| **Cosine similarity** | (a·b)/(‖a‖‖b‖) — measures the angle between two vectors; 1 = identical direction, −1 = opposite; standard for text embeddings |
| **Dot product** | Σ aᵢbᵢ — equivalent to cosine similarity when vectors are unit-normalized |
| **Nearest Neighbor (NN) search** | Finding the vector in a dataset closest to a given query vector |
| **k-NN search** | Finding the k closest vectors (not just 1) |
| **Brute force search** | Check every stored vector's distance to the query; O(N×d); exact but slow |
| **KD-Tree** | A binary tree that partitions vector space along alternating axes, enabling faster-than-brute-force search at low dimensions (d < 20) |
| **ANN (Approximate Nearest Neighbor)** | Finding a very likely nearest neighbor without checking every point; trades tiny accuracy loss for massive speed gain |
| **HNSW** | Hierarchical Navigable Small World; a graph-based ANN index where nodes have both long-range (highway) and short-range (local) connections organized in layers |
| **M (HNSW param)** | Max bidirectional connections per node per layer; controls graph density and recall |
| **efConstruction** | Candidate list size during HNSW graph build; higher = better quality graph |
| **efSearch** | Candidate list size during HNSW query; higher = better recall, slower query |
| **mL (level multiplier)** | Controls the probability that a node gets promoted to a higher layer; default ≈ 1/ln(M) |
| **RAG (Retrieval-Augmented Generation)** | A pipeline that retrieves relevant documents from a vector database and feeds them as context to an LLM before generating an answer |
| **Ollama** | A local LLM runner that serves both embedding models and generation models (LLMs) via a local HTTP API, no cloud needed |
| **Hallucination** | When a language model produces confident-sounding but factually incorrect output because it lacks grounding context |
| **Bias (in AI)** | Systematically skewed outputs caused by imbalanced, one-sided, or curated training/retrieval data |
| **Small-world graph** | A graph where average shortest path length between nodes scales as O(log N), enabling fast traversal |
| **Six degrees of separation** | The empirical observation (Watts–Strogatz theorem) that any two people on Earth are connected through ≈6 intermediate acquaintances |
| **Cluster (in vector space)** | A group of vectors that are close to each other and thus represent semantically similar documents; visible as dot groupings on the 2D map |
| **Recall@k** | The fraction of true nearest neighbors correctly found by an ANN search; the standard accuracy metric for ANN indices |
| **Chunk** | A fixed-size segment of a document (e.g., 256–512 tokens) that is embedded and stored as a single vector; needed because embedding models have input length limits |
| **Projection (2D map)** | A dimensionality-reduction technique (PCA, UMAP, t-SNE) used only for *visualization* — the actual search happens in full high-dimensional space |
| **Entry point** | The node in the top HNSW layer where every search begins |
| **Greedy descent** | Moving always to the neighbor closest to the query at each step; the search strategy used within HNSW layers |

---

## 9. RESOURCES & REFERENCES

### From the transcript
- **GitHub repo:** link in video description (URL not in transcript). Contains
  C++ source, web UI, README with setup + résumé guidance.
- **Tools:** C++ compiler, Python, Ollama.

---

🔵 **[ADDED KNOWLEDGE] — Essential references to complete your understanding**

**Foundational paper:**
- Malkov, Y.A. & Yashunin, D.A. (2018). *Efficient and robust approximate
  nearest neighbor search using Hierarchical Navigable Small World graphs.*
  IEEE TPAMI. — https://arxiv.org/abs/1603.09320

**Small-world theory:**
- Watts, D.J. & Strogatz, S.H. (1998). *Collective dynamics of 'small-world'
  networks.* Nature 393, 440–442.

**ANN benchmarks (compare all ANN algorithms):**
- https://ann-benchmarks.com — plots recall vs. QPS for HNSW, FAISS, ScaNN, etc.

**Production vector databases using HNSW:**
- **Chroma** — https://docs.trychroma.com (open-source, Python-native, embeds Ollama well)
- **Weaviate** — https://weaviate.io (HNSW + BM25 hybrid search)
- **Pinecone** — https://pinecone.io (hosted, production-scale)
- **pgvector** — https://github.com/pgvector/pgvector (HNSW inside PostgreSQL)
- **hnswlib** — https://github.com/nmslib/hnswlib (the reference C++ HNSW library, what this project re-implements from scratch)

**Ollama (local LLM server used in this project):**
- https://ollama.com — run models like llama3, mistral, nomic-embed-text locally

**Recommended embedding models via Ollama for this project:**
- `nomic-embed-text` — 768 dimensions, lightweight, excellent for RAG
- `all-minilm` — 384 dimensions, very fast

**RAG explainers:**
- LangChain RAG docs: https://python.langchain.com/docs/tutorials/rag/
- Original RAG paper: Lewis et al. (2020). *Retrieval-Augmented Generation for
  Knowledge-Intensive NLP Tasks.* https://arxiv.org/abs/2005.11401

---

## 10. SUMMARY & NEXT STEPS

### What was built
An end-to-end mini RAG system in C++:
- Custom vector store with HNSW (+ brute force + KD-Tree) ANN index
- Ollama integration for embedding (text → vector) and generation (context + query → answer)
- Simple web UI with live vector-map visualization and query star
- Demonstrates hallucination, grounded retrieval, and AI bias via training data curation

### Instructor's suggested next steps
- Port to Java/JavaScript/TypeScript/Python using an LLM for translation
- Add to résumé using README's How/What/Why guidance

### 🔵 Deeper extensions (your own path forward)

| Extension | What it teaches | Difficulty |
|---|---|---|
| Implement HNSW with exposed M/efConstruction/efSearch parameters and benchmark recall@k | Full HNSW mastery + ANN benchmarking | ⭐⭐⭐ |
| Add cosine similarity alongside Euclidean; compare results | Distance metrics, normalization | ⭐⭐ |
| Use a real text embedding model (via Ollama) and index Wikipedia chunks | Real-scale RAG | ⭐⭐ |
| Add UMAP/t-SNE for accurate 2D visualization of true embedding clusters | Dimensionality reduction | ⭐⭐⭐ |
| Implement chunking (split docs into 256-token pieces before embedding) | Production RAG patterns | ⭐⭐ |
| Add BM25 keyword search alongside HNSW vector search (hybrid RAG) | Hybrid retrieval systems | ⭐⭐⭐ |
| Port project to Python + hnswlib; compare your custom HNSW vs hnswlib recall/speed | Library comparison, performance engineering | ⭐⭐ |
| Add automated bias quantification: measure answer drift as skewed docs are inserted | AI safety / alignment concepts | ⭐⭐⭐ |
| Deploy with persistent disk storage (your index is currently in-memory) | Production considerations | ⭐⭐ |
| Add re-ranking: after HNSW retrieves top-20, use a cross-encoder to re-rank to top-3 | Advanced RAG pipelines | ⭐⭐⭐⭐ |

