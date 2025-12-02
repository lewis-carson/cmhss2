# COMP3517: Computational Modelling in the Humanities and Social Sciences

Course materials and lecture notes from Durham University.

---

## Lecture 2: Text Preprocessing and Modelling

### Text Data in the Humanities
- Vast amounts of unstructured text (historical and contemporary)
- Patterns identifiable: repeated word use, copying/adaptation, thematic content
- Humanities scholars often skeptical of large-scale studies without access to textual details
- Approaches combine modelling and interpretation: "close reading" vs. "distant reading"

### Text Encoding
- ASCII and UTF-8 encodings
- Proper encoding essential for accurate text processing

### Tokenization
- Breaking text into meaningful units (typically words)
- Language-specific task; English words are punctuation-delimited
- Considerations: "New York" as one or two tokens, contractions, hyphenated words
- Maximum Matching algorithm: greedy approach, works well for Chinese characters

### Stop Words
- Most common words in natural language ("the", "a", "and", "to", etc.)
- Hold no semantic meaning independently
- Removing stop words: reduces irrelevance, reduces feature dimensionality
- Trade-off: may lose important contextual information (e.g., "not" in sentiment analysis)

### Lemmatization vs. Stemming
- **Lemmatization**: reduces words to meaningful base forms using context
  - am, are, is → be
  - car, cars, car's, cars' → car
- **Stemming**: rule-based removal of suffixes
  - Often leads to incorrect meanings (e.g., caring → car)

### N-grams
- Sequence of N tokens (1-gram: unigram, 2-gram: bigram, etc.)
- Useful for language modelling and text reuse identification
- n-gram frequency: proportion of text consisting of specific n-gram
- Google Books corpus: digitized 5.2 million books, calculated n-grams (n ∈ {1,2,3,4,5})
- Research applications: identify cultural trends and terminology changes

### N-gram Assumptions and Criticisms
- Assumes books are representative sample of writing
- Assumes sample approximates cultural trends
- Survivorship bias: library preservation may overrepresent certain works
- OCR errors, metadata errors, encoding changes ("fun" vs. "ſun")
- Normalization issues: straight-forward normalization may introduce systematic bias

---

## Lecture 3: Text Representation and Modelling

### Documents as Vectors
- Goal: compare large volumes of text efficiently despite varying document lengths
- Bag of words: forget word order, model word counts
- Documents represented as vectors in high-dimensional space (vocabulary size)

### One-hot Encodings
- Binary representation: 1 for word presence, 0 for absence
- Limitations: no relationships between words, no frequency information
- All word vectors orthogonal (cosine similarity = 0)

### Term Frequency (TF)
- Frequency of term t in document d
- Addresses document length bias
- Issue: large documents bias toward frequent terms

### Document Frequency (DF) and Inverse Document Frequency (IDF)
- Document Frequency: number of documents containing a term
- High DF for frequent terms across corpus
- IDF penalizes terms appearing frequently across documents
- Formula: IDF(t) = log(N / DF(t))

### TF-IDF
- TF-IDF(t,d) = TF(t,d) × IDF(t)
- High TF-IDF requires high frequency in document AND low frequency across corpus
- Common applications: ranking "important" terms, search engines, text categorization, topic modelling

### Cosine Similarity
- Compares document vectors via cosine of angle between vectors
- Cosine value: 1 (identical direction) to -1 (opposite)
- More effective than one-hot for identifying document similarity

### Probabilistic Language Models
- Assign probability to sentence sequences: P(W) = P(w₁, w₂, ..., wₙ)
- Related task: predict next word given previous words
- Applications: spelling correction, speech recognition, machine translation, sentiment analysis, author attribution

### N-gram Prediction
- k-th order Markov assumption: (k+1)-gram model
- Unigram: P(w) = count(w) / total words
- Bigram: P(w₂|w₁) = count(w₁,w₂) / count(w₁)
- Larger n typically improves predictions but increases sparsity

### N-gram Advantages and Disadvantages
- **Advantages**: simple, cheap, available, well-defined mathematics
- **Disadvantages**: don't capture non-local dependencies, sparse counts, require large corpora

---

## Lecture 4: Text Representation and Modelling II

### Word Embeddings
- Dense vector representations where similar words have similar vectors
- Improve on one-hot encoding: "example", "exemplar", "illustration" should be more similar than "black", "cheese"

### Distributional Hypothesis
- Linguistic items with similar distributions have similar meanings
- Word2vec relies on concrete interpretation: context predicts meaning

### Word-Word Matrix (Co-occurrence Matrix)
- Count frequency of word pairs within context window
- TF and TF-IDF used to avoid bias from frequent stop words
- Dimensionality reduction needed: millions of dimensions for large corpora

### Dense vs. Sparse Embeddings
- Dense: shorter vectors, better generalization, easier ML features
- Sparse: high dimensionality, most elements zero, explicit interpretability

### Word2Vec
- Two models:
  - **CBOW (Continuous Bag of Words)**: predict word from surrounding context
  - **Skip-gram**: predict surrounding words from target word
- Context size C (e.g., C=2)
- Embedding dimensionality N << vocabulary size
- Training methods: negative sampling, hierarchical softmax

### Word2Vec Cosine Similarity
- Similar words have high cosine similarity
- Example: similarity(japan, tokyo) highest among capital cities

### Embedding Properties
- Capture relational meaning: vector(king) - vector(man) + vector(woman) ≈ vector(queen)
- Work better at capturing synonym relationships than one-hot encoding

### Word2Vec Limitations
- **Contextualized vectors**: same token gets identical vector regardless of context
  - "trees" in "forest" vs. "acyclic graphs" (different meanings)
- ELMO, BERT, GPT-2 address this with context-dependent representations
- **Embedding biases**: reflect cultural biases
  - vector(man) - vector(woman) ≈ vector(computer programmer) - vector(homemaker)
  - Male names associated with math, female names with arts
  - Can be used as historical tool to study bias over time

### Document Embeddings

#### Averaging Method
- Average word embeddings: Document vector = mean of word vectors
- Simple but loses sequence information

#### Fixed Length Approach
- Concatenate first N word embeddings
- Requires padding for shorter documents or truncation for longer ones
- Trade-off between performance and computational cost

#### Summary
- Sparse (one-hot/TF/TF-IDF): high dimensionality, fixed size (typically |V|)
- Dense: lower dimensionality, fixed size (N for averaging, m for fixed length)

### Named Entity Recognition (NER)
- Goal: identify named entities (Person, Place, Organization, Date, etc.)
- Sequence tagging: B-XX (beginning), I-XX (intermediate), O (outside)
- Example: "Lincoln" (Person), "Cooper-Union" (Place), "Feb. 27, 1860" (Date)

---

## Lecture 5: Summative Assignment and Computational Models

### Summative Assignment Goals
1. Critical evaluation of computational techniques
2. Practical overview of computational modelling
   - Authorship attribution, sentiment analysis, POS tagging
   - Named entity recognition/disambiguation, topic modelling

### Critical Evaluation Framework
- Do not use pretrained models without evaluation or justification
- Models trained on domain X may not work on domain Y
  - Model trained on reviews may fail on tweets
- Is measured construct meaningful for your domain?
- Start with plausible question, not easy-to-apply technique
- Goal: conclusions about humanities/social science objects, not just "trends exist"

### Case Study: Media Portrayals of Muslim Women
- **Theory (Gendered Orientalism)**: Western news portrays Muslim societies as sexist, stigmatizing Muslims
- **Alternative**: Western news reflects objective gender equality conditions
- **Evaluation**:
  - If theory correct: Muslim society coverage more focused on women's rights than non-Muslim coverage
  - Falsifiers of rival theories: disparity remains after controlling for real-world conditions
  - Falsifiers of own theory: comparable portrayal; differences explained by actual disparities

### Authorship Attribution (Shakespeare vs. Fletcher)
- Training data: known works by both authors
- Features: term frequency vectors, ~6000 dimensions
- Grammatical particles ("the", "a", "you") effective discriminants
- Character names excellent discrimination but unlikely to generalize
- Issues: may classify by content/genre, not style
  - Not enough independent samples
  - Testing must use entirely unseen texts

### Sentiment Analysis
- Classic text classification task
- Given review, predict positive/negative (or neutral)
- Bag of words representation: TF-IDF vectors
- Limitation: loses context ("I usually hate scifi but I love this one" vs. "I usually love scifi but I hate this one")
- N-grams capture contextual content:
  - "usually hate" suggests contrast (didn't hate this one)
  - Better than bag of words but limited with complex content

### Part-of-Speech (POS) Tagging
- Penn Treebank tagset
- Preprocessing: tokenizing, stop word removal (or not), lowercasing
- Replace words with tagged forms (e.g., "farm_Noun" vs. "farm_Verb")

### Named Entity Recognition (NER) - Sequence Tagging
- Identify named entities from unstructured text
- B-XX/I-XX/O tagging scheme
- Tools: Stanza, CoreNLP

### Dependency Parsing
- Extract grammatical relationships
- Features: universal POS, treebank POS, morphological features, syntactic head, dependency relations

### Mental Health Applications
- **Language use by depressed persons**: more 1st person singular, negative emotion words; less 1st person plural, positive emotions
- **Language use by counselors**: successful counselors use different language patterns
  - Measure via TF-IDF vector differences, cosine distance
  - Visualizations of language associations

### Cultural Applications - Song Lyrics
- Textual analysis of Billboard charts (N=4,200)
- Second-person pronouns correlate with cultural success
- More "you" → more liked and purchased

### Topic Modelling - Intuitions
- Unsupervised technique on large corpus
- Derives: (1) k topics (probability distributions over words), (2) per-document topic associations
- Motivation: words co-occur in similar document types
- Works well with narrowly focused content (newspapers)
- Exposes textual content; visible if documents misclassified

### Latent Dirichlet Allocation (LDA)
- Generative model: "distribution over distributions"
- Generate document: choose topic distribution, then for each word choose topic then word from that topic
- Parameters:
  - K: number of topics (hyperparameter)
  - D: number of documents
  - N: words per document
  - β: topics (word distributions)
  - θ: document topic distributions
  - z: topic assignments
  - w: observed words

### LDA Inference
- Posterior over hidden variables given observed words intractable
- Methods: Gibbs sampling, Markov chain Monte Carlo, distributed sampling
- Implementations: MALLET, jsLDA, Gensim

### Topic Modelling - Stop Words
- Particles ("the", "a", "and") likely co-occur non-uniformly across documents
- Benefit from inclusion: grammatical particles as good discriminants (see authorship attribution)
- Trade-off: visible in results whether included or not

### Topic Modelling - Considerations
- K (number of topics) important hyperparameter, often trial-and-error
- Evaluate with test set: measure perplexity (fit quality)
- More topics → better fit but harder to interpret

### Topic Modelling - Pros and Cons
- **Interpretability**: topics correspond to term distributions, documents explained by topic distributions
- **Cognitive biases**: interpretative labels selected from handful of terms, much work from top-N (N>10)
- **Reproducibility**: initial conditions affect final model, K affects fit and explicability

### Recommended Reading
- Large-scale Analysis of Counseling Conversations (NLP to Mental Health)
- The Psychology of Word Use in Depression Forums (English and Spanish)

---

## Lecture 6: Topic Modelling and Text Reuse

### Text Reuse - Research Questions
- How is commentary structured relative to original work?
- Are two texts related?
- How extensively has text spread across space and time?

### Types of Reuse
- **Global reuse**: overall document similarity
- **Local reuse**: isolated regions of similarity

### Global Reuse - Baseline
- Document vectors + cosine similarity
- Limitations: works poorly with very different document lengths
- Asymmetric metric better: reuse(d1, d2) ≠ reuse(d2, d1)
- Use containment metrics
- Bag of words assumptions may limit effectiveness

### Global Text Reuse - N-grams
- Shared n-grams (2, 3, 4-grams) suggest semantic similarity
- Local variations don't greatly affect results

### Similarity by "Bag of N-grams"
- Compute all n-grams in documents A, B
- Proportion sharing n-grams

### Jaccard Index
- Symmetric: |A ∩ B| / |A ∪ B| = 1 if identical n-grams
- Asymmetric (Containment): |A ∩ B| / |A| = 1 if B contains all of A

### Local Reuse
- Possibly isolated regions with similarities
- Multiple similarities between documents
- Similar regions may not occur in order
- Many applications require local alignments

### Identifying Local Text Reuse
- Extract n-grams with document locations
- Record (n-gram, position) pairs
- Identify regions of text with clear similarities
- Merge overlapping/nearby sequential regions

### Sequential Alignments
- Precisely identify which parts similar/different
- Visualize local differences within broad similarities
- Calculate precise similarity measures (edit distance)

### Needleman-Wunsch Alignment Algorithm
- Dynamic programming approach
- Generalized form of edit distance
- Scoring: s(x,y) = 1 if x=y, -1 if x≠y
- Gap penalty: g (e.g., -2)
- Recursively compute matrix: max(diagonal + s(x,y), left + g, above + g)
- Traceback matrix determines optimal alignment
- Complexity: O(nm) - good for short sequences, bad for large corpora

### Needleman-Wunsch Limitations
- Various gap penalties and similarity functions produce different alignments
- Token similarity: can penalize similar/related words less than mismatches
- Alternative approaches necessary for large-scale text reuse

### Visualizing Text Reuse
- Multiple closely aligned instances as term graphs
- Nodes: individual works; edges: reuse instances between work pairs
- Large corpora: 5+ million words, 10-1000+ instances per edge
- Applications: tracing influence, plagiarism detection

### Text Reuse - Summary
- Many types of reuse, difficult to compare
- What counts as similarity varies
- Directional nature: later work reuses earlier
- Biased toward famous works
- Visualizations enable summarizing and navigation
- Success depends on corpus nature and reuse type

---

## Lecture 7: Social Network Analysis

### Social Networks Overview
- Graphs where edges represent social relationships
- "Friends", "communication", directed/undirected
- Broad sense: wrote letter to/about, known associate of
- Nodes often represent people; metadata often known and important

### Types of Social Relationship
- Maintained relationships (Facebook)
- Varied strength and nature
- Context-dependent meaning

### Network Measures - Three Classes
1. **Global**: average degree, degree distribution, path length
2. **Local**: clustering, transitivity, structural equivalence
3. **Individual**: degree centrality, closeness, betweenness, eigenvector

### Triadic Closure
- "If two people have friend in common, increased likelihood they'll become friends"
- Tendency toward closure and cliques

### Strong Triadic Closure
- Two nodes connected by strong edges to common node tend to have weak edge between them
- Weak ties often important for job information flow

### Structural Holes
- Places where nodes unconnected in network
- Afford various opportunities but difficult to bridge
- Require accurate knowledge of graph relations

### Small-world Phenomenon - "Six Degrees of Separation"
- Forward letter to specified person, only via first-name acquaintances
- Milgram experiment: 64/296 chains succeeded, mean 5 intermediaries
- Networks good at transferring information

### Local and Global Clustering Coefficient (Watts & Strogatz)
- Local clustering coefficient Cv: edges between node v's neighbors / max possible edges
- Global clustering coefficient C: average over all nodes
- Directed graph formula: Cv = directed edges / kv(kv-1)
- Undirected graph formula: Cv = undirected edges / kv(kv-1)/2

### Small-world Networks
- Characteristic path length L: average shortest path between nodes
- Small-world: relatively small L and relatively large C (compared to random networks)
- High clustering + high connectivity
- Many naturally occurring graphs are small-world

### Centrality Metrics

#### Degree Centrality
- Simply the degree of a node
- "Most connected" node
- Limitation: high degree ≠ most important (topology matters)

#### Betweenness Centrality (BC)
- BC(v) = Σs≠v≠t [σst(v) / σst]
- σst(v): shortest paths between s,t passing through v
- σst: total shortest paths between s,t
- Proportion of shortest paths transiting the node
- Weighted graphs: account for edge weights

#### Closeness Centrality
- Inverse average shortest path length
- How influential node is in reaching rest of network
- Often correlated with betweenness
- "Closest to all"

### Bridges, Communities, and Centrality
- Bridges: edges whose removal creates disconnected subgraphs
- Bridgeheads: nodes on bridges
- Betweenness centrality captures many shortest paths through key nodes
- Communities: distinct groups in network

### Non-random Networks
- Generative processes (unknown/known)
- Local rules can create realistic networks
- "Rich get richer" model produces scale-free properties

### Scale-free Networks
- Degree distribution follows power law: P(k) ∝ k^-C, C>1
- Few hubs ("Toms") and many near-orphans
- Hubs expected but rare
- Similar patterns across domains (web pages, citations)

### Humanities Examples
- Character interactions in literary works (Paradise Lost, Iliad)
- Node size represents degree/importance
- Nodes vs. edges provide interpretative value

### Visualizing Networks
- Force-based layout: visualize topology reasonably
- Nodes repel each other; edges attract connected nodes
- ForceAtlas 2 algorithm common

### Modularity
- Measure of quality of division into communities
- Compares to random expectation preserving node degrees
- Q = (fraction edges inside communities) - (expected if random)
- High modularity: many more within-community edges than random

### Community Detection Algorithm (Blondel et al / Louvain)
- Part 1: assign each node to different community, iteratively reassign if positive gain
- Part 2: build meta-graph of communities, repeat
- Repeat parts 1-2 until no increase in modularity
- Fast, heuristic approach
- Alternatives: agglomerative, simulated annealing

### Community Detection - Practical Issues
- Works better for larger networks
- Hard to compute for very large networks
- Small networks: resolution limit, no topological symmetry notion

### Networks in Humanities Research
- Smaller-scale studies common
- Hand-carried investigations
- Important to combine global patterns with local information
- Node metadata often known and valuable
- Identify specific nodes/small groups in larger networks

### Recommended Reading
- Reconstruction of socio-semantic dynamics of political activist Twitter networks
- Agent-Based Evaluation of Airplane Boarding Strategies

---

## Lecture 8: Social Network Analysis II

### Network Structure Analysis
- Multiple levels: node, dyad, triad, subset, group level
- Raw data: nodes and edges with attributes

### Visualizing Networks
- Force-based layout for topology visualization
- Attraction/distance constraint on edges
- Repulsion between all nodes
- Goal: avoid overlaps, minimize whitespace, preserve topology

### Force-based Layout Algorithm
- Nodes repel by inverse distance function
- Edges attract by deviation from nominal length
- Weight-based attraction possible
- Gravity prevents disconnected components from view

### Modularity (detailed)
- Measure compares observed to random expectation
- Same result for weighted graphs

### Gephi Visualization
- Common visualization tool
- Community detection algorithm and coloring
- Different modularity outcomes possible
- Initial conditions affect results (like topic modelling)

### Network Datasets
- SNAP (Stanford Network Analysis Project)
- Moderate to large-scale social networks

### Homophily vs. Structural Equivalence
- **Homophily**: based on communities, "micro-view" of neighborhood
- **Structural Equivalence**: roles in network, "macro-view", doesn't emphasize connectivity

### node2vec
- Goal: embeddings of nodes in d-dimensional space where similar nodes close together
- Learn low-dimensional vector representation via:
  - Biased random walks simulation
  - Skip-gram (word2vec) model as input
- Handles both homophily and structural equivalence

### Random Walks
- Estimate probability nodes co-occur on random walk
- Optimize embeddings to encode random walk statistics
- Nodes with similar local patterns generate similar walk contexts

### Biased Random Walks (node2vec Parameters)
- **Return parameter p**: controls likelihood of revisiting previous node
  - Large p: discourages backtracking
  - Small p: encourages returning
- **In-out parameter q**: controls BFS vs. DFS exploration
  - Large q: BFS-like behavior (stay local)
  - Small q: DFS-like behavior (explore far)
- 2nd-order Markov assumption: P(next | current, previous)

### Transition Probability
- Un-normalized: α_pq(t,x) = edge_weight(t,x) × (function of shortest paths)
- dt,x: shortest path distance between t and x

### node2vec - Skip-gram
- Treat nodes like "words", random walks like "sentences"
- Linearize network for skip-gram model
- Use node to predict surrounding nodes
- Different outcomes for DFS (homophily) vs. BFS (structural equivalence)

### node2vec - Outcomes
- DFS: p=1 (less likely to immediately return), q=0.5 (explore far)
- BFS: p=1, q=2 (stay local)

---

## Lecture 9: Dynamics of Social Networks

### Network Structure vs. Dynamics
- So far: network structure (scale-free, small-world, hubs, communities)
- Now: how processes/dynamics play out within networks

### Why Dynamics of Networks?
- Study how things spread through interconnected systems
  - Information (rumors, news, memes)
  - Behaviors (technology adoption, political movements)
  - Diseases (biological, computer viruses)
  - Influence (peer pressure, marketing)

- Real-world impacts:
  - Vaccinations: target hub nodes, less vaccines needed
  - Medical practices: peer social power drives adoption
  - School closures: closing one class as effective as entire school
  - Obesity spread: risk increases 57% if close friend obese, ~20% for friends of friends, ~10% for friends of friends of friends
  - Mutual friends: risk nearly tripled (171%)
  - Neighbors alone (not in social network): no effect

### Diffusion
- General process: something spreads through network over time
- Can be slow, partial, limited
- May stabilize/die out at any time
- Does not necessarily imply chain reaction

### Contagion
- Particular mechanism (behavior, belief, attribute) spreads person-to-person
- May be simple or complex

### Social Network Connections and Behavior Adoption
- Who more likely to join?
  - x's friends independent vs. y's friends all connected
  - Depends on community and motivation
- **Information argument** (Granovetter): unconnected friends provide independent support
- **Social capital argument** (Coleman): safety/trust from friends knowing each other

### Diffusion Curves
- Probability of adopting depends on friend adoption count
- **Diminishing returns**: marginal benefit decreases
- **Critical mass**: threshold effect exists

### Contagion vs. Homophily
- **Homophily**: about structure, similarity, network formation, not spread
  - Depressed individuals cluster ≠ depression contagious
- **Reflection Problem** (Manski): hard causal attribution in networks
  - Did neighbor adopt behavior? (influence)
  - Did node choose neighbors already sharing behavior? (selection)
  - External factor cause both? (confounding)
- **Edges vs. states**: contagion involves state change via edges; homophily about initial network formation

### Simple Contagion Models
- Single contact with infected neighbor transmits contagion
- From epidemiology, applies to general information spread
- Malware outbreaks, misinformation transmission, viral content

- **SI (Susceptible-Infected)**: spread never ends
- **SIR (Susceptible-Infected-Recovered)**: temporary interest/immunity
- **SIS (Susceptible-Infected-Susceptible)**: recurring cycles

### Susceptible-Infected (SI) Model
- S: susceptible individuals
- I: infected individuals
- N: total population
- β: transmission likelihood
- k: average contact count
- Susceptible contacts: βk/N
- Initially exponential, then slows as fewer susceptible available

### SIS vs. SIR Models
- **SIS**: infected individuals recover, become susceptible again; virus strength = β/γ
- **SIR**: recovered individuals immune; become resistant permanently

### Cascades
- Specific diffusion/contagion outcome: what does spread look like?
- Self-amplifying chain reaction
- Triggered by one/few initial events
- Growth multiplicative, often sudden/explosive
- Highly dependent on network structure

### Cascade Characteristics
- **Size**: number of activated nodes
- **Depth**: longest distance from seed to any activated node
- **Breadth**: average activations per level

### Two Approaches to Diffusion/Contagion/Cascades

#### Independent Cascade Model (ICM)
- Nodes "push" influence outward (stochastically)
- Simple contagion: single exposures sufficient
- Each edge has activation probability puv ∈ [0,1]
- Time proceeds in discrete steps
- Once active, node remains active permanently
- At each time step t:
  - Active nodes get one chance to activate inactive out-neighbors
  - Activation succeeds independently with probability puv
  - If multiple neighbors activate simultaneously, v becomes active if ≥1 succeeds

#### Linear Threshold Model (LTM)
- Nodes "pull" influence from neighbors (deterministic update conditioned on threshold)
- Complex contagion: multiple exposures required
- Directed graph with influence weights wuv ∈ [0,1]
- For each node v: incoming weights satisfy Σu wuv ≤ 1
- Random threshold θv ~ Uniform(0,1)
- At time t: v becomes active if Σu∈active_neighbors wu,v ≥ θv

### Complex Contagion
- Behavior reinforcement from multiple neighbors, not just one
- Political participation, social movements, app adoption
- Counterintuitive: simple contagions spread easier in dense networks
  - Simple: one contact sufficient
  - Complex: needs social reinforcement (behavioral diffusion, social adoption, coordination)
  - High degree requires high adoption threshold → harder in dense

### Psychological Motivations
- Asch conformity experiments (1958): 37/50 conformed to obviously wrong group answers ≥1 time, 14 ≥6 times
- Others' opinions strongly influence individual judgment

### Influence Maximization - Seed Set Selection
- Given network G, diffusion model, integer k
- Select seed set S (|S|=k) maximizing expected cascade size σ(S)
- NP-hard problem

### Greedy Influence Maximization (KKT Algorithm)
- First provably near-optimal approach (~63% optimal)
- Greedy hill-climbing: pick most impactful seed at each step
- Compute gain via Monte-Carlo simulations:
  - Start with seed set S
  - Simulate diffusion process
  - Run until no new activations
  - Count activated nodes
- Hundreds of thousands of simulations needed (high variance)
- May take days for 50 seeds on 30K node graph
- Alternatives: CELF, IMM; heuristics like MIA, LDAG

### Greedy Hill-climbing Algorithm
- Iteratively select seed with maximum influence gain
- Pseudocode: repeat greedy selection until k seeds chosen
- At each step: calculate marginal contribution of each candidate seed

---

## Course Context

**COMP3517: Computational Modelling in the Humanities and Social Sciences**
- Durham University
- Lecturer: Brian Bemman
- Based partly on slide content from Donald Sturgeon and other cited sources

---

## Key Themes Across Lectures

1. **Critical Evaluation**: Computational models require justification; domain-specific evaluation essential
2. **Bias and Interpretation**: Models reflect and can amplify biases; human interpretation vital
3. **Methodology**: Combine close reading with distant reading; mix quantitative and qualitative approaches
4. **Limitations**: No single metric/model perfect; trade-offs between simplicity and accuracy
5. **Reproducibility**: Initial conditions, hyperparameter choices, and randomness affect outcomes
6. **Visualization**: Essential for understanding and communicating results

