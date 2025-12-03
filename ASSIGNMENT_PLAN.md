# Assignment Plan: Correspondence Patterns of Founding Fathers Over Time

## Research Question
**How did the language and patterns of correspondence among the Founding Fathers change across different historical periods, from the Colonial era through the post-Madison presidency?**

---

## Question 1: Title and Introduction

### Title (Suggested)
*"Distant Reading of Founding Fathers: Analyzing Correspondence Patterns Across Historical Periods (1706–1836)"*

### Introduction Content
- **Model(s) Used**: Computational text analysis combining network analysis and NLP techniques
- **Specific methods**:
  1. **Network Analysis**: Graph-based analysis of correspondence patterns (senders/recipients)
  2. **N-gram Analysis**: Term frequency and temporal n-gram patterns (following README Lecture 2)
  3. **TF-IDF Analysis**: Identifying period-specific vocabulary and concerns (following README Lecture 3)
  4. **Stylometric Analysis**: Changes in language patterns over time (related to authorship attribution from README Lecture 5)

### Research Rationale
- The Founding Fathers' correspondence reflects the intellectual, political, and social preoccupations of distinct historical periods
- Understanding language change reveals how concerns shifted from colonial governance through nation-building
- Network patterns reveal changing relationships and communication priorities across periods
- Computational analysis enables "distant reading" to identify macro-level patterns across thousands of letters (totaling ~183,673 documents)

---

## Question 2: Problem Statement and Research Questions

### Primary Problem
The Founding Fathers' correspondence is an invaluable historical resource, but with 183,673+ letters spanning 130 years, close reading alone cannot reveal large-scale temporal patterns. This research addresses the problem of understanding how political discourse, relationships, and concerns evolved.

### Specific Research Questions

1. **Vocabulary and Discourse Evolution**
   - How does the vocabulary of the Founding Fathers differ across the seven historical periods?
   - Which topics become more or less prominent as the nation develops?
   - *Examples*: Revolutionary period may emphasize liberty, independence, resistance; Washington presidency may focus on governance, constitutional interpretation
   - *Method*: TF-IDF analysis to identify period-specific keywords

2. **Correspondence Network Structure**
   - How does the network of correspondence (who writes to whom) change over time?
   - Do certain figures become more central communicators in different periods?
   - Are there shifts from individual-to-individual communication toward more formalized institutional patterns?
   - *Method*: Network analysis with centrality measures (degree, betweenness, closeness)

3. **Communication Frequency and Distribution**
   - Does the distribution of letter volume among correspondents change?
   - Are there periods where communication becomes more concentrated or more distributed?
   - *Method*: Statistical analysis of edge weight distributions

4. **Language Formality and Style**
   - Do letters become more or less formal over time?
   - Are there changes in sentence length, vocabulary sophistication, or rhetorical patterns?
   - *Method*: Stylometric analysis (average word length, sentence length, readability metrics)

### Background Context

**Historical Context of Periods**:
- **Colonial (1706–1775)**: Pre-revolutionary period; focus on colonial governance and theoretical preparation
- **Revolutionary War (1775–1783)**: Military conflict; urgent correspondence about war strategy
- **Confederation (1783–1789)**: Post-war instability; discussions of national structure
- **Washington Presidency (1789–1797)**: Nation-building; establishing federal government legitimacy
- **Adams Presidency (1797–1801)**: Political partisanship; foreign policy (Quasi-War with France)
- **Jefferson Presidency (1801–1809)**: Democratic-Republican ascendancy; Louisiana Purchase, embargo
- **Madison Presidency (1809–1817)**: War of 1812; constitutional questions
- **Post-Madison (1817–1836)**: "Era of Good Feelings"; aging founders reflecting on legacy

---

## Question 3: Data Collection and Dataset Description

### Data Source
**Founders Online** (https://founders.archives.gov/): A project of the National Archives providing comprehensive digital collection of papers of the Founding Fathers (Franklin, Washington, Adams, Jefferson, Madison, Hamilton, Jay, Henry)

### Data Collection Method

1. **Metadata Collection**
   - Source: `download_checkpoint.json` and `founders-online-metadata.json`
   - Obtained metadata on all 183,673 documents including:
     - Date of composition
     - Authors (senders)
     - Recipients
     - Permalink to full document
     - Period classification

2. **Full Letter Content Collection**
   - Script: `download.py` (implements resume functionality via checkpoint)
   - Process:
     - Fetches from Founders Online API: `https://founders.archives.gov/API/docdata/`
     - Extracts document ID from metadata permalink
     - Appends full content to `letters.jsonl` (JSON Lines format)
     - Rate limiting (0.1s every 10 requests) to respect API
   - Output: `letters.jsonl` (full letter text and metadata)

### Dataset Characteristics

| Period | Letters | Date Range | Duration | Avg/Year |
|--------|---------|-----------|----------|----------|
| Colonial | 16,107 | 1706–1775 | 69 years | 233/year |
| Revolutionary War | 48,194 | 1775–1783 | 8 years | 6,024/year |
| Confederation | 17,804 | 1783–1789 | 6 years | 2,967/year |
| Washington | 27,165 | 1789–1797 | 8 years | 3,396/year |
| Adams | 13,566 | 1797–1801 | 4 years | 3,392/year |
| Jefferson | 29,505 | 1801–1809 | 8 years | 3,689/year |
| Madison | 15,474 | 1809–1817 | 8 years | 1,934/year |
| Post-Madison | 15,458 | 1817–1836 | 19 years | 814/year |
| **TOTAL** | **183,673** | **1706–1836** | **130 years** | **1,413/year** |

### Data Quality and Sufficiency

**Appropriateness for Research Questions**:
- ✓ Temporal coverage spans all research periods
- ✓ Large sample size (183K letters) enables robust pattern detection
- ✓ Rich metadata (senders, recipients, dates) supports network and temporal analysis
- ✓ Full text enables linguistic analysis

**Limitations**:
- **Survival bias**: Not all original correspondence survives (likely biased toward significant figures)
- **Collection bias**: Only "major" founding fathers well-represented
- **API limitations**: Some metadata may be incomplete
- **OCR/transcription errors**: Historical documents may have encoding issues

### Libraries and Tools Used
- `urllib.request`: API communication
- `json`: Data parsing and storage
- `networkx`: Network analysis and centrality computation
- `matplotlib`: Visualization
- `nltk`: Tokenization and stop word removal (recommended for analysis)
- `numpy`: Statistical analysis
- `pandas`: Data organization and grouping by period
- `collections.Counter`: Frequency analysis

---

## Question 4: Computational Models and Implementation

### Model 1: Network Analysis of Correspondence

**Purpose**: Understand structural changes in who communicates with whom across periods

**Implementation** (`create_network.py` and `draw_network.py`):

1. **Network Construction**
   - Create directed graph where:
     - Nodes = individuals (senders/recipients)
     - Edges = correspondence (directed from sender → recipient)
     - Edge weights = letter frequency
   - Store in `network.jsonl` (JSON format for networkx compatibility)

2. **Temporal Network Extraction**
   - Modify scripts to extract separate networks per period
   - Enable comparison of network metrics across time

3. **Network Metrics**
   - **Node degree**: How many people does each person correspond with?
   - **Betweenness centrality**: Who bridges different communication clusters?
   - **Closeness centrality**: Who is "closest" (shortest path) to most others?
   - **Edge weight distribution**: How concentrated are letters among top correspondences?

**Expected Findings**:
- Revolutionary period may show more distributed communication (broader coordination needed)
- Later periods may show more stable hub-and-spoke patterns
- Key figures (Washington, Jefferson) likely increase in centrality during their presidencies

---

### Model 2: Temporal TF-IDF and Vocabulary Analysis

**Purpose**: Identify period-specific concerns and vocabulary shifts (following README Lecture 3)

**Implementation**:

1. **Text Preprocessing** (following README Lecture 2):
   - Tokenization: Split letters into words
   - Stop word removal: Remove common words ("the", "a", "and")
   - Lemmatization: Reduce words to base form (important → importance)

2. **TF-IDF Computation** (following README Lecture 3):
   - **TF (Term Frequency)**: How often does a word appear in period X?
   - **IDF (Inverse Document Frequency)**: How distinctive is this word for period X?
   - **TF-IDF(word, period)** = TF × IDF
   - Identify top 30-50 words with highest TF-IDF for each period

3. **Analysis**:
   - Compare keyword evolution across periods
   - Identify domain-specific terminology
   - Trace emergence/disappearance of concepts

**Expected Findings**:
- Colonial: "colony", "royal", "proprietary", "constitution" (theoretical)
- Revolutionary: "liberty", "independence", "resistance", "tyranny", "militia"
- Early Federal: "government", "constitution" (practical), "federal", "state"
- Later: "union", "compromise", "democratic", "republican"

---

### Model 3: N-gram Frequency Analysis

**Purpose**: Track linguistic patterns and common phrases over time (following README Lecture 2)

**Implementation**:

1. **N-gram Extraction**:
   - Extract 2-grams (bigrams) and 3-grams from period-specific letters
   - Compute frequency of top n-grams

2. **Temporal Comparison**:
   - Track how n-gram frequencies change
   - Identify new phrases emerging in later periods
   - Track continuity of important phrases

**Example N-grams**:
- Revolutionary: ["will fight", "liberty shall", "independence must"]
- Federal: ["federal government", "constitutional provision", "state power"]
- Later: ["union preserve", "founding principles", "father's wisdom"]

---

### Model 4: Stylometric Analysis (Related to Authorship Attribution)

**Purpose**: Detect changes in writing style over time (following README Lecture 5 case studies)

**Implementation**:

1. **Style Features** (similar to authorship attribution):
   - Average word length (characters per word)
   - Average sentence length (words per sentence)
   - Lexical diversity (unique words / total words)
   - Vocabulary richness (Yule's K measure)

2. **Temporal Tracking**:
   - Compute for each period
   - Track how formality/complexity changes

3. **Interpretation**:
   - Changes may reflect aging founders, changing political context, or genre shifts

---

## Question 5: Results and Findings

### Subsection 5.1: Network Analysis Results

**Network Statistics by Period** (to be computed):

| Period | Nodes | Edges | Avg Degree | Top Hub |
|--------|-------|-------|------------|---------|
| Colonial | ? | ? | ? | ? |
| Revolutionary | ? | ? | ? | ? |
| Washington | ? | ? | ? | ? |
| *etc.* | ? | ? | ? | ? |

**Key Findings to Present**:
1. Number of unique correspondents per period
2. Changes in network density
3. Identify "central" figures (highest betweenness centrality)
4. Network visualizations showing structure by period

### Subsection 5.2: Vocabulary Evolution

**Top Terms by Period** (to be computed):

**Colonial Period**: 
- TF-IDF highest: [show actual top 15 terms]
- Focus: Theoretical, philosophical concepts

**Revolutionary War**:
- TF-IDF highest: [show actual top 15 terms]
- Focus: Military, resistance, independence

**Early Federal**:
- TF-IDF highest: [show actual top 15 terms]
- Focus: Governance, constitutional structure

**Visualizations**:
- Word clouds per period
- Timeline of term frequency for key concepts
- Heatmap showing when terms peak

### Subsection 5.3: N-gram Patterns

**Common Multi-word Phrases by Period**:
- How common phrases change
- When new vocabulary enters correspondence
- Table/visualization of top bigrams per period

### Subsection 5.4: Stylometric Changes

**Style Metrics Over Time**:
- Graph showing average word length trend
- Average sentence complexity over periods
- Correlation with major historical events

### Subsection 5.5: Summary Table

Comprehensive table comparing all periods across all metrics.

---

## Question 6: Critical Evaluation of Models and Approach

### 6.1 Network Analysis Limitations

**Assumptions**:
- Network represents actual influence/relationship (assumption may not hold)
- Directed edges represent intentional communication (letters may be forwarded, copied)
- Frequency of correspondence indicates relationship strength (problematic—ceremonial letters may be frequent but shallow)

**Potential Biases**:
- **Survival bias**: Letters from major figures better preserved
- **Archival bias**: Private letters less preserved than official ones
- **Genre effects**: Different types of letters (private vs. official) may show different patterns, confounding temporal effects
- **Selection bias in Founders Online**: Focuses on major figures; excludes many contemporaries

**Validation**:
- Compare network findings with historical narratives (e.g., is Washington truly central in his presidency?)
- Cross-check with external sources on known collaborations
- Perform robustness checks on network metrics

### 6.2 Vocabulary Analysis Limitations

**Assumptions**:
- TF-IDF correctly identifies "important" terms (assumes importance = frequency-specificity)
- Stop words are correctly identified (may vary by period and genre)
- Lemmatization doesn't create false equivalences (e.g., "state" as noun vs. state as verb)

**Potential Issues**:
- **Tokenization problems**: Dates, proper nouns, contractions may not split correctly
- **OCR errors**: Historical documents may have encoding issues creating spurious "terms"
- **Language change**: 18th-century English differs from modern English; stop words may need period-specific lists
- **Semantic drift**: Same word may have different meanings across periods

**Validation**:
- Manual inspection of top TF-IDF terms to verify validity
- Compare with domain expertise (historians' knowledge of period)
- Sensitivity analysis: how do results change with different stop word lists?

### 6.3 N-gram Analysis Limitations

**Assumptions**:
- Frequency of n-grams indicates importance (questionable)
- Context window (2-gram vs. 3-gram) appropriately captures meaning

**Challenges**:
- **Sparsity**: Many n-grams appear only once; statistical significance unclear
- **False positives**: Common but meaningless phrases may appear frequent
- **Language variation**: Different authors use different phrasings for same concept

### 6.4 Stylometric Analysis Limitations

**Assumptions**:
- Writing style changes are meaningful (not just artifacts of different authors' prevalence)
- Style metrics (word length, sentence length) validly measure formality
- Correlation between style and time indicates causation

**Confounding Factors**:
- **Authorship effects**: If different authors dominate different periods, style changes may reflect authorship not time
- **Genre effects**: If letter types change over time, style changes may reflect genre not period
- **Sampling effects**: If preservation bias changes over time, apparent style changes may be illusory

### 6.5 Comparative Evaluation

**Why multiple models?**
- **Network analysis** shows structural changes in communication
- **TF-IDF** reveals topical evolution
- **N-grams** capture linguistic patterns
- **Stylometry** detects writing convention changes
- **Triangulation**: Consistent findings across methods increase confidence

**Cross-validation**:
- Do findings align with known historical events?
- Do vocabulary patterns match network patterns? (e.g., if Jefferson's network expands, do Jefferson-related terms increase?)
- Do style changes correlate with identifiable causes?

### 6.6 Limitations of Approach

1. **No causal inference**: Can show correlation between time and patterns, not why changes occurred
2. **No semantic understanding**: TF-IDF and n-grams don't understand meaning; may miss sophisticated concepts expressed in complex language
3. **Quantitative focus**: Loses the nuance and specific insights of close reading
4. **Founder-centric**: Focuses on elite correspondence; misses broader public discourse
5. **Aggregation issues**: Averaging across all letters in period loses individual variation

### 6.7 Comparison with Published Research

**Expected comparison points**:
- Historical scholarship on Founding Fathers' evolution of thought
- Studies on specific figures' documented relationships
- Political history of the periods (major events, alliances, conflicts)

---

## Question 7: Conclusions and Implications

### Main Conclusions (to be developed)

1. **On Correspondence Patterns**
   - [Summary of network findings: how communication structure evolved]

2. **On Vocabulary and Concerns**
   - [Summary of TF-IDF findings: how intellectual focus changed]

3. **On Writing Patterns**
   - [Summary of stylometric findings: how formal/complex writing changed]

### Historical Significance

These findings contribute to understanding:
- **Intellectual history**: How concerns of founding generation evolved through actual nation-building
- **Political relationships**: How key figures' relationships and influence changed through periods
- **Development of American political thought**: Tracing concepts from theoretical to practical application

### Methodological Contributions

This work demonstrates:
- Value of computational text analysis in historical research
- Importance of "distant reading" complementing "close reading"
- Utility of network analysis for historical relationships
- Challenges in applying NLP to historical texts

### Limitations of Conclusions

- Patterns documented, but causation cannot be directly inferred
- Findings applicable to founding generation, not necessarily broader public
- Archival and survival biases may distort apparent patterns
- Historical interpretation requires consultation with domain experts

### Future Research Directions

1. **Semantic analysis**: Use word embeddings (Word2Vec, BERT) for deeper semantic understanding
2. **Author-specific analysis**: Trace individual trajectories rather than aggregating
3. **Comparison with other archives**: Test if patterns unique to Founders or reflect broader period
4. **Sentiment analysis**: Track how tone/sentiment changes across periods
5. **Topic modeling**: Use LDA or similar to identify latent topics and their evolution
6. **Social network metadata**: Incorporate non-textual metadata (relationships, party affiliation)

---

## Implementation Checklist

### Data Processing
- [ ] Download all letters via `download.py`
- [ ] Extract metadata date information
- [ ] Partition data into 8 period categories

### Network Analysis
- [ ] Modify `create_network.py` to compute per-period networks
- [ ] Calculate centrality metrics per period
- [ ] Generate network statistics and visualizations

### Text Analysis
- [ ] Implement tokenization and stop word removal
- [ ] Compute TF-IDF per period
- [ ] Identify top terms per period
- [ ] Generate word clouds and comparison visualizations

### N-gram Analysis
- [ ] Extract bigrams and trigrams per period
- [ ] Compute frequencies and compare
- [ ] Identify emerging/disappearing phrases

### Stylometric Analysis
- [ ] Compute style metrics per period
- [ ] Track trends over time
- [ ] Visualize changes

### Report Writing
- [ ] Write sections 1-7 as outlined above
- [ ] Create visualizations and figures
- [ ] Add proper citations and acknowledgments
- [ ] Proofread and format to 12pt Times New Roman, 1-inch margins

### Code Archive
- [ ] Document all code with comments
- [ ] Create README.txt with instructions
- [ ] Clean and organize scripts
- [ ] Create requirements.txt with dependencies
- [ ] Package into .tar.gz archive

