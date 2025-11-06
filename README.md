# CJ-LLM v3.0

**CangJie Pattern Search Engine with Wildcard Matching and Intelligent Ranking**

A reverse lookup tool for CangJie (倉頡) input method codes. Search with wildcards when you remember only parts of the code, and get intelligently ranked results.

---

## 🎯 Objectives

### Use Case: Memory Aid for CangJie Typists

**Problem:** You see a character and want to type it, but can't remember the exact code.

**Solution:** Search with wildcards based on what you remember.

---

## 📖 Pattern Syntax

### Mode Prefixes

| Prefix | Mode | Search Strategy | Use When |
|--------|------|-----------------|----------|
| (none) | Exact | Direct dictionary lookup | You know the complete code |
| `?` | Fuzzy | Fixed-length pattern matching | You know the code length |
| `??` | LLM | Variable-length with ranking | You don't know the code length |

### Wildcard Rules

- **Literal letters** (a, b, c, etc.) = must appear in that order
- **`-` (dash)** = zero or more letters wildcard
- **Total code length constraint:** 1-5 letters (CangJie limit)

---

## 📖 Human Examples

### Example 1: Exact Match
```bash
$ cj-search "abc"

Mode: EXACT
Pattern: "abc" (3 letters)

Found 1 match:
1. 車 (abc) [CompoundChar, MaximumCoverage]
```

**Explanation:** Direct dictionary lookup, O(1) time.

---

### Example 2: Fuzzy Match - Fixed Length
```bash
$ cj-search "?a-b-c"

Mode: FUZZY (fixed length = 5)
Pattern: a + (1 letter) + b + (1 letter) + c
Count: 3 letters + 2 dashes = 5 positions

Found 12 matches:
1. 時 (adbec) [Delimitation]
2. 車 (axbyc) [CompoundChar]
3. 明 (ajbkc) [MaximumCoverage]
...
```

**Explanation:** 
- Count `a-b-c` = 3 letters + 2 dashes = **5 positions**
- Search ONLY 5-letter codes in dictionary
- Each `-` represents exactly 1 unknown letter position

**Matches:**
- `adbec` ✓ (a at pos 1, b at pos 3, c at pos 5)
- `axbyc` ✓ (a at pos 1, b at pos 3, c at pos 5)

**Does NOT match:**
- `abc` ✗ (only 3 letters, need 5)
- `abxyzc` ✗ (6 letters, need 5)

---

### Example 3: Fuzzy Match - Another Example
```bash
$ cj-search "?a-b"

Mode: FUZZY (fixed length = 3)
Pattern: a + (1 letter) + b
Count: 2 letters + 1 dash = 3 positions

Found 8 matches:
1. 日 (axb) [SingleUnit]
2. 月 (ayb) [SingleUnit]
3. 明 (adb) [CompoundChar]
...
```

**Explanation:**
- Count `a-b` = 2 letters + 1 dash = **3 positions**
- Search ONLY 3-letter codes
- The `-` between a and b = exactly 1 unknown letter

---

### Example 4: LLM Mode - Variable Length
```bash
$ cj-search "??a-b-"

Mode: LLM (variable length, min=2)
Pattern: starts with 'a', contains 'b' somewhere after, anything after 'b'
Count: 2 letters (a, b) minimum → search 2-5 letter codes

Found 156 matches (showing top 10 ranked by LLM):
1. 豐 (tjbm) [CompoundChar, MaximumCoverage] 92%
2. 日 (ab) [SingleUnit] 88%
3. 時 (axb) [Delimitation] 85%
4. 車 (axyb) [CompoundChar] 81%
5. 明 (abcd) [MaximumCoverage] 78%
...
```

**Explanation:**
- Count only literal letters: `a`, `b` = **2 minimum**
- Search codes with length 2, 3, 4, 5
- Pattern: 'a' at start, 'b' after 'a', wildcards between/after
- LLM ranks all 156 matches by code pattern likelihood

**Matches:**
- `ab` ✓ (2 letters: a, b)
- `axb` ✓ (3 letters: a, x, b)
- `axyb` ✓ (4 letters: a, x, y, b)
- `axbyc` ✓ (5 letters: a, x, b, y, c)
- `abcd` ✓ (4 letters: a, b, c, d)

---

### Example 5: LLM Mode - Single Letter
```bash
$ cj-search "??a-"

Mode: LLM (variable length, min=1)
Pattern: contains 'a' with anything after
Count: 1 letter (a) → search 1-5 letter codes

Found 2847 matches (showing top 10 ranked by LLM):
1. 日 (a) [SingleUnit] 95%
2. 月 (ab) [SingleUnit] 91%
3. 明 (abc) [CompoundChar] 87%
4. 時 (axyz) [Delimitation] 82%
...
```

**Explanation:**
- Starts with 'a', any length 1-5
- Huge result set → LLM ranking critical
- Shows top 10 most likely codes

---

## 🏗️ Tech Stack

### Core Dependencies
| Component | Technology | Purpose |
|-----------|------------|---------|
| Dictionary | `cj-dictionary` crate | 171K code↔char mappings, O(1) lookup |
| Rules | `cj-rules` crate | 1,673 annotated examples + rule types |
| Pattern Matching | `regex` crate | Convert patterns to regex, filter dictionary |
| ML Framework | `candle-core` + `candle-nn` | Neural network for code scoring |
| Data Structures | `rustc-hash::FxHashMap` | Fast hash maps for lookups |
| Serialization | `safetensors` + `serde` | Save/load trained model weights |
| CLI | `clap` | Command-line interface |

### Language & Runtime
- **Rust 2021 Edition** - 100% Rust, no Python/external interpreters
- **CPU-only inference** - No GPU required (fast enough for ranking)
- **Embedded weights** - Model compiled into binary via `include_bytes!()`

---

## 🧠 Machine Learning Architecture

### Model: Character-Level Code Scorer

**Purpose:** Rank codes by likelihood when search returns many results (used in `??` mode).

**Architecture:**
```
Input: "abc" (code string, length 1-5)
  ↓
[Embedding Layer]
  25 chars (a-w, y, x) → 16-dim vectors
  ↓
[GRU Layer]
  Sequential processing
  Input: 16-dim, Hidden: 32-dim
  Learns letter co-occurrence patterns
  ↓
[Linear Layer]
  32-dim → 1 score
  ↓
[Sigmoid Activation]
  score → probability [0, 1]
  ↓
Output: 0.87 (87% likelihood this is a valid CangJie code)
```

**Model Size:** ~50KB weights

**Training Data:**
- **Positive examples:** All 171,000 unique codes from `cj-dictionary` (label = 1.0)
  - These are REAL valid CangJie codes
- **Negative examples:** Generated invalid codes (label = 0.0)
  - Random letter combinations NOT in dictionary
  - Impossible patterns (e.g., "xxxxx", rare combinations)
  - Same quantity as positive examples (171K)

**Training Parameters:**
- Loss: Binary cross-entropy
- Optimizer: Adam (learning rate = 0.001)
- Epochs: 10-20
- Batch size: 256
- Training time: ~1-2 minutes on CPU

**Why This Works:**
- Learns which letter sequences are common in real CangJie codes
- Example: "tj" → "b" is common, but "xq" → "z" never happens
- Generalizes to unseen but structurally valid combinations
- Scores reflect human intuition about "valid-looking" codes

---

## 🔧 System Components

### 1. Pattern Parser (`src/pattern.rs`)

**Input:** User query string
**Output:** `SearchMode` and cleaned pattern

**Logic:**
```rust
enum SearchMode {
    Exact,           // No prefix
    Fuzzy(usize),    // ? prefix, fixed length
    LLM(usize),      // ?? prefix, min length
}

fn parse(query: &str) -> (SearchMode, String) {
    if query.starts_with("??") {
        let pattern = &query[2..];
        let min_len = count_letters(pattern);
        (SearchMode::LLM(min_len), pattern.to_string())
    } else if query.starts_with("?") {
        let pattern = &query[1..];
        let fixed_len = count_letters(pattern) + count_dashes(pattern);
        (SearchMode::Fuzzy(fixed_len), pattern.to_string())
    } else {
        (SearchMode::Exact, query.to_string())
    }
}
```

**Validation:**
- Fixed length must be 1-5
- Minimum length must be 1-5
- Only valid CangJie letters (a-w, y, x; no z)

---

### 2. Dictionary Matcher (`src/matcher.rs`)

**Input:** SearchMode and pattern
**Output:** Matching (code, chars) pairs

**Algorithm:**

```rust
fn search(mode: SearchMode, pattern: &str, dict: &Dictionary) 
    -> Vec<(String, Vec<String>)> {
    
    match mode {
        SearchMode::Exact => {
            // Direct O(1) lookup
            dict.code_to_char(pattern)
        }
        
        SearchMode::Fuzzy(length) => {
            // Filter by length first, then regex
            let regex = pattern_to_regex(pattern); // "a-b" → "a.b"
            dict.codes_with_length(length)
                .filter(|code| regex.is_match(code))
                .collect()
        }
        
        SearchMode::LLM(min_length) => {
            // Filter by length range, then regex
            let regex = pattern_to_regex(pattern); // "a-b-" → "a.*b.*"
            (min_length..=5)
                .flat_map(|len| dict.codes_with_length(len))
                .filter(|code| regex.is_match(code))
                .collect()
        }
    }
}
```

**Pattern to Regex:**
- In Fuzzy mode: `-` → `.` (exactly one char)
- In LLM mode: `-` → `.*` (zero or more chars)

---

### 3. Code Scorer (`src/scorer.rs`)

**Input:** Code string
**Output:** Likelihood score [0.0, 1.0]

**When Used:** Only in LLM mode when results > threshold (e.g., 50)

**Model Loading:**
```rust
// Load pre-trained model
use cj_llm::CodeScorer;
use candle_core::Device;

let device = Device::Cpu;
let scorer = CodeScorer::from_pretrained("data/model_weights.bin", &device)?;

// Score a code
let code_ids = vec![0, 1, 2]; // 'a', 'b', 'c'
let score = scorer.forward(&code_ids)?;  // → 0.87
```

**Character Encoding:**
```rust
// Map CangJie letters to indices
// a=0, b=1, ..., w=22, y=23, x=24
fn char_to_idx(c: char) -> usize {
    match c {
        'a'..='w' => (c as usize) - ('a' as usize),
        'y' => 23,
        'x' => 24,
        _ => panic!("Invalid CangJie char"),
    }
}
```

---

### 4. Rule Classifier (`src/rules.rs`)

**Input:** (character, code) pair
**Output:** List of applicable rule types

**Rule Types:**
- `SingleUnit` - Single component character (1 letter code)
- `CompoundChar` - Multi-component with 'x' separator
- `MaximumCoverage` - Uses maximum radical coverage
- `Delimitation` - Delimitation between components
- `NoCrossing` - Components don't cross boundaries
- `General` - Uncategorized

**Classification Logic:**
```rust
fn classify(char: &str, code: &str) -> Vec<RuleType> {
    let mut rules = Vec::new();
    
    // SingleUnit: 1-letter codes
    if code.len() == 1 {
        rules.push(RuleType::SingleUnit);
    }
    
    // CompoundChar: contains 'x'
    if code.contains('x') {
        rules.push(RuleType::CompoundChar);
    }
    
    // Use cj-rules examples for validation
    if let Some(example) = rules_engine.find_example(char, code) {
        rules.push(example.rule_type);
    }
    
    if rules.is_empty() {
        rules.push(RuleType::General);
    }
    
    rules
}
```

---

## 📂 Project Structure

```
cj-llm/
├── Cargo.toml                  # Dependencies and config
├── README.md                   # This file
├── IMPLEMENTATION.md           # Detailed implementation checklist
├── src/
│   ├── lib.rs                  # Main library API exports
│   ├── model.rs                # CodeScorer + ScoreFusion (neural ranking)
│   ├── search.rs               # CJSearch orchestrator
│   ├── matcher.rs              # Dictionary filtering
│   ├── pattern.rs              # Pattern parsing and mode detection
│   ├── rules.rs                # Rule classification logic
│   ├── types.rs                # Shared types (SearchResult, etc.)
│   ├── data.rs                 # Data loader
│   ├── vocab.rs                # Vocabulary management
│   └── bin/
│       ├── train.rs            # Binary: Train neural network
│       └── search.rs           # Binary: CLI search tool
├── data/
│   └── model.safetensors       # Trained model weights (generated)
├── tests/
│   ├── integration_test.rs     # End-to-end tests
│   └── pattern_test.rs         # Pattern matching tests
└── benches/
    └── search_bench.rs         # Performance benchmarks
```

---

## 🚀 Usage

### Quick Start

```bash
# Build the project
cargo build --release

# Search using the CLI
./target/release/search "a" --limit 10
./target/release/search "?a-b" --rules
./target/release/search "??a-b-" --scores
```

### Build and Compile

```bash
# Development build
cargo build

# Optimized release build
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo run --release --bin search -- --help
```

### CLI Tool Examples

#### Exact Match
```bash
$ ./search "a"
Mode: Exact
✅ Found 1 matches:

1. a              → 曰, 日
```

#### Fuzzy Match with Rules
```bash
$ ./search "?a-b" --limit 5 --rules
Mode: Fuzzy
✅ Found 5 matches:

1. awb            → 𣉌
      Rules: General

2. atb            → 𦡉
      Rules: General
```

#### LLM Mode with Scores
```bash
$ ./search "??a-b-" --limit 5 --scores
Mode: LLM (Ranking)
✅ Found 5 matches:

1. arbuu          → 𧢈
      Score: 91% [█████████░]

2. aobuu          → 𧡨
      Score: 91% [█████████░]
```

#### Verbose Mode
```bash
$ ./search -v "a" --limit 3
🔍 Loading CJSearch engine...
✅ Dictionary loaded: 53426 codes in 5 categories

🔎 Searching in Exact mode: a
─────────────────────────────────────────────────

✅ Found 1 matches:

1. a              → 曰, 日

─────────────────────────────────────────────────
✨ Search completed successfully!
```

### CLI Arguments

```
Usage: search [OPTIONS] <PATTERN>

Arguments:
  <PATTERN>  Search pattern (prefix rules apply)

Options:
  -l, --limit <LIMIT>  Maximum results to display [default: 10]
  -r, --rules          Show rule annotations for each result
  -s, --scores         Show LLM scores (LLM mode only)
  -v, --verbose        Show detailed information
  -h, --help           Print help
  -V, --version        Print version
```

---

## 🎯 Performance Goals

| Metric | Target | Notes |
|--------|--------|-------|
| Exact match | < 1ms | O(1) hash lookup |
| Fuzzy match | < 10ms | Filter by length + regex |
| LLM mode (small) | < 50ms | Includes neural network scoring |
| LLM mode (large) | < 200ms | 1000+ results need ranking |
| Training time | < 2 min | One-time setup |
| Model size | < 100KB | Embedded in binary |
| Binary size | < 5MB | With embedded dictionary + model |

---

## 📊 Data Sources

### CJ Dictionary (`cj-dictionary`)
- **Size:** 171,000 code→char mappings, 64,000 char→code mappings
- **Format:** Binary HashMap (bincode serialized)
- **Access:** O(1) lookups in both directions
- **Letter set:** a-w, y, x (25 letters, z excluded)

### CJ Rules (`cj-rules`)
- **Size:** 1,673 annotated (char, code) examples
- **Format:** MessagePack serialized
- **Usage:** Rule classification validation

---

## 🔬 Validation Strategy

### Unit Tests
- Pattern parser correctness (mode detection, length counting)
- Regex generation accuracy (fuzzy vs LLM patterns)
- Dictionary lookup consistency

### Integration Tests
- End-to-end search queries for all modes
- Score ranking correctness
- Rule annotation accuracy

### Benchmarks
- Search latency across pattern types
- Model inference speed
- Memory usage profiling

---

## 📝 Next Steps

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for detailed implementation checklist.

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 👤 Author

OM <om@Mova.Club>
