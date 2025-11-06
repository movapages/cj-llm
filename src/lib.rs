//! # CJ-LLM: CangJie Pattern Search Engine
//!
//! A high-performance reverse lookup tool for CangJie codes with wildcard matching and intelligent neural ranking.
//!
//! ## Three Search Modes
//!
//! 1. **Exact** - Direct dictionary lookup (O(1), < 1ms)
//!    - `search("abc")` - direct lookup
//! 2. **Fuzzy** - Fixed-length pattern matching (< 10ms)
//!    - `search("?a-b")` - 4-letter codes matching pattern
//! 3. **LLM** - Variable-length with neural ranking (< 200ms)
//!    - `search("??a-b-")` - codes of any length, ranked by score
//!
//! ## Pattern Syntax
//!
//! - No prefix: Exact match (must be 1-5 characters)
//! - `?` prefix: Fuzzy match (pattern must be exactly 1-5 chars)
//! - `??` prefix: LLM mode (pattern 1-5+ chars, flexible length)
//! - `-` (dash): Wildcard (one char in Fuzzy, zero+ in LLM)
//!
//! ## Example Usage
//!
//! ```ignore
//! use cj_llm::CJSearch;
//!
//! let search = CJSearch::new()?;
//!
//! // Exact search
//! let results = search.search("a")?;
//!
//! // Fuzzy search
//! let results = search.search("?a-b")?;
//!
//! // LLM mode with ranking
//! let results = search.search_limit("??a-b-", 10)?;
//!
//! // Statistics
//! let (total_codes, categories) = search.stats()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Architecture
//!
//! - **Pattern Parser** - Converts user queries to regex patterns
//! - **Dictionary Matcher** - Filters codes using regex
//! - **Code Scorer** - Ranks results using neural network (LLM mode)
//! - **Rule Classifier** - Annotates results with CangJie rules
//! - **CJSearch API** - Main entry point combining all components

pub mod data;
pub mod matcher;
pub mod model;
pub mod pattern;
pub mod rules;
pub mod search;
pub mod types;
pub mod vocab;

// Re-export main types and functions for convenience
pub use data::{DataInfo, DataLoader};
pub use matcher::DictionaryMatcher;
pub use model::{CodeScorer, ScoreFusion};
pub use pattern::{parse_query, pattern_to_regex};
pub use rules::RuleClassifier;
pub use search::CJSearch;
pub use types::{ParsedQuery, PatternError, RuleType, SearchMode, SearchResult};
pub use vocab::Vocab;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
