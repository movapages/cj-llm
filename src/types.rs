// CJ-LLM Type Definitions
// Core types for pattern parsing and search results

use thiserror::Error;

/// Search modes for CangJie code lookup
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    /// Exact match: No prefix, direct dictionary lookup (O(1))
    /// Example: "abc" → search for exact code "abc"
    Exact,

    /// Fuzzy match: `?` prefix, fixed-length pattern
    /// Example: "?a-b-c" → search only 5-letter codes matching pattern
    /// Length includes letters + dashes
    Fuzzy(usize),

    /// LLM mode: `??` prefix, variable-length with ranking
    /// Example: "??a-b-" → search 2-5 letter codes, rank by likelihood
    /// Length = count of literal letters only (dashes are wildcards)
    LLM(usize),
}

impl std::fmt::Display for SearchMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchMode::Exact => write!(f, "Exact"),
            SearchMode::Fuzzy(len) => write!(f, "Fuzzy({})", len),
            SearchMode::LLM(len) => write!(f, "LLM({})", len),
        }
    }
}

/// Rule type classifications for CangJie codes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuleType {
    /// Single component character (1-letter code)
    SingleUnit,
    /// Multi-component with 'x' separator
    CompoundChar,
    /// Uses maximum radical coverage
    MaximumCoverage,
    /// Delimitation between components
    Delimitation,
    /// Components don't cross boundaries
    NoCrossing,
    /// Uncategorized
    General,
}

impl std::fmt::Display for RuleType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RuleType::SingleUnit => write!(f, "SingleUnit"),
            RuleType::CompoundChar => write!(f, "CompoundChar"),
            RuleType::MaximumCoverage => write!(f, "MaximumCoverage"),
            RuleType::Delimitation => write!(f, "Delimitation"),
            RuleType::NoCrossing => write!(f, "NoCrossing"),
            RuleType::General => write!(f, "General"),
        }
    }
}

/// Search result containing code, characters, and metadata
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The CangJie code (e.g., "abc")
    pub code: String,

    /// Characters that map to this code
    pub characters: Vec<String>,

    /// Applicable rule types
    pub rules: Vec<RuleType>,

    /// Likelihood score [0.0, 1.0] (only set in LLM mode)
    pub score: Option<f32>,
}

impl SearchResult {
    /// Create a new search result
    pub fn new(code: String, characters: Vec<String>) -> Self {
        Self {
            code,
            characters,
            rules: vec![RuleType::General],
            score: None,
        }
    }

    /// Set rule types
    pub fn with_rules(mut self, rules: Vec<RuleType>) -> Self {
        self.rules = rules;
        self
    }

    /// Set likelihood score
    pub fn with_score(mut self, score: f32) -> Self {
        self.score = Some(score);
        self
    }
}

/// Pattern parsing errors
#[derive(Debug, Clone, Error)]
pub enum PatternError {
    #[error("Invalid pattern: empty")]
    EmptyPattern,

    #[error("Invalid pattern: code length {actual} exceeds maximum of 5")]
    LengthTooLong { actual: usize },

    #[error("Invalid pattern: code length {actual} is less than minimum of 1")]
    LengthTooShort { actual: usize },

    #[error("Invalid character '{char}' in pattern: only a-w, y, x allowed")]
    InvalidCharacter { char: char },

    #[error("Invalid pattern: only dashes and letters allowed, got '{char}'")]
    InvalidSymbol { char: char },

    #[error("Regex compilation failed: {0}")]
    RegexError(String),

    #[error("Invalid prefix: use no prefix, '?', or '??' at the start")]
    InvalidPrefix,
}

/// Parsed query result
#[derive(Debug, Clone)]
pub struct ParsedQuery {
    /// Detected search mode
    pub mode: SearchMode,

    /// The pattern string (without prefix)
    pub pattern: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_mode_display() {
        assert_eq!(SearchMode::Exact.to_string(), "Exact");
        assert_eq!(SearchMode::Fuzzy(3).to_string(), "Fuzzy(3)");
        assert_eq!(SearchMode::LLM(2).to_string(), "LLM(2)");
    }

    #[test]
    fn test_rule_type_display() {
        assert_eq!(RuleType::SingleUnit.to_string(), "SingleUnit");
        assert_eq!(RuleType::CompoundChar.to_string(), "CompoundChar");
    }

    #[test]
    fn test_search_result_builder() {
        let result = SearchResult::new("abc".to_string(), vec!["字".to_string()])
            .with_rules(vec![RuleType::SingleUnit])
            .with_score(0.95);

        assert_eq!(result.code, "abc");
        assert_eq!(result.characters, vec!["字"]);
        assert_eq!(result.score, Some(0.95));
    }
}
