// CJ-LLM Search Engine
// Main API that orchestrates all components

use crate::matcher::DictionaryMatcher;
use crate::model::CodeScorer;
use crate::pattern::parse_query;
use crate::rules::RuleClassifier;
use crate::types::{SearchMode, SearchResult};

/// Main CJ-LLM search engine
///
/// Combines all components:
/// - Pattern parsing (Exact/Fuzzy/LLM modes)
/// - Dictionary matching (code lookup)
/// - Neural network scoring (LLM ranking)
/// - Rule classification (annotation)
pub struct CJSearch {
    /// Dictionary matcher for code lookups
    matcher: DictionaryMatcher,

    /// Code scorer for LLM mode ranking
    scorer: CodeScorer,

    /// Rule classifier for annotations
    classifier: RuleClassifier,
}

impl CJSearch {
    /// Create a new CJSearch engine
    ///
    /// # Returns
    /// CJSearch with all components loaded
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let matcher = DictionaryMatcher::new()?;
        let scorer = CodeScorer::new(candle_core::Device::Cpu)?;
        let classifier = RuleClassifier::new()?;

        Ok(Self {
            matcher,
            scorer,
            classifier,
        })
    }

    /// Execute a search query
    ///
    /// # Arguments
    /// * `query` - Search pattern (e.g., "abc", "?a-b", "??a-b-")
    ///
    /// # Returns
    /// Vec of SearchResult sorted by relevance
    pub fn search(&self, query: &str) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
        // Step 1: Parse the query
        let parsed = parse_query(query)?;

        // Step 2: Search dictionary
        let matches = self.matcher.search(parsed.mode, &parsed.pattern)?;

        // Step 3: Score results (for LLM mode)
        let mut results = Vec::new();

        for (code, characters) in matches {
            // Get first character for rule classification
            let first_char = characters.first().cloned().unwrap_or_default();

            // Classify rules
            let rules = self
                .classifier
                .classify(&first_char, &code)
                .ok()
                .unwrap_or_default();

            // Score the code (especially important for LLM mode)
            let score = if matches!(parsed.mode, SearchMode::LLM(_)) {
                let code_ids: Vec<u32> = code.chars().map(|c| (c as u32) - ('a' as u32)).collect();
                Some(self.scorer.forward(&code_ids)?)
            } else {
                None
            };

            let result = SearchResult {
                code,
                characters,
                rules,
                score,
            };

            results.push(result);
        }

        // Step 4: Sort by relevance
        match parsed.mode {
            SearchMode::LLM(_) => {
                // Sort by score descending (higher is better)
                results.sort_by(|a, b| {
                    let score_a = a.score.unwrap_or(0.0);
                    let score_b = b.score.unwrap_or(0.0);
                    score_b
                        .partial_cmp(&score_a)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            _ => {
                // For Exact and Fuzzy, keep original order
            }
        }

        Ok(results)
    }

    /// Search with limit on results
    ///
    /// # Arguments
    /// * `query` - Search pattern
    /// * `limit` - Maximum number of results to return
    pub fn search_limit(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
        let mut results = self.search(query)?;
        results.truncate(limit);
        Ok(results)
    }

    /// Get statistics about the current dictionary
    pub fn stats(&self) -> Result<(usize, usize), Box<dyn std::error::Error>> {
        let counts = self.matcher.codes_count_by_length();
        let total: usize = counts.values().sum();
        Ok((total, counts.len()))
    }
}

impl Default for CJSearch {
    fn default() -> Self {
        Self::new().expect("Failed to create default CJSearch")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_search() -> Result<CJSearch, Box<dyn std::error::Error>> {
        CJSearch::new()
    }

    #[test]
    fn test_search_creation() -> Result<(), Box<dyn std::error::Error>> {
        let _search = create_search()?;
        Ok(())
    }

    #[test]
    fn test_exact_search() -> Result<(), Box<dyn std::error::Error>> {
        let search = create_search()?;
        let results = search.search("a")?;

        assert!(!results.is_empty());
        assert_eq!(results[0].code, "a");
        assert!(results[0].score.is_none()); // Exact mode has no score
        Ok(())
    }

    #[test]
    fn test_fuzzy_search() -> Result<(), Box<dyn std::error::Error>> {
        let search = create_search()?;
        let results = search.search("?a-b")?;

        assert!(!results.is_empty());
        for result in results {
            assert_eq!(result.code.len(), 3);
            assert!(result.score.is_none()); // Fuzzy mode has no score
        }
        Ok(())
    }

    #[test]
    fn test_llm_search() -> Result<(), Box<dyn std::error::Error>> {
        let search = create_search()?;
        let results = search.search("??a-b-")?;

        assert!(!results.is_empty());

        // LLM mode should have scores
        for result in &results {
            assert!(result.score.is_some());
            let score = result.score.unwrap();
            assert!(score >= 0.0 && score <= 1.0);
        }

        // Results should be sorted by score (descending)
        for i in 0..results.len() - 1 {
            let score_curr = results[i].score.unwrap_or(0.0);
            let score_next = results[i + 1].score.unwrap_or(0.0);
            assert!(score_curr >= score_next);
        }
        Ok(())
    }

    #[test]
    fn test_search_with_limit() -> Result<(), Box<dyn std::error::Error>> {
        let search = create_search()?;
        let results = search.search_limit("??a-", 5)?;

        assert!(results.len() <= 5);
        Ok(())
    }

    #[test]
    fn test_search_results_have_rules() -> Result<(), Box<dyn std::error::Error>> {
        let search = create_search()?;
        let results = search.search("a")?;

        assert!(!results.is_empty());
        for result in results {
            assert!(!result.rules.is_empty()); // Should have at least General rule
        }
        Ok(())
    }

    #[test]
    fn test_invalid_query() -> Result<(), Box<dyn std::error::Error>> {
        let search = create_search()?;
        let result = search.search("??z-");

        // Should error (z is invalid in CJ)
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_stats() -> Result<(), Box<dyn std::error::Error>> {
        let search = create_search()?;
        let (total, _category_count) = search.stats()?;

        assert!(total > 0);
        Ok(())
    }
}
