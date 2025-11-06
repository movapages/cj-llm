// CJ-LLM Dictionary Matcher
// Filters dictionary codes based on pattern and search mode

use crate::data::DataLoader;
use crate::types::SearchMode;
use regex::Regex;
use rustc_hash::FxHashMap;

/// Dictionary matcher with pre-indexed codes by length
#[derive(Clone)]
pub struct DictionaryMatcher {
    /// Code to characters mapping: code → [char1, char2, ...]
    c2h: FxHashMap<String, Vec<String>>,

    /// Codes indexed by length: length → [code1, code2, ...]
    /// CJ codes range from 1-5 letters, so we have indices 1, 2, 3, 4, 5
    codes_by_length: FxHashMap<usize, Vec<String>>,
}

impl DictionaryMatcher {
    /// Create a new DictionaryMatcher, loading and indexing the dictionary
    ///
    /// This precomputes all codes organized by length for fast filtering.
    ///
    /// # Returns
    /// Result containing the matcher or error if dictionary fails to load
    ///
    /// # Example
    /// ```
    /// # use cj_llm::matcher::DictionaryMatcher;
    /// let matcher = DictionaryMatcher::new().unwrap();
    /// ```
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Load C2H (code to characters) mapping from embedded binary data
        let c2h_bytes = DataLoader::c2h_data();
        let c2h: FxHashMap<String, Vec<String>> = bincode::deserialize(c2h_bytes)
            .map_err(|e| format!("Failed to deserialize dictionary: {}", e))?;

        // Pre-index codes by length
        let mut codes_by_length: FxHashMap<usize, Vec<String>> = FxHashMap::default();

        // Initialize empty vectors for each length
        for len in 1..=5 {
            codes_by_length.insert(len, Vec::new());
        }

        // Get all codes and index by length
        for code in c2h.keys() {
            let len = code.len();
            if len >= 1 && len <= 5 {
                if let Some(codes_vec) = codes_by_length.get_mut(&len) {
                    codes_vec.push(code.clone());
                }
            }
        }

        Ok(Self {
            c2h,
            codes_by_length,
        })
    }

    /// Search for codes matching the pattern in the given mode
    ///
    /// # Arguments
    /// * `mode` - The search mode (Exact, Fuzzy, or LLM)
    /// * `pattern` - The pattern string (without prefix)
    ///
    /// # Returns
    /// Vector of (code, characters) tuples
    ///
    /// # Example
    /// ```
    /// # use cj_llm::matcher::DictionaryMatcher;
    /// # use cj_llm::types::SearchMode;
    /// # let matcher = DictionaryMatcher::new().unwrap();
    /// // Search for exact code "a"
    /// let results = matcher.search(SearchMode::Exact, "a").unwrap();
    /// ```
    pub fn search(
        &self,
        mode: SearchMode,
        pattern: &str,
    ) -> Result<Vec<(String, Vec<String>)>, Box<dyn std::error::Error>> {
        match mode {
            SearchMode::Exact => self.search_exact(pattern),
            SearchMode::Fuzzy(len) => self.search_fuzzy(len, pattern),
            SearchMode::LLM(min_len) => self.search_llm(min_len, pattern),
        }
    }

    /// Exact match: Direct O(1) dictionary lookup
    fn search_exact(
        &self,
        code: &str,
    ) -> Result<Vec<(String, Vec<String>)>, Box<dyn std::error::Error>> {
        if let Some(chars) = self.c2h.get(code) {
            Ok(vec![(code.to_string(), chars.clone())])
        } else {
            Ok(vec![]) // Code not found
        }
    }

    /// Fuzzy match: Fixed-length pattern with regex filtering
    fn search_fuzzy(
        &self,
        fixed_len: usize,
        pattern: &str,
    ) -> Result<Vec<(String, Vec<String>)>, Box<dyn std::error::Error>> {
        // Convert pattern to regex
        let regex_str = crate::pattern::pattern_to_regex(pattern, SearchMode::Fuzzy(fixed_len))?;
        let re = Regex::new(&regex_str)?;

        // Get all codes of the target length
        let codes = self
            .codes_by_length
            .get(&fixed_len)
            .ok_or("No codes of this length")?;

        // Filter codes that match the regex pattern
        let mut results = Vec::new();
        for code in codes {
            if re.is_match(code) {
                if let Some(chars) = self.c2h.get(code) {
                    results.push((code.clone(), chars.clone()));
                }
            }
        }

        Ok(results)
    }

    /// LLM match: Variable-length with all matching codes
    fn search_llm(
        &self,
        min_len: usize,
        pattern: &str,
    ) -> Result<Vec<(String, Vec<String>)>, Box<dyn std::error::Error>> {
        // Convert pattern to regex
        let regex_str = crate::pattern::pattern_to_regex(pattern, SearchMode::LLM(min_len))?;
        let re = Regex::new(&regex_str)?;

        let mut results = Vec::new();

        // Search through all lengths from min_len to 5
        for len in min_len..=5 {
            if let Some(codes) = self.codes_by_length.get(&len) {
                for code in codes {
                    if re.is_match(code) {
                        if let Some(chars) = self.c2h.get(code) {
                            results.push((code.clone(), chars.clone()));
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    /// Get all codes of a specific length
    ///
    /// # Arguments
    /// * `length` - The code length (1-5)
    ///
    /// # Returns
    /// Slice of codes with that length
    pub fn codes_with_length(&self, length: usize) -> Option<&Vec<String>> {
        self.codes_by_length.get(&length)
    }

    /// Get the total count of codes across all lengths
    pub fn total_codes(&self) -> usize {
        self.codes_by_length.values().map(|v| v.len()).sum()
    }

    /// Get counts of codes by length
    ///
    /// # Returns
    /// FxHashMap with length → count
    pub fn codes_count_by_length(&self) -> FxHashMap<usize, usize> {
        self.codes_by_length
            .iter()
            .map(|(len, codes)| (*len, codes.len()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matcher_creation() {
        let matcher = DictionaryMatcher::new().unwrap();
        assert!(matcher.total_codes() > 0);
    }

    #[test]
    fn test_codes_indexed() {
        let matcher = DictionaryMatcher::new().unwrap();

        // Should have codes for each length 1-5
        for len in 1..=5 {
            assert!(matcher.codes_with_length(len).is_some());
            let codes = matcher.codes_with_length(len).unwrap();
            assert!(!codes.is_empty(), "Should have codes of length {}", len);
        }
    }

    #[test]
    fn test_exact_search() {
        let matcher = DictionaryMatcher::new().unwrap();

        // "a" should be a valid single-letter code
        let results = matcher.search(SearchMode::Exact, "a").unwrap();
        assert!(!results.is_empty(), "Code 'a' should exist");
        assert_eq!(results[0].0, "a");
    }

    #[test]
    fn test_exact_search_not_found() {
        let matcher = DictionaryMatcher::new().unwrap();

        // "zzz" should not be a valid code (z is excluded from CJ)
        let results = matcher.search(SearchMode::Exact, "zzz").unwrap();
        assert!(results.is_empty(), "Invalid code should return empty");
    }

    #[test]
    fn test_fuzzy_search_single_length() {
        let matcher = DictionaryMatcher::new().unwrap();

        // Search for 3-letter codes matching "a-b"
        let results = matcher.search(SearchMode::Fuzzy(3), "a-b").unwrap();

        // Should have results
        assert!(!results.is_empty(), "Should find codes matching a-b");

        // All results should be 3 letters
        for (code, _) in results {
            assert_eq!(code.len(), 3, "Code '{}' should be 3 letters", code);
        }
    }

    #[test]
    fn test_llm_search_variable_length() {
        let matcher = DictionaryMatcher::new().unwrap();

        // Search for codes starting with 'a' (LLM mode with min_len=1)
        let results = matcher.search(SearchMode::LLM(1), "a").unwrap();

        // Should have many results (a is at the start)
        assert!(!results.is_empty(), "Should find codes starting with 'a'");

        // All results should start with 'a'
        for (code, _) in results {
            assert!(
                code.starts_with('a'),
                "Code '{}' should start with 'a'",
                code
            );
        }
    }

    #[test]
    fn test_codes_count() {
        let matcher = DictionaryMatcher::new().unwrap();
        let counts = matcher.codes_count_by_length();

        // Should have counts for each length
        assert_eq!(counts.len(), 5, "Should have 5 length categories");

        for len in 1..=5 {
            assert!(
                counts.contains_key(&len),
                "Should have count for length {}",
                len
            );
            assert!(
                *counts.get(&len).unwrap() > 0,
                "Should have codes of length {}",
                len
            );
        }
    }
}
