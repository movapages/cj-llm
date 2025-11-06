// CJ-LLM Rule Classifier
// Annotates codes with CangJie rule types

use crate::data::DataLoader;
use crate::types::RuleType;

/// Rule classifier for annotating CangJie codes
///
/// Determines which rule types apply to a given (character, code) pair
pub struct RuleClassifier {
    /// Loaded examples: (code, character) pairs
    examples: Vec<(String, String)>,
}

impl RuleClassifier {
    /// Create a new RuleClassifier
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Load rules from embedded binary data
        let rules_bytes = DataLoader::rules_data();
        let examples: Vec<(String, String)> = rmp_serde::from_slice(rules_bytes)?;

        Ok(Self { examples })
    }

    /// Classify a code with applicable rule types
    ///
    /// # Arguments
    /// * `character` - The Unicode character (e.g., "字")
    /// * `code` - The CangJie code (e.g., "pnbc")
    ///
    /// # Returns
    /// Vec of applicable RuleType enums
    pub fn classify(
        &self,
        character: &str,
        code: &str,
    ) -> Result<Vec<RuleType>, Box<dyn std::error::Error>> {
        let mut rules = Vec::new();

        // Rule 1: SingleUnit - 1-letter code
        if code.len() == 1 {
            rules.push(RuleType::SingleUnit);
        }

        // Rule 2: CompoundChar - contains 'x' separator
        if code.contains('x') {
            rules.push(RuleType::CompoundChar);
        }

        // Rule 3: Check examples for matching (code, character) pairs
        for (ex_code, ex_char) in &self.examples {
            if ex_char == character && ex_code == code {
                // Found matching example - it's a known rule
                rules.push(RuleType::General);
                break;
            }
        }

        // Default: General rule if nothing else matched
        if rules.is_empty() {
            rules.push(RuleType::General);
        }

        Ok(rules)
    }

    /// Classify multiple (character, code) pairs in batch
    pub fn classify_batch(
        &self,
        pairs: &[(String, String)],
    ) -> Result<Vec<Vec<RuleType>>, Box<dyn std::error::Error>> {
        let mut results = Vec::new();
        for (character, code) in pairs {
            let rules = self.classify(character, code)?;
            results.push(rules);
        }
        Ok(results)
    }

    /// Get heuristic rule type based only on code structure
    pub fn classify_by_structure(code: &str) -> RuleType {
        match code.len() {
            1 => RuleType::SingleUnit,
            _ if code.contains('x') => RuleType::CompoundChar,
            _ => RuleType::General,
        }
    }

    /// Check if a code matches a specific rule type pattern
    pub fn matches_rule(code: &str, rule: RuleType) -> bool {
        match rule {
            RuleType::SingleUnit => code.len() == 1,
            RuleType::CompoundChar => code.contains('x'),
            _ => false,
        }
    }
}

impl Default for RuleClassifier {
    fn default() -> Self {
        Self::new().expect("Failed to create default RuleClassifier")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classifier_creation() {
        let _classifier = RuleClassifier::new().unwrap();
    }

    #[test]
    fn test_single_unit_detection() {
        let classifier = RuleClassifier::new().unwrap();
        let rules = classifier.classify("日", "a").unwrap();
        assert!(rules.contains(&RuleType::SingleUnit));
    }

    #[test]
    fn test_compound_char_detection() {
        let classifier = RuleClassifier::new().unwrap();
        let rules = classifier.classify("字", "pxnb").unwrap();
        assert!(rules.contains(&RuleType::CompoundChar));
    }

    #[test]
    fn test_classify_by_structure() {
        assert_eq!(
            RuleClassifier::classify_by_structure("a"),
            RuleType::SingleUnit
        );
        assert_eq!(
            RuleClassifier::classify_by_structure("pxnb"),
            RuleType::CompoundChar
        );
        assert_eq!(
            RuleClassifier::classify_by_structure("pnbc"),
            RuleType::General
        );
    }

    #[test]
    fn test_matches_rule() {
        assert!(RuleClassifier::matches_rule("a", RuleType::SingleUnit));
        assert!(!RuleClassifier::matches_rule("ab", RuleType::SingleUnit));
        assert!(RuleClassifier::matches_rule("axb", RuleType::CompoundChar));
        assert!(!RuleClassifier::matches_rule("abc", RuleType::CompoundChar));
    }

    #[test]
    fn test_batch_classify() {
        let classifier = RuleClassifier::new().unwrap();
        let pairs = vec![
            ("日".to_string(), "a".to_string()),
            ("字".to_string(), "pxnb".to_string()),
            ("明".to_string(), "pnbc".to_string()),
        ];

        let results = classifier.classify_batch(&pairs).unwrap();
        assert_eq!(results.len(), 3);
        assert!(results[0].contains(&RuleType::SingleUnit));
        assert!(results[1].contains(&RuleType::CompoundChar));
    }

    #[test]
    fn test_default() {
        let classifier = RuleClassifier::default();
        let rules = classifier.classify("🔥", "xyz").unwrap();
        assert!(!rules.is_empty());
    }
}
