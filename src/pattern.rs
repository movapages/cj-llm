// CJ-LLM Pattern Parser
// Parses user queries and converts them to searchable patterns

use crate::types::{ParsedQuery, PatternError, SearchMode};
use regex::Regex;

/// Valid CangJie letters (a-w, y, x; z is excluded)
const VALID_CJ_LETTERS: &[char] = &[
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
    't', 'u', 'v', 'w', 'y', 'x',
];

/// Parse a user query and detect the search mode
///
/// # Pattern Syntax
/// - No prefix: Exact mode → "abc"
/// - `?` prefix: Fuzzy mode → "?a-b" (fixed length = letters + dashes)
/// - `??` prefix: LLM mode → "??a-b-" (min length = letters only)
///
/// # Examples
/// ```
/// # use cj_llm::pattern::parse_query;
/// let parsed = parse_query("abc").unwrap();
/// assert_eq!(parsed.pattern, "abc");
///
/// let parsed = parse_query("?a-b").unwrap();
/// assert_eq!(parsed.pattern, "a-b");
///
/// let parsed = parse_query("??a-b-").unwrap();
/// assert_eq!(parsed.pattern, "a-b-");
/// ```
pub fn parse_query(query: &str) -> Result<ParsedQuery, PatternError> {
    let query = query.trim();

    if query.is_empty() {
        return Err(PatternError::EmptyPattern);
    }

    // Detect mode and strip prefix
    let (mode, pattern) = if query.starts_with("??") {
        let pattern = &query[2..];
        let min_len = count_letters(pattern);
        validate_length(min_len)?;
        (SearchMode::LLM(min_len), pattern)
    } else if query.starts_with("?") {
        let pattern = &query[1..];
        let fixed_len = count_letters(pattern) + count_dashes(pattern);
        validate_length(fixed_len)?;
        (SearchMode::Fuzzy(fixed_len), pattern)
    } else {
        let fixed_len = query.len();
        validate_length(fixed_len)?;
        (SearchMode::Exact, query)
    };

    // Validate pattern characters
    validate_pattern(pattern)?;

    Ok(ParsedQuery {
        mode,
        pattern: pattern.to_string(),
    })
}

/// Convert a pattern to a regex string for matching
///
/// # Arguments
/// * `pattern` - The pattern string (e.g., "a-b-c")
/// * `mode` - The search mode (Exact, Fuzzy, or LLM)
///
/// # Conversion Rules
/// - Fuzzy: Each `-` becomes `.` (exactly one character)
/// - LLM: Each `-` becomes `.*` (zero or more characters)
/// - Literal letters: Escaped and matched exactly
///
/// # Examples
/// ```
/// # use cj_llm::pattern::{pattern_to_regex, parse_query};
/// # use cj_llm::types::SearchMode;
/// let regex = pattern_to_regex("a-b", SearchMode::Fuzzy(3)).unwrap();
/// assert_eq!(regex, "^a.b$");
///
/// let regex = pattern_to_regex("a-b-", SearchMode::LLM(2)).unwrap();
/// assert_eq!(regex, "^a.*b.*$");
/// ```
pub fn pattern_to_regex(pattern: &str, mode: SearchMode) -> Result<String, PatternError> {
    let mut regex = String::from("^");

    let dash_replacement = match mode {
        SearchMode::Fuzzy(_) => ".",
        SearchMode::LLM(_) => ".*",
        SearchMode::Exact => "", // No wildcards in exact mode
    };

    let chars: Vec<char> = pattern.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];

        if ch == '-' {
            if mode == SearchMode::Exact {
                // Exact mode shouldn't have dashes
                return Err(PatternError::InvalidSymbol { char: ch });
            }
            regex.push_str(dash_replacement);
        } else if is_valid_cj_letter(ch) {
            regex.push(ch);
        } else {
            return Err(PatternError::InvalidCharacter { char: ch });
        }

        i += 1;
    }

    regex.push('$');

    // Validate regex compiles
    Regex::new(&regex).map_err(|e| PatternError::RegexError(e.to_string()))?;

    Ok(regex)
}

/// Count the number of letters in a pattern
#[inline]
fn count_letters(pattern: &str) -> usize {
    pattern.chars().filter(|c| is_valid_cj_letter(*c)).count()
}

/// Count the number of dashes in a pattern
#[inline]
fn count_dashes(pattern: &str) -> usize {
    pattern.chars().filter(|c| *c == '-').count()
}

/// Check if a character is a valid CangJie letter
#[inline]
fn is_valid_cj_letter(ch: char) -> bool {
    VALID_CJ_LETTERS.contains(&ch)
}

/// Validate pattern has only valid characters
fn validate_pattern(pattern: &str) -> Result<(), PatternError> {
    for ch in pattern.chars() {
        match ch {
            '-' => continue,
            c if is_valid_cj_letter(c) => continue,
            c => return Err(PatternError::InvalidCharacter { char: c }),
        }
    }
    Ok(())
}

/// Validate length is within bounds (1-5)
#[inline]
fn validate_length(len: usize) -> Result<(), PatternError> {
    match len {
        0 => Err(PatternError::LengthTooShort { actual: 0 }),
        1..=5 => Ok(()),
        n => Err(PatternError::LengthTooLong { actual: n }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============ Mode Detection Tests ============

    #[test]
    fn test_exact_mode_detection() {
        let parsed = parse_query("abc").unwrap();
        assert_eq!(parsed.mode, SearchMode::Exact);
        assert_eq!(parsed.pattern, "abc");
    }

    #[test]
    fn test_fuzzy_mode_detection() {
        let parsed = parse_query("?a-b").unwrap();
        assert!(matches!(parsed.mode, SearchMode::Fuzzy(3)));
        assert_eq!(parsed.pattern, "a-b");
    }

    #[test]
    fn test_llm_mode_detection() {
        let parsed = parse_query("??a-b-").unwrap();
        assert!(matches!(parsed.mode, SearchMode::LLM(2)));
        assert_eq!(parsed.pattern, "a-b-");
    }

    // ============ Length Counting Tests ============

    #[test]
    fn test_fuzzy_length_counting() {
        // 3 letters + 2 dashes = 5 positions
        let parsed = parse_query("?a-b-c").unwrap();
        assert!(matches!(parsed.mode, SearchMode::Fuzzy(5)));
    }

    #[test]
    fn test_llm_length_counting() {
        // Only count letters (a, b, c) = 3 letters
        let parsed = parse_query("??a-b-c").unwrap();
        assert!(matches!(parsed.mode, SearchMode::LLM(3)));
    }

    #[test]
    fn test_single_letter_exact() {
        let parsed = parse_query("a").unwrap();
        assert_eq!(parsed.mode, SearchMode::Exact);
    }

    #[test]
    fn test_single_letter_fuzzy() {
        let parsed = parse_query("?a").unwrap();
        assert!(matches!(parsed.mode, SearchMode::Fuzzy(1)));
    }

    #[test]
    fn test_max_length_exact() {
        let parsed = parse_query("abcde").unwrap();
        assert_eq!(parsed.mode, SearchMode::Exact);
    }

    #[test]
    fn test_max_length_fuzzy() {
        let parsed = parse_query("?a-b-c").unwrap();
        assert!(matches!(parsed.mode, SearchMode::Fuzzy(5)));
    }

    // ============ Regex Generation Tests ============

    #[test]
    fn test_regex_exact_simple() {
        let regex = pattern_to_regex("abc", SearchMode::Exact).unwrap();
        assert_eq!(regex, "^abc$");
    }

    #[test]
    fn test_regex_fuzzy_simple() {
        let regex = pattern_to_regex("a-b", SearchMode::Fuzzy(3)).unwrap();
        assert_eq!(regex, "^a.b$");
    }

    #[test]
    fn test_regex_fuzzy_multiple_dashes() {
        let regex = pattern_to_regex("a-b-c", SearchMode::Fuzzy(5)).unwrap();
        assert_eq!(regex, "^a.b.c$");
    }

    #[test]
    fn test_regex_llm_simple() {
        let regex = pattern_to_regex("a-b-", SearchMode::LLM(2)).unwrap();
        assert_eq!(regex, "^a.*b.*$");
    }

    #[test]
    fn test_regex_llm_complex() {
        let regex = pattern_to_regex("a-b-c-", SearchMode::LLM(3)).unwrap();
        assert_eq!(regex, "^a.*b.*c.*$");
    }

    #[test]
    fn test_regex_single_letter() {
        let regex = pattern_to_regex("a", SearchMode::LLM(1)).unwrap();
        assert_eq!(regex, "^a$");
    }

    #[test]
    fn test_regex_compiles() {
        let regex_str = pattern_to_regex("a-b-c", SearchMode::Fuzzy(5)).unwrap();
        Regex::new(&regex_str).unwrap(); // Should not panic
    }

    // ============ Validation Error Tests ============

    #[test]
    fn test_empty_pattern_error() {
        let result = parse_query("");
        assert!(matches!(result, Err(PatternError::EmptyPattern)));
    }

    #[test]
    fn test_whitespace_pattern_error() {
        let result = parse_query("   ");
        assert!(matches!(result, Err(PatternError::EmptyPattern)));
    }

    #[test]
    fn test_too_long_exact() {
        let result = parse_query("abcdef");
        assert!(matches!(
            result,
            Err(PatternError::LengthTooLong { actual: 6 })
        ));
    }

    #[test]
    fn test_too_long_fuzzy() {
        let result = parse_query("?a-b-c-d");
        // 4 letters + 3 dashes = 7 > 5
        assert!(matches!(
            result,
            Err(PatternError::LengthTooLong { actual: 7 })
        ));
    }

    #[test]
    fn test_invalid_character_z() {
        let result = parse_query("?a-z");
        assert!(matches!(
            result,
            Err(PatternError::InvalidCharacter { char: 'z' })
        ));
    }

    #[test]
    fn test_invalid_character_number() {
        let result = parse_query("a1b");
        assert!(matches!(
            result,
            Err(PatternError::InvalidCharacter { char: '1' })
        ));
    }

    #[test]
    fn test_invalid_character_space() {
        let result = parse_query("a b");
        assert!(matches!(
            result,
            Err(PatternError::InvalidCharacter { char: ' ' })
        ));
    }

    // ============ Edge Cases ============

    #[test]
    fn test_fuzzy_all_dashes() {
        let result = parse_query("?-----");
        // 0 letters + 5 dashes = 5 length - valid
        assert!(result.is_ok());
    }

    #[test]
    fn test_llm_single_dash() {
        let result = parse_query("??-");
        // 0 letters = LLM(0) - invalid
        assert!(matches!(
            result,
            Err(PatternError::LengthTooShort { actual: 0 })
        ));
    }

    #[test]
    fn test_pattern_with_y_and_x() {
        let result = parse_query("?y-x");
        // y, x are valid CJ letters
        assert!(result.is_ok());
        assert_eq!(result.unwrap().pattern, "y-x");
    }

    #[test]
    fn test_case_sensitivity() {
        let result = parse_query("A");
        // 'A' is not in valid letters (only lowercase)
        assert!(matches!(
            result,
            Err(PatternError::InvalidCharacter { char: 'A' })
        ));
    }
}
