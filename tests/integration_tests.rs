// Integration tests for Phase 1: Pattern Parser

use cj_llm::{parse_query, pattern_to_regex, PatternError, SearchMode};

// ============ Mode Detection Tests ============

#[test]
fn test_exact_mode_workflow() {
    let parsed = parse_query("abc").unwrap();
    assert_eq!(parsed.mode, SearchMode::Exact);
    assert_eq!(parsed.pattern, "abc");

    let regex = pattern_to_regex(&parsed.pattern, parsed.mode).unwrap();
    assert_eq!(regex, "^abc$");
}

#[test]
fn test_fuzzy_mode_workflow() {
    let parsed = parse_query("?a-b-c").unwrap();
    assert!(matches!(parsed.mode, SearchMode::Fuzzy(5)));
    assert_eq!(parsed.pattern, "a-b-c");

    let regex = pattern_to_regex(&parsed.pattern, parsed.mode).unwrap();
    assert_eq!(regex, "^a.b.c$");
}

#[test]
fn test_llm_mode_workflow() {
    let parsed = parse_query("??a-b-").unwrap();
    assert!(matches!(parsed.mode, SearchMode::LLM(2)));
    assert_eq!(parsed.pattern, "a-b-");

    let regex = pattern_to_regex(&parsed.pattern, parsed.mode).unwrap();
    assert_eq!(regex, "^a.*b.*$");
}

// ============ README Examples ============

#[test]
fn test_readme_example_1_exact() {
    // Example 1: Exact Match
    let query = "abc";
    let parsed = parse_query(query).unwrap();

    assert_eq!(parsed.mode, SearchMode::Exact);
    assert_eq!(parsed.pattern, "abc");

    let regex = pattern_to_regex(&parsed.pattern, parsed.mode).unwrap();
    assert_eq!(regex, "^abc$");
}

#[test]
fn test_readme_example_2_fuzzy() {
    // Example 2: Fuzzy Match - Fixed Length
    let query = "?a-b-c";
    let parsed = parse_query(query).unwrap();

    assert!(matches!(parsed.mode, SearchMode::Fuzzy(5)));
    assert_eq!(parsed.pattern, "a-b-c");

    let regex = pattern_to_regex(&parsed.pattern, parsed.mode).unwrap();
    assert_eq!(regex, "^a.b.c$");

    // Verify some matches
    let re = regex::Regex::new(&regex).unwrap();
    assert!(re.is_match("adbec")); // a at 1, b at 3, c at 5
    assert!(re.is_match("axbyc")); // a at 1, b at 3, c at 5
    assert!(!re.is_match("abc")); // too short
}

#[test]
fn test_readme_example_3_fuzzy_short() {
    // Example 3: Fuzzy Match - Another Example
    let query = "?a-b";
    let parsed = parse_query(query).unwrap();

    assert!(matches!(parsed.mode, SearchMode::Fuzzy(3)));
    assert_eq!(parsed.pattern, "a-b");

    let regex = pattern_to_regex(&parsed.pattern, parsed.mode).unwrap();
    assert_eq!(regex, "^a.b$");

    // Verify matches
    let re = regex::Regex::new(&regex).unwrap();
    assert!(re.is_match("axb")); // a_b pattern
    assert!(re.is_match("ayb")); // a_b pattern
    assert!(!re.is_match("abc")); // wrong pattern
}

#[test]
fn test_readme_example_4_llm() {
    // Example 4: LLM Mode - Variable Length
    let query = "??a-b-";
    let parsed = parse_query(query).unwrap();

    assert!(matches!(parsed.mode, SearchMode::LLM(2)));
    assert_eq!(parsed.pattern, "a-b-");

    let regex = pattern_to_regex(&parsed.pattern, parsed.mode).unwrap();
    assert_eq!(regex, "^a.*b.*$");

    // Verify matches
    let re = regex::Regex::new(&regex).unwrap();
    assert!(re.is_match("ab")); // 2 letters
    assert!(re.is_match("axb")); // 3 letters
    assert!(re.is_match("axyb")); // 4 letters
    assert!(re.is_match("axbyc")); // 5 letters
    assert!(!re.is_match("ba")); // wrong order
}

#[test]
fn test_readme_example_5_llm_single() {
    // Example 5: LLM Mode - Single Letter
    let query = "??a-";
    let parsed = parse_query(query).unwrap();

    assert!(matches!(parsed.mode, SearchMode::LLM(1)));
    assert_eq!(parsed.pattern, "a-");

    let regex = pattern_to_regex(&parsed.pattern, parsed.mode).unwrap();
    assert_eq!(regex, "^a.*$");

    // Verify matches
    let re = regex::Regex::new(&regex).unwrap();
    assert!(re.is_match("a")); // 1 letter
    assert!(re.is_match("ab")); // a followed by anything
    assert!(re.is_match("axyz")); // a followed by anything
}

// ============ Error Handling ============

#[test]
fn test_error_empty_query() {
    let result = parse_query("");
    assert!(matches!(result, Err(PatternError::EmptyPattern)));
}

#[test]
fn test_error_invalid_char_z() {
    let result = parse_query("abz");
    assert!(matches!(
        result,
        Err(PatternError::InvalidCharacter { char: 'z' })
    ));
}

#[test]
fn test_error_too_long() {
    let result = parse_query("abcdef");
    assert!(matches!(
        result,
        Err(PatternError::LengthTooLong { actual: 6 })
    ));
}

#[test]
fn test_error_display_messages() {
    let err = PatternError::EmptyPattern;
    assert_eq!(err.to_string(), "Invalid pattern: empty");

    let err = PatternError::InvalidCharacter { char: 'z' };
    assert!(err.to_string().contains("Invalid character"));

    let err = PatternError::LengthTooLong { actual: 10 };
    assert!(err.to_string().contains("exceeds maximum"));
}

// ============ Pattern Validation ============

#[test]
fn test_valid_cj_letters_all() {
    // Test all valid CJ letters (note: z is excluded, y and x are valid)
    // Valid: a-w (23 letters) = 23 letters total (max is 5)
    let valid = "abcde";
    let result = parse_query(valid);
    assert!(result.is_ok());

    // Test with y and x
    let valid = "abcyx";
    let result = parse_query(valid);
    assert!(result.is_ok());
}

#[test]
fn test_valid_y_and_x() {
    let result = parse_query("?y-x");
    assert!(result.is_ok());
    assert_eq!(result.unwrap().pattern, "y-x");
}

#[test]
fn test_invalid_z() {
    let result = parse_query("z");
    assert!(matches!(
        result,
        Err(PatternError::InvalidCharacter { char: 'z' })
    ));
}

#[test]
fn test_case_sensitive() {
    let result = parse_query("ABC");
    // Uppercase should fail
    assert!(matches!(result, Err(PatternError::InvalidCharacter { .. })));
}

// ============ Regex Validation ============

#[test]
fn test_regex_matching_fuzzy() {
    let parsed = parse_query("?a-b").unwrap();
    let regex_str = pattern_to_regex(&parsed.pattern, parsed.mode).unwrap();
    let re = regex::Regex::new(&regex_str).unwrap();

    // Should match: a_b (3 letters, specific pattern)
    assert!(re.is_match("axb"));
    assert!(re.is_match("a1b")); // dash means any char
    assert!(re.is_match("a-b")); // even dash itself

    // Should NOT match
    assert!(!re.is_match("ab")); // too short
    assert!(!re.is_match("axbx")); // too long
    assert!(!re.is_match("ba")); // wrong order
}

#[test]
fn test_regex_matching_llm() {
    let parsed = parse_query("??a-b").unwrap();
    let regex_str = pattern_to_regex(&parsed.pattern, parsed.mode).unwrap();
    let re = regex::Regex::new(&regex_str).unwrap();

    // Should match: a followed by anything, then b
    assert!(re.is_match("ab")); // 2 letters
    assert!(re.is_match("axb")); // 3 letters
    assert!(re.is_match("axxxb")); // many letters between

    // Should NOT match
    assert!(!re.is_match("ba")); // wrong order
    assert!(!re.is_match("ax")); // missing b
}

// ============ Edge Cases ============

#[test]
fn test_single_letter_modes() {
    // Exact
    let exact = parse_query("a").unwrap();
    assert_eq!(exact.mode, SearchMode::Exact);

    // Fuzzy
    let fuzzy = parse_query("?a").unwrap();
    assert!(matches!(fuzzy.mode, SearchMode::Fuzzy(1)));

    // LLM
    let llm = parse_query("??a").unwrap();
    assert!(matches!(llm.mode, SearchMode::LLM(1)));
}

#[test]
fn test_max_length_modes() {
    // Exact: 5 letters
    let exact = parse_query("abcde").unwrap();
    assert_eq!(exact.mode, SearchMode::Exact);

    // Fuzzy: 5 total (3 letters + 2 dashes)
    let fuzzy = parse_query("?a-b-c").unwrap();
    assert!(matches!(fuzzy.mode, SearchMode::Fuzzy(5)));

    // LLM: 5 letters (min length)
    let llm = parse_query("??a-b-c-d-e").unwrap();
    assert!(matches!(llm.mode, SearchMode::LLM(5)));
}

#[test]
fn test_fuzzy_all_wildcards() {
    let parsed = parse_query("?-----").unwrap();
    assert!(matches!(parsed.mode, SearchMode::Fuzzy(5)));
    assert_eq!(parsed.pattern, "-----");

    let regex = pattern_to_regex(&parsed.pattern, parsed.mode).unwrap();
    assert_eq!(regex, "^.....$");
}

// ============ Integration Workflow ============

#[test]
fn test_full_workflow_exact() {
    let query = "abc";

    // Step 1: Parse
    let parsed = parse_query(query).unwrap();
    assert_eq!(parsed.mode, SearchMode::Exact);

    // Step 2: Generate regex
    let regex = pattern_to_regex(&parsed.pattern, parsed.mode).unwrap();

    // Step 3: Use for matching (simulated)
    let re = regex::Regex::new(&regex).unwrap();
    assert!(re.is_match("abc"));
    assert!(!re.is_match("abcd"));
}

#[test]
fn test_full_workflow_fuzzy() {
    let query = "?a-b";

    // Step 1: Parse
    let parsed = parse_query(query).unwrap();
    assert!(matches!(parsed.mode, SearchMode::Fuzzy(3)));

    // Step 2: Generate regex
    let regex = pattern_to_regex(&parsed.pattern, parsed.mode).unwrap();

    // Step 3: Use for matching (simulated)
    let re = regex::Regex::new(&regex).unwrap();
    assert!(re.is_match("axb"));
    assert!(!re.is_match("axbx"));
}

#[test]
fn test_full_workflow_llm() {
    let query = "??a-b-";

    // Step 1: Parse
    let parsed = parse_query(query).unwrap();
    assert!(matches!(parsed.mode, SearchMode::LLM(2)));

    // Step 2: Generate regex
    let regex = pattern_to_regex(&parsed.pattern, parsed.mode).unwrap();

    // Step 3: Use for matching (simulated)
    let re = regex::Regex::new(&regex).unwrap();
    assert!(re.is_match("ab"));
    assert!(re.is_match("axb"));
    assert!(re.is_match("axyb"));
    assert!(!re.is_match("ba"));
}
