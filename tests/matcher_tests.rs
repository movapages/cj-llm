// Integration tests for Phase 2: Dictionary Matcher

use cj_llm::{parse_query, DictionaryMatcher, SearchMode};

// ============ Matcher Creation ============

#[test]
fn test_matcher_creates_successfully() {
    let matcher = DictionaryMatcher::new().unwrap();
    assert!(matcher.total_codes() > 0);
}

#[test]
fn test_matcher_indexes_all_lengths() {
    let matcher = DictionaryMatcher::new().unwrap();

    for len in 1..=5 {
        let codes = matcher.codes_with_length(len).unwrap();
        assert!(!codes.is_empty(), "Should have codes of length {}", len);
    }
}

#[test]
fn test_codes_count_by_length() {
    let matcher = DictionaryMatcher::new().unwrap();
    let counts = matcher.codes_count_by_length();

    // Should have 5 categories (1-5 letters)
    assert_eq!(counts.len(), 5);

    // Total should match
    let total: usize = counts.values().sum();
    assert_eq!(total, matcher.total_codes());

    println!("Code distribution: {:?}", counts);
}

// ============ Exact Search Tests ============

#[test]
fn test_exact_single_letter_code() {
    let matcher = DictionaryMatcher::new().unwrap();

    // "a" should be valid
    let results = matcher.search(SearchMode::Exact, "a").unwrap();
    assert!(!results.is_empty(), "Code 'a' should exist");
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, "a");
}

#[test]
fn test_exact_invalid_code() {
    let matcher = DictionaryMatcher::new().unwrap();

    // "zzz" should not exist (z is excluded)
    let results = matcher.search(SearchMode::Exact, "zzz").unwrap();
    assert!(results.is_empty(), "Invalid code should return empty");
}

#[test]
fn test_exact_search_workflow() {
    let matcher = DictionaryMatcher::new().unwrap();

    // Parse query and search
    let parsed = parse_query("abc").unwrap();
    assert_eq!(parsed.mode, SearchMode::Exact);

    let results = matcher.search(parsed.mode, &parsed.pattern).unwrap();

    // Should find the exact code or be empty (depending on dictionary)
    for (code, chars) in results {
        assert_eq!(code, "abc");
        assert!(!chars.is_empty());
        println!("Code 'abc' maps to: {:?}", chars);
    }
}

// ============ Fuzzy Search Tests ============

#[test]
fn test_fuzzy_length_3_matches() {
    let matcher = DictionaryMatcher::new().unwrap();

    // Parse "?a-b" to get Fuzzy(3)
    let parsed = parse_query("?a-b").unwrap();
    assert!(matches!(parsed.mode, SearchMode::Fuzzy(3)));

    let results = matcher.search(parsed.mode, &parsed.pattern).unwrap();

    // Should find codes matching a_b pattern
    assert!(
        !results.is_empty(),
        "Should find codes matching 'a-b' pattern"
    );

    for (code, _) in &results {
        assert_eq!(code.len(), 3, "Code '{}' should be 3 letters", code);
        // Pattern is "a-b" → regex "^a.b$"
        assert!(
            code.starts_with('a'),
            "Code '{}' should start with 'a'",
            code
        );
        assert!(code.ends_with('b'), "Code '{}' should end with 'b'", code);
    }

    println!("Found {} codes matching 'a-b' pattern", results.len());
}

#[test]
fn test_fuzzy_length_5_matches() {
    let matcher = DictionaryMatcher::new().unwrap();

    // Parse "?a-b-c" to get Fuzzy(5)
    let parsed = parse_query("?a-b-c").unwrap();
    assert!(matches!(parsed.mode, SearchMode::Fuzzy(5)));

    let results = matcher.search(parsed.mode, &parsed.pattern).unwrap();

    // Should find codes matching a_b_c pattern (5 letters)
    assert!(
        !results.is_empty(),
        "Should find codes matching 'a-b-c' pattern"
    );

    for (code, _) in &results {
        assert_eq!(code.len(), 5, "Code '{}' should be 5 letters", code);
        assert_eq!(
            code.chars().nth(0).unwrap(),
            'a',
            "First char should be 'a'"
        );
        assert_eq!(
            code.chars().nth(2).unwrap(),
            'b',
            "Third char should be 'b'"
        );
        assert_eq!(
            code.chars().nth(4).unwrap(),
            'c',
            "Fifth char should be 'c'"
        );
    }

    println!("Found {} codes matching 'a-b-c' pattern", results.len());
}

#[test]
fn test_fuzzy_single_letter_exact() {
    let matcher = DictionaryMatcher::new().unwrap();

    // Parse "?abc" to get Fuzzy(3) with pattern "abc"
    let parsed = parse_query("?abc").unwrap();
    assert!(matches!(parsed.mode, SearchMode::Fuzzy(3)));

    let results = matcher.search(parsed.mode, &parsed.pattern).unwrap();

    // Should match only exact 3-letter code "abc"
    for (code, _) in &results {
        assert_eq!(code, "abc", "Only 'abc' matches this exact pattern");
    }
}

// ============ LLM Search Tests ============

#[test]
fn test_llm_mode_returns_variable_length() {
    let matcher = DictionaryMatcher::new().unwrap();

    // Parse "??a-b-" to get LLM(2)
    let parsed = parse_query("??a-b-").unwrap();
    assert!(matches!(parsed.mode, SearchMode::LLM(2)));

    let results = matcher.search(parsed.mode, &parsed.pattern).unwrap();

    // Should find codes of length 2-5 containing 'a' then 'b'
    assert!(!results.is_empty(), "Should find codes for LLM mode");

    let mut lengths_found = std::collections::HashSet::new();
    for (code, _) in &results {
        lengths_found.insert(code.len());
        // All should have 'a' before 'b'
        let a_pos = code.find('a').unwrap_or(usize::MAX);
        let b_pos = code.find('b').unwrap_or(usize::MAX);
        assert!(a_pos < b_pos, "Code '{}' should have 'a' before 'b'", code);
    }

    // Should have multiple lengths
    assert!(
        lengths_found.len() > 1 || *lengths_found.iter().next().unwrap() >= 2,
        "Should find codes of various lengths"
    );

    println!(
        "Found {} codes with lengths: {:?}",
        results.len(),
        lengths_found
    );
}

#[test]
fn test_llm_single_letter_prefix() {
    let matcher = DictionaryMatcher::new().unwrap();

    // Parse "??a-" to get LLM(1)
    let parsed = parse_query("??a-").unwrap();
    assert!(matches!(parsed.mode, SearchMode::LLM(1)));

    let results = matcher.search(parsed.mode, &parsed.pattern).unwrap();

    // Should find many codes starting with 'a'
    assert!(!results.is_empty(), "Should find codes starting with 'a'");

    for (code, _) in &results {
        assert!(
            code.starts_with('a'),
            "Code '{}' should start with 'a'",
            code
        );
    }

    println!("Found {} codes starting with 'a'", results.len());
}

// ============ Integration Workflows ============

#[test]
fn test_full_exact_workflow() {
    let matcher = DictionaryMatcher::new().unwrap();

    // Step 1: Parse query
    let parsed = parse_query("abc").unwrap();
    assert_eq!(parsed.mode, SearchMode::Exact);

    // Step 2: Search
    let results = matcher.search(parsed.mode, &parsed.pattern).unwrap();

    // Step 3: Process results
    for (code, chars) in results {
        println!("Code: {}, Characters: {:?}", code, chars);
        assert_eq!(code.len(), 3);
        assert!(!chars.is_empty());
    }
}

#[test]
fn test_full_fuzzy_workflow() {
    let matcher = DictionaryMatcher::new().unwrap();

    // Step 1: Parse query
    let parsed = parse_query("?a-b").unwrap();

    // Step 2: Search
    let results = matcher.search(parsed.mode, &parsed.pattern).unwrap();

    // Step 3: Verify results
    assert!(!results.is_empty());
    let result_count = results.len();
    for (code, chars) in &results {
        assert_eq!(code.len(), 3);
        assert!(code.starts_with('a'));
        assert!(code.ends_with('b'));
        assert!(!chars.is_empty());
    }

    println!("Fuzzy search found {} matches", result_count);
}

#[test]
fn test_full_llm_workflow() {
    let matcher = DictionaryMatcher::new().unwrap();

    // Step 1: Parse query
    let parsed = parse_query("??a-b-").unwrap();

    // Step 2: Search
    let results = matcher.search(parsed.mode, &parsed.pattern).unwrap();

    // Step 3: Verify results
    assert!(!results.is_empty());
    assert!(
        results.len() < 1000,
        "LLM should find reasonable number of matches"
    );

    for (code, _) in &results {
        // All should match the pattern
        let a_pos = code.find('a').unwrap_or(usize::MAX);
        let b_pos = code.find('b').unwrap_or(usize::MAX);
        assert!(a_pos < b_pos, "Pattern 'a-b-' not matched in '{}'", code);
    }

    println!("LLM search found {} matches", results.len());
}

// ============ Edge Cases ============

#[test]
fn test_search_nonexistent_length() {
    let matcher = DictionaryMatcher::new().unwrap();

    // Length 6 doesn't exist in CangJie
    let codes = matcher.codes_with_length(6);
    assert!(codes.is_none());
}

#[test]
fn test_fuzzy_max_length() {
    let matcher = DictionaryMatcher::new().unwrap();

    // Parse "?abcde" to get Fuzzy(5)
    let parsed = parse_query("?abcde").unwrap();
    assert!(matches!(parsed.mode, SearchMode::Fuzzy(5)));

    // Should work fine
    let results = matcher.search(parsed.mode, &parsed.pattern).unwrap();
    for (code, _) in results {
        assert_eq!(code, "abcde");
    }
}

#[test]
fn test_llm_max_length() {
    let matcher = DictionaryMatcher::new().unwrap();

    // Parse "??abcde" to get LLM(5) - max
    let parsed = parse_query("??abcde").unwrap();
    assert!(matches!(parsed.mode, SearchMode::LLM(5)));

    let results = matcher.search(parsed.mode, &parsed.pattern).unwrap();

    // Should find 5-letter codes matching the pattern
    for (code, _) in results {
        assert_eq!(code.len(), 5);
    }
}

// ============ Performance Tests ============

#[test]
fn test_exact_search_is_fast() {
    let matcher = DictionaryMatcher::new().unwrap();

    let start = std::time::Instant::now();
    for _ in 0..100 {
        let _ = matcher.search(SearchMode::Exact, "a").unwrap();
    }
    let elapsed = start.elapsed();

    println!("100 exact searches: {:?}", elapsed);
    // Should be very fast (< 10ms for 100 searches)
    assert!(elapsed.as_millis() < 100, "Exact search should be fast");
}

#[test]
fn test_fuzzy_search_reasonable_performance() {
    let matcher = DictionaryMatcher::new().unwrap();

    let start = std::time::Instant::now();
    for _ in 0..10 {
        let _ = matcher.search(SearchMode::Fuzzy(3), "a-b").unwrap();
    }
    let elapsed = start.elapsed();

    println!("10 fuzzy searches: {:?}", elapsed);
    // Should complete in reasonable time
    assert!(
        elapsed.as_secs() < 1,
        "Fuzzy search should be reasonably fast"
    );
}
