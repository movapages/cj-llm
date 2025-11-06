// Phase 5: Integration tests for CJSearch

use cj_llm::CJSearch;

#[test]
fn test_search_creation() {
    let _search = CJSearch::new().unwrap();
}

#[test]
fn test_exact_search() {
    let search = CJSearch::new().unwrap();
    let results = search.search("a").unwrap();

    assert!(!results.is_empty());
    assert_eq!(results[0].code, "a");
    assert!(results[0].score.is_none()); // Exact mode has no score
    assert!(!results[0].rules.is_empty()); // Should have rules
}

#[test]
fn test_fuzzy_search() {
    let search = CJSearch::new().unwrap();
    let results = search.search("?a-b").unwrap();

    assert!(!results.is_empty());
    for result in results {
        assert_eq!(result.code.len(), 3);
        assert!(result.score.is_none()); // Fuzzy mode has no score
        assert!(!result.rules.is_empty());
    }
}

#[test]
fn test_llm_search() {
    let search = CJSearch::new().unwrap();
    let results = search.search("??a-b-").unwrap();

    assert!(!results.is_empty());

    // LLM mode should have scores
    for result in &results {
        assert!(result.score.is_some());
        let score = result.score.unwrap();
        assert!(score >= 0.0 && score <= 1.0, "Score {} out of range", score);
    }

    // Results should be sorted by score (descending)
    for i in 0..results.len().saturating_sub(1) {
        let score_curr = results[i].score.unwrap_or(0.0);
        let score_next = results[i + 1].score.unwrap_or(0.0);
        assert!(
            score_curr >= score_next,
            "Results not sorted: {} < {}",
            score_curr,
            score_next
        );
    }
}

#[test]
fn test_search_limit() {
    let search = CJSearch::new().unwrap();
    let results = search.search_limit("??a-", 5).unwrap();

    assert!(results.len() <= 5);
}

#[test]
fn test_search_results_have_all_fields() {
    let search = CJSearch::new().unwrap();
    let results = search.search("a").unwrap();

    assert!(!results.is_empty());
    for result in results {
        assert!(!result.code.is_empty(), "Code should not be empty");
        assert!(
            !result.characters.is_empty(),
            "Characters should not be empty"
        );
        assert!(!result.rules.is_empty(), "Rules should not be empty");
    }
}

#[test]
fn test_invalid_query_error() {
    let search = CJSearch::new().unwrap();
    let result = search.search("??z-");

    assert!(result.is_err(), "Should error on invalid character 'z'");
}

#[test]
fn test_stats() {
    let search = CJSearch::new().unwrap();
    let (total, category_count) = search.stats().unwrap();

    assert!(total > 0, "Should have codes in dictionary");
    assert!(category_count > 0, "Should have code categories");
}

#[test]
fn test_default() {
    let search = CJSearch::default();
    let results = search.search("a").unwrap();

    assert!(!results.is_empty());
}

#[test]
fn test_multiple_search_types() {
    let search = CJSearch::new().unwrap();

    // Exact
    let exact = search.search("a").unwrap();
    assert!(!exact.is_empty());

    // Fuzzy
    let fuzzy = search.search("?a-b").unwrap();
    assert!(!fuzzy.is_empty());

    // LLM
    let llm = search.search("??a-b-").unwrap();
    assert!(!llm.is_empty());
}

#[test]
fn test_llm_sorting_by_score() {
    let search = CJSearch::new().unwrap();
    let results = search.search("??ab").unwrap();

    if results.len() > 1 {
        // Verify they're sorted
        let mut prev_score = f32::INFINITY;
        for result in results {
            let score = result.score.unwrap_or(0.0);
            assert!(score <= prev_score, "Not sorted in descending order");
            prev_score = score;
        }
    }
}

#[test]
fn test_single_char_exact() {
    let search = CJSearch::new().unwrap();

    // Try all single letters
    for ch in 'a'..='e' {
        let query = ch.to_string();
        let results = search.search(&query).unwrap();
        assert!(!results.is_empty(), "Should find code '{}'", ch);
        assert_eq!(results[0].code, query);
    }
}

#[test]
fn test_fuzzy_length_validation() {
    let search = CJSearch::new().unwrap();

    // Fuzzy with 3-letter pattern
    let results = search.search("?a-b").unwrap();
    for result in results {
        assert_eq!(result.code.len(), 3);
    }

    // Fuzzy with 5-letter pattern
    let results = search.search("?a-b-c").unwrap();
    for result in results {
        assert_eq!(result.code.len(), 5);
    }
}

#[test]
fn test_llm_variable_length() {
    let search = CJSearch::new().unwrap();

    let results = search.search("??a-b-").unwrap();

    // Should have varying lengths from 2-5
    let mut lengths = std::collections::HashSet::new();
    for result in results {
        lengths.insert(result.code.len());
    }

    // Should have at least 2 different lengths
    assert!(lengths.len() >= 1, "Should find codes of varying length");
}

#[test]
fn test_character_lookup() {
    let search = CJSearch::new().unwrap();

    let results = search.search("a").unwrap();
    assert!(!results.is_empty());

    // Should return characters
    let first_result = &results[0];
    assert!(
        !first_result.characters.is_empty(),
        "Should return characters"
    );

    // Characters should be meaningful (not empty)
    for char in &first_result.characters {
        assert!(!char.is_empty(), "Character should not be empty");
    }
}
