// Performance benchmarks for cj-llm search operations

use cj_llm::CJSearch;
use std::time::Instant;

fn main() {
    println!("🏃 CJ-LLM Performance Benchmarks\n");

    let search = CJSearch::new().expect("Failed to load search engine");

    // Warmup
    let _ = search.search("a");

    bench_exact_match(&search);
    bench_fuzzy_match(&search);
    bench_llm_match(&search);
    bench_batch_operations(&search);

    println!("\n✅ Benchmarks completed!");
}

fn bench_exact_match(search: &CJSearch) {
    println!("📍 EXACT MATCH (O(1) lookup)");
    println!("─────────────────────────────");

    let patterns = vec!["a", "b", "c", "w"];

    for pattern in patterns {
        let start = Instant::now();
        let results = search.search(pattern).expect("Search failed");
        let duration = start.elapsed();

        println!(
            "  {:<10} → {} results in {:.3}ms",
            pattern,
            results.len(),
            duration.as_secs_f64() * 1000.0
        );
    }
    println!();
}

fn bench_fuzzy_match(search: &CJSearch) {
    println!("🔤 FUZZY MATCH (Pattern filtering)");
    println!("─────────────────────────────────");

    let patterns = vec!["?a-b", "?a-c", "?ab-"];

    for pattern in patterns {
        let start = Instant::now();
        let results = search.search(pattern).expect("Search failed");
        let duration = start.elapsed();

        println!(
            "  {:<10} → {} results in {:.3}ms",
            pattern,
            results.len(),
            duration.as_secs_f64() * 1000.0
        );
    }
    println!();
}

fn bench_llm_match(search: &CJSearch) {
    println!("🧠 LLM MODE (Neural ranking)");
    println!("──────────────────────────────");

    let patterns = vec!["??ab", "??abc", "??a-b-"];

    for pattern in patterns {
        let start = Instant::now();
        let results = search.search_limit(pattern, 10).expect("Search failed");
        let duration = start.elapsed();

        println!(
            "  {:<10} → {} results in {:.3}ms",
            pattern,
            results.len(),
            duration.as_secs_f64() * 1000.0
        );
    }
    println!();
}

fn bench_batch_operations(search: &CJSearch) {
    println!("📦 BATCH OPERATIONS");
    println!("─────────────────────");

    let patterns = vec!["a", "b", "c", "?ab-", "?cd-", "??ab", "??cd-"];

    let start = Instant::now();
    for pattern in patterns {
        let _ = search.search(pattern);
    }
    let total = start.elapsed();

    println!(
        "  7 searches in {:.3}ms ({:.3}ms avg)",
        total.as_secs_f64() * 1000.0,
        (total.as_secs_f64() / 7.0) * 1000.0
    );

    // Stats
    let (total_codes, categories) = search.stats().expect("Stats failed");
    println!("\n📊 Dictionary Statistics");
    println!("─────────────────────────");
    println!("  Total codes: {} codes", total_codes);
    println!("  Categories: {} length categories", categories);
}
