// CJ-LLM Search CLI Tool
// Command-line interface for CangJie code search

use cj_llm::CJSearch;
use clap::Parser;

/// CangJie Search Tool - Search CangJie codes with three modes
#[derive(Parser, Debug)]
#[command(name = "cj-search")]
#[command(about = "Search CangJie codes using exact, fuzzy, or LLM modes", long_about = None)]
#[command(version = "0.3.0")]
struct Args {
    /// Search pattern
    /// - No prefix: exact match (e.g., "abc")
    /// - "?" prefix: fuzzy match (e.g., "?a-b-c")
    /// - "??" prefix: LLM mode with ranking (e.g., "??a-b-")
    #[arg(value_name = "PATTERN")]
    pattern: String,

    /// Maximum number of results to display
    #[arg(short, long, default_value = "10")]
    limit: usize,

    /// Show rule annotations for each result
    #[arg(short, long)]
    rules: bool,

    /// Show LLM scores (only in LLM mode)
    #[arg(short, long)]
    scores: bool,

    /// Show detailed information
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Load search engine
    if args.verbose {
        println!("🔍 Loading CJSearch engine...");
    }

    let search = CJSearch::new()?;

    if args.verbose {
        let (total, categories) = search.stats()?;
        println!(
            "✅ Dictionary loaded: {} codes in {} categories\n",
            total, categories
        );
    }

    // Determine search mode
    let mode = if args.pattern.starts_with("??") {
        "LLM (Ranking)"
    } else if args.pattern.starts_with("?") {
        "Fuzzy"
    } else {
        "Exact"
    };

    // Execute search
    if args.verbose {
        println!("🔎 Searching in {} mode: {}", mode, args.pattern);
        println!("─────────────────────────────────────────────────\n");
    } else {
        println!("Mode: {}", mode);
    }

    let results = search.search_limit(&args.pattern, args.limit)?;

    if results.is_empty() {
        println!("❌ No matches found.");
        return Ok(());
    }

    println!("✅ Found {} matches:\n", results.len());

    // Display results
    for (idx, result) in results.iter().enumerate() {
        print!("{}. ", idx + 1);
        print!("{:<15}", result.code);

        // Show characters
        print!("→ ");
        for (i, ch) in result.characters.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{}", ch);
        }
        println!();

        // Show score if LLM mode and scores requested
        if args.scores && result.score.is_some() {
            let score = result.score.unwrap();
            let percent = (score * 100.0) as i32;
            println!("      Score: {}% {}", percent, score_bar(score));
        }

        // Show rules if requested
        if args.rules {
            print!("      Rules: ");
            for (i, rule) in result.rules.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print!("{:?}", rule);
            }
            println!();
        }

        println!();
    }

    if args.verbose {
        println!("─────────────────────────────────────────────────");
        println!("✨ Search completed successfully!");
    }

    Ok(())
}

/// Generate a visual score bar
fn score_bar(score: f32) -> String {
    let bar_len = 10;
    let filled = (score * bar_len as f32) as usize;
    let mut bar = String::from("[");
    for i in 0..bar_len {
        if i < filled {
            bar.push('█');
        } else {
            bar.push('░');
        }
    }
    bar.push(']');
    bar
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_bar() {
        let bar = score_bar(0.5);
        assert!(bar.contains('█'));
        assert!(bar.contains('░'));
        // bar is "[█████░░░░░]" which is 12 characters total in UTF-8, but len() counts bytes
        assert!(bar.contains('['));
        assert!(bar.contains(']'));
    }

    #[test]
    fn test_score_bar_full() {
        let bar = score_bar(1.0);
        assert_eq!(bar, "[██████████]");
    }

    #[test]
    fn test_score_bar_empty() {
        let bar = score_bar(0.0);
        assert_eq!(bar, "[░░░░░░░░░░]");
    }
}
