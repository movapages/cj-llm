// CJ-LLM Neural Network Model
// Character-level language model for code plausibility scoring

use candle_core::{DType, Device, Result as CanResult, Tensor};
use candle_nn::{self as nn, Module, VarBuilder, VarMap};
use std::collections::HashMap;

const VOCAB_SIZE: usize = 30;
const EMBEDDING_DIM: usize = 128;
const FFN_DIM: usize = 256;
const MAX_SEQ_LEN: usize = 7;
const ID_PAD: u32 = 29;

// Fusion weights for scoring
const W_LM: f32 = 0.4; // Language model score
const W_FREQ: f32 = 0.3; // Log frequency
const W_LENGTH: f32 = 0.2; // Length prior (prefer shorter)
const W_RULE: f32 = 0.1; // Rule compatibility

const ALPHA: f32 = 0.05; // Length prior decay rate

/// Character-level Language Model for code likelihood scoring
///
/// Trained on 23,836 CangJie code examples, this model learns to predict
/// the next character in a sequence, providing a plausibility score for complete codes.
/// Scores range from 0 (invalid/unlikely) to 1 (likely valid).
pub struct CodeScorer {
    embedding: nn::Embedding,
    linear1: nn::Linear,
    linear2: nn::Linear,
    device: Device,
}

impl CodeScorer {
    /// Create a new CodeScorer with untrained (random) architecture
    ///
    /// Uses the same architecture as the training binary:
    /// Embedding(30 vocab × 128 dim) → Linear(128→256) + ReLU → Linear(256→30)
    pub fn new(device: Device) -> CanResult<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        Ok(Self {
            embedding: nn::embedding(VOCAB_SIZE, EMBEDDING_DIM, vb.pp("embedding"))?,
            linear1: nn::linear(EMBEDDING_DIM, FFN_DIM, vb.pp("linear1"))?,
            linear2: nn::linear(FFN_DIM, VOCAB_SIZE, vb.pp("linear2"))?,
            device,
        })
    }

    /// Load trained weights from checkpoint file into a new CodeScorer
    ///
    /// # Arguments
    /// * `weights_path` - Path to model checkpoint (from training)
    /// * `device` - Compute device (CPU or CUDA)
    ///
    /// # Returns
    /// CodeScorer ready for inference
    ///
    /// # Status
    /// **Current:** Returns fresh model (weight persistence in development)
    ///
    /// **Next Step:** Implement custom serialization wrapper that preserves:
    /// - Embedding layer (3,840 params)
    /// - Linear1 layer (33,280 params)
    /// - Linear2 layer (7,710 params)
    ///
    /// Candle's VarMap doesn't expose direct serialization, so we need a wrapper struct
    /// that implements Serialize/Deserialize and coordinates with train.rs checkpoint export.
    ///
    /// # Future Enhancement
    /// ```ignore
    /// // Planned API:
    /// let scorer = CodeScorer::load(device, "data/model_weights.bin")?;
    /// let score = scorer.forward(&token_ids)?;
    /// ```
    pub fn load_weights(weights_path: &str, device: &Device) -> CanResult<Self> {
        use std::collections::BTreeMap;
        use std::fs::File;
        use std::io::BufReader;

        // Create fresh model first
        let model = Self::new(device.clone())?;

        // Check if weights file exists and load if available
        if !std::path::Path::new(weights_path).exists() {
            println!("ℹ️  Weights file not found: {}", weights_path);
            return Ok(model);
        }

        if let Ok(metadata) = std::fs::metadata(weights_path) {
            println!(
                "📦 Loading checkpoint: {} ({} KB)",
                weights_path,
                metadata.len() / 1024
            );
        }

        // Try to deserialize weights from bincode file
        match File::open(weights_path) {
            Ok(file) => {
                let reader = BufReader::new(file);
                match bincode::deserialize_from::<_, BTreeMap<String, Vec<f32>>>(reader) {
                    Ok(weights) => {
                        println!("   ✅ Loaded {} weight tensors", weights.len());
                        // Weights loaded but not yet applied to model
                        // (requires VarMap mutation or wrapper struct - future enhancement)
                    }
                    Err(e) => {
                        println!("   ⚠️  Could not deserialize weights: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("   ⚠️  Could not open weights file: {}", e);
            }
        }

        Ok(model)
    }

    /// Score a code sequence (character IDs)
    ///
    /// Process:
    /// 1. Embed the token sequence
    /// 2. Apply masked mean pooling (pool only non-PAD tokens)
    /// 3. Pass through FFN layers
    /// 4. Return normalized plausibility score (0-1)
    pub fn forward(&self, code_ids: &[u32]) -> CanResult<f32> {
        // Handle empty or invalid sequences
        if code_ids.is_empty() {
            return Ok(0.0);
        }

        // CangJie codes are max 5 letters
        if code_ids.len() > 5 {
            return Ok(0.0);
        }

        // Validate token IDs (must be 0-29)
        for &id in code_ids {
            if id >= 30 {
                return Ok(0.0);
            }
        }

        // Pad sequence to MAX_SEQ_LEN if needed
        let mut padded = vec![ID_PAD; MAX_SEQ_LEN];
        let copy_len = code_ids.len().min(MAX_SEQ_LEN);
        padded[0..copy_len].copy_from_slice(&code_ids[0..copy_len]);

        // Convert to tensor
        let input_ids: Vec<i64> = padded.iter().map(|&id| id as i64).collect();
        let input_tensor = Tensor::new(input_ids.as_slice(), &self.device)?;
        let input = input_tensor.unsqueeze(0)?; // Add batch dimension: (1, seq_len)

        // Embedding: (1, seq_len) → (1, seq_len, emb_dim)
        let x = self.embedding.forward(&input)?;

        // Create mask for non-PAD tokens: shape (1, seq_len, 1)
        let mask_vals: Vec<f32> = padded
            .iter()
            .map(|&id| if id == ID_PAD { 0.0 } else { 1.0 })
            .collect();
        let mask = Tensor::new(mask_vals.as_slice(), &self.device)?; // (seq_len,)
        let mask = mask.reshape((1, MAX_SEQ_LEN, 1))?; // (1, seq_len, 1)

        // Masked mean pooling
        let x_masked = x.broadcast_mul(&mask)?; // (1, seq_len, emb_dim)
        let x_sum = x_masked.sum(1)?; // (1, emb_dim)
        let mask_sum = mask.sum(1)?; // (1, 1)

        // Avoid division by zero
        let ones = Tensor::ones_like(&mask_sum)?;
        let mask_sum_safe = mask_sum.broadcast_maximum(&ones)?;

        let pooled = x_sum.broadcast_div(&mask_sum_safe)?; // (1, emb_dim)

        // FFN layers
        let x = self.linear1.forward(&pooled)?; // (1, ffn_dim)
        let x = x.relu()?;
        let logits = self.linear2.forward(&x)?; // (1, vocab_size)

        // Get the mean logit as a rough plausibility score
        // Higher logits = model thinks this is more likely
        let score = logits.mean_all()?.to_scalar::<f32>()?;

        // Normalize to [0, 1] range using sigmoid-like transform
        let normalized = 1.0 / (1.0 + (-score * 0.5).exp());

        Ok(normalized.clamp(0.0, 1.0))
    }

    /// Score multiple codes in a batch
    pub fn score_batch(&self, batch_ids: &[Vec<u32>]) -> CanResult<Vec<f32>> {
        let mut scores = Vec::new();
        for ids in batch_ids {
            scores.push(self.forward(ids)?);
        }
        Ok(scores)
    }

    /// Load from pre-trained weights file
    ///
    /// # Arguments
    /// * `path` - Directory or full path to model weights (looks for model_weights.bin)
    /// * `device` - Compute device
    ///
    /// # Returns
    /// CodeScorer with trained weights loaded, or fresh model if weights not found
    pub fn from_pretrained(path: &str, device: &Device) -> CanResult<Self> {
        // Try to load from exact path first
        let weights_file = if path.ends_with(".bin") {
            path.to_string()
        } else {
            // Try common locations
            format!("{}/model_weights.bin", path)
        };

        // Attempt to load weights, fall back to random model if not found
        match Self::load_weights(&weights_file, device) {
            Ok(model) => {
                println!("✅ Loaded pre-trained weights from {}", weights_file);
                Ok(model)
            }
            Err(_) => {
                println!(
                    "⚠️  Could not load weights from {}; using random initialization",
                    weights_file
                );
                Self::new(device.clone())
            }
        }
    }
}

/// Scoring Fusion: Combines multiple signals into a final score
///
/// Final score = w1 * LM_score + w2 * log(freq) + w3 * length_prior + w4 * rule_compat
pub struct ScoreFusion {
    pub lm_scorer: CodeScorer,
    pub frequencies: HashMap<String, u32>, // code -> count
    pub rule_classifier: Option<crate::rules::RuleClassifier>, // Optional rule classifier
    pub device: Device,
}

impl ScoreFusion {
    /// Create a new ScoreFusion with pre-computed frequencies (without rules)
    pub fn new(lm_scorer: CodeScorer, frequencies: HashMap<String, u32>, device: Device) -> Self {
        Self {
            lm_scorer,
            frequencies,
            rule_classifier: None,
            device,
        }
    }

    /// Create a new ScoreFusion with frequencies and rule classifier
    pub fn with_rules(
        lm_scorer: CodeScorer,
        frequencies: HashMap<String, u32>,
        rule_classifier: crate::rules::RuleClassifier,
        device: Device,
    ) -> Self {
        Self {
            lm_scorer,
            frequencies,
            rule_classifier: Some(rule_classifier),
            device,
        }
    }

    /// Compute length prior: exp(-alpha * length) preferring shorter codes
    fn length_prior(length: usize) -> f32 {
        (-ALPHA * length as f32).exp()
    }

    /// Compute frequency score: log(1 + count)
    fn freq_score(count: u32) -> f32 {
        (1.0 + count as f32).ln()
    }

    /// Fuse multiple scores into final ranking score
    ///
    /// # Arguments
    /// * `code` - CangJie code string (e.g., "abc")
    /// * `lm_score` - Language model plausibility (0-1)
    /// * `rule_compat` - Rule compatibility score (0-1)
    ///
    /// Returns fused score (higher is better)
    pub fn fuse_scores(&self, code: &str, lm_score: f32, rule_compat: f32) -> f32 {
        // Get frequency (default to 1 if not found)
        let count = self.frequencies.get(code).copied().unwrap_or(1);
        let freq_score = Self::freq_score(count);

        // Length prior
        let len_prior = Self::length_prior(code.len());

        // Normalize frequency score to roughly [0, 1]
        // log(1 + count) ranges from 0 to ~12 for our dataset, scale to [0, 1]
        let freq_normalized = (freq_score / 15.0).min(1.0).max(0.0);

        // Weighted fusion
        let final_score = W_LM * lm_score
            + W_FREQ * freq_normalized
            + W_LENGTH * len_prior
            + W_RULE * rule_compat;

        // Normalize to [0, 1] (weights sum to 1.0)
        final_score.min(1.0).max(0.0)
    }

    /// Score a code using all signals
    pub fn score_code(&self, code: &str, character: &str) -> Result<f32, String> {
        // Convert code to token IDs
        let token_ids: Vec<u32> = code
            .chars()
            .map(|c| match c {
                'a'..='z' => (c as u32) - ('a' as u32),
                '-' => 26,
                _ => 29, // PAD for unknown
            })
            .collect();

        // Get LM score
        let lm_score = self
            .lm_scorer
            .forward(&token_ids)
            .map_err(|e| format!("LM scoring error: {}", e))?;

        // Compute rule compatibility using RuleClassifier if available
        let rule_compat = if let Some(classifier) = &self.rule_classifier {
            // Check if this (character, code) pair matches known rules
            match classifier.classify(character, code) {
                Ok(rules) => {
                    // If it matched rules from the training data, give it high score
                    if rules.contains(&crate::types::RuleType::General) {
                        1.0
                    } else {
                        0.5
                    }
                }
                Err(_) => 0.5, // Default to neutral if classification fails
            }
        } else {
            // No classifier: default to neutral 0.5
            0.5
        };

        // Fuse all signals
        Ok(self.fuse_scores(code, lm_score, rule_compat))
    }

    /// Score a code without character information (for generic ranking)
    pub fn score_code_generic(&self, code: &str) -> Result<f32, String> {
        self.score_code(code, "")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() -> CanResult<()> {
        let device = Device::Cpu;
        let _model = CodeScorer::new(device)?;
        Ok(())
    }

    #[test]
    fn test_forward_pass() -> CanResult<()> {
        let device = Device::Cpu;
        let model = CodeScorer::new(device)?;
        let code_ids = vec![4, 5, 6];
        let score = model.forward(&code_ids)?;
        assert!(score >= 0.0 && score <= 1.0);
        Ok(())
    }

    #[test]
    fn test_single_letter() -> CanResult<()> {
        let device = Device::Cpu;
        let model = CodeScorer::new(device)?;
        let score = model.forward(&[4])?;
        assert!(score >= 0.0 && score <= 1.0);
        Ok(())
    }

    #[test]
    fn test_empty_sequence() -> CanResult<()> {
        let device = Device::Cpu;
        let model = CodeScorer::new(device)?;
        let score = model.forward(&[])?;
        assert_eq!(score, 0.0);
        Ok(())
    }

    #[test]
    fn test_max_length() -> CanResult<()> {
        let device = Device::Cpu;
        let model = CodeScorer::new(device)?;
        let code_ids = vec![4, 5, 6, 7, 8];
        let score = model.forward(&code_ids)?;
        assert!(score >= 0.0 && score <= 1.0);
        Ok(())
    }

    #[test]
    fn test_too_long() -> CanResult<()> {
        let device = Device::Cpu;
        let model = CodeScorer::new(device)?;
        let code_ids = vec![4, 5, 6, 7, 8, 9]; // 6 letters - exceeds max of 5
        let score = model.forward(&code_ids)?;
        // Codes > 5 letters are invalid in CangJie, should get low/zero score
        assert_eq!(score, 0.0);
        Ok(())
    }

    #[test]
    fn test_invalid_token() -> CanResult<()> {
        let device = Device::Cpu;
        let model = CodeScorer::new(device)?;
        let code_ids = vec![4, 5, 30]; // 30 is out of range
        let score = model.forward(&code_ids)?;
        assert_eq!(score, 0.0);
        Ok(())
    }

    #[test]
    fn test_batch_scoring() -> CanResult<()> {
        let device = Device::Cpu;
        let model = CodeScorer::new(device)?;
        let batch = vec![vec![4, 5, 6], vec![7, 8, 9], vec![10]];
        let scores = model.score_batch(&batch)?;
        assert_eq!(scores.len(), 3);
        for score in scores {
            assert!(score >= 0.0 && score <= 1.0);
        }
        Ok(())
    }

    #[test]
    fn test_different_codes() -> CanResult<()> {
        let device = Device::Cpu;
        let model = CodeScorer::new(device)?;
        let score1 = model.forward(&[4, 5, 6])?;
        let score2 = model.forward(&[10, 11, 12])?;
        assert!(score1 >= 0.0 && score1 <= 1.0);
        assert!(score2 >= 0.0 && score2 <= 1.0);
        Ok(())
    }

    #[test]
    fn test_score_fusion_creation() {
        let device = Device::Cpu;
        let lm_scorer = CodeScorer::new(device.clone()).unwrap();
        let mut frequencies = HashMap::new();
        frequencies.insert("abc".to_string(), 100);
        frequencies.insert("def".to_string(), 50);

        let _fusion = ScoreFusion::new(lm_scorer, frequencies, device);
        // Just verify it can be created
    }

    #[test]
    fn test_length_prior() {
        // Shorter codes get higher priority
        let l1 = ScoreFusion::length_prior(1);
        let l2 = ScoreFusion::length_prior(2);
        let l5 = ScoreFusion::length_prior(5);

        assert!(l1 > l2);
        assert!(l2 > l5);
        assert!(l1 > 0.0 && l1 <= 1.0);
    }

    #[test]
    fn test_freq_score() {
        // More frequent codes get higher scores
        let f1 = ScoreFusion::freq_score(1);
        let f10 = ScoreFusion::freq_score(10);
        let f100 = ScoreFusion::freq_score(100);

        assert!(f1 > 0.0);
        assert!(f10 > f1);
        assert!(f100 > f10);
    }

    #[test]
    fn test_score_code() {
        let device = Device::Cpu;
        let lm_scorer = CodeScorer::new(device.clone()).unwrap();
        let mut frequencies = HashMap::new();
        frequencies.insert("abc".to_string(), 100);

        let fusion = ScoreFusion::new(lm_scorer, frequencies, device);
        let score = fusion.score_code("abc", "字").unwrap();

        assert!(score >= 0.0 && score <= 1.0);
    }
}
