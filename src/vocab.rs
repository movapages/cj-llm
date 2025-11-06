// CJ-LLM Vocabulary
// Maps CangJie letters to token IDs for the neural network

use std::collections::HashMap;

/// Vocabulary for CangJie code tokenization
///
/// CangJie uses 25 letters: a-w (23 letters), y, x (z is excluded)
/// Plus special tokens for padding, unknown, beginning, end
#[derive(Debug, Clone)]
pub struct Vocab {
    /// Text token → ID mapping
    pub token_to_id: HashMap<String, u32>,

    /// ID → Text token mapping (for decoding)
    pub id_to_token: Vec<String>,

    /// Vocabulary size
    pub size: usize,
}

impl Vocab {
    /// Create vocabulary with CangJie letters and special tokens
    ///
    /// Structure:
    /// - 0: <pad> (padding)
    /// - 1: <unk> (unknown)
    /// - 2: <bos> (beginning of sequence)
    /// - 3: <eos> (end of sequence)
    /// - 4-28: a-w, y, x (CangJie letters)
    pub fn new() -> Self {
        let mut token_to_id = HashMap::new();
        let mut id_to_token = Vec::new();

        // Special tokens
        let special_tokens = vec!["<pad>", "<unk>", "<bos>", "<eos>"];
        for (idx, token) in special_tokens.iter().enumerate() {
            token_to_id.insert(token.to_string(), idx as u32);
            id_to_token.push(token.to_string());
        }

        // CangJie letters: a-w (23 letters)
        for ch in 'a'..='w' {
            let id = id_to_token.len() as u32;
            token_to_id.insert(ch.to_string(), id);
            id_to_token.push(ch.to_string());
        }

        // Special CJ letters: y, x
        for ch in &['y', 'x'] {
            let id = id_to_token.len() as u32;
            token_to_id.insert(ch.to_string(), id);
            id_to_token.push(ch.to_string());
        }

        let size = id_to_token.len();

        Self {
            token_to_id,
            id_to_token,
            size,
        }
    }

    /// Get token ID from character
    pub fn get_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Get character from token ID
    pub fn get_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(|s| s.as_str())
    }

    /// Encode a code string to token IDs
    ///
    /// Unknown characters map to `[unk]` token
    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.chars()
            .map(|ch| {
                self.get_id(&ch.to_string())
                    .unwrap_or_else(|| self.get_id("<unk>").unwrap())
            })
            .collect()
    }

    /// Decode token IDs back to string
    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter().filter_map(|&id| self.get_token(id)).collect()
    }

    /// Get embedding dimension (should match model hidden size)
    pub fn embedding_dim(&self) -> usize {
        16 // Match model architecture
    }

    /// Get GRU hidden dimension
    pub fn hidden_dim(&self) -> usize {
        32 // Match model architecture
    }
}

impl Default for Vocab {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_creation() {
        let vocab = Vocab::new();
        assert_eq!(vocab.size, 29); // 4 special + 25 CJ letters
    }

    #[test]
    fn test_special_tokens() {
        let vocab = Vocab::new();

        assert_eq!(vocab.get_id("<pad>"), Some(0));
        assert_eq!(vocab.get_id("<unk>"), Some(1));
        assert_eq!(vocab.get_id("<bos>"), Some(2));
        assert_eq!(vocab.get_id("<eos>"), Some(3));
    }

    #[test]
    fn test_cj_letters() {
        let vocab = Vocab::new();

        // Test a-w
        for ch in 'a'..='w' {
            assert!(vocab.get_id(&ch.to_string()).is_some());
        }

        // Test y and x
        assert!(vocab.get_id("y").is_some());
        assert!(vocab.get_id("x").is_some());

        // Test z is not valid
        assert_eq!(vocab.get_id("z"), None);
    }

    #[test]
    fn test_encoding() {
        let vocab = Vocab::new();

        let ids = vocab.encode("abc");
        assert_eq!(ids.len(), 3);
        assert_eq!(
            ids,
            vec![
                vocab.get_id("a").unwrap(),
                vocab.get_id("b").unwrap(),
                vocab.get_id("c").unwrap()
            ]
        );
    }

    #[test]
    fn test_encoding_invalid_char() {
        let vocab = Vocab::new();

        let ids = vocab.encode("a1b");
        assert_eq!(ids.len(), 3);
        // '1' should map to <unk>
        assert_eq!(ids[1], vocab.get_id("<unk>").unwrap());
    }

    #[test]
    fn test_decoding() {
        let vocab = Vocab::new();

        let ids = vec![
            vocab.get_id("a").unwrap(),
            vocab.get_id("b").unwrap(),
            vocab.get_id("c").unwrap(),
        ];
        let decoded = vocab.decode(&ids);
        assert_eq!(decoded, "abc");
    }

    #[test]
    fn test_roundtrip() {
        let vocab = Vocab::new();

        let original = "abcde";
        let encoded = vocab.encode(original);
        let decoded = vocab.decode(&encoded);
        assert_eq!(decoded, original);
    }

    #[test]
    fn test_vocab_size() {
        let vocab = Vocab::new();
        // 4 special tokens + 25 CJ letters (a-w, y, x)
        assert_eq!(vocab.size, 29);
    }

    #[test]
    fn test_dimensions() {
        let vocab = Vocab::new();
        assert_eq!(vocab.embedding_dim(), 16);
        assert_eq!(vocab.hidden_dim(), 32);
    }
}
