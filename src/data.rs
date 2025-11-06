// Data loader for embedded binary files
// Loads dictionary and rules data at compile time

/// Embedded dictionary data (C2H - Character to Hash mapping)
pub const C2H_DATA: &[u8] = include_bytes!("../data/C2H.bin");

/// Embedded dictionary data (H2C - Hash to Character mapping)
pub const H2C_DATA: &[u8] = include_bytes!("../data/H2C.bin");

/// Embedded rules data (MessagePack format)
pub const RULES_DATA: &[u8] = include_bytes!("../data/examples.msgpack");

/// Data loader utility
pub struct DataLoader;

impl DataLoader {
    /// Get C2H (Character to Hash) dictionary data
    pub fn c2h_data() -> &'static [u8] {
        C2H_DATA
    }

    /// Get H2C (Hash to Character) dictionary data
    pub fn h2c_data() -> &'static [u8] {
        H2C_DATA
    }

    /// Get rules (examples) data
    pub fn rules_data() -> &'static [u8] {
        RULES_DATA
    }

    /// Get all data info
    pub fn info() -> DataInfo {
        DataInfo {
            c2h_size: C2H_DATA.len(),
            h2c_size: H2C_DATA.len(),
            rules_size: RULES_DATA.len(),
            total_size: C2H_DATA.len() + H2C_DATA.len() + RULES_DATA.len(),
        }
    }
}

/// Information about embedded data
#[derive(Debug, Clone)]
pub struct DataInfo {
    /// Size of C2H dictionary in bytes
    pub c2h_size: usize,
    /// Size of H2C dictionary in bytes
    pub h2c_size: usize,
    /// Size of rules data in bytes
    pub rules_size: usize,
    /// Total size of all embedded data
    pub total_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_loaded() {
        assert!(!C2H_DATA.is_empty(), "C2H data should be loaded");
        assert!(!H2C_DATA.is_empty(), "H2C data should be loaded");
        assert!(!RULES_DATA.is_empty(), "Rules data should be loaded");
    }

    #[test]
    fn test_data_sizes() {
        let info = DataLoader::info();
        assert!(info.c2h_size > 0);
        assert!(info.h2c_size > 0);
        assert!(info.rules_size > 0);
        assert_eq!(
            info.total_size,
            info.c2h_size + info.h2c_size + info.rules_size
        );
    }

    #[test]
    fn test_loader_methods() {
        assert!(!DataLoader::c2h_data().is_empty());
        assert!(!DataLoader::h2c_data().is_empty());
        assert!(!DataLoader::rules_data().is_empty());
    }
}
