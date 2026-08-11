use super::{
    lattice::Lattice,
    trainer::UnigramTrainer,
    trie::{Trie, TrieBuilder},
};
use crate::tokenizer::{Model, Result, Token};
use crate::utils::cache::{Cache, MAX_LENGTH};
use std::collections::HashMap;

use ahash::AHashMap;
use std::convert::TryInto;
use std::fs::read_to_string;
use std::path::{Path, PathBuf};

type TokenMap = AHashMap<String, u32>;
type Vocab = Vec<(String, f64)>;

/// Numerically stable log-sum-exp for two f32 values in log-space.
#[inline(always)]
fn log_sum_exp_f32(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let max = if a > b { a } else { b };
    let min = if a > b { b } else { a };
    max + (min - max).exp().ln_1p()
}

/// A precomputed forward graph over one sentence:
/// edges[s] = list of (end_pos, vocab_id) matches starting at byte offset s.
/// Includes the UNK fallback edge at s if no mblen-token exists.
#[derive(Debug)]
pub struct PreparedDP<'a> {
    sentence: &'a str,
    len: usize,
    edges: Vec<Vec<(usize /*end_pos*/, usize /*vocab_id*/)>>, // size = len+1
}

impl<'a> PreparedDP<'a> {
    #[inline]
    pub fn score_only_f32(&self, weights: &[f32], unk_id: usize, unk_score: f32) -> f32 {
        if self.len == 0 {
            return 0.0;
        }
        let mut best = vec![f32::NEG_INFINITY; self.len + 1];
        best[0] = 0.0;

        // Topological order: increasing start byte position
        for s in 0..self.len {
            let base = best[s];
            if base == f32::NEG_INFINITY {
                continue;
            }
            for &(e, id) in &self.edges[s] {
                let w = if id == unk_id { unk_score } else { unsafe { *weights.get_unchecked(id) } };
                let sc = base + w;
                if sc > best[e] {
                    best[e] = sc;
                }
            }
        }
        best[self.len]
    }

    /// Like `score_only_f32`, but also returns the number of tokens in the best path.
    #[inline]
    pub fn score_and_count_f32(&self, weights: &[f32], unk_id: usize, unk_score: f32) -> (f32, u32) {
        if self.len == 0 {
            return (0.0, 0);
        }
        let mut best = vec![f32::NEG_INFINITY; self.len + 1];
        let mut count = vec![0u32; self.len + 1];
        best[0] = 0.0;

        for s in 0..self.len {
            let base = best[s];
            if base == f32::NEG_INFINITY {
                continue;
            }
            let base_count = count[s];
            for &(e, id) in &self.edges[s] {
                let w = if id == unk_id { unk_score } else { unsafe { *weights.get_unchecked(id) } };
                let sc = base + w;
                if sc > best[e] {
                    best[e] = sc;
                    count[e] = base_count + 1;
                }
            }
        }
        (best[self.len], count[self.len])
    }

    /// Forward algorithm: compute log p(s|l) by marginalizing over all segmentations.
    /// Uses log-sum-exp instead of max, giving the total probability of the string
    /// under the language model rather than just the best segmentation's probability.
    #[inline]
    pub fn forward_score_f32(&self, weights: &[f32], unk_id: usize, unk_score: f32) -> f32 {
        if self.len == 0 {
            return 0.0;
        }
        let mut alpha = vec![f32::NEG_INFINITY; self.len + 1];
        alpha[0] = 0.0;

        for s in 0..self.len {
            let base = alpha[s];
            if base == f32::NEG_INFINITY {
                continue;
            }
            for &(e, id) in &self.edges[s] {
                let w = if id == unk_id {
                    unk_score
                } else {
                    unsafe { *weights.get_unchecked(id) }
                };
                let sc = base + w;
                // log-sum-exp: alpha[e] = log(exp(alpha[e]) + exp(sc))
                alpha[e] = log_sum_exp_f32(alpha[e], sc);
            }
        }
        alpha[self.len]
    }

    /// Compute score and the best tokenization; fuses consecutive UNKs if requested.
    pub fn tokens_and_score_f32(
        &self,
        weights: &[f32],
        unk_id: usize,
        unk_score: f32,
        fuse_unk: bool,
    ) -> (Vec<String>, f32) {
        if self.len == 0 {
            return (Vec::new(), 0.0);
        }
        let mut best = vec![f32::NEG_INFINITY; self.len + 1];
        let mut link_start = vec![usize::MAX; self.len + 1];
        let mut link_id = vec![0usize; self.len + 1];
        best[0] = 0.0;

        for s in 0..self.len {
            let base = best[s];
            if base == f32::NEG_INFINITY {
                continue;
            }
            for &(e, id) in &self.edges[s] {
                let w = if id == unk_id { unk_score } else { unsafe { *weights.get_unchecked(id) } };
                let sc = base + w;
                if sc > best[e] {
                    best[e] = sc;
                    link_start[e] = s;
                    link_id[e] = id;
                }
            }
        }

        // Backtrace
        let mut tokens: Vec<String> = Vec::new();
        let mut agg = String::new(); // for fusing UNKs
        let mut e = self.len;
        while e > 0 {
            let s = link_start[e];
            let id = link_id[e];
            let piece = &self.sentence[s..e];
            if fuse_unk && id == unk_id {
                // accumulate and continue
                agg.insert_str(0, piece);
            } else {
                if !agg.is_empty() {
                    tokens.push(agg.clone());
                    agg.clear();
                }
                tokens.push(piece.to_string());
            }
            e = s;
        }
        if !agg.is_empty() {
            tokens.push(agg);
        }
        tokens.reverse();
        (tokens, best[self.len])
    }
}

/// A `Unigram` model to encode sentences.
pub struct Unigram {
    token_to_ids: TokenMap,
    pub(crate) vocab: Vocab,
    cache: Cache<String, Vec<String>>,
    trie: Trie<u8>,
    pub min_score: f64,
    pub(super) unk_id: Option<usize>,
    pub(super) bos_id: usize,
    pub(super) eos_id: usize,

    fuse_unk: bool,
    is_optimized: bool,
    byte_fallback: bool,

    // ---- NEW: keep per-language weights in Rust (lang-major) ----
    // Each item is a dense vector of length vocab.len(), f32 to cut bandwidth.
    cached_weight_sets: Option<Vec<Box<[f32]>>>,
}
impl PartialEq for Unigram {
    fn eq(&self, other: &Self) -> bool {
        self.unk_id == other.unk_id && self.vocab == other.vocab
    }
}

impl Clone for Unigram {
    // `Clone` can't be derive because it's not implemented for `Cache`.
    // To keep things simple when we clone, the new Unigram will start with a fresh cache.
    fn clone(&self) -> Self {
        let fresh_cache = self.cache.fresh();
        Self {
            vocab: self.vocab.clone(),
            cache: fresh_cache,
            token_to_ids: self.token_to_ids.clone(),
            trie: self.trie.clone(),
            min_score: self.min_score,
            unk_id: self.unk_id,
            bos_id: self.bos_id,
            eos_id: self.eos_id,
            fuse_unk: self.fuse_unk,
            is_optimized: self.is_optimized,
            byte_fallback: self.byte_fallback,
            cached_weight_sets: self
                .cached_weight_sets
                .as_ref()
                .map(|v| v.iter().map(|x| x.clone()).collect()),
        }
    }
}

impl std::fmt::Debug for Unigram {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        fmt.debug_struct("Unigram")
            .field("vocab", &self.vocab.len())
            .field("unk_id", &self.unk_id)
            .field("byte_fallback", &self.byte_fallback)
            .finish()
    }
}

static K_UNK_PENALTY: f64 = 10.0;

#[derive(thiserror::Error, Debug)]
pub enum UnigramError {
    #[error("The vocabulary is empty but at least <unk> is needed")]
    EmptyVocabulary,
    #[error("The `unk_id` is larger than vocabulary size")]
    UnkIdNotInVocabulary,
    #[error("Encountered an unknown token but `unk_id` is missing")]
    MissingUnkId,
    #[error("Weights length mismatch: expected {expected}, got {got}")]
    MismatchedWeightLength { expected: usize, got: usize },
    #[error("No cached weight sets have been provided")]
    NoCachedWeights,
    #[error("Weight set index {index} is out of range: only {num_sets} cached weight sets are available")]
    WeightSetIndexOutOfRange { index: usize, num_sets: usize },
    #[error("Batch length mismatch: {sentences} sentences but {indices} indices")]
    MismatchedBatchLength { sentences: usize, indices: usize },
}

impl Default for Unigram {
    fn default() -> Self {
        let vocab = vec![("<unk>".to_string(), 0.0)];
        Self::from(vocab, Some(0), false).unwrap()
    }
}

impl Unigram {
    /// Create a `Unigram` model from a given vocabulary.
    /// Vocabulary are the various tokens and their associated score which is a sort of a logprob of
    /// their frequency, which will enable tokenization and sampling.
    /// unk_id, is the index within the vocabulary.
    /// For now `Unigram` *requires* at least `unk` because we might find a never seen char.
    /// Further versions might allow that part to be hidden.
    pub fn from(
        vocab: Vec<(String, f64)>,
        unk_id: Option<usize>,
        byte_fallback: bool,
    ) -> Result<Self> {
        let n = vocab.len();
        let mut token_to_ids: TokenMap = AHashMap::new();
        let mut builder = TrieBuilder::default();

        if let Some(unk_id) = unk_id {
            if vocab.is_empty() {
                return Err(Box::new(UnigramError::EmptyVocabulary));
            }
            if unk_id >= vocab.len() {
                return Err(Box::new(UnigramError::UnkIdNotInVocabulary));
            }
        }
        let bos_id = n + 1;
        let eos_id = n + 2;

        let mut min_score = f64::INFINITY;
        for (id, (token, score)) in vocab.iter().enumerate() {
            token_to_ids.insert(token.to_string(), id as u32);
            builder.push(token.as_bytes());
            if score < &min_score {
                min_score = *score;
            }
        }
        let trie = builder.build();
        let fuse_unk = true;
        let is_optimized = true;

        Ok(Self {
            vocab,
            token_to_ids,
            trie,
            min_score,
            bos_id,
            eos_id,
            unk_id,
            fuse_unk,
            cache: Cache::default(),
            is_optimized,
            byte_fallback,
            cached_weight_sets: None,
        })
    }

    #[cfg(test)]
    pub(super) fn set_fuse_unk(&mut self, fuse_unk: bool) {
        self.fuse_unk = fuse_unk;
        self.cache = self.cache.fresh();
    }

    #[cfg(test)]
    pub(super) fn set_optimized(&mut self, is_optimized: bool) {
        self.is_optimized = is_optimized;
    }
    pub fn byte_fallback(&self) -> bool {
        self.byte_fallback
    }
    pub(super) fn len(&self) -> usize {
        self.vocab.len()
    }

    pub(super) fn populate_nodes(&self, lattice: &mut Lattice) {
        let unk_score = self.min_score - K_UNK_PENALTY;

        let len = lattice.len();

        let mut begin_pos = 0;
        while begin_pos < len {
            let mblen = lattice.sentence[begin_pos..]
                .chars()
                .next()
                .unwrap()
                .len_utf8();

            let mut has_single_node = false;

            for bytes in self
                .trie
                .common_prefix_search(lattice.sentence.bytes().skip(begin_pos))
            {
                let n = bytes.len();
                let tok = String::from_utf8(bytes).unwrap();
                let id = *self.token_to_ids.get(&tok).unwrap();

                let item = &self.vocab[id as usize];
                assert_eq!(item.0, tok);
                let score: f64 = item.1;
                lattice.insert(begin_pos, n, score, id.try_into().unwrap());
                if !has_single_node && n == mblen {
                    has_single_node = true;
                }
            }

            if !has_single_node {
                if let Some(unk_id) = self.unk_id {
                    lattice.insert(begin_pos, mblen, unk_score, unk_id);
                }
            }
            begin_pos += mblen
        }
    }

    /// This functions take a String, and will encode it in a Vec of Strings,
    /// of the best tokenization available to the current model.
    /// ```
    /// use tokenizers::models::unigram::Unigram;
    ///
    /// let pieces = vec![
    ///     ("<unk>".to_string(), 0.0),
    ///     ("a".to_string(), 0.0),
    ///     ("b".to_string(), 0.0),
    ///     ("c".to_string(), 0.0),
    ///     ("d".to_string(), 0.0),
    ///     ("cd".to_string(), 1.0),
    ///     ("ab".to_string(), 2.0),
    ///     ("abc".to_string(), 5.0),
    ///     ("abcd".to_string(), 10.0),
    /// ];
    /// let model = Unigram::from(pieces, Some(0), false).unwrap();
    /// let result = model.encode("abcdacdxx").unwrap();
    /// assert_eq!(result, vec!["abcd", "a", "cd", "xx"]);
    /// ```
    pub fn encode(&self, sentence: &str) -> Result<Vec<String>> {
        if sentence.is_empty() {
            return Ok(vec![]);
        }
        if let Some(result) = self.cache.get(sentence) {
            Ok(result.to_vec())
        } else {
            let result = if self.is_optimized {
                self.encode_optimized(sentence)?
            } else {
                self.encode_unoptimized(sentence)?
            };
            if sentence.len() < MAX_LENGTH {
                self.cache.set(sentence.to_owned(), result.clone());
            }
            Ok(result)
        }
    }

    fn encode_optimized(&self, sentence: &str) -> Result<Vec<String>> {
        // Keep the original optimized single-pass for the common path.
        #[derive(Debug, Clone)]
        struct BestPathNode {
            /// The vocab id. (maybe UNK)
            id: usize,
            /// The total score of the best path ending at this node.
            best_path_score: f64,
            /// The starting position (in utf-8) of this node. The entire best
            /// path can be constructed by backtracking along this link.
            starts_at: Option<usize>,
        }
        impl Default for BestPathNode {
            fn default() -> Self {
                Self {
                    id: 0,
                    best_path_score: 0.0,
                    starts_at: None,
                }
            }
        }
        let size = sentence.len();
        let unk_score = self.min_score - K_UNK_PENALTY;

        let mut best_path_ends_at = vec![BestPathNode::default(); size + 1];
        let mut starts_at = 0;
        while starts_at < size {
            let best_path_score_till_here = best_path_ends_at[starts_at].best_path_score;
            let mut has_single_node = false;
            let mblen = sentence[starts_at..].chars().next().unwrap().len_utf8();
            for tok_bytes in self
                .trie
                .common_prefix_search(sentence.bytes().skip(starts_at))
            {
                let key_pos = starts_at + tok_bytes.len();
                let token: String = String::from_utf8(tok_bytes).unwrap();
                let target_node = &mut best_path_ends_at[key_pos];
                let length = key_pos - starts_at;
                let id = self.token_to_ids.get(&token).unwrap();
                let score = self.vocab.get(*id as usize).unwrap().1;
                let candidate_best_path_score = score + best_path_score_till_here;
                if target_node.starts_at.is_none()
                    || candidate_best_path_score > target_node.best_path_score
                {
                    target_node.best_path_score = candidate_best_path_score;
                    target_node.starts_at = Some(starts_at);
                    target_node.id = *id as usize;
                }
                if !has_single_node && length == mblen {
                    has_single_node = true;
                }
            }
            if !has_single_node {
                let target_node = &mut best_path_ends_at[starts_at + mblen];
                let candidate_best_path_score = unk_score + best_path_score_till_here;
                if target_node.starts_at.is_none()
                    || candidate_best_path_score > target_node.best_path_score
                {
                    target_node.best_path_score = candidate_best_path_score;
                    target_node.starts_at = Some(starts_at);
                    target_node.id = self.unk_id.ok_or(UnigramError::MissingUnkId)?;
                }
            }
            starts_at += mblen
        }
        let mut ends_at = size;
        let mut results: Vec<String> = vec![];
        let mut token = vec![];
        while ends_at > 0 {
            let node = &best_path_ends_at[ends_at];
            let starts_at = node.starts_at.unwrap();
            if self.fuse_unk && Some(node.id) == self.unk_id {
                token.push(sentence[starts_at..ends_at].to_string());
            } else {
                if !token.is_empty() {
                    token.reverse();
                    results.push(token.concat());
                    token = vec![];
                }
                results.push(sentence[starts_at..ends_at].to_string());
            }
            ends_at = starts_at;
        }
        if !token.is_empty() {
            token.reverse();
            results.push(token.concat());
        }
        results.reverse();
        Ok(results)
    }

    fn encode_unoptimized(&self, sentence: &str) -> Result<Vec<String>> {
        let mut lattice = Lattice::from(sentence, self.bos_id, self.eos_id);
        self.populate_nodes(&mut lattice);
        if self.fuse_unk {
            let mut results = vec![];
            let mut token = String::new();
            for node in lattice.viterbi().iter() {
                let item = lattice.piece(&node.borrow());
                if node.borrow().id == self.unk_id.ok_or(UnigramError::MissingUnkId)? {
                    token.push_str(&item);
                } else {
                    if !token.is_empty() {
                        results.push(token);
                        token = String::new();
                    }
                    results.push(item);
                }
            }
            if !token.is_empty() {
                results.push(token);
            }
            Ok(results)
        } else {
            Ok(lattice.tokens())
        }
    }

    // ---------------- NEW: DP preparation & multi-weight APIs ----------------

    /// Build the forward graph (edges) once for this sentence.
    fn prepare_dp<'a>(&'a self, sentence: &'a str) -> PreparedDP<'a> {
        let size = sentence.len();
        if size == 0 {
            return PreparedDP { sentence, len: 0, edges: vec![Vec::new()] };
        }

        let mut edges: Vec<Vec<(usize, usize)>> = vec![Vec::new(); size + 1];
        let mut s = 0usize;
        while s < size {
            let mblen = sentence[s..].chars().next().unwrap().len_utf8();
            let mut has_single_node = false;

            for tok_bytes in self.trie.common_prefix_search(sentence.bytes().skip(s)) {
                let n = tok_bytes.len();
                let end = s + n;
                let tok = unsafe { String::from_utf8_unchecked(tok_bytes) };
                if let Some(&id_u32) = self.token_to_ids.get(&tok) {
                    let id = id_u32 as usize;
                    if !has_single_node && n == mblen {
                        has_single_node = true;
                    }
                    edges[s].push((end, id));
                }
            }
            if !has_single_node {
                if let Some(unk_id) = self.unk_id {
                    edges[s].push((s + mblen, unk_id));
                }
            }
            s += mblen;
        }

        PreparedDP { sentence, len: size, edges }
    }

    /// Keep (or replace) language weight sets in Rust (converted to f32).
    pub fn set_weight_sets(&mut self, sets: Vec<Vec<f64>>) -> Result<()> {
        if sets.is_empty() {
            self.cached_weight_sets = Some(Vec::new());
            return Ok(());
        }
        let v = self.vocab.len();
        if !sets.iter().all(|w| w.len() == v) {
            return Err(Box::new(UnigramError::MismatchedWeightLength {
                expected: v,
                got: sets[0].len(),
            }));
        }
        let packed: Vec<Box<[f32]>> = sets
            .into_iter()
            .map(|w| w.into_iter().map(|x| x as f32).collect::<Vec<_>>().into_boxed_slice())
            .collect();
        self.cached_weight_sets = Some(packed);
        Ok(())
    }

    /// Keep (or replace) language weight sets already in f32 (zero-conversion path).
    pub fn set_weight_sets_f32(&mut self, sets: Vec<Box<[f32]>>) -> Result<()> {
        if sets.is_empty() {
            self.cached_weight_sets = Some(Vec::new());
            return Ok(());
        }
        let v = self.vocab.len();
        if !sets.iter().all(|w| w.len() == v) {
            return Err(Box::new(UnigramError::MismatchedWeightLength {
                expected: v,
                got: sets[0].len(),
            }));
        }
        self.cached_weight_sets = Some(sets);
        Ok(())
    }

    pub fn clear_weight_sets(&mut self) {
        self.cached_weight_sets = None;
    }

    /// Like your earlier `best_of_weight_sets`, but **no FFI copying**:
    /// Uses the cached weights and the prepared graph for this sentence.
    /// Returns (winner_index, tokens, score_f32).
    pub fn best_of_cached_weight_sets(&self, sentence: &str) -> Result<(usize, Vec<String>, f32)> {
        let sets = self
            .cached_weight_sets
            .as_ref()
            .ok_or_else(|| Box::new(UnigramError::NoCachedWeights) as Box<dyn std::error::Error + Send + Sync>)?;

        if sentence.is_empty() {
            return Ok((0, Vec::new(), 0.0));
        }
        let unk_id = self.unk_id.ok_or(UnigramError::MissingUnkId)?;
        let unk_score = (self.min_score - K_UNK_PENALTY) as f32;

        let prep = self.prepare_dp(sentence);

        // Pass 1: score only
        let mut best_i = 0usize;
        let mut best_s = f32::NEG_INFINITY;
        for (i, ws) in sets.iter().enumerate() {
            let s = prep.score_only_f32(ws, unk_id, unk_score);
            if s > best_s {
                best_s = s;
                best_i = i;
            }
        }
        // Pass 2: tokens for the winner
        let (tokens, score) = if sets.is_empty() {
            // no weights => fall back to model's own scores (not typical)
            let (t, sc) = prep.tokens_and_score_f32(&[], unk_id, unk_score, self.fuse_unk);
            (t, sc)
        } else {
            prep.tokens_and_score_f32(&sets[best_i], unk_id, unk_score, self.fuse_unk)
        };
        Ok((best_i, tokens, score))
    }

    /// Batch version of best_of_cached_weight_sets using Rayon for parallelism.
    /// Processes multiple sentences in parallel while sharing cached weights (no extra RAM).
    /// Returns Vec of (winner_index, tokens, score_f32) for each sentence.
    pub fn best_of_cached_weight_sets_batch(&self, sentences: &[String]) -> Result<Vec<(usize, Vec<String>, f32)>> {
        use rayon::prelude::*;

        // Validate weights are cached
        self.cached_weight_sets
            .as_ref()
            .ok_or_else(|| Box::new(UnigramError::NoCachedWeights) as Box<dyn std::error::Error + Send + Sync>)?;

        sentences.par_iter()
            .map(|sentence| self.best_of_cached_weight_sets(sentence.as_str()))
            .collect()
    }

    /// Top-k languages by raw summed score (no bias). Returns up to k (idx, score)
    /// pairs sorted descending. Used to fit a learned per-language bias offline.
    pub fn top_k_of_cached_weight_sets(&self, sentence: &str, k: usize) -> Result<Vec<(usize, f32)>> {
        let sets = self
            .cached_weight_sets
            .as_ref()
            .ok_or_else(|| Box::new(UnigramError::NoCachedWeights) as Box<dyn std::error::Error + Send + Sync>)?;
        if sentence.is_empty() || sets.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        let unk_id = self.unk_id.ok_or(UnigramError::MissingUnkId)?;
        let unk_score = (self.min_score - K_UNK_PENALTY) as f32;
        let prep = self.prepare_dp(sentence);
        let mut scored: Vec<(usize, f32)> = sets
            .iter()
            .enumerate()
            .map(|(i, ws)| (i, prep.score_only_f32(ws, unk_id, unk_score)))
            .collect();
        let kk = k.min(scored.len());
        scored.select_nth_unstable_by(kk - 1, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(kk);
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored)
    }

    /// Batch version of top_k_of_cached_weight_sets using Rayon.
    pub fn top_k_of_cached_weight_sets_batch(&self, sentences: &[String], k: usize) -> Result<Vec<Vec<(usize, f32)>>> {
        use rayon::prelude::*;
        self.cached_weight_sets
            .as_ref()
            .ok_or_else(|| Box::new(UnigramError::NoCachedWeights) as Box<dyn std::error::Error + Send + Sync>)?;
        sentences.par_iter()
            .map(|sentence| self.top_k_of_cached_weight_sets(sentence.as_str(), k))
            .collect()
    }

    /// Like `best_of_cached_weight_sets`, but adds a per-language bias `biases[i]`
    /// (a language prior / calibration offset) to each language's summed score
    /// before the argmax. `biases` shorter than the weight sets is padded with 0.
    pub fn best_of_cached_weight_sets_biased(&self, sentence: &str, biases: &[f32]) -> Result<(usize, Vec<String>, f32)> {
        let sets = self
            .cached_weight_sets
            .as_ref()
            .ok_or_else(|| Box::new(UnigramError::NoCachedWeights) as Box<dyn std::error::Error + Send + Sync>)?;
        if sentence.is_empty() {
            return Ok((0, Vec::new(), 0.0));
        }
        let unk_id = self.unk_id.ok_or(UnigramError::MissingUnkId)?;
        let unk_score = (self.min_score - K_UNK_PENALTY) as f32;
        let prep = self.prepare_dp(sentence);
        let mut best_i = 0usize;
        let mut best_s = f32::NEG_INFINITY;
        for (i, ws) in sets.iter().enumerate() {
            let b = if i < biases.len() { biases[i] } else { 0.0 };
            let s = prep.score_only_f32(ws, unk_id, unk_score) + b;
            if s > best_s {
                best_s = s;
                best_i = i;
            }
        }
        let (tokens, score) = if sets.is_empty() {
            prep.tokens_and_score_f32(&[], unk_id, unk_score, self.fuse_unk)
        } else {
            prep.tokens_and_score_f32(&sets[best_i], unk_id, unk_score, self.fuse_unk)
        };
        Ok((best_i, tokens, score))
    }

    /// Batch version of best_of_cached_weight_sets_biased using Rayon.
    pub fn best_of_cached_weight_sets_biased_batch(&self, sentences: &[String], biases: &[f32]) -> Result<Vec<(usize, Vec<String>, f32)>> {
        use rayon::prelude::*;
        self.cached_weight_sets
            .as_ref()
            .ok_or_else(|| Box::new(UnigramError::NoCachedWeights) as Box<dyn std::error::Error + Send + Sync>)?;
        sentences.par_iter()
            .map(|sentence| self.best_of_cached_weight_sets_biased(sentence.as_str(), biases))
            .collect()
    }

    /// Like `best_of_cached_weight_sets`, but selects the winner by
    /// **length-normalized** score (score / n_tokens) instead of raw score.
    /// Returns (winner_index, tokens, normalized_score_f32).
    ///
    /// Note: n_tokens is the number of edges in the Viterbi best path (before
    /// UNK fusion), so `len(tokens) * normalized_score != raw_score` when
    /// consecutive UNKs are fused.
    pub fn best_of_cached_weight_sets_normalized(&self, sentence: &str, alpha: f32) -> Result<(usize, Vec<String>, f32)> {
        let sets = self
            .cached_weight_sets
            .as_ref()
            .ok_or_else(|| Box::new(UnigramError::NoCachedWeights) as Box<dyn std::error::Error + Send + Sync>)?;

        if sentence.is_empty() {
            return Ok((0, Vec::new(), 0.0));
        }
        let unk_id = self.unk_id.ok_or(UnigramError::MissingUnkId)?;
        let unk_score = (self.min_score - K_UNK_PENALTY) as f32;

        let prep = self.prepare_dp(sentence);

        // Pass 1: score + count, compare normalized by count^alpha
        let mut best_i = 0usize;
        let mut best_norm = f32::NEG_INFINITY;
        for (i, ws) in sets.iter().enumerate() {
            let (s, c) = prep.score_and_count_f32(ws, unk_id, unk_score);
            let norm = if c > 0 { s / (c as f32).powf(alpha) } else { f32::NEG_INFINITY };
            if norm > best_norm {
                best_norm = norm;
                best_i = i;
            }
        }
        // Pass 2: tokens for the winner
        let (tokens, _raw_score) = if sets.is_empty() {
            prep.tokens_and_score_f32(&[], unk_id, unk_score, self.fuse_unk)
        } else {
            prep.tokens_and_score_f32(&sets[best_i], unk_id, unk_score, self.fuse_unk)
        };
        Ok((best_i, tokens, best_norm))
    }

    /// Batch version of best_of_cached_weight_sets_normalized using Rayon.
    pub fn best_of_cached_weight_sets_normalized_batch(&self, sentences: &[String], alpha: f32) -> Result<Vec<(usize, Vec<String>, f32)>> {
        use rayon::prelude::*;

        self.cached_weight_sets
            .as_ref()
            .ok_or_else(|| Box::new(UnigramError::NoCachedWeights) as Box<dyn std::error::Error + Send + Sync>)?;

        sentences.par_iter()
            .map(|sentence| self.best_of_cached_weight_sets_normalized(sentence.as_str(), alpha))
            .collect()
    }

    /// Like `best_of_cached_weight_sets`, but returns the tokens and score for a
    /// caller-specified weight set `index` instead of the argmax over all cached
    /// weight sets. Useful when the winning language has already been chosen
    /// (e.g. by an external classifier) and only its segmentation is needed.
    pub fn tokens_of_cached_weight_set(&self, sentence: &str, index: usize) -> Result<(Vec<String>, f32)> {
        let sets = self
            .cached_weight_sets
            .as_ref()
            .ok_or_else(|| Box::new(UnigramError::NoCachedWeights) as Box<dyn std::error::Error + Send + Sync>)?;
        if index >= sets.len() {
            return Err(Box::new(UnigramError::WeightSetIndexOutOfRange {
                index,
                num_sets: sets.len(),
            }));
        }
        if sentence.is_empty() {
            return Ok((Vec::new(), 0.0));
        }
        let unk_id = self.unk_id.ok_or(UnigramError::MissingUnkId)?;
        let unk_score = (self.min_score - K_UNK_PENALTY) as f32;
        let prep = self.prepare_dp(sentence);
        Ok(prep.tokens_and_score_f32(&sets[index], unk_id, unk_score, self.fuse_unk))
    }

    /// Batch version of tokens_of_cached_weight_set using Rayon.
    pub fn tokens_of_cached_weight_set_batch(&self, sentences: &[String], indices: &[usize]) -> Result<Vec<(Vec<String>, f32)>> {
        use rayon::prelude::*;
        self.cached_weight_sets
            .as_ref()
            .ok_or_else(|| Box::new(UnigramError::NoCachedWeights) as Box<dyn std::error::Error + Send + Sync>)?;
        if sentences.len() != indices.len() {
            return Err(Box::new(UnigramError::MismatchedBatchLength {
                sentences: sentences.len(),
                indices: indices.len(),
            }));
        }
        sentences
            .par_iter()
            .zip(indices.par_iter())
            .map(|(sentence, &index)| self.tokens_of_cached_weight_set(sentence.as_str(), index))
            .collect()
    }

    /// Forward algorithm variant: select best language by marginalizing over all
    /// segmentations (log-sum-exp) instead of taking only the Viterbi-best.
    /// Returns (winner_index, tokens_from_viterbi, forward_score).
    pub fn best_of_cached_weight_sets_forward(&self, sentence: &str) -> Result<(usize, Vec<String>, f32)> {
        let sets = self
            .cached_weight_sets
            .as_ref()
            .ok_or_else(|| Box::new(UnigramError::NoCachedWeights) as Box<dyn std::error::Error + Send + Sync>)?;

        if sentence.is_empty() {
            return Ok((0, Vec::new(), 0.0));
        }
        let unk_id = self.unk_id.ok_or(UnigramError::MissingUnkId)?;
        let unk_score = (self.min_score - K_UNK_PENALTY) as f32;

        let prep = self.prepare_dp(sentence);

        // Pass 1: forward score (marginalize over all segmentations)
        let mut best_i = 0usize;
        let mut best_s = f32::NEG_INFINITY;
        for (i, ws) in sets.iter().enumerate() {
            let s = prep.forward_score_f32(ws, unk_id, unk_score);
            if s > best_s {
                best_s = s;
                best_i = i;
            }
        }
        // Pass 2: Viterbi tokens for the winner (for display/debugging)
        let (tokens, _viterbi_score) = if sets.is_empty() {
            prep.tokens_and_score_f32(&[], unk_id, unk_score, self.fuse_unk)
        } else {
            prep.tokens_and_score_f32(&sets[best_i], unk_id, unk_score, self.fuse_unk)
        };
        Ok((best_i, tokens, best_s))
    }

    /// Batch version of best_of_cached_weight_sets_forward using Rayon.
    pub fn best_of_cached_weight_sets_forward_batch(&self, sentences: &[String]) -> Result<Vec<(usize, Vec<String>, f32)>> {
        use rayon::prelude::*;

        self.cached_weight_sets
            .as_ref()
            .ok_or_else(|| Box::new(UnigramError::NoCachedWeights) as Box<dyn std::error::Error + Send + Sync>)?;

        sentences.par_iter()
            .map(|sentence| self.best_of_cached_weight_sets_forward(sentence.as_str()))
            .collect()
    }

    /// Keep a convenience version that takes weights via FFI (one-shot),
    /// but internally reuses the prepared DP so it's also faster than the old lattice path.
    /// Returns f64 score for backward-compat.
    pub fn best_of_weight_sets(
        &self,
        sentence: &str,
        weight_sets: &[Vec<f64>],
    ) -> Result<(usize, Vec<String>, f64)> {
        if sentence.is_empty() {
            return Ok((0, Vec::new(), 0.0));
        }
        let v = self.vocab.len();
        if !weight_sets.is_empty() && !weight_sets.iter().all(|w| w.len() == v) {
            return Err(Box::new(UnigramError::MismatchedWeightLength {
                expected: v,
                got: weight_sets[0].len(),
            }));
        }

        let unk_id = self.unk_id.ok_or(UnigramError::MissingUnkId)?;
        let unk_score = (self.min_score - K_UNK_PENALTY) as f32;
        let prep = self.prepare_dp(sentence);

        // score-only in f32
        let mut best_i = 0usize;
        let mut best_s = f32::NEG_INFINITY;
        for (i, ws64) in weight_sets.iter().enumerate() {
            let tmp: Vec<f32> = ws64.iter().map(|x| *x as f32).collect();
            let s = prep.score_only_f32(&tmp, unk_id, unk_score);
            if s > best_s {
                best_s = s;
                best_i = i;
            }
        }
        // tokens + winner
        let (tokens, score32) = if weight_sets.is_empty() {
            prep.tokens_and_score_f32(&[], unk_id, unk_score, self.fuse_unk)
        } else {
            let tmp: Vec<f32> = weight_sets[best_i].iter().map(|x| *x as f32).collect();
            prep.tokens_and_score_f32(&tmp, unk_id, unk_score, self.fuse_unk)
        };
        Ok((best_i, tokens, score32 as f64))
    }

    /// Iterate of vocabulary of the model as a pair of `(token, score)`.
    pub fn iter(&self) -> UnigramIterator<'_> {
        UnigramIterator { model: self, i: 0 }
    }

    /// Loads a SentencePiece output model after being trained by tokenizers.
    /// After that you can use the model with tokenizers library.
    /// ```no_run
    /// use tokenizers::models::unigram::Unigram;
    /// use std::path::Path;
    ///
    /// let model = Unigram::load("mymodel-unigram.json").unwrap();
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Unigram> {
        let string = read_to_string(path)?;
        Ok(serde_json::from_str(&string)?)
    }

    /// Clears the internal cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Resize the cache
    pub fn resize_cache(&mut self, capacity: usize) {
        self.cache.resize(capacity);
    }
}

/// Iterator to iterate of vocabulary of the model, and their relative score.
pub struct UnigramIterator<'a> {
    model: &'a Unigram,
    i: usize,
}

impl<'a> Iterator for UnigramIterator<'a> {
    type Item = &'a (String, f64);

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.i;
        if i < self.model.len() {
            let r = Some(&self.model.vocab[i]);
            self.i += 1;
            r
        } else {
            None
        }
    }
}

impl Model for Unigram {
    type Trainer = UnigramTrainer;

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.token_to_ids.clone().into_iter().collect()
    }

    fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn tokenize(&self, sentence: &str) -> Result<Vec<Token>> {
        let str_tokens = self.encode(sentence)?;
        let mut offset = 0;
        let mut tokens = Vec::with_capacity(str_tokens.len());
        for string in str_tokens {
            let len = string.len();
            let offsets = (offset, offset + len);
            let id: u32 = match self.token_to_ids.get(&string) {
                Some(id) => *id,
                None => {
                    if self.byte_fallback {
                        let byte_tokens: Option<Vec<_>> = string
                            .bytes()
                            .map(|byte| -> Option<Token> {
                                let byte_string = format!("<0x{byte:02X}>");
                                let id = self.token_to_ids.get(&byte_string);
                                id.map(|id| Token::new(*id, byte_string, (offset, offset + len)))
                            })
                            .collect();
                        if let Some(byte_tokens) = byte_tokens {
                            for token in byte_tokens {
                                tokens.push(token);
                            }
                            offset += len;
                            continue;
                        }
                    }
                    self.unk_id.ok_or(UnigramError::MissingUnkId)? as u32
                }
            };
            offset += len;
            tokens.push(Token::new(id, string, offsets));
        }
        Ok(tokens)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_ids.get(token).copied()
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.vocab.get(id as usize).map(|item| item.0.clone())
    }

    fn save(&self, folder: &Path, name: Option<&str>) -> Result<Vec<PathBuf>> {
        let name = match name {
            Some(name) => format!("{name}-unigram.json"),
            None => "unigram.json".to_string(),
        };
        let mut fullpath = PathBuf::new();
        fullpath.push(folder);
        fullpath.push(name);
        let string = serde_json::to_string_pretty(self)?;
        std::fs::write(&fullpath, string)?;
        Ok(vec![fullpath])
    }

    fn get_trainer(&self) -> Self::Trainer {
        UnigramTrainer::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_populate_nodes_unk() {
        let pieces = vec![("<unk>".to_string(), 0.0)];
        let model = Unigram::from(pieces, Some(0), false).unwrap();

        let mut lattice = Lattice::from("abc", model.bos_id, model.eos_id);
        model.populate_nodes(&mut lattice);

        assert_eq!(lattice.begin_nodes[0].len(), 1);
        assert_eq!(lattice.begin_nodes[1].len(), 1);
        assert_eq!(lattice.begin_nodes[2].len(), 1);
        assert_eq!(lattice.begin_nodes[0][0].borrow().id, 0);
        assert_eq!(lattice.begin_nodes[1][0].borrow().id, 0);
        assert_eq!(lattice.begin_nodes[2][0].borrow().id, 0);
        assert_eq!(lattice.begin_nodes[0][0].borrow().node_id, 2);
        assert_eq!(lattice.begin_nodes[1][0].borrow().node_id, 3);
        assert_eq!(lattice.begin_nodes[2][0].borrow().node_id, 4);
    }

    #[test]
    fn test_populate_nodes() {
        let pieces = vec![
            ("<unk>".to_string(), 0.0),
            ("a".to_string(), 0.1),
            ("b".to_string(), 0.2),
            ("ab".to_string(), 0.3),
            ("bc".to_string(), 0.4),
        ];
        let model = Unigram::from(pieces, Some(0), false).unwrap();

        let mut lattice = Lattice::from("abc", model.bos_id, model.eos_id);
        model.populate_nodes(&mut lattice);

        assert_eq!(lattice.begin_nodes[0].len(), 2); // a, ab
        assert_eq!(lattice.begin_nodes[1].len(), 2); // b, bc
        assert_eq!(lattice.begin_nodes[2].len(), 1); // c(unk)

        // Id is the vocabulary id from Unigram model
        // node_id is simply the rank of the given node in the lattice.
        assert_eq!(lattice.begin_nodes[0][0].borrow().id, 1);
        assert_eq!(lattice.begin_nodes[0][1].borrow().id, 3);
        assert_eq!(lattice.begin_nodes[1][0].borrow().id, 2);
        assert_eq!(lattice.begin_nodes[1][1].borrow().id, 4);
        assert_eq!(lattice.begin_nodes[2][0].borrow().id, 0);
        assert_eq!(lattice.begin_nodes[0][0].borrow().node_id, 2);
        assert_eq!(lattice.begin_nodes[0][1].borrow().node_id, 3);
        assert_eq!(lattice.begin_nodes[1][0].borrow().node_id, 4);
        assert_eq!(lattice.begin_nodes[1][1].borrow().node_id, 5);
        assert_eq!(lattice.begin_nodes[2][0].borrow().node_id, 6);
    }

    #[test]
    fn test_encode() {
        let sentencepieces = vec![
            ("<unk>".to_string(), 0.0),
            ("a".to_string(), 0.0),
            ("b".to_string(), 0.0),
            ("c".to_string(), 0.0),
            ("d".to_string(), 0.0),
            ("cd".to_string(), 1.0),
            ("ab".to_string(), 2.0),
            ("abc".to_string(), 5.0),
            ("abcd".to_string(), 10.0),
        ];

        let model = Unigram::from(sentencepieces, Some(0), false).unwrap();
        let result = model.encode("abcd").unwrap();
        assert_eq!(result, vec!["abcd"]);
    }

    #[test]
    fn test_unigram_bytefallback() {
        let sentencepieces = vec![
            ("<unk>".to_string(), 0.0),
            ("<0xC3>".to_string(), -0.01),
            ("<0xA9>".to_string(), -0.03),
        ];
        let unigram = Unigram::from(sentencepieces, Some(0), true).unwrap();
        let tokens = unigram.tokenize("é").unwrap();
        assert_eq!(tokens.len(), 2);
    }
}