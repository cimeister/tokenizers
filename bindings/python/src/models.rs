use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use crate::token::PyToken;
use crate::trainers::PyTrainer;
use ahash::AHashMap;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::*;
use serde::{Deserialize, Serialize};
use tk::models::bpe::{BpeBuilder, Merges, BPE};
use tk::models::unigram::Unigram;
use tk::models::wordlevel::WordLevel;
use tk::models::wordpiece::{WordPiece, WordPieceBuilder};
use tk::models::ModelWrapper;
use tk::{Model, Token};
use tokenizers as tk;

use super::error::{deprecation_warning, ToPyResult};

/// Base class for all models
#[pyclass(module = "tokenizers.models", name = "Model", subclass)]
#[derive(Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PyModel {
    pub model: Arc<RwLock<ModelWrapper>>,
}

impl PyModel {
    pub(crate) fn get_as_subtype(&self, py: Python<'_>) -> PyResult<PyObject> {
        let base = self.clone();
        Ok(match *self.model.as_ref().read().unwrap() {
            ModelWrapper::BPE(_) => Py::new(py, (PyBPE {}, base))?
                .into_pyobject(py)?
                .into_any()
                .into(),
            ModelWrapper::WordPiece(_) => Py::new(py, (PyWordPiece {}, base))?
                .into_pyobject(py)?
                .into_any()
                .into(),
            ModelWrapper::WordLevel(_) => Py::new(py, (PyWordLevel {}, base))?
                .into_pyobject(py)?
                .into_any()
                .into(),
            ModelWrapper::Unigram(_) => Py::new(py, (PyUnigram {}, base))?
                .into_pyobject(py)?
                .into_any()
                .into(),
        })
    }
}

impl Model for PyModel {
    type Trainer = PyTrainer;

    fn tokenize(&self, tokens: &str) -> tk::Result<Vec<Token>> {
        self.model.read().unwrap().tokenize(tokens)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.model.read().unwrap().token_to_id(token)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.model.read().unwrap().id_to_token(id)
    }

    fn get_vocab(&self) -> HashMap<String, u32> {
        self.model.read().unwrap().get_vocab()
    }

    fn get_vocab_size(&self) -> usize {
        self.model.read().unwrap().get_vocab_size()
    }

    fn save(&self, folder: &Path, name: Option<&str>) -> tk::Result<Vec<PathBuf>> {
        self.model.read().unwrap().save(folder, name)
    }

    fn get_trainer(&self) -> Self::Trainer {
        self.model.read().unwrap().get_trainer().into()
    }
}

impl<I> From<I> for PyModel
where
    I: Into<ModelWrapper>,
{
    fn from(model: I) -> Self {
        Self {
            model: Arc::new(RwLock::new(model.into())),
        }
    }
}

#[pymethods]
impl PyModel {
    #[new]
    #[pyo3(text_signature = None)]
    fn __new__() -> Self {
        PyModel {
            model: Arc::new(RwLock::new(BPE::default().into())),
        }
    }

    fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
        let data = serde_json::to_string(&self.model).map_err(|e| {
            exceptions::PyException::new_err(format!("Error while attempting to pickle Model: {e}"))
        })?;
        Ok(PyBytes::new(py, data.as_bytes()).into())
    }

    fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
        match state.extract::<&[u8]>(py) {
            Ok(s) => {
                self.model = serde_json::from_slice(s).map_err(|e| {
                    exceptions::PyException::new_err(format!(
                        "Error while attempting to unpickle Model: {e}"
                    ))
                })?;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Tokenize a sequence
    #[pyo3(text_signature = "(self, sequence)")]
    fn tokenize(&self, sequence: &str) -> PyResult<Vec<PyToken>> {
        Ok(ToPyResult(self.model.read().unwrap().tokenize(sequence))
            .into_py()?
            .into_iter()
            .map(|t| t.into())
            .collect())
    }

    /// Get the ID associated to a token
    #[pyo3(text_signature = "(self, tokens)")]
    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.model.read().unwrap().token_to_id(token)
    }

    /// Get the token associated to an ID
    #[pyo3(text_signature = "(self, id)")]
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.model.read().unwrap().id_to_token(id)
    }

    /// Save the current model
    #[pyo3(signature = (folder, prefix=None, name=None), text_signature = "(self, folder, prefix)")]
    fn save<'a>(
        &self,
        py: Python<'_>,
        folder: &str,
        mut prefix: Option<&'a str>,
        name: Option<&'a str>,
    ) -> PyResult<Vec<String>> {
        if name.is_some() {
            deprecation_warning(
                py,
                "0.10.0",
                "Parameter `name` of Model.save has been renamed `prefix`",
            )?;
            if prefix.is_none() {
                prefix = name;
            }
        }

        let saved: PyResult<Vec<_>> =
            ToPyResult(self.model.read().unwrap().save(Path::new(folder), prefix)).into();

        Ok(saved?
            .into_iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect())
    }

    /// Get the associated Trainer
    #[pyo3(text_signature = "(self)")]
    fn get_trainer(&self, py: Python<'_>) -> PyResult<PyObject> {
        PyTrainer::from(self.model.read().unwrap().get_trainer()).get_as_subtype(py)
    }

    fn __repr__(&self) -> PyResult<String> {
        crate::utils::serde_pyo3::repr(self)
            .map_err(|e| exceptions::PyException::new_err(e.to_string()))
    }

    fn __str__(&self) -> PyResult<String> {
        crate::utils::serde_pyo3::to_string(self)
            .map_err(|e| exceptions::PyException::new_err(e.to_string()))
    }
}

/// ===== BPE =====

#[pyclass(extends=PyModel, module = "tokenizers.models", name = "BPE")]
pub struct PyBPE {}

impl PyBPE {
    fn with_builder(
        mut builder: BpeBuilder,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<(Self, PyModel)> {
        if let Some(kwargs) = kwargs {
            for (key, value) in kwargs {
                let key: String = key.extract()?;
                match key.as_ref() {
                    "cache_capacity" => builder = builder.cache_capacity(value.extract()?),
                    "dropout" => {
                        if let Some(dropout) = value.extract()? {
                            builder = builder.dropout(dropout);
                        }
                    }
                    "unk_token" => {
                        if let Some(unk) = value.extract()? {
                            builder = builder.unk_token(unk);
                        }
                    }
                    "continuing_subword_prefix" => {
                        builder = builder.continuing_subword_prefix(value.extract()?)
                    }
                    "end_of_word_suffix" => builder = builder.end_of_word_suffix(value.extract()?),
                    "fuse_unk" => builder = builder.fuse_unk(value.extract()?),
                    "byte_fallback" => builder = builder.byte_fallback(value.extract()?),
                    "ignore_merges" => builder = builder.ignore_merges(value.extract()?),
                    _ => println!("Ignored unknown kwarg option {key}"),
                };
            }
        }

        match builder.build() {
            Err(e) => Err(exceptions::PyException::new_err(format!(
                "Error while initializing BPE: {e}"
            ))),
            Ok(bpe) => Ok((PyBPE {}, bpe.into())),
        }
    }
}

macro_rules! getter {
    ($self: ident, $variant: ident, $($name: tt)+) => {{
        let super_ = $self.as_ref();
        let model = super_.model.read().unwrap();
        if let ModelWrapper::$variant(ref mo) = *model {
            mo.$($name)+
        } else {
            unreachable!()
        }
    }};
}

macro_rules! setter {
    ($self: ident, $variant: ident, $name: ident, $value: expr) => {{
        let super_ = $self.as_ref();
        let mut model = super_.model.write().unwrap();
        if let ModelWrapper::$variant(ref mut mo) = *model {
            mo.$name = $value;
        }
    }};
}

#[derive(FromPyObject)]
enum PyVocab {
    Vocab(HashMap<String, u32>),
    Filename(String),
}

#[derive(FromPyObject)]
enum PyMerges {
    Merges(Merges),
    Filename(String),
}

#[pymethods]
impl PyBPE {
    #[getter]
    fn get_dropout(self_: PyRef<Self>) -> Option<f32> {
        getter!(self_, BPE, dropout)
    }
    #[setter]
    fn set_dropout(self_: PyRef<Self>, dropout: Option<f32>) {
        setter!(self_, BPE, dropout, dropout);
    }

    #[getter]
    fn get_unk_token(self_: PyRef<Self>) -> Option<String> {
        getter!(self_, BPE, unk_token.clone())
    }
    #[setter]
    fn set_unk_token(self_: PyRef<Self>, unk_token: Option<String>) {
        setter!(self_, BPE, unk_token, unk_token);
    }

    #[getter]
    fn get_continuing_subword_prefix(self_: PyRef<Self>) -> Option<String> {
        getter!(self_, BPE, continuing_subword_prefix.clone())
    }
    #[setter]
    fn set_continuing_subword_prefix(self_: PyRef<Self>, continuing_subword_prefix: Option<String>) {
        setter!(self_, BPE, continuing_subword_prefix, continuing_subword_prefix);
    }

    #[getter]
    fn get_end_of_word_suffix(self_: PyRef<Self>) -> Option<String> {
        getter!(self_, BPE, end_of_word_suffix.clone())
    }
    #[setter]
    fn set_end_of_word_suffix(self_: PyRef<Self>, end_of_word_suffix: Option<String>) {
        setter!(self_, BPE, end_of_word_suffix, end_of_word_suffix);
    }

    #[getter]
    fn get_fuse_unk(self_: PyRef<Self>) -> bool {
        getter!(self_, BPE, fuse_unk)
    }
    #[setter]
    fn set_fuse_unk(self_: PyRef<Self>, fuse_unk: bool) {
        setter!(self_, BPE, fuse_unk, fuse_unk);
    }

    #[getter]
    fn get_byte_fallback(self_: PyRef<Self>) -> bool {
        getter!(self_, BPE, byte_fallback)
    }
    #[setter]
    fn set_byte_fallback(self_: PyRef<Self>, byte_fallback: bool) {
        setter!(self_, BPE, byte_fallback, byte_fallback);
    }

    #[getter]
    fn get_ignore_merges(self_: PyRef<Self>) -> bool {
        getter!(self_, BPE, ignore_merges)
    }
    #[setter]
    fn set_ignore_merges(self_: PyRef<Self>, ignore_merges: bool) {
        setter!(self_, BPE, ignore_merges, ignore_merges);
    }

    #[new]
    #[pyo3(
        signature = (vocab=None, merges=None, **kwargs),
        text_signature = "(self, vocab=None, merges=None, cache_capacity=None, dropout=None, unk_token=None, continuing_subword_prefix=None, end_of_word_suffix=None, fuse_unk=None, byte_fallback=False, ignore_merges=False)"
    )]
    fn new(
        py: Python<'_>,
        vocab: Option<PyVocab>,
        merges: Option<PyMerges>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<(Self, PyModel)> {
        if (vocab.is_some() && merges.is_none()) || (vocab.is_none() && merges.is_some()) {
            return Err(exceptions::PyValueError::new_err(
                "`vocab` and `merges` must be both specified",
            ));
        }

        let mut builder = BPE::builder();
        if let (Some(vocab), Some(merges)) = (vocab, merges) {
            match (vocab, merges) {
                (PyVocab::Vocab(vocab), PyMerges::Merges(merges)) => {
                    let vocab: AHashMap<_, _> = vocab.into_iter().collect();
                    builder = builder.vocab_and_merges(vocab, merges);
                }
                (PyVocab::Filename(vocab_filename), PyMerges::Filename(merges_filename)) => {
                    deprecation_warning(
                        py,
                        "0.9.0",
                        "BPE.__init__ will not create from files anymore, try `BPE.from_file` instead",
                    )?;
                    builder = builder.files(vocab_filename.to_string(), merges_filename.to_string());
                }
                _ => {
                    return Err(exceptions::PyValueError::new_err(
                        "`vocab` and `merges` must be both be from memory or both filenames",
                    ));
                }
            }
        }

        PyBPE::with_builder(builder, kwargs)
    }

    #[staticmethod]
    #[pyo3(text_signature = "(self, vocab, merges)")]
    fn read_file(vocab: &str, merges: &str) -> PyResult<(HashMap<String, u32>, Merges)> {
        let (vocab, merges) = BPE::read_file(vocab, merges).map_err(|e| {
            exceptions::PyException::new_err(format!(
                "Error while reading vocab & merges files: {e}"
            ))
        })?;
        let vocab = vocab.into_iter().collect();
        Ok((vocab, merges))
    }

    #[classmethod]
    #[pyo3(signature = (vocab, merges, **kwargs))]
    #[pyo3(text_signature = "(cls, vocab, merge, **kwargs)")]
    fn from_file(
        _cls: &Bound<'_, PyType>,
        py: Python,
        vocab: &str,
        merges: &str,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<Self>> {
        let (vocab, merges) = BPE::read_file(vocab, merges).map_err(|e| {
            exceptions::PyException::new_err(format!("Error while reading BPE files: {e}"))
        })?;
        let vocab = vocab.into_iter().collect();
        Py::new(
            py,
            PyBPE::new(
                py,
                Some(PyVocab::Vocab(vocab)),
                Some(PyMerges::Merges(merges)),
                kwargs,
            )?,
        )
    }

    #[pyo3(signature = ())]
    #[pyo3(text_signature = "(self)")]
    fn _clear_cache(self_: PyRef<Self>) -> PyResult<()> {
        let super_ = self_.as_ref();
        let mut model = super_.model.write().map_err(|e| {
            exceptions::PyException::new_err(format!("Error while clearing BPE cache: {e}"))
        })?;
        model.clear_cache();
        Ok(())
    }

    #[pyo3(signature = (capacity))]
    #[pyo3(text_signature = "(self, capacity)")]
    fn _resize_cache(self_: PyRef<Self>, capacity: usize) -> PyResult<()> {
        let super_ = self_.as_ref();
        let mut model = super_.model.write().map_err(|e| {
            exceptions::PyException::new_err(format!("Error while resizing BPE cache: {e}"))
        })?;
        model.resize_cache(capacity);
        Ok(())
    }
}

/// ===== WordPiece =====

#[pyclass(extends=PyModel, module = "tokenizers.models", name = "WordPiece")]
pub struct PyWordPiece {}

impl PyWordPiece {
    fn with_builder(
        mut builder: WordPieceBuilder,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<(Self, PyModel)> {
        if let Some(kwargs) = kwargs {
            for (key, val) in kwargs {
                let key: String = key.extract()?;
                match key.as_ref() {
                    "unk_token" => {
                        builder = builder.unk_token(val.extract()?);
                    }
                    "max_input_chars_per_word" => {
                        builder = builder.max_input_chars_per_word(val.extract()?);
                    }
                    "continuing_subword_prefix" => {
                        builder = builder.continuing_subword_prefix(val.extract()?);
                    }
                    _ => println!("Ignored unknown kwargs option {key}"),
                }
            }
        }

        match builder.build() {
            Err(e) => Err(exceptions::PyException::new_err(format!(
                "Error while initializing WordPiece: {e}"
            ))),
            Ok(wordpiece) => Ok((PyWordPiece {}, wordpiece.into())),
        }
    }
}

#[pymethods]
impl PyWordPiece {
    #[getter]
    fn get_unk_token(self_: PyRef<Self>) -> String {
        getter!(self_, WordPiece, unk_token.clone())
    }
    #[setter]
    fn set_unk_token(self_: PyRef<Self>, unk_token: String) {
        setter!(self_, WordPiece, unk_token, unk_token);
    }

    #[getter]
    fn get_continuing_subword_prefix(self_: PyRef<Self>) -> String {
        getter!(self_, WordPiece, continuing_subword_prefix.clone())
    }
    #[setter]
    fn set_continuing_subword_prefix(self_: PyRef<Self>, continuing_subword_prefix: String) {
        setter!(self_, WordPiece, continuing_subword_prefix, continuing_subword_prefix);
    }

    #[getter]
    fn get_max_input_chars_per_word(self_: PyRef<Self>) -> usize {
        getter!(self_, WordPiece, max_input_chars_per_word)
    }
    #[setter]
    fn set_max_input_chars_per_word(self_: PyRef<Self>, max: usize) {
        setter!(self_, WordPiece, max_input_chars_per_word, max);
    }

    #[new]
    #[pyo3(signature = (vocab=None, **kwargs), text_signature = "(self, vocab, unk_token, max_input_chars_per_word)")]
    fn new(
        py: Python<'_>,
        vocab: Option<PyVocab>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<(Self, PyModel)> {
        let mut builder = WordPiece::builder();

        if let Some(vocab) = vocab {
            match vocab {
                PyVocab::Vocab(vocab) => {
                    let vocab: AHashMap<_, _> = vocab.into_iter().collect();
                    builder = builder.vocab(vocab);
                }
                PyVocab::Filename(vocab_filename) => {
                    deprecation_warning(
                        py,
                        "0.9.0",
                        "WordPiece.__init__ will not create from files anymore, try `WordPiece.from_file` instead",
                    )?;
                    builder = builder.files(vocab_filename.to_string());
                }
            }
        }

        PyWordPiece::with_builder(builder, kwargs)
    }

    #[staticmethod]
    #[pyo3(text_signature = "(vocab)")]
    fn read_file(vocab: &str) -> PyResult<HashMap<String, u32>> {
        let vocab = WordPiece::read_file(vocab).map_err(|e| {
            exceptions::PyException::new_err(format!("Error while reading WordPiece file: {e}"))
        })?;
        Ok(vocab.into_iter().collect())
    }

    #[classmethod]
    #[pyo3(signature = (vocab, **kwargs))]
    #[pyo3(text_signature = "(vocab, **kwargs)")]
    fn from_file(
        _cls: &Bound<'_, PyType>,
        py: Python,
        vocab: &str,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<Self>> {
        let vocab = WordPiece::read_file(vocab).map_err(|e| {
            exceptions::PyException::new_err(format!("Error while reading WordPiece file: {e}"))
        })?;
        let vocab = vocab.into_iter().collect();
        Py::new(py, PyWordPiece::new(py, Some(PyVocab::Vocab(vocab)), kwargs)?)
    }
}

/// ===== WordLevel =====

#[pyclass(extends=PyModel, module = "tokenizers.models", name = "WordLevel")]
pub struct PyWordLevel {}

#[pymethods]
impl PyWordLevel {
    #[getter]
    fn get_unk_token(self_: PyRef<Self>) -> String {
        getter!(self_, WordLevel, unk_token.clone())
    }
    #[setter]
    fn set_unk_token(self_: PyRef<Self>, unk_token: String) {
        setter!(self_, WordLevel, unk_token, unk_token);
    }

    #[new]
    #[pyo3(signature = (vocab=None, unk_token = None), text_signature = "(self, vocab, unk_token)")]
    fn new(
        py: Python<'_>,
        vocab: Option<PyVocab>,
        unk_token: Option<String>,
    ) -> PyResult<(Self, PyModel)> {
        let mut builder = WordLevel::builder();

        if let Some(vocab) = vocab {
            match vocab {
                PyVocab::Vocab(vocab) => {
                    let vocab = vocab.into_iter().collect();
                    builder = builder.vocab(vocab);
                }
                PyVocab::Filename(vocab_filename) => {
                    deprecation_warning(
                        py,
                        "0.9.0",
                        "WordLevel.__init__ will not create from files anymore, \
                            try `WordLevel.from_file` instead",
                    )?;
                    builder = builder.files(vocab_filename.to_string());
                }
            };
        }
        if let Some(unk_token) = unk_token {
            builder = builder.unk_token(unk_token);
        }

        Ok((
            PyWordLevel {},
            builder
                .build()
                .map_err(|e| exceptions::PyException::new_err(e.to_string()))?
                .into(),
        ))
    }

    #[staticmethod]
    #[pyo3(text_signature = "(vocab)")]
    fn read_file(vocab: &str) -> PyResult<HashMap<String, u32>> {
        let vocab = WordLevel::read_file(vocab).map_err(|e| {
            exceptions::PyException::new_err(format!("Error while reading WordLevel file: {e}"))
        })?;
        let vocab: HashMap<_, _> = vocab.into_iter().collect();
        Ok(vocab)
    }

    #[classmethod]
    #[pyo3(signature = (vocab, unk_token = None))]
    #[pyo3(text_signature = "(vocab, unk_token)")]
    fn from_file(
        _cls: &Bound<'_, PyType>,
        py: Python,
        vocab: &str,
        unk_token: Option<String>,
    ) -> PyResult<Py<Self>> {
        let vocab = WordLevel::read_file(vocab).map_err(|e| {
            exceptions::PyException::new_err(format!("Error while reading WordLevel file: {e}"))
        })?;
        let vocab = vocab.into_iter().collect();
        Py::new(py, PyWordLevel::new(py, Some(PyVocab::Vocab(vocab)), unk_token)?)
    }
}

/// ===== Unigram =====

#[pyclass(extends=PyModel, module = "tokenizers.models", name = "Unigram")]
pub struct PyUnigram {}

#[pymethods]
impl PyUnigram {
    #[new]
    #[pyo3(signature = (vocab=None, unk_id=None, byte_fallback=None), text_signature = "(self, vocab, unk_id, byte_fallback)")]
    fn new(
        vocab: Option<Vec<(String, f64)>>,
        unk_id: Option<usize>,
        byte_fallback: Option<bool>,
    ) -> PyResult<(Self, PyModel)> {
        match (vocab, unk_id, byte_fallback) {
            (Some(vocab), unk_id, byte_fallback) => {
                let model = Unigram::from(vocab, unk_id, byte_fallback.unwrap_or(false)).map_err(
                    |e| exceptions::PyException::new_err(format!("Error while loading Unigram: {e}")),
                )?;
                Ok((PyUnigram {}, model.into()))
            }
            (None, None, _) => Ok((PyUnigram {}, Unigram::default().into())),
            _ => Err(exceptions::PyValueError::new_err(
                "`vocab` and `unk_id` must be both specified",
            )),
        }
    }

    /// Keep the older one-shot API (weights from Python) for convenience
    #[pyo3(text_signature = "(self, text, weight_sets)")]
    fn best_of_weight_sets(
        self_: PyRef<Self>,
        text: &str,
        weight_sets: Vec<Vec<f64>>,
    ) -> PyResult<(usize, Vec<String>, f64)> {
        let super_ = self_.as_ref();
        let model_guard = super_.model.read().unwrap();
        if let ModelWrapper::Unigram(ref uni) = *model_guard {
            uni.best_of_weight_sets(text, &weight_sets).map_err(|e| {
                exceptions::PyException::new_err(format!("best_of_weight_sets failed: {e}"))
            })
        } else {
            Err(exceptions::PyException::new_err(
                "best_of_weight_sets is only available for Unigram",
            ))
        }
    }

    /// ---- NEW ---- Cache weight sets in Rust
    #[pyo3(text_signature = "(self, weight_sets)")]
    fn set_weight_sets(self_: PyRefMut<Self>, weight_sets: Vec<Vec<f64>>) -> PyResult<()> {
        let super_ = self_.as_ref();
        let mut model = super_.model.write().unwrap();
        if let ModelWrapper::Unigram(ref mut uni) = *model {
            uni.set_weight_sets(weight_sets)
                .map_err(|e| exceptions::PyException::new_err(format!("{e}")))
        } else {
            Err(exceptions::PyException::new_err(
                "set_weight_sets is only available for Unigram",
            ))
        }
    }

    #[pyo3(text_signature = "(self)")]
    fn clear_weight_sets(self_: PyRefMut<Self>) -> PyResult<()> {
        let super_ = self_.as_ref();
        let mut model = super_.model.write().unwrap();
        if let ModelWrapper::Unigram(ref mut uni) = *model {
            uni.clear_weight_sets();
            Ok(())
        } else {
            Err(exceptions::PyException::new_err(
                "clear_weight_sets is only available for Unigram",
            ))
        }
    }

    /// ---- NEW ---- Evaluate against cached weights (no FFI copying)
    #[pyo3(text_signature = "(self, text)")]
    fn best_of_cached_weight_sets(
        self_: PyRef<Self>,
        text: &str,
    ) -> PyResult<(usize, Vec<String>, f32)> {
        let super_ = self_.as_ref();
        let model_guard = super_.model.read().unwrap();
        if let ModelWrapper::Unigram(ref uni) = *model_guard {
            uni.best_of_cached_weight_sets(text)
                .map_err(|e| exceptions::PyException::new_err(format!("{e}")))
        } else {
            Err(exceptions::PyException::new_err(
                "best_of_cached_weight_sets is only available for Unigram",
            ))
        }
    }

    /// Batch version: process multiple texts in parallel using Rayon.
    /// Returns list of (winner_index, tokens, score) for each text.
    #[pyo3(text_signature = "(self, texts)")]
    fn best_of_cached_weight_sets_batch(
        self_: PyRef<Self>,
        texts: Vec<String>,
    ) -> PyResult<Vec<(usize, Vec<String>, f32)>> {
        let super_ = self_.as_ref();
        let model_guard = super_.model.read().unwrap();
        if let ModelWrapper::Unigram(ref uni) = *model_guard {
            uni.best_of_cached_weight_sets_batch(&texts)
                .map_err(|e| exceptions::PyException::new_err(format!("{e}")))
        } else {
            Err(exceptions::PyException::new_err(
                "best_of_cached_weight_sets_batch is only available for Unigram",
            ))
        }
    }

    /// Biased batch version: adds per-language bias `biases[i]` (a language prior)
    /// to each language's score before the argmax. Returns (idx, tokens, score).
    #[pyo3(text_signature = "(self, texts, biases)")]
    fn best_of_cached_weight_sets_biased_batch(
        self_: PyRef<Self>,
        texts: Vec<String>,
        biases: Vec<f32>,
    ) -> PyResult<Vec<(usize, Vec<String>, f32)>> {
        let super_ = self_.as_ref();
        let model_guard = super_.model.read().unwrap();
        if let ModelWrapper::Unigram(ref uni) = *model_guard {
            uni.best_of_cached_weight_sets_biased_batch(&texts, &biases)
                .map_err(|e| exceptions::PyException::new_err(format!("{e}")))
        } else {
            Err(exceptions::PyException::new_err(
                "best_of_cached_weight_sets_biased_batch is only available for Unigram",
            ))
        }
    }

    /// Top-k batch version: returns, per text, a list of (idx, score) for the top-k
    /// languages by raw score (no bias). Used to fit a learned per-language bias.
    #[pyo3(text_signature = "(self, texts, k)")]
    fn top_k_of_cached_weight_sets_batch(
        self_: PyRef<Self>,
        texts: Vec<String>,
        k: usize,
    ) -> PyResult<Vec<Vec<(usize, f32)>>> {
        let super_ = self_.as_ref();
        let model_guard = super_.model.read().unwrap();
        if let ModelWrapper::Unigram(ref uni) = *model_guard {
            uni.top_k_of_cached_weight_sets_batch(&texts, k)
                .map_err(|e| exceptions::PyException::new_err(format!("{e}")))
        } else {
            Err(exceptions::PyException::new_err(
                "top_k_of_cached_weight_sets_batch is only available for Unigram",
            ))
        }
    }

    /// Length-normalized version: selects winner by score/n_tokens^alpha.
    #[pyo3(signature = (text, alpha = 1.0))]
    fn best_of_cached_weight_sets_normalized(
        self_: PyRef<Self>,
        text: &str,
        alpha: f32,
    ) -> PyResult<(usize, Vec<String>, f32)> {
        let super_ = self_.as_ref();
        let model_guard = super_.model.read().unwrap();
        if let ModelWrapper::Unigram(ref uni) = *model_guard {
            uni.best_of_cached_weight_sets_normalized(text, alpha)
                .map_err(|e| exceptions::PyException::new_err(format!("{e}")))
        } else {
            Err(exceptions::PyException::new_err(
                "best_of_cached_weight_sets_normalized is only available for Unigram",
            ))
        }
    }

    /// Batch version of length-normalized scoring.
    #[pyo3(signature = (texts, alpha = 1.0))]
    fn best_of_cached_weight_sets_normalized_batch(
        self_: PyRef<Self>,
        texts: Vec<String>,
        alpha: f32,
    ) -> PyResult<Vec<(usize, Vec<String>, f32)>> {
        let super_ = self_.as_ref();
        let model_guard = super_.model.read().unwrap();
        if let ModelWrapper::Unigram(ref uni) = *model_guard {
            uni.best_of_cached_weight_sets_normalized_batch(&texts, alpha)
                .map_err(|e| exceptions::PyException::new_err(format!("{e}")))
        } else {
            Err(exceptions::PyException::new_err(
                "best_of_cached_weight_sets_normalized_batch is only available for Unigram",
            ))
        }
    }

    /// Clears the internal cache
    #[pyo3(signature = ())]
    #[pyo3(text_signature = "(self)")]
    fn _clear_cache(self_: PyRef<Self>) -> PyResult<()> {
        let super_ = self_.as_ref();
        let mut model = super_.model.write().map_err(|e| {
            exceptions::PyException::new_err(format!("Error while clearing Unigram cache: {e}"))
        })?;
        model.clear_cache();
        Ok(())
    }

    /// Resize the internal cache
    #[pyo3(signature = (capacity))]
    #[pyo3(text_signature = "(self, capacity)")]
    fn _resize_cache(self_: PyRef<Self>, capacity: usize) -> PyResult<()> {
        let super_ = self_.as_ref();
        let mut model = super_.model.write().map_err(|e| {
            exceptions::PyException::new_err(format!("Error while resizing Unigram cache: {e}"))
        })?;
        model.resize_cache(capacity);
        Ok(())
    }
}

/// Models Module
#[pymodule]
pub fn models(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    m.add_class::<PyBPE>()?;
    m.add_class::<PyWordPiece>()?;
    m.add_class::<PyWordLevel>()?;
    m.add_class::<PyUnigram>()?;
    Ok(())
}

#[cfg(test)]
mod test {
    use crate::models::PyModel;
    use pyo3::prelude::*;
    use tk::models::bpe::BPE;
    use tk::models::ModelWrapper;

    #[test]
    fn get_subtype() {
        Python::with_gil(|py| {
            let py_model = PyModel::from(BPE::default());
            let py_bpe = py_model.get_as_subtype(py).unwrap();
            assert_eq!("BPE", py_bpe.bind(py).get_type().qualname().unwrap());
        })
    }

    #[test]
    fn serialize() {
        let rs_bpe = BPE::default();
        let rs_bpe_ser = serde_json::to_string(&rs_bpe).unwrap();
        let rs_wrapper: ModelWrapper = rs_bpe.into();
        let rs_wrapper_ser = serde_json::to_string(&rs_wrapper).unwrap();

        let py_model = PyModel::from(rs_wrapper);
        let py_ser = serde_json::to_string(&py_model).unwrap();
        assert_eq!(py_ser, rs_bpe_ser);
        assert_eq!(py_ser, rs_wrapper_ser);

        let py_model: PyModel = serde_json::from_str(&rs_bpe_ser).unwrap();
        match *py_model.model.as_ref().read().unwrap() {
            ModelWrapper::BPE(_) => (),
            _ => panic!("Expected BPE."),
        };

        let py_model: PyModel = serde_json::from_str(&rs_wrapper_ser).unwrap();
        match *py_model.model.as_ref().read().unwrap() {
            ModelWrapper::BPE(_) => (),
            _ => panic!("Expected BPE."),
        };
    }
}