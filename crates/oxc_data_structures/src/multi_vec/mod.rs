//! Struct-of-arrays vectors.
//!
//! The public interface is the [`multi_vec!`] macro, which generates a friendly named table type,
//! with a named struct per element, named reference / slice views, and per-field accessors.
//!
//! Internally:
//!
//! * `MultiVec`: a growable struct-of-arrays vector backed by a single allocation,
//!   indexed by a typed index. Performs all allocation, layout, and pointer arithmetic.
//! * `Fields`: trait describing a set of fields stored in a `MultiVec` - a thin
//!   typed translation layer over the field pointers that `MultiVec` computes.
//!
//! The generated table type wraps a `MultiVec`, and the macro implements `Fields` for
//! the element struct. Neither `MultiVec` nor `Fields` is public API: users interact
//! with them only through the generated table types.
//!
//! All the logic containing unsafe code lives in `MultiVec` and the `columns`, `utils`,
//! and `iter` modules, written once as ordinary generic Rust. The only unsafe code the
//! [`multi_vec!`] macro generates is the `Fields` impl - pure pointer casts
//! and calls to the `utils` helpers, with no arithmetic or control flow.

mod clone;
mod columns;
mod fields;
mod iter;
mod macros;
mod shape;
mod utils;
mod vec;

use vec::MultiVec;

pub use macros::multi_vec;

/// Not public API. Referenced by the expansion of the [`multi_vec!`] macro.
#[doc(hidden)]
pub mod __private {
    pub use std::alloc::Layout;

    pub use oxc_index::IndexSlice;
    pub use pastey::paste;

    pub use super::{
        clone::{CloneColumn, ColumnCloner, CopyColumn, SrcAndDstPtrs},
        fields::{CloneFields, Fields, SliceFields},
        iter::{IntoIter, Iter, IterMut},
        shape::Shape,
        utils::{drop_column, index_slice_from_raw_parts, index_slice_from_raw_parts_mut},
        vec::MultiVec,
    };
}

#[cfg(test)]
mod tests;
