//! Column-by-column cloning for [`MultiVec`], used by the expansion of the [`multi_vec!`] macro.
//!
//! Cloning a struct-of-arrays vector is *field-wise*: each field array (column) is
//! cloned in turn. [`ColumnCloner`] picks the strategy for a column by autoderef
//! specialization on the field type: a single bitwise `memcpy` for `Copy` types
//! ([`CopyColumn`]), element-by-element cloning for other `Clone` types ([`CloneColumn`]).
//!
//! The only caller is [`CloneFields::clone_columns`]: the [`multi_vec!`] macro
//! implements it - one `clone_column` call per field - only for tables declared with
//! `#[derive(Clone)]`. Tables without the derive use nothing in this module.
//!
//! Cloning is panic-safe, via [`ColumnDropGuard`]s at two levels: if a value's `clone`
//! panics, the guard inside [`CloneColumn::clone_column`] drops the values already
//! cloned into the current column, and the guards held by `clone_columns` drop every
//! earlier, fully-cloned column. Nothing is leaked.
//!
//! [`MultiVec`]: super::MultiVec
//! [`multi_vec!`]: super::multi_vec
//! [`CloneFields::clone_columns`]: super::fields::CloneFields::clone_columns

#![expect(
    rustdoc::private_intra_doc_links,
    reason = "items are only reachable via `#[doc(hidden)]`, so links to private items are \
    only ever seen in internal docs (`--document-private-items`), where they resolve"
)]

use std::{
    marker::PhantomData,
    mem::{MaybeUninit, needs_drop},
    ptr::{self, NonNull},
    slice,
};

/// One column's source and destination pointers.
///
/// [`CloneFields::clone_columns`] takes one per column, zipped into a single array,
/// and passes each on to its column's `clone_column` call.
///
/// A single array (rather than separate `src` and `dst` arrays) lets the macro-generated
/// `clone_columns` destructure it with one binding per field, like every other
/// [`Fields`] method - two arrays would need two `paste!`-derived binding names per field.
///
/// [`CloneFields::clone_columns`]: super::fields::CloneFields::clone_columns
/// [`Fields`]: super::fields::Fields
#[derive(Clone, Copy)]
pub struct SrcAndDstPtrs {
    /// Pointer to the start of the source column.
    pub src_ptr: NonNull<u8>,
    /// Pointer to the start of the destination column.
    pub dst_ptr: NonNull<u8>,
}

/// Dispatcher for cloning one field array (column) of a [`MultiVec`].
///
/// [`ColumnCloner`] uses *autoderef specialization*: the [`CopyColumn`] and [`CloneColumn`]
/// traits both have a `clone_column` method, implemented for `&&ColumnCloner<T>` and
/// `&ColumnCloner<T>` respectively. A call `(&&ColumnCloner::<T>::NEW).clone_column(...)`
/// (with `T` a concrete type, as it always is in the code the [`multi_vec!`] macro generates)
/// resolves to the first applicable implementation, dereferencing the receiver one step at a time:
///
/// 1. `T: Copy` - [`CopyColumn`]: single bitwise copy (guaranteed `memcpy`).
/// 2. `T: Clone` - [`CloneColumn`]: clone the values one by one.
///
/// The 2 traits must not be merged, and the reference levels of their implementations
/// must not be changed - both would break this resolution order.
///
/// `clone_column` calls appear only in [`CloneFields::clone_columns`], which the
/// [`multi_vec!`] macro generates only for tables declared with `#[derive(Clone)]`, whose
/// field types must all be `Clone` - so one of the two tiers always applies. (A
/// non-`Clone` field type in such a table is a compile-time error at the macro
/// expansion.)
///
/// [`MultiVec`]: super::MultiVec
/// [`multi_vec!`]: super::multi_vec
/// [`CloneFields::clone_columns`]: super::fields::CloneFields::clone_columns
pub struct ColumnCloner<T>(PhantomData<T>);

impl<T> ColumnCloner<T> {
    /// A `ColumnCloner` for column type `T`.
    pub const NEW: Self = Self(PhantomData);
}

/// Tier 1 of [`ColumnCloner`]'s autoderef specialization: `Copy` types, copied bitwise.
pub trait CopyColumn {
    /// Copy the first `len` values of a field array from its source column to its
    /// destination column, bitwise.
    ///
    /// Returns `()`, unlike [`CloneColumn::clone_column`] which returns a [`ColumnDropGuard`].
    /// `Copy` types can never need dropping, so the copied column requires no cleanup
    /// even if a later column's `clone` panics.
    /// The caller treats both uniformly - `mem::forget(())` is a no-op.
    ///
    /// # SAFETY
    ///
    /// * `src_ptr` must be aligned for `T`, and valid for reads of `len` consecutive
    ///   values of `T`, all initialized.
    /// * `dst_ptr` must be aligned for `T`, and valid for writes of `len` consecutive
    ///   values of `T`.
    /// * The two ranges must not overlap.
    /// * Dangling but aligned pointers are sufficient if `len == 0` or `T` is zero-sized.
    unsafe fn clone_column(self, src_and_dst_ptrs: SrcAndDstPtrs, len: usize);
}

impl<T: Copy> CopyColumn for &&ColumnCloner<T> {
    #[inline]
    unsafe fn clone_column(self, src_and_dst_ptrs: SrcAndDstPtrs, len: usize) {
        let SrcAndDstPtrs { src_ptr, dst_ptr } = src_and_dst_ptrs;

        // SAFETY: Caller guarantees `src_ptr` is aligned and valid for reads of `len`
        // initialized values of `T`, `dst_ptr` is aligned and valid for writes of `len`
        // values of `T`, and the two ranges do not overlap. `T: Copy`, so a bitwise copy
        // is a valid clone.
        unsafe {
            ptr::copy_nonoverlapping(
                src_ptr.cast::<T>().as_ptr(),
                dst_ptr.cast::<T>().as_ptr(),
                len,
            );
        }
    }
}

/// Tier 2 of [`ColumnCloner`]'s autoderef specialization: `Clone` types, cloned one by one.
pub trait CloneColumn {
    /// The column's value type. [`ColumnCloner`]'s type parameter.
    type Value;

    /// Clone the first `len` values of a field array from its source column to its
    /// destination column.
    ///
    /// Returns a fully-armed [`ColumnDropGuard`] for the destination column.
    /// The caller must hold it until all columns are cloned, then [`mem::forget`] it
    /// (see [`ColumnDropGuard`]).
    ///
    /// If a value's `clone` panics, the values already written to the destination column
    /// are dropped (by this method's own guard) during unwinding.
    ///
    /// # SAFETY
    ///
    /// Same requirements as [`CopyColumn::clone_column`].
    ///
    /// [`mem::forget`]: std::mem::forget
    unsafe fn clone_column(
        self,
        src_and_dst_ptrs: SrcAndDstPtrs,
        len: usize,
    ) -> ColumnDropGuard<Self::Value>;
}

impl<T: Clone> CloneColumn for &ColumnCloner<T> {
    type Value = T;

    #[inline]
    unsafe fn clone_column(
        self,
        src_and_dst_ptrs: SrcAndDstPtrs,
        len: usize,
    ) -> ColumnDropGuard<T> {
        // The loop lives in an inner function taking the columns as slices. `noalias` is
        // only emitted for reference-typed function *parameters* (references created
        // mid-function carry no aliasing information), and it survives inlining as
        // scoped-alias metadata - so this shape lets LLVM prove the two slices are disjoint.
        // For trivially-cloneable types (e.g. types which could be `Copy` but only
        // implement `Clone` for API design reasons), that collapses the loop to a single
        // `memcpy`; for types with real `Clone` impls, it enables vectorization without a
        // runtime overlap check. (The guard's `initialized` updates don't defeat this:
        // when `clone` cannot panic, LLVM sinks them to a single store after the loop,
        // and the never-dropped guard then vanishes entirely.)
        #[inline]
        fn clone_into<T: Clone>(src: &[T], dst: &mut [MaybeUninit<T>], initialized: &mut usize) {
            for (value, out) in src.iter().zip(dst.iter_mut()) {
                out.write(value.clone());
                // Only incremented after the value is fully written, so if the *next*
                // value's `clone` panics, the guard drops exactly the initialized values.
                *initialized += 1;
            }
        }

        let SrcAndDstPtrs { src_ptr, dst_ptr } = src_and_dst_ptrs;

        let mut guard = ColumnDropGuard { ptr: dst_ptr.cast::<T>(), initialized: 0 };

        // SAFETY: Caller guarantees `src` is aligned and valid for reads of `len`
        // consecutive initialized values of `T`, which are not mutated while the slice lives
        // (the only writes are through `dst`, which does not overlap).
        let src = unsafe { slice::from_raw_parts(src_ptr.cast::<T>().as_ptr(), len) };

        // SAFETY: Caller guarantees `dst` is aligned and valid for reads and writes of
        // `len` consecutive values of `T`. `MaybeUninit<T>` has the same layout as `T`,
        // and is valid even for uninitialized memory. The values are not accessed through
        // any other pointer while the slice lives (`src` does not overlap, and the guard's
        // copy of the pointer is not read unless the guard is dropped, after this slice
        // is dead).
        let dst =
            unsafe { slice::from_raw_parts_mut(dst_ptr.cast::<MaybeUninit<T>>().as_ptr(), len) };

        clone_into(src, dst, &mut guard.initialized);

        // All `len` values cloned without panicking. Return the guard to caller.
        // At this point `guard.initialized == len`, so dropping the guard drops all the values.
        guard
    }
}

/// Drop guard for a partially or fully cloned column, returned by [`CloneColumn::clone_column`].
///
/// If dropped, drops the first `initialized` values of the column.
/// This provides panic safety for cloning, at two levels:
///
/// * Within one column: `clone_column` keeps `initialized` up to date as it clones,
///   and the guard is a local variable inside it until it returns - so if a value's
///   `clone` panics, unwinding drops the values already cloned into this column.
///
/// * Across columns: The caller (macro-generated [`CloneFields::clone_columns`]) holds
///   the returned guards (each fully armed, `initialized == len`) in a tuple until *all*
///   columns are cloned, and only then [`mem::forget`]s the tuple. So if a later column's
///   `clone` panics mid-way through building the tuple, unwinding drops the already-built
///   guards - and with them, every earlier column in full.
///
/// On the success path the guards are forgotten, and for types whose `clone` cannot panic
/// (no unwind edges), the compiler removes the guard machinery entirely.
///
/// [`mem::forget`]: std::mem::forget
/// [`CloneFields::clone_columns`]: super::fields::CloneFields::clone_columns
pub struct ColumnDropGuard<T> {
    /// Pointer to the start of the column.
    ptr: NonNull<T>,
    /// Number of values at the start of the column which are initialized.
    initialized: usize,
}

impl<T> Drop for ColumnDropGuard<T> {
    // This method is shaped to keep the guard cheap on the success path, and landing pads
    // small on the panic path:
    //
    // * `#[inline(always)]`, with the actual dropping outlined in `drop_column_prefix`,
    //   which is passed `ptr` and `initialized` *by value*. After inlining, the guard's
    //   address never escapes, so the guard's fields live in registers
    //   (`clone_column`'s `*initialized += 1` is a register increment, not a memory write),
    //   and each landing pad is just "load 2 registers, call, resume".
    // * `drop_column_prefix` is `#[inline(never)]`, so the drop loop can never be
    //   inlined into (and bloat) landing pads, and `#[cold]` (it only runs during unwinding),
    //   placing it in the cold section.
    // * The `const` `needs_drop` guard is needed *because* of the outlining.
    //   `#[inline(never)]` means the compiler cannot see into `drop_column_prefix` to know that
    //   dropping is a no-op when `T` is not `Drop`. This check states that fact at the call site,
    //   so for such types `drop` compiles to nothing (no call, no landing pad).
    #[expect(clippy::inline_always, reason = "trivial: a compile-time check + a call")]
    #[inline(always)]
    fn drop(&mut self) {
        if const { needs_drop::<T>() } {
            // SAFETY: `ptr` is aligned and valid for reads and writes of `initialized`
            // initialized values of `T` (upheld by `clone_column`, which creates the
            // guard with `initialized = 0` and increments it only after each value is written).
            // The values are never used again - the guard is only dropped when cloning panicked,
            // and the caller's partially-built `MultiVec` (with `len == 0`) does not read them.
            unsafe { drop_column_prefix(self.ptr, self.initialized) };
        }
    }
}

/// Outlined cleanup for [`ColumnDropGuard`]: drop the first `initialized` values of the column
/// starting at `ptr`.
///
/// Only called during unwinding, from landing pads - see [`ColumnDropGuard`]'s `Drop` impl
/// for why it is `#[cold]` and `#[inline(never)]`, and takes its arguments by value.
///
/// # SAFETY
///
/// * `ptr` must be aligned for `T`, and valid for reads and writes of `initialized`
///   consecutive values of `T`, all initialized.
/// * The values must not be used after this call.
#[cold]
#[inline(never)]
unsafe fn drop_column_prefix<T>(ptr: NonNull<T>, initialized: usize) {
    // SAFETY: Caller guarantees `ptr` is aligned and valid for reads and writes of
    // `initialized` initialized values of `T`, which are never used again
    unsafe {
        ptr::drop_in_place(ptr::slice_from_raw_parts_mut(ptr.as_ptr(), initialized));
    }
}
