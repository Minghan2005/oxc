//! [`MultiVec`]: a growable struct-of-arrays vector backed by a single allocation,
//! indexed by a typed index.
//!
//! The allocation and its layout maths live in [`Columns`] (see the `columns` module for
//! the allocation layout). Field arrays are stored in descending alignment order, which
//! packs them with no padding. This makes all offsets simple multiples of compile-time
//! constants: hot paths use plain unchecked arithmetic, justified by overflow checks
//! performed once, at allocation time (in the cold grow / clone paths).

#![expect(
    rustdoc::private_intra_doc_links,
    reason = "items are only reachable via `#[doc(hidden)]`, so links to private items are \
    only ever seen in internal docs (`--document-private-items`), where they resolve"
)]

use std::{
    fmt::{self, Debug},
    iter::FusedIterator,
    marker::PhantomData,
};

use oxc_index::Idx;

use crate::assert_unchecked;

use super::{
    clone::SrcAndDstPtrs,
    columns::Columns,
    fields::{CloneFields, Fields, SliceFields},
    iter::{IntoIter, Iter, IterMut},
    shape::CopyArray,
};

/// Minimum default capacity for a [`MultiVec`].
///
/// This is minimum capacity when the `MultiVec` grows due to [`push`] or [`reserve`].
/// [`with_capacity`] can request (and receive) a `MultiVec` with smaller capacity.
///
/// [`push`]: MultiVec::push
/// [`reserve`]: MultiVec::reserve
/// [`with_capacity`]: MultiVec::with_capacity
const MIN_CAPACITY: usize = 4;

/// A growable struct-of-arrays vector, backed by a single allocation, indexed by
/// a typed index `I`.
///
/// Instead of N separate `IndexVec`s each with their own `ptr + len + capacity`,
/// `MultiVec` stores one array (column) per field of `F` within a single allocation,
/// with a single `len` and `capacity`.
///
/// Usually used via the [`multi_vec!`] macro, which generates a named wrapper around
/// a `MultiVec`, and the [`Fields`] impl that `MultiVec` requires.
///
/// The columns themselves - the allocation, its layout, and the pointer arithmetic to
/// address any field of any element - are described by [`Columns`] (see its docs for the
/// allocation layout). `MultiVec` wraps a `Columns` with *ownership* (of the elements and
/// the allocation) and the typed index `I`. [`Fields`] (usually implemented by the
/// [`multi_vec!`] macro) is only a thin typed translation layer over the field pointers
/// that `Columns` computes.
///
/// # Invariants
///
/// * `columns`' invariants hold (see [`Columns`]) - they are geometry only.
/// * `columns.capacity <= Self::MAX_CAPACITY` (tightens `Columns`' allocation-size limit
///   with the index type's range).
/// * The first `len` elements of every column are initialized. (This is `MultiVec`'s
///   refinement of `Columns`' contents-agnostic invariants.)
///
/// Field types may need dropping (e.g. `String`): [`Drop`] drops the first `len` elements
/// of every column before freeing the allocation. Note that elements only ever exist
/// as scattered field values, so `F`'s own `Drop` impl (if it has one) is never invoked -
/// only the field values are dropped, individually.
///
/// [`new`] is the only constructor, and trivially establishes the invariants ([`Columns::empty`]).
/// The invariants are preserved by every method:
///
/// * [`grow`] is the only method which changes `capacity` or `base_ptr`: it panics unless
///   the new capacity is `<= MAX_CAPACITY`, allocates with `layout_for(new_capacity)`, moves the
///   `len` initialized elements to the new allocation (a bitwise copy, after which the old
///   copies are never used again), and only then updates `base_ptr` and `capacity` - so
///   `self` is untouched if allocation fails or panics.
/// * [`push`] is the only method which increases `len`: it initializes element `len` in
///   every column before incrementing `len`.
/// * [`clone`] builds a fresh `MultiVec`: it sets the clone's `len` only after [`CloneFields::clone_columns`]
///   has initialized all its elements (guaranteed by [`CloneFields`]' contract).
/// * [`into_iter`] transfers ownership of the elements and the allocation to an [`IntoIter`]
///   (which snapshots the `Columns`), then forgets the `MultiVec` without running its `Drop`.
///   The `IntoIter` reads / drops the elements directly, and frees the allocation
///   ([`Columns::deallocate`]) when it is dropped.
///
/// Every unsafe operation's SAFETY comment argues from these invariants.
///
/// [`multi_vec!`]: super::multi_vec
/// [`Drop`]: MultiVec::drop
/// [`new`]: MultiVec::new
/// [`grow`]: MultiVec::grow
/// [`push`]: MultiVec::push
/// [`clone`]: MultiVec::clone
/// [`into_iter`]: IntoIterator::into_iter
pub struct MultiVec<I: Idx, F: Fields> {
    /// The columns: the allocation, `len`, and `capacity`.
    columns: Columns<F>,
    /// `I` is a branding parameter only - no `I` is ever stored (indices round-trip through
    /// `usize` at the API boundary). So this marker has no bearing on `Send`/`Sync` -
    /// the manual impls below decide those.
    /// It is `fn(I) -> I`, not `PhantomData<I>`, to keep `MultiVec` invariant in `I`
    /// (the correct variance for a brand used in both argument and return position -
    /// load-bearing once index types carry lifetimes), to own no `I` for drop-check,
    /// and to leave the other auto traits (`Unpin` etc.) independent of `I`.
    index_marker: PhantomData<fn(I) -> I>,
}

// SAFETY: `MultiVec` owns its data, and its API upholds Rust's aliasing rules (shared data is
// only exposed via `&self` methods, exclusive via `&mut self`). So it is `Send`/`Sync` under
// the same conditions as `Vec` of each field type: when the field types are.
//
// The bound is on `F` only, not `I`. No `I` is ever stored, so sending a `MultiVec` transfers
// no `I` across threads - indices are only built from a `usize` (`push`, `iter_ids`) or turned
// back into one (`get`), always on the calling thread. `!Send` forbids *moving* an existing
// value to another thread, not *creating* a fresh one there, and no `I` value ever crosses the
// boundary - so `I: !Send` cannot be broken. `I: Send` would be needlessly restrictive.
unsafe impl<I: Idx, F: Fields + Send> Send for MultiVec<I, F> {}

// SAFETY: See `Send` impl above
unsafe impl<I: Idx, F: Fields + Sync> Sync for MultiVec<I, F> {}

impl<I: Idx, F: Fields> MultiVec<I, F> {
    /// Maximum capacity.
    ///
    /// Capacity is limited by 2 factors:
    ///
    /// 1. All valid indices must be representable as `I`: `I::MAX + 1`.
    /// 2. Allocations cannot exceed `isize::MAX` bytes: [`F::SHAPE.max_alloc_capacity()`].
    ///
    /// The 2nd limit comes into play on 32-bit platforms (e.g. WASM), and theoretically
    /// could on 64-bit too, if field types are massive.
    ///
    /// `MAX_CAPACITY <= isize::MAX`, so `capacity * 2` in `grow` cannot overflow.
    ///
    /// The `saturating_add` is just to avoid overflow if `I::MAX == usize::MAX` -
    /// the allocation limit is necessarily far lower.
    ///
    /// [`F::SHAPE.max_alloc_capacity()`]: super::shape::Shape::max_alloc_capacity
    pub const MAX_CAPACITY: usize = min(I::MAX.saturating_add(1), F::SHAPE.max_alloc_capacity());

    /// Create a new empty `MultiVec`.
    ///
    /// Does not allocate.
    ///
    /// Field sets consisting only of zero-sized types are rejected at compile time, when
    /// the table is defined: the [`multi_vec!`] macro's [`Fields`] impl gives `F::SHAPE`
    /// a concrete type, so rustc eagerly evaluates it, and [`Shape::new`] (inside it)
    /// panics on an all-ZST field set (see the macro's docs for an example).
    ///
    /// [`multi_vec!`]: super::multi_vec
    /// [`Fields`]: super::fields::Fields
    /// [`Shape::new`]: super::shape::Shape::new
    pub const fn new() -> Self {
        Self { columns: Columns::empty(), index_marker: PhantomData }
    }

    /// Create a new `MultiVec` with capacity for `capacity` elements.
    ///
    /// Does not allocate if `capacity == 0`.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` exceeds [`MAX_CAPACITY`](Self::MAX_CAPACITY).
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return Self::new();
        }

        assert!(capacity <= Self::MAX_CAPACITY, "`capacity` exceeds maximum capacity");

        // SAFETY: `0 < capacity <= MAX_CAPACITY <= MAX_ALLOC_CAPACITY` (checked above)
        let columns = unsafe { Columns::<F>::allocate(capacity) };
        Self { columns, index_marker: PhantomData }
    }

    /// Returns the number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        let len = self.columns.len;
        // Communicate the bound on the returned value to compiler
        // (e.g. so `a.len() + b.len()` is provably overflow-free).
        // SAFETY: `len <= capacity <= MAX_CAPACITY` by `MultiVec`'s invariants.
        unsafe { assert_unchecked!(len <= Self::MAX_CAPACITY) };
        len
    }

    /// Returns `true` if there are no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.columns.len == 0
    }

    /// Returns the number of elements the `MultiVec` can hold without reallocating.
    #[inline]
    pub fn capacity(&self) -> usize {
        let capacity = self.columns.capacity;
        // Communicate the bound on the returned value to compiler.
        // SAFETY: `capacity <= MAX_CAPACITY` by `MultiVec`'s invariants.
        unsafe { assert_unchecked!(capacity <= Self::MAX_CAPACITY) };
        capacity
    }

    /// Get a clone of the [`Columns`].
    ///
    /// Used by the iterators (in the `iter` module) to snapshot the [`MultiVec`]'s state.
    #[inline]
    pub(super) fn columns(&self) -> Columns<F> {
        self.columns.clone()
    }

    /// Push a new element (splitting its fields across the columns).
    /// Returns the ID of the new element.
    ///
    /// # Panics
    ///
    /// Panics if the new length would exceed [`MAX_CAPACITY`].
    ///
    /// [`MAX_CAPACITY`]: Self::MAX_CAPACITY
    #[inline]
    pub fn push(&mut self, value: F) -> I {
        // If full, grow.
        // Growth is `#[cold] #[inline(never)]` as it's a rare event.
        //
        // Then fall through to the single shared write.
        // Keep the write on this one path, do NOT complete the push inside a cold
        // "grow, then write" continuation that takes `value`.
        // Handing `value` to a separate function stages it to the stack on every
        // iteration, and also stops the compiler holding `len` / `capacity` in registers -
        // it reloads them from memory each iteration instead.
        //
        // With the current formulation, a bulk-push loop keeps `len` / `capacity`
        // in registers (reloading only after a grow) and stores `value`'s fields
        // straight from registers into the heap, instead of writing to stack,
        // then copying to heap. `value` rides through the rare grow call
        // in callee-saved registers, so it costs no hot-path spill.
        if self.columns.len == self.columns.capacity {
            // SAFETY: Just checked `len == capacity`
            unsafe { self.grow_for_push() };
        }

        // SAFETY: If was full, `grow_for_push` above grew the allocation.
        // Either way, there's now space for `value` (`len < capacity`).
        unsafe { self.push_unchecked(value) }
    }

    /// Grow the allocation to hold at least 1 more element when the current allocation is full.
    ///
    /// # SAFETY
    ///
    /// `self.columns.len` must be equal to `self.columns.capacity`.
    ///
    /// # Panics
    ///
    /// Panics if capacity is already [`MAX_CAPACITY`].
    ///
    /// [`MAX_CAPACITY`]: Self::MAX_CAPACITY
    #[cold]
    #[inline(never)]
    unsafe fn grow_for_push(&mut self) {
        // Inform compiler that `len == capacity`.
        // If `grow` is inlined, compiler can treat the 2 as equivalent, rather than reading both.
        // SAFETY: Caller guarantees `len == capacity`
        unsafe { assert_unchecked!(self.columns.len == self.columns.capacity) };

        assert!(self.columns.capacity < Self::MAX_CAPACITY, "Maximum capacity exceeded");

        let min_capacity = self.columns.capacity + 1;
        // SAFETY:
        // `min_capacity > self.columns.capacity`.
        // `capacity < MAX_CAPACITY` therefore `min_capacity <= MAX_CAPACITY`
        unsafe { self.grow(min_capacity) };
    }

    /// Push a new element, without checking capacity.
    ///
    /// # SAFETY
    ///
    /// Caller must ensure that `self.len < self.capacity`.
    #[expect(clippy::inline_always, reason = "trivial writes on the hot path")]
    #[inline(always)]
    unsafe fn push_unchecked(&mut self, value: F) -> I {
        debug_assert!(self.columns.len < self.columns.capacity);

        let index = self.columns.len;

        // SAFETY: `index = len < capacity` per this function's safety contract, so
        // `field_ptrs` returns pointers to the element's (uninitialized) field slots
        // within the allocation, valid for writes and aligned, as `F::write` requires.
        unsafe { F::write(value, self.columns.field_ptrs(index)) };

        self.columns.len = index + 1;

        // SAFETY: `index < capacity <= MAX_CAPACITY <= I::MAX + 1`, so `index <= I::MAX`
        unsafe { I::from_usize_unchecked(index) }
    }

    /// Reserve capacity for at least `additional` more elements.
    ///
    /// # Panics
    ///
    /// Panics if the required capacity exceeds [`MAX_CAPACITY`] (which includes the case
    /// where it overflows `usize`).
    ///
    /// [`MAX_CAPACITY`]: Self::MAX_CAPACITY
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        // This is equivalent to `len + additional > capacity`, but is a single check,
        // and cannot overflow. The subtraction cannot underflow (`len` is always `<= capacity`),
        // and if `len + additional` would overflow `usize`, then `additional` necessarily
        // exceeds `capacity - len`.
        if additional > self.columns.capacity - self.columns.len {
            // SAFETY: Just checked `additional > capacity - len`
            unsafe { self.grow_for_reserve(additional) };
        }
    }

    /// Grow the allocation to hold at least `additional` more elements.
    ///
    /// # SAFETY
    ///
    /// `additional` must be `> self.columns.capacity - self.columns.len`.
    /// i.e. there is insufficient capacity to hold `additional` more elements.
    ///
    /// # Panics
    ///
    /// Panics if the required capacity (`len + additional`) exceeds [`MAX_CAPACITY`].
    ///
    /// [`MAX_CAPACITY`]: Self::MAX_CAPACITY
    #[cold]
    #[inline(never)]
    unsafe fn grow_for_reserve(&mut self, additional: usize) {
        // This check detects both `len + additional` exceeding `MAX_CAPACITY` and it overflowing `usize`.
        // The subtraction cannot underflow: `len <= MAX_CAPACITY` by `MultiVec`'s invariants.
        assert!(additional <= Self::MAX_CAPACITY - self.columns.len, "Maximum capacity exceeded");

        // Cannot overflow: `additional <= MAX_CAPACITY - len` (checked above)
        let min_capacity = self.columns.len + additional;
        // SAFETY:
        // Caller guarantees `additional > capacity - len`, so `min_capacity > capacity`.
        // Assertion above guarantees `min_capacity <= MAX_CAPACITY`.
        unsafe { self.grow(min_capacity) };
    }

    /// Grow the allocation to hold at least `min_capacity` elements.
    ///
    /// # SAFETY
    ///
    /// * `min_capacity` must be `> self.columns.capacity`.
    /// * `min_capacity` must be `<= MAX_CAPACITY`.
    unsafe fn grow(&mut self, min_capacity: usize) {
        debug_assert!(min_capacity > self.columns.capacity);
        debug_assert!(min_capacity <= Self::MAX_CAPACITY);

        // Grow by doubling (clamped to `MAX_CAPACITY`), with a minimum capacity of 4.
        // The doubling cannot overflow: `capacity <= MAX_CAPACITY <= isize::MAX`,
        // so `capacity * 2 <= usize::MAX - 1`.
        // Note that `.min(Self::MAX_CAPACITY)` must be after `.max(MIN_CAPACITY)`.
        // `MIN_CAPACITY` can be larger than `MAX_CAPACITY` if `ELEMENT_SIZE` is huge.
        let new_capacity =
            (self.columns.capacity * 2).max(min_capacity).max(MIN_CAPACITY).min(Self::MAX_CAPACITY);

        debug_assert!(new_capacity > self.columns.capacity);
        debug_assert!(new_capacity <= Self::MAX_CAPACITY);

        // Allocation happens before any of `self`'s fields are mutated and before the old
        // allocation is deallocated, so `self`'s invariants hold even if `allocate` aborts.
        // SAFETY: `0 < new_capacity <= MAX_CAPACITY <= MAX_ALLOC_CAPACITY`
        // (clamped above; `new_capacity >= 4`).
        let new_columns = unsafe { Columns::<F>::allocate(new_capacity) };
        let new_base_ptr = new_columns.base_ptr;

        if self.columns.capacity > 0 {
            // SAFETY: By `self`'s invariants, the first `len` elements of every column are
            // initialized. The new allocation has layout `layout_for(new_capacity)`, and
            // `self.len <= self.capacity < new_capacity`. It is freshly allocated, so the
            // two allocations do not overlap.
            // The old copies of the elements are never used again: the old allocation is
            // deallocated immediately below, without reading or dropping them.
            unsafe { self.columns.copy_fields(new_base_ptr, new_capacity) };
            // SAFETY: `capacity > 0` (checked above). The old allocation is never used
            // again: `base_ptr` and `capacity` are replaced immediately below, and the
            // `Columns` was not copied anywhere.
            unsafe { self.columns.deallocate() };
        }

        self.columns.base_ptr = new_base_ptr;
        self.columns.capacity = new_capacity;
    }

    /// Get references to every field of the element at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    #[inline]
    pub fn get(&self, index: I) -> F::Ref<'_> {
        let index = self.bounds_checked_index(index);

        // SAFETY: `index < len <= capacity`, so `field_ptrs` returns pointers to the
        // element's field values, which are initialized (by `MultiVec`'s invariants).
        // The references borrow `self`, so the values cannot be mutated while they live.
        unsafe { F::create_ref(self.columns.field_ptrs(index)) }
    }

    /// Get mutable references to every field of the element at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    #[inline]
    pub fn get_mut(&mut self, index: I) -> F::Mut<'_> {
        let index = self.bounds_checked_index(index);

        // SAFETY: Same as `get`, plus the references borrow `self` exclusively,
        // so the values cannot be otherwise accessed while they live.
        unsafe { F::create_mut(self.columns.field_ptrs(index)) }
    }

    /// Check `index` is in bounds and convert it to `usize`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= len`.
    #[inline]
    fn bounds_checked_index(&self, index: I) -> usize {
        let index = index.index();
        if index >= self.columns.len {
            // Cold function, so that the `fmt::Arguments` construction for the panic
            // message does not put stack frame + spill instructions on the hot path.
            // Takes `&self` rather than `len`, so that the hot path's bounds check can
            // compare against `len` in memory, without loading it into a register.
            self.out_of_bounds(index);
        }
        index
    }

    /// Panic with out-of-bounds message.
    #[cold]
    #[inline(never)]
    fn out_of_bounds(&self, index: usize) -> ! {
        let len = self.columns.len;
        panic!("Index out of bounds: `len` is {len} but `index` is {index}");
    }

    /// Get references to every field of the element at `index`, without checking
    /// that `index` is in bounds.
    ///
    /// # SAFETY
    ///
    /// `index` must be in bounds: `index.index() < self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: I) -> F::Ref<'_> {
        let index = index.index();
        debug_assert!(index < self.columns.len);

        // SAFETY: `index < len` per this function's safety contract, and `len <= capacity`,
        // so `field_ptrs` returns pointers to the element's field values, which are initialized
        // (by `MultiVec`'s invariants).
        // The references borrow `self`, so the values cannot be mutated while they live.
        unsafe { F::create_ref(self.columns.field_ptrs(index)) }
    }

    /// Get mutable references to every field of the element at `index`, without
    /// checking that `index` is in bounds.
    ///
    /// # SAFETY
    ///
    /// `index` must be in bounds: `index.index() < self.len()`.
    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, index: I) -> F::Mut<'_> {
        let index = index.index();
        debug_assert!(index < self.columns.len);

        // SAFETY: Same as `get_unchecked`, plus the references borrow `self` exclusively,
        // so the values cannot be otherwise accessed while they live
        unsafe { F::create_mut(self.columns.field_ptrs(index)) }
    }

    /// Get slices over every column.
    //
    // Bounded `F: SliceFields<I>` (not just `Fields`): the slice views are keyed
    // by the index type `I`, so they live on the `SliceFields<I>` trait - see its docs.
    #[inline]
    pub fn slices(&self) -> F::Slices<'_>
    where
        F: SliceFields<I>,
    {
        let len = self.columns.len;

        // Communicate the bound on the slices' length to compiler.
        // SAFETY: `len <= capacity <= MAX_CAPACITY` by `MultiVec`'s invariants.
        unsafe { assert_unchecked!(len <= Self::MAX_CAPACITY) };

        // SAFETY: `slice_ptrs` returns aligned pointers to the columns' starts
        // (dangling if `capacity == 0`, but `len == 0` then too),
        // whose first `len` elements are initialized (by `MultiVec`'s invariants).
        // The slices borrow `self`, so the values cannot be mutated while they live.
        unsafe { F::create_slices(self.columns.slice_ptrs(), len) }
    }

    /// Get mutable slices over every column.
    //
    // Bounded `F: SliceFields<I>` - see `slices`.
    #[inline]
    pub fn slices_mut(&mut self) -> F::SlicesMut<'_>
    where
        F: SliceFields<I>,
    {
        let len = self.columns.len;

        // Communicate the bound on the slices' length to compiler.
        // SAFETY: `len <= capacity <= MAX_CAPACITY` by `MultiVec`'s invariants.
        unsafe { assert_unchecked!(len <= Self::MAX_CAPACITY) };

        // SAFETY: Same as `slices`, plus the slices borrow `self` exclusively,
        // so the values cannot be otherwise accessed while they live
        unsafe { F::create_slices_mut(self.columns.slice_ptrs(), len) }
    }

    /// Iterate over all valid indices.
    ///
    /// The returned iterator does not borrow `self` (the `use<...>` clause omits
    /// the `&self` lifetime, opting out of edition 2024's capture-everything default).
    /// It snapshots `len`, so the table can be mutated while the iterator lives.
    /// IDs of elements pushed after the `iter_ids` call are not included.
    #[inline]
    pub fn iter_ids(&self) -> impl ExactSizeIterator<Item = I> + FusedIterator + use<I, F> {
        let len = self.columns.len;

        // Communicate the bound on the returned iterator's length (`size_hint`) to compiler.
        // SAFETY: `len <= capacity <= MAX_CAPACITY` by `MultiVec`'s invariants.
        unsafe { assert_unchecked!(len <= Self::MAX_CAPACITY) };

        (0..len).map(|i| {
            // Communicate the bound on the yielded index to compiler.
            // SAFETY: `len <= capacity <= MAX_CAPACITY` by `MultiVec`'s invariants.
            // `len <= Self::MAX_CAPACITY`, and max value of `i` is `len - 1`.
            unsafe { assert_unchecked!(i < Self::MAX_CAPACITY) };

            // SAFETY: `i < len <= MAX_CAPACITY <= I::MAX + 1`, so `i <= I::MAX`
            unsafe { I::from_usize_unchecked(i) }
        })
    }

    /// Iterate over the elements, yielding references to every field of each element.
    #[inline]
    pub fn iter(&self) -> Iter<'_, F> {
        Iter::new(self)
    }

    /// Iterate over the elements, yielding mutable references to every field of each element.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, F> {
        IterMut::new(self)
    }

    /// Iterate over the elements, yielding each element's ID and references to
    /// every field of it.
    #[inline]
    pub fn iter_enumerated(
        &self,
    ) -> impl ExactSizeIterator<Item = (I, F::Ref<'_>)> + FusedIterator {
        self.iter().enumerate().map(|(index, item)| {
            // SAFETY: `index < len <= MAX_CAPACITY <= I::MAX + 1`, so `index <= I::MAX`
            let index = unsafe { I::from_usize_unchecked(index) };
            (index, item)
        })
    }

    /// Iterate over the elements, yielding each element's ID and mutable references to
    /// every field of it.
    #[inline]
    pub fn iter_mut_enumerated(
        &mut self,
    ) -> impl ExactSizeIterator<Item = (I, F::Mut<'_>)> + FusedIterator {
        self.iter_mut().enumerate().map(|(index, item)| {
            // SAFETY: `index < len <= MAX_CAPACITY <= I::MAX + 1`, so `index <= I::MAX`
            let index = unsafe { I::from_usize_unchecked(index) };
            (index, item)
        })
    }

    /// Consume the `MultiVec`, yielding each element's ID and the element as an owned `F`
    /// value (reassembled from its stored field values).
    #[inline]
    pub fn into_iter_enumerated(self) -> impl ExactSizeIterator<Item = (I, F)> + FusedIterator {
        self.into_iter().enumerate().map(|(index, item)| {
            // SAFETY: `index < len <= MAX_CAPACITY <= I::MAX + 1`, so `index <= I::MAX`
            let index = unsafe { I::from_usize_unchecked(index) };
            (index, item)
        })
    }
}

impl<I: Idx, F: Fields> IntoIterator for MultiVec<I, F> {
    type Item = F;
    type IntoIter = IntoIter<F>;

    /// Consume the `MultiVec`, yielding each element as an owned `F` value (reassembled
    /// from its stored field values).
    fn into_iter(self) -> IntoIter<F> {
        IntoIter::new(self)
    }
}

impl<'v, I: Idx, F: Fields> IntoIterator for &'v MultiVec<I, F> {
    type Item = F::Ref<'v>;
    type IntoIter = Iter<'v, F>;

    fn into_iter(self) -> Iter<'v, F> {
        self.iter()
    }
}

impl<'v, I: Idx, F: Fields> IntoIterator for &'v mut MultiVec<I, F> {
    type Item = F::Mut<'v>;
    type IntoIter = IterMut<'v, F>;

    fn into_iter(self) -> IterMut<'v, F> {
        self.iter_mut()
    }
}

impl<I: Idx, F: Fields> Default for MultiVec<I, F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Cloning is *field-wise*: each column is cloned, column by column (bitwise, for
/// `Copy` field types). `F`'s own `Clone` impl is never invoked - elements only ever exist
/// as scattered field values, so there is no `F` value to clone. [`CloneFields`] is
/// implemented by the `multi_vec!` macro only for tables declared with `#[derive(Clone)]`,
/// and its `Clone` supertrait requires the fields struct to opt in too (see the trait's
/// docs).
impl<I: Idx, F: CloneFields> Clone for MultiVec<I, F> {
    fn clone(&self) -> Self {
        let len = self.columns.len;
        if len == 0 {
            return Self::new();
        }

        // Allocate exactly `len` (not `capacity`), like `Vec::clone`.
        // `new.len` remains 0 until all elements are cloned. If an element's `clone`
        // panics, `new` is dropped with `len == 0`, so its `Drop` does not touch the
        // partially-initialized elements, and frees the allocation. The already-cloned
        // elements are dropped during unwinding by `clone_columns`' drop guards, so
        // nothing is leaked.
        // SAFETY: `0 < len <= capacity <= MAX_ALLOC_CAPACITY` by `MultiVec`'s invariants
        // (`len > 0` checked above).
        let new_columns = unsafe { Columns::<F>::allocate(len) };
        let mut new_vec = Self { columns: new_columns, index_marker: PhantomData };

        let src_ptrs = self.columns.slice_ptrs();
        let dst_ptrs = new_vec.columns.slice_ptrs();
        let src_and_dst_ptrs =
            CopyArray::from_fn(|i| SrcAndDstPtrs { src_ptr: src_ptrs[i], dst_ptr: dst_ptrs[i] });

        // SAFETY: `slice_ptrs` returns aligned pointers to each column's start, so each
        // `src_and_dst_ptrs[i]` pairs the `i`th `self` (source) column with the `i`th `new`
        // (destination) column. By `self`'s invariants, the first `len` elements of every
        // `self` column are initialized. `new`'s columns are valid for writes of `len`
        // elements (`new.capacity == len`). `new`'s allocation is freshly allocated,
        // so the two do not overlap.
        unsafe { F::clone_columns(src_and_dst_ptrs, len) };

        // `clone_columns` initialized the first `len` elements of every `new` column
        // (guaranteed by `CloneFields`' contract), so `new`'s invariants hold with `len` set
        new_vec.columns.len = len;

        new_vec
    }
}

impl<I: Idx, F: Fields> Drop for MultiVec<I, F> {
    fn drop(&mut self) {
        // If never allocated, no backing allocation to free, and no values to drop either
        if self.columns.capacity == 0 {
            return;
        }

        // Drop values.
        // SAFETY: By `MultiVec`'s invariants, the first `len` elements of every column are
        // initialized, and `slice_ptrs` returns aligned pointers to the columns' starts.
        // The values are never used again - the allocation is freed below without reading them.
        // If an element's `Drop` panics, the remaining elements and the allocation are leaked -
        // `drop` is not re-entered.
        unsafe { F::drop_columns(self.columns.slice_ptrs(), self.columns.len) };

        // Free backing allocation.
        // SAFETY: `capacity > 0` (checked above). The allocation is never used again:
        // `self` is being dropped, and its `Columns` was not copied anywhere (iterators
        // copy it, but only by consuming the `MultiVec` or holding a borrow of it, which
        // has expired).
        unsafe { self.columns.deallocate() };
    }
}

impl<I: Idx, F: Fields> Debug for MultiVec<I, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MultiVec")
            .field("len", &self.columns.len)
            .field("capacity", &self.columns.capacity)
            .finish_non_exhaustive()
    }
}

/// Get the minimum of two `usize`s.
///
/// Equivalent to [`std::cmp::min`] but can be used in const context.
const fn min(a: usize, b: usize) -> usize {
    if a < b { a } else { b }
}
