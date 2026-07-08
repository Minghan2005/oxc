use std::{
    marker::PhantomData,
    sync::atomic::{AtomicUsize, Ordering},
};

use oxc_index::{Idx, define_index_type};

use super::{MultiVec, multi_vec};

define_index_type! {
    pub struct TestId = u32;
}

define_index_type! {
    pub struct SmallId = u8;
    MAX_INDEX = 3;
}

// Field types deliberately not ordered by alignment, and of mixed sizes/alignments
// (8, 2, 8, 1), to exercise padding between field arrays.
multi_vec! {
    /// Table of [`Scope`]s.
    #[derive(Clone, Debug)]
    pub table ScopeTable<TestId, Scope>;

    /// Test scope.
    #[derive(Clone, Copy, Debug)]
    pub struct Scope {
        parent_id: Option<TestId>,
        #[plural(all_flags)]
        flags: u16,
        big: u64,
        small: u8,
    }
}

multi_vec! {
    /// Table testing every combination of doc comments and `#[plural(...)]` on fields.
    /// The attributes are peeled off one at a time by the macro's `@field` rules, so all
    /// combinations and orders are accepted. Each `#[plural(...)]` names its slice field
    /// something the default (append `s`) would not produce, so the test observes that
    /// the attribute was consumed, wherever the doc comments sit.
    table AttrTable<TestId, AttrItem>;

    struct AttrItem {
        bare: u32,

        /// Doc comment, no `#[plural(...)]`.
        documented: u32,

        #[plural(plural_only_column)]
        plural_only: u32,

        /// Doc comment before `#[plural(...)]`.
        #[plural(doc_before_column)]
        doc_before: u32,

        #[plural(doc_after_column)]
        /// Doc comment after `#[plural(...)]`.
        doc_after: u32,

        /// First doc comment before `#[plural(...)]`.
        /// Second doc comment before `#[plural(...)]`.
        #[plural(doc_around_column)]
        /// Doc comment after `#[plural(...)]`.
        doc_around: u32,
    }
}

multi_vec! {
    /// Table with an index type whose range is only 0..=3.
    table SmallTable<SmallId, Small>;

    struct Small {
        value: u32,
    }
}

multi_vec! {
    /// Table with a zero-sized field alongside a non-zero-sized one.
    table ZstTable<TestId, Zst>;

    struct Zst {
        value: u32,
        unit: (),
    }
}

/// Over-aligned type (align 16).
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(align(16))]
struct Aligned(u8);

multi_vec! {
    /// Table with an over-aligned field after a byte-sized one.
    table AlignedTable<TestId, AlignedPair>;

    struct AlignedPair {
        byte: u8,
        aligned: Aligned,
    }
}

multi_vec! {
    /// Table with a field type which needs dropping (`String`), alongside `Copy` fields.
    /// (`#[derive(Debug, Clone)]` also covers a multi-derive list in the reverse order.)
    #[derive(Debug, Clone)]
    table StringTable<TestId, StringItem>;

    #[derive(Clone, Debug)]
    struct StringItem {
        name: String,
        value: u32,
    }
}

/// Type which counts its drops.
struct Counted(#[expect(dead_code)] u8);

static COUNTED_DROPS: AtomicUsize = AtomicUsize::new(0);

impl Drop for Counted {
    fn drop(&mut self) {
        COUNTED_DROPS.fetch_add(1, Ordering::Relaxed);
    }
}

multi_vec! {
    /// Table with a drop-counting field. Neither `Copy`, `Clone`, nor `Debug`, so the
    /// table can derive neither `Clone` nor `Debug` - which also verifies that tables
    /// with non-`Clone` / non-`Debug` fields compile.
    table CountedTable<TestId, CountedItem>;

    struct CountedItem {
        counted: Counted,
        value: u32,
    }
}

/// Zero-sized type which counts its drops.
#[derive(Debug)]
struct CountedZst;

static COUNTED_ZST_DROPS: AtomicUsize = AtomicUsize::new(0);

impl Drop for CountedZst {
    fn drop(&mut self) {
        COUNTED_ZST_DROPS.fetch_add(1, Ordering::Relaxed);
    }
}

multi_vec! {
    /// Table with a zero-sized field which needs dropping.
    table CountedZstTable<TestId, CountedZstItem>;

    struct CountedZstItem {
        value: u32,
        marker: CountedZst,
    }
}

/// Type which counts its drops. `Clone` but not `Copy`, so its column is cloned via
/// tier 2 ([`CloneColumn`](crate::multi_vec::__private::CloneColumn)).
#[derive(Debug)]
struct DropTracked(u32);

static DROP_TRACKED_DROPS: AtomicUsize = AtomicUsize::new(0);

impl Clone for DropTracked {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

impl Drop for DropTracked {
    fn drop(&mut self) {
        DROP_TRACKED_DROPS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Type whose 3rd clone panics. Counts its clones and drops.
#[derive(Debug)]
struct PanicOnThirdClone(u32);

static PANIC_CLONES: AtomicUsize = AtomicUsize::new(0);
static PANIC_DROPS: AtomicUsize = AtomicUsize::new(0);

impl Clone for PanicOnThirdClone {
    fn clone(&self) -> Self {
        assert!(PANIC_CLONES.fetch_add(1, Ordering::Relaxed) != 2, "Clone panicked");
        Self(self.0)
    }
}

impl Drop for PanicOnThirdClone {
    fn drop(&mut self) {
        PANIC_DROPS.fetch_add(1, Ordering::Relaxed);
    }
}

multi_vec! {
    /// Table for testing panic safety of `clone`.
    #[derive(Clone)]
    table PanicTable<TestId, PanicItem>;

    #[derive(Clone)]
    struct PanicItem {
        tracked: DropTracked,
        panicky: PanicOnThirdClone,
    }
}

/// Type which counts its drops, for testing `into_iter`.
/// (Separate from [`Counted`]: that type's drop count is asserted by another test,
/// and tests run in parallel.)
#[derive(Debug)]
struct IterCounted(#[expect(dead_code)] u8);

static ITER_COUNTED_DROPS: AtomicUsize = AtomicUsize::new(0);

impl Drop for IterCounted {
    fn drop(&mut self) {
        ITER_COUNTED_DROPS.fetch_add(1, Ordering::Relaxed);
    }
}

multi_vec! {
    /// Table with a drop-counting field, for testing that `into_iter` drops unyielded elements.
    table IterDropTable<TestId, IterDropItem>;

    struct IterDropItem {
        counted: IterCounted,
        value: u32,
    }
}

multi_vec! {
    /// Table with a single lifetime param, without derives.
    table BorrowTable<'a, TestId, Borrowed<'a>>;

    struct Borrowed<'a> {
        text: &'a str,
        size: usize,
    }
}

multi_vec! {
    /// Table with a single lifetime param, named something other than `'a`.
    /// `other_names` both borrows and needs dropping.
    table ThingTable<'s, TestId, Thing<'s>>;

    struct Thing<'s> {
        name: &'s str,
        // The default pluralization (append `s`) would produce the wrong name here
        #[plural(all_other_names)]
        other_names: Vec<&'s str>,
    }
}

multi_vec! {
    /// Table with a single lifetime param, with `Clone` and `Debug` derives.
    /// `extra` both borrows and needs dropping.
    #[derive(Clone, Debug)]
    table LabelTable<'a, TestId, Label<'a>>;

    #[derive(Clone, Debug)]
    struct Label<'a> {
        text: &'a str,
        extra: Vec<&'a str>,
    }
}

multi_vec! {
    /// Table whose fields borrow, with two lifetime params. `tags` is `Clone`-but-not-`Copy`
    /// and needs dropping, with a lifetime inside - exercising the element-by-element clone
    /// tier and `drop_columns` with borrowed data.
    #[derive(Clone, Debug)]
    table AliasTable<'a, 'b, TestId, Alias<'a, 'b>>;

    /// Test alias.
    #[derive(Clone, Debug)]
    struct Alias<'a, 'b> {
        name: &'a str,
        #[plural(all_nicknames)]
        nickname: &'b str,
        count: u32,
        #[plural(all_tags)]
        tags: Vec<&'a str>,
    }
}

/// Key type with a lifetime param, standing in for a future branded ID type.
///
/// [`Idx`] requires `'static`, so only the `'static` instantiation can implement it
/// currently - enough to exercise the macro's handling of lifetime arguments on the key.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
struct BrandedId<'brand>(u32, PhantomData<&'brand ()>);

impl Idx for BrandedId<'static> {
    const MAX: usize = u32::MAX as usize;

    unsafe fn from_usize_unchecked(idx: usize) -> Self {
        // Caller guarantees `idx <= Self::MAX == u32::MAX`, so the cast is lossless
        #[expect(clippy::cast_possible_truncation)]
        BrandedId(idx as u32, PhantomData)
    }

    fn index(self) -> usize {
        self.0 as usize
    }
}

multi_vec! {
    /// Table whose key type takes a lifetime argument, alongside a table lifetime.
    table BrandedTable<'a, BrandedId<'static>, BrandedItem<'a>>;

    struct BrandedItem<'a> {
        text: &'a str,
    }
}

// Table with a mix of field visibilities, defined in a nested module. The struct field,
// the view-type fields, and the accessor methods each inherit the visibility written on
// the `struct` field: `secret` is private to `vis_inner`, `shared` is visible within the
// crate, `open` is fully public. That the crate-visible and public fields are reachable
// from *outside* `vis_inner` is proven by `vis_field_visibilities` below; that the private
// field's own accessors are usable *within* `vis_inner` is proven by `read_secret` here.
mod vis_inner {
    use super::{TestId, multi_vec};

    multi_vec! {
        /// Table exercising per-field visibility.
        #[derive(Clone, Debug)]
        pub table VisTable<TestId, VisItem>;

        /// Item with fields of three different visibilities.
        #[derive(Clone, Debug)]
        pub struct VisItem {
            secret: u32,
            pub(crate) shared: u32,
            pub open: u32,
        }
    }

    // Builds a `VisItem` - only possible inside this module, since `secret` is private here
    // (constructing the struct literal elsewhere would be a "field is private" error).
    pub fn make(secret: u32, shared: u32, open: u32) -> (VisTable, TestId) {
        let mut table = VisTable::new();
        let id = table.push(VisItem { secret, shared, open });
        (table, id)
    }

    // Uses the private field and its accessors - only possible inside this module.
    pub fn read_secret(table: &VisTable, id: TestId) -> u32 {
        let via_accessor = *table.secret(id);
        let via_view = *table.get(id).secret;
        let via_slice = table.secrets()[id];
        assert_eq!(via_accessor, via_view);
        assert_eq!(via_accessor, via_slice);
        via_accessor
    }
}

#[test]
fn vis_field_visibilities() {
    let (mut table, id) = vis_inner::make(1, 2, 3);

    // `pub(crate)` and `pub` fields, accessors, and slice accessors are reachable here,
    // outside the module `VisTable` is defined in.
    assert_eq!(*table.shared(id), 2);
    assert_eq!(*table.open(id), 3);
    assert_eq!(*table.get(id).shared, 2);
    assert_eq!(*table.get(id).open, 3);
    assert_eq!(table.slices().shareds[id], 2);
    assert_eq!(table.slices().opens[id], 3);

    *table.shared_mut(id) = 20;
    *table.open_mut(id) = 30;
    assert_eq!(*table.get(id).shared, 20);
    assert_eq!(table.get_mut(id).open, &mut 30);

    // The private `secret` field is only reachable from within `vis_inner`.
    assert_eq!(vis_inner::read_secret(&table, id), 1);
}

fn scope(i: u32) -> Scope {
    Scope {
        parent_id: i.checked_sub(1).map(TestId::from_raw),
        flags: u16::try_from(i % 1000).unwrap(),
        big: u64::from(i) * 1_000_000_007,
        small: u8::try_from(i % 256).unwrap(),
    }
}

#[test]
fn empty() {
    let table = ScopeTable::new();
    assert_eq!(table.len(), 0);
    assert!(table.is_empty());
    assert_eq!(table.iter_ids().count(), 0);

    let slices = table.slices();
    assert!(slices.parent_ids.is_empty());
    assert!(slices.all_flags.is_empty());
    assert!(slices.bigs.is_empty());
    assert!(slices.smalls.is_empty());

    // Per-field slice methods
    assert!(table.parent_ids().is_empty());
    assert!(table.all_flags().is_empty());

    let clone = table.clone();
    assert!(clone.is_empty());

    assert_eq!(format!("{table:?}"), "{}");
}

#[test]
fn push_and_get() {
    let mut table = ScopeTable::new();
    let id0 = table.push(scope(0));
    let id1 = table.push(scope(1));
    assert_eq!(table.len(), 2);
    assert!(!table.is_empty());
    assert_eq!(id0.raw(), 0);
    assert_eq!(id1.raw(), 1);

    // Per-field accessors
    assert_eq!(*table.parent_id(id0), None);
    assert_eq!(*table.parent_id(id1), Some(id0));
    assert_eq!(*table.flags(id1), 1);
    assert_eq!(*table.big(id1), 1_000_000_007);
    assert_eq!(*table.small(id1), 1);

    // Whole-element accessor
    let scope_ref = table.get(id1);
    assert_eq!(*scope_ref.parent_id, Some(id0));
    assert_eq!(*scope_ref.flags, 1);
    assert_eq!(*scope_ref.big, 1_000_000_007);
    assert_eq!(*scope_ref.small, 1);
}

#[test]
fn field_attr_combinations() {
    let mut table = AttrTable::new();
    let id = table.push(AttrItem {
        bare: 1,
        documented: 2,
        plural_only: 3,
        doc_before: 4,
        doc_after: 5,
        doc_around: 6,
    });

    // Accessors exist for every field, whatever its attributes.
    assert_eq!(*table.bare(id), 1);
    assert_eq!(*table.doc_around(id), 6);

    let slices = table.slices();

    // Fields without `#[plural(...)]` get the default plural (append `s`).
    assert_eq!(slices.bares[id], 1);
    assert_eq!(slices.documenteds[id], 2);

    // `#[plural(...)]` names the slice field, wherever the doc comments sit.
    assert_eq!(slices.plural_only_column[id], 3);
    assert_eq!(slices.doc_before_column[id], 4);
    assert_eq!(slices.doc_after_column[id], 5);
    assert_eq!(slices.doc_around_column[id], 6);

    // Per-field slice methods take the same names as the slice fields.
    assert_eq!(table.bares()[id], 1);
    assert_eq!(table.documenteds()[id], 2);
    assert_eq!(table.plural_only_column()[id], 3);
    assert_eq!(table.doc_before_column()[id], 4);
    assert_eq!(table.doc_after_column()[id], 5);
    assert_eq!(table.doc_around_column()[id], 6);

    // `_mut` variants exist for both the default plural name and a `#[plural(...)]` name.
    table.bares_mut()[id] += 10;
    table.doc_around_column_mut()[id] += 10;
    assert_eq!(*table.bare(id), 11);
    assert_eq!(*table.doc_around(id), 16);
}

#[test]
fn get_unchecked() {
    let mut table = ScopeTable::new();
    for i in 0..3 {
        table.push(scope(i));
    }

    // SAFETY: `0..3` are in bounds (`len == 3`)
    unsafe {
        assert_eq!(*table.get_unchecked(TestId::from_raw(0)).parent_id, None);
        assert_eq!(*table.get_unchecked(TestId::from_raw(1)).big, 1_000_000_007);
        *table.get_unchecked_mut(TestId::from_raw(2)).flags = 42;
    }
    assert_eq!(*table.flags(TestId::from_raw(2)), 42);
}

#[test]
fn mutation() {
    let mut table = ScopeTable::new();
    let id = table.push(scope(0));

    *table.flags_mut(id) = 7;
    assert_eq!(*table.flags(id), 7);

    let scope_mut = table.get_mut(id);
    *scope_mut.parent_id = Some(id);
    *scope_mut.big = 42;
    assert_eq!(*table.parent_id(id), Some(id));
    assert_eq!(*table.big(id), 42);

    let slices_mut = table.slices_mut();
    slices_mut.smalls[id] = 99;
    slices_mut.bigs[id] += 1;
    assert_eq!(*table.small(id), 99);
    assert_eq!(*table.big(id), 43);
}

#[test]
fn field_slices() {
    let mut table = ScopeTable::new();
    let id0 = table.push(scope(0));
    let id1 = table.push(scope(1));

    // Each per-field slice method returns the same slice as the corresponding
    // `slices()` field.
    assert_eq!(table.parent_ids(), table.slices().parent_ids);
    assert_eq!(table.all_flags(), table.slices().all_flags);

    assert_eq!(table.parent_ids().len(), 2);
    assert_eq!(table.parent_ids()[id1], Some(id0));
    assert_eq!(table.bigs()[id1], 1_000_000_007);
    assert_eq!(table.smalls()[id0], 0);

    // Mutations through a field's slice are visible through every other accessor.
    table.all_flags_mut()[id0] = 7;
    table.bigs_mut()[id1] *= 2;
    assert_eq!(*table.flags(id0), 7);
    assert_eq!(table.slices().bigs[id1], 2_000_000_014);
    assert_eq!(*table.get(id1).big, 2_000_000_014);
}

#[test]
fn growth_preserves_values() {
    let mut table = ScopeTable::new();
    let ids = (0..1000).map(|i| table.push(scope(i))).collect::<Vec<_>>();
    assert_eq!(table.len(), 1000);

    for (i, &id) in ids.iter().enumerate() {
        let i = u32::try_from(i).unwrap();
        assert_eq!(id.raw(), i);
        let expected = scope(i);
        let actual = table.get(id);
        assert_eq!(*actual.parent_id, expected.parent_id);
        assert_eq!(*actual.flags, expected.flags);
        assert_eq!(*actual.big, expected.big);
        assert_eq!(*actual.small, expected.small);
    }

    let slices = table.slices();
    assert_eq!(slices.parent_ids.len(), 1000);
    assert_eq!(slices.all_flags.len(), 1000);
    assert_eq!(slices.bigs[TestId::from_raw(999)], 999 * 1_000_000_007);
    assert_eq!(slices.smalls[TestId::from_raw(999)], u8::try_from(999 % 256).unwrap());
}

#[test]
fn reserve_then_push() {
    let mut table = ScopeTable::new();
    table.reserve(100);
    for i in 0..100 {
        table.push(scope(i));
    }
    assert_eq!(table.len(), 100);
    assert_eq!(*table.big(TestId::from_raw(50)), 50 * 1_000_000_007);
    // Reserving less than current capacity is a no-op.
    table.reserve(1);
    assert_eq!(table.len(), 100);
}

#[test]
fn with_capacity() {
    let mut vec = MultiVec::<TestId, Scope>::with_capacity(10);
    assert!(vec.is_empty());
    assert_eq!(vec.capacity(), 10);
    for i in 0..10 {
        vec.push(scope(i));
    }
    // No reallocation was required.
    assert_eq!(vec.capacity(), 10);
    assert_eq!(vec.len(), 10);

    let empty = MultiVec::<TestId, Scope>::with_capacity(0);
    assert_eq!(empty.capacity(), 0);

    let mut table = ScopeTable::with_capacity(5);
    assert!(table.is_empty());
    let id = table.push(scope(0));
    assert_eq!(*table.big(id), 0);
}

#[test]
fn iter_ids() {
    let mut table = ScopeTable::new();
    table.push(scope(0));
    table.push(scope(1));
    table.push(scope(2));
    let iter = table.iter_ids();
    assert_eq!(iter.len(), 3);
    let ids = iter.collect::<Vec<_>>();
    assert_eq!(ids, vec![TestId::from_raw(0), TestId::from_raw(1), TestId::from_raw(2)]);
}

#[test]
fn iter() {
    let mut table = ScopeTable::new();
    for i in 0..3 {
        table.push(scope(i));
    }

    let mut iter = table.iter();
    assert_eq!(iter.len(), 3);
    let first = iter.next().unwrap();
    assert_eq!(*first.parent_id, None);
    assert_eq!(*first.big, 0);
    assert_eq!(iter.len(), 2);

    // Yielded elements match `get`, in order.
    for (id, item) in table.iter_ids().zip(table.iter()) {
        let expected = table.get(id);
        assert_eq!(item.parent_id, expected.parent_id);
        assert_eq!(item.flags, expected.flags);
        assert_eq!(item.big, expected.big);
        assert_eq!(item.small, expected.small);
    }

    // `&table` also iterates, via `IntoIterator`.
    let mut count = 0u64;
    for item in &table {
        assert_eq!(*item.big, count * 1_000_000_007);
        count += 1;
    }
    assert_eq!(count, 3);

    assert!(ScopeTable::new().iter().next().is_none());
}

#[test]
fn iter_mut() {
    let mut table = ScopeTable::new();
    for i in 0..3 {
        table.push(scope(i));
    }

    let iter = table.iter_mut();
    assert_eq!(iter.len(), 3);
    drop(iter);

    for (i, item) in table.iter_mut().enumerate() {
        *item.flags = u16::try_from(i).unwrap() * 10;
        *item.big += 1;
    }
    assert_eq!(*table.flags(TestId::from_raw(2)), 20);
    assert_eq!(*table.big(TestId::from_raw(1)), 1_000_000_008);

    // `&mut table` also iterates, via `IntoIterator`.
    for item in &mut table {
        *item.small = 42;
    }
    assert_eq!(*table.small(TestId::from_raw(0)), 42);
    assert_eq!(*table.small(TestId::from_raw(2)), 42);
}

#[test]
fn into_iter() {
    let mut table = ScopeTable::new();
    for i in 0..3 {
        table.push(scope(i));
    }

    let mut iter = table.into_iter();
    assert_eq!(iter.len(), 3);
    let first = iter.next().unwrap();
    assert_eq!(first.parent_id, None);
    assert_eq!(iter.len(), 2);

    let rest = iter.collect::<Vec<_>>();
    assert_eq!(rest.len(), 2);
    for (i, item) in rest.iter().enumerate() {
        let expected = scope(u32::try_from(i).unwrap() + 1);
        assert_eq!(item.parent_id, expected.parent_id);
        assert_eq!(item.flags, expected.flags);
        assert_eq!(item.big, expected.big);
        assert_eq!(item.small, expected.small);
    }

    // A table can be consumed directly by a `for` loop, via `IntoIterator`.
    let mut table = ScopeTable::new();
    table.push(scope(0));
    let mut count = 0;
    for item in table {
        assert_eq!(item.big, 0);
        count += 1;
    }
    assert_eq!(count, 1);

    // `into_iter` on a never-allocated table: nothing to yield, drop, or free.
    let empty = ScopeTable::new();
    assert!(empty.into_iter().next().is_none());
}

#[test]
fn iter_enumerated() {
    let mut table = ScopeTable::new();
    for i in 0..3 {
        table.push(scope(i));
    }

    let iter = table.iter_enumerated();
    assert_eq!(iter.len(), 3);
    for (id, item) in iter {
        let expected = table.get(id);
        assert_eq!(item.big, expected.big);
    }

    for (id, item) in table.iter_mut_enumerated() {
        *item.flags = u16::try_from(id.raw()).unwrap() + 100;
    }
    assert_eq!(*table.flags(TestId::from_raw(2)), 102);

    let items = table.into_iter_enumerated().collect::<Vec<_>>();
    assert_eq!(items.len(), 3);
    for (i, (id, item)) in items.iter().enumerate() {
        assert_eq!(id.raw(), u32::try_from(i).unwrap());
        assert_eq!(item.flags, u16::try_from(i).unwrap() + 100);
    }
}

#[test]
fn clone_is_independent() {
    let mut table = ScopeTable::new();
    let id = table.push(scope(0));
    let mut clone = table.clone();

    *table.flags_mut(id) = 1;
    *clone.flags_mut(id) = 2;
    assert_eq!(*table.flags(id), 1);
    assert_eq!(*clone.flags(id), 2);

    // Clone of a table which has grown (`capacity > len`) allocates tightly.
    table.reserve(100);
    let clone2 = table.clone();
    assert_eq!(clone2.len(), 1);
    assert_eq!(*clone2.flags(id), 1);

    // Clone of an exactly-full table (`capacity == len`).
    let mut full = ScopeTable::with_capacity(4);
    let ids = (0..4).map(|i| full.push(scope(i))).collect::<Vec<_>>();
    let full_clone = full.clone();
    assert_eq!(full.len(), 4);
    assert_eq!(full_clone.len(), 4);
    for (i, &id) in ids.iter().enumerate() {
        let i = u32::try_from(i).unwrap();
        let expected = scope(i);
        let actual = full_clone.get(id);
        assert_eq!(*actual.parent_id, expected.parent_id);
        assert_eq!(*actual.flags, expected.flags);
        assert_eq!(*actual.big, expected.big);
        assert_eq!(*actual.small, expected.small);
    }
}

#[test]
#[should_panic(expected = "Index out of bounds: `len` is 1 but `index` is 1")]
fn get_out_of_bounds_panics() {
    let mut table = ScopeTable::new();
    table.push(scope(0));
    let _ = table.get(TestId::from_raw(1));
}

#[test]
#[should_panic(expected = "Index out of bounds")]
fn field_accessor_out_of_bounds_panics() {
    let mut table = ScopeTable::new();
    table.push(scope(0));
    let _ = table.parent_id(TestId::from_raw(1));
}

#[test]
fn push_up_to_index_type_max() {
    // `SmallId` has `MAX_INDEX = 3`, so the table's capacity is limited to 4.
    assert_eq!(SmallTable::MAX_CAPACITY, 4);

    let mut table = SmallTable::new();
    for i in 0..4 {
        let id = table.push(Small { value: i });
        assert_eq!(id.raw(), u8::try_from(i).unwrap());
    }
    assert_eq!(table.len(), 4);
}

#[test]
#[should_panic(expected = "Maximum capacity exceeded")]
fn push_beyond_index_type_max_panics() {
    let mut table = SmallTable::new();
    for i in 0..5 {
        table.push(Small { value: i });
    }
}

#[test]
#[should_panic(expected = "Maximum capacity exceeded")]
fn reserve_beyond_index_type_max_panics() {
    let mut table = SmallTable::new();
    table.reserve(5);
}

#[test]
fn zst_field() {
    // A zero-sized field alongside a non-zero-sized one is fine.
    let mut table = ZstTable::new();
    let id0 = table.push(Zst { value: 10, unit: () });
    let id1 = table.push(Zst { value: 20, unit: () });
    assert_eq!(*table.value(id0), 10);
    assert_eq!(*table.value(id1), 20);
    let slices = table.slices();
    assert_eq!(slices.values.raw, [10, 20]);
    assert_eq!(slices.units.len(), 2);

    // Iteration works with a zero-sized field (its pointer never advances).
    assert_eq!(table.iter().map(|item| *item.value).collect::<Vec<_>>(), [10, 20]);
}

#[test]
fn over_aligned_field() {
    let mut table = AlignedTable::new();
    for i in 0..100u8 {
        table.push(AlignedPair { byte: i, aligned: Aligned(i) });
    }
    let slices = table.slices();
    assert_eq!(slices.aligneds.raw.as_ptr().addr() % 16, 0);
    for i in 0..100u8 {
        let id = TestId::from_raw(u32::from(i));
        assert_eq!(slices.bytes[id], i);
        assert_eq!(slices.aligneds[id], Aligned(i));
    }
}

#[test]
fn table_debug() {
    let mut table = ScopeTable::new();
    table.push(scope(0));
    let debug = format!("{table:?}");
    assert!(debug.starts_with("{0: ScopeRef {"));
    assert!(debug.contains("flags: 0"));
    assert!(debug.ends_with("}}"));
}

#[test]
fn drop_type_fields() {
    let mut table = StringTable::new();
    let id0 = table.push(StringItem { name: "hello".to_string(), value: 1 });
    let id1 = table.push(StringItem { name: "world".to_string(), value: 2 });
    assert_eq!(table.name(id0), "hello");
    assert_eq!(table.name(id1), "world");
    assert_eq!(*table.value(id1), 2);

    // Overwriting through a mutable reference drops the old value
    *table.name_mut(id1) = "there".to_string();
    assert_eq!(table.name(id1), "there");

    // Growth moves the `String`s bitwise, without dropping or cloning them
    let ids = (0..100)
        .map(|i| table.push(StringItem { name: i.to_string(), value: i }))
        .collect::<Vec<_>>();
    assert_eq!(table.name(id0), "hello");
    for (i, &id) in ids.iter().enumerate() {
        assert_eq!(table.name(id), &i.to_string());
    }
    // Dropping the table drops all the `String`s (verified by Miri: no leaks)
}

#[test]
fn drop_type_fields_clone() {
    let mut table = StringTable::new();
    let id0 = table.push(StringItem { name: "hello".to_string(), value: 1 });

    // Clone is deep and independent: the `String` column is cloned element-wise.
    let mut clone = table.clone();
    *table.name_mut(id0) = "changed".to_string();
    assert_eq!(clone.name(id0), "hello");
    *clone.name_mut(id0) = "also changed".to_string();
    assert_eq!(table.name(id0), "changed");

    // Clone of a table which has grown (`capacity > len`) allocates tightly.
    table.reserve(100);
    let clone2 = table.clone();
    assert_eq!(clone2.len(), 1);
    assert_eq!(clone2.name(id0), "changed");
}

#[test]
fn elements_dropped_exactly_once() {
    let mut table = CountedTable::new();
    for i in 0..10 {
        table.push(CountedItem { counted: Counted(i), value: u32::from(i) });
    }
    // No drops so far: pushes move the values in, and growth moves elements bitwise
    // without dropping them.
    assert_eq!(COUNTED_DROPS.load(Ordering::Relaxed), 0);

    drop(table);
    assert_eq!(COUNTED_DROPS.load(Ordering::Relaxed), 10);
}

#[test]
fn clone_panic_drops_cloned_elements() {
    let mut table = PanicTable::new();
    for i in 0..4 {
        table.push(PanicItem { tracked: DropTracked(i), panicky: PanicOnThirdClone(i) });
    }
    assert_eq!(DROP_TRACKED_DROPS.load(Ordering::Relaxed), 0);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| table.clone()));
    assert!(result.is_err());

    // Columns are cloned in field declaration order: the `tracked` column was fully
    // cloned (4 elements), then the `panicky` column's 3rd clone panicked (2 cloned).
    // Unwinding must drop every cloned value - the 4 `DropTracked`s via the guard
    // returned for the completed column, and the 2 `PanicOnThirdClone`s via the
    // in-progress column's internal guard. Nothing leaks (verified by Miri).
    assert_eq!(PANIC_CLONES.load(Ordering::Relaxed), 3);
    assert_eq!(DROP_TRACKED_DROPS.load(Ordering::Relaxed), 4);
    assert_eq!(PANIC_DROPS.load(Ordering::Relaxed), 2);

    // The original table is untouched.
    assert_eq!(table.len(), 4);
    assert_eq!(table.tracked(TestId::from_raw(3)).0, 3);
    assert_eq!(table.panicky(TestId::from_raw(3)).0, 3);

    // Dropping the original drops its own 4 elements per column.
    drop(table);
    assert_eq!(DROP_TRACKED_DROPS.load(Ordering::Relaxed), 8);
    assert_eq!(PANIC_DROPS.load(Ordering::Relaxed), 6);
}

#[test]
fn into_iter_drops_unyielded_elements() {
    let mut table = IterDropTable::new();
    for i in 0..10 {
        table.push(IterDropItem { counted: IterCounted(i), value: u32::from(i) });
    }
    assert_eq!(ITER_COUNTED_DROPS.load(Ordering::Relaxed), 0);

    let mut iter = table.into_iter();
    let item = iter.next().unwrap();
    assert_eq!(item.value, 0);
    drop(item);
    assert_eq!(ITER_COUNTED_DROPS.load(Ordering::Relaxed), 1);

    // Dropping the iterator drops the 9 unyielded elements, and frees the allocation
    // (verified by Miri: no leaks).
    drop(iter);
    assert_eq!(ITER_COUNTED_DROPS.load(Ordering::Relaxed), 10);
}

#[test]
fn into_iter_drop_type_fields() {
    let mut table = StringTable::new();
    for i in 0..3 {
        table.push(StringItem { name: i.to_string(), value: i });
    }

    // `into_iter` moves the `String`s out, without cloning them.
    let mut iter = table.into_iter();
    let first = iter.next().unwrap();
    assert_eq!(first.name, "0");
    assert_eq!(first.value, 0);

    // Dropping the iterator drops the 2 unyielded elements' `String`s
    // (verified by Miri: no leaks).
    drop(iter);
}

#[test]
fn zst_field_dropped() {
    let mut table = CountedZstTable::new();
    for i in 0..3 {
        table.push(CountedZstItem { value: i, marker: CountedZst });
    }
    assert_eq!(COUNTED_ZST_DROPS.load(Ordering::Relaxed), 0);

    drop(table);
    assert_eq!(COUNTED_ZST_DROPS.load(Ordering::Relaxed), 3);
}

#[test]
fn function_scope_invocation() {
    // The macro can be invoked inside a function, not just at module scope.
    // The generated types become function-local.
    multi_vec! {
        /// Table defined inside a function.
        #[derive(Clone, Debug)]
        table FnScopeTable<TestId, FnScopeItem>;

        #[derive(Clone, Debug)]
        struct FnScopeItem {
            value: u32,
        }
    }

    let mut table = FnScopeTable::new();
    let id = table.push(FnScopeItem { value: 7 });
    assert_eq!(*table.value(id), 7);
    assert_eq!(table.clone().len(), 1);
    assert_eq!(format!("{table:?}"), "{0: FnScopeItemRef { value: 7 }}");
}

#[test]
fn lifetimes_locals_in_function() {
    // Borrowed data built at runtime inside a function - lifetimes genuinely local.
    let texts: Vec<String> = (0..20).map(|i| format!("text-{i}")).collect();

    let mut table = BorrowTable::with_capacity(4);
    for text in &texts {
        table.push(Borrowed { text, size: text.len() });
    }

    // Growth (capacity 4 -> 20+) moves the borrowed field values bitwise.
    assert_eq!(table.len(), 20);
    assert_eq!(*table.text(TestId::from_raw(19)), "text-19");
    assert_eq!(table.texts().raw.concat().len(), texts.iter().map(String::len).sum::<usize>());
}

#[test]
fn lifetimes_single() {
    let names: Vec<String> = (0..8).map(|i| format!("thing-{i}")).collect();

    let mut table = ThingTable::new();
    for name in &names {
        table.push(Thing { name, other_names: vec![&names[0], name] });
    }
    assert_eq!(table.len(), 8);

    let id0 = TestId::from_raw(0);
    assert_eq!(*table.name(id0), "thing-0");
    assert_eq!(table.get(id0).other_names.len(), 2);
    assert_eq!(table.names().len(), 8);
    assert_eq!(table.all_other_names().raw.iter().map(Vec::len).sum::<usize>(), 16);

    for thing in &table {
        assert!(thing.name.starts_with("thing-"));
    }
    for thing in &mut table {
        thing.other_names.push(*thing.name);
    }
    assert_eq!(table.all_other_names().raw.iter().map(Vec::len).sum::<usize>(), 24);
    *table.name_mut(id0) = names[7].as_str();
    assert_eq!(*table.name(id0), "thing-7");

    // `into_iter`: partially consume; the iterator's `Drop` drops the 7 unyielded
    // elements' `Vec`s (verified by Miri: no leaks)
    let mut iter = table.into_iter();
    let first = iter.next().unwrap();
    assert_eq!(first.other_names, [&names[0], "thing-0", "thing-0"]);
    drop(iter);
}

#[test]
fn lifetimes_single_with_clone_debug() {
    let strings: Vec<String> = (0..4).map(|i| format!("label-{i}")).collect();

    let mut table = LabelTable::new();
    for text in &strings {
        table.push(Label { text, extra: vec![text] });
    }

    let id0 = TestId::from_raw(0);
    assert_eq!(*table.text(id0), "label-0");
    *table.text_mut(id0) = "relabeled";
    assert_eq!(table.texts().raw[0], "relabeled");
    assert_eq!(table.extras().len(), 4);
    table.extra_mut(id0).push("added");

    for label in &table {
        assert!(!label.extra.is_empty());
    }

    let cloned = table.clone();
    assert!(format!("{cloned:?}").contains("relabeled"));

    let labels: Vec<Label<'_>> = table.into_iter().collect();
    assert_eq!(labels.len(), 4);
    assert_eq!(labels[0].extra, ["label-0", "added"]);
}

#[test]
fn lifetimes_full_surface() {
    let name = String::from("hello");
    let nickname = String::from("hi");

    // The borrowed strings are declared before the table: `MultiVec` has a `Drop` impl
    // (and, on stable, no `#[may_dangle]`), so dropck requires everything the elements
    // borrow to outlive the table.
    let mut table = AliasTable::new();
    let id0 = table.push(Alias { name: &name, nickname: &nickname, count: 1, tags: vec![&name] });
    let id1 = table.push(Alias { name: "static", nickname: &nickname, count: 2, tags: vec![] });

    // `get` / `get_mut` / per-field accessors
    assert_eq!(*table.name(id0), "hello");
    assert_eq!(table.get(id0).tags, &vec![&*name]);
    *table.get_mut(id0).count += 10;
    assert_eq!(*table.count(id0), 11);
    *table.name_mut(id1) = "replaced";
    assert_eq!(*table.name(id1), "replaced");

    // Slices (plural names, from `#[plural(...)]` and the `s` default)
    assert_eq!(table.names().raw, ["hello", "replaced"]);
    assert_eq!(table.all_nicknames().len(), 2);
    assert_eq!(table.counts().raw, [11, 2]);
    assert_eq!(table.all_tags().len(), 2);

    // Iteration, by `&` and `&mut`
    for item in &table {
        assert_eq!(*item.nickname, "hi");
    }
    for item in &mut table {
        *item.count += 1;
    }
    assert_eq!(table.counts().raw, [12, 3]);
    let ids: Vec<TestId> = table.iter_ids().collect();
    assert_eq!(ids, [id0, id1]);
    for (id, item) in table.iter_enumerated() {
        let expected = if id == id0 { 12 } else { 3 };
        assert_eq!(*item.count, expected);
    }

    // Clone is independent of the original (borrows the same data)
    let cloned = table.clone();
    assert_eq!(cloned.len(), 2);
    assert_eq!(*cloned.name(id0), "hello");

    // Debug formats through the ref views
    let debug = format!("{table:?}");
    assert!(debug.contains("AliasRef"));
    assert!(debug.contains("hello"));

    // `into_iter` reassembles owned elements; dropping it part-way drops the
    // unyielded elements' `Vec`s (verified by Miri: no leaks)
    let mut iter = table.into_iter();
    let first = iter.next().unwrap();
    assert_eq!(first.count, 12);
    assert_eq!(first.tags, [&*name]);
    drop(iter);
    drop(cloned);
}

#[test]
fn lifetimes_key_with_lifetime_arguments() {
    let strings: Vec<String> = (0..3).map(|i| format!("branded-{i}")).collect();

    let mut table = BrandedTable::new();
    for text in &strings {
        table.push(BrandedItem { text });
    }

    let ids: Vec<BrandedId<'static>> = table.iter_ids().collect();
    assert_eq!(ids.len(), 3);
    let id0 = ids[0];
    assert_eq!(*table.text(id0), "branded-0");
    *table.text_mut(id0) = "rebranded";
    assert_eq!(table.get(id0).text, &"rebranded");
    assert_eq!(table.texts().len(), 3);
}

/// The generated table must be covariant in its lifetime params (like `Vec` is in `T`'s):
/// a table borrowing longer-lived data coerces to one borrowing shorter-lived data.
/// This is a compile-time property; calling the function is not required.
#[expect(dead_code)]
fn lifetimes_are_covariant<'short>(
    table: AliasTable<'static, 'static>,
    iter: AliasTableIter<'short, 'static, 'static>,
) -> (AliasTable<'short, 'short>, AliasTableIter<'short, 'short, 'short>) {
    (table, iter)
}

#[test]
fn shape_ordered_field_sizes_and_offsets() {
    use super::{fields::Fields, shape::CopyArray};

    fn check<F: Fields>() {
        let field_count = <F::Array<usize>>::LEN;

        let ordered = F::SHAPE.ordered_field_sizes_and_offsets();
        let pairs: Vec<(usize, usize)> =
            (0..field_count).map(|i| (ordered[i].size, ordered[i].offset)).collect();

        // Every field appears exactly once: the pairs are the declaration-order
        // `(size, offset)` pairs, as a set
        let (sizes, offsets) = (F::SHAPE.field_sizes(), F::SHAPE.field_offsets());
        let mut expected: Vec<(usize, usize)> =
            (0..field_count).map(|i| (sizes[i], offsets[i])).collect();
        let mut sorted_pairs = pairs.clone();
        sorted_pairs.sort_unstable();
        expected.sort_unstable();
        assert_eq!(sorted_pairs, expected);

        // The columns are contiguous: the first starts at offset 0, each starts where
        // the previous one ends, and the last ends at `element_size`
        assert_eq!(pairs[0].1, 0);
        for window in pairs.windows(2) {
            let ((prev_size, prev_offset), (_, offset)) = (window[0], window[1]);
            assert_eq!(offset, prev_offset + prev_size);
        }
        let (last_size, last_offset) = *pairs.last().unwrap();
        assert_eq!(last_offset + last_size, F::SHAPE.element_size());
    }

    // Mixed sizes/alignments declared out of alignment order; a zero-sized field;
    // an over-aligned field after a byte-sized one; borrowing fields.
    check::<Scope>();
    check::<Zst>();
    check::<AlignedPair>();
    check::<Alias<'static, 'static>>();
}
