//! [`multi_vec!`] macro, generating a named struct-of-arrays table type wrapping a [`MultiVec`].
//!
//! The only unsafe code the macro generates is the `Fields` impl:
//! a thin translation layer between untyped field pointers and the typed field values /
//! references, consisting purely of pointer casts and calls to helper functions in the
//! `utils` module - no arithmetic, no allocation, and no control flow.
//! All the logic (allocation, layout, pointer arithmetic, bounds checks, clone / drop
//! loops) lives in `MultiVec` and the `utils` module, in plain Rust code.
//!
//! [`MultiVec`]: super::MultiVec

/// Create a struct-of-arrays table type.
///
/// Generates:
///
/// * The plain struct itself (e.g. `Scope`), which is what `push` takes.
/// * A table type (e.g. `ScopeTable`) wrapping a `MultiVec` (private), which stores the struct's fields
///   as struct-of-arrays in a single allocation, indexed by a typed index (e.g. `ScopeId`).
/// * A shared-reference view (e.g. `ScopeRef`) returned by `get`, with a `&T` field
///   per struct field.
/// * An mutable-reference view (e.g. `ScopeMut`) returned by `get_mut`, with a `&mut T`
///   field per struct field.
/// * Slice views (e.g. `ScopeSlices` / `ScopeSlicesMut`) returned by `slices` / `slices_mut`,
///   with an [`IndexSlice`] field per struct field, for bulk access to whole field arrays.
///
/// The view type names are derived from the struct's name: a struct `Scope` produces
/// `ScopeRef`, `ScopeMut`, `ScopeSlices`, and `ScopeSlicesMut`.
/// * Iterators: `iter` / `iter_mut` yield the reference views (e.g. `ScopeRef` / `ScopeMut`),
///   and the table implements [`IntoIterator`] (by value, by `&`, and by `&mut`), with
///   `into_iter` yielding each element as an owned struct (e.g. `Scope`). Enumerated
///   variants (`iter_enumerated` etc.) also yield each element's ID. The iterator type
///   names are derived from the table's name: a table `ScopeTable` produces
///   `ScopeTableIter`, `ScopeTableIterMut`, and `ScopeTableIntoIter`.
/// * Per-field accessor methods, e.g. `.parent_id(id)` / `.parent_id_mut(id)`.
/// * The `Fields` impl for the struct, which `MultiVec` requires.
///
/// All generated types are defined directly in the invoking scope, each with the
/// visibility given on the `table` declaration (e.g. `pub table ScopeTable<ScopeId, Scope>;`).
/// The `struct` must repeat that same visibility (e.g. `pub struct Scope`) - it is a
/// compile-time error for the two to differ.
///
/// Each field carries its own visibility, exactly as written on the `struct` field.
/// That visibility is reflected in the generated types (the struct field and the
/// corresponding fields of `ScopeRef` / `ScopeMut` / `ScopeSlices` / `ScopeSlicesMut`),
/// and in the per-field accessor methods (e.g. `.name()` / `.names()`).
/// As always, a field's effective visibility is bounded by the struct's.
///
/// Everything else the macro generates (the impls, and the helper items only they need)
/// is hidden inside an anonymous `const` block, adding no other names to the invoking
/// scope. The macro can be invoked at module scope or inside a function.
///
/// Types which need dropping (e.g. `String`) are supported: the stored field values are
/// dropped when the table is dropped.
///
/// There must be at least one field, and at least one field must have non-zero size
/// (a table whose fields are all zero-sized types would have a zero-sized allocation).
/// Both are enforced when the table is defined, at check time:
///
/// ```compile_fail
/// # use oxc_data_structures::multi_vec::multi_vec;
/// # oxc_index::define_index_type! { struct Id = u32; }
/// // All fields are zero-sized types - error
/// multi_vec! {
///     table AllZstTable<Id, AllZst>;
///
///     struct AllZst {
///         x: (),
///         y: (),
///     }
/// }
/// ```
///
/// ```compile_fail
/// # use oxc_data_structures::multi_vec::multi_vec;
/// # oxc_index::define_index_type! { struct Id = u32; }
/// // `Empty` has no fields - error
/// multi_vec! {
///     table EmptyTable<Id, Empty>;
///
///     struct Empty {}
/// }
/// ```
///
/// The item type in the `table` declaration must be the struct declared below it.
/// Naming a different fields struct (e.g. another table's, which would otherwise
/// compile, silently binding the table to the wrong element type) is a compile-time
/// error:
///
/// ```compile_fail,E0308
/// # use oxc_data_structures::multi_vec::multi_vec;
/// # oxc_index::define_index_type! { struct Id = u32; }
/// # multi_vec! {
/// #     table OtherTable<Id, Other>;
/// #
/// #     struct Other {
/// #         value: u32,
/// #     }
/// # }
/// // `Other` is another table's struct, not the one declared below - error
/// multi_vec! {
///     table WrongItemTable<Id, Other>;
///
///     struct Item {
///         value: u32,
///     }
/// }
/// ```
///
/// # Lifetimes
///
/// The struct's fields may borrow. Declare the lifetimes at the start of the `table`
/// declaration's generics, and apply them to the item type - the struct's lifetime
/// params, with the same names, in its declaration order.
///
/// ```
/// # use oxc_data_structures::multi_vec::multi_vec;
/// # oxc_index::define_index_type! { pub struct AliasId = u32; }
/// multi_vec! {
///     #[derive(Clone, Debug)]
///     table Aliases<'a, 'b, AliasId, Alias<'a, 'b>>;
///
///     #[derive(Clone, Debug)]
///     struct Alias<'a, 'b> {
///         name: &'a str,
///         alias_name: &'b str,
///     }
/// }
///
/// let name = String::from("hello");
/// let mut aliases = Aliases::new();
/// let id = aliases.push(Alias { name: &name, alias_name: "hi" });
/// assert_eq!(*aliases.name(id), "hello");
/// for alias in &aliases {
///     assert_eq!(*alias.alias_name, "hi");
/// }
/// ```
///
/// The table and iterator types take the table's lifetimes (e.g. `Aliases<'a, 'b>`,
/// `AliasesIter<'v, 'a, 'b>`); the view types take the borrow lifetime first, then the
/// struct's (e.g. `AliasRef<'v, 'a, 'b>`). All are covariant in the struct's lifetimes,
/// like `Vec` is in `T`'s.
///
/// The two declarations must use the same lifetime names - the struct's lifetimes
/// (under their own names) parameterize the generated view types and per-field methods.
/// Different names are a compile-time error:
///
/// ```compile_fail,E0261
/// # use oxc_data_structures::multi_vec::multi_vec;
/// # oxc_index::define_index_type! { pub struct AliasId = u32; }
/// multi_vec! {
///     // `'x` does not match the struct's `'a` - error
///     table Aliases<'x, AliasId, Alias<'x>>;
///
///     struct Alias<'a> {
///         name: &'a str,
///     }
/// }
/// ```
///
/// The key type may also take lifetime arguments (e.g. a branded ID type), declared in
/// the same leading list when not `'static`. (`oxc_index::Idx` currently requires
/// `'static`, so non-`'static` key lifetimes await an `Idx` relaxation.)
///
/// The borrowed data must outlive the table *strictly*: the table's drop code runs while
/// the borrows must still be valid (unlike `Vec`, which is exempted from this rule by
/// unstable machinery). In practice: declare the table *after* the data it borrows, so
/// it is dropped first. This fails because `name` is dropped before `aliases`:
///
/// ```compile_fail,E0597
/// # use oxc_data_structures::multi_vec::multi_vec;
/// # oxc_index::define_index_type! { pub struct AliasId = u32; }
/// # multi_vec! {
/// #     table Aliases<'a, AliasId, Alias<'a>>;
/// #
/// #     struct Alias<'a> {
/// #         name: &'a str,
/// #     }
/// # }
/// let mut aliases = Aliases::new();
/// let name = String::from("hello"); // declared after `aliases` - dropped before it
/// aliases.push(Alias { name: &name });
/// ```
///
/// # Cloning
///
/// The generated types implement `Clone` only if you derive it, like ordinary structs:
///
/// * `#[derive(Clone)]` on the `table` declaration makes the table cloneable.
///   This requires the struct to also be `Clone` (derive it too) - it is a compile-time error otherwise.
/// * `#[derive(Clone)]` on the struct requires all field types to be `Clone`, as usual.
///
/// Cloning a table clones column by column: `Copy` field types are copied bitwise
/// (one `memcpy` per field array); other field types are cloned element by element.
/// The struct's own `Clone` impl is never invoked by table cloning (elements are stored
/// as scattered field values, so there is never a whole struct value to clone).
///
/// ```compile_fail
/// # use oxc_data_structures::multi_vec::multi_vec;
/// # oxc_index::define_index_type! { struct Id = u32; }
/// // No `#[derive(Clone)]` on the `Item`struct - error
/// multi_vec! {
///     #[derive(Clone)]
///     table CloneTable<Id, Item>;
///
///     struct Item { value: u32 }
/// }
/// ```
///
/// # Debug
///
/// Likewise, the generated types implement `Debug` only if you derive it:
///
/// * `#[derive(Debug)]` on the `table` declaration makes the table and the view types
///   (e.g. `ScopeRef`) implement `Debug`. This requires the struct to also be `Debug`
///   (derive it too) - it is a compile-time error otherwise.
/// * `#[derive(Debug)]` on the struct requires all field types to be `Debug`, as usual.
///
/// The table prints as a map from each element's ID to the element, formatted through
/// the shared-reference view - e.g. `{ 0: ScopeRef { parent_id: None, ... }, 1: ... }`.
///
/// ```compile_fail
/// # use oxc_data_structures::multi_vec::multi_vec;
/// # oxc_index::define_index_type! { struct Id = u32; }
/// // No `#[derive(Debug)]` on the `Item` struct - error
/// multi_vec! {
///     #[derive(Debug)]
///     table DebugTable<Id, Item>;
///
///     struct Item { value: u32 }
/// }
/// ```
///
/// `Clone` and `Debug` are the only derives accepted on the `table` declaration.
/// Any other derive is a compile-time error:
///
/// ```compile_fail
/// # use oxc_data_structures::multi_vec::multi_vec;
/// # oxc_index::define_index_type! { struct Id = u32; }
/// multi_vec! {
///     // `PartialEq` is not supported - error
///     #[derive(Clone, PartialEq)]
///
///     // Any derive is fine on the struct - it is a plain struct
///     table PartialEqTable<Id, Item>;
///     #[derive(Clone, PartialEq)]
///     struct Item { value: u32 }
/// }
/// ```
///
/// # Dropping
///
/// The struct's own `Drop` impl (if it has one) is never invoked, for the same reason -
/// the stored field values are dropped individually. Do not implement `Drop` on the struct.
///
/// # Pluralization
///
/// The fields of the slice views are pluralized: each holds a whole column, so e.g. a
/// `parent_id` field produces a `parent_ids` slice field. By default the plural is formed
/// by appending `s`. Where that produces the wrong name (e.g. `flags` -> `flagss`),
/// specify the plural with a `#[plural(...)]` attribute on the field. The attribute may
/// appear anywhere among the field's doc comments.
///
/// # Field visibility
///
/// Each field carries the visibility written on the `struct` field - a bare field is
/// private, `pub name` is public, `pub(crate) name` is crate-visible, and so on. That
/// visibility is applied to the field in every generated type (the struct, and the
/// `Ref` / `Mut` / `Slices` / `SlicesMut` views) and to the field's accessor methods
/// (`.name()` / `.name_mut()` / `.names()` / `.names_mut()`). A field's visibility is
/// bounded by the struct's, as always. (Visibility goes on the `struct` fields; the
/// generated types and slice fields all inherit it - do not write it on them.)
///
/// ```
/// mod inner {
///     # use oxc_data_structures::multi_vec::multi_vec;
///     # oxc_index::define_index_type! { pub struct Id = u32; }
///     multi_vec! {
///         pub table Items<Id, Item>;
///
///         pub struct Item {
///             secret: u32,           // private to `inner`
///             pub value: u32,        // public
///         }
///     }
///
///     // Built here because the `secret` field is private to this module.
///     pub fn make() -> (Items, Id) {
///         let mut items = Items::new();
///         let id = items.push(Item { secret: 1, value: 10 });
///         (items, id)
///     }
/// }
///
/// let (items, id) = inner::make();
/// // `value` and its accessor are public, so reachable here.
/// assert_eq!(*items.value(id), 10);
/// ```
///
/// A private field's accessor is not callable from outside the defining module:
///
/// ```compile_fail
/// mod inner {
///     # use oxc_data_structures::multi_vec::multi_vec;
///     # oxc_index::define_index_type! { pub struct Id = u32; }
///     multi_vec! {
///         pub table Items<Id, Item>;
///
///         pub struct Item {
///             secret: u32, // private to `inner`
///         }
///     }
/// }
///
/// let items = inner::Items::new();
/// let id = inner::Id::from_raw(0);
/// let _ = items.secret(id); // error: method `secret` is private
/// ```
///
/// The `struct`'s own visibility must match the `table`'s. A mismatch (here `pub` table,
/// private struct) is a compile-time error:
///
/// ```compile_fail
/// # use oxc_data_structures::multi_vec::multi_vec;
/// # oxc_index::define_index_type! { pub struct Id = u32; }
/// multi_vec! {
///     pub table Items<Id, Item>;
///
///     // Missing `pub` - error
///     struct Item {
///         value: u32,
///     }
/// }
/// ```
///
/// # Field attributes
///
/// Fields accept only doc comments and `#[plural(...)]`. Doc comments are applied to the
/// struct's field, and to the corresponding fields of the reference views (e.g.
/// `ScopeRef` / `ScopeMut`). Any other attribute is a compile-time error - it would
/// apply only to the generated struct's field, not to the fields of the generated view
/// types:
///
/// ```compile_fail
/// # use oxc_data_structures::multi_vec::multi_vec;
/// # oxc_index::define_index_type! { struct Id = u32; }
/// multi_vec! {
///     table ItemTable<Id, Item>;
///
///     struct Item {
///         #[serde(skip)] // not supported - error
///         value: u32,
///     }
/// }
/// ```
///
/// A duplicate `#[plural(...)]` is also a compile-time error:
///
/// ```compile_fail
/// # use oxc_data_structures::multi_vec::multi_vec;
/// # oxc_index::define_index_type! { struct Id = u32; }
/// multi_vec! {
///     table ItemTable<Id, Item>;
///
///     struct Item {
///         #[plural(a)]
///         #[plural(b)] // duplicate - error
///         value: u32,
///     }
/// }
/// ```
///
/// # Example
///
/// ```
/// use oxc_data_structures::multi_vec::multi_vec;
/// # oxc_index::define_index_type! { pub struct ScopeId = u32; }
/// # #[derive(Debug, Clone, Copy)] pub struct NodeId(u32);
/// # #[derive(Debug, Clone, Copy)] pub struct ScopeFlags(u8);
///
/// multi_vec! {
///     /// SoA table of [`Scope`]s.
///     #[derive(Clone, Debug)]
///     pub table ScopeTable<ScopeId, Scope>;
///
///     /// A scope.
///     #[derive(Clone, Debug)]
///     pub struct Scope {
///         parent_id: Option<ScopeId>,
///         node_id: NodeId,
///         #[plural(all_flags)]
///         flags: ScopeFlags,
///     }
/// }
///
/// let mut table = ScopeTable::new();
/// let id = table.push(Scope { parent_id: None, node_id: NodeId(0), flags: ScopeFlags(0) });
/// assert_eq!(*table.parent_id(id), None);
/// assert_eq!(table.get(id).parent_id, &None);
/// assert_eq!(table.slices().parent_ids.len(), 1);
/// assert_eq!(table.slices().all_flags.len(), 1);
/// *table.parent_id_mut(id) = Some(id);
/// assert_eq!(*table.parent_id(id), Some(id));
/// let clone = table.clone();
/// assert_eq!(*clone.parent_id(id), Some(id));
/// for scope in &table {
///     assert_eq!(*scope.parent_id, Some(id));
/// }
/// let scopes: Vec<Scope> = table.into_iter().collect();
/// assert_eq!(scopes.len(), 1);
/// ```
///
/// Note: doc comments and other attributes on the struct are applied to the generated
/// struct; those on the `table` declaration are applied to the generated table type.
/// They are not applied to the other generated types, so e.g. `#[cfg(...)]` will not
/// work as expected.
///
/// [`IndexSlice`]: oxc_index::IndexSlice
#[macro_export]
macro_rules! multi_vec {
    // Entry point.
    //
    // The `table` declaration's attributes are classified first, one at a time, by the
    // `@table_attrs` rules: ordinary attributes (doc comments) pass through to the
    // generated table type; `#[derive(Clone)]` passes through too, and is also recorded
    // in `config`'s `clone` slot (the `@generate_clone` rules generate the clone machinery on it);
    // `#[derive(Debug)]` is recorded in the `debug` slot (it gates the `Debug` impls
    // `@generate` emits); and any other derive is a compile error.
    // An attribute with several derives (e.g. `#[derive(Clone, Debug)]`) is first split
    // into one attribute per derive.
    //
    // Fields are then normalized one at a time: `@normalize` matches one whole field -
    // attributes, name, and type - and builds its normalized record. The `@field`
    // rules then classify the attributes one at a time, refining the record in place:
    // doc comments into `field_comments` (emitted onto the struct field),
    // `#[plural(...)]` into `field_plural_name`, and anything else is a compile error.
    // Once every field is normalized, the `@generate` rule generates the output.
    // To add a new field attribute: add a record slot, a classify rule for it, and
    // pass the slot through the other `@field` rules.
    //
    // Attributes are captured as raw token trees, NOT `:meta` fragments: a captured
    // `:meta` fragment is opaque, so the classify rules could not tell `#[plural(...)]`
    // from `#[doc]`. (Doc comments reach the macro already lowered to `#[doc = "..."]`
    // attribute tokens, so they are matched structurally like any other attribute.)
    //
    // Normalized fields have the form
    // `{ field_vis = [...] field_name = [...] field_ty = [...] field_plural_name = [...] field_comments = [...] }`.
    // `field_vis` is the field's visibility (from the `struct` field, e.g. `pub` /
    // `pub(crate)` / empty), captured as `:vis` and threaded through as an opaque `:tt`
    // like `field_ty`; `@generate` re-parses it as `:vis` and applies it to the field in
    // every generated type and to the field's accessor methods.
    // `field_plural_name` is a single token tree: either the ident from a `#[plural(...)]`
    // attribute, or `[<field_name s>]`, which `paste!` concatenates to e.g. `parent_ids`.
    //
    // The table's generics are lifetimes (optional), then the key, then the item type.
    //
    // The key is matched structurally - `$key_name:ident` plus optional lifetime
    // arguments - not as one `:ty`: after the optional lifetime list, the matcher must
    // decide "another lifetime, or the key?" one token ahead, which is only allowed
    // between single-token fragments - a `ty` fragment there is a `local ambiguity`
    // error. The pieces are reassembled into `config`'s `key` slot, which `@normalize`
    // re-matches as a single `:ty` fragment (legal there, with nothing repeatable in
    // front of it), so the rest of the macro handles the key as one fragment.
    // `$key_name` is also recorded on its own, as the `key_name` slot: `paste!` derives
    // the `$id` method-argument name from it, which it could not do from a type.
    //
    // `$struct_ty` is `:ty`, matched by the real Rust parser, so `Thing<'a, 'b>>;` works:
    // the parser consumes the struct type's half of the trailing `>>` token and leaves the
    // other half for the matcher's closing `>`. (Token-by-token matchers cannot do this -
    // a literal `>` in a pattern does not match half of a `>>`.)
    (
        $(# [ $($table_attr:tt)* ])*
        $vis:vis table $table_name:ident<
            $( $($table_lifetime:lifetime,)+ )?
            $key_name:ident $(< $($key_lifetime:lifetime),+ $(,)? >)?,
            $struct_ty:ty
        >;

        $(#[$struct_attr:meta])*
        $struct_vis:vis struct $struct_name:ident $(< $($struct_lifetime:lifetime),+ $(,)? >)? {
            $($fields:tt)*
        }
    ) => {
        // The `struct`'s visibility must match the `table`'s: the macro applies one
        // visibility to every generated type, so the two declarations have to agree.
        // Checked here rather than in the matcher because `macro_rules!` cannot compare
        // two captured `:vis` fragments for equality - so the check is a compile-time
        // assertion over their `stringify!`d forms. `const_str_eq` is used because `str`'s
        // `==` is not `const`. Both visibilities go through the same `stringify!`, which has
        // already discarded whitespace (`pub(crate)` and `pub (crate)` stringify alike), so
        // the textual comparison is exact. Emitted alongside the recursive call, so it runs
        // exactly once per invocation and needs no plumbing through the internal rules.
        const _: () = ::std::assert!(
            $crate::str::const_str_eq(
                ::std::stringify!($vis),
                ::std::stringify!($struct_vis),
            ),
            "the `struct` visibility must match the `table` visibility \
             (both `pub`, both `pub(crate)`, both private, etc)",
        );

        $crate::multi_vec! {
            @table_attrs
            config = {
                key_ty = [ $key_name $(< $($key_lifetime),+ >)? ]
                key_name = [ $key_name ]
                vis = [ $vis ]
                table_name = [ $table_name ]
                table_lifetimes = [ $( $($table_lifetime),+ )? ]
                table_attrs = []
                struct_ty = [ $struct_ty ]
                struct_name = [ $struct_name ]
                struct_lifetimes = [ $( $($struct_lifetime),+ )? ]
                struct_attrs = [ $(#[$struct_attr])* ]
                clone = []
                debug = []
            }
            pending = [ $([ $($table_attr)* ])* ]
            fields = []
            rest = [ $($fields)* ]
        }
    };

    // Classify a `#[derive(Clone)]` on the `table` declaration: pass it through to the
    // generated table type, and record it in the `clone` slot - the `@generate_clone`
    // rules generate the clone machinery (a `CloneFields` impl) when the slot is full.
    // The passthrough works as a real derive: `MultiVec` implements `Clone` where the
    // fields struct implements `CloneFields`.
    // (A duplicate `#[derive(Clone)]` just passes through twice and gets rustc's
    // duplicate-impl error, like any duplicated derive; the slot is set idempotently.)
    (
        @table_attrs
        config = {
            key_ty = $key_ty:tt
            key_name = $key_name:tt
            vis = $vis:tt
            table_name = $table_name:tt
            table_lifetimes = $table_lifetimes:tt
            table_attrs = [ $($table_attrs:tt)* ]
            struct_ty = $struct_ty:tt
            struct_name = $struct_name:tt
            struct_lifetimes = $struct_lifetimes:tt
            struct_attrs = $struct_attrs:tt
            clone = $_clone:tt
            debug = $debug:tt
        }
        pending = [
            [ derive ( Clone $(,)? ) ]
            $($pending:tt)*
        ]
        fields = $fields:tt
        rest = $rest:tt
    ) => {
        $crate::multi_vec! {
            @table_attrs
            config = {
                key_ty = $key_ty
                key_name = $key_name
                vis = $vis
                table_name = $table_name
                table_lifetimes = $table_lifetimes
                table_attrs = [
                    $($table_attrs)*
                    #[derive(Clone)]
                ]
                struct_ty = $struct_ty
                struct_name = $struct_name
                struct_lifetimes = $struct_lifetimes
                struct_attrs = $struct_attrs
                clone = [ Clone ]
                debug = $debug
            }
            pending = [ $($pending)* ]
            fields = $fields
            rest = $rest
        }
    };

    // Classify a `#[derive(Debug)]` on the `table` declaration: record it in the `debug`
    // slot, and do NOT pass it through. It cannot be a real derive: the view types'
    // `Debug` impls must also be conditional on it, which a derive on the table type
    // alone could not achieve. `@generate` emits the `Debug` impls only when the slot
    // is filled.
    (
        @table_attrs
        config = {
            key_ty = $key_ty:tt
            key_name = $key_name:tt
            vis = $vis:tt
            table_name = $table_name:tt
            table_lifetimes = $table_lifetimes:tt
            table_attrs = $table_attrs:tt
            struct_ty = $struct_ty:tt
            struct_name = $struct_name:tt
            struct_lifetimes = $struct_lifetimes:tt
            struct_attrs = $struct_attrs:tt
            clone = $clone:tt
            debug = []
        }
        pending = [
            [ derive ( Debug $(,)? ) ]
            $($pending:tt)*
        ]
        fields = $fields:tt
        rest = $rest:tt
    ) => {
        $crate::multi_vec! {
            @table_attrs
            config = {
                key_ty = $key_ty
                key_name = $key_name
                vis = $vis
                table_name = $table_name
                table_lifetimes = $table_lifetimes
                table_attrs = $table_attrs
                struct_ty = $struct_ty
                struct_name = $struct_name
                struct_lifetimes = $struct_lifetimes
                struct_attrs = $struct_attrs
                clone = $clone
                debug = [ Debug ]
            }
            pending = [ $($pending)* ]
            fields = $fields
            rest = $rest
        }
    };

    // A 2nd `#[derive(Debug)]` on the `table` (the `debug` slot is already full) - reject.
    // (A real derive would get rustc's duplicate-impl error; the slot must produce the
    // equivalent.)
    (
        @table_attrs
        config = {
            key_ty = $_key_ty:tt
            key_name = $_key_name:tt
            vis = $_vis:tt
            table_name = $_table_name:tt
            table_lifetimes = $_table_lifetimes:tt
            table_attrs = $_table_attrs:tt
            struct_ty = $_struct_ty:tt
            struct_name = $_struct_name:tt
            struct_lifetimes = $_struct_lifetimes:tt
            struct_attrs = $_struct_attrs:tt
            clone = $_clone:tt
            debug = [ $_debug:ident ]
        }
        pending = [
            [ derive ( Debug $(,)? ) ]
            $($_pending:tt)*
        ]
        fields = $_fields:tt
        rest = $_rest:tt
    ) => {
        ::std::compile_error!("duplicate `#[derive(Debug)]` attribute");
    };

    // A `#[derive(...)]` with more than one derive (e.g. `#[derive(Clone, Debug)]`) -
    // split off the first derive into its own attribute, and re-queue both halves, so
    // the rules above and below classify each derive separately.
    //
    // `$first:ident`, not `:path`: an `ident` capture stays transparent, so the re-queued
    // `[ derive ( $first ) ]` still matches the literal `Clone` / `Debug` in the rules
    // above. (A `:path` capture would become an opaque fragment, which they cannot match.
    // Path derives like `serde::Serialize` fall through to the rule below and are
    // rejected - only `Clone` and `Debug` are accepted anyway.)
    (
        @table_attrs
        config = $config:tt
        pending = [
            [ derive ( $first:ident , $($more:tt)+ ) ]
            $($pending:tt)*
        ]
        fields = $fields:tt
        rest = $rest:tt
    ) => {
        $crate::multi_vec! {
            @table_attrs
            config = $config
            pending = [
                [ derive ( $first ) ]
                [ derive ( $($more)+ ) ]
                $($pending)*
            ]
            fields = $fields
            rest = $rest
        }
    };

    // Any other derive on the `table` - reject. The table type wraps a `MultiVec`, which
    // supports only `Clone` and `Debug`.
    // (Must come after the `Clone` / `Debug` rules and the splitting rule above, which
    // this rule would also match.)
    (
        @table_attrs
        config = $_config:tt
        pending = [
            [ derive ( $($derives:tt)* ) ]
            $($_pending:tt)*
        ]
        fields = $_fields:tt
        rest = $_rest:tt
    ) => {
        ::std::compile_error!(::std::concat!(
            "`multi_vec!` tables accept only `#[derive(Clone)]` and `#[derive(Debug)]`, \
             not `#[derive(",
            ::std::stringify!($($derives)*),
            ")]`",
        ));
    };

    // Any other attribute on the `table` (doc comments etc.) - pass it through to the
    // generated table type.
    // (Must come after the derive rules above: this rule would also match a derive.)
    (
        @table_attrs
        config = {
            key_ty = $key_ty:tt
            key_name = $key_name:tt
            vis = $vis:tt
            table_name = $table_name:tt
            table_lifetimes = $table_lifetimes:tt
            table_attrs = [ $($table_attrs:tt)* ]
            struct_ty = $struct_ty:tt
            struct_name = $struct_name:tt
            struct_lifetimes = $struct_lifetimes:tt
            struct_attrs = $struct_attrs:tt
            clone = $clone:tt
            debug = $debug:tt
        }
        pending = [
            [ $($attr:tt)* ]
            $($pending:tt)*
        ]
        fields = $fields:tt
        rest = $rest:tt
    ) => {
        $crate::multi_vec! {
            @table_attrs
            config = {
                key_ty = $key_ty
                key_name = $key_name
                vis = $vis
                table_name = $table_name
                table_lifetimes = $table_lifetimes
                table_attrs = [ $($table_attrs)* #[$($attr)*] ]
                struct_ty = $struct_ty
                struct_name = $struct_name
                struct_lifetimes = $struct_lifetimes
                struct_attrs = $struct_attrs
                clone = $clone
                debug = $debug
            }
            pending = [ $($pending)* ]
            fields = $fields
            rest = $rest
        }
    };

    // All table attributes classified - start normalizing the fields.
    (
        @table_attrs
        config = $config:tt
        pending = []
        fields = $fields:tt
        rest = $rest:tt
    ) => {
        $crate::multi_vec! {
            @normalize
            config = $config
            fields = $fields
            rest = $rest
        }
    };

    // Start normalizing the next field: build its normalized record up front, in the
    // exact form `@generate` consumes, then refine it in place as the `@field`
    // rules classify the attributes.
    //
    // `ctx` bundles the state which the classify rules carry through unexamined.
    // `field_plural_name` starts out empty, meaning "no `#[plural(...)]` seen yet";
    // if it is still empty once all attributes are classified, a dedicated rule fills
    // in the default plural name (append `s`).
    (
        @normalize
        config = $config:tt
        fields = $fields:tt
        rest = [
            $(# [ $($field_attr:tt)* ])*
            $field_vis:vis $field_name:ident: $field_ty:ty
            $(, $($rest:tt)*)?
        ]
    ) => {
        $crate::multi_vec! {
            @field
            ctx = {
                config = $config
                fields = $fields
                rest = [ $($($rest)*)? ]
            }
            field = {
                field_vis = [ $field_vis ]
                field_name = [ $field_name ]
                field_ty = [ $field_ty ]
                field_plural_name = []
                field_plural_name_mut = []
                field_comments = []
            }
            pending = [ $([ $($field_attr)* ])* ]
        }
    };

    // Classify a doc comment (`#[doc ...]`): collect it into `field_comments`, which
    // `@generate` emits onto the struct's field.
    (
        @field
        ctx = $ctx:tt
        field = {
            field_vis = $field_vis:tt
            field_name = $field_name:tt
            field_ty = $field_ty:tt
            field_plural_name = $field_plural_name:tt
            field_plural_name_mut = $field_plural_name_mut:tt
            field_comments = [ $($field_comments:tt)* ]
        }
        pending = [
            [ doc $($body:tt)* ]
            $($pending:tt)*
        ]
    ) => {
        $crate::multi_vec! {
            @field
            ctx = $ctx
            field = {
                field_vis = $field_vis
                field_name = $field_name
                field_ty = $field_ty
                field_plural_name = $field_plural_name
                field_plural_name_mut = $field_plural_name_mut
                field_comments = [
                    $($field_comments)*
                    #[doc $($body)*]
                ]
            }
            pending = [ $($pending)* ]
        }
    };

    // Classify a `#[plural(...)]` attribute: set the plural name.
    (
        @field
        ctx = $ctx:tt
        field = {
            field_vis = $field_vis:tt
            field_name = $field_name:tt
            field_ty = $field_ty:tt
            field_plural_name = []
            field_plural_name_mut = []
            field_comments = $field_comments:tt
        }
        pending = [
            [ plural ( $field_plural_name:ident ) ]
            $($pending:tt)*
        ]
    ) => {
        $crate::multi_vec! {
            @field
            ctx = $ctx
            field = {
                field_vis = $field_vis
                field_name = $field_name
                field_ty = $field_ty
                field_plural_name = [ $field_plural_name ]
                field_plural_name_mut = [ [<$field_plural_name _mut>] ]
                field_comments = $field_comments
            }
            pending = [ $($pending)* ]
        }
    };

    // A 2nd `#[plural(...)]` on the same field (`field_plural_name` is already full) - reject.
    (
        @field
        ctx = $_ctx:tt
        field = {
            field_vis = $_field_vis:tt
            field_name = $_field_name:tt
            field_ty = $_field_ty:tt
            field_plural_name = [ $_field_plural_name:tt ]
            field_plural_name_mut = $_field_plural_name_mut:tt
            field_comments = $_field_comments:tt
        }
        pending = [
            [ plural $($_body:tt)* ]
            $($_pending:tt)*
        ]
    ) => {
        ::std::compile_error!("duplicate `#[plural(...)]` attribute");
    };

    // A malformed `#[plural ...]` (not a single identifier in parentheses) - reject.
    // (A well-formed `#[plural(...)]` was already consumed by the rules above.)
    (
        @field
        ctx = $_ctx:tt
        field = $_field:tt
        pending = [
            [ plural $($_body:tt)* ]
            $($_pending:tt)*
        ]
    ) => {
        ::std::compile_error!(
            "`#[plural(...)]` expects a single identifier, e.g. `#[plural(all_flags)]`"
        );
    };

    // Any other attribute - reject. It would apply only to the generated struct's field,
    // not to the fields of the generated view types, which is a footgun (`#[cfg(...)]`
    // especially so).
    (
        @field
        ctx = $_ctx:tt
        field = $_field:tt
        pending = [
            [ $($body:tt)* ]
            $($_pending:tt)*
        ]
    ) => {
        ::std::compile_error!(::std::concat!(
            "`multi_vec!` fields accept only doc comments and `#[plural(...)]` on struct fields, not `#[",
            ::std::stringify!($($body)*),
            "]`",
        ));
    };

    // All attributes classified, but no `#[plural(...)]` was given - fill in the default
    // plural name (append `s`), and re-run, so the rule below pushes the field.
    (
        @field
        ctx = $ctx:tt
        field = {
            field_vis = $field_vis:tt
            field_name = [ $field_name:ident ]
            field_ty = $field_ty:tt
            field_plural_name = []
            field_plural_name_mut = []
            field_comments = $field_comments:tt
        }
        pending = []
    ) => {
        $crate::multi_vec! {
            @field
            ctx = $ctx
            field = {
                field_vis = $field_vis
                field_name = [ $field_name ]
                field_ty = $field_ty
                field_plural_name = [ [<$field_name s>] ]
                field_plural_name_mut = [ [<$field_name s_mut>] ]
                field_comments = $field_comments
            }
            pending = []
        }
    };

    // All attributes classified - push the completed record, and hand the remaining
    // fields back to `@normalize`.
    //
    // Must come after the fill rule above: this rule would also match a record whose
    // `field_plural_name` is still empty, so such records must be filled first.
    (
        @field
        ctx = {
            config = $config:tt
            fields = [ $($fields:tt)* ]
            rest = $rest:tt
        }
        field = $field:tt
        pending = []
    ) => {
        $crate::multi_vec! {
            @normalize
            config = $config
            fields = [
                $($fields)*
                $field
            ]
            rest = $rest
        }
    };

    // All fields normalized, but there are none - reject with a clear error.
    // (Without this rule, an empty field list fails to match any rule, producing an
    // inscrutable macro-expansion error.)
    (
        @normalize
        config = $_config:tt
        fields = []
        rest = []
    ) => {
        ::std::compile_error!("`multi_vec!` requires at least one field");
    };

    // All fields normalized - build the derived config slots, and generate the output.
    // (Must come after the empty-fields rule above, which would also match here.)
    //
    // The lifetime lists are precompiled here into the token groups `@generate` emits
    // (see the slot comments on the `@generate` rule), so the nested-optional shape -
    // `$(< $($table_lifetime),+ >)?` etc., emitting nothing when the list is empty -
    // is written once per form here, instead of at every one of `@generate`'s many
    // use sites.
    (
        @normalize
        config = {
            // The key's tokens, re-matched as one `:ty` fragment - see the key note on
            // the entry rule.
            key_ty = [ $key_ty:ty ]
            key_name = [ $key_name:ident ]
            vis = $vis:tt
            table_name = [ $table_name:ident ]
            table_lifetimes = [ $( $($table_lifetime:lifetime),+ )? ]
            table_attrs = $table_attrs:tt
            struct_ty = $struct_ty:tt
            struct_name = [ $struct_name:ident ]
            struct_lifetimes = [ $( $($struct_lifetime:lifetime),+ )? ]
            struct_attrs = $struct_attrs:tt
            clone = $clone:tt
            debug = $debug:tt
        }
        fields = $fields:tt
        rest = []
    ) => {
        $crate::multi_vec! {
            @generate
            config = {
                key_ty = [ $key_ty ]
                key_name = [ $key_name ]
                id = [ [<$key_name:snake>] ]
                vis = $vis
                table_name = [ $table_name ]
                table_lifetimes = [ $( $($table_lifetime),+ )? ]
                table_generics = [ $(< $($table_lifetime),+ >)? ]
                table_ref_generics = [ < 'v $(, $($table_lifetime),+ )? > ]
                table_attrs = $table_attrs
                struct_ty = $struct_ty
                struct_name = [ $struct_name ]
                struct_generics = [ $(< $($struct_lifetime),+ >)? ]
                struct_ref_generics = [ < 'v $(, $($struct_lifetime),+ )? > ]
                struct_attrs = $struct_attrs
                // The two reference-view types, name and generics together.
                struct_ref_ty = [ [<$struct_name Ref>] < 'v $(, $($struct_lifetime),+ )? > ]
                struct_mut_ty = [ [<$struct_name Mut>] < 'v $(, $($struct_lifetime),+ )? > ]
                // The two slice-view types. Unlike the reference views, these take the
                // table's lifetimes: their fields name the key (`IndexSlice<$key_ty, ...>`).
                struct_slices_ty = [ [<$struct_name Slices>] < 'v $(, $($table_lifetime),+ )? > ]
                struct_slices_mut_ty = [ [<$struct_name SlicesMut>] < 'v $(, $($table_lifetime),+ )? > ]
                // The three iterator types, name and generics together (the wrappers take
                // the struct's lifetimes). `paste!` concatenates the `[<...>]` in `@generate`.
                table_iter_ty = [ [<$table_name Iter>] < 'v $(, $($struct_lifetime),+ )? > ]
                table_iter_mut_ty = [ [<$table_name IterMut>] < 'v $(, $($struct_lifetime),+ )? > ]
                table_into_iter_ty = [ [<$table_name IntoIter>] $(< $($struct_lifetime),+ >)? ]
                clone = $clone
                debug = $debug
            }
            fields = $fields
        }
    };

    // Generate the output: the struct, the table type, the view types, the iterators, and all their impls.
    //
    // Lifetime plumbing: the `*_generics` / `*_lifetimes*` slots are precompiled token
    // groups `@normalize` builds from the two declarations' lifetime lists (see the
    // slot comments below); each is empty - emitting nothing - when the corresponding
    // list is empty. The table type and the iterator types take the table's lifetimes;
    // the struct and the view types take the struct's, as do the `Fields` /
    // `CloneFields` impls. The item type (`$struct_ty`, e.g. `Things<'a, 'b>`) is how the
    // table instantiates the struct's lifetimes from its own; where table-side code
    // needs a view type, it goes through `<$struct_ty as Fields>::Ref<'v>` etc. (see the
    // type aliases below), which substitutes the struct's lifetimes correctly by
    // construction.
    //
    // The two declarations must use the same lifetime names
    // (`assert_item_type_is_the_struct` below rejects a mismatch), so the field types -
    // written with the struct's names - can appear directly in the table impl's
    // per-field methods.
    //
    // Comments below show the content/expansion for this example invocation:
    //
    // ```
    // multi_vec! {
    //     /// Lots of things.
    //     #[derive(Clone, Debug)]
    //     pub table Things<'k, 'a, ThingId<'k>, Thing<'a>>;
    //
    //     /// The thing itself.
    //     #[derive(Clone, Debug)]
    //     pub struct Thing<'a> {
    //         /// ID of the thing
    //         pub(crate) id: u32,
    //         /// Name of the thing
    //         pub name: &'a str,
    //     }
    // }
    // ```
    (
        @generate
        config = {
            // `ThingId<'k>`
            key_ty = [ $key_ty:ty ]
            // `ThingId`
            key_name = [ $key_name:ident ]
            // `thing_id`
            id = [ $id:tt ]
            // `pub`
            vis = [ $vis:vis ]

            // `Things`
            table_name = [ $table_name:ident ]
            // `'k, 'a` (bare - the `use<...>` capture list)
            table_lifetimes = [ $($table_lifetimes:tt)* ]
            // `<'k, 'a>` (the table type, its `impl` binders, and `IntoIter`)
            table_generics = [ $($table_generics:tt)* ]
            // `<'v, 'k, 'a>` (the views, iterators, and borrowing methods)
            table_ref_generics = [ $($table_ref_generics:tt)* ]
            // `/// Lots of things.` `#[derive(Clone)]`
            table_attrs = [ $(#[$table_attr:meta])* ]

            // `Thing<'a>`
            struct_ty = [ $struct_ty:ty ]
            // `Thing`
            struct_name = [ $struct_name:ident ]
            // `<'a>`
            struct_generics = [ $($struct_generics:tt)* ]
            // `<'v, 'a>`
            struct_ref_generics = [ $($struct_ref_generics:tt)* ]
            // `/// The thing itself.` `#[derive(Clone, Debug)]`
            struct_attrs = [ $(#[$struct_attr:meta])* ]

            // `ThingRef<'v, 'a>`
            struct_ref_ty = [ $($struct_ref_ty:tt)* ]
            // `ThingMut<'v, 'a>`
            struct_mut_ty = [ $($struct_mut_ty:tt)* ]
            // `ThingSlices<'v, 'k, 'a>` (the slice views take the table's lifetimes)
            struct_slices_ty = [ $($struct_slices_ty:tt)* ]
            // `ThingSlicesMut<'v, 'k, 'a>`
            struct_slices_mut_ty = [ $($struct_slices_mut_ty:tt)* ]

            // `ThingsIter<'v, 'a>`
            table_iter_ty = [ $($table_iter_ty:tt)* ]
            // `ThingsIterMut<'v, 'a>`
            table_iter_mut_ty = [ $($table_iter_mut_ty:tt)* ]
            // `ThingsIntoIter<'a>`
            table_into_iter_ty = [ $($table_into_iter_ty:tt)* ]

            // `Clone` (empty if not derived - drives `@generate_clone`)
            clone = $clone:tt
            // `Debug` (empty if not derived - gates the `Debug` impls)
            debug = [ $($debug:ident)? ]
        }
        fields = [ $(
            {
                // `pub`
                field_vis = [ $field_vis:vis ]
                // `name`
                field_name = [ $field_name:ident ]
                // `&'a str`
                field_ty = [ $field_ty:ty ]
                // `[<name s>]` (`paste!` converts to `names`)
                field_plural_name = [ $field_plural_name:tt ]
                // `[<name s_mut>]` (`paste!` converts to `names_mut`)
                field_plural_name_mut = [ $field_plural_name_mut:tt ]
                // `/// Name of the thing`
                field_comments = [ $(#[$field_comment:meta])* ]
            }
        )+ ]
    ) => {
        $crate::multi_vec::__private::paste! {
            // ```
            // /// Lots of things.
            // #[derive(Clone)]
            // #[derive(Default)]
            // pub struct Things<'k, 'a> {
            //     inner: MultiVec<ThingId<'k>, Thing<'a>>,
            // }
            // ```
            $(#[$table_attr])*
            #[derive(Default)]
            #[allow(dead_code, clippy::allow_attributes)]
            $vis struct $table_name $($table_generics)* {
                inner: $crate::multi_vec::__private::MultiVec<$key_ty, $struct_ty>,
            }

            // ```
            // /// The thing itself.
            // #[derive(Clone, Debug)]
            // pub struct Thing<'a> {
            //     /// ID of the thing
            //     pub(crate) id: u32,
            //     /// Name of the thing
            //     pub name: &'a str,
            // }
            // ```
            $(#[$struct_attr])*
            #[allow(dead_code, clippy::allow_attributes)]
            $vis struct $struct_name $($struct_generics)* {
                $(
                    $(#[$field_comment])*
                    $field_vis $field_name: $field_ty,
                )+
            }

            // ```
            // /// References to one element of a [`Things`].
            // ///
            // /// Returned by [`Things::get`].
            // #[derive(Debug)]
            // pub struct ThingRef<'v, 'a> {
            //     /// ID of the thing
            //     pub(crate) id: &'v u32,
            //     /// Name of the thing
            //     pub name: &'v &'a str,
            // }
            // ```
            #[doc = concat!("References to one element of a [`", stringify!($table_name), "`].")]
            #[doc = ""]
            #[doc = concat!("Returned by [`", stringify!($table_name), "::get`].")]
            $(#[derive($debug)])?
            #[allow(dead_code, clippy::allow_attributes)]
            $vis struct $($struct_ref_ty)* {
                $(
                    $(#[$field_comment])*
                    $field_vis $field_name: &'v $field_ty,
                )+
            }

            // ```
            // /// Mutable references to one element of a [`Things`].
            // ///
            // /// Returned by [`Things::get_mut`].
            // #[derive(Debug)]
            // pub struct ThingMut<'v, 'a> {
            //     /// ID of the thing
            //     pub(crate) id: &'v mut u32,
            //     /// Name of the thing
            //     pub name: &'v mut &'a str,
            // }
            // ```
            #[doc = concat!("Mutable references to one element of a [`", stringify!($table_name), "`].")]
            #[doc = ""]
            #[doc = concat!("Returned by [`", stringify!($table_name), "::get_mut`].")]
            $(#[derive($debug)])?
            #[allow(dead_code, clippy::allow_attributes)]
            $vis struct $($struct_mut_ty)* {
                $(
                    $(#[$field_comment])*
                    $field_vis $field_name: &'v mut $field_ty,
                )+
            }

            // ```
            // /// Slices over each field array of a [`Things`].
            // ///
            // /// Returned by [`Things::slices`].
            // #[derive(Debug)]
            // pub struct ThingSlices<'v, 'k, 'a> {
            //     pub(crate) ids: &'v IndexSlice<ThingId<'k>, [u32]>,
            //     pub names: &'v IndexSlice<ThingId<'k>, [&'a str]>,
            // }
            // ```
            #[doc = concat!("Slices over each field array of a [`", stringify!($table_name), "`].")]
            #[doc = ""]
            #[doc = concat!("Returned by [`", stringify!($table_name), "::slices`].")]
            $(#[derive($debug)])?
            #[allow(dead_code, clippy::allow_attributes)]
            $vis struct $($struct_slices_ty)* {
                $(
                    $field_vis $field_plural_name: &'v $crate::multi_vec::__private::IndexSlice<$key_ty, [$field_ty]>,
                )+
            }

            // ```
            // /// Mutable slices over each field array of a [`Things`].
            // ///
            // /// Returned by [`Things::slices_mut`].
            // #[derive(Debug)]
            // pub struct ThingSlicesMut<'v, 'k, 'a> {
            //     pub(crate) ids: &'v mut IndexSlice<ThingId<'k>, [u32]>,
            //     pub names: &'v mut IndexSlice<ThingId<'k>, [&'a str]>,
            // }
            // ```
            #[doc = concat!("Mutable slices over each field array of a [`", stringify!($table_name), "`].")]
            #[doc = ""]
            #[doc = concat!("Returned by [`", stringify!($table_name), "::slices_mut`].")]
            $(#[derive($debug)])?
            #[allow(dead_code, clippy::allow_attributes)]
            $vis struct $($struct_slices_mut_ty)* {
                $(
                    $field_vis $field_plural_name: &'v mut $crate::multi_vec::__private::IndexSlice<$key_ty, [$field_ty]>,
                )+
            }

            // ```
            // /// Iterator over a [`Things`]'s elements, yielding a [`ThingRef`] for each.
            // ///
            // /// Returned by [`Things::iter`].
            // #[must_use = "iterators are lazy and do nothing unless consumed"]
            // pub struct ThingsIter<'v, 'a>(Iter<'v, Thing<'a>>);
            // ```
            #[doc = concat!("Iterator over a [`", stringify!($table_name), "`]'s elements, yielding a [`", stringify!($struct_name), "Ref`] for each.")]
            #[doc = ""]
            #[doc = concat!("Returned by [`", stringify!($table_name), "::iter`].")]
            #[must_use = "iterators are lazy and do nothing unless consumed"]
            #[allow(dead_code, clippy::allow_attributes)]
            $vis struct $($table_iter_ty)* (
                $crate::multi_vec::__private::Iter<'v, $struct_ty>,
            );

            // ```
            // /// Iterator over a [`Things`]'s elements, yielding a [`ThingMut`] for each.
            // ///
            // /// Returned by [`Things::iter_mut`].
            // #[must_use = "iterators are lazy and do nothing unless consumed"]
            // pub struct ThingsIterMut<'v, 'a>(IterMut<'v, Thing<'a>>);
            // ```
            #[doc = concat!("Iterator over a [`", stringify!($table_name), "`]'s elements, yielding a [`", stringify!($struct_name), "Mut`] for each.")]
            #[doc = ""]
            #[doc = concat!("Returned by [`", stringify!($table_name), "::iter_mut`].")]
            #[must_use = "iterators are lazy and do nothing unless consumed"]
            #[allow(dead_code, clippy::allow_attributes)]
            $vis struct $($table_iter_mut_ty)* (
                $crate::multi_vec::__private::IterMut<'v, $struct_ty>,
            );

            // ```
            // /// Iterator over a [`Things`]'s elements, yielding each element as an owned [`Thing`].
            // ///
            // /// Returned by [`Things`]'s [`IntoIterator`] impl.
            // #[must_use = "iterators are lazy and do nothing unless consumed"]
            // pub struct ThingsIntoIter<'a>(IntoIter<Thing<'a>>);
            // ```
            #[doc = concat!("Iterator over a [`", stringify!($table_name), "`]'s elements, yielding each element as an owned [`", stringify!($struct_name), "`].")]
            #[doc = ""]
            #[doc = concat!("Returned by [`", stringify!($table_name), "`]'s [`IntoIterator`] impl.")]
            #[must_use = "iterators are lazy and do nothing unless consumed"]
            #[allow(dead_code, clippy::allow_attributes)]
            $vis struct $($table_into_iter_ty)* (
                $crate::multi_vec::__private::IntoIter<$struct_ty>,
            );

            // All the impls, and the items only they need (imports, `FIELD_COUNT`,
            // the compile-time assertions, and the type aliases), live in this anonymous
            // `const` block. It keeps them out of the invoking scope, while the impls
            // still apply to the types above as normal (and rustdoc still documents them).
            // Only the types themselves must be defined outside it, to be nameable by the user.
            #[allow(
                dead_code,
                forgetting_copy_types,
                private_interfaces,
                clippy::extra_unused_lifetimes,
                clippy::inline_always,
                clippy::macro_metavars_in_unsafe,
                clippy::undocumented_unsafe_blocks,
                clippy::allow_attributes
            )]
            const _: () = {
                use ::std::{iter::FusedIterator, ptr::NonNull};

                use $crate::multi_vec::__private as __p;

                // Number of fields in the struct
                const FIELD_COUNT: usize = [$(stringify!($field_name)),+].len();

                // Verify that the item type named in the `table` declaration is the struct
                // declared below it, with the struct's own lifetimes applied. (Without this
                // check, a `table` declaration naming a *different* struct which also
                // implements `Fields` - e.g. another table's struct - would compile,
                // silently binding this table to the wrong element type.)
                //
                // Never called. A `*mut` pointer is invariant in its pointee, so returning
                // `item_ptr` requires the two types to be *exactly* equal, lifetimes
                // included. A wrong struct name fails with a type mismatch (E0308), a wrong
                // number of lifetimes with E0107, elided lifetimes with E0621, and swapped /
                // repeated / `'static` lifetime applications fail borrow checking
                // ("lifetime may not live long enough"). The fn declares the *struct's*
                // lifetime names, so a `table` declaration whose item type uses different
                // names fails with E0261 (undeclared lifetime) - the two declarations must
                // use the same names.
                //
                // ```
                // fn assert_item_type_is_the_struct<'a>(item_ptr: *mut Thing<'a>) -> *mut Thing<'a> {
                //     item_ptr
                // }
                // ```
                fn assert_item_type_is_the_struct$($struct_generics)*(
                    item_ptr: *mut $struct_ty,
                ) -> *mut $struct_name$($struct_generics)* {
                    item_ptr
                }

                // `impl<'k, 'a> Things<'k, 'a> {`
                impl$($table_generics)* $table_name$($table_generics)* {
                    #[doc = concat!("Maximum capacity of a `", stringify!($table_name), "`.")]
                    ///
                    /// Capacity is limited by the index type's range, and the maximum
                    /// allocation size (`isize::MAX` bytes).
                    pub const MAX_CAPACITY: usize = __p::MultiVec::<$key_ty, $struct_ty>::MAX_CAPACITY;

                    #[doc = concat!("Create a new empty `", stringify!($table_name), "`. Does not allocate.")]
                    #[inline(always)]
                    pub const fn new() -> Self {
                        Self { inner: __p::MultiVec::new() }
                    }

                    #[doc = concat!("Create a new `", stringify!($table_name), "` with capacity for `capacity` elements.")]
                    ///
                    /// Does not allocate if `capacity == 0`.
                    ///
                    /// # Panics
                    ///
                    /// Panics if `capacity` exceeds [`MAX_CAPACITY`].
                    ///
                    /// [`MAX_CAPACITY`]: Self::MAX_CAPACITY
                    #[inline(always)]
                    pub fn with_capacity(capacity: usize) -> Self {
                        Self { inner: __p::MultiVec::with_capacity(capacity) }
                    }

                    /// Returns the number of elements.
                    #[inline(always)]
                    pub fn len(&self) -> usize {
                        self.inner.len()
                    }

                    /// Returns `true` if there are no elements.
                    #[inline(always)]
                    pub fn is_empty(&self) -> bool {
                        self.inner.is_empty()
                    }

                    #[doc = concat!("Push a [`", stringify!($struct_name), "`] to this [`", stringify!($table_name), "`].")]
                    #[doc = concat!("Returns the [`", stringify!($key_name), "`] of the new element.")]
                    ///
                    /// # Panics
                    ///
                    /// Panics if already full to [`MAX_CAPACITY`].
                    ///
                    /// [`MAX_CAPACITY`]: Self::MAX_CAPACITY
                    #[inline(always)]
                    // `pub fn push(&mut self, value: Thing<'a>) -> ThingId<'k> {`
                    pub fn push(&mut self, value: $struct_ty) -> $key_ty {
                        self.inner.push(value)
                    }

                    /// Reserve capacity for at least `additional` more elements.
                    ///
                    /// # Panics
                    ///
                    /// Panics if the required capacity exceeds [`MAX_CAPACITY`].
                    ///
                    /// [`MAX_CAPACITY`]: Self::MAX_CAPACITY
                    #[inline(always)]
                    pub fn reserve(&mut self, additional: usize) {
                        self.inner.reserve(additional);
                    }

                    #[doc = concat!("Get references to the element at `", stringify!($id), "`.")]
                    ///
                    /// # Panics
                    ///
                    #[doc = concat!("Panics if `", stringify!($id), "` is out of bounds.")]
                    #[inline(always)]
                    // `pub fn get<'v>(&'v self, thing_id: ThingId<'k>) -> ThingRef<'v, 'a> {`
                    pub fn get<'v>(&'v self, $id: $key_ty) -> $($struct_ref_ty)* {
                        self.inner.get($id)
                    }

                    #[doc = concat!("Get mutable references to the element at `", stringify!($id), "`.")]
                    ///
                    /// # Panics
                    ///
                    #[doc = concat!("Panics if `", stringify!($id), "` is out of bounds.")]
                    #[inline(always)]
                    // `pub fn get_mut<'v>(&'v mut self, thing_id: ThingId<'k>) -> ThingMut<'v, 'a> {`
                    pub fn get_mut<'v>(&'v mut self, $id: $key_ty) -> $($struct_mut_ty)* {
                        self.inner.get_mut($id)
                    }

                    #[doc = concat!("Get references to the element at `", stringify!($id), "`,")]
                    #[doc = concat!("without checking that `", stringify!($id), "` is in bounds.")]
                    ///
                    /// # SAFETY
                    ///
                    #[doc = concat!("`", stringify!($id), "` must be in bounds - less than [`len`](Self::len).")]
                    #[inline(always)]
                    // `pub unsafe fn get_unchecked<'v>(&'v self, thing_id: ThingId<'k>) -> ThingRef<'v, 'a> {`
                    pub unsafe fn get_unchecked<'v>(&'v self, $id: $key_ty) -> $($struct_ref_ty)* {
                        // SAFETY: Caller guarantees `$id` is in bounds`
                        unsafe { self.inner.get_unchecked($id) }
                    }

                    #[doc = concat!("Get mutable references to the element at `", stringify!($id), "`,")]
                    #[doc = concat!("without checking that `", stringify!($id), "` is in bounds.")]
                    ///
                    /// # SAFETY
                    ///
                    #[doc = concat!("`", stringify!($id), "` must be in bounds - less than [`len`](Self::len).")]
                    #[inline(always)]
                    // `pub unsafe fn get_unchecked_mut<'v>(&'v mut self, thing_id: ThingId<'k>) -> ThingMut<'v, 'a> {`
                    pub unsafe fn get_unchecked_mut<'v>(&'v mut self, $id: $key_ty) -> $($struct_mut_ty)* {
                        // SAFETY: Caller guarantees `$id` is in bounds
                        unsafe { self.inner.get_unchecked_mut($id) }
                    }

                    /// Get slices over all field arrays.
                    #[inline(always)]
                    // `pub fn slices<'v>(&'v self) -> ThingSlices<'v, 'k, 'a> {`
                    pub fn slices<'v>(&'v self) -> $($struct_slices_ty)* {
                        self.inner.slices()
                    }

                    /// Get mutable slices over all field arrays.
                    #[inline(always)]
                    // `pub fn slices_mut<'v>(&'v mut self) -> ThingSlicesMut<'v, 'k, 'a> {`
                    pub fn slices_mut<'v>(&'v mut self) -> $($struct_slices_mut_ty)* {
                        self.inner.slices_mut()
                    }

                    /// Iterate over all valid indices.
                    ///
                    /// The returned iterator does not borrow `self` (the `use<...>` clause omits
                    /// the `&self` lifetime, opting out of edition 2024's capture-everything default).
                    /// It snapshots `len`, so the table can be mutated while the iterator lives.
                    /// IDs of elements pushed after the `iter_ids` call are not included.
                    #[inline(always)]
                    // `pub fn iter_ids(&self) -> impl ExactSizeIterator<Item = ThingId<'k>> + FusedIterator + use<'k, 'a> {`
                    pub fn iter_ids(&self) -> impl ExactSizeIterator<Item = $key_ty> + FusedIterator + use<$($table_lifetimes)*> {
                        self.inner.iter_ids()
                    }

                    #[doc = concat!("Iterate over all elements, yielding a [`", stringify!($struct_name), "Ref`] (references to every field) for each.")]
                    #[inline(always)]
                    // ```
                    // pub fn iter<'v>(&'v self) -> ThingsIter<'v, 'a> {
                    //     ThingsIter(self.inner.iter())
                    // }
                    // ```
                    pub fn iter<'v>(&'v self) -> $($table_iter_ty)* {
                        [<$table_name Iter>](self.inner.iter())
                    }

                    #[doc = concat!("Iterate over all elements, yielding a [`", stringify!($struct_name), "Mut`] (mutable references to every field) for each.")]
                    #[inline(always)]
                    // ```
                    // pub fn iter_mut<'v>(&'v mut self) -> ThingsIterMut<'v, 'a> {
                    //     ThingsIterMut(self.inner.iter_mut())
                    // }
                    // ```
                    pub fn iter_mut<'v>(&'v mut self) -> $($table_iter_mut_ty)* {
                        [<$table_name IterMut>](self.inner.iter_mut())
                    }

                    #[doc = concat!("Iterate over all elements, yielding each element's [`", stringify!($key_name), "`] and a [`", stringify!($struct_name), "Ref`].")]
                    #[inline(always)]
                    // `pub fn iter_enumerated<'v>(&'v self) -> impl ExactSizeIterator<Item = (ThingId<'k>, ThingRef<'v, 'a>)> + FusedIterator {`
                    pub fn iter_enumerated<'v>(&'v self) -> impl ExactSizeIterator<Item = ($key_ty, $($struct_ref_ty)*)> + FusedIterator {
                        self.inner.iter_enumerated()
                    }

                    #[doc = concat!("Iterate over all elements, yielding each element's [`", stringify!($key_name), "`] and a [`", stringify!($struct_name), "Mut`].")]
                    #[inline(always)]
                    // `pub fn iter_mut_enumerated<'v>(&'v mut self) -> impl ExactSizeIterator<Item = (ThingId<'k>, ThingMut<'v, 'a>)> + FusedIterator {`
                    pub fn iter_mut_enumerated<'v>(&'v mut self) -> impl ExactSizeIterator<Item = ($key_ty, $($struct_mut_ty)*)> + FusedIterator {
                        self.inner.iter_mut_enumerated()
                    }

                    #[doc = concat!("Consume the table, yielding each element's [`", stringify!($key_name), "`] and the element as an owned [`", stringify!($struct_name), "`].")]
                    #[inline(always)]
                    // `pub fn into_iter_enumerated(self) -> impl ExactSizeIterator<Item = (ThingId<'k>, Thing<'a>)> + FusedIterator {`
                    pub fn into_iter_enumerated(self) -> impl ExactSizeIterator<Item = ($key_ty, $struct_ty)> + FusedIterator {
                        self.inner.into_iter_enumerated()
                    }

                    // Per-field accessor methods e.g. `.parent_id(id)` / `.parent_id_mut(id)`
                    $(
                        #[doc = concat!("Get reference to the `", stringify!($field_name), "` field of the element at `", stringify!($id), "`.")]
                        ///
                        /// # Panics
                        ///
                        #[doc = concat!("Panics if `", stringify!($id), "` is out of bounds.")]
                        #[inline(always)]
                        // `pub fn name(&self, thing_id: ThingId<'k>) -> &&'a str {` (one per field)
                        $field_vis fn $field_name(&self, $id: $key_ty) -> &$field_ty {
                            self.get($id).$field_name
                        }

                        #[doc = concat!("Get mutable reference to the `", stringify!($field_name), "` field of the element at `", stringify!($id), "`.")]
                        ///
                        /// # Panics
                        ///
                        #[doc = concat!("Panics if `", stringify!($id), "` is out of bounds.")]
                        #[inline(always)]
                        // `pub fn name_mut(&mut self, thing_id: ThingId<'k>) -> &mut &'a str {`
                        $field_vis fn [<$field_name _mut>](&mut self, $id: $key_ty) -> &mut $field_ty {
                            self.get_mut($id).$field_name
                        }
                    )+

                    // Per-field slice methods e.g. `.parent_ids(id)` / `.parent_ids_mut(id)`
                    $(
                        #[doc = concat!("Get slice of `", stringify!($field_name), "` fields of all elements.")]
                        #[inline(always)]
                        // `pub fn names(&self) -> &IndexSlice<ThingId<'k>, [&'a str]> {`
                        $field_vis fn $field_plural_name(&self) -> &__p::IndexSlice<$key_ty, [$field_ty]> {
                            self.slices().$field_plural_name
                        }

                        #[doc = concat!("Get mutable slice of `", stringify!($field_name), "` fields of all elements.")]
                        #[inline(always)]
                        // `pub fn names_mut(&mut self) -> &mut IndexSlice<ThingId<'k>, [&'a str]> {`
                        $field_vis fn $field_plural_name_mut(&mut self) -> &mut __p::IndexSlice<$key_ty, [$field_ty]> {
                            self.slices_mut().$field_plural_name
                        }
                    )+
                }

                // `impl<'k, 'a> IntoIterator for Things<'k, 'a> {`
                impl$($table_generics)* IntoIterator for $table_name$($table_generics)* {
                    type Item = $struct_ty;
                    type IntoIter = $($table_into_iter_ty)*;

                    #[doc = concat!("Consume the table, yielding each element as an owned [`", stringify!($struct_name), "`], reassembled from its stored fields.")]
                    #[inline(always)]
                    // ```
                    // fn into_iter(self) -> ThingsIntoIter<'a> {
                    //     ThingsIntoIter(self.inner.into_iter())
                    // }
                    // ```
                    fn into_iter(self) -> $($table_into_iter_ty)* {
                        [<$table_name IntoIter>](self.inner.into_iter())
                    }
                }

                // `impl<'v, 'k, 'a> IntoIterator for &'v Things<'k, 'a> {`
                impl$($table_ref_generics)* IntoIterator for &'v $table_name$($table_generics)* {
                    type Item = $($struct_ref_ty)*;
                    type IntoIter = $($table_iter_ty)*;

                    #[inline(always)]
                    // `fn into_iter(self) -> ThingsIter<'v, 'a> {`
                    fn into_iter(self) -> $($table_iter_ty)* {
                        self.iter()
                    }
                }

                // `impl<'v, 'k, 'a> IntoIterator for &'v mut Things<'k, 'a> {`
                impl$($table_ref_generics)* IntoIterator for &'v mut $table_name$($table_generics)* {
                    type Item = $($struct_mut_ty)*;
                    type IntoIter = $($table_iter_mut_ty)*;

                    #[inline(always)]
                    // `fn into_iter(self) -> ThingsIterMut<'v, 'a> {`
                    fn into_iter(self) -> $($table_iter_mut_ty)* {
                        self.iter_mut()
                    }
                }

                // Implement `Debug` on the table type if `#[derive(Debug)]` is present.
                // Prints as a map from ID to element e.g. `{ 0: ScopeRef { ... }, 1: ... }`.
                // `@generate_debug` expands to nothing when `debug` is empty.
                // (A sub-rule, not a `$( ... )?` group gated on the `debug` slot: all
                // metavariables directly under a repetition must repeat the same number
                // of times, so the generics lists cannot appear under a `$debug`-driven
                // repetition - `$debug` repeats once, the lists per lifetime.)
                //
                // Braces, not brackets, delimit the generics slots here (and on
                // `@generate_clone` below): this invocation's tokens pass through the
                // surrounding `paste!`, and a bracketed generics list would start with
                // `[ <`, which `paste!` treats as its own `[< ... >]` paste syntax and
                // mangles.
                $crate::multi_vec! {
                    @generate_debug
                    debug = [ $($debug)? ]
                    table_name = [ $table_name ]
                    struct_name = [ $struct_name ]
                    table_generics = { $($table_generics)* }
                    struct_generics = { $($struct_generics)* }
                }

                // The struct's lifetime params (if any) are declared by `field_layouts`, and
                // erased when it is called in `SHAPE`'s initializer (a `Layout` does not
                // depend on lifetimes). `SHAPE` must stay a free const: free consts are
                // evaluated when checked, so `Shape::new`'s compile-time rejection of
                // all-zero-sized field sets fires when the table is *defined*.
                // An associated const is only evaluated when used.
                //
                // ```
                // const fn field_layouts<'a>() -> [Layout; FIELD_COUNT] {
                //     [Layout::new::<u32>(), Layout::new::<&'a str>()]
                // }
                // ```
                const fn field_layouts$($struct_generics)*() -> [__p::Layout; FIELD_COUNT] {
                    [ $( __p::Layout::new::<$field_ty>() ),+ ]
                }
                const SHAPE: __p::Shape<[usize; FIELD_COUNT]> = __p::Shape::new(field_layouts());

                // Implement `Fields` on the struct type (e.g. `Scope`).
                // These methods are a thin layer of type casting.
                //
                // They all either:
                // 1. Convert `NonNull<u8>` pointers to typed references, or
                // 2. Call typed methods.
                //
                // The view types here are the raw generated names (not the `Ref` etc.
                // aliases above, which take the *table's* lifetimes): this impl is
                // parameterized by the *struct's* own lifetimes.
                //
                // `unsafe impl<'a> Fields for Thing<'a> {`
                unsafe impl$($struct_generics)* __p::Fields for $struct_ty {
                    // `type Ref<'v> = ThingRef<'v, 'a> where Self: 'v;`
                    type Ref<'v> = $($struct_ref_ty)* where Self: 'v;

                    // `type Mut<'v> = ThingMut<'v, 'a> where Self: 'v;`
                    type Mut<'v> = $($struct_mut_ty)* where Self: 'v;

                    type Array<T: Copy> = [T; FIELD_COUNT];

                    const SHAPE: __p::Shape<[usize; FIELD_COUNT]> = SHAPE;

                    // ```
                    // unsafe fn write(self, ptrs: [NonNull<u8>; FIELD_COUNT]) {
                    //     let [id, name] = ptrs;
                    //     unsafe {
                    //         id.cast::<u32>().write(self.id);
                    //         name.cast::<&'a str>().write(self.name);
                    //     }
                    // }
                    // ```
                    #[inline]
                    unsafe fn write(self, ptrs: [NonNull<u8>; FIELD_COUNT]) {
                        let [ $($field_name),+ ] = ptrs;
                        unsafe {
                            $(
                                $field_name.cast::<$field_ty>().write(self.$field_name);
                            )+
                        }
                    }

                    // ```
                    // unsafe fn create_owned(ptrs: [NonNull<u8>; FIELD_COUNT]) -> Self {
                    //     let [id, name] = ptrs;
                    //     unsafe {
                    //         Self {
                    //             id: id.cast::<u32>().read(),
                    //             name: name.cast::<&'a str>().read(),
                    //         }
                    //     }
                    // }
                    // ```
                    #[inline]
                    unsafe fn create_owned(ptrs: [NonNull<u8>; FIELD_COUNT]) -> Self {
                        let [ $($field_name),+ ] = ptrs;
                        unsafe {
                            Self {
                                $(
                                    $field_name: $field_name.cast::<$field_ty>().read(),
                                )+
                            }
                        }
                    }

                    // ```
                    // unsafe fn create_ref<'v>(ptrs: [NonNull<u8>; FIELD_COUNT]) -> ThingRef<'v, 'a> {
                    //     let [id, name] = ptrs;
                    //     unsafe {
                    //         ThingRef {
                    //             id: id.cast::<u32>().as_ref(),
                    //             name: name.cast::<&'a str>().as_ref(),
                    //         }
                    //     }
                    // }
                    // ```
                    #[inline]
                    unsafe fn create_ref<'v>(ptrs: [NonNull<u8>; FIELD_COUNT]) -> $($struct_ref_ty)* {
                        let [ $($field_name),+ ] = ptrs;
                        unsafe {
                            [<$struct_name Ref>] {
                                $(
                                    $field_name: $field_name.cast::<$field_ty>().as_ref(),
                                )+
                            }
                        }
                    }

                    // ```
                    // unsafe fn create_mut<'v>(ptrs: [NonNull<u8>; FIELD_COUNT]) -> ThingMut<'v, 'a> {
                    //     let [id, name] = ptrs;
                    //     unsafe {
                    //         ThingMut {
                    //             id: id.cast::<u32>().as_mut(),
                    //             name: name.cast::<&'a str>().as_mut(),
                    //         }
                    //     }
                    // }
                    // ```
                    #[inline]
                    unsafe fn create_mut<'v>(ptrs: [NonNull<u8>; FIELD_COUNT]) -> $($struct_mut_ty)* {
                        let [ $($field_name),+ ] = ptrs;
                        unsafe {
                            [<$struct_name Mut>] {
                                $(
                                    $field_name: $field_name.cast::<$field_ty>().as_mut(),
                                )+
                            }
                        }
                    }

                    // ```
                    // unsafe fn drop_columns(ptrs: [NonNull<u8>; FIELD_COUNT], len: usize) {
                    //     let [id, name] = ptrs;
                    //     unsafe {
                    //         drop_column::<u32>(id, len);
                    //         drop_column::<&'a str>(name, len);
                    //     }
                    // }
                    // ```
                    #[inline]
                    unsafe fn drop_columns(ptrs: [NonNull<u8>; FIELD_COUNT], len: usize) {
                        let [ $($field_name),+ ] = ptrs;
                        unsafe {
                            $(
                                __p::drop_column::<$field_ty>($field_name, len);
                            )+
                        }
                    }
                }

                // Implement `SliceFields` on the struct type (e.g. `Scope`).
                //
                // Unlike the `Fields` impl above (which binds only the struct's lifetimes,
                // `$struct_generics`), this binds the *table's* - `$table_generics`, i.e. the
                // key's lifetimes as well as the struct's - so the key type in the slice views
                // (`ThingSlices`'s `IndexSlice<ThingId<'k>, _>` fields) is nameable here. See
                // the `SliceFields` trait docs for why the views can't live on `Fields`.
                //
                // `unsafe impl<'k, 'a> SliceFields<ThingId<'k>> for Thing<'a> {`
                unsafe impl$($table_generics)* __p::SliceFields<$key_ty> for $struct_ty {
                    // `type Slices<'v> = ThingSlices<'v, 'k, 'a> where Self: 'v;`
                    type Slices<'v> = $($struct_slices_ty)* where Self: 'v;

                    // `type SlicesMut<'v> = ThingSlicesMut<'v, 'k, 'a> where Self: 'v;`
                    type SlicesMut<'v> = $($struct_slices_mut_ty)* where Self: 'v;

                    // ```
                    // unsafe fn create_slices<'v>(ptrs: [NonNull<u8>; FIELD_COUNT], len: usize) -> ThingSlices<'v, 'k, 'a> {
                    //     let [id, name] = ptrs;
                    //     unsafe {
                    //         ThingSlices {
                    //             ids: index_slice_from_raw_parts(id, len),
                    //             names: index_slice_from_raw_parts(name, len),
                    //         }
                    //     }
                    // }
                    // ```
                    #[inline]
                    unsafe fn create_slices<'v>(ptrs: [NonNull<u8>; FIELD_COUNT], len: usize) -> $($struct_slices_ty)* {
                        let [ $($field_name),+ ] = ptrs;
                        unsafe {
                            [<$struct_name Slices>] {
                                $(
                                    $field_plural_name: __p::index_slice_from_raw_parts($field_name, len),
                                )+
                            }
                        }
                    }

                    // ```
                    // unsafe fn create_slices_mut<'v>(ptrs: [NonNull<u8>; FIELD_COUNT], len: usize) -> ThingSlicesMut<'v, 'k, 'a> {
                    //     let [id, name] = ptrs;
                    //     unsafe {
                    //         ThingSlicesMut {
                    //             ids: index_slice_from_raw_parts_mut(id, len),
                    //             names: index_slice_from_raw_parts_mut(name, len),
                    //         }
                    //     }
                    // }
                    // ```
                    #[inline]
                    unsafe fn create_slices_mut<'v>(ptrs: [NonNull<u8>; FIELD_COUNT], len: usize) -> $($struct_slices_mut_ty)* {
                        let [ $($field_name),+ ] = ptrs;
                        unsafe {
                            [<$struct_name SlicesMut>] {
                                $(
                                    $field_plural_name: __p::index_slice_from_raw_parts_mut($field_name, len),
                                )+
                            }
                        }
                    }
                }

                // Implement `CloneFields` on the struct type (e.g. `Scope`).
                // `@generate_clone` expands to nothing when `clone` is empty.
                $crate::multi_vec! {
                    @generate_clone
                    clone = $clone
                    struct_name = [ $struct_name ]
                    struct_generics = { $($struct_generics)* }
                    fields = [ $(
                        {
                            field_name = [ $field_name ]
                            field_ty = [ $field_ty ]
                        }
                    )+ ]
                }

                // Iterator of refs (e.g. `ScopeTableIter`)

                // `impl<'v, 'a> Iterator for ThingsIter<'v, 'a> {`
                impl$($struct_ref_generics)* Iterator for $($table_iter_ty)* {
                    // `type Item = ThingRef<'v, 'a>`
                    type Item = $($struct_ref_ty)*;

                    #[inline(always)]
                    // `fn next(&mut self) -> Option<ThingRef<'v, 'a>> {`
                    fn next(&mut self) -> Option<$($struct_ref_ty)*> {
                        self.0.next()
                    }

                    #[inline(always)]
                    fn size_hint(&self) -> (usize, Option<usize>) {
                        self.0.size_hint()
                    }
                }

                // `impl<'v, 'a> ExactSizeIterator for ThingsIter<'v, 'a> {}`
                impl$($struct_ref_generics)* ExactSizeIterator for $($table_iter_ty)* {}

                // `impl<'v, 'a> FusedIterator for ThingsIter<'v, 'a> {}`
                impl$($struct_ref_generics)* FusedIterator for $($table_iter_ty)* {}

                // Iterator of mut refs (e.g. `ScopeTableIterMut`)

                // `impl<'v, 'a> Iterator for ThingsIterMut<'v, 'a> {`
                impl$($struct_ref_generics)* Iterator for $($table_iter_mut_ty)* {
                    // `type Item = ThingMut<'v, 'a>`
                    type Item = $($struct_mut_ty)*;

                    #[inline(always)]
                    // `fn next(&mut self) -> Option<ThingMut<'v, 'a>> {`
                    fn next(&mut self) -> Option<$($struct_mut_ty)*> {
                        self.0.next()
                    }

                    #[inline(always)]
                    fn size_hint(&self) -> (usize, Option<usize>) {
                        self.0.size_hint()
                    }
                }

                // `impl<'v, 'a> ExactSizeIterator for ThingsIterMut<'v, 'a> {}`
                impl$($struct_ref_generics)* ExactSizeIterator for $($table_iter_mut_ty)* {}

                // `impl<'v, 'a> FusedIterator for ThingsIterMut<'v, 'a> {}`
                impl$($struct_ref_generics)* FusedIterator for $($table_iter_mut_ty)* {}

                // Iterator of owned items (e.g. `ScopeTableIntoIter`)

                // `impl<'a> Iterator for ThingsIntoIter<'a> {`
                impl$($struct_generics)* Iterator for $($table_into_iter_ty)* {
                    // `type Item = Thing<'a>`
                    type Item = $struct_ty;

                    #[inline(always)]
                    // `fn next(&mut self) -> Option<Thing<'a>> {`
                    fn next(&mut self) -> Option<$struct_ty> {
                        self.0.next()
                    }

                    #[inline(always)]
                    fn size_hint(&self) -> (usize, Option<usize>) {
                        self.0.size_hint()
                    }
                }

                // `impl<'a> ExactSizeIterator for ThingsIntoIter<'a> {}`
                impl$($struct_generics)* ExactSizeIterator for $($table_into_iter_ty)* {}

                // `impl<'a> FusedIterator for ThingsIntoIter<'a> {}`
                impl$($struct_generics)* FusedIterator for $($table_into_iter_ty)* {}
            };
        }
    };

    // Generate the `Debug` machinery - the table's `Debug` impl, and a struct-is-`Debug`
    // assertion - for a table declared with `#[derive(Debug)]`.
    //
    // Invoked by `@generate`, unconditionally, from inside the generated `const` block,
    // so the block's imports are in scope. The rule below expands to nothing when the
    // `debug` slot is empty. The choice must be made by rule matching, in a separate rule -
    // see the invocation in `@generate` for why a `$( ... )?` group cannot make it.
    (
        @generate_debug
        debug = [ $debug:ident ]
        table_name = [ $table_name:ident ]
        struct_name = [ $struct_name:ident ]
        table_generics = { $($table_generics:tt)* }
        struct_generics = { $($struct_generics:tt)* }
    ) => {
        use std::fmt;

        // `#[derive(Debug)]` on the `table` requires the struct to be `Debug` too
        // (like `Clone` - see "Debug" in the macro docs). Verify it at check time.
        // The impl below will also fail to compile if the struct is not `Debug`,
        // but with a poor error message. This fails first, naming the offending type.
        // (A fn, not a free `const _: fn(..)`: a free const's type cannot name the
        // struct's lifetime params; a fn declares them.)
        //
        // ```
        // fn assert_struct_is_debug<'a>() {
        //     let _: fn(&Thing<'a>, &mut fmt::Formatter<'_>) -> fmt::Result =
        //         <Thing<'a> as fmt::Debug>::fmt;
        // }
        // ```
        fn assert_struct_is_debug$($struct_generics)*() {
            let _: fn(&$struct_name$($struct_generics)*, &mut fmt::Formatter<'_>) -> fmt::Result =
                <$struct_name$($struct_generics)* as fmt::$debug>::fmt;
        }

        // `impl<'k, 'a> fmt::Debug for Things<'k, 'a> {`
        impl$($table_generics)* fmt::Debug for $table_name$($table_generics)* {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_map().entries(self.iter_enumerated()).finish()
            }
        }
    };

    // No `#[derive(Debug)]` on the `table` - no `Debug` machinery.
    (
        @generate_debug
        debug = []
        table_name = $_table_name:tt
        struct_name = $_struct_name:tt
        table_generics = $_table_generics:tt
        struct_generics = $_struct_generics:tt
    ) => {};

    // Generate the clone machinery - the `CloneFields` impl, and a per-field `Clone`
    // assertion - for a table declared with `#[derive(Clone)]`.
    //
    // Tables without the derive get no clone machinery at all.
    // `MultiVec` is `Clone` only where its fields struct implements `CloneFields`,
    // so the `#[derive(Clone)]` passed through to the table type resolves exactly
    // when this impl exists.
    //
    // Invoked by `@generate`, unconditionally, from inside the generated `const` block,
    // so the block's imports and `FIELD_COUNT` are in scope.
    // The rule below expands to nothing when the `clone` slot is empty.
    // The choice must be made by rule matching, in a separate rule -
    // `@generate` could only make it with a `$( ... )?` group gated on the `clone` slot
    // and this impl's per-field code cannot live inside one - all metavariables
    // directly under a repetition must repeat the same number of times, and the
    // field list repeats per field while `$clone` repeats once.
    (
        @generate_clone
        clone = [ Clone ]
        struct_name = [ $struct_name:ident ]
        struct_generics = { $($struct_generics:tt)* }
        fields = [ $(
            {
                field_name = [ $field_name:ident ]
                field_ty = [ $field_ty:ty ]
            }
        )+ ]
    ) => {
        // `CloneColumn` and `CopyColumn` must be in scope for `.clone_column()` calls to resolve
        #[allow(unused_imports, clippy::allow_attributes)]
        use $crate::multi_vec::__private::{CloneColumn as _, CopyColumn as _};

        // `#[derive(Clone)]` on the `table` requires every field type to be `Clone`.
        // The `clone_column` calls below would also fail to resolve for a non-`Clone` field,
        // but with a poor error message. These fail first, naming the offending type.
        // (A fn, not free `const _: fn(..)`s: a free const's type cannot name the struct's
        // lifetime params; a fn declares them.)
        //
        // ```
        // fn assert_fields_are_clone<'a>() {
        //     let _: fn(&u32) -> u32 = <u32 as Clone>::clone;
        //     let _: fn(&&'a str) -> &'a str = <&'a str as Clone>::clone;
        // }
        // ```
        fn assert_fields_are_clone$($struct_generics)*() {
            $(
                let _: fn(&$field_ty) -> $field_ty = <$field_ty as Clone>::clone;
            )+
        }

        // Implement `CloneFields` on the struct type (e.g. `Scope`)
        //
        // `unsafe impl<'a> CloneFields for Thing<'a> {`
        unsafe impl$($struct_generics)* __p::CloneFields for $struct_name$($struct_generics)* {
            #[inline]
            // ```
            // unsafe fn clone_columns(src_and_dst_ptrs: [SrcAndDstPtrs; FIELD_COUNT], len: usize) {
            //     let [id, name] = src_and_dst_ptrs;
            //     let drop_guards = unsafe {
            //         (
            //             (&&ColumnCloner::<u32>::NEW).clone_column(id, len),
            //             (&&ColumnCloner::<&'a str>::NEW).clone_column(name, len),
            //         )
            //     };
            //     std::mem::forget(drop_guards);
            // }
            // ```
            unsafe fn clone_columns(src_and_dst_ptrs: [__p::SrcAndDstPtrs; FIELD_COUNT], len: usize) {
                let [ $($field_name),+ ] = src_and_dst_ptrs;

                // Each `Clone` (non-`Copy`) column's `clone_column` returns a `ColumnDropGuard`
                // (`Copy` columns return `()`). The guards are held in a tuple until all columns
                // are cloned, then forgotten - so if a column's `clone` panics, unwinding drops
                // all values cloned so far, in this column (by `clone_column`'s internal guard)
                // and in earlier columns (by the already-built guards of the partially-evaluated tuple).
                //
                // TODO: It'd be ideal to clone fields in memory order not declaration order,
                // for the same reason that `MultiVec::grow` does - though that'd be a tiny optimization.
                let drop_guards = unsafe {
                    (
                        $(
                            (&&__p::ColumnCloner::<$field_ty>::NEW).clone_column($field_name, len),
                        )+
                    )
                };
                std::mem::forget(drop_guards);
            }
        }
    };

    // No `#[derive(Clone)]` on the `table` - no clone machinery.
    (
        @generate_clone
        clone = []
        struct_name = $_struct_name:tt
        struct_generics = $_struct_generics:tt
        fields = $_fields:tt
    ) => {};
}
pub use multi_vec;
