//! Whole-program symbol liveness for unused-declaration removal (#13105).
//!
//! `Scoping::symbol_is_unused` is reference-count based, so a declaration
//! that only references itself (`function f() { f() }`) or participates in a
//! reference cycle (`function c() { d() } function d() { c() }`) never reaches
//! zero references and survives even though no live code can reach it.
//!
//! This module computes reachability instead: roots are references that occur
//! in live-executing context (including `export { f }` specifiers, which carry
//! real references); references inside a *candidate* declaration's deferred
//! region (function bodies, side-effect-free initializers, class bodies) only
//! mark their targets live once the candidate itself is marked live. Dead
//! cycles are simply never reached — no cycle detection is needed.
//!
//! ## Where the analysis runs
//!
//! No standalone walk: traversals that already happen collect the analysis
//! inputs as they go (the `collect_*` hooks below write into
//! [`LivenessCollect`]). `Normalize`'s pre-loop traversal produces the
//! initial set — unfiltered, since no previous candidate set exists yet to
//! filter edges against — so source-level dead cycles are visible to pass 1.
//! Each peephole pass then collects with the edge filter seeded by the
//! previous pass, and [`propagate_collected`] turns the collection into a
//! fresh dead set at every flush — including after quiet passes, whose
//! collection observes the previous pass's late mutations. The set consumed
//! by a pass is therefore always exactly one pass stale, the same freshness
//! the standalone recompute cadence had. That freshness is load-bearing:
//! deferring consumption to the loop's fixed point was measured to LOSE
//! output, because other passes rewrite late-exposed dead cycles into
//! non-candidate shapes before a deferred recompute ever sees them.
//!
//! Collection during a mutating pass can diverge from a settled-tree walk in
//! exactly one dangerous direction: a reference MINTED behind the traversal
//! cursor is never visited, so its target could be marked dead while a live
//! reference to it exists. Every bound-reference mint funnels through
//! `TraverseScoping::create_bound_reference`, which logs the symbol; the log
//! is force-rooted at flush. All other divergences are toward-live and
//! self-correct one pass later (a debug validator in [`propagate_collected`]
//! checks the dead set against a ground-truth walk across the whole test
//! corpus).
//!
//! ## The gate-mirror invariant (correctness, not style)
//!
//! Candidacy here must be a SUBSET of what the removal sites in
//! `remove_unused_declaration.rs` will actually remove — cleanly, with no
//! residue. Every member of a dead cycle is removed in the same pass; if one
//! member's removal were blocked (or left residue carrying references), the
//! survivors would reference bindings that no longer exist. Concretely:
//!
//! - global gates: shared structurally — `can_remove_unused_declarators`
//!   (the removal sites' guard) delegates to [`analysis_enabled`], the one
//!   definition of `unused != Keep` plus the root `ScopeFlags::DirectEval`
//!   skip (the flag propagates to the root from any direct eval in the
//!   program, subsuming the per-site current-scope checks).
//! - script-mode: statements whose removal site runs at the root scope are
//!   position-ineligible — the hooks consult the removal sites' own
//!   `keep_top_level_var_in_script_mode` (for-init declarators via its
//!   scope-parameterized core, `statement_scope_keeps_top_level_var`, since
//!   their removal site runs at the scope containing the `for` statement,
//!   one above the declarator's own visitation scope).
//! - declarator inits must be dropped whole by `remove_unused_expression`:
//!   kinds with specialized handlers there can leave residue, so candidacy
//!   excludes them via the shared
//!   [`PeepholeOptimizations::expr_has_specialized_unused_handler`].
//! - class candidacy requires
//!   [`PeepholeOptimizations::classify_class_removability`] (shared with
//!   `remove_unused_class` itself) to return `RemovesClean` — removal must
//!   not bail AND must extract nothing into live code.
//!
//! ## Per-symbol candidacy: position failures force-root
//!
//! Candidacy is granted per declaration SITE, but the dead bit is consumed
//! per SYMBOL — and `var`s redeclare. A site that fails a POSITION gate is
//! not merely a non-candidate: it force-roots the symbols it binds (marks
//! them live outright), because otherwise a removable sibling site of the
//! same symbol could strip the initializer while the ineligible site keeps
//! the binding observable. Position gates:
//!
//! - export-wrapped declarations (`export var f;` carries no reference, yet
//!   importers observe the binding);
//! - script-mode bindings that live in the ROOT scope wherever their
//!   statement sits (a block `var` hoists its BINDING to the root, where
//!   other scripts observe it), plus the visitation-root mirror above;
//! - for-in/of head declarators (no removal site handles them at all);
//! - Annex B block-level function declarations in sloppy code whose binding
//!   was NOT hoisted into the var scope (the binder hoists only the first
//!   same-named function per var scope, so a later duplicate's runtime
//!   var-alias re-assignment is invisible to reference resolution);
//! - `using` declarators (removal always bails on them).
//!
//! SHAPE failures (residue-leaving init kinds, destructuring patterns,
//! `Extracts`/`Keep` classes) do NOT force-root: such a site either keeps
//! its declaration (still binding the symbol) or leaves residue whose
//! references were never deferred — they attribute to the enclosing region
//! and root the peers on their own.
//!
//! Side-effect judgments here use a context with strictly less information
//! than the peephole `TraverseCtx` (no tracked constant values). Extra
//! information only ever proves MORE expressions pure, so "pure here" implies
//! "pure at the removal site" — the direction the invariant needs. The
//! in-pass collection uses the same lean context so its candidacy agrees
//! with the ground-truth walk.
//!
//! ## Allocation discipline
//!
//! Everything sized by the program lives in the arena, like
//! `PassDirty::dead_refs`. References are filtered at record time on
//! candidate-kind `SymbolFlags` (most references cost one flag check and no
//! storage), roots are deduplicated at record time by marking them live
//! immediately, and the edge "graph" is a flat list sorted by source symbol,
//! range-scanned via `partition_point` — no per-candidate allocations. The
//! in-pass collection buffers are `clear()`-reused across passes (arena vecs
//! leak only on capacity growth, which high-watermarks after early passes),
//! keeping system allocations untouched.

#[cfg(debug_assertions)]
use std::cell::Cell;

use oxc_allocator::{Allocator, BitSet, GetAllocator, Vec as ArenaVec};
use oxc_ast::ast::*;
#[cfg(debug_assertions)]
use oxc_ast_visit::{
    Visit,
    walk::{
        walk_class, walk_export_default_declaration, walk_export_named_declaration,
        walk_for_statement_init, walk_for_statement_left, walk_function, walk_variable_declarator,
    },
};
use oxc_ecmascript::{
    BoundNames, GlobalContext,
    side_effects::{
        MayHaveSideEffects, MayHaveSideEffectsContext, PropertyReadSideEffects, is_pure_function,
    },
};
use oxc_semantic::{IsGlobalReference, Scoping};
#[cfg(debug_assertions)]
use oxc_syntax::scope::{ScopeFlags, ScopeId};
use oxc_syntax::symbol::{SymbolFlags, SymbolId};

use crate::{
    CompressOptions, CompressOptionsUnused, TraverseCtx,
    generated::ancestor::Ancestor,
    peephole::{ClassRemovability, PeepholeOptimizations},
};

/// Symbol kinds a liveness candidate can have: function/class declarations
/// and `var`/`let`/`const` bindings (`SymbolFlags::Variable`). A necessary
/// (not sufficient) condition for candidacy.
const CANDIDATE_KINDS: SymbolFlags =
    SymbolFlags::Variable.union(SymbolFlags::Class).union(SymbolFlags::Function);

/// Side-effect context for use outside the peephole traversal. Mirrors the
/// `TraverseCtx` impls in `traverse_context/ecma_context.rs`, minus the
/// tracked-constant lookups (the trait defaults are strictly more
/// conservative).
struct LivenessCtx<'b> {
    scoping: &'b Scoping,
    options: &'b CompressOptions,
}

impl<'a> GlobalContext<'a> for LivenessCtx<'_> {
    fn is_global_reference(&self, ident: &IdentifierReference<'a>) -> bool {
        ident.is_global_reference(self.scoping)
    }
}

impl MayHaveSideEffectsContext<'_> for LivenessCtx<'_> {
    fn annotations(&self) -> bool {
        self.options.treeshake.annotations
    }

    fn manual_pure_functions(&self, callee: &Expression) -> bool {
        is_pure_function(callee, &self.options.treeshake.manual_pure_functions)
    }

    fn property_read_side_effects(&self) -> PropertyReadSideEffects {
        self.options.treeshake.property_read_side_effects
    }

    fn property_write_side_effects(&self) -> bool {
        self.options.treeshake.property_write_side_effects
    }

    fn unknown_global_side_effects(&self) -> bool {
        self.options.treeshake.unknown_global_side_effects
    }
}

/// In script mode a binding that lives in the ROOT scope is observable
/// by other scripts (global object property / global lexical binding)
/// no matter where its declaration statement sits: a block `var` hoists
/// its BINDING to the root while its statement stays at removable block
/// depth.
fn symbol_is_script_global(scoping: &Scoping, is_script: bool, symbol_id: SymbolId) -> bool {
    is_script && scoping.symbol_scope_id(symbol_id) == scoping.root_scope_id()
}

/// Annex B.3.3: in sloppy code, a plain block-level function declaration
/// also creates a runtime var-alias binding in the enclosing var scope.
/// The binder models that alias by MOVING the first same-named
/// function's binding into the var scope (accurate — references resolve
/// to it); a plain function whose binding still sits in a sloppy non-var
/// scope is a later duplicate (or a blocked hoist), whose runtime
/// re-assignment of the alias is invisible to reference resolution — no
/// such declaration is a safe candidate. Applies to sloppy non-script
/// code too (`.cjs`), so this checks strictness, not `is_script`.
fn is_unmodeled_annex_b_alias(scoping: &Scoping, func: &Function<'_>, symbol_id: SymbolId) -> bool {
    // Annex B.3.3 covers plain functions only; async/generator block
    // functions are purely lexical and accurately modeled.
    if func.r#async || func.generator {
        return false;
    }
    let flags = scoping.scope_flags(scoping.symbol_scope_id(symbol_id));
    !flags.is_var() && !flags.is_strict_mode()
}

/// Init shapes `remove_unused_expression` fully drops when pure. Kinds
/// with specialized handlers can leave residue (a surviving expression
/// whose references would dangle once the cycle is removed), so they are
/// not candidates.
fn init_fully_removable(init: Option<&Expression<'_>>, ctx: &LivenessCtx<'_>) -> bool {
    match init {
        None | Some(Expression::FunctionExpression(_) | Expression::ArrowFunctionExpression(_)) => {
            true
        }
        Some(e) => {
            !PeepholeOptimizations::expr_has_specialized_unused_handler(e)
                && !e.may_have_side_effects(ctx)
        }
    }
}

/// Whether the analysis is enabled for this program state. The single
/// definition of the global gates: `can_remove_unused_declarators` (the
/// removal sites' guard) delegates here, so candidacy and removal cannot
/// drift. Any direct eval flags the root via ancestor propagation — see
/// `refresh_direct_eval_flags`.
pub fn analysis_enabled(scoping: &Scoping, options: &CompressOptions) -> bool {
    options.unused != CompressOptionsUnused::Keep
        && !scoping.root_scope_flags().contains_direct_eval()
}

/// Compute the dead set with a standalone walk of a settled tree. Returns
/// `(dead, candidates)`; bits are `SymbolId::index()`.
///
/// Debug-only ground truth: the release pipeline never walks — collection
/// rides the Normalize and peephole traversals — but every
/// [`propagate_collected`] validates its dead set against this walk of the
/// post-flush tree, so the whole test corpus checks the in-pass collection.
#[cfg(debug_assertions)]
pub fn compute_dead_symbols<'a>(
    program: &Program<'a>,
    scoping: &Scoping,
    options: &CompressOptions,
    allocator: &'a Allocator,
) -> (BitSet<'a>, BitSet<'a>) {
    let empty = || (BitSet::new_in(0, allocator), BitSet::new_in(0, allocator));
    if !analysis_enabled(scoping, options) {
        return empty();
    }

    let symbols_len = scoping.symbols_len();
    let mut collector = Collector {
        ctx: LivenessCtx { scoping, options },
        is_script: program.source_type.is_script(),
        scope_depth: 0,
        for_init_depth: None,
        for_head_depth: None,
        in_export: false,
        enclosing_candidate: None,
        candidates: BitSet::new_in(symbols_len, allocator),
        live: BitSet::new_in(symbols_len, allocator),
        roots: ArenaVec::new_in(&allocator),
        edges: ArenaVec::new_in(&allocator),
    };
    collector.visit_program(program);

    let Collector { candidates, mut live, roots, mut edges, .. } = collector;
    if candidates.is_empty() {
        return empty();
    }

    let mut worklist = roots;
    propagate(&candidates, &mut live, &mut worklist, &mut edges);

    let mut dead = BitSet::new_in(symbols_len, allocator);
    for candidate in candidates.ones() {
        if !live.contains(candidate) {
            dead.set_bit(candidate);
        }
    }
    (dead, candidates)
}

/// Worklist propagation shared by the standalone walk and the in-pass
/// collection. The flat edge list sorted by source symbol is the adjacency
/// "map": a live symbol's targets are one `partition_point` range scan away.
/// Roots are already marked live (record-time dedup); marking non-candidate
/// targets live is harmless — only candidates are consulted for deadness.
fn propagate(
    candidates: &BitSet<'_>,
    live: &mut BitSet<'_>,
    worklist: &mut ArenaVec<'_, SymbolId>,
    edges: &mut ArenaVec<'_, (SymbolId, SymbolId)>,
) {
    edges.sort_unstable_by_key(|&(from, _)| from.index());
    while let Some(symbol_id) = worklist.pop() {
        // Only candidates have outgoing edges by construction.
        if !candidates.contains(symbol_id.index()) {
            continue;
        }
        let start = edges.partition_point(|&(from, _)| from.index() < symbol_id.index());
        for i in start..edges.len() {
            let (from, target) = edges[i];
            if from != symbol_id {
                break;
            }
            if !live.contains(target.index()) {
                live.set_bit(target.index());
                worklist.push(target);
            }
        }
    }
}

#[cfg(debug_assertions)]
struct Collector<'a, 'b> {
    ctx: LivenessCtx<'b>,
    is_script: bool,
    /// Scope nesting depth; 1 = the program (root) scope.
    scope_depth: usize,
    /// Depth at which a `ForStatementInit` subtree is being visited (the
    /// for's own scope is already entered). Its direct declarators' removal
    /// site runs one scope above; the exact-depth match leaves declarators
    /// nested deeper inside the init (function bodies) unadjusted.
    for_init_depth: Option<usize>,
    /// Depth at which a `ForStatementLeft` (for-in/of head) subtree is being
    /// visited. Head declarators have no removal site at all.
    for_head_depth: Option<usize>,
    /// The visited node is the `declaration` of an export statement (or a
    /// sibling declarator of one). Cleared for the declaration's subtree so
    /// declarations nested inside an exported function/class stay eligible.
    in_export: bool,
    /// Innermost candidate whose deferred region we are inside.
    enclosing_candidate: Option<SymbolId>,
    candidates: BitSet<'a>,
    /// Marked at record time for roots, during propagation for edge targets.
    live: BitSet<'a>,
    /// Deduplicated live-context references to candidate-kind symbols (plus
    /// force-rooted position-ineligible declarations); the propagation
    /// worklist.
    roots: ArenaVec<'a, SymbolId>,
    /// `(innermost enclosing candidate, referenced candidate-kind symbol)`.
    /// May contain duplicates (one entry per reference); propagation dedups
    /// via the `live` bitset.
    edges: ArenaVec<'a, (SymbolId, SymbolId)>,
}

#[cfg(debug_assertions)]
impl Collector<'_, '_> {
    /// `keep_top_level_var_in_script_mode` mirror: script-mode statements
    /// whose removal site runs at the root scope are not removable. `depth`
    /// is the visitation depth of that removal site.
    fn scope_allows_removal(&self, depth: usize) -> bool {
        !(self.is_script && depth <= 1)
    }

    /// Mark a symbol live and enqueue it for propagation; deduplicated at
    /// record time via the `live` bitset. Used for references in live
    /// context and for force-rooted position-ineligible declarations.
    fn mark_live_root(&mut self, symbol_id: SymbolId) {
        if !self.live.contains(symbol_id.index()) {
            self.live.set_bit(symbol_id.index());
            self.roots.push(symbol_id);
        }
    }

    /// Shared visitor skeleton for the three declaration kinds: clear
    /// `in_export` for the subtree, track the innermost enclosing candidate
    /// across the walk, restore both.
    fn walk_declaration(&mut self, candidate: Option<SymbolId>, walk: impl FnOnce(&mut Self)) {
        let saved_export = std::mem::replace(&mut self.in_export, false);
        let saved_candidate = self.enclosing_candidate;
        if let Some(symbol_id) = candidate {
            self.candidates.set_bit(symbol_id.index());
            self.enclosing_candidate = Some(symbol_id);
        }
        walk(self);
        self.enclosing_candidate = saved_candidate;
        self.in_export = saved_export;
    }
}

#[cfg(debug_assertions)]
impl<'a> Visit<'a> for Collector<'_, '_> {
    fn enter_scope(&mut self, _flags: ScopeFlags, _scope_id: &Cell<Option<ScopeId>>) {
        self.scope_depth += 1;
    }

    fn leave_scope(&mut self) {
        self.scope_depth -= 1;
    }

    fn visit_identifier_reference(&mut self, it: &IdentifierReference<'a>) {
        if let Some(reference_id) = it.reference_id.get()
            && let Some(symbol_id) = self.ctx.scoping.get_reference(reference_id).symbol_id()
            && self.ctx.scoping.symbol_flags(symbol_id).intersects(CANDIDATE_KINDS)
        {
            match self.enclosing_candidate {
                None => self.mark_live_root(symbol_id),
                Some(from) => self.edges.push((from, symbol_id)),
            }
        }
    }

    fn visit_export_named_declaration(&mut self, it: &ExportNamedDeclaration<'a>) {
        let saved = std::mem::replace(&mut self.in_export, true);
        walk_export_named_declaration(self, it);
        self.in_export = saved;
    }

    fn visit_export_default_declaration(&mut self, it: &ExportDefaultDeclaration<'a>) {
        let saved = std::mem::replace(&mut self.in_export, true);
        walk_export_default_declaration(self, it);
        self.in_export = saved;
    }

    fn visit_for_statement_init(&mut self, it: &ForStatementInit<'a>) {
        let saved = self.for_init_depth.replace(self.scope_depth);
        walk_for_statement_init(self, it);
        self.for_init_depth = saved;
    }

    fn visit_for_statement_left(&mut self, it: &ForStatementLeft<'a>) {
        let saved = self.for_head_depth.replace(self.scope_depth);
        walk_for_statement_left(self, it);
        self.for_head_depth = saved;
    }

    fn visit_function(&mut self, it: &Function<'a>, flags: ScopeFlags) {
        let mut candidate = None;
        if it.is_declaration()
            && let Some(symbol_id) = it.id.as_ref().and_then(|id| id.symbol_id.get())
        {
            if self.in_export
                || !self.scope_allows_removal(self.scope_depth)
                || symbol_is_script_global(self.ctx.scoping, self.is_script, symbol_id)
                || is_unmodeled_annex_b_alias(self.ctx.scoping, it, symbol_id)
            {
                self.mark_live_root(symbol_id);
            } else {
                candidate = Some(symbol_id);
            }
        }
        self.walk_declaration(candidate, |v| walk_function(v, it, flags));
    }

    fn visit_class(&mut self, it: &Class<'a>) {
        let mut candidate = None;
        if it.is_declaration()
            && let Some(symbol_id) = it.id.as_ref().and_then(|id| id.symbol_id.get())
        {
            if self.in_export
                || !self.scope_allows_removal(self.scope_depth)
                || symbol_is_script_global(self.ctx.scoping, self.is_script, symbol_id)
            {
                self.mark_live_root(symbol_id);
            } else if matches!(
                PeepholeOptimizations::classify_class_removability(it, &self.ctx, self.ctx.scoping),
                ClassRemovability::RemovesClean
            ) {
                // Shape failures (`Keep` / `Extracts`) do NOT force-root:
                // the class survives whole, or its extraction's references
                // were never deferred (non-candidates attribute outward).
                candidate = Some(symbol_id);
            }
        }
        self.walk_declaration(candidate, |v| walk_class(v, it));
    }

    fn visit_variable_declarator(&mut self, it: &VariableDeclarator<'a>) {
        // The for-init removal site (`handle_for_statement`'s retain) runs
        // while the traversal sits at the scope CONTAINING the for statement
        // — one level above this visit, since the walk has already entered
        // the for's own scope. Mirror its script-root gate at that depth.
        let statement_depth = if self.for_init_depth == Some(self.scope_depth) {
            self.scope_depth - 1
        } else {
            self.scope_depth
        };
        let site_removable = !self.in_export
            && self.for_head_depth != Some(self.scope_depth)
            && !it.kind.is_using()
            && self.scope_allows_removal(statement_depth);
        let mut candidate = None;
        if site_removable {
            if let BindingPattern::BindingIdentifier(id) = &it.id
                && let Some(symbol_id) = id.symbol_id.get()
            {
                if symbol_is_script_global(self.ctx.scoping, self.is_script, symbol_id) {
                    self.mark_live_root(symbol_id);
                } else if init_fully_removable(it.init.as_ref(), &self.ctx) {
                    candidate = Some(symbol_id);
                }
            }
        } else {
            // Position-ineligible site: force-root every symbol it binds
            // (see the module doc). Shape failures land in the `if` arm
            // above and deliberately do not force-root.
            it.id.bound_names(&mut |ident| {
                if let Some(symbol_id) = ident.symbol_id.get() {
                    self.mark_live_root(symbol_id);
                }
            });
        }
        self.walk_declaration(candidate, |v| walk_variable_declarator(v, it));
    }
}

/// Per-pass in-traversal liveness collection: the peephole `Traverse` hooks
/// record candidates, roots, and edges as the pass visits each node, so no
/// standalone walk is needed to refresh the dead set. All buffers are
/// `clear()`-reused across passes.
pub struct LivenessCollect<'a> {
    /// Collection is running this pass. Set at `enter_program` from
    /// [`analysis_enabled`]; stable for the whole pass, so every hook's
    /// enter/exit pair is balanced.
    active: bool,
    /// Whether `prev_candidates` filters edge recording. `false` for the
    /// Normalize seeding pass: no previous candidate set exists yet, and
    /// filtering against an EMPTY set would route every deferred reference
    /// to a root — everything live, no dead cycle visible to pass 1, a
    /// regression of the core `function f() { f() }` case. The seeding pass
    /// records all candidate-kind edges (paying the junk-edge cost once);
    /// its propagation then seeds the filter for every loop pass.
    filtered: bool,
    /// Candidates admitted this pass (per-site gates at enter time, on the
    /// pre-fold shape — folds only purify, so this under-approves relative
    /// to the settled tree: the sound direction, self-correcting next pass).
    candidates: BitSet<'a>,
    /// Marked at record time for roots, during propagation for edge targets.
    live: BitSet<'a>,
    /// The PREVIOUS pass's candidate set: record-time edge filter. A
    /// reference whose target was not a candidate last pass is marked live
    /// outright instead of recorded as an edge — junk targets (parameters,
    /// gate-failing locals) never earn storage, and a genuinely NEW
    /// candidate merely stays alive one extra pass.
    prev_candidates: BitSet<'a>,
    /// Deduplicated roots; doubles as the propagation worklist.
    roots: ArenaVec<'a, SymbolId>,
    /// `(innermost enclosing candidate, referenced candidate-kind symbol)`.
    edges: ArenaVec<'a, (SymbolId, SymbolId)>,
    /// Saved `current_candidate` values; one frame per declaration node
    /// (function/class/declarator), pushed at enter, popped at exit.
    frames: ArenaVec<'a, Option<SymbolId>>,
    /// Innermost candidate whose deferred region the traversal is inside.
    current_candidate: Option<SymbolId>,
    /// Scratch for the next dead set; swapped with
    /// `MinifierState::dead_symbols` at flush.
    dead_next: BitSet<'a>,
}

impl<'a> LivenessCollect<'a> {
    /// Everything starts zero-capacity (like `MinifierState::dead_symbols`):
    /// the first ACTIVE `reset_for_pass` sizes the bitsets, so a compile with
    /// the analysis off (`unused: Keep`, root direct eval) never allocates.
    pub fn new(allocator: &'a Allocator) -> Self {
        Self {
            active: false,
            filtered: false,
            candidates: BitSet::new_in(0, allocator),
            live: BitSet::new_in(0, allocator),
            prev_candidates: BitSet::new_in(0, allocator),
            roots: ArenaVec::new_in(&allocator),
            edges: ArenaVec::new_in(&allocator),
            frames: ArenaVec::new_in(&allocator),
            current_candidate: None,
            dead_next: BitSet::new_in(0, allocator),
        }
    }

    fn reset_for_pass(
        &mut self,
        active: bool,
        filtered: bool,
        symbols_len: usize,
        allocator: &'a Allocator,
    ) {
        self.active = active;
        self.filtered = filtered;
        self.current_candidate = None;
        self.frames.clear();
        if !active {
            return;
        }
        self.roots.clear();
        self.edges.clear();
        // Symbols minted mid-pass are past these capacities and read as
        // live everywhere (the `PassDirty::dead_refs` convention); their
        // declarations enter the analysis next pass.
        for bits in [&mut self.candidates, &mut self.live, &mut self.dead_next] {
            if bits.capacity() == symbols_len {
                bits.clear();
            } else {
                *bits = BitSet::new_in(symbols_len, allocator);
            }
        }
        // `prev_candidates` deliberately survives: it is the previous pass's
        // output (or the Normalize seeding pass's), consumed as this pass's
        // filter.
    }

    /// Mark a symbol live and enqueue it for propagation; deduplicated at
    /// record time.
    fn mark_live_root(&mut self, symbol_id: SymbolId) {
        let index = symbol_id.index();
        if index < self.live.capacity() && !self.live.has_bit(index) {
            self.live.set_bit(index);
            self.roots.push(symbol_id);
        }
    }

    /// Admit a candidacy-eligible declaration; refuses mid-pass-minted
    /// symbols (past capacity — they read as live and retry next pass).
    fn admit_candidate(&mut self, symbol_id: SymbolId) -> bool {
        let index = symbol_id.index();
        if index < self.candidates.capacity() {
            self.candidates.set_bit(index);
            true
        } else {
            false
        }
    }

    fn push_region(&mut self, candidate: Option<SymbolId>) {
        self.frames.push(self.current_candidate);
        if candidate.is_some() {
            self.current_candidate = candidate;
        }
    }

    fn pop_region(&mut self) {
        debug_assert!(!self.frames.is_empty(), "unbalanced liveness region frames");
        self.current_candidate = self.frames.pop().unwrap_or(None);
    }

    /// Shared tail of the `collect_enter_*` declaration hooks: apply the
    /// position-failure force-root, admit the candidate (refusing mid-pass
    /// mints), and open the region frame. Which of the two a declaration
    /// gets — force-root for POSITION failures, plain skip for SHAPE
    /// failures — is decided by the callers (see the module doc).
    fn enter_region(&mut self, force_root: Option<SymbolId>, candidate: Option<SymbolId>) {
        if let Some(symbol_id) = force_root {
            self.mark_live_root(symbol_id);
        }
        let candidate = candidate.filter(|&symbol_id| self.admit_candidate(symbol_id));
        self.push_region(candidate);
    }
}

/// `Traverse::enter_program` body for peephole passes: reset the collection,
/// edge filter seeded by the previous pass's candidates.
pub fn begin_pass(ctx: &mut TraverseCtx<'_>) {
    begin(ctx, /* filtered */ true);
}

/// `Traverse::enter_program` body for the Normalize seeding pass: collect
/// UNFILTERED — see [`LivenessCollect::filtered`]. Its propagation produces
/// the initial dead set for pass 1 and seeds every later pass's edge filter.
pub fn begin_seeding_pass(ctx: &mut TraverseCtx<'_>) {
    begin(ctx, /* filtered */ false);
}

fn begin(ctx: &mut TraverseCtx<'_>, filtered: bool) {
    let allocator = ctx.allocator();
    let TraverseCtx { state, scoping, .. } = ctx;
    let enabled = analysis_enabled(scoping.scoping(), &state.options);
    let symbols_len = scoping.scoping().symbols_len();
    state.liveness.reset_for_pass(enabled, filtered, symbols_len, allocator);
    // A pass observes mints made DURING it; anything already in the log
    // predates this pass's traversal, which will visit those references as
    // part of the tree. (`propagate_collected` drains the log at every
    // flush, so this is a defensive reset — nothing mints between a flush
    // and the next pass start.)
    scoping.clear_minted_symbols();
}

/// `Traverse::enter_identifier_reference` body. Runs for every
/// `IdentifierReference` (including assignment targets), so the inactive
/// path is one load and branch.
pub fn collect_identifier_reference<'a>(
    ident: &IdentifierReference<'a>,
    ctx: &mut TraverseCtx<'a>,
) {
    let TraverseCtx { state, scoping, .. } = ctx;
    let lv = &mut state.liveness;
    if !lv.active {
        return;
    }
    let scoping = scoping.scoping();
    let Some(reference_id) = ident.reference_id.get() else { return };
    let Some(symbol_id) = scoping.get_reference(reference_id).symbol_id() else { return };
    if lv.filtered
        && let Some(from) = lv.current_candidate
    {
        // Deferred-region references — most of the pass on bundle-shaped
        // input — skip the `symbol_flags` load: the previous pass's
        // candidate set subsumes the kind test (only candidate-kind
        // declarations are ever admitted), so one cache-friendly bitset
        // probe decides edge-vs-root, and rooting a filtered-out
        // non-candidate-kind symbol is a deduplicated no-op (only
        // candidates' live bits are consulted). The dead set is unchanged.
        if lv.prev_candidates.contains(symbol_id.index()) {
            lv.edges.push((from, symbol_id));
        } else {
            lv.mark_live_root(symbol_id);
        }
        return;
    }
    // Live-context references keep the kind test so the roots list stays
    // candidate-kind only (junk roots would grow the arena worklist for
    // symbols whose live bits nothing consults); the seeding pass keeps it
    // because it is what bounds the unfiltered edge list.
    if !scoping.symbol_flags(symbol_id).intersects(CANDIDATE_KINDS) {
        return;
    }
    match lv.current_candidate {
        None => lv.mark_live_root(symbol_id),
        Some(from) => lv.edges.push((from, symbol_id)),
    }
}

fn parent_is_export(parent: Ancestor<'_, '_>) -> bool {
    matches!(
        parent,
        Ancestor::ExportNamedDeclarationDeclaration(_)
            | Ancestor::ExportDefaultDeclarationDeclaration(_)
    )
}

/// Position gates shared by function and class declarations: export-wrapped
/// site, removal site at the script root (the hooks fire before the
/// declaration's own scope is entered, so `keep_top_level_var_in_script_mode`
/// evaluates the identical visitation scope the removal site sees), or a
/// script-global binding. Position failures force-root (see the module doc).
fn declaration_position_fails(ctx: &TraverseCtx<'_>, symbol_id: SymbolId) -> bool {
    parent_is_export(ctx.parent())
        || PeepholeOptimizations::keep_top_level_var_in_script_mode(ctx)
        || symbol_is_script_global(ctx.scoping(), ctx.state.source_type.is_script(), symbol_id)
}

/// `Traverse::enter_function` body: decide candidacy at the same position
/// the removal site evaluates (the hook fires before the function's scope is
/// entered, so `current_scope_id` is the containing scope — identical to the
/// `exit_statement` frame that runs `remove_unused_function_declaration`).
pub fn collect_enter_function<'a>(func: &Function<'a>, ctx: &mut TraverseCtx<'a>) {
    if !ctx.state.liveness.active {
        return;
    }
    let mut candidate = None;
    let mut force_root = None;
    if func.is_declaration()
        && let Some(symbol_id) = func.id.as_ref().and_then(|id| id.symbol_id.get())
    {
        if declaration_position_fails(ctx, symbol_id)
            || is_unmodeled_annex_b_alias(ctx.scoping(), func, symbol_id)
        {
            force_root = Some(symbol_id);
        } else {
            candidate = Some(symbol_id);
        }
    }
    ctx.state.liveness.enter_region(force_root, candidate);
}

/// `Traverse::enter_class` body; see [`collect_enter_function`].
pub fn collect_enter_class<'a>(class: &Class<'a>, ctx: &mut TraverseCtx<'a>) {
    if !ctx.state.liveness.active {
        return;
    }
    let mut candidate = None;
    let mut force_root = None;
    if class.is_declaration()
        && let Some(symbol_id) = class.id.as_ref().and_then(|id| id.symbol_id.get())
    {
        if declaration_position_fails(ctx, symbol_id) {
            force_root = Some(symbol_id);
        } else {
            let scoping = ctx.scoping();
            if matches!(
                PeepholeOptimizations::classify_class_removability(
                    class,
                    &LivenessCtx { scoping, options: &ctx.state.options },
                    scoping
                ),
                ClassRemovability::RemovesClean
            ) {
                // Shape failures (`Keep` / `Extracts`) do NOT force-root:
                // the class survives whole, or its extraction's references
                // were never deferred (non-candidates attribute outward).
                candidate = Some(symbol_id);
            }
        }
    }
    ctx.state.liveness.enter_region(force_root, candidate);
}

/// `Traverse::enter_variable_declarator` body. Position gates come from
/// ancestry, which matches the removal sites exactly: the for-init retain
/// runs at the scope containing the `for` statement (the walk has already
/// entered the for's own scope here), and for-in/of heads have no removal
/// site at all.
pub fn collect_enter_variable_declarator<'a>(
    decl: &VariableDeclarator<'a>,
    ctx: &mut TraverseCtx<'a>,
) {
    if !ctx.state.liveness.active {
        return;
    }
    let grandparent = ctx.ancestor(1);
    let in_export = matches!(grandparent, Ancestor::ExportNamedDeclarationDeclaration(_));
    let in_for_head =
        matches!(grandparent, Ancestor::ForInStatementLeft(_) | Ancestor::ForOfStatementLeft(_));
    let scoping = ctx.scoping();
    let is_script = ctx.state.source_type.is_script();
    let position_fail = in_export
        || in_for_head
        || decl.kind.is_using()
        // Lazy tail: modules (the common bundler input) and already-failed
        // gates never pay the scope lookups.
        || (is_script
            && PeepholeOptimizations::statement_scope_keeps_top_level_var(ctx, {
                // The for-init removal site (`handle_for_statement`'s
                // retain) runs at the scope containing the `for` statement
                // — one above this visit, since the walk has already
                // entered the for's own scope.
                if matches!(grandparent, Ancestor::ForStatementInit(_)) {
                    scoping
                        .scope_parent_id(ctx.current_scope_id())
                        .unwrap_or_else(|| ctx.current_scope_id())
                } else {
                    ctx.current_scope_id()
                }
            }));
    let mut candidate = None;
    let mut force_root = None;
    if !position_fail
        && let BindingPattern::BindingIdentifier(id) = &decl.id
        && let Some(symbol_id) = id.symbol_id.get()
    {
        if symbol_is_script_global(scoping, is_script, symbol_id) {
            force_root = Some(symbol_id);
        } else if init_fully_removable(
            decl.init.as_ref(),
            &LivenessCtx { scoping, options: &ctx.state.options },
        ) {
            candidate = Some(symbol_id);
        }
    }
    let lv = &mut ctx.state.liveness;
    if position_fail {
        // Position-ineligible site: force-root every symbol it binds (see
        // the module doc). Shape failures land in the arm above and
        // deliberately do not force-root.
        decl.id.bound_names(&mut |ident| {
            if let Some(symbol_id) = ident.symbol_id.get() {
                lv.mark_live_root(symbol_id);
            }
        });
    }
    lv.enter_region(force_root, candidate);
}

/// `Traverse::exit_function` / `exit_class` / `exit_variable_declarator`
/// body: close the region opened by the matching enter hook.
pub fn collect_exit_region(ctx: &mut TraverseCtx<'_>) {
    let lv = &mut ctx.state.liveness;
    if lv.active {
        lv.pop_region();
    }
}

/// Consume the pass's collection: force-root symbols that received freshly
/// minted references (the one toward-dead divergence of in-pass collection),
/// propagate liveness, refresh `MinifierState::dead_symbols`, and report
/// whether the driver must run another pass:
///
/// - a NEW dead symbol appeared — its removal must be consumed with the
///   same one-pass freshness as everything else; or
/// - the candidate set GREW — references to a candidate the edge filter had
///   not seen yet were conservatively rooted (kept alive), so one more pass
///   is needed for the filter to defer them. Without this, a candidate that
///   first becomes eligible on the loop's final pass would silently survive
///   (e.g. the whole analysis coming online after the last direct eval is
///   dropped, or an init folding into an eligible shape).
///
/// Both continue reasons are bounded: dead and candidate bits only ever
/// come from the symbol table, and on an unchanged tree a repeat pass
/// collects the identical candidate set, so a quiet re-run converges
/// immediately.
///
/// Must run after `flush_pass_dirty` so the debug ground-truth walk sees
/// post-flush scoping.
pub fn propagate_collected<'a>(
    #[cfg_attr(not(debug_assertions), expect(unused_variables))] program: &Program<'a>,
    ctx: &mut TraverseCtx<'a>,
) -> bool {
    // `allocator` and `options` feed only the debug ground-truth walk below.
    #[cfg_attr(not(debug_assertions), expect(unused_variables))]
    let allocator = ctx.allocator();
    let TraverseCtx { state, scoping, .. } = ctx;
    #[cfg_attr(not(debug_assertions), expect(unused_variables))]
    let crate::state::MinifierState { liveness, dead_symbols, options, .. } = state;
    let lv = liveness;
    if !lv.active {
        // Keep the mint log bounded even when the analysis is off.
        scoping.clear_minted_symbols();
        return false;
    }
    debug_assert!(lv.frames.is_empty(), "unbalanced liveness region frames at flush");

    // References minted behind the traversal cursor were never visited by
    // the collection; their targets must be live.
    for &symbol_id in scoping.minted_symbols() {
        lv.mark_live_root(symbol_id);
    }
    scoping.clear_minted_symbols();

    // No candidates admitted ⇒ no edges (an edge needs an enclosing
    // candidate region), no dead set, no growth: skip the worklist drain
    // and the set scans (mirrors `compute_dead_symbols`' early-out). The
    // swaps still run so a stale dead set from the previous pass clears
    // and the next pass's filter sees the empty candidate set.
    if lv.candidates.is_empty() {
        std::mem::swap(dead_symbols, &mut lv.dead_next);
        std::mem::swap(&mut lv.prev_candidates, &mut lv.candidates);
        return false;
    }

    // The seeding pass recorded edges unfiltered; candidacy is fully known
    // now, so drop the (measured 77-84%) edges whose targets can never be
    // dead before they hit the sort. A removed edge's only effect was a
    // live mark on a non-candidate, which nothing consults. Loop passes
    // filter at record time instead.
    if !lv.filtered {
        let LivenessCollect { edges, candidates, .. } = lv;
        edges.retain(|&(_, target)| candidates.contains(target.index()));
    }

    propagate(&lv.candidates, &mut lv.live, &mut lv.roots, &mut lv.edges);

    for candidate in lv.candidates.ones() {
        if !lv.live.contains(candidate) {
            lv.dead_next.set_bit(candidate);
        }
    }

    // Per-bit scans; word-level set-difference helpers would make these
    // O(words) — a possible `oxc_allocator` follow-up. Until then, gate the
    // common terminal-pass case (empty dead set) and short-circuit the
    // growth scan when a new dead symbol already demands the pass.
    let found_new_dead =
        !lv.dead_next.is_empty() && lv.dead_next.ones().any(|bit| !dead_symbols.contains(bit));
    let needs_liveness_pass =
        found_new_dead || lv.candidates.ones().any(|bit| !lv.prev_candidates.contains(bit));

    // Ground-truth net: everything the in-pass collection calls dead must be
    // dead per a standalone walk of the settled tree — for symbols whose
    // declarations still exist (a declaration removed this pass legitimately
    // lingers in the stale set for one flush, with nothing left to remove).
    // Runs only when the set is non-empty (the subset check is vacuous
    // otherwise), so the whole test and conformance corpus doubles as a
    // collection-vs-truth differ at zero release cost.
    #[cfg(debug_assertions)]
    if !lv.dead_next.is_empty() {
        let (walk_dead, walk_candidates) =
            compute_dead_symbols(program, scoping.scoping(), options, allocator);
        for bit in lv.dead_next.ones() {
            assert!(
                !walk_candidates.contains(bit) || walk_dead.contains(bit),
                "in-pass liveness collection marked symbol {bit} dead but the ground-truth walk \
                 sees it live",
            );
        }
    }

    std::mem::swap(dead_symbols, &mut lv.dead_next);
    std::mem::swap(&mut lv.prev_candidates, &mut lv.candidates);
    needs_liveness_pass
}

/// Generates the shared liveness-collection `Traverse` hook delegations.
/// Both collecting traversals (`Normalize` — the seeding pass — and
/// `PeepholeOptimizations`) invoke this, so the hook-set membership is
/// single-sourced: extending collection to a new node kind is one edit here.
/// `enter_program` (seeding vs loop pass) and `exit_variable_declarator`
/// (the peephole impl prepends to a pre-existing body) stay per-impl. The
/// hooks have no dce gating by design — collection runs in dce mode too.
macro_rules! liveness_collect_hooks {
    () => {
        fn enter_identifier_reference(
            &mut self,
            node: &mut IdentifierReference<'a>,
            ctx: &mut TraverseCtx<'a>,
        ) {
            crate::symbol_liveness::collect_identifier_reference(node, ctx);
        }

        fn enter_function(&mut self, node: &mut Function<'a>, ctx: &mut TraverseCtx<'a>) {
            crate::symbol_liveness::collect_enter_function(node, ctx);
        }

        fn exit_function(&mut self, _node: &mut Function<'a>, ctx: &mut TraverseCtx<'a>) {
            crate::symbol_liveness::collect_exit_region(ctx);
        }

        fn enter_class(&mut self, node: &mut Class<'a>, ctx: &mut TraverseCtx<'a>) {
            crate::symbol_liveness::collect_enter_class(node, ctx);
        }

        fn exit_class(&mut self, _node: &mut Class<'a>, ctx: &mut TraverseCtx<'a>) {
            crate::symbol_liveness::collect_exit_region(ctx);
        }

        fn enter_variable_declarator(
            &mut self,
            node: &mut VariableDeclarator<'a>,
            ctx: &mut TraverseCtx<'a>,
        ) {
            crate::symbol_liveness::collect_enter_variable_declarator(node, ctx);
        }
    };
}
pub(crate) use liveness_collect_hooks;
