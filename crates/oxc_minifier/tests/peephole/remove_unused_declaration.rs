use oxc_span::SourceType;

use crate::{
    CompressOptions, CompressOptionsUnused, TreeShakeOptions, test_options,
    test_options_source_type, test_same_options, test_same_options_source_type, test_same_smallest,
    test_smallest,
};

// Leak regression: dropping an unused declarator must walk the whole
// declarator, not just the init — references can also live in the binding's
// TS type annotation (e.g. computed keys in a type literal). A leaked type
// ref makes the symbol look used, blocking its own removal.
#[test]
fn remove_unused_declarator_walks_type_annotation_refs() {
    let options = CompressOptions::smallest();
    test_options_source_type(
        "function f() { const a = Symbol('a'); const b = Symbol('b'); const reg: { [a]: string; [b]: string } = { foo: 1, bar: 2 }; return 1; } g(f());",
        "function f() { return 1; } g(f());",
        SourceType::ts(),
        &options,
    );
}

// Leak regression (single-use inlining, `stmts.pop()` site): after the lone
// declarator's init is inlined into the next statement, the whole declaration
// statement is popped — the discarded declarator's type annotation still holds
// a ref to `a`.
#[test]
fn single_use_inline_pop_walks_type_annotation_refs() {
    let options = CompressOptions::smallest();
    test_options_source_type(
        "function f() { const a = Symbol('a'); const x: { [a]: string } = g(); return x; } h(f());",
        "function f() { return g(); } h(f());",
        SourceType::ts(),
        &options,
    );
}

// Leak regression (single-use inlining, `declarations.truncate()` site): only
// the tail declarator `x` is inlined; the truncate discards it while `keep`
// survives — `x`'s type annotation still holds a ref to `a`.
#[test]
fn single_use_inline_truncate_walks_type_annotation_refs() {
    let options = CompressOptions::smallest();
    test_options_source_type(
        "function f() { const a = Symbol('a'); const keep = g(), x: { [a]: string } = h(); return [keep, keep, x]; } j(f());",
        "function f() { let keep = g(); return [keep, keep, h()]; } j(f());",
        SourceType::ts(),
        &options,
    );
}

// Leak regression (single-use inlining, `declarations.drain()` site): `x` is
// inlined into the sibling declarator `y`'s init within the same declaration;
// the drain discards `x`'s declarator — its type annotation still holds a ref
// to `a`.
#[test]
fn single_use_inline_drain_walks_type_annotation_refs() {
    let options = CompressOptions::smallest();
    test_options_source_type(
        "function f() { const a = Symbol('a'); const x: { [a]: string } = g(), y = [x]; return y; } j(f());",
        "function f() { return [g()]; } j(f());",
        SourceType::ts(),
        &options,
    );
}

// Leak regression (dead-code identity-drop site): an init-less `var` after
// `return` is classified as an identity drop (KeepVar re-emits it), skipping
// the drop walk — but KeepVar's re-emit strips the type annotation, so the
// annotation's ref to `b` leaks.
#[test]
fn dead_code_identity_drop_checks_type_annotation() {
    let options = CompressOptions::smallest();
    test_options_source_type(
        "function f() { const b = Symbol('b'); return 1; var a: { [b]: string }; } g(f());",
        "function f() { return 1; } g(f());",
        SourceType::ts(),
        &options,
    );
}

// Near-miss: dropping the annotated declarator must only kill the annotation's
// own ref — `a`'s other (value) uses keep `const a = Symbol('a')` alive.
#[test]
fn type_annotation_drop_keeps_symbol_used_elsewhere() {
    let options = CompressOptions::smallest();
    test_options_source_type(
        "function f() { const a = Symbol('a'); const x: { [a]: string } = g(); return [x, a, a]; } h(f());",
        "function f() { let a = Symbol('a'); return [g(), a, a]; } h(f());",
        SourceType::ts(),
        &options,
    );
}

#[test]
fn remove_unused_variable_declaration() {
    let options = CompressOptions::smallest();
    test_options("var x", "", &options);
    test_options("var x = 1", "", &options);
    test_options("var x = foo", "foo", &options);

    test_options("var [] = []", "", &options);
    test_options("var [] = [1]", "", &options);
    test_options("var [] = [foo]", "foo", &options);
    test_options("var [] = 'foo'", "", &options);
    test_same_options("export var f = () => { var [] = arguments }", &options);
    test_options(
        "export function f() { var [] = arguments }",
        "export function f() { arguments; }",
        &options,
    );
    test_options(
        "function foo() {return (()=>{ var []=arguments })()};foo()",
        "function foo() {arguments;} foo();",
        &options,
    );
    test_same_options_source_type(
        "globalThis.f = function () { var [] = arguments }",
        SourceType::cjs(),
        &options,
    );
    test_same_options("var [] = arguments", &options);
    test_same_options("var [] = null", &options);
    test_same_options("var [] = void 0", &options);
    test_same_options("var [] = 1", &options);
    test_same_options("var [] = a", &options);

    test_options("var {} = {}", "", &options);
    test_options("var {} = { a: 1 }", "", &options);
    test_options("var {} = { foo }", "foo", &options);
    test_same_options("var {} = null", &options);
    test_same_options("var {} = a", &options);
    test_same_options("var {} = null", &options);
    test_same_options("var {} = void 0", &options);

    test_same_options("var x; foo(x)", &options);
    test_same_options("export var x", &options);
    test_same_options("using x = foo", &options);
    test_same_options("await using x = foo", &options);

    test_options("for (var x; ; );", "for (; ;);", &options);
    test_options("for (var x = 1; ; );", "for (; ;);", &options);
    test_same_options("for (var x = foo; ; );", &options); // can be improved
}

#[test]
fn remove_unused_pure_iife_init() {
    // https://github.com/oxc-project/oxc/issues/17480
    test_smallest("var x = /* @__PURE__ */ foo()", "");
    test_smallest("var x = /* @__PURE__ */ new Foo()", "");
    test_smallest("var x = /* @__PURE__ */ foo(a)", "a;");
    test_smallest("var x = /* @__PURE__ */ foo(bar())", "bar();");
    test_smallest("var x = /* @__PURE__ */ new Foo(bar())", "bar();");
    test_smallest("var x = /* @__PURE__ */ foo(/* @__PURE__ */ bar(z))", "z;");

    test_smallest("var x = /* @__PURE__ */ (() => foo())()", "");
    test_smallest("var x = /* @__PURE__ */ (() => new Foo())()", "");
    test_smallest("var x = /* @__PURE__ */ (() => { return foo() })()", "");
    test_smallest("var x = /* @__PURE__ */ (() => { foo() })()", "");

    test_smallest("var x = /* @__PURE__ */ (() => g.x)()", "");
    test_smallest("var x = /* @__PURE__ */ (() => g[k])()", "");
    test_smallest("var x = /* @__PURE__ */ (() => foo`tpl`)()", "");
    test_smallest("var x = /* @__PURE__ */ (() => [a, b])()", "");
    test_smallest("var x = /* @__PURE__ */ (() => ({ a }))()", "");
    test_smallest("var x = /* @__PURE__ */ (() => a + b)()", "");
    test_smallest("var x = /* @__PURE__ */ (() => `${a}`)()", "");
    test_smallest("var x = /* @__PURE__ */ (() => foo()?.bar())()", "");
    test_smallest("var x = /* @__PURE__ */ (() => a ? b : c)()", "");

    test_smallest("var x = /* @__PURE__ */ (function() { return foo() })()", "");

    test_smallest("let x = /* @__PURE__ */ (() => g.x)()", "");
    test_smallest("const x = /* @__PURE__ */ (() => g.x)()", "");

    test_smallest("var x = /* @__PURE__ */ foo(), y = bar(); use(y);", "var y = bar(); use(y);");

    // Referenced bindings keep the declarator — `symbol_is_unused` blocks
    // the drop. Propagation still inlines the IIFE body.
    test_same_smallest("var x = /* @__PURE__ */ foo(); use(x);");
    test_smallest(
        "var x = /* @__PURE__ */ (() => foo())(); use(x);",
        "var x = /* @__PURE__ */ foo(); use(x);",
    );
    test_smallest("var x = /* @__PURE__ */ (() => g.x)(); use(x);", "var x = g.x; use(x);");
    test_smallest(
        "var x = /* @__PURE__ */ (() => { return foo() })(); use(x);",
        "var x = /* @__PURE__ */ foo(); use(x);",
    );
    // Conditional body — propagation only fires on Call/New, so the
    // top-level conditional is inlined without an annotation.
    test_smallest(
        "var x = /* @__PURE__ */ (() => a ? b : c)(); use(x);",
        "var x = a ? b : c; use(x);",
    );

    // Exported bindings are cross-module reachable — the export-ancestor
    // check blocks the early drop.
    test_smallest("export var x = /* @__PURE__ */ foo()", "export var x = /* @__PURE__ */ foo();");
    test_smallest(
        "export const x = /* @__PURE__ */ (() => foo())();",
        "export const x = /* @__PURE__ */ foo();",
    );
    test_smallest("export const x = /* @__PURE__ */ (() => g.x)();", "export const x = g.x;");
    test_same_smallest("var x = /* @__PURE__ */ foo(); export { x }");

    test_smallest("var x = (() => g.x)();", "g.x;");

    // `using` runs `[Symbol.dispose]` at scope exit, so the declarator stays.
    test_smallest("using x = /* @__PURE__ */ (() => foo())()", "using x = /* @__PURE__ */ foo();");
    test_smallest(
        "await using x = /* @__PURE__ */ (() => foo())()",
        "await using x = /* @__PURE__ */ foo();",
    );

    // Function-local var inside an exported function — the export ancestor
    // walk must not be fooled by `f` being exported. `x` is a local.
    test_smallest(
        "export function f() { var x = /* @__PURE__ */ (() => foo())(); } f();",
        "export function f() {} f();",
    );

    // Empty async/generator IIFE in unused-var-init position now collapses
    // through `is_expression_result_unused` (which the widening newly covers).
    test_smallest("var x = (async () => {})()", "");
    test_smallest("var x = (function* () {})()", "");

    // `can_remove_unused_declarators` blocks top-level `var` drops in script
    // mode (the binding is an observable global). The IIFE inlines with
    // propagation as in any other position, but the declarator stays.
    test_options_source_type(
        "var x = /* @__PURE__ */ (() => stuff())()",
        "var x = /* @__PURE__ */ stuff();",
        SourceType::cjs().with_script(true),
        &CompressOptions::smallest(),
    );

    // Direct eval at the root scope blocks the drop — eval might reference
    // the binding even when static analysis sees no use.
    test_smallest(
        "eval('x'); var x = /* @__PURE__ */ (() => stuff())()",
        "eval('x'); var x = /* @__PURE__ */ stuff();",
    );
}

#[test]
fn remove_unused_function_declaration() {
    let options = CompressOptions::smallest();
    test_options("function foo() {}", "", &options);
    test_same_options("function foo() { bar } foo()", &options);
    test_same_options("export function foo() {} foo()", &options);
    test_same_options("function foo() { bar } eval('foo()')", &options);
}

#[test]
fn remove_unused_declaration_after_dead_direct_eval() {
    let options = CompressOptions::smallest();
    test_options("function f(){if(false)eval('x');var x}f()", "", &options);
    // Live eval still keeps `var x` alive after the refresh.
    test_same_options("function f(){eval('x');var x}f()", &options);
    // Parenthesized eval is still direct eval; the wrapped form must keep `var x` alive.
    test_options(
        "function f(){if(false)y;(eval)('x');var x}f()",
        "function f(){(eval)('x');var x}f()",
        &options,
    );
    // Eval nested inside another call's arguments still keeps `var x` alive.
    // The dead `if(false)y` triggers a peephole change so the refresh actually runs.
    test_options(
        "function f(){if(false)y;foo(eval('x'));var x}f()",
        "function f(){foo(eval('x'));var x}f()",
        &options,
    );
    // Eval in a nested scope: clearing must propagate up through both nested and
    // outer chains, then re-set only what's still live (here, nothing).
    test_options(
        "function outer(){function inner(){if(false)eval('x')}inner();var x}outer()",
        "",
        &options,
    );
    // Live eval at the program root keeps an otherwise-unused `var x` alive
    // (the root flag is the global witness checked by `can_remove_unused_declarators`).
    test_same_options("eval('x');var x", &options);
}

#[test]
fn remove_unused_declaration_with_optional_eval() {
    let options = CompressOptions::smallest();
    test_options("function f(){if(false)eval?.('x');var x}f()", "", &options);
    // Live optional eval is indirect — it doesn't set `DirectEval`, so `var x` is
    // removable even though the call itself stays as a side-effectful expression.
    // Contrast with the live-direct-eval root case above, where `var x` is kept.
    test_options("eval?.('x');var x", "eval?.('x');", &options);
}

#[test]
fn remove_unused_class_declaration() {
    let options = CompressOptions::smallest();
    test_options("class C {}", "", &options);
    test_same_options("export class C {}", &options);
    test_options("class C {} C", "", &options);
    test_same_options("class C {} eval('C')", &options);

    // extends
    test_options("class C {}", "", &options);
    test_options("class C extends Foo {}", "Foo", &options);

    // static block
    test_options("class C { static {} }", "", &options);
    test_same_options("class C { static { foo } }", &options);

    // method
    test_options("class C { foo() {} }", "", &options);
    test_options("class C { [foo]() {} }", "foo", &options);
    test_options("class C { static foo() {} }", "", &options);
    test_options("class C { static [foo]() {} }", "foo", &options);
    test_options("class C { [1]() {} }", "", &options);
    test_options("class C { static [1]() {} }", "", &options);

    // property
    test_options("class C { foo }", "", &options);
    test_options("class C { foo = bar }", "", &options);
    test_options("class C { foo = 1 }", "", &options);
    // TODO: would be nice if this is removed but the one with `this` is kept.
    test_same_options("class C { static foo = bar }", &options);
    test_same_options("class C { static foo = this.bar = {} }", &options);
    test_options("class C { static foo = 1 }", "", &options);
    test_options("class C { [foo] = bar }", "foo", &options);
    test_options("class C { [foo] = 1 }", "foo", &options);
    test_same_options("class C { static [foo] = bar }", &options);
    test_options("class C { static [foo] = 1 }", "foo", &options);

    // accessor
    test_options("class C { accessor foo = 1 }", "", &options);
    test_options("class C { accessor [foo] = 1 }", "foo", &options);

    // order
    test_options("class _ extends A { [B] = C; [D]() {} }", "A, B, D", &options);

    // decorators
    test_same_options("class C { @dec foo() {} }", &options);
    test_same_options("@dec class C {}", &options);

    // TypeError
    test_same_options("class C extends (() => {}) {}", &options);
}

#[test]
fn keep_in_script_mode() {
    let options = CompressOptions::smallest();
    let source_type = SourceType::cjs().with_script(true);
    test_same_options_source_type("var x = 1; x = 2;", source_type, &options);
    test_same_options_source_type("var x = 1; x = 2, foo(x)", source_type, &options);
    test_options_source_type("var x = 1; x = 2;", "", SourceType::cjs(), &options);

    test_options_source_type("class C {}", "class C {}", source_type, &options);
}

// #13105: a declaration whose every reference lives inside its own body (or
// inside the bodies of a cycle it belongs to) can never execute — no live
// code can reach it, so the whole group is removable. Reference counting
// alone can't see this: the internal references keep the count above zero.
#[test]
fn remove_recursive_unused_function_declaration() {
    // Self-recursion.
    test_smallest("function f() { f() }", "");
    // Side effects inside the dead body never run.
    test_smallest("function f() { console.log(1); f() }", "");
    // Mutual recursion.
    test_smallest("function c() { d() } function d() { c() }", "");
    // Self-reference as a value.
    test_smallest("function f() { return f }", "");
    test_smallest("function f() { g(f) }", "");
    // The cycle's only external reference is inside dead code.
    test_smallest("if (false) c(); function c() { d() } function d() { c() }", "");
}

#[test]
fn remove_recursive_unused_declarator_and_class() {
    // const arrow cycle.
    test_smallest("const a = () => b(); const b = () => a();", "");
    // var closing over its own binding.
    test_smallest("var f = function() { f() };", "");
    // Class cycle with side-effect-free evaluation.
    test_smallest("class A { m() { new B() } } class B { m() { new A() } }", "");
    // Mixed function / const arrow / class cycle.
    test_smallest("function a() { b() } const b = () => { new C() }; class C { m() { a() } }", "");
}

#[test]
fn remove_recursive_unused_nested_in_live_function() {
    // Dead recursion inside a used function: statement-level tree shaking
    // (rolldown's linker) cannot see inside bodies, so this must be handled
    // here.
    test_smallest(
        "function live() { function inner() { inner() } return 1; } g(live());",
        "function live() { return 1; } g(live());",
    );
}

#[test]
fn remove_recursive_unused_multi_declarator() {
    // A dead cycle member sharing a declaration statement with a used
    // declarator (rolldown's statement-granular shaking keeps the whole
    // statement).
    test_smallest(
        "const a = () => b(), keep = 1; function b() { a() } console.log(keep);",
        "console.log(1);",
    );
}

#[test]
fn remove_recursive_unused_in_script_function_scope() {
    // Script mode only protects top-level bindings; recursion dead inside a
    // kept function is still removable.
    let options = CompressOptions::smallest();
    let source_type = SourceType::cjs().with_script(true);
    test_options_source_type(
        "function o() { function f() { f() } return 1 } g(o());",
        "function o() { return 1 } g(o());",
        source_type,
        &options,
    );
}

#[test]
fn keep_recursive_function_with_live_references() {
    // A read from live code roots the cycle.
    test_same_smallest("function f() { f() } console.log(f);");
    // A write from live code also roots it (dropping the function would leave
    // `f = null` assigning to a missing binding).
    test_same_smallest("function f() { f() } f = null;");
    // Exports are roots.
    test_same_smallest("export function f() { f() }");
    test_same_smallest("function f() { f() } export { f };");
    test_same_smallest("export default function f() { f() }");
    // Direct eval in the declaring scope blocks removal.
    test_same_smallest("function o() { function f() { f() } eval('x') } o();");
}

#[test]
fn keep_recursive_cycle_with_side_effectful_evaluation() {
    // The side-effectful initializer survives, and its reference to `b`
    // roots the cycle.
    test_smallest(
        "const a = (console.log(1), () => b()); const b = () => a();",
        "const a = (console.log(1), () => b()), b = () => a();",
    );
    // Side-effectful heritage keeps the class cycle.
    test_same_smallest(
        "class A extends (console.log(1), Object) { m() { new B() } } class B { m() { new A() } }",
    );
    // A PURE static value still keeps the class cycle: `remove_unused_class`
    // extracts every present static value, so removal would not be clean —
    // the extracted `B` would reference a removed cycle member (see
    // `classify_class_removability`).
    test_same_smallest("class A { static x = B; m() { new B() } } class B { m() { new A() } }");
}

#[test]
fn keep_class_cycle_with_wrapped_arrow_heritage() {
    // `classify_class_removability` must see the arrow heritage through a
    // pure sequence/paren wrapper: liveness classifies the pre-fold shape,
    // but a fold surfaces the literal arrow before the removal site
    // re-classifies, and the Keep flip would strand `A` referencing a
    // removed `B`.
    test_smallest(
        "class A extends (0, () => {}) { m() { new B() } } class B { m() { new A() } } console.log(1);",
        "class A extends (() => {}) {\n\tm() {\n\t\tnew B();\n\t}\n}\nclass B {\n\tm() {\n\t\tnew A();\n\t}\n}\nconsole.log(1);",
    );
    // The single, fully-unused class is kept for the same reason a literal
    // arrow heritage is kept: evaluating it is a guaranteed TypeError.
    test_smallest("class C extends (0, () => {}) {}", "class C extends (() => {}) {}");
}

#[test]
fn keep_class_with_tdz_or_undefined_heritage() {
    // test262 language/statements/class/name-binding/in-extends-expression.js:
    // the class's own name is in its TDZ while the heritage evaluates, so the
    // declaration is a guaranteed ReferenceError that must survive.
    test_same_smallest("class C extends C {}");
    // The test262 shape: the class lives in a callback whose call is live.
    // (A NAMED function wrapper additionally hits a pre-existing hole in the
    // pure-function model — `may_have_side_effects` does not model heritage
    // TDZ throws, so `f` reads as pure and the call is dropped on `main`
    // too; that is a separate `oxc_ecmascript` issue, not covered here.)
    test_same_smallest("g(function() {\n\tclass C extends C {}\n});");
    // The wrapped variant classifies through the same heritage unwrap.
    test_smallest("class C extends (0, C) {}", "class C extends C {}");
    // A forward lexical heritage also evaluates in its TDZ; reference order
    // cannot be proven mid-minification (transforms copy and move spans), so
    // any class/lexical/`var` heritage keeps the class.
    test_same_smallest(
        "class A extends B {\n\tm() {\n\t\tnew A();\n\t}\n}\nclass B {\n\tm() {\n\t\tnew A();\n\t}\n}",
    );
    // `var` heritage: a hoisted-but-unassigned binding is `undefined`, and
    // `extends undefined` is a TypeError.
    test_same_smallest("var B = class {};\nclass A extends B {\n\tm() {\n\t\tnew A();\n\t}\n}");
}

#[test]
fn remove_class_with_hoisted_function_heritage() {
    // Hoisted function declarations are initialized before any statement
    // runs, so a dead cycle whose heritage is a plain function declaration
    // is still removable.
    test_smallest("function F() {}\nclass A extends F {\n\tm() {\n\t\tnew A();\n\t}\n}", "");
}

#[test]
fn keep_recursive_function_in_script_mode_top_level() {
    let options = CompressOptions::smallest();
    let source_type = SourceType::cjs().with_script(true);
    test_same_options_source_type("function f() { f() }", source_type, &options);
}

#[test]
fn keep_recursive_function_with_unused_keep_option() {
    let options =
        CompressOptions { unused: CompressOptionsUnused::Keep, ..CompressOptions::smallest() };
    test_same_options("function f() { f() }", &options);
}

// Candidacy is granted per declaration SITE but the dead bit is consumed per
// SYMBOL, so a declaration site the removal machinery can never remove —
// export-wrapped, script top-level (including bindings var-hoisted to the
// script root from inside blocks), for-in/of heads, Annex-B block-level
// functions — must force-root its symbol: one ineligible site keeps the
// whole symbol alive even when another site of the same symbol is removable.
#[test]
fn keep_recursive_cycle_with_exported_redeclaration() {
    // `export var f;` carries no reference, but importers observe the
    // binding: removing the initializing redeclaration would export
    // undefined.
    test_same_smallest("export var f; var f = function() { setTimeout(f) };");
}

#[test]
fn keep_recursive_cycle_in_script_mode_positions() {
    let options = CompressOptions::smallest();
    let source_type = SourceType::cjs().with_script(true);
    // A top-level for-init declarator's removal site runs at the scope
    // CONTAINING the for statement — the protected script root — so it can
    // never be removed there, and its cycle peers must stay alive too. (The
    // var-only blocks flatten into the parent; the bindings must survive.)
    test_options_source_type(
        "for (var f = g;;) break; { var g = function() { f() }; } console.log('done');",
        "for (var f = g;;) break; var g = function() { f() }; console.log('done');",
        source_type,
        &options,
    );
    test_options_source_type(
        "{ var p = function() { f() }; } for (let f = p;;) break;",
        "var p = function() { f() }; for (let f = p;;) break;",
        source_type,
        &options,
    );
    // A var inside a block still binds at the script root (var hoisting), so
    // cross-script access can observe it exactly like a top-level statement.
    test_options_source_type(
        "{ var g = function() { g() }; }",
        "var g = function() { g() };",
        source_type,
        &options,
    );
    test_options_source_type(
        "var f; { var f = function() { f() }; }",
        "var f, f = function() { f() };",
        source_type,
        &options,
    );
}

#[test]
fn keep_recursive_cycle_in_for_in_head() {
    // No removal site handles for-in/of head declarators, so the head
    // survives; its Annex-B initializer must keep referencing a live `p`.
    let options = CompressOptions::smallest();
    let source_type = SourceType::cjs().with_script(true);
    test_same_options_source_type(
        "function o() { var p = function() { console.log(x) }; for (var x = p in {}); return 1; } g(o());",
        source_type,
        &options,
    );
}

#[test]
fn keep_annex_b_block_level_function() {
    let options = CompressOptions::smallest();
    let source_type = SourceType::cjs().with_script(true);
    // A sloppy block-level function declaration also assigns a var-alias
    // binding at runtime (Annex B.3.3), and `Scoping` resolves outer
    // references to the FIRST same-named block function only, so later
    // duplicates' runtime re-assignments are invisible to reference
    // resolution. None of these declarations are safe liveness candidates.
    // (The zero-reference variant `{ function f() {} }` is still removed by
    // the count-based path — pre-existing.)
    test_same_options_source_type("{ function f() { f() } }", source_type, &options);
    test_same_options_source_type(
        "function o() { { function f() { console.log(1) } } f(); { function f() { console.log(2), g(f) } } f(); } o();",
        source_type,
        &options,
    );
    // Annex B applies to any sloppy code, not just script mode: `.cjs`
    // (rolldown's CommonJS source type) is sloppy but not `is_script`.
    test_same_options_source_type(
        "function o() { { function f() { console.log(1) } } f(); { function f() { console.log(2), g(f) } } f(); } o();",
        SourceType::cjs(),
        &options,
    );
}

// A symbol declared both by a candidate site and a non-candidate site must
// stay consistent at BOTH removal sites.
#[test]
fn recursive_function_with_var_redeclaration() {
    // The array init has a specialized (residue-leaving) handler, so the
    // declarator site is a non-candidate and its references root from live
    // context: the function site's removal is enabled, the residue survives.
    test_smallest("function f() { f() } var f = [g()];", "g();");
    test_same_smallest("function f() { f() } var f = [f];");
}

// Module-mode for-init declarators have no script protection and keep being
// removed.
#[test]
fn remove_recursive_for_init_declarator_in_module() {
    test_smallest("for (let f = () => f();;) break;", "for (;;) break;");
}

// A dropped direct eval must re-enable the analysis: the initial compute
// skips the whole program while the root scope carries `DirectEval`, so only
// the `eval_dropped` recompute trigger lets a later pass remove the cycle.
#[test]
fn remove_recursive_function_after_eval_dropped() {
    test_smallest("if (false) eval('x'); function f() { f() }", "");
}

#[test]
fn remove_unused_import_specifiers() {
    let options = CompressOptions::smallest();

    test_options("import a from 'a'", "import 'a';", &options);
    test_options("import a from 'a'; foo()", "import 'a'; foo();", &options);
    test_same_options(
        "import a from 'a'",
        &CompressOptions {
            treeshake: TreeShakeOptions {
                invalid_import_side_effects: true,
                ..TreeShakeOptions::default()
            },
            ..CompressOptions::smallest()
        },
    );

    test_options("import { a } from 'a'", "import 'a';", &options);
    test_options("import { a, b } from 'a'", "import 'a';", &options);

    test_options("import * as a from 'a'", "import 'a';", &options);

    test_options("import a, { b } from 'a'", "import 'a';", &options);
    test_options("import a, * as b from 'a'", "import 'a';", &options);

    test_same_options("import a from 'a'; foo(a);", &options);
    test_same_options("import { a } from 'a'; foo(a);", &options);
    test_same_options("import * as a from 'a'; foo(a);", &options);
    test_same_options("import a, { b } from 'a'; foo(a, b);", &options);

    test_options("import { a, b } from 'a'; foo(a);", "import { a } from 'a'; foo(a);", &options);
    test_options(
        "import { a, b, c } from 'a'; foo(b);",
        "import { b } from 'a'; foo(b);",
        &options,
    );
    test_options("import a, { b } from 'a'; foo(a);", "import a from 'a'; foo(a);", &options);
    test_options("import a, { b } from 'a'; foo(b);", "import { b } from 'a'; foo(b);", &options);

    test_options(
        "import a from 'a'; import { b } from 'b'; if (false) { console.log(b) }",
        "import 'a'; import 'b';",
        &options,
    );

    test_same_options("import 'a';", &options);

    test_options("import {} from 'a'", "import 'a';", &options);

    test_options(
        "import a from 'a' with { type: 'json' }",
        "import 'a' with { type: 'json' };",
        &options,
    );
    test_options(
        "import {} from 'a' with { type: 'json' }",
        "import 'a' with { type: 'json' };",
        &options,
    );

    test_options("import { a as b } from 'a'", "import 'a';", &options);
    test_same_options("import { a as b } from 'a'; foo(b);", &options);

    test_same_options("import { a } from 'a'; export { a };", &options);
    // Keep imports when direct eval is present
    test_same_options("import { a } from 'a'; eval('a');", &options);
    test_same_options("import a from 'a'; eval('a');", &options);
    test_same_options("import * as a from 'a'; eval('a');", &options);
    test_same_options("import { a } from 'a'; function f() { eval('a'); }", &options);
}

#[test]
fn remove_unused_import_source_statement() {
    let options = CompressOptions::smallest();

    test_options("import source a from 'a'", "", &options);
    test_options("import source a from 'a'; if (false) { console.log(a) }", "", &options);
    test_same_options("import source a from 'a'; foo(a);", &options);
    test_same_options(
        "import source a from 'a'",
        &CompressOptions {
            treeshake: TreeShakeOptions {
                invalid_import_side_effects: true,
                ..TreeShakeOptions::default()
            },
            ..CompressOptions::smallest()
        },
    );
}

#[test]
fn remove_unused_import_defer_statements() {
    let options = CompressOptions::smallest();

    test_options("import defer * as a from 'a'", "", &options);
    test_options("import defer * as a from 'a'; if (false) { console.log(a.foo) }", "", &options);
    test_same_options("import defer * as a from 'a'; foo(a);", &options);
    test_same_options("import defer * as a from 'a'; foo(a.bar);", &options);
    test_same_options(
        "import defer * as a from 'a'",
        &CompressOptions {
            treeshake: TreeShakeOptions {
                invalid_import_side_effects: true,
                ..TreeShakeOptions::default()
            },
            ..CompressOptions::smallest()
        },
    );
}
