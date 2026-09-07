# Result constructor typing ergonomics implementation plan

**Goal:** Model concrete variants by their active payload so `Ok(value)` and `Err(error)` fit annotated `Result[A, E]` values without phantom type arguments.

**Architecture:** Use `Ok[T]` and `Err[E]` as the concrete immutable variants and `Result[A, E] = Ok[A] | Err[E]`. Refactor branch methods, guards, helpers, tests, and documentation to preserve payload narrowing and error unions.

**Validation:**
- Replace all concrete `Ok[...]`/`Err[...]` constructor calls with `Ok(...)`/`Err(...)`.
- Add static coverage for constructor assignability, branch narrowing, matching, and sync/async chaining.
- Run `ruff format src tests`, `.venv/bin/ruff check src tests`, `.venv/bin/ty check src tests`, and `.venv/bin/pytest --cov`.
