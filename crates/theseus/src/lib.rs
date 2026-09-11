//! **Theseus** — Force Density Method form-finding solver with hand-coded adjoints.
//!
//! This crate implements the complete FDM optimisation pipeline:
//!
//! 1. **Forward solve** (`fdm`): assemble A(q), factorise, triangular solve.
//! 2. **Objectives** (`objectives`): 13 loss functions on geometry / forces / reactions.
//! 3. **Gradients** (`gradients`): hand-coded adjoint + explicit derivatives.
//! 4. **Optimiser** (`optimizer`): argmin L-BFGS for soft barriers and
//!    Fortran-v3.0-compatible L-BFGS-B for direct box constraints. Box-mode
//!    stopping uses the projected-gradient infinity norm and relative
//!    accepted-iterate function reduction.
//! 5. **FFI** (`ffi`): C-compatible API for Grasshopper / C# P/Invoke.
//!
//! All public functions return `Result<_, TheseusError>` — the crate never
//! panics in normal operation.

pub mod fdm;
pub mod ffi;
pub mod gradients;
pub mod inverse;
pub mod nullspace;
pub mod objectives;
pub mod optimizer;
pub mod sparse;
pub mod types;
pub mod variable_supports;

pub use types::ObjectiveTrait;
pub use types::TheseusError;
