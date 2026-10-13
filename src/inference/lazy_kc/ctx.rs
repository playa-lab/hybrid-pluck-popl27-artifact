use crate::language::define::DefinitionRegistry;
use crate::language::types::{StringInterner, Symbol, TypeRegistry};

use super::state::LazyKCState;

pub struct CompilerCtx<'a> {
    pub types: &'a TypeRegistry,
    pub defs: &'a DefinitionRegistry,
    pub interner: &'a StringInterner,
    pub true_sym: Symbol,
    pub false_sym: Symbol,
    pub state: &'a mut LazyKCState,
}
