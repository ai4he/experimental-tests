/* tslint:disable */
/* eslint-disable */
export const memory: WebAssembly.Memory;
export const __wbg_aimodelsimulation_free: (a: number, b: number) => void;
export const aimodelsimulation_new: (a: number) => number;
export const aimodelsimulation_train_iteration: (a: number) => void;
export const aimodelsimulation_get_drift_comparison: (a: number) => [number, number];
export const __wbg_quantumsimulation_free: (a: number, b: number) => void;
export const quantumsimulation_new: () => number;
export const quantumsimulation_simulate_decoherence: (a: number) => void;
export const quantumsimulation_get_coherence_report: (a: number) => [number, number];
export const analyze_recursive_stability: (a: number) => [number, number];
export const run_assembly: (a: number, b: number) => [number, number];
export const add: (a: number, b: number, c: number, d: number) => [number, number];
export const to_duodecimal: (a: number, b: number) => [number, number];
export const from_duodecimal: (a: number, b: number) => number;
export const tune_weights: (a: number, b: number, c: number) => [number, number];
export const tune_weights_at_cycle: (a: number, b: number, c: number, d: number) => [number, number];
export const evaluate_drift: (a: number, b: number, c: number, d: number, e: number) => number;
export const evaluate_without_middleware: () => number;
export const evaluate_with_middleware: () => number;
export const main: () => void;
export const __wbindgen_exn_store: (a: number) => void;
export const __externref_table_alloc: () => number;
export const __wbindgen_export_2: WebAssembly.Table;
export const __wbindgen_free: (a: number, b: number, c: number) => void;
export const __wbindgen_malloc: (a: number, b: number) => number;
export const __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
export const __wbindgen_start: () => void;
