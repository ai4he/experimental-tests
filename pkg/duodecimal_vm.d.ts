/* tslint:disable */
/* eslint-disable */
export function analyze_recursive_stability(iterations: number): string;
export function run_assembly(assembly: string): string;
export function add(a: string, b: string): string;
export function to_duodecimal(value: number, scale: number): string;
export function from_duodecimal(s: string): number;
export function tune_weights(weights: Float64Array, scale: number): Float64Array;
export function tune_weights_at_cycle(weights: Float64Array, scale: number, cycle: number): Float64Array;
export function evaluate_drift(weights: Float64Array, iterations: number, scale: number, with_tuning: boolean): number;
export function evaluate_without_middleware(): number;
export function evaluate_with_middleware(): number;
export function main(): void;
export class AIModelSimulation {
  private constructor();
  free(): void;
  static new(size: number): AIModelSimulation;
  train_iteration(): void;
  get_drift_comparison(): string;
}
export class QuantumSimulation {
  private constructor();
  free(): void;
  static new(): QuantumSimulation;
  simulate_decoherence(): void;
  get_coherence_report(): string;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_aimodelsimulation_free: (a: number, b: number) => void;
  readonly aimodelsimulation_new: (a: number) => number;
  readonly aimodelsimulation_train_iteration: (a: number) => void;
  readonly aimodelsimulation_get_drift_comparison: (a: number) => [number, number];
  readonly __wbg_quantumsimulation_free: (a: number, b: number) => void;
  readonly quantumsimulation_new: () => number;
  readonly quantumsimulation_simulate_decoherence: (a: number) => void;
  readonly quantumsimulation_get_coherence_report: (a: number) => [number, number];
  readonly analyze_recursive_stability: (a: number) => [number, number];
  readonly run_assembly: (a: number, b: number) => [number, number];
  readonly add: (a: number, b: number, c: number, d: number) => [number, number];
  readonly to_duodecimal: (a: number, b: number) => [number, number];
  readonly from_duodecimal: (a: number, b: number) => number;
  readonly tune_weights: (a: number, b: number, c: number) => [number, number];
  readonly tune_weights_at_cycle: (a: number, b: number, c: number, d: number) => [number, number];
  readonly evaluate_drift: (a: number, b: number, c: number, d: number, e: number) => number;
  readonly evaluate_without_middleware: () => number;
  readonly evaluate_with_middleware: () => number;
  readonly main: () => void;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly __externref_table_alloc: () => number;
  readonly __wbindgen_export_2: WebAssembly.Table;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
