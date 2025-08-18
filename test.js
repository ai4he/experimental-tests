module.evaluate_without_middleware().then(time => console.log(`Without middleware time: ${time} ms`)); // Run base-10 benchmark
module.evaluate_with_middleware().then(time => console.log(`With middleware time: ${time} ms`)); // Run base-12 benchmark
const assembly_dec = `...` // Copy ASSEMBLY_WITHOUT
module.run_assembly(assembly_dec).then(result => console.log(`Dec sum: ${result}`)); // "5000050000"
const assembly_duo = `...` // Copy ASSEMBLY_WITH
module.run_assembly(assembly_duo).then(result => console.log(`Duo sum: ${result}`)); // Base-12 string for 5000050000 in base-12
