// lower-mlir test.mlir -lower-torch-ops -print-mlir-after-all
// mlir-translate spirv.mlir --serialize-spirv -no-implicit-module >& output.spv
// Feed output.spv to SPIRV target compiler.

module {
	func.func @kernel(%primals_1: tensor<100xi32>, %primals_2: tensor<100xi32>, %out_add: tensor<100xi32>) {
		%add = mhlo.add %primals_1, %primals_2 : tensor<100xi32>
		linalg.copy ins(%add : tensor<100xi32>) outs(%out_add : tensor<100xi32>) -> tensor<100xi32>
		return
	}
}