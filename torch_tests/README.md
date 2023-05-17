### Pytorch Build steps

### Step 1: 

Clone llvm-project, mlir-hlo , pytorch repo & checkout 'dev/mhlo_exp' branch in all the three repos.

``` git clone https://github.com/ekam-one/llvm-project ```

``` git clone https://github.com/ekam-one/pytorch.git ```

``` git clone https://github.com/ekam-one/mlir-hlo.git ```

 
### Step 2: 

cd llvm-project && mkdir build && cd build  

```cmake -GNinja -B. ../llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="mlir;clang" -DLLVM_EXTERNAL_PROJECTS="mlir_hlo" -DLLVM_EXTERNAL_MLIR_HLO_SOURCE_DIR=<path to mlir-hlo repo>/mlir-hlo -DLLVM_TARGETS_TO_BUILD=host -DPython3_EXECUTABLE=/usr/bin/python3.8 -DMLIR_ENABLE_BINDINGS_PYTHON=ON -DMHLO_ENABLE_BINDINGS_PYTHON=ON```

After successful cmake, 

``` ninja clang MLIRHLOPythonModules ```


### Step 3: 

Add build/bin to PATH

```export PYTHONPATH=<build>/tools/mlir_hlo/python_packages/mlir_hlo```

 
### Step 4:

### Build PyTorch

```cd <pytorch-repo>```
- Build and install PyTorch package <br />
```USE_DISTRIBUTED=0 USE_SYSTEM_SLEEF=1 USE_MKLDNN=0 USE_CUDA=0 BUILD_CAFFE2=0 BUILD_CAFFE2_OPS=0 USE_OPENMP=0 CC=clang<8.0> CXX=clang<8.0>++ python3.8 setup.py install --user```

### Running Tests

#### Inductor MLIR backend

``` TORCH_COMPILE_DEBUG=1 python3.8 trig.py ```

#### Custom backend
``` TORCH_COMPILE_DEBUG=1 python3.8 custom_backend.py ```