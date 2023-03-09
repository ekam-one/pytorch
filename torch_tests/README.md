# PyTorch - MLIR Experiments

Experiments with PyTorch to generate MLIR from inductor backend and custom backend

## Building

### Build MLIR Tool
- Clone llvm repo  <br />
`` git clone https://github.com/ekam-one/llvm-project``
- Configure cmake <br />
`` cmake -G Ninja ../llvm -DCMAKE_BUILD_TYPE="Debug" -DLLVM_TARGETS_TO_BUILD="host" -DBUILD_SHARED_LIBS=1 -DCMAKE_C_COMPILER=clang-10 -DCMAKE_CXX_COMPILER=clang++-10 -DLLVM_ENABLE_PROJECTS="clang;mlir" -DLLVM_ENABLE_ASSERTIONS=ON``

- Build required binaries <br />
``ninja clang lower-mlir``
- Add build/bin to **PATH** so that Pytorch can detect lower-mlir tool and lower-mlir tool can detect clang-17 compiler to lower generated llvm ir to dynamic library.


### Build PyTorch

- Clone PyTorch Repo <br />
```git clone https://github.com/ekam-one/pytorch.git```
- Build and install PyTorch package <br />
```USE_DISTRIBUTED=0 USE_SYSTEM_SLEEF=1 USE_MKLDNN=0 python3.8  setup.py  install  --user```

### Running Tests

#### Inductor MLIR backend

``` TORCH_COMPILE_DEBUG=1 python3.8 trig.py ```

#### Custom backend
``` TORCH_COMPILE_DEBUG=1 python3.8 custom_backend.py ```

