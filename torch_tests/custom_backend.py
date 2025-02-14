import warnings
import torch
import torch._dynamo
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import make_boxed_func
import enum
from typing import Callable
import subprocess
from ctypes import cdll
from ctypes import c_void_p
from torch import empty_strided
from typing import List
from torch._dispatch.python import enable_python_dispatcher
from functorch import make_fx
from torch.fx import immutable_collections, Interpreter
from torch.nn.utils import stateless
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree
from torch._functorch.aot_autograd import create_forward_or_joint_functionalized
from torch._functorch.aot_autograd import CompiledRuntimeMetadata
from torch._functorch.aot_autograd import run_functionalized_fw_and_collect_metadata
from torch._functorch.aot_autograd import merge_view_inputs
from torch import Tensor
from torch._inductor.decomposition import select_decomp_table
import torch.fx._symbolic_trace as fx
from torch._functorch.aot_autograd import AOTConfig

class MlirDType(enum.Enum):
    f32 = 1
    i32 = 2
    f16 = 3
    bf16 = 4


mlir_type_dict = {}
mlir_type_dict[torch.float32] = MlirDType.f32
mlir_type_dict[torch.float16] = MlirDType.f16
mlir_type_dict[torch.int32] = MlirDType.i32
mlir_type_dict[torch.bfloat16] = MlirDType.bf16

torch_type_dict = {}
torch_type_dict[MlirDType.f32] = torch.float32
torch_type_dict[MlirDType.i32] = torch.int32
torch_type_dict[MlirDType.f16] = torch.float16
torch_type_dict[MlirDType.bf16] = torch.bfloat16


class MlirType:
    def __init__(self, dtype: MlirDType, shape: List):
        self.dtype = dtype
        self.shape = shape

    def __str__(self):
        string = ""

        if len(self.shape) > 0:
            string += "tensor<"
            for d in self.shape:
                string = string + str(d) + "x"

        string = string + self.dtype.name
        if len(self.shape) > 0:
            string += ">"
        return string

    def get_memref_string(self):
        string = ""

        if len(self.shape) > 0:
            string += "memref<"
            for d in self.shape:
                string = string + str(d) + "x"

        string = string + self.dtype.name
        if len(self.shape) > 0:
            string += ">"
        return string

    def __eq__(self, other):
        if not isinstance(other, MlirType):
            assert False, "Comparing objects of different class"
        return self.dtype == other.dtype and self.shape == other.shape

    def __ne__(self, other):
        if not isinstance(other, MlirType):
            assert False, "Comparing objects of different class"

        return self.dtype != other.dtype or self.shape != other.shape


class Codegen:
    def __init__(self, file_name, kernel_name, arg_types, arg_names, node_types):
        self.code = []
        self.node_types = node_types
        self.file_name = file_name
        self.prefix = str()
        self.codegen_func_prologue(kernel_name, arg_types, arg_names)

    def codegen_func_prologue(self, kernel_name, arg_types, arg_names):
        string = "module {\n"
        self.code.append(string)
        self.prefix += "\t"

        string = self.prefix + "func.func @" + kernel_name + "("

        assert len(arg_names) == len(arg_types)
        for name, type in zip(arg_names, arg_types):
            string = string + "%" + name + ": " + str(type) + ", "
        string = string[:-2]
        string = string + ") {\n"
        self.code.append(string)
        self.prefix = self.prefix + "\t"

    def codegen_func_epilogue(self):
        string = self.prefix + "return\n"

        self.code.append(string)
        if len(self.prefix) >= 1:
            self.prefix = self.prefix[:-1]
        self.code.append(self.prefix + "}\n")
        if len(self.prefix) >= 1:
            self.prefix = self.prefix[:-1]
        self.code.append(self.prefix + "}")

    def primals_1(self, node):
        pass

    def placeholder(self, node):
        pass

    def output(self, node):
        for i in range(len(node.all_input_nodes)):
            in_node = node.all_input_nodes[i]
            memref_type = self.node_types[in_node].get_memref_string()
            tempnode_name = in_node.name + "_tmp"
            string = (
                self.prefix
                + "%"
                + tempnode_name
                + " = bufferization.to_memref %"
                + ("out_"+in_node.name)
                + " : "
                + memref_type
                + "\n"
            )

            self.code.append(string)
            string = (
                self.prefix
                + "memref.tensor_store %"
                + in_node.name
                + ", %"
                + tempnode_name
                + " : "
                + memref_type
                + "\n"
            )
            self.code.append(string)

    def sin(self, node):
        string = self.prefix + "%" + node.name + " = math.sin %"
        assert len(node.all_input_nodes) == 1, "Expecting only one input for sin node"
        in_node = node.all_input_nodes[0]
        assert in_node in self.node_types
        string += in_node.name + " : " + str(self.node_types[in_node]) + "\n"
        self.code.append(string)

    def cos(self, node):
        string = self.prefix + "%" + node.name + " = math.cos %"
        assert len(node.all_input_nodes) == 1, "Expecting only one input for sin node"
        in_node = node.all_input_nodes[0]
        assert in_node in self.node_types
        string += in_node.name + " : " + str(self.node_types[in_node]) + "\n"
        self.code.append(string)

    def add(self, node):
        string = self.prefix + "%" + node.name + " = arith.addf "
        assert len(node.all_input_nodes) == 2, "Expecting only one input for add node"
        in_node1 = node.all_input_nodes[0]
        in_node2 = node.all_input_nodes[1]
        assert in_node1 in self.node_types and in_node2 in self.node_types
        string += (
            "%"
            + in_node1.name
            + ", %"
            + in_node2.name
            + " : "
            + str(self.node_types[in_node1])
            # + ", "
            # + str(self.node_types[in_node2])
            # + ") -> "
            # + str(self.node_types[in_node1])
            + "\n"
        )
        self.code.append(string)

    def mul(self, node):
        string = self.prefix + "%" + node.name + " = arith.mulf "
        assert len(node.all_input_nodes) == 2, "Expecting only one input for mul node"
        in_node1 = node.all_input_nodes[0]
        in_node2 = node.all_input_nodes[1]
        assert in_node1 in self.node_types and in_node2 in self.node_types
        string += (
            "%"
            + in_node1.name
            + ", %"
            + in_node2.name
            + " : "
            + str(self.node_types[in_node1])
            # + ", "
            # + str(self.node_types[in_node2])
            # + ") -> "
            # + str(self.node_types[in_node1])
            + "\n"
        )
        self.code.append(string)


    def sub(self, node):
        string = self.prefix + "%" + node.name + " = arith.subf "
        assert len(node.all_input_nodes) == 2, "Expecting only one input for sub node"
        in_node1 = node.all_input_nodes[0]
        in_node2 = node.all_input_nodes[1]
        assert in_node1 in self.node_types and in_node2 in self.node_types
        string += (
            "%"
            + in_node1.name
            + ", %"
            + in_node2.name
            + " : "
            + str(self.node_types[in_node1])
            # + ", "
            # + str(self.node_types[in_node2])
            # + ") -> "
            # + str(self.node_types[in_node1])
            + "\n"
        )
        self.code.append(string)

    def call_function(self, node):
        target_fn = str()
        if isinstance(node.target, Callable):
            target_fn = getattr(node.target, "__name__", "unknown")
        elif isinstance(node.target, str):
            target_fn = node.target
        else:
            assert False, "Unknown target type in call_function"
        
        if target_fn.endswith(".default"):
            target_fn = target_fn[:-len(".default")]

        if target_fn.endswith(".Tensor"):
            target_fn = target_fn[:-len(".Tensor")]

        getattr(self, target_fn)(node)

    def codegen_node(self, node):
        print(node)
        getattr(self, node.op)(node)

    def unknwon(self, node):
        assert False, "Fix call_function"

    def flush(self):
        with open(self.file_name, "w") as fd:
            fd.writelines(self.code)


decompositions = select_decomp_table()

class MlirFxBackend:
    def __init__(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        print("MlirFxBackend init called")
        print(gm)
        self.gm = gm
        self.node_types = {}
        self.inputs = example_inputs
        self.arg_types = []
        self.arg_names = []
        self.output_types = []
        self.kernel_file_name = "torch_kernel.mlir"
        self.kernel_name = "kernel"
        self.lib_name = "./torch_library.so"
        self.kernel = self.compile_and_load()

    def get_lower_cmd(self):
        return [
            "lower-mlir",
            self.kernel_file_name,
            "-lower-torch-ops",
            "-o",
            self.lib_name,
        ]

    def lower_and_load(self):
        cmd = self.get_lower_cmd()
        print("Lowering with command ", cmd)
        try:
            subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            raise Exception("Compilation failed ", e)

        try:
            lib_handle = cdll.LoadLibrary(self.lib_name)
        except OSError as e:
            assert False, "Failed to load the library"

        return lib_handle.kernel

    def kernel_func(self, arg1, arg2):
        print("************This is dummmy lowering! remove *************")
        x = torch.sin(arg1) + torch.sin(arg2)
        return (x,)

    def __call__(self, *args, **kwargs):

        print("__call__ custom called")

        output_args = []

        for out_type in self.output_types:
            out_arg =  empty_strided(out_type.shape, (1,), dtype=torch_type_dict[out_type.dtype])
            output_args.append(out_arg)

        actual_args = list(args)
        ptrs = []
        for a_arg in actual_args:
            ptrs.append(c_void_p(a_arg.data_ptr()))
        
        for out_arg in output_args:
           ptrs.append(c_void_p(out_arg.data_ptr())) 

        self.kernel(*ptrs)
        return tuple(output_args)

    def infer_arg_type(self, node, i):
        input = self.inputs[i]
        shape = input.shape
        tensor_shape = []
        for d in shape:
            tensor_shape.append(d)

        mlir_type = MlirType(mlir_type_dict[input.dtype], tensor_shape)
        self.node_types[node] = mlir_type
        self.arg_types.append(mlir_type)
        self.arg_names.append(node.name)

    def infer_call_function(self, node):
        input_nodes = node.all_input_nodes
        if len(input_nodes) == 0:
            assert False, "Unhandled node in shape inference"

        if len(input_nodes) == 1:
            self.node_types[node] = self.node_types[input_nodes[0]]
            return

        first_type = self.node_types[input_nodes[0]]
        for i in range(len(input_nodes) - 1):
            assert (
                input_nodes[i] in self.node_types
            ), "Argument type is not yet inferred"
            current_type = self.node_types[input_nodes[i]]
            assert (
                current_type == first_type
            ), "Nodes with different argument types are not handled, for node " + str(
                node
            )

        self.node_types[node] = first_type

    def infer_output(self, node):
        input_nodes = node.all_input_nodes
        
        for i in range(len(input_nodes)):
           assert input_nodes[i] in self.node_types
           self.output_types.append(self.node_types[input_nodes[i]])
           self.arg_types.append(self.node_types[input_nodes[i]])
           self.arg_names.append("out_"+input_nodes[i].name)

    def compile_and_load(self):
        i = 0
        for node in self.gm.graph.nodes:
            if node.op == "placeholder":
                self.infer_arg_type(node, i)
                i += 1
                continue
            getattr(self, "infer_" + node.op)(node)

        cg = Codegen(
            self.kernel_file_name,
            self.kernel_name,
            self.arg_types,
            self.arg_names,
            self.node_types,
        )

        print(self.gm.graph)
        for node in self.gm.graph.nodes:
            cg.codegen_node(node)
        cg.codegen_func_epilogue()
        cg.flush()

        return self.lower_and_load()



def my_compiler(gm, example_inputs):
  backend = MlirFxBackend(gm, example_inputs)
  return make_boxed_func(backend.__call__)


my_backend = aot_autograd(
            fw_compiler=my_compiler,
            bw_compiler=my_compiler,
            decompositions=decompositions,)



def fn(x):
    return torch.sin(x)
    #return torch.softmax(x, -1)

input_tensor = torch.range(0, 99, requires_grad=True)

model_opt = torch.compile(fn, backend=my_backend)

out = model_opt(input_tensor)
print(out)

print(out.sum().backward())