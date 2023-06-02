import logging

from ..backends.common import aot_autograd
from ..backends.registry import register_experimental_backend as register_backend

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
import numpy as np

import mlir
from mlir.ir import *
import mlir.dialects.mhlo as mhlo
import mlir.dialects.vector as vector
import mlir.dialects.arith as arith
from mlir.dialects import func
from mlir.dialects import linalg
import mlir.dialects.tensor as tensor


class MlirDType(enum.Enum):
    f32 = 1
    i32 = 2
    f16 = 3
    bf16 = 4
    i64 = 5


mlir_type_dict = {}
mlir_type_dict[torch.float32] = MlirDType.f32
mlir_type_dict[torch.float16] = MlirDType.f16
mlir_type_dict[torch.int32] = MlirDType.i32
mlir_type_dict[torch.bfloat16] = MlirDType.bf16
mlir_type_dict[torch.int64] = MlirDType.i64

torch_type_dict = {}
torch_type_dict[MlirDType.f32] = torch.float32
torch_type_dict[MlirDType.i32] = torch.int32
torch_type_dict[MlirDType.f16] = torch.float16
torch_type_dict[MlirDType.bf16] = torch.bfloat16
torch_type_dict[MlirDType.i64] = torch.int64

# core_mlir_type_dict = {}
# with Context() as ctx:


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

    def get_mlir_tensor_type(self):
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


def make_ir_context():
    """Creates an MLIR context suitable for JAX IR."""
    context = Context()
    context.enable_multithreading(False)
    mhlo.register_mhlo_dialect(context)
    # linalg.register_dialect(context)
    return context


class Codegen:
    def __init__(self, file_name, kernel_name, arg_types, arg_names, node_types):
        self.code = []
        self.input_args = 0
        self.node_types = node_types
        self.file_name = file_name
        self.prefix = str()
        self.module = None
        self.func = None
        self.op_dict = {}
        self.ctx = make_ir_context()
        self.core_mlir_type_dict = {}
        self.core_mlir_type_to_np_dict = {}
        with self.ctx:
            self.locA = Location.file(self.file_name, 1, 0)
            f32Ty = F32Type.get()
            self.core_mlir_type_dict[MlirDType.f32] = f32Ty
            self.core_mlir_type_to_np_dict[f32Ty] = np.float32
        self.codegen_func_prologue(kernel_name, arg_types, arg_names)

    # assuming that we are not returning anything frm any kernel since we are converting
    # return statement into store
    def codegen_func_prologue(self, kernel_name, arg_types, arg_names):
        assert len(arg_names) == len(arg_types)
        for arg_name in arg_names:
            if "out" not in arg_name:
                self.input_args += 1
        argTypes = []
        with self.ctx, self.locA:
            for type in arg_types:
                if isinstance(type, MlirType):
                    argTypes.append(
                        RankedTensorType.get(
                            tuple(type.shape), self.core_mlir_type_dict[type.dtype]
                        )
                    )
                else:
                    assert (False, "handle scalar args")
            args = argTypes
            temp = []
            final_arg = (args, temp)

            self.module = Module.create()
            with InsertionPoint(self.module.body):
                self.func = func.FuncOp(kernel_name, final_arg)
            with InsertionPoint(self.func.add_entry_block()):
                func.ReturnOp([])
            for name, barg in zip(arg_names, self.func.regions[0].blocks[0].arguments):
                self.op_dict[name] = barg

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
        with self.ctx, self.locA:
            with InsertionPoint.at_block_terminator(self.func.regions[0].blocks[0]):
                for i in range(len(node.all_input_nodes)):
                    in_node = node.all_input_nodes[i]
                    inval = self.op_dict[in_node.name]
                    arg_num = self.input_args + i
                    if arg_num >= len(self.func.regions[0].blocks[0].arguments):
                        break
                    outval = self.func.regions[0].blocks[0].arguments[arg_num]
                    input = (inval,)
                    output = (outval,)
                    outputTy = (outval.type,)
                    tnsrTy = RankedTensorType(inval.type)
                    tensor.InsertSliceOp(
                        inval,
                        outval,
                        [],
                        [],
                        [],
                        DenseI64ArrayAttr.get([0] * len(tnsrTy.shape)),
                        DenseI64ArrayAttr.get(tnsrTy.shape),
                        DenseI64ArrayAttr.get([1] * len(tnsrTy.shape)),
                    )

    def sin(self, node):
        assert len(node.all_input_nodes) == 1, "Expecting only one input for sin node"
        in_node1 = node.all_input_nodes[0]
        lhs = self.op_dict[in_node1.name]
        with self.ctx, self.locA:
            with InsertionPoint.at_block_terminator(self.func.regions[0].blocks[0]):
                add_op = mhlo.SineOp(lhs).result
                self.op_dict[node.name] = add_op

    def cos(self, node):
        assert len(node.all_input_nodes) == 1, "Expecting only one input for sin node"
        in_node1 = node.all_input_nodes[0]
        lhs = self.op_dict[in_node1.name]
        with self.ctx, self.locA:
            with InsertionPoint.at_block_terminator(self.func.regions[0].blocks[0]):
                add_op = mhlo.CosineOp(lhs).result
                self.op_dict[node.name] = add_op

    def exp(self, node):
        assert len(node.all_input_nodes) == 1, "Expecting only one input for sin node"
        in_node1 = node.all_input_nodes[0]
        lhs = self.op_dict[in_node1.name]
        with self.ctx, self.locA:
            with InsertionPoint.at_block_terminator(self.func.regions[0].blocks[0]):
                add_op = mhlo.ExpOp(lhs).result
                self.op_dict[node.name] = add_op

    # we are assuming reduction happens only along 1 dimension as of now
    def amax(self, node):
        assert len(node.all_input_nodes) == 1, "Expecting only one input for sin node"

        in_node1 = node.all_input_nodes[0]
        lhs = self.op_dict[in_node1.name]
        in_node1 = node.all_input_nodes[0]
        lhs = self.op_dict[in_node1.name]
        nodeTy = lhs.type
        nodeTy = RankedTensorType(lhs.type)
        eleTy = nodeTy.element_type
        with self.ctx, self.locA:
            with InsertionPoint.at_block_terminator(self.func.regions[0].blocks[0]):
                init_val = mhlo.ConstantOp(
                    DenseElementsAttr.get(
                        np.zeros([], dtype=self.core_mlir_type_to_np_dict[eleTy]),
                        type=eleTy,
                    )
                ).result
                reduce_dim = node.args[1][0]
                input_shape = nodeTy.shape
                num_dims = len(input_shape)
                del input_shape[reduce_dim]
                resty = RankedTensorType.get(shape=input_shape, element_type=eleTy)
                add_op = mhlo.ReduceOp(
                    [resty],
                    [lhs],
                    [init_val],
                    DenseIntElementsAttr.get(
                        np.asarray(reduce_dim, np.int64), shape=[1]
                    ),
                )
                block = Block.create_at_start(
                    add_op.body,
                    [
                        RankedTensorType.get(shape=[], element_type=eleTy),
                        RankedTensorType.get(shape=[], element_type=eleTy),
                    ],
                )
                with InsertionPoint.at_block_begin(block):
                    maxop = mhlo.MaxOp(
                        add_op.body.blocks[0].arguments[0],
                        add_op.body.blocks[0].arguments[1],
                    ).result
                    mhlo.ReturnOp([maxop])
                broadcast_dims = [dd for dd in range(num_dims)]
                del broadcast_dims[reduce_dim]
                broadcast = mhlo.BroadcastInDimOp(
                    lhs.type,
                    add_op.result,
                    DenseIntElementsAttr.get(np.asarray(broadcast_dims, np.int64)),
                )
                self.op_dict[node.name] = broadcast.result

    # we are assuming reduction happens only along 1 dimension as of now
    def sum_1(self, node):
        assert len(node.all_input_nodes) == 1, "Expecting only one input for sin node"

        in_node1 = node.all_input_nodes[0]
        lhs = self.op_dict[in_node1.name]
        nodeTy = lhs.type
        nodeTy = RankedTensorType(lhs.type)
        eleTy = nodeTy.element_type
        with self.ctx, self.locA:
            with InsertionPoint.at_block_terminator(self.func.regions[0].blocks[0]):
                init_val = mhlo.ConstantOp(
                    DenseElementsAttr.get(
                        np.zeros([], dtype=self.core_mlir_type_to_np_dict[eleTy]),
                        type=eleTy,
                    )
                ).result
                reduce_dim = node.args[1][0]
                input_shape = nodeTy.shape
                num_dims = len(input_shape)
                del input_shape[reduce_dim]
                resty = RankedTensorType.get(shape=input_shape, element_type=eleTy)
                add_op = mhlo.ReduceOp(
                    [resty],
                    [lhs],
                    [init_val],
                    DenseIntElementsAttr.get(
                        np.asarray(reduce_dim, np.int64), shape=[1]
                    ),
                )
                block = Block.create_at_start(
                    add_op.body,
                    [
                        RankedTensorType.get(shape=[], element_type=eleTy),
                        RankedTensorType.get(shape=[], element_type=eleTy),
                    ],
                )
                with InsertionPoint.at_block_begin(block):
                    maxop = mhlo.AddOp(
                        add_op.body.blocks[0].arguments[0],
                        add_op.body.blocks[0].arguments[1],
                    ).result
                    mhlo.ReturnOp([maxop])
                broadcast_dims = [dd for dd in range(num_dims)]
                del broadcast_dims[reduce_dim]
                broadcast = mhlo.BroadcastInDimOp(
                    lhs.type,
                    add_op.result,
                    DenseIntElementsAttr.get(np.asarray(broadcast_dims, np.int64)),
                )
                self.op_dict[node.name] = broadcast.result

    def add(self, node):
        assert len(node.all_input_nodes) == 2, "Expecting only one input for add node"
        in_node1 = node.all_input_nodes[0]
        in_node2 = node.all_input_nodes[1]
        assert in_node1 in self.node_types and in_node2 in self.node_types

        lhs = self.op_dict[in_node1.name]
        rhs = self.op_dict[in_node2.name]

        with self.ctx, self.locA:
            with InsertionPoint.at_block_terminator(self.func.regions[0].blocks[0]):
                add_op = mhlo.AddOp(lhs, rhs).result
                self.op_dict[node.name] = add_op

    def mul(self, node):
        assert len(node.all_input_nodes) == 2, "Expecting only one input for mul node"
        assert in_node1 in self.node_types and in_node2 in self.node_types
        in_node1 = node.all_input_nodes[0]
        in_node2 = node.all_input_nodes[1]
        lhs = self.op_dict[in_node1.name]
        rhs = self.op_dict[in_node2.name]
        with self.ctx, self.locA:
            with InsertionPoint.at_block_terminator(self.func.regions[0].blocks[0]):
                add_op = mhlo.MulOp(lhs, rhs).result
                self.op_dict[node.name] = add_op

    def sub(self, node):
        assert len(node.all_input_nodes) == 2, "Expecting only one input for sub node"
        in_node1 = node.all_input_nodes[0]
        in_node2 = node.all_input_nodes[1]
        assert in_node1 in self.node_types and in_node2 in self.node_types
        lhs = self.op_dict[in_node1.name]
        rhs = self.op_dict[in_node2.name]
        with self.ctx, self.locA:
            with InsertionPoint.at_block_terminator(self.func.regions[0].blocks[0]):
                add_op = mhlo.SubtractOp(lhs, rhs).result
                self.op_dict[node.name] = add_op

    def mm(self, node):
        in_node1 = node.all_input_nodes[0]
        in_node2 = node.all_input_nodes[1]
        lhs = self.op_dict[in_node1.name]
        rhs = self.op_dict[in_node2.name]
        ty = self.node_types[node]
        with self.ctx, self.locA:
            with InsertionPoint.at_block_terminator(self.func.regions[0].blocks[0]):
                res_ty = RankedTensorType.get(
                    shape=ty.shape, element_type=self.core_mlir_type_dict[ty.dtype]
                )
                add_op = mhlo.DotOp(res_ty, lhs, rhs).result
                self.op_dict[node.name] = add_op

    def div(self, node):
        in_node1 = node.all_input_nodes[0]
        in_node2 = node.all_input_nodes[1]
        lhs = self.op_dict[in_node1.name]
        rhs = self.op_dict[in_node2.name]
        with self.ctx, self.locA:
            with InsertionPoint.at_block_terminator(self.func.regions[0].blocks[0]):
                add_op = mhlo.DivOp(lhs, rhs).result
                self.op_dict[node.name] = add_op

    def addmm(self, node):
        in_node0 = node.all_input_nodes[0]
        in_node1 = node.all_input_nodes[1]
        in_node2 = node.all_input_nodes[2]
        alhs = self.op_dict[in_node0.name]
        lhs = self.op_dict[in_node1.name]
        rhs = self.op_dict[in_node2.name]
        input_ty = lhs.type
        tnsr_ty = RankedTensorType(input_ty)
        shape = tnsr_ty.shape
        input_ty1 = rhs.type
        tnsr_ty1 = RankedTensorType(input_ty1)
        shape1 = tnsr_ty1.shape
        ele_ty = tnsr_ty.element_type
        new_shape = [shape[0], shape1[1]]
        with self.ctx, self.locA:
            with InsertionPoint.at_block_terminator(self.func.regions[0].blocks[0]):
                res_ty = RankedTensorType.get(shape=new_shape, element_type=ele_ty)
                add_op = mhlo.DotOp(res_ty, lhs, rhs).result
                broadcast = mhlo.BroadcastInDimOp(
                    res_ty, alhs, DenseIntElementsAttr.get(np.asarray([1], np.int64))
                ).result
                fadd = mhlo.AddOp(broadcast, add_op).result
                self.op_dict[node.name] = fadd

    def permute(self, node):
        assert len(node.all_input_nodes) == 1, "Expecting only one input for sin node"
        in_node1 = node.all_input_nodes[0]
        assert in_node1 in self.node_types
        lhs = self.op_dict[in_node1.name]
        ty = self.node_types[node]
        with self.ctx, self.locA:
            with InsertionPoint.at_block_terminator(self.func.regions[0].blocks[0]):
                res_ty = RankedTensorType.get(
                    shape=ty.shape, element_type=self.core_mlir_type_dict[ty.dtype]
                )
                add_op = mhlo.ReshapeOp(res_ty, lhs).result
                self.op_dict[node.name] = add_op

    def unsqueeze(self, node):
        assert len(node.all_input_nodes) == 1, "Expecting only one input for sin node"
        in_node1 = node.all_input_nodes[0]
        assert in_node1 in self.node_types
        lhs = self.op_dict[in_node1.name]
        ty = self.node_types[node]
        with self.ctx, self.locA:
            with InsertionPoint.at_block_terminator(self.func.regions[0].blocks[0]):
                res_ty = RankedTensorType.get(
                    shape=ty.shape, element_type=self.core_mlir_type_dict[ty.dtype]
                )
                add_op = mhlo.ReshapeOp(res_ty, lhs).result
                self.op_dict[node.name] = add_op

    def squeeze(self, node):
        assert len(node.all_input_nodes) == 1, "Expecting only one input for sin node"
        in_node1 = node.all_input_nodes[0]
        assert in_node1 in self.node_types
        lhs = self.op_dict[in_node1.name]
        ty = self.node_types[node]
        with self.ctx, self.locA:
            with InsertionPoint.at_block_terminator(self.func.regions[0].blocks[0]):
                res_ty = RankedTensorType.get(
                    shape=ty.shape, element_type=self.core_mlir_type_dict[ty.dtype]
                )
                add_op = mhlo.ReshapeOp(res_ty, lhs).result
                self.op_dict[node.name] = add_op

    def relu(self, node):
        assert len(node.all_input_nodes) == 1, "Expecting only one input for sin node"
        in_node1 = node.all_input_nodes[0]
        assert in_node1 in self.node_types
        lhs = self.op_dict[in_node1.name]
        input_ty = lhs.type
        tnsr_ty = RankedTensorType(input_ty)
        shape = tnsr_ty.shape
        ele_ty = tnsr_ty.element_type
        with self.ctx, self.locA:
            with InsertionPoint.at_block_terminator(self.func.regions[0].blocks[0]):
                init_val = mhlo.ConstantOp(
                    DenseElementsAttr.get(
                        np.zeros(shape, dtype=self.core_mlir_type_to_np_dict[ele_ty]),
                        type=ele_ty,
                    )
                ).result
                add_op = mhlo.MaxOp(lhs, init_val).result
                self.op_dict[node.name] = add_op

    def call_function(self, node):
        target_fn = str()
        if isinstance(node.target, Callable):
            target_fn = getattr(node.target, "__name__", "unknown")
        elif isinstance(node.target, str):
            target_fn = node.target
        else:
            assert False, "Unknown target type in call_function"

        if target_fn.endswith(".default"):
            target_fn = target_fn[: -len(".default")]

        if target_fn.endswith(".Tensor"):
            target_fn = target_fn[: -len(".Tensor")]

        getattr(self, target_fn)(node)

    def codegen_node(self, node):
        print(node)
        if node.name == "sum_1":
            getattr(self, node.name)(node)
        elif "squeeze" in node.name and "un" not in node.name:
            getattr(self, "squeeze")(node)
        else:
            getattr(self, node.op)(node)

    def unknwon(self, node):
        assert False, "Fix call_function"

    def flush(self):
        tempvar = str(self.module)
        with open(self.file_name, "w") as fd:
            fd.write(tempvar)


decompositions = select_decomp_table()


class MlirFxBackend:
    def __init__(self, gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        print("MlirFxBackend init called")
        print(gm)  # .print_readable())
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
        # try:
        #     subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        # except subprocess.CalledProcessError as e:
        #     raise Exception("Compilation failed ", e)

        # try:
        #     lib_handle = cdll.LoadLibrary(self.lib_name)
        # except OSError as e:
        #     assert False, "Failed to load the library"

        # return lib_handle.kernel

    def kernel_func(self, arg1, arg2):
        print("************This is dummmy lowering! remove *************")
        x = torch.sin(arg1) + torch.sin(arg2)
        return (x,)

    def __call__(self, *args, **kwargs):
        print("__call__ custom called")

        output_args = []

        for out_type in self.output_types:
            out_arg = empty_strided(
                out_type.shape, (1,), dtype=torch_type_dict[out_type.dtype]
            )
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

        if "permute" in node.name:
            assert (
                input_nodes[0] in self.node_types
            ), "Argument type is not yet inferred"
            input_ty = self.node_types[input_nodes[0]]
            input_dtype = input_ty.dtype
            input_shape = input_ty.shape
            new_shape = [0] * len(input_shape)
            dim = node.args[1]
            for i in range(len(input_shape)):
                new_shape[i] = input_shape[dim[i]]
            self.node_types[node] = MlirType(input_dtype, new_shape)
            return

        if "unsqueeze" in node.name:
            assert (
                input_nodes[0] in self.node_types
            ), "Argument type is not yet inferred"
            input_ty = self.node_types[input_nodes[0]]
            input_dtype = input_ty.dtype
            input_shape = input_ty.shape
            dim = node.args[1]
            new_shape = input_shape[:]
            new_shape.insert(dim, 1)
            self.node_types[node] = MlirType(input_dtype, new_shape)
            return

        if "squeeze" in node.name:
            assert (
                input_nodes[0] in self.node_types
            ), "Argument type is not yet inferred"
            input_ty = self.node_types[input_nodes[0]]
            input_dtype = input_ty.dtype
            input_shape = input_ty.shape
            dim = node.args[1]
            new_shape = input_shape[:]
            del new_shape[dim]
            self.node_types[node] = MlirType(input_dtype, new_shape)
            return

        if "mm" in node.name:
            input_ty = self.node_types[input_nodes[0]]
            input_ty1 = self.node_types[input_nodes[1]]
            input_dtype = input_ty.dtype
            input_shape = input_ty.shape
            input_shape1 = input_ty1.shape
            new_shape = [input_shape[0], input_shape1[1]]
            self.node_types[node] = MlirType(input_dtype, new_shape)
            return

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
            if "primals" not in input_nodes[i].name:
                assert input_nodes[i] in self.node_types
                self.output_types.append(self.node_types[input_nodes[i]])
                self.arg_types.append(self.node_types[input_nodes[i]])
                self.arg_names.append("out_" + input_nodes[i].name)

    def compile_and_load(self):
        i = 0
        for node in self.gm.graph.nodes:
            if node.op == "placeholder":
                self.infer_arg_type(node, i)
                i += 1
                continue
            getattr(self, "infer_" + node.op)(node)

        print(self.gm.graph)

        cg = Codegen(
            self.kernel_file_name,
            self.kernel_name,
            self.arg_types,
            self.arg_names,
            self.node_types,
        )

        for node in self.gm.graph.nodes:
            cg.codegen_node(node)
        cg.codegen_func_epilogue()
        cg.flush()
        print(cg.module)
        return self.lower_and_load()


@register_backend
def my_compiler(gm, example_inputs):
    backend = MlirFxBackend(gm, example_inputs)
    return make_boxed_func(backend.__call__)


aot_torchmhlo = aot_autograd(
    fw_compiler=my_compiler,
    bw_compiler=my_compiler,
    decompositions=decompositions,
)

register_backend(name="torchmhlo", compiler_fn=aot_torchmhlo)
