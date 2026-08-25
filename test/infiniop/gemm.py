import ctypes
from ctypes import c_uint64

import torch
from libinfiniop import (
    LIBINFINIOP,
    InfiniDeviceNames,
    InfiniDtype,
    InfiniDtypeNames,
    TestTensor,
    TestWorkspace,
    check_error,
    debug,
    get_args,
    get_test_devices,
    get_tolerance,
    infiniopOperatorDescriptor_t,
    profile_operation,
    test_operator,
)

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES = [
    # alpha, beta, a_shape, b_shape, c_shape, a_stride, b_stride, c_stride
    (1.0, 0.0, (1, 2048), (2048, 2048), (1, 2048), None, None, None),
    (1.0, 0.0, (2, 4, 2048), (2, 2048, 2048), (2, 4, 2048), None, None, None),
    (1.0, 0.0, (1, 2048), (2048, 2048), (1, 2048), (4096, 1), (4096, 1), (4096, 1)),
    (1.0, 1.0, (6, 2048), (2048, 2560), (6, 2560), (2048, 1), (1, 2048), (2560, 1)),
    (1.0 / 8.0, 0.0, (4, 8 * 6, 64), (4, 64, 6), (4, 8 * 6, 6), None, None, None),
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 0, "rtol": 1e-2},
    InfiniDtype.F32: {"atol": 0, "rtol": 1e-3},
    InfiniDtype.BF16: {"atol": 0, "rtol": 5e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


# PyTorch implementation for matrix multiplication
def gemm(d, _c, beta, _a, _b, alpha):
    # Match the kernel's FP32 accumulation instead of platform-specific CPU GEMM.
    if _a.device.type == "cpu" and _a.dtype in (torch.float16, torch.bfloat16):
        if _c.ndim == 2:
            result = torch.addmm(
                _c.float(), _a.float(), _b.float(), beta=beta, alpha=alpha
            )
        elif _c.ndim == 3:
            result = torch.baddbmm(
                _c.float(), _a.float(), _b.float(), beta=beta, alpha=alpha
            )
        else:
            raise ValueError(f"Unsupported tensor rank for GEMM: {_c.ndim}")
        d.copy_(result)
        return

    try:
        if _c.ndim == 2:
            torch.addmm(_c, _a, _b, beta=beta, alpha=alpha, out=d)
        elif _c.ndim == 3:
            torch.baddbmm(_c, _a, _b, beta=beta, alpha=alpha, out=d)
        else:
            raise
    except Exception:
        torch.matmul(_a, _b, out=d)
        d.mul_(alpha).add_(_c, alpha=beta)


# The argument list should be (lib, handle, torch_device, <param list>, dtype)
# The <param list> should keep the same order as the one specified in _TEST_CASES
def test(
    handle,
    device,
    alpha,
    beta,
    a_shape,
    b_shape,
    c_shape,
    a_stride=None,
    b_stride=None,
    c_stride=None,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing Gemm on {InfiniDeviceNames[device]} with alpha:{alpha}, beta:{beta},"
        f" a_shape:{a_shape}, b_shape:{b_shape}, c_shape:{c_shape},"
        f" a_stride:{a_stride}, b_stride:{b_stride}, c_stride:{c_stride}, dtype:{InfiniDtypeNames[dtype]}"
    )

    # Initialize tensors
    a = TestTensor(a_shape, a_stride, dtype, device)
    b = TestTensor(b_shape, b_stride, dtype, device)
    c = TestTensor(c_shape, c_stride, dtype, device, mode="ones")
    ans = TestTensor(c_shape, c_stride, dtype, device, mode="zeros")

    # Compute the PyTorch reference result
    def torch_gemm():
        gemm(
            ans.torch_tensor(),
            c.torch_tensor(),
            beta,
            a.torch_tensor(),
            b.torch_tensor(),
            alpha,
        )

    torch_gemm()

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateGemmDescriptor(
            handle,
            ctypes.byref(descriptor),
            c.descriptor,
            a.descriptor,
            b.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [a, b, c]:
        tensor.destroy_desc()

    # Get workspace size and create workspace
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetGemmWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    # Execute infiniop gemm operator
    def lib_gemm():
        check_error(
            LIBINFINIOP.infiniopGemm(
                descriptor,
                workspace.data(),
                workspace_size.value,
                c.data(),
                a.data(),
                b.data(),
                alpha,
                beta,
                None,
            )
        )

    lib_gemm()

    # Validate results
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)

    if DEBUG:
        debug(c.actual_tensor(), ans.torch_tensor(), atol=atol, rtol=rtol)

    assert torch.allclose(c.actual_tensor(), ans.torch_tensor(), atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch_gemm(), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_gemm(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyGemmDescriptor(descriptor))


# ==============================================================================
#  Main Execution
# ==============================================================================
if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
