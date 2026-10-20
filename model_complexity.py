"""Algorithmic forward FLOPs: MAC=2; scalar transcendental=1.
CUDA scan counts the mathematical recurrence, not hardware parallel-scan work.
Comparisons, indexing, memory transfers and integer arithmetic are not FLOPs.
"""
import json
import math
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode
from torch.utils._python_dispatch import TorchDispatchMode


class ArithmeticAudit(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.extra = Counter()
        self.unknown = Counter()
        self.excluded = Counter()
        self.cuda_calls = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = func._schema.name.split('::')[-1]
        t = out if isinstance(out, torch.Tensor) else out[0] if isinstance(out, (list, tuple)) and out else None
        n = t.numel() if isinstance(t, torch.Tensor) else 0
        x = args[0] if args and isinstance(args[0], torch.Tensor) else None
        floating = x is not None and (x.is_floating_point() or x.is_complex())
        cost = None
        if name in {'mm', 'bmm', 'matmul', 'addmm', 'baddbmm', 'convolution', '_convolution'}:
            # The base counter supplies MACs; add bias arithmetic separately.
            cost = n if name in {'addmm', 'baddbmm'} or (name in {'convolution', '_convolution'} and args[2] is not None) else 0
        elif name in {'add','add_','sub','sub_','mul','mul_','div','div_','neg','abs','exp','log','sqrt','rsqrt','reciprocal','pow'}:
            cost = n if floating else 0
        elif name in {'sigmoid','sigmoid_'}:
            cost = 4*n  # negate, exp, add, divide
        elif name in {'silu','silu_'}:
            cost = 5*n
        elif name == 'softplus':
            cost = 4*n  # beta*x, exp, log1p, divide by beta
        elif name == 'gelu':
            approximate = (kwargs or {}).get('approximate', 'none')
            cost = (9 if approximate == 'tanh' else 5)*n
        elif name in {'relu','relu_','leaky_relu','leaky_relu_','clamp','clamp_min','clamp_max'}:
            cost = n if name.startswith('leaky_relu') else 0
        elif name in {'sum','mean'}:
            cost = max(0, x.numel()-n) + (n if name == 'mean' else 0)
        elif name == 'linalg_vector_norm':
            if len(args) > 1 and args[1] != 2:
                self.unknown[str(func)] += 1
                return out
            cost = 2*x.numel()  # squares, sum and sqrt
        elif name in {'native_layer_norm','native_group_norm'}:
            cost = 8*x.numel()  # mean,var,epsilon,rsqrt,normalize,affine
        elif name in {'native_batch_norm','_native_batch_norm_legit_no_training','_native_batch_norm_legit','cudnn_batch_norm'}:
            cost = 4*x.numel() + 2*x.shape[1]  # inference normalization and affine
        elif name.startswith('avg_pool'):
            kernel=args[1]
            kernel=kernel if isinstance(kernel,(tuple,list)) else [kernel]
            cost=n*math.prod(kernel)  # kernel sum plus division; includes padded positions
        elif name.startswith('_adaptive_avg_pool') or name.startswith('adaptive_avg_pool'):
            # Sum/divide over each potentially overlapping adaptive bin.
            dims=len(t.shape)-2
            bins=1
            for a,b in zip(x.shape[-dims:],t.shape[-dims:]):
                bins*=sum(math.ceil((i+1)*a/b)-math.floor(i*a/b) for i in range(b))
            cost=x.shape[0]*x.shape[1]*bins
        elif (not floating or name.startswith(('max_pool','adaptive_max_pool','_adaptive_max_pool')) or
              name in {'view','_unsafe_view','reshape','permute','transpose','t','slice','select','expand','as_strided',
                       'unsqueeze','squeeze','detach','clone','copy_','contiguous','_to_copy','to','empty','empty_like',
                       'zeros','zeros_like','ones','ones_like','new_zeros','new_empty','fill_','zero_','cat','stack',
                       'split','split_with_sizes','unbind','chunk','index','index_select','index_put_',
                       'gather','scatter_','scatter','flip','roll','constant_pad_nd','alias','lift_fresh',
                       'ge','gt','le','lt','eq','ne','max','min','amax','amin','topk','sort','where'}):
            self.excluded[str(func)] += 1
            cost=0
        if cost is None:
            self.unknown[str(func)] += 1
        else:
            self.extra[str(func)] += int(cost)
        return out

    @contextmanager
    def cuda_kernels(self):
        patches=[]
        # Patch only the low-level CUDA entry points during this profiling pass.
        # Python-side projections/activations remain counted by the dispatcher.
        for module_name, attr, kind in [('selective_scan_cuda','fwd','scan'),
                                         ('causal_conv1d_cuda','causal_conv1d_fwd','causal')]:
            module=sys.modules.get(module_name)
            if module is None:
                continue
            original=getattr(module,attr)
            def wrapped(*args, _original=original, _kind=kind, **kwargs):
                result=_original(*args,**kwargs)
                if _kind == 'scan':
                    u, delta, A, B, C, D, z, bias, softplus = args[:9]
                    if A.is_complex():
                        raise ValueError('Complex selective scan is not covered by this FLOP convention')
                    m=u.numel(); state=A.shape[-1]
                    # exp(delta*A):2N; delta*B*u:2N; state recurrence:2N;
                    # C*state sum:2N-1. Plus skip, delta bias/softplus and SiLU gate.
                    cost=m*(8*state-1)
                    cost+=m*(int(bias is not None)+4*int(bool(softplus)))
                    cost+=m*(2*int(D is not None)+6*int(z is not None))
                else:
                    x, weight, bias = args[:3]
                    activation=args[-1]
                    cost=x.numel()*(2*weight.shape[-1]+int(bias is not None)+5*int(bool(activation)))
                self.extra['cuda::'+_kind]+=int(cost)
                self.cuda_calls[_kind]+=1
                return result
            setattr(module,attr,wrapped)
            patches.append((module,attr,original))
        try:
            yield
        finally:
            for module,attr,original in reversed(patches):
                setattr(module,attr,original)


def report_model_complexity(model, input_shape, output_path=None):
    parameter = next(model.parameters())
    modes = {module: module.training for module in model.modules()}
    devices = [parameter.device.index] if parameter.is_cuda else []
    try:
        model.eval()
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            sample = torch.zeros(tuple(input_shape), device=parameter.device, dtype=parameter.dtype)
            audit = ArithmeticAudit()
            with audit.cuda_kernels(), audit, FlopCounterMode(display=False) as counter:
                prediction = model(sample)
            if parameter.is_cuda:
                torch.cuda.synchronize(parameter.device)
            del prediction, sample
    finally:
        for module, training in modes.items():
            module.training = training
    counts = counter.get_flop_counts().get('Global', {})
    result = {
        'input_shape_BCzxy': list(input_shape),
        'input_type': 'zeros; data-dependent routing may differ for real images',
        'parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'base_flops': counter.get_total_flops(),
        'additional_flops': sum(audit.extra.values()),
        'estimated_forward_flops': counter.get_total_flops() + sum(audit.extra.values()),
        'estimated_gflops': (counter.get_total_flops() + sum(audit.extra.values())) / 1e9,
        'flops_by_operator': {str(op): value for op, value in counts.items()},
        'additional_flops_by_operator': dict(audit.extra),
        'custom_cuda_calls': dict(audit.cuda_calls),
        'unaccounted_operators': dict(audit.unknown),
        'non_flop_operators': dict(audit.excluded),
        'scope': 'Algorithmic forward estimate: multiply-add=2; scalar exp/log/sqrt/divide=1. '
                 'Includes custom causal convolution, selective scan recurrence, activations, '
                 'normalization and pooling. Normalization uses analytical estimates. '
                 'Comparisons/topk, indexing, copies and integer operations are not floating arithmetic. '
                 'Excludes loss/backward/optimizer/NMS and hardware parallel scan overhead. '
                 'Unaccounted operators must be empty before reporting as covered.',

    }
    print(f"[model] Input (B,C,z,x,y)={tuple(input_shape)}")
    print(f"[model] Params={result['parameters']:,} ({result['parameters']/1e6:.6f} M); "
          f"trainable={result['trainable_parameters']:,}")
    print(f"[model] Estimated forward GFLOPs={result['estimated_gflops']:.4f} (multiply-add=2)")
    print(f"[model] Custom CUDA calls={dict(audit.cuda_calls)}; unaccounted operators={dict(audit.unknown)}")
    print('[model] ' + result['scope'])
    if output_path is not None:
        Path(output_path).write_text(json.dumps(result, indent=2) + '\n')
    return result
