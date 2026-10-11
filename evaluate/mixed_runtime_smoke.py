"""Small deployed per-group dispatch check against independent chunk products."""


def mixed_smoke_layout():
    # Exact binary fractions exercise both equal-to-cut boundaries and every rung.
    return dict(amplitudes=[[0.0,.25,.5,.75],[1.0,.5,.25,1.0]],
                levels=[32,79,128],cuts=[.25,.75],
                rungs=[[0,0,1,1],[2,1,0,2]])


def check_mixed_runtime(recipe, device, dispatch):
    import torch
    from scmp_kernels import sc_matmul
    from scmp_kernels.mp import MPConfig
    from models.sc_integration import sc_linear

    spec = mixed_smoke_layout()
    amplitudes = torch.tensor(spec['amplitudes'],device=device,dtype=torch.float32)
    x = amplitudes.repeat_interleave(recipe.chunk,dim=1).contiguous()
    generator = torch.Generator(device=device).manual_seed(1701)
    linear = torch.nn.Linear(x.shape[1],8,device=device)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(linear.weight.shape,device=device,generator=generator)*.05)
        linear.bias.copy_(torch.randn(linear.bias.shape,device=device,generator=generator)*.01)
    linear._sc_smooth_scales = torch.linspace(.5,1.5,x.shape[1],device=device)
    expected_rungs = torch.tensor(spec['rungs'],device=device,dtype=torch.int32)
    old_config,old_modules = sc_linear._MP_CONFIG,sc_linear._MP_PER_MODULE
    sc_linear._MP_CONFIG = MPConfig(stoc_len_levels=spec['levels'],level_fractions=[1.0,0.0,0.0])
    sc_linear._MP_PER_MODULE = {('qkv',0):dict(fractions=None,invert=False,group_thresholds=spec['cuts'])}
    try:
        # Capture what the deployment adapter actually sends to the fused kernel.
        native = sc_linear.sc_matmul
        captured = []
        def traced(*args,**kwargs):
            captured.append(kwargs['rung_table'].clone())
            return native(*args,**kwargs)
        sc_linear.sc_matmul = traced
        try:
            with torch.no_grad():
                actual = dispatch(x,linear.weight,linear,x.shape,recipe.precision,
                                  torch.float32,op='qkv',block_idx=0)
        finally:
            sc_linear.sc_matmul = native
        if len(captured)!=1 or not torch.equal(captured[0],expected_rungs):
            raise RuntimeError('Mixed dispatch or equal-to-cut boundary mismatch')
        expected = torch.zeros_like(actual)
        with torch.no_grad():
            for chunk in range(amplitudes.shape[1]):
                sl = slice(chunk*recipe.chunk,(chunk+1)*recipe.chunk)
                for rung,length in enumerate(spec['levels']):
                    rows = (expected_rungs[:,chunk]==rung).nonzero(as_tuple=True)[0]
                    if rows.numel():
                        expected[rows] += sc_matmul(x[rows,sl].contiguous(),linear.weight[:,sl].contiguous(),
                            granularity='per_row',mode='bipolar',sc_prec=recipe.precision,stoc_len=length,
                            halve_bipolar_stoc_len=recipe.halve,smooth_scales=linear._sc_smooth_scales[sl].contiguous())
            expected += linear.bias
        if not torch.allclose(actual,expected,rtol=2e-5,atol=2e-5):
            raise RuntimeError('Fused mixed result differs from independent chunk products')
        lengths = torch.tensor(spec['levels'],device=device)
        return dict(levels=spec['levels'],rungs=spec['rungs'],
                    avg_cycles=float(lengths[expected_rungs.long()].float().mean()),
                    max_abs_error=float((actual-expected).abs().max()),
                    boundary_identity=True,chunk_product_identity=True)
    finally:
        sc_linear._MP_CONFIG,sc_linear._MP_PER_MODULE = old_config,old_modules
