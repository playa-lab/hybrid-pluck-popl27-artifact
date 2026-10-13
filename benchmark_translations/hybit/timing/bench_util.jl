macro timed_infer(ex)
    quote
        Base.cumulative_compile_timing(true)
        local _c0 = Base.cumulative_compile_time_ns()
        local _g0 = Base.gc_num()
        local _t0 = time_ns()
        local _val = $(esc(ex))
        local _t1 = time_ns()
        local _c1 = Base.cumulative_compile_time_ns()
        local _gd = Base.GC_Diff(Base.gc_num(), _g0)
        local _wall    = (_t1 - _t0) / 1e9
        local _compile = (_c1[1] - _c0[1]) / 1e9   # [1] = total compile incl. recompile
        (value   = _val,
         time    = _wall - _compile,               # inference-only (drop-in for @timed .time)
         wall    = _wall,
         compile = _compile,
         gctime  = _gd.total_time / 1e9)
    end
end
