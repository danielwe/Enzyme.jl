module RecursiveMaps

using EnzymeCore: EnzymeCore, isvectortype, isscalartype
using EnzymeCore.EnzymeRules: EnzymeRules, VectorSpace
using ..Compiler: guaranteed_const, guaranteed_const_nongen, guaranteed_nonactive,
    guaranteed_nonactive_nongen
using LinearAlgebra

### Config type for setting inactive/nonactive options
"""
    config = InactiveConfig(
        extra=(T -> false); copy_if_inactive=Val(false), runtime_inactive=Val(false)
    )
    config = InactiveConfig{copy_if_inactive::Bool,runtime_inactive::Bool}(extra)
    newconfig = InactiveConfig(config::InactiveConfig, extra)

Config type for specifying which parts of objects should be skipped by `recursive_map{!}`.

At a minimum, parts that Enzyme always considers inactive are skipped. An inactive type is a
type for which Enzyme can prove that a differentiable value can never be reached from any
instance of the type.

The optional argument `extra` takes a function defining additional types that should be
skipped regardless of their nominal activity. `extra` should be a plain function
or callable of a singleton type, not a closure or otherwise stateful callable; this is to
ensure that an `InactiveConfig` instance is fully specified by its type.

The parameter `copy_if_inactive` specifies whether `recursive_map{!}` should share (if
`Val(false)`, the default) or deep-copy (if `Val(true)`) inactive/skipped parts from inputs
to outputs.

The parameter `runtime_inactive` specifies whether `recursive_map{!}` should respect runtime
semantics when determining if a type is guaranteed inactive. If `Val(false)`, guaranteed
inactivity is determined once during compilation of the internal generated function
`active_reg_nothrow`, and won't be invalidated by subsequent changes to the
`EnzymeRules.inactive_type` method table. If `Val(true)`, the generated function is not used
and changes to `EnzymeRules.inactive_type` are picked up through invalidation as usual.

Using `runtime_inactive = Val(false)` may be preferred in interactive sessions, but
performance may sometimes suffer if the activity states of all types cannot be resolved at
compile time, and in some cases this mode has been observed to break gradient compilation
when `recursive_map{!}` is used inside custom rules. Hence `runtime_inactive = Val(true)` is
recommended for non-interactive usage and is the default.

The updating constructor `InactiveConfig(config::InactiveConfig, extra)` returns a new
config that extends `config` with an additional `extra` function.
"""
struct InactiveConfig{copy_if_inactive, runtime_inactive, E}
    extra::E
    function InactiveConfig{C, R}(extra::E) where {C, R, E}
        @assert Base.issingletontype(E)
        return new{C::Bool, R::Bool, E}(extra)
    end
end

function InactiveConfig(
        extra::E = (_ -> (@nospecialize; false));
        copy_if_inactive::Val{C} = Val(false), runtime_inactive::Val{R} = Val(false),
    ) where {E, C, R}
    return InactiveConfig{C, R}(extra)
end

function InactiveConfig(config::InactiveConfig{C, R}, extra::E) where {C, R, E}
    @inline combinedextra(::Type{T}) where {T} = (config.extra(T) || extra(T))
    return InactiveConfig{C, R}(combinedextra)
end

function isinactivetype(::Type{T}, config::InactiveConfig{C, false}) where {T, C}
    return guaranteed_const(T) || config.extra(T) # call guaranteed_const first, as this is a constant at runtime
end
function isinactivetype(::Type{T}, config::InactiveConfig{C, true}) where {T, C}
    return config.extra(T) || guaranteed_const_nongen(T, nothing) # call config.extra first, as guaranteed_const_nongen may incur runtime dispatch
end

function isnonactivetype(::Type{T}, config::InactiveConfig{C, false}) where {T, C}
    return guaranteed_nonactive(T) || config.extra(T) # call guaranteed_const first, as this is a constant at runtime
end
function isnonactivetype(::Type{T}, config::InactiveConfig{C, true}) where {T, C}
    return config.extra(T) || guaranteed_nonactive_nongen(T, nothing) # call config.extra first, as guaranteed_nonactive_nongen may incur runtime dispatch
end

### traits defining active leaf types for recursive_map
@inline EnzymeCore.isvectortype(::Type{T}) where {T} = isscalartype(T)
@inline function EnzymeCore.isvectortype(::Type{<:DenseArray{U}}) where {U}
    return isbitstype(U) && isscalartype(U)
end

@inline EnzymeCore.isscalartype(::Type) = false
@inline EnzymeCore.isscalartype(::Type{T}) where {T <: AbstractFloat} = isconcretetype(T)
@inline function EnzymeCore.isscalartype(::Type{Complex{T}}) where {T <: AbstractFloat}
    return isconcretetype(T)
end

### recursive_map: walk arbitrary objects and map a function over scalar and vector leaves
"""
    ys = recursive_map(
        [seen::Union{Nothing,IdDict},]
        f,
        ::Val{Nout}
        xs::NTuple{Nin,T},
        config::InactiveConfig=InactiveConfig(),
    )::T
    newys = recursive_map(
        [seen::Union{Nothing,IdDict},]
        f,
        ys::NTuple{Nout,T},
        xs::NTuple{Nin,T},
        config::InactiveConfig=InactiveConfig(),
    )::T

Recurse through `Nin` objects `xs = (x1::T, x2::T, ..., xNin::T)` of the same type, mapping
the function `f` over every differentiable value encountered and building `Nout` new objects
`(y1::T, ...)` from the resulting values `(y1_i, ...) = f(x1_i, ..., xNin_i)`. Only
`Nout == 1` and `Nout == 2` are supported.

The trait `EnzymeCore.isvectortype`(@ref) determines which values are considered
leaf nodes at which to terminate recursion invoke `f`. See the docstring for
[`EnzymeCore.isvectortype`](@ref) and the related [`EnzymeCore.isscalartype`](@ref) for more
information.

A tuple of existing objects `ys = (y1::T, ...)` can be passed, in which case the `ys` are
updated "partially-in-place": any parts of the `ys` that are mutable or non-differentiable
are reused in the returned object tuple `newys`, while immutable differentiable parts are
handled out-of-place as if the `ys` were not passed. If `T` itself is a mutable type, the
`ys` are modified in-place and returned, such that `newys === ys`.

The recursion and mapping operate on the structure of `T` as defined by struct fields and
plain array elements, not on the values provided through iteration or array interfaces. For
example, given a structured matrix wrapper or sparse array type, this function recurses into
the struct type and operates on the plain arrays held within, rather than operating on the
array that the type notionally represents.

# Arguments

* `seen::Union{IdDict,Nothing}` (optional): Dictionary for tracking object identity as
  needed to construct `y` such that its internal graph of object references is identical to
  that of the `xs`, including cycles (i.e., recursive substructures) and multiple paths to
  the same objects. If not provided, an `IdDict` will be allocated internally if required.

  If `nothing` is provided, object identity is tracking is turned off. In this case, objects
  with multiple references are duplicated such that the `ys`s object reference graph becomes
  a tree, but cycles will result in infinite recursion and stack overflow.

* `f`: Function mapping leaf nodes within the `xs` to the corresponding leaf nodes in the
  `ys`, that is, `(y1_i, ...) = f(x1_i::U, ..., xNin_i::U)::NTuple{Nout,U}`. The function
  `f` must be applicable to the type of every leaf node, and must return a tuple of values
  of the same type as its arguments.

  When an existing object tuple `ys` is passed and contains leaf nodes of a non-isbits
  non-scalar type `U`, `f` should also have a partially-in-place method
  `(newy1_i, ...) === f(y1_i::U, ..., yNout_i::U, x1_i::U, ..., xNin_i::U)::NTuple{Nout,U}`
  that modifies and reuses any mutable parts of the `yj_i`; in particular, if `U` is a
  mutable type, this method should return `newyj_i === yj_i`.

  If a non-isbits leaf type `U` must always be handled using the out-of-place signature,
  define the method `EnzymeCore.isscalartype(::Type{U}) = true`.

  See [`EnzymeCore.isvectortype`](@ref) and [`EnzymeCore.isscalartype`](@ref) for more
  details about leaf types and scalar types.

* `::Val{Nout}` or `ys::NTuple{Nout,T}`: For out-of-place operation, pass `Val(Nout)` where
  `Nout in (1, 2)` matches the length of the tuple returned by `f`. For partially-in-place
  operation, pass the existing tuple `ys::NTuple{Nout,T}` containing the values to be
  modified.

* `xs::NTuple{N,T}`: Tuple of `N` objects of the same type `T`.

  The first object `x1 = first(xs)` is the reference for graph structure and
  non-differentiable values when constructing the returned object. In particular:
  * When `ys` is not passed, the returned `ys` take any non-differentiable parts from `x1`.
  * When `ys` is passed, its non-differentiable parts are kept unchanged, unless they are
    uninitialized, in which case they are taken from `x1`.
  * The graph of object references in `x1` is the one which is reproduced in the returned
    object. For each instance of multiple paths and cycles within `x1`, the same structure
    must be present in the other objects `x2, ..., xN`, otherwise the corresponding values
    in the `ys` would not be uniquely defined. However, `x2, ..., xN` may contain additional
    converging paths or cycles that are not present in `x1`; these do not affect the `ys`.
  * If any values within `x1` are not initialized (that is, struct fields are undefined or
    array elements are unassigned), they are left uninitialized in the returned object. If
    any such values are mutable and `ys` is passed, the corresponding value in `y` must not
    already be initialized (initialized values cannot be nulled). Conversely, for every
    value in `x1` that is initialized, the corresponding values in `x2, ..., xN` must also
    be initialized, such that the corresponding values of the `ys` can be computed (however,
    `x2, ..., xN` may have initialized values where `x1` has uninitialized values).

* `config::InactiveConfig` (optional): Config object detailing how to deal with
  non-differentiable (inactive) parts. The config specifies whether non-differentiable parts
  should be shared or deep-copied from `x1` to the `ys`, and whether any additional types
  should be skipped in addition to those Enzyme always considers inactive. See
  [`InactiveConfig`](@ref) for details.
"""
function recursive_map end

## type alias for unified handling of out-of-place and partially-in-place recursive_map
const YS{Nout, T} = Union{Val{Nout}, NTuple{Nout, T}}
@inline hasvalues(::Val{Nout}) where {Nout} = (Nout::Int; false)
@inline hasvalues(::NTuple) = true

## main entry point: set default arguments, allocate IdDict if needed, exit early if possible
function recursive_map(
        f::F, ys::YS{Nout, T}, xs::NTuple{Nin, T}, config::InactiveConfig = InactiveConfig()
    ) where {F, Nout, Nin, T}
    @assert (Nout == 1) || (Nout == 2)
    newys = if isinactivetype(T, config)
        recursive_map_inactive(nothing, ys, xs, config)
    elseif isvectortype(T) || isbitstype(T)
        recursive_map_inner(nothing, f, ys, xs, config)
    else
        recursive_map_inner(IdDict(), f, ys, xs, config)
    end
    return newys::NTuple{Nout, T}
end

## recursive methods
function recursive_map(
        seen::Union{Nothing, IdDict},
        f::F,
        ys::YS{Nout, T},
        xs::NTuple{Nin, T},
        config::InactiveConfig = InactiveConfig(),
    ) where {F, Nout, Nin, T}
    # determine whether to continue recursion, copy/share, or retrieve from cache
    @assert (Nout == 1) || (Nout == 2)
    newys = if isinactivetype(T, config)
        recursive_map_inactive(seen, ys, xs, config)
    elseif isbitstype(T)  # no object identity to to track in this branch
        recursive_map_inner(nothing, f, ys, xs, config)
    elseif hascache(seen, xs)
        getcached(seen, Val(Nout), xs)
    else
        recursive_map_inner(seen, f, ys, xs, config)
    end
    return newys::NTuple{Nout, T}
end

@inline function recursive_map_inner(
        seen, f::F, ys::YS{Nout, T}, xs::NTuple{Nin, T}, config
    ) where {F, Nout, Nin, T}
    # forward to appropriate handler for leaf vs. mutable vs. immutable type
    @assert !isabstracttype(T)
    @assert isconcretetype(T)
    newys = if isvectortype(T)
        recursive_map_leaf(seen, f, ys, xs)
    elseif ismutabletype(T)
        recursive_map_mutable(seen, f, ys, xs, config)
    else
        recursive_map_immutable(seen, f, ys, xs, config)
    end
    return newys::NTuple{Nout, T}
end

@inline function recursive_map_mutable(
        seen, f::F, ys::YS{Nout, T}, xs::NTuple{Nin, T}, config
    ) where {F, Nout, Nin, T}
    @assert ismutabletype(T)
    if !hasvalues(ys) && !(T <: DenseArray) && all(isbitstype, fieldtypes(T))
        # fast path for out-of-place handling when all fields are bitstypes, which rules
        # out undefined fields and circular references
        newys = recursive_map_new(seen, f, ys, xs, config)
        maybecache!(seen, newys, xs)
    else
        newys = if hasvalues(ys)
            ys
        else
            x1 = first(xs)
            ntuple(_ -> (@inline; _similar(x1)), Val(Nout))
        end
        maybecache!(seen, newys, xs)
        recursive_map_mutable_inner!(seen, f, newys, ys, xs, config)
    end
    return newys::NTuple{Nout, T}
end

@inline function recursive_map_mutable_inner!(
        seen, f::F, newys::NTuple{Nout, T}, ys::YS{Nout, T}, xs::NTuple{Nin, T}, config
    ) where {F, Nout, Nin, T <: DenseArray}
    if (Nout == 1) && isbitstype(eltype(T))
        newy = only(newys)
        if hasvalues(ys)
            y = only(ys)
            broadcast!(newy, y, xs...) do y_i, xs_i...
                only(recursive_map(nothing, f, (y_i,), xs_i, config))
            end
        else
            broadcast!(newy, xs...) do xs_i...
                only(recursive_map(nothing, f, Val(1), xs_i, config))
            end
        end
    else
        @inbounds for i in eachindex(newys..., xs...)
            recursive_map_item!(i, seen, f, newys, ys, xs, config)
        end
    end
    return nothing
end

@generated function recursive_map_mutable_inner!(
        seen, f::F, newys::NTuple{Nout, T}, ys::YS{Nout, T}, xs::NTuple{Nin, T}, config
    ) where {F, Nout, Nin, T}
    return quote
        @inline
        Base.Cartesian.@nexprs $(fieldcount(T)) i -> @inbounds begin
            recursive_map_item!(i, seen, f, newys, ys, xs, config)
        end
        return nothing
    end
end

@inline function recursive_map_immutable(
        seen, f::F, ys::YS{Nout, T}, xs::NTuple{Nin, T}, config
    ) where {F, Nout, Nin, T}
    @assert !ismutabletype(T)
    nf = fieldcount(T)
    if nf == 0  # nothing to do (also no known way to hit this branch)
        newys = recursive_map_inactive(seen, ys, xs, config)
    else
        newys = if isinitialized(first(xs), nf)  # fast path when all fields are defined
            check_allinitialized(Base.tail(xs), nf)
            recursive_map_new(seen, f, ys, xs, config)
        else
            recursive_map_immutable_inner(seen, f, ys, xs, config)
        end
        # maybecache! _should_ be a no-op here; call it anyway for consistency
        maybecache!(seen, newys, xs)
    end
    return newys::NTuple{Nout, T}
end

@generated function recursive_map_immutable_inner(
        seen, f::F, ys::YS{Nout, T}, xs::NTuple{Nin, T}, config
    ) where {F, Nout, Nin, T}
    nf = fieldcount(T)
    return quote
        @inline
        x1, xtail = first(xs), Base.tail(xs)
        fields = Base.@ntuple $Nout _ -> Vector{Any}(undef, $(nf - 1))
        Base.Cartesian.@nexprs $(nf - 1) i -> begin  # unrolled loop over struct fields
            @inbounds if isinitialized(x1, i)
                check_allinitialized(xtail, i)
                newys_i = recursive_map_item(i, seen, f, ys, xs, config)
                Base.Cartesian.@nexprs $Nout j -> (fields[j][i] = newys_i[j])
            else
                return new_structvs(T, fields, i - 1)
            end
        end
        @assert !isinitialized(x1, $nf)
        return new_structvs(T, fields, $(nf - 1))
    end
end

@generated function recursive_map_new(
        seen, f::F, ys::YS{Nout, T}, xs::NTuple{Nin, T}, config
    ) where {F, Nout, Nin, T}
    # direct construction of fully initialized non-cyclic structs
    nf = fieldcount(T)
    return quote
        @inline
        Base.Cartesian.@nexprs $nf i -> @inbounds begin
            newys_i = recursive_map_item(i, seen, f, ys, xs, config)
        end
        newys = Base.@ntuple $Nout j -> begin
            $(Expr(:splatnew, :T, :(Base.@ntuple $nf i -> newys_i[j])))
        end
        return newys::NTuple{Nout, T}
    end
end

Base.@propagate_inbounds function recursive_map_item!(
        i, seen, f::F, newys::NTuple{Nout, T}, ys::YS{Nout, T}, xs::NTuple{Nin, T}, config
    ) where {F, Nout, Nin, T}
    if isinitialized(first(xs), i)
        check_allinitialized(Base.tail(xs), i)
        newys_i = recursive_map_item(i, seen, f, ys, xs, config)
        setitems!(newys, i, newys_i)
    elseif hasvalues(ys)
        check_allinitialized(ys, i, false)
    end
    return nothing
end

Base.@propagate_inbounds function recursive_map_item(
        i, seen, f::F, ys::YS{Nout, T}, xs::NTuple{Nin, T}, config
    ) where {F, Nout, Nin, T}
    # recurse into the xs and apply recursive_map to items with index i
    xs_i = getitems(xs, i)
    newys_i = if hasvalues(ys) && isinitialized(first(ys), i)
        check_allinitialized(Base.tail(ys), i)
        ys_i = getitems(ys, i)
        recursive_map_barrier!!(seen, f, ys_i..., config, xs_i...)
    else
        recursive_map_barrier(seen, f, Val(Nout), config, xs_i...)
    end
    return newys_i
end

# function barriers such that abstractly typed items trigger minimal runtime dispatch
function recursive_map_barrier(
        seen, f::F, ::Val{Nout}, config::InactiveConfig, xs_i::Vararg{ST, Nin}
    ) where {F, Nout, Nin, ST}
    return recursive_map(seen, f, Val(Nout), xs_i, config)::NTuple{Nout, ST}
end

function recursive_map_barrier!!(
        seen, f::F, y_i::ST, config::InactiveConfig, xs_i::Vararg{ST, Nin}
    ) where {F, Nin, ST}
    return recursive_map(seen, f, (y_i,), xs_i, config)::NTuple{1, ST}
end

function recursive_map_barrier!!(  # may not show in coverage but is covered via accumulate_into! TODO: ensure coverage via VectorSpace once implemented
        seen, f::F, y1_i::ST, y2_i::ST, config::InactiveConfig, xs_i::Vararg{ST, Nin}
    ) where {F, Nin, ST}
    ys_i = (y1_i, y2_i)
    return recursive_map(seen, f, ys_i, xs_i, config)::NTuple{2, ST}
end

## recursion base case handlers
@inline function recursive_map_leaf(
        seen, f::F, ys::YS{Nout, T}, xs::NTuple{Nin, T}
    ) where {F, Nout, Nin, T}
    # apply the mapped function to leaf values
    if !hasvalues(ys) || isbitstype(T) || isscalartype(T)
        newys = f(xs...)::NTuple{Nout, T}
    else  # !isbitstype(T)
        newys = f(ys..., xs...)::NTuple{Nout, T}
        if ismutabletype(T)
            @assert newys === ys
        end
    end
    maybecache!(seen, newys, xs)
    return newys::NTuple{Nout, T}
end

@inline function recursive_map_inactive(
        _, ys::NTuple{Nout, T}, xs::NTuple{Nin, T}, ::InactiveConfig{copy_if_inactive}
    ) where {Nout, Nin, T, copy_if_inactive}
    return ys::NTuple{Nout, T}
end

@inline function recursive_map_inactive(
        seen, ::Val{Nout}, (x1,)::NTuple{Nin, T}, ::InactiveConfig{copy_if_inactive}
    ) where {Nout, Nin, T, copy_if_inactive}
    @inline
    y = if copy_if_inactive && !isbitstype(T)
        if isnothing(seen)
            deepcopy(x1)
        else
            Base.deepcopy_internal(x1, seen)
        end
    else
        x1
    end
    return ntuple(_ -> (@inline; y), Val(Nout))::NTuple{Nout, T}
end

### recursive_map!: fully in-place wrapper around recursive_map
"""
    recursive_map!(
        [seen::Union{Nothing,IdDict},]
        f!!,
        ys::NTuple{Nout,T},
        xs::NTuple{Nin,T},
        isinactivetype::InactiveConfig=InactiveConfig(),
    )::Nothing

Recurse through `Nin` objects `xs = (x1::T, x2::T, ..., xNin::T)` of the same type, mapping
the function `f!!` over every differentiable value encountered and updating `(y1::T, ...)`
in-place with the resulting values.

This is a simple wrapper that verifies that `T` is a type where all differentiable values
can be updated in-place, calls `recursive_map`, and verifies that the returned value is
indeed identically the same tuple `ys`. See [`recursive_map`](@ref) for details.
"""
function recursive_map! end

function recursive_map!(
        f!!::F,
        ys::NTuple{Nout, T},
        xs::NTuple{Nin, T},
        config::InactiveConfig = InactiveConfig(),
    ) where {F, Nout, Nin, T}
    check_nonactive(T, config)
    newys = recursive_map(f!!, ys, xs, config)
    @assert newys === ys
    return nothing
end

function recursive_map!(
        seen::Union{Nothing, IdDict},
        f!!::F,
        ys::NTuple{Nout, T},
        xs::NTuple{Nin, T},
        config::InactiveConfig = InactiveConfig(),
    ) where {F, Nout, Nin, T}
    check_nonactive(T, config)
    newys = recursive_map(seen, f!!, ys, xs, config)
    @assert newys === ys
    return nothing
end

### recursive_map helpers
@generated function new_structvs(
        ::Type{T}, fields::NTuple{N, Vector{Any}}, nfields_
    ) where {T, N}
    return quote
        @inline
        return Base.@ntuple $N j -> begin
            ccall(:jl_new_structv, Any, (Any, Ptr{Any}, UInt32), T, fields[j], nfields_)::T
        end
    end
end

@inline _similar(::T) where {T} = ccall(:jl_new_struct_uninit, Any, (Any,), T)::T
@inline _similar(x::T) where {T <: DenseArray} = similar(x)::T
Base.@propagate_inbounds isinitialized(x, i) = isdefined(x, i)
Base.@propagate_inbounds isinitialized(x::DenseArray, i) = isassigned(x, i)
Base.@propagate_inbounds getitem(x, i) = getfield(x, i)
Base.@propagate_inbounds getitem(x::DenseArray, i) = x[i]
Base.@propagate_inbounds setitem!(x, i, v) = setfield_force!(x, i, v)
Base.@propagate_inbounds setitem!(x::DenseArray, i, v) = (x[i] = v; nothing)

Base.@propagate_inbounds function setfield_force!(x::T, i, v) where {T}
    if Base.isconst(T, i)
        ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), x, i - 1, v)
    else
        setfield!(x, i, v)
    end
    return nothing
end

Base.@propagate_inbounds function getitems(
        (x1, xtail...)::Tuple{T, T, Vararg{T, N}}, i
    ) where {T, N}
    return (getitem(x1, i), getitems(xtail, i)...)
end

Base.@propagate_inbounds getitems((x1,)::Tuple{T}, i) where {T} = (getitem(x1, i),)

Base.@propagate_inbounds function setitems!(  # may not show in coverage but is covered via accumulate_into! TODO: ensure coverage via VectorSpace once implemented
        (x1, xtail...)::Tuple{T, T, Vararg{T, N}},
        i,
        (v1, vtail...)::Tuple{ST, ST, Vararg{ST, N}},
    ) where {T, ST, N}
    setitem!(x1, i, v1)
    setitems!(xtail, i, vtail)
    return nothing
end

Base.@propagate_inbounds function setitems!((x1,)::Tuple{T}, i, (v1,)::Tuple{ST}) where {T, ST}
    setitem!(x1, i, v1)
    return nothing
end

## cache (seen) helpers
@inline function iscachedtype(::Type{T}) where {T}
    # cache all mutable types and any non-isbits types that are also leaf types
    return ismutabletype(T) || ((!isbitstype(T)) && isvectortype(T))
end

@inline shouldcache(::IdDict, ::Type{T}) where {T} = iscachedtype(T)
@inline shouldcache(::Nothing, ::Type{T}) where {T} = false

@inline function maybecache!(
        seen, newys::NTuple{Nout, T}, (x1, xtail...)::NTuple{Nin, T}
    ) where {Nout, Nin, T}
    if shouldcache(seen, T)
        seen[x1] = if (Nout == 1) && (Nin == 1)
            only(newys)
        else  # may not show in coverage but is covered via accumulate_into! TODO: ensure coverage via VectorSpace once implemented
            (newys..., xtail...)
        end
    end
    return nothing
end

@inline function hascache(seen, (x1,)::NTuple{Nin, T}) where {Nin, T}
    return shouldcache(seen, T) ? haskey(seen, x1) : false
end

@inline function getcached(
        seen::IdDict, ::Val{Nout}, (x1, xtail...)::NTuple{Nin, T}
    ) where {Nout, Nin, T}
    newys = if (Nout == 1) && (Nin == 1)
        (seen[x1]::T,)
    else   # may not show in coverage but is covered via accumulate_into! TODO: ensure coverage via VectorSpace once implemented
        cache = seen[x1]::NTuple{(Nout + Nin - 1), T}
        cachedtail = cache[(Nout + 1):end]
        check_identical(cachedtail, xtail)  # check compatible structure
        cache[1:Nout]
    end
    return newys::NTuple{Nout, T}
end

## argument validation
Base.@propagate_inbounds function check_initialized(x, i, initialized = true)
    if isinitialized(x, i) != initialized
        throw_initialized()  # TODO: hit this when VectorSpace implemented
    end
    return nothing
end

Base.@propagate_inbounds function check_allinitialized(  # TODO: hit this when VectorSpace implemented
        (x1, xtail...)::Tuple{T, T, Vararg{T, N}}, i, initialized = true
    ) where {T, N}
    check_initialized(x1, i, initialized)
    check_allinitialized(xtail, i, initialized)
    return nothing
end

Base.@propagate_inbounds function check_allinitialized(
        (x1,)::Tuple{T}, i, initialized = true
    ) where {T}
    check_initialized(x1, i, initialized)
    return nothing
end

Base.@propagate_inbounds check_allinitialized(::Tuple{}, i, initialized = true) = nothing

@inline function check_identical(u, v)  # TODO: hit this when VectorSpace implemented
    if u !== v
        throw_identical()
    end
    return nothing
end

@inline function check_nonactive(::Type{T}, config) where {T}
    if !isnonactivetype(T, config)
        throw_nonactive()
    end
    return nothing
end

# TODO: hit all of these via check_* when VectorSpace implemented
@noinline function throw_nout()
    throw(ArgumentError("recursive_map(!) only supports mapping to 1 or 2 outputs"))
end

@noinline function throw_initialized()
    msg = "recursive_map(!) called on objects whose undefined fields/unassigned elements "
    msg *= "don't line up"
    throw(ArgumentError(msg))
end

@noinline function throw_identical()
    msg = "recursive_map(!) called on objects whose structure don't match"
    throw(ArgumentError(msg))
end

@noinline function throw_nonactive()
    msg = "recursive_map! called on objects containing immutable differentiable values"
    throw(ArgumentError(msg))
end

### EnzymeCore.make_zero(!) implementation
@inline function EnzymeCore.make_zero(prev::T, args::Vararg{Any, M}; kws...) where {T, M}
    config = make_zero_config(args...; kws...)
    new = if iszero(M) && isempty(kws) && !isinactivetype(T, config) && isvectortype(T)  # fallback
        # isinactivetype precedes over isvectortype for consistency with recursive handler
        convert(T, zero(prev))  # convert because zero(prev)::T may not hold when eltype(T) is abstract
    else
        only(recursive_map(_make_zero!!, Val(1), (prev,), config))::T
    end
    return new::T
end

@inline function EnzymeCore.make_zero!(val::T, args::Vararg{Any, M}; kws...) where {T, M}
    @assert !isscalartype(T)  # not appropriate for in-place handler
    if iszero(M) && isempty(kws) && !isinactivetype(T, make_zero!_config()) && isvectortype(T)  # fallback
        # isinactivetype precedes over isvectortype for consistency with recursive handler
        fill!(val, false)
    else
        _make_zero_inner!(val, args...; kws...)
    end
    return nothing
end

@inline function _make_zero_inner!(val, args::Vararg{Any, M}; kws...) where {M}
    return recursive_map!(_make_zero!!, (val,), (val,), make_zero!_config(args...; kws...))
end
@inline function _make_zero_inner!(val, seen::IdDict, args::Vararg{Any, M}; kws...) where {M}
    config = make_zero!_config(args...; kws...)
    return recursive_map!(seen, _make_zero!!, (val,), (val,), config)
end

# map make_zero(!) args/kws to config
@inline make_zero_config(C) = InactiveConfig(; copy_if_inactive = C)
@inline make_zero_config(C, R) = InactiveConfig(; copy_if_inactive = C, runtime_inactive = R)
@inline make_zero_config(; kws...) = InactiveConfig(; kws...)

@inline make_zero!_config(R) = InactiveConfig(; runtime_inactive = R)
@inline function make_zero!_config(; runtime_inactive = nothing)
    if isnothing(runtime_inactive)
        return InactiveConfig()
    else
        return InactiveConfig(; runtime_inactive)
    end
end

# the mapped function: assert leaf type and call back into single-arg make_zero(!)
function _make_zero!!(prev::T) where {T}
    @assert isvectortype(T)  # otherwise infinite loop
    return (EnzymeCore.make_zero(prev)::T,)
end

function _make_zero!!(val::T, _val::T) where {T}
    @assert !isscalartype(T)  # not appropriate for in-place handler
    @assert isvectortype(T)   # otherwise infinite loop
    @assert val === _val
    EnzymeCore.make_zero!(val)
    return (val::T,)
end

# alternative entry point for passing custom IdDict
@inline function EnzymeCore.make_zero(
        ::Type{T}, seen::IdDict, prev::T, args::Vararg{Any, M}; kws...
    ) where {T, M}
    config = make_zero_config(args...; kws...)
    news = recursive_map(seen, _make_zero!!, Val(1), (prev,), config)
    return only(news)::T
end

### VectorSpace implementation
## constructors
@inline EnzymeRules.VectorSpace(val; kws...) = VectorSpace(InactiveConfig(; kws...), val)

@inline function EnzymeRules.VectorSpace(v::VectorSpace{C}, newval::T) where {C, T}
    return VectorSpace(getconfig(v), newval)::VectorSpace{C, T}
end

@inline getconfig(v::VectorSpace) = v.config

## comparison
Base.:(==)(u::VectorSpace{C}, v::VectorSpace{C}) where {C} = (u.val == v.val)
Base.:(==)(::VectorSpace, ::VectorSpace) = false

Base.isequal(u::VectorSpace{C}, v::VectorSpace{C}) where {C} = isequal(u.val, v.val)
Base.isequal(::VectorSpace, ::VectorSpace) = false

Base.hash(v::VectorSpace, h::UInt) = hash(v.val, h)
Base.widen(::Type{VS}) where {VS <: VectorSpace} = VS  # apparently required

## similar
function Base.similar(v::VectorSpace{C, T}) where {C, T}
    function similar_or_zero(prev::U) where {U}
        @assert isvectortype(U)  # otherwise infinite loop
        new = if isscalartype(U)
            zero(prev)
        else
            similar(prev)
        end
        return (convert(U, new)::U,)
    end
    newval = first(recursive_map(similar_or_zero, Val(1), (v.val,), getconfig(v)))::T
    return VectorSpace(v, newval)
end

## unary arithmetic
Base.:(+)(v::VS) where {VS <: VectorSpace} = v::VS

function Base.:(-)(v::VectorSpace{C, T}) where {C, T}
    neg(x) = (oftype(x, -x),)
    newval = first(recursive_map(neg, Val(1), (v.val,), getconfig(v)))::T
    return VectorSpace(v, newval)
end

Base.zero(v::VS) where {VS <: VectorSpace} = VectorSpace(v, EnzymeCore.make_zero(v.val))::VS

## binary arithmetic
function Base.:(+)(v1::VectorSpace{C, T}, vrest::Vararg{VectorSpace{C}, N}) where {C, T, N}
    add(x1, xrest...) = (oftype(x1, +(x1, xrest...)),)
    xs = (v1.val, ntuple(i -> (@inline; vrest[i].val), Val(N))...)
    newval = first(recursive_map(add, Val(1), xs, getconfig(v1)))::T
    return VectorSpace(v1, newval)
end

function Base.:(-)(u::VectorSpace{C, T}, v::VectorSpace{C}) where {C, T}
    sub(x, y) = (oftype(x, x - y),)
    newval = first(recursive_map(sub, Val(1), (u.val, v.val), getconfig(u)))::T
    return VectorSpace(u, newval)
end

function Base.:(*)(a::Number, v::VectorSpace{C, T}) where {C, T}
    mul(x) = (oftype(x, a * x),)
    newval = first(recursive_map(mul, Val(1), (v.val,), getconfig(v)))::T
    return VectorSpace(v, newval)
end

@inline Base.:(*)(v::VectorSpace, a::Number) = a * v

function Base.:(/)(v::VectorSpace{C, T}, a::Number) where {C, T}
    _div(x) = (oftype(x, x / a),)
    newval = first(recursive_map(_div, Val(1), (v.val,), getconfig(v)))::T
    return VectorSpace(v, newval)
end

@inline Base.:(\)(a::Number, v::VectorSpace) = v / a

function LinearAlgebra.lmul!(a::Number, v::VS) where {VS <: VectorSpace}
    f!!(x) = (oftype(x, a * x),)
    f!!(x, _x) = (@assert x === _x; LinearAlgebra.lmul!(a, x); (x,))
    recursive_map!(f!!, (v.val,), (v.val,), getconfig(v))
    return v::VS
end

@inline LinearAlgebra.rmul!(v::VectorSpace, a::Number) = LinearAlgebra.lmul!(a, v)

function LinearAlgebra.rdiv!(v::VS, a::Number) where {VS <: VectorSpace}
    f!!(x) = (oftype(x, x / a),)
    f!!(x, _x) = (@assert x === _x; LinearAlgebra.rdiv!(x, a); (x,))
    recursive_map!(f!!, (v.val,), (v.val,), getconfig(v))
    return v::VS
end

@inline LinearAlgebra.ldiv!(a::Number, v::VectorSpace) = LinearAlgebra.rdiv!(v, a)

## norms and inner products
# recursive map isn't really designed for this, but it's a pretty benign hack
# the only issue is we don't know the correctly promoted float type upfront, so to avoid
# catastrophic performance loss we need to choose a type upfront
# Is there a better way to do this? Perhaps add a normtype parameter to VectorSpace?
function LinearAlgebra.norm(v::VectorSpace, p::Real = 2)
    acc = Float64[]  # XXX: hardcoded output type
    f!!(x) = (push!(acc, LinearAlgebra.norm(x, p)); (x,))
    f!!(x, _x) = (@assert x === _x; f!!(x))
    val = v.val
    recursive_map(f!!, (val,), (val,), getconfig(v))
    return LinearAlgebra.norm(acc, p)
end

function LinearAlgebra.dot(u::VectorSpace{C}, v::VectorSpace{C}) where {C}
    acc = Ref{Float64}(0.0)  # XXX: hardcoded output type
    f!!(x, y) = (acc[] += LinearAlgebra.dot(x, y); (x, y))
    f!!(x, y, _x, _y) = (@assert x === _x; @assert y === _y; f!!(x, y))
    uval, vval = u.val, v.val
    recursive_map(f!!, (uval, vval), (uval, vval), getconfig(u))
    return acc[]
end

## broadcasting
struct VSStyle <: Broadcast.BroadcastStyle end

Broadcast.BroadcastStyle(::Type{<:VectorSpace}) = VSStyle()
Broadcast.BroadcastStyle(s::VSStyle, ::Broadcast.AbstractArrayStyle{0}) = s

Base.similar(bc::Broadcast.Broadcasted{VSStyle}, ::Any) = _similar_to_first_VS(bc.args)

@inline function _similar_to_first_VS(args::Tuple{VS, Vararg{Any}}) where {VS <: VectorSpace}
    return similar(first(args))::VS
end
@inline _similar_to_first_VS(args::Tuple) = _similar_to_first_VS(Base.tail(args))
@inline _similar_to_first_VS(::Tuple{}) = error("no VectorSpace instance found")

Base.axes(::VectorSpace) = nothing
Broadcast.broadcastable(vs::VectorSpace) = vs
Broadcast.instantiate(bc::Broadcast.Broadcasted{VSStyle}) = bc

function Base.copy(bc::Broadcast.Broadcasted{VSStyle})
    (; f, args) = Broadcast.flatten(bc)
    return _copy(f, args, getvectors(args))
end

function Base.copyto!(out::VectorSpace, bc::Broadcast.Broadcasted{VSStyle})
    (; f, args) = Broadcast.flatten(bc)
    return _copyto!(f, args, out, getvectors(args))
end

function _copy(
        f::A, args, vectors::Tuple{VectorSpace{C, T}, Vararg{VectorSpace{C}, N}}
    ) where {A, C, T, N}
    function h(value::U, values::Vararg{Any, N}) where {U}
        return (broadcast(f, replacevectors(args, (value, values...))...)::U,)
    end
    vvalues = ntuple(i -> (@inline; vectors[i].val), Val(N + 1))
    v1 = first(vectors)
    newval = first(recursive_map(h, Val(1), vvalues, getconfig(v1)))::T
    return VectorSpace(v1, newval)
end

function _copyto!(
        f::A,
        args,
        out::VectorSpace{C, T},
        vectors::Tuple{Vararg{VectorSpace{C}, M}},
    ) where {A, C, T, M}
    function h!!(value::U, values::Vararg{Any, N}) where {U, N}
        return (broadcast(f, replacevectors(args, (value, values...))...)::U,)
    end
    function h!!(outvalue::U, values::Vararg{Any, M}) where {U}
        broadcast!(f, outvalue, replacevectors(args, values)...)
        return (outvalue::U,)
    end
    vvalues = ntuple(i -> (@inline; vectors[i].val), Val(M))
    recursive_map!(h!!, (out.val,), vvalues, getconfig(first(vectors)))
    return out::VectorSpace{C, T}
end

## broadcast helpers
@inline function getvectors(args::Tuple{VectorSpace, Vararg{Any}})
    return (first(args), getvectors(Base.tail(args))...)
end
@inline getvectors(args::Tuple{Any, Vararg{Any}}) = getvectors(Base.tail(args))
@inline getvectors(args::Tuple{VectorSpace}) = args
@inline getvectors(::Tuple{Any}) = ()
@inline getvectors(::Tuple{}) = ()

@inline function replacevectors(args::Tuple{Any, Vararg{Any}}, values::Tuple{Any, Vararg{Any}})
    return (first(args), replacevectors(Base.tail(args), values)...)
end
@inline function replacevectors(
        args::Tuple{VectorSpace, Vararg{Any}}, values::Tuple{Any, Vararg{Any}}
    )
    return (first(values), replacevectors(Base.tail(args), Base.tail(values))...)
end
@inline function replacevectors(
        args::Tuple{VectorSpace, Vararg{VectorSpace}}, values::Tuple{Any, Vararg{Any}}
    )
    @assert length(args) == length(values)
    return values
end
@inline function replacevectors(args::Tuple{Vararg{Any}}, ::Tuple{})
    @assert !any(arg -> (@inline; isa(arg, VectorSpace)), args)
    return args
end

end  # module RecursiveMaps
