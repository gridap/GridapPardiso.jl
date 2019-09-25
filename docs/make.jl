using Documenter, GridapPardiso

makedocs(;
    modules=[GridapPardiso],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/gridap/GridapPardiso.jl/blob/{commit}{path}#L{line}",
    sitename="GridapPardiso.jl",
    authors="Víctor Sande Veiga, Large Scale Scientific Computing",
    assets=String[],
)

deploydocs(;
    repo="github.com/gridap/GridapPardiso.jl",
)
