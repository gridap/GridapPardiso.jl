using Documenter, GridapPardiso

makedocs(;
    modules=[GridapPardiso],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/gridap/GridapPardiso.jl/blob/{commit}{path}#L{line}",
    sitename="GridapPardiso.jl",
    authors=["Francesc Verdugo <fverdugo@cimne.upc.edu>", "Víctor Sande <vsande@cimne.upc.edu>"],
    assets=String[],
)

deploydocs(;
    repo="github.com/gridap/GridapPardiso.jl",
)
