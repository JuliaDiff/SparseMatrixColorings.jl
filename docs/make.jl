using Documenter
using DocumenterInterLinks
using SparseMatrixColorings

links = InterLinks("ADTypes" => "https://sciml.github.io/ADTypes.jl/stable/")

# The README is reused as the documentation homepage.
# Julia's Markdown parser does not support raw HTML, so the all-contributors table
# has to be wrapped in a `@raw html` block, and the anchor of the "Contributors"
# heading has to be capitalized the way Documenter generates it.
const CONTRIBUTORS_START = "<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->"
const CONTRIBUTORS_END = "<!-- ALL-CONTRIBUTORS-LIST:END -->"

readme = read(joinpath(@__DIR__, "..", "README.md"), String)
index = replace(
    readme,
    "](#contributors)" => "](#Contributors)",
    CONTRIBUTORS_START => "```@raw html\n" * CONTRIBUTORS_START,
    CONTRIBUTORS_END => CONTRIBUTORS_END * "\n```",
)
write(joinpath(@__DIR__, "src", "index.md"), index)

makedocs(;
    modules=[SparseMatrixColorings],
    authors="Guillaume Dalle and Alexis Montoison",
    sitename="SparseMatrixColorings.jl",
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "tutorial.md",
        "api.md",
        "Developer Documentation" => ["dev.md", "vis.md"],
    ],
    plugins=[links],
)

deploydocs(;
    repo="github.com/JuliaDiff/SparseMatrixColorings.jl",
    push_preview=true,
    devbranch="main",
)
