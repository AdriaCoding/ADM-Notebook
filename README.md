# Algorithms for Data ~~Mining~~ Science project
For the first deliverable of the project we used Julia, and a notebook environment similar to Jupyter named Pluto. The notebook in question is called "MLP.jl", you can find it in the notebook-packages directory.

To reproduce our code, you might need to intall Julia along with the Pluto.jl package

To install Julia, you can go the [official website](https://julialang.org/downloads/) and follow the appropriate steps for your system. 

If you have added Julia to PATH, you can start a new session into the Julia REPL by just running "julia" (or else, run "/path/to/julia/bin/julia)" on your favorite shell.
```
$ julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.8.1 (2022-09-06)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia>
```
Feel free to play arround with the REPL to be more familiarized with the language. I encourage checking out the [offical documentation manual](https://docs.julialang.org/en/v1/manual/getting-started/), and this [cool tutorial](https://youtu.be/EkgCENBFrAY?si=DTJ3SP1Shm0wYKTk) by Miguel Raz.

Now you need to install PLuto.jl in your Julia environment. You can just type inside the REPL
```
julia> using Pkg; Pkg.add(Pluto)
```
Or equivalently, after pressing ``]``
```
(@v1.10) pkg> add Pluto
```

Now you can laung a Pluto session from localhost with
```
julia> using PLuto; Pluto.run()
```
and just by entering your local path to ``notebook-packages/MLP.jl`` you can edit, run and replicate our work.