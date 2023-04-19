module VehicleSim

using ColorTypes
using Dates
using GeometryBasics
using MeshCat
using MeshCatMechanisms
using Random
using Rotations
using RigidBodyDynamics
using Infiltrator
using LinearAlgebra
using SparseArrays
using Suppressor
using Sockets
using Serialization
using StaticArrays
using Zygote

include("view_car.jl")
include("objects.jl")
include("sim.jl")
include("control.jl")
include("sink.jl")
include("measurements.jl")
include("client.jl")
include("map.jl")
include("localization.jl")

export server, shutdown!, keyboard_client, test_client

end
