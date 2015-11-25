module FsCudafy1.EnumGPU

open System
// You need to make refernce to System.Xml and System.Xml.Linq to open top level Cudafy.
open Cudafy
open Cudafy.Host

let print_GPU_Info (prop: GPGPUProperties) i =
    printfn "   --- General Information for device %i ---" i
    printfn "Name: %s" prop.Name
    printfn "Platform Name:  %s" prop.PlatformName
    printfn "Device Id:  %i" prop.DeviceId
    printfn "Compute capability:  %i.%i" prop.Capability.Major prop.Capability.Minor
    printfn "Clock rate: %i" prop.ClockRate
    printfn "Simulated: %b" prop.IsSimulated
    printfn ""
    printfn "   --- Memory Information for device %i ---" i
    printfn "Total global mem:  %u" prop.TotalMemory
    printfn "Total constant mem:  %u" prop.TotalConstantMemory
    printfn "Max mem pitch:  %i" prop.MemoryPitch
    printfn "Texture Alignment:  %i" prop.TextureAlignment
    printfn ""
    printfn "   --- MP Information for device %i ---" i
    printfn "Shared mem per mp: %i" prop.SharedMemoryPerBlock
    printfn "Registers per mp:  %i" prop.RegistersPerBlock
    printfn "Threads in warp:  %i" prop.WarpSize
    printfn "Max threads per block:  %i" prop.MaxThreadsPerBlock
    printfn "Max thread dimensions:  (%i, %i, %i)" prop.MaxThreadsSize.x prop.MaxThreadsSize.y prop.MaxThreadsSize.z
    printfn "Max grid dimensions:  (%i, %i, %i)" prop.MaxGridSize.x prop.MaxGridSize.y prop.MaxGridSize.z
    printfn ""
    ()

let execute() =
    CudafyHost.GetDeviceProperties(CudafyModes.Target, false)
    |> Seq.iteri(fun i p -> print_GPU_Info p i )