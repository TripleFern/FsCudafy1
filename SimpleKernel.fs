module FsCudafy1.SimpleKernel

open System
open Cudafy.Host
open Cudafy.Translator

[<Cudafy.Cudafy>]
let computeKernel() =
    ()

let execute() =
    let km = CudafyTranslator.Cudafy()
    let gpu = CudafyHost.GetDevice(Cudafy.CudafyModes.Target, Cudafy.CudafyModes.DeviceId)
    gpu.LoadModule(km)
    gpu.Launch(Cudafy.dim3  1, Cudafy.dim3 1, "computeKernel")
    Console.WriteLine("Hello Cudafy !")
    gpu.UnloadModule()
    ()
