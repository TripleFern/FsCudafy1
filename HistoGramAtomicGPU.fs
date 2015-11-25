namespace FsCudafy1

open System
open Cudafy
open Cudafy.Host
open Cudafy.Atomics
open Cudafy.Translator
open System.Diagnostics
open System.Threading.Tasks
open FSharp.Collections.ParallelSeq
open MathNet.Numerics.Random

exception FsCudaException of string

type HistogramAtomicGPU() =
    static let rng = MersenneTwister()
    static let SIZE = 104857600 // 100 * 1024 * 1024 not allowed for [<Literal>]

    [<Cudafy>]
    static member HistoKernel(thread: GThread, buffer: byte array, size: int, histo: uint32 array) =
        let temp = thread.AllocateShared<uint32>("temp", 256)
        temp.[thread.threadIdx.x] <- 0u
        thread.SyncThreads()

        let mutable i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x
        let stride = thread.blockDim.x * thread.gridDim.x
        while i < size do
            let _ = thread.atomicAdd(& temp.[int buffer.[i]], 1u) // Using ref will make compiling fail.
            i <- i + stride
        thread.SyncThreads()
        thread.atomicAdd(& histo.[thread.threadIdx.x],temp.[thread.threadIdx.x]) |> ignore
        ()

    static member BigRandomBlock(size: int) = rng.NextBytes(size)

    static member Execute() =
        let km = CudafyTranslator.Cudafy()

        let gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId)
        match gpu with
        | :? CudaGPU -> if gpu.GetDeviceProperties().Capability < Version(1, 2) then
                            raise (FsCudaException("Compute capability 1.2 or higher required for atomics."))
        | _ -> ()

        gpu.LoadModule(km)

        let buffer = HistogramAtomicGPU.BigRandomBlock(SIZE)

        let mutable prop = Unchecked.defaultof<GPGPUProperties>
        try
            prop <- gpu.GetDeviceProperties(true)
        with
            | :? DllNotFoundException ->
                prop <- gpu.GetDeviceProperties(false)

        gpu.StartTimer()
        let dev_buffer = gpu.CopyToDevice(buffer)
        let dev_histo = gpu.Allocate<uint32>(256)
        gpu.Set(dev_histo)

        let blocks =
            if prop.MultiProcessorCount = 0 then 16
            else  prop.MultiProcessorCount
        printfn "Processors: %i" blocks

        let gridSize = blocks * 2
        let blockSize = 256
        gpu.Launch(dim3 gridSize, dim3 blockSize, "HistoKernel", dev_buffer, SIZE, dev_histo)

        let histo = Array.zeroCreate 256
        gpu.CopyFromDevice(dev_histo, histo)

        let elapsedTime = gpu.StopTimer()
        printfn "Time to generate: %f ms" elapsedTime

        let histoCount = histo |> Array.sum
        printfn "Histogram Sum: %i" histoCount

        for i in 0 .. SIZE-1 do
            histo.[int buffer.[i]] <- histo.[int buffer.[i]] - 1u

        Parallel.For(0, 256,
            (fun i ->
                if histo.[i] <>  0u then
                    printfn "Failure at %i" i )
            ) |> ignore

        gpu.FreeAll()