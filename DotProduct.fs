namespace FsCudafy1

open System
open Cudafy
open Cudafy.Host
open Cudafy.Translator
open System.Diagnostics
open System.Threading.Tasks
open FSharp.Collections.ParallelSeq

type DotProductGPU()=
        static let imin =
            fun (a: float32) (b: float32) -> 
                if a < b then int a
                else int b
        static let sum_squares =
            fun (x: float32) ->
                x * (x+1.0f) * (2.0f*x+1.0f) / 6.0f
        static let sum_squares_double =
            fun x->  x * (x+1.0) * (2.0*x+1.0) / 6.0
        static let N = 33 * 1024
        [<Literal>]
        static let threadsPerBlock = 256
        static let blocksPerGrid = imin 32.0f (float32 ((N+threadsPerBlock-1) / threadsPerBlock))
        
        [<Cudafy>]
        static member Dot(thread: GThread, a: float32 array, b: float32 array, c: float32 array, n ) =
            let cache = thread.AllocateShared<float32>("cache", threadsPerBlock)

            let mutable tid = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;
            let cacheIndex = thread.threadIdx.x;

            let mutable temp = 0.0f
            while tid < n do
                temp <- temp + a.[tid] * b.[tid]
                tid <- tid + thread.blockDim.x * thread.gridDim.x
            
            cache.[cacheIndex] <- temp
            thread.SyncThreads()

            // for reductions, threadsPerBlock must be a power of 2
            // because of the following code
            let mutable i = thread.blockDim.x / 2
            while i <> 0 do
                if cacheIndex < i then
                    cache.[cacheIndex] <- cache.[cacheIndex] + cache.[cacheIndex + i]
                    thread.SyncThreads()
                i <- i / 2
            if cacheIndex = 0 then
                c.[thread.blockIdx.x] <- cache.[0]
            ()
        
        static member Execute() =
            let km: CudafyModule = CudafyTranslator.Cudafy()
            let gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId)
            gpu.LoadModule(km)
            //CPU
            let a: float32 array = Array.zeroCreate N
            let b: float32 array = Array.zeroCreate N
            let partial_c: float32 array = Array.zeroCreate blocksPerGrid
            //GPU
            let dev_a: float32 array = gpu.Allocate<float32>(N)
            let dev_b: float32 array = gpu.Allocate<float32>(N)
            let dev_partial_c: float32 array = gpu.Allocate<float32>(blocksPerGrid)

            let dev_test: float32 array = gpu.Allocate<float32>(blocksPerGrid * blocksPerGrid)
            Parallel.For(0, N, fun i -> a.[i] <- float32 i
                                        b.[i] <- float32 i * 2.0f) |>ignore

            gpu.CopyToDevice(a, dev_a)
            gpu.CopyToDevice(b, dev_b)
            gpu.Launch(dim3 blocksPerGrid, dim3 threadsPerBlock, "Dot", dev_a, dev_b, dev_partial_c, N)
            gpu.CopyFromDevice(dev_partial_c, partial_c)
            
            let c = partial_c |> Array.fold (fun s i -> s + double i) 0.0

            printfn "Does GPU value %f = %f?" c  (2.0 * sum_squares_double(float (N - 1)))

            gpu.FreeAll()