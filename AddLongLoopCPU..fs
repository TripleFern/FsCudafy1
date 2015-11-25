module FsCudafy1.AddLongLoopCPU

open System
open System.Threading.Tasks
open System.Diagnostics

let N = 32 * 1024

let execute() =
    let a = [| for i in 0 .. N-1 -> i|]
    let b = [| for i in 0 .. N-1 -> 2 * i|]
    let c: int array = Array.zeroCreate N

    let stopWatch = new Stopwatch()
    stopWatch.Start()
    Parallel.For(0, N, (fun i -> (c.[i] <- a.[i] + b.[i]) )) |> ignore
    stopWatch.Stop()
    let timeSpan = stopWatch.Elapsed
    printfn "Add long loop on CPU: %f (ms)" timeSpan.TotalMilliseconds
    ()