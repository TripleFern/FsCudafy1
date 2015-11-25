module FsCudafy1.AddLoopCPU

open System
open Cudafy
open Cudafy.Host
open Cudafy.Translator

open FSharp.Collections.ParallelSeq


let N = 10

let execute() =
    let a = [| for i in 0 .. N-1 -> -i |]
    let b = [| for i in 0 .. N-1 ->  i * i |]
    let c = (a, b) ||> Array.map2(fun x y -> x + y) //Should I use PSeq here, order of c would be random!
    (a, b, c) |||> Seq.zip3 |> Seq.iter (fun (x, y, z) -> printfn "%i + %i = %i" x y z)
    ()