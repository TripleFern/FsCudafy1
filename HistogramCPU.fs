module FsCudafy1.HistogramCPU

open System
open System.Threading.Tasks
open System.Diagnostics
open MathNet.Numerics.Random
open FSharp.Control

let rng = MersenneTwister()
let SIZE = 104857600

let BigRandomBlock(size: int) = rng.NextBytes(size)

let buffer = BigRandomBlock(SIZE)

let stopWatch = new Stopwatch()

type 'a MailContent =
    | Add of int
    | End of AsyncReplyChannel<'a>

type HistMailBox()=
    static let histo = Array.zeroCreate(256)

    static member CreateMailBox(postal_code: int) =
        let mailbox = 
            MailboxProcessor<'a MailContent>.Start(fun inbox ->
                let id = postal_code
                let rec loop() =
                    async {
                        let! msg = inbox.Receive()
                        match msg with
                        | Add(ib) -> histo.[ib] <- histo.[ib] + 1u; return! loop()
                        | End(reply) -> reply.Reply(id); return ()  }
                loop() )
        mailbox

    static member Execute(N: int) =
        let stride = 256 / N
        let mailboxes: MailboxProcessor<'b> array = Array.init N (fun i -> HistMailBox.CreateMailBox(i))
        stopWatch.Reset()
        stopWatch.Start()
        Parallel.For(0, SIZE,
            (fun i ->
                let ib = int buffer.[i]
                let postal_code = ib / stride
                mailboxes.[postal_code].Post(Add(ib))
                )
            ) |> ignore
        mailboxes |> Array.iter(fun mailbox ->
                                    let id = mailbox.PostAndReply(fun reply -> End(reply))
                                    printfn "Mailbox %i finished." id)
        stopWatch.Stop()
        let timeSpan = stopWatch.Elapsed
        printfn "Histogram on CPU: %f (ms)" timeSpan.TotalMilliseconds
        let histoCount = histo |> Array.sum
        printfn "Histogram Sum: %i" histoCount
        ()


type Single() =
    static member Execute() =
        let histo2 = Array.zeroCreate 256 // Array.Clear somehow not working.
        stopWatch.Reset()
        stopWatch.Start()
        for i in 0 .. SIZE-1 do
            histo2.[int buffer.[i]] <- histo2.[int buffer.[i]] + 1u
        stopWatch.Stop()
        let timeSpan = stopWatch.Elapsed
        printfn "Histogram on CPU using a single thread: %f (ms)" timeSpan.TotalMilliseconds
        let histoCount = histo2 |> Array.sum
        printfn "Histogram Sum: %i" histoCount


type BatchContent =
    | Partial of int * int
    | Result of  AsyncReplyChannel<uint32 array>

type HistMailBox2()=
    static let histo = Array.zeroCreate(256)

    static member CreateMailBox(postal_code: int) =
        let mailbox = 
            MailboxProcessor<'a>.Start(fun inbox ->
                let temp = Array.zeroCreate 256
                // let count = ref 0
                let id = postal_code
                let rec loop() =
                    async {
                        let! msg = inbox.Receive()
                        match msg with
                        | Partial(is, ie) ->
                            for i in is .. ie do
                                temp.[int buffer.[i]] <- temp.[int buffer.[i]] + 1u
                                // count := !count + 1
                            // printfn "%i" !count
                            return! loop()
                        | Result(reply) -> reply.Reply(temp); return ()  }
                loop() )
        mailbox

    static member Execute(N: int) =
        printfn "Using %i mailboxes." N
        let band = SIZE / N
        let mailboxes: MailboxProcessor<'b> array = Array.init N (fun i -> HistMailBox2.CreateMailBox(i))
        stopWatch.Reset()
        stopWatch.Start()
        for i in 0 .. N-1 do
            let is = band * i
            let ie = is + band - 1
            mailboxes.[i].Post(Partial(is, ie))
        for i in 0 .. N-1 do
            let temp = mailboxes.[i].PostAndReply(fun reply -> Result(reply))
            for j in 0 .. 255 do
                histo.[j] <- histo.[j] + temp.[j] // temp just 16 in total?
        stopWatch.Stop()
        let timeSpan = stopWatch.Elapsed
        printfn "Histogram on CPU: %f (ms)" timeSpan.TotalMilliseconds
        let histoCount = histo |> Array.sum
        printfn "Histogram Sum: %i" histoCount
        for j in 0 .. 255 do
                histo.[j] <- 0u
        ()

type PFor() =
    static let histo = Array.zeroCreate(256)

    static member Execute(N: int) =
        printfn "Parallel.For using %i sequences." N
        stopWatch.Reset()
        stopWatch.Start()
        let temps = Array.init N (fun _ -> Array.zeroCreate 256)
        let band = SIZE / N
        Parallel.For(0, N,
            (fun i ->
                let is = band * i
                let ie = is + band - 1
                for j in is .. ie do
                    temps.[i].[int buffer.[j]] <- temps.[i].[int buffer.[j]] + 1u
                )) |> ignore
        Parallel.For(0, 256,
            (fun k ->
                for l in 0 .. N-1 do
                    histo.[k] <- histo.[k] + temps.[l].[k]
                )) |> ignore
        stopWatch.Stop()
        let timeSpan = stopWatch.Elapsed
        printfn "Histogram on CPU using Parallel.For: %f (ms)" timeSpan.TotalMilliseconds
        let histoCount = histo |> Array.sum
        printfn "Histogram Sum: %i" histoCount
        for j in 0 .. 255 do
                histo.[j] <- 0u
        ()