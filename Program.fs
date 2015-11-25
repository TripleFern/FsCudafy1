module FsCudafy1.EntryPoint

open System
open Cudafy.Host
open Cudafy.Translator


[<EntryPoint>]
let main argv = 
    Cudafy.CudafyModes.Target <- Cudafy.eGPUType.Cuda
    Cudafy.CudafyModes.DeviceId <- 0
    
    CudafyTranslator.Language <- 
        if Cudafy.CudafyModes.Target = Cudafy.eGPUType.OpenCL then
            Cudafy.eLanguage.OpenCL
        else
            Cudafy.eLanguage.Cuda

    let deviceCount = CudafyHost.GetDeviceCount(Cudafy.CudafyModes.Target)
    printfn "Number of devices: %i" deviceCount

    let gpu = CudafyHost.GetDevice(Cudafy.CudafyModes.Target, Cudafy.CudafyModes.DeviceId)
    let name =  gpu.GetDeviceProperties(false).Name
    printfn "Running examples using '%s'." name 
    EnumGPU.execute()
    Console.WriteLine "-- Chapter 3 examples. --"
    SimpleKernel.execute()
    SimpleKernelParams.execute()
    Console.WriteLine "-- Chapter 4 examples. --"
    Console.WriteLine "Add loop on CPU."
    AddLoopCPU.execute()
    Console.WriteLine "Add loop on GPU."
    AddLoopGPU.execute()
    AddLongLoopCPU.execute()
    AddLongLoopGPU.execute()    
    Console.WriteLine "-- Chapter 5 examples. --"
    Console.WriteLine "Add loop on GPU using multiple threads."
    AddLoopBlocksGPU.execute()
    AddLongLoopBlocksGPU.execute()
    Console.WriteLine "Dot product calculation on GPU."
    DotProductGPU.Execute()
    Console.WriteLine "-- Chapter 9 examples. --"
    Console.WriteLine "Histogram processing using GPU atomics."
    HistogramAtomicGPU.Execute()
    Console.WriteLine "Histogram processing using CPU MailboxProcessors."
    // HistogramCPU.HistMailBox.Execute(8) // With 4 or 32 workers, the program sleeps..., not coming back!!!???
    //HistogramCPU.HistMailBox2.Execute(16)
    HistogramCPU.HistMailBox2.Execute(64)
    //HistogramCPU.HistMailBox2.Execute(256)
    //HistogramCPU.HistMailBox2.Execute(1024)
    //HistogramCPU.PFor.Execute(4)
    HistogramCPU.PFor.Execute(8)
    //HistogramCPU.PFor.Execute(16)
    HistogramCPU.PFor.Execute(32)
    //HistogramCPU.PFor.Execute(64)
    HistogramCPU.Single.Execute()
    Console.WriteLine "-- Chapter 10 examples. --"
    Console.WriteLine "Measuring data transfer between CPU and GPU."
    let copyTimed = CopyTimedGPU()
    copyTimed.Execute()
    Console.WriteLine "Basic streams example."
    BasicStreamGPU.Execute()
    // Tests.NormalClass.execute()...does not work!
    Console.WriteLine "-- Cudafy samples completed. --"
    Console.ReadLine() |> ignore
    0 // 整数の終了コードを返します
