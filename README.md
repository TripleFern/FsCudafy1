FsCudafy1
=========


 This is F# translation of Cudafy by examples, which is in turn C# translation of the original Cuda by examples.
Please note that some of the examples are not working. Especially, F# structs failed to be translated into CUDA C kernels. According to ILSpy, it appears F# automatically adds several interface implementations. This might be interfering with Cudafying process. Also constant memory allocation is not working. F# has very different syntax for array allocation. I could not come up with work arounds. Those examples with graphics output, namely Julia example and ray-tracing example will be in a different repository.

