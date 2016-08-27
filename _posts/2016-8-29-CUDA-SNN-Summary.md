---
layout: post
title:  "CUDA Programming Tips (Unfinished)"
date:   2016-08-29 22:26:44 +0800
categories: jekyll update
---

Using `cuda-gdb`, you are able to know where the code crashed and see the related variables. But it does not mean that you just need to examine the code around the incident area. Instead, look at how you allocate this variable. Are you calling the right pointer? Did you allocate it with the right size and type? Did you call `cudaDeviceSynchronize()` properly if you are using `cudaMallocManged` ? Did you call `__syncthreads()` if there are any concurrent operation on the same address (which you should use Atomic Functions)

### Problem 1: You find your array's value is not what you want or is chaotic.
Check if your memory allocation or initialization is correct.

### Problem 8: Your array seems to fail right in the middle.
If you call your `cuda-gdb` and find your code crashed right in the middle stage. E.g. you are visiting a array and right in the middle `cuda-gdb` told you that the array address is illegal. No boundary problems. No initialization problems. You may suspect that previous operations have modified here illegally. But the first thing to do is check whether it is really the middle. CUDA execute in a rather unreasonably parallel way. Thread in a warp execute at the same time while different warps execute in approximately concurrent time but differ stochastically. It is straightforward but ignorable to notice that threads with quite-non-zero indices are the first to execute (say <<<(1,0,0),(92,0,0)>>> warp 2 in one of my failures).
So my advice to this situation is to change you configuration from multiple threads to single threads.
If your program fail at the first thread, then it is probable that you initialization or parameter passing or something like that is problematic.

### Problem 9: Using cuda-gdb `set cuda memcheck on` getting a different result
I once encountered a situation that when I turn on this option and figured out bugs but the program fails again when running normally. Later I found that running normal cuda-gdb will not cover the bug, though it cannot dive into kernel functions. Based on my experience, I examined  `cudaDeviceSynchronize()` and find one missing. Then the bug was fixed. So I guess using this option result in automatic memory synchronization.

But later, I faced the problem again. cuda-gdb said `Invalid Managed Memory Access`. Well, it may be a synchronization problem again. But this time, `cudaDeviceSynchronize` does not help any more.

### Tip 1: Using class
CUDA does not support
