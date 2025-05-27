# Debug in Vscode

Add to CMakeLists.txt (Seems this is not needed)

```
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
```

Use the synchronized `CUDA_CHECK_LAST()` version in [utils.h](../include/utils.h) so exceptions are thrown just after the cuda kernel that throws it.

Example launcher.json

```
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
       {
      "name": "Debug feed_forward_layer_test_ASYNC_ALLOC",
      "type": "cppdbg",
      "request": "launch",
      "program": "${workspaceFolder}/build/feed_forward_layer_test_ASYNC_ALLOC",
      "args": [],
      "stopAtEntry": false,
      "cwd": "${workspaceFolder}",
      "environment": [],
      "externalConsole": false,
      "MIMode": "gdb",
      "setupCommands": [
        {
          "description": "Enable pretty-printing for gdb",
          "text": "-enable-pretty-printing",
          "ignoreFailures": true
        }
      ]
    } 

    ]
}
```