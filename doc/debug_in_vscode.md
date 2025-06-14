# Debug in Vscode

* Make sure gdb is installed.
* `make debug_build`

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