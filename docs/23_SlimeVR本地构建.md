# Local Windows Build

This checkout uses a reproducible local build entry point:

```powershell
cd D:\dyh\MotionRe\base_mobileposer\SlimeVR-Server
.\scripts\build-windows-local.ps1
```

The script pins the verified local tool locations:

- CMake: `D:\Software\cmake\bin\cmake.exe`
- JDK: `D:\Software\jdk-17`
- pnpm: `D:\Nodejs\pnpm.cmd`
- Compiler: the latest registered Visual Studio 2022 installation with x64/x86 C++ tools
- Target: Windows x64 Release

It builds the Java server, native bindings provider, GUI, and unpacked Windows app. The output is:

```text
gui\dist\artifacts\win\win-unpacked\SlimeVR.exe
```

Validate the toolchain without building:

```powershell
.\scripts\build-windows-local.ps1 -ValidateOnly
```

Create a ZIP with PowerShell after packaging:

```powershell
.\scripts\build-windows-local.ps1 -Archive
```

The defaults can be overridden per process with `SLIMEVR_CMAKE_EXE`, `SLIMEVR_JAVA_HOME`, and `SLIMEVR_PNPM_CMD`.
