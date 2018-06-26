# Configure VIM to use YCM

clone vim from github.

## Build from Source

Configure options:

```
./configure --prefix=/home/atlantix/usr/local/ \
    --with-features=huge \
    --enable-rubyinterp \
    --enable-python3interp \
    --with-python3-config-dir=/usr/lib/python3.5/config-3.5m-x86_64-linux-gnu \
    --enable-pythoninterp \
    --with-python-config-dir=/home/atlantix/anaconda2/lib/python2.7/config \
    --enable-perlinterp \
    --enable-gui=gtk2 \
    --enable-cscope \
    --enable-luainterp \
    --with-lua-prefix=/home/atlantix/torch/install/ \
    --enable-fail-if-missing
```

./configure --prefix=/home/jianjin/home/usr/local/ \
    --with-features=huge \
    --enable-pythoninterp \
    --with-python-config-dir=/home/jianjin/home/anaconda2/lib/python2.7/config \
    --enable-gui=gtk2 \
    --enable-cscope \
    --enable-luainterp \
    --with-lua-prefix=/home/jianjin/home/torch/install/ \
    --enable-fail-if-missing

Need to make sure lua perl ruby is installed.

```
make && make install
```

Then vim is installed locally. Make sure that the local bin directory is in the path.

## Configure vimrc

I select the [Ultimate vimrc](https://github.com/amix/vimrc) from github. First install this configuration.

Then add [YouCompleteMe](https://github.com/Valloric/YouCompleteMe) to this configuration.

```
git clone https://github.com/Valloric/YouCompleteMe
cd YouCompleteMe
git submodule update --init --recursive
```

Clone YCM to the ~/.vim_runtime/my_plugins directory.

Install all the complete feature of YCM: make sure xbuild, go, tsserver, node, npm, rustc, and cargo tools are installed.

1. Go. Pass.

2. Install tsserver: `npm i typescript -g`. This is the typescript server. If you need to use proxy, then do 

```
$ npm config set proxy http://server:port
$ npm config set https-proxy https://server:port
```

3. node. Pass.

4. npm. Pass.

5. rustc. To install this, it is recommended to install rustup as a whole.

` curl https://sh.rustup.rs -sSf | sh`

```
Welcome to Rust!

This will download and install the official compiler for the Rust programming
language, and its package manager, Cargo.

It will add the cargo, rustc, rustup and other commands to Cargo's bin
directory, located at:

  /home/atlantix/.cargo/bin

This path will then be added to your PATH environment variable by modifying the
profile files located at:

  /home/atlantix/.profile
  /home/atlantix/.zprofile
  /home/atlantix/.bash_profile

You can uninstall at any time with rustup self uninstall and these changes will
be reverted.

Current installation options:

   default host triple: x86_64-unknown-linux-gnu
     default toolchain: stable
  modify PATH variable: yes

1) Proceed with installation (default)
2) Customize installation
3) Cancel installation
```

Sometimes proxy may cause several issues. Disable them make things work.

4. cargo. Installed together with rustup.

Native configuration failed for me. run `./install.py --all` will result in a HASH error. Because the program failed to download a specific clang prebuilt libraries. So here I try to build YCM manually.

Download clang and llvm from official site and extract the prebuilt binary:

http://releases.llvm.org/download.html

Follow the instructions: 

```
cd ~ && mkdir ycm_build && cd ycm_build
cmake -G "Unix Makefiles" -DPATH_TO_LLVM_ROOT=/home/atlantix/.vim_runtime/my_plugins/YouCompleteMe/third_party/ycmd/clang_archives/clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-14.04 . ~/.vim_runtime/my_plugins/YouCompleteMe/third_party/ycmd/cpp
cmake --build . --target ycm_core --config Release
```

or this for Ubuntu 16.04

```
cd ~ && mkdir ycm_build && cd ycm_build
cmake -G "Unix Makefiles" -DPATH_TO_LLVM_ROOT=~/.vim_runtime/my_plugins/YouCompleteMe/third_party/ycmd/clang_archives/clang+llvm-6.0.0-x86_64-linux-gnu-ubuntu-16.04 . ~/.vim_runtime/my_plugins/YouCompleteMe/third_party/ycmd/cpp
cmake --build . --target ycm_core --config Release
```

Optional:

cd into this directory

```
cmake -G "Unix Makefiles" . ~/.vim_runtime/my_plugins/YouCompleteMe/third_party/ycmd/third_party/cregex
cmake --build . --target _regex --config Release
```

And build support for other languages as instructed in full installation guide.

To enable completion in C language project, you need to add a few step to your project. If you are using CMake, then add a line `set(CMAKE_EXPORT_COMPILE_COMMANDS ON)` to cmake and build it. This will generate a json named compile_commands.json. Make sure this file is at the root of your project.

As for other methods, I would recommend using `.ycm_extra_conf.py`.