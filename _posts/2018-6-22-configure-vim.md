# Configure VIM to use YCM

clone vim from github.

# Build from Source

Configure options:

```
./configure --prefix=/home/atlantix/usr/local/ \
    --with-features=huge \
    --enable-rubyinterp \
    --enable-python3interp \
    --with-python3-config-dir=/usr/lib/python3.5/config-3.5m-x86_64-linux-gnu \
    --enable-python2interp \
    --with-python2-config-dir=/home/atlantix/anaconda2/lib/python2.7/config \
    --enable-perlinterp \
    --enable-gui=gtk2 \
    --enable-cscope \
    --enable-luainterp \
    --with-lua-prefix=/home/atlantix/torch/install/ \
    --enable-fail-if-missing
```

Need to make sure lua perl ruby is installed.

```
make && make install
```