# experimental_syftcore

## Purpose

Validate the core Rust library approach for Syft Reboot.

## Structure

- SyftClientCore
- SyftWorkerCore
- PrivacyCore

## Key Components

- Pointers
- Plans
- Protocols
- Capabilities

## Setup

- rustup - https://rustup.rs/
- bloomrpc - https://github.com/uw-labs/bloomrpc
- protoc - https://github.com/protocolbuffers/protobuf

### Rust Toolchain

```
$ rustup toolchain install stable
```

### VSCode Configuration

Install Rust Format:

```
$ rustup component add rustfmt
```

Install Rust Language Server:

```
$ rustup component add rls
```

Install Rust Linting:

```
$ rustup component add clippy
```

Install Rust VSCode Extension:
https://marketplace.visualstudio.com/items?itemName=rust-lang.rust

```
$ ext install rust-lang.rust
```

# Python

## Setup

Make sure you have `python3.5+`

Upgrade pip:

```
$ pip install --upgrade pip
```

Install pipenv:

```
$ pip install pipenv
```

Enter virtualenv:

```
$ cd syft_core
$ pipenv shell
```

Install packages:

```
$ pipenv install --dev --skip-lock
```

## Python Development

You can compile and install the python library in one command:

```
$ maturin develop
```

## Compile Protobufs

Make sure you have protoc available in your path.

```
$ protoc -I=../proto --python_out=./example ../proto/capabilities.proto
```

## Python Wheel

```
$ pipenv shell
$ maturin build -i python
$ pip install `find -L ./target/wheels -name "*.whl"`
```

# Hello World Demo

## Start Worker from Python

```
$ pipenv shell
$ maturin develop
$ python -i example/worker.py
```

You should see:

```
[src/python.rs:17] "Calling hello in rust" = "Calling hello in rust"
>>> Worker listening on [::1]:50051
>>>
```

## Start Client from Python

```
$ pipenv shell
$ maturin develop
$ python -i example/client.py
```

You should see:

```
[src/python.rs:17] "Calling hello in rust" = "Calling hello in rust"
>>>
```

Try issuing a command like:

```
>>> remote_addr = "http://[::1]:50051"
>>> send_addone(remote_addr, 42)
```

# Jupyter Notebook

You can run the Hello World demo with jupyter by opening the two notebooks.

```
$ pipenv shell
$ maturin develop
$ jupyter notebook
```

Make sure to initialize the worker and register capability functions before sending requests from the client.

## Use BloomRPC

Open bloomrpc:

- load `capabilities.proto`
- set address to `[::1]:50051`
- send a message
