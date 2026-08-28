# fusion-store fs-serve 容器镜像（F-OPS-6 部署配置）
# 多阶段构建：builder 编译 release 二进制，runtime 精简镜像
# 用法：
#   docker build -t fusion-store:latest -f deploy/Dockerfile .
#   docker run -d --name fusion-store \
#     -p 11463:11463 \
#     -v fusion-store-data:/data \
#     -e FS_AUTH_TOKEN=<secret> \
#     fusion-store:latest serve --home /data --port 11463
#
# 注：fusion-store 定位 Apple Silicon 统一内存 + mmap，容器化仅作部署封装，
# 跨平台 Linux 构建需 heed/LMDB + mmap 支持（linux/amd64 + linux/arm64均可）。
# macOS 生产建议用 launchd plist（见 deploy/fusion-store.plist）。

FROM rust:1.94-bookworm AS builder
WORKDIR /build
# 先拷 manifests 利用 docker 层缓存
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
# release 构建（排除 pyo3 cdylib，容器内不构建 Python 绑定）
RUN cargo build --release -p fs-serve -p fs-cli --exclude fs-ffi-py

FROM debian:bookworm-slim
# heed/LMDB 运行时依赖 + ca-certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    liblmdb0 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /build/target/release/fusion-store /usr/local/bin/fusion-store
# 数据目录（建议挂载 volume）
RUN mkdir -p /data
VOLUME ["/data"]
# fs-serve 默认端口
EXPOSE 11463
# 默认 serve 子命令，home=/data，环回绑定时容器内需 0.0.0.0（--bind 0.0.0.0）
ENTRYPOINT ["fusion-store"]
CMD ["serve", "--home", "/data", "--port", "11463", "--bind", "0.0.0.0"]
