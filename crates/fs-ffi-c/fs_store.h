// fs_store.h —— fusion-store C-ABI [v2 §3.4/E3]
//
// 由 crates/fs-ffi-c/src/lib.rs 的 #[no_mangle] extern "C" 函数导出。
// 当前手写对齐（接口小）；随接口增长可改 cbindgen 自动生成。
// 错误码：0 = Ok，负数 = StoreError 变体（见 lib.rs err_code）。
//
// 目录布局：单 path 下 vec/kv 各占独立子目录。
//   <path>/vec/{data,meta}   向量索引
//   <path>/kv/{data,meta}    KV store

#ifndef FS_STORE_H
#define FS_STORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 不透明句柄（caller 不直接访问字段）
typedef struct FsStoreHandle FsStoreHandle;

// 错误码
#define FS_OK 0
#define FS_ERR_IO -1
#define FS_ERR_HEED -2
#define FS_ERR_NOT_FOUND -3
#define FS_ERR_DUP_VECTOR -4
#define FS_ERR_DIM_MISMATCH -5
#define FS_ERR_QUOTA -6
#define FS_ERR_BUSY -7
#define FS_ERR_LOCK -8
#define FS_ERR_CORRUPT -9
#define FS_ERR_TIMEOUT -10
#define FS_ERR_SEGMENT_FULL -11
#define FS_ERR_ARROW -12
#define FS_ERR_SERDE -13
#define FS_ERR_OTHER -99

// 新建 store：建向量索引（锁定 schema dim）+ KV store。out 接收堆分配句柄。
int fs_store_create(const char* path, size_t dim, FsStoreHandle** out);

// 打开已存在 store：从持久化 schema 读回向量索引 + 重开 KV。
int fs_store_open(const char* path, FsStoreHandle** out);

// 关闭句柄，释放堆内存（NULL 安全）。
void fs_store_close(FsStoreHandle* h);

// 写 KV。
int fs_store_put_kv(FsStoreHandle* h, const uint8_t* key, size_t klen,
                    const uint8_t* val, size_t vlen);

// 零拷贝读 KV → 强制拷贝到 caller buffer（E3）。
// out_cap 不足返 FS_ERR_OTHER，vlen_out 写所需长度；key 不存在返 FS_ERR_NOT_FOUND。
int fs_store_get_kv(FsStoreHandle* h, const uint8_t* key, size_t klen,
                    uint8_t* out_val, size_t out_cap, size_t* vlen_out);

// 插入向量（id + vlen 个 f32）。
int fs_store_insert_vector(FsStoreHandle* h, uint64_t id, const float* v, size_t vlen);

// KNN 检索（读强制拷贝，E3）。out_ids/out_dists 预分配容量 >= top_k，out_n 写实际数。
int fs_store_search_knn(FsStoreHandle* h, const float* q, size_t qlen, size_t top_k,
                        uint64_t* out_ids, float* out_dists, size_t* out_n);

// 返回向量维度（0 = 出错或空）。
size_t fs_store_vector_dim(const FsStoreHandle* h);

// checkpoint：HNSW 图 snapshot 落盘 + 段 flush（close 前调，重开可恢复图）。
int fs_store_checkpoint(FsStoreHandle* h);

#ifdef __cplusplus
}
#endif

#endif // FS_STORE_H
