declare void @llvm.memset.p0.i64(ptr writeonly, i8, i64, i1 immarg)
declare void @llvm.memcpy.p0.p0.i64(ptr writeonly, ptr readonly, i64, i1 immarg)
declare void @llvm.memmove.p0.p0.i64(ptr writeonly, ptr readonly, i64, i1 immarg)
declare ptr @memset(ptr, i32, i64)
declare ptr @memcpy(ptr, ptr, i64)
declare ptr @memmove(ptr, ptr, i64)
declare ptr @pth_memset(ptr, i32, i64)
declare ptr @pth_memcpy(ptr, ptr, i64)
declare ptr @pth_memmove(ptr, ptr, i64)

define void @write_ops(ptr %dst, ptr %src, ptr %atomic) {
entry:
  store i32 1, ptr %dst, align 4
  %old = atomicrmw add ptr %atomic, i64 1 seq_cst, align 8
  %cmp = cmpxchg ptr %atomic, i64 1, i64 2 seq_cst seq_cst, align 8
  call void @llvm.memset.p0.i64(ptr %dst, i8 0, i64 16, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 16, i1 false)
  call void @llvm.memmove.p0.p0.i64(ptr %dst, ptr %src, i64 16, i1 false)
  call ptr @memset(ptr %dst, i32 0, i64 16)
  call ptr @memcpy(ptr %dst, ptr %src, i64 16)
  call ptr @memmove(ptr %dst, ptr %src, i64 16)
  call ptr @pth_memset(ptr %dst, i32 0, i64 16)
  call ptr @pth_memcpy(ptr %dst, ptr %src, i64 16)
  call ptr @pth_memmove(ptr %dst, ptr %src, i64 16)
  ret void
}
