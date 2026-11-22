//go:build linux || darwin

package np

import (
	"os"
	"runtime"
	"syscall"
	"unsafe"
)

const mmapSupported = true

func mapFile(f *os.File, length int, offset int64, mode string) ([]byte, error) {
	prot, flags := syscall.PROT_READ, syscall.MAP_SHARED
	if mode != "r" {
		prot |= syscall.PROT_WRITE
	}
	if mode == "c" {
		flags = syscall.MAP_PRIVATE
	}
	return syscall.Mmap(int(f.Fd()), offset, length, prot, flags)
}
func unmapBytes(data []byte) error {
	if len(data) == 0 {
		return nil
	}
	return syscall.Munmap(data)
}
func syncMappedBytes(data []byte) error {
	if len(data) == 0 {
		return nil
	}
	_, _, e := syscall.Syscall(syscall.SYS_MSYNC, uintptr(unsafe.Pointer(&data[0])), uintptr(len(data)), syscall.MS_SYNC)
	runtime.KeepAlive(data)
	if e != 0 {
		return e
	}
	return nil
}
