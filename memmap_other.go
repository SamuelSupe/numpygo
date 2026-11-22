//go:build !linux && !darwin

package np

import "os"

const mmapSupported = false

func mapFile(*os.File, int, int64, string) ([]byte, error) { return nil, ErrMappingUnsupported }
func unmapBytes([]byte) error                              { return nil }
func syncMappedBytes([]byte) error                         { return nil }
