//go:build arm64 && !purego

#include "textflag.h"

TEXT ·partial64NEON(SB),NOSPLIT,$0-32
	MOVD dst+0(FP),R0
	MOVD src_base+8(FP),R1
	MOVD src_len+16(FP),R2
	LSR $3,R2,R2
	VLD1.P 64(R1),[V0.B16,V1.B16,V2.B16,V3.B16]
	SUBS $1,R2,R2
	BEQ sum64Done
sum64Loop:
	VLD1.P 64(R1),[V4.B16,V5.B16,V6.B16,V7.B16]
	WORD $0x4e64d400 // fadd v0.2d,v0.2d,v4.2d
	WORD $0x4e65d421 // fadd v1.2d,v1.2d,v5.2d
	WORD $0x4e66d442 // fadd v2.2d,v2.2d,v6.2d
	WORD $0x4e67d463 // fadd v3.2d,v3.2d,v7.2d
	SUBS $1,R2,R2
	BNE sum64Loop
sum64Done:
	VST1 [V0.B16,V1.B16,V2.B16,V3.B16],(R0)
	RET

TEXT ·partial32NEON(SB),NOSPLIT,$0-32
	MOVD dst+0(FP),R0
	MOVD src_base+8(FP),R1
	MOVD src_len+16(FP),R2
	LSR $3,R2,R2
	VLD1.P 32(R1),[V0.B16,V1.B16]
	SUBS $1,R2,R2
	BEQ sum32Done
sum32Loop:
	VLD1.P 32(R1),[V2.B16,V3.B16]
	WORD $0x4e22d400 // fadd v0.4s,v0.4s,v2.4s
	WORD $0x4e23d421 // fadd v1.4s,v1.4s,v3.4s
	SUBS $1,R2,R2
	BNE sum32Loop
sum32Done:
	VST1 [V0.B16,V1.B16],(R0)
	RET

TEXT ·widenNEON(SB), NOSPLIT, $0-24
	MOVD dst+0(FP),R0
	MOVD src+8(FP),R1
	MOVD n+16(FP),R2
	LSR $2,R2,R2
	CBZ R2,widenDone
widenLoop:
	VLD1.P 16(R1),[V2.B16]
	WORD $0x0e617840 // fcvtl v0.2d,v2.2s
	WORD $0x4e617841 // fcvtl2 v1.2d,v2.4s
	VST1.P [V0.B16,V1.B16],32(R0)
	SUBS $1,R2,R2
	BNE widenLoop
widenDone:
	RET

TEXT ·narrowNEON(SB), NOSPLIT, $0-24
	MOVD dst+0(FP),R0
	MOVD src+8(FP),R1
	MOVD n+16(FP),R2
	LSR $2,R2,R2
	CBZ R2,narrowDone
narrowLoop:
	VLD1.P 32(R1),[V0.B16,V1.B16]
	WORD $0x0e616802 // fcvtn v2.2s,v0.2d
	WORD $0x4e616822 // fcvtn2 v2.4s,v1.2d
	VST1.P [V2.B16],16(R0)
	SUBS $1,R2,R2
	BNE narrowLoop
narrowDone:
	RET

// Go 1.24's assembler does not expose all packed floating operations.
// These are baseline AArch64 AdvSIMD FADD/FSUB/FMUL encodings for
// 2D registers (D=destination, N/M=source registers), no fused operations.
#define FADD(D,N,M) WORD $(0x4e60d400 | ((M)<<16) | ((N)<<5) | (D))
#define FSUB(D,N,M) WORD $(0x4ee0d400 | ((M)<<16) | ((N)<<5) | (D))
#define FMUL(D,N,M) WORD $(0x6e60dc00 | ((M)<<16) | ((N)<<5) | (D))

// VLD2/VST2 separate and rejoin two interleaved complex128 numbers.
TEXT ·butterflyNEON(SB), NOSPLIT, $0-72
	MOVD left_base+0(FP), R0
	MOVD right_base+24(FP), R1
	MOVD roots_base+48(FP), R2
	MOVD left_len+8(FP), R3
	LSR $1, R3, R3
	CBZ R3, bdone
bloop:
	VLD2 (R0), [V0.D2,V1.D2]
	VLD2 (R1), [V2.D2,V3.D2]
	VLD2.P 32(R2), [V4.D2,V5.D2]
	FMUL(6,2,4)
	FMUL(7,3,5)
	FSUB(6,6,7)
	FMUL(7,2,5)
	FMUL(8,3,4)
	FADD(7,7,8)
	FADD(2,0,6)
	FADD(3,1,7)
	FSUB(0,0,6)
	FSUB(1,1,7)
	VST2.P [V2.D2,V3.D2], 32(R0)
	VST2.P [V0.D2,V1.D2], 32(R1)
	SUBS $1,R3,R3
	BNE bloop
bdone:
	RET

// Multiply an interleaved pair already split into R/I vectors by
// a split twiddle in V8/V9. Temporary vectors are V10..V12.
#define CMUL(R,I) \
	FMUL(10,R,8); \
	FMUL(11,I,9); \
	FMUL(12,R,9); \
	FMUL(I,I,8); \
	FADD(I,12,I); \
	FSUB(R,10,11)

TEXT ·radix4NEON(SB), NOSPLIT, $0-104
	MOVD x_base+0(FP), R0
	MOVD w1_base+24(FP), R4
	MOVD w2_base+48(FP), R5
	MOVD w3_base+72(FP), R6
	MOVD w1_len+32(FP), R7
	LSL $4,R7,R8
	ADD R8,R0,R1
	ADD R8,R1,R2
	ADD R8,R2,R3
	FMOVD sign+96(FP), F13
	VDUP V13.D[0], V13.D2
	WORD $0x6ee0f9ae // fneg v14.2d,v13.2d
	LSR $1,R7,R7
	CBZ R7, rdone
rloop:
	VLD2 (R0), [V0.D2,V1.D2]
	VLD2 (R1), [V2.D2,V3.D2]
	VLD2.P 32(R4), [V8.D2,V9.D2]
	CMUL(2,3)
	VLD2 (R2), [V4.D2,V5.D2]
	VLD2.P 32(R5), [V8.D2,V9.D2]
	CMUL(4,5)
	VLD2 (R3), [V6.D2,V7.D2]
	VLD2.P 32(R6), [V8.D2,V9.D2]
	CMUL(6,7)
	FADD(8,0,2)
	FADD(9,1,3)
	FSUB(0,0,2)
	FSUB(1,1,3)
	FADD(10,4,6)
	FADD(11,5,7)
	FSUB(4,4,6)
	FSUB(5,5,7)
	FMUL(6,5,14)
	FMUL(7,4,13)
	FADD(2,8,10)
	FADD(3,9,11)
	FSUB(8,8,10)
	FSUB(9,9,11)
	FADD(10,0,6)
	FADD(11,1,7)
	FSUB(0,0,6)
	FSUB(1,1,7)
	VST2.P [V2.D2,V3.D2], 32(R0)
	VST2.P [V10.D2,V11.D2], 32(R1)
	VST2.P [V8.D2,V9.D2], 32(R2)
	VST2.P [V0.D2,V1.D2], 32(R3)
	SUBS $1,R7,R7
	BNE rloop
rdone:
	RET

#define FNEG(D,N) WORD $(0x6ee0f800 | ((N)<<5) | (D))
#define MRMUL(R,I) \
	FMUL(22,R,20); \
	FMUL(23,I,21); \
	FMUL(24,R,21); \
	FMUL(I,I,20); \
	FADD(I,24,I); \
	FSUB(R,22,23)

TEXT ·radix3NEON(SB),NOSPLIT,$0-80
	MOVD x_base+0(FP),R0
	MOVD w1_base+24(FP),R4
	MOVD w2_base+48(FP),R5
	MOVD w1_len+32(FP),R6
	LSL $4,R6,R7
	ADD R7,R0,R1
	ADD R7,R1,R2
	MOVD $·mixedConstants(SB),R7
	VLD1R.P 8(R7),[V25.D2]
	VLD1R (R7),[V26.D2]
	FMOVD sign+72(FP),F27
	VDUP V27.D[0],V27.D2
	FMUL(26,26,27)
	LSR $1,R6,R6
m3loop:
	VLD2 (R0),[V0.D2,V1.D2]
	VLD2 (R1),[V2.D2,V3.D2]
	VLD2.P 32(R4),[V20.D2,V21.D2]
	MRMUL(2,3)
	VLD2 (R2),[V4.D2,V5.D2]
	VLD2.P 32(R5),[V20.D2,V21.D2]
	MRMUL(4,5)
	FADD(6,2,4)
	FADD(7,3,5)
	FSUB(8,2,4)
	FSUB(9,3,5)
	FADD(12,0,6)
	FADD(13,1,7)
	FMUL(6,6,25)
	FMUL(7,7,25)
	FSUB(0,0,6)
	FSUB(1,1,7)
	FMUL(10,9,26)
	FNEG(10,10)
	FMUL(11,8,26)
	FADD(2,0,10)
	FADD(3,1,11)
	FSUB(4,0,10)
	FSUB(5,1,11)
	VST2.P [V12.D2,V13.D2],32(R0)
	VST2.P [V2.D2,V3.D2],32(R1)
	VST2.P [V4.D2,V5.D2],32(R2)
	SUBS $1,R6,R6
	BNE m3loop
	RET

TEXT ·radix5NEON(SB),NOSPLIT,$0-56
	MOVD x_base+0(FP),R0
	MOVD roots_base+24(FP),R5
	MOVD roots_len+32(FP),R9
	LSR $2,R9,R9
	LSL $4,R9,R10
	ADD R10,R0,R1
	ADD R10,R1,R2
	ADD R10,R2,R3
	ADD R10,R3,R4
	ADD R10,R5,R6
	ADD R10,R6,R7
	ADD R10,R7,R8
	MOVD $·mixedConstants+16(SB),R11
	VLD1R.P 8(R11),[V25.D2]
	VLD1R.P 8(R11),[V26.D2]
	VLD1R.P 8(R11),[V27.D2]
	VLD1R (R11),[V28.D2]
	FMOVD sign+48(FP),F29
	VDUP V29.D[0],V29.D2
	FMUL(27,27,29)
	FMUL(28,28,29)
	LSR $1,R9,R9
m5loop:
	VLD2 (R0),[V0.D2,V1.D2]
	VLD2 (R1),[V2.D2,V3.D2]
	VLD2.P 32(R5),[V20.D2,V21.D2]
	MRMUL(2,3)
	VLD2 (R2),[V4.D2,V5.D2]
	VLD2.P 32(R6),[V20.D2,V21.D2]
	MRMUL(4,5)
	VLD2 (R3),[V6.D2,V7.D2]
	VLD2.P 32(R7),[V20.D2,V21.D2]
	MRMUL(6,7)
	VLD2 (R4),[V8.D2,V9.D2]
	VLD2.P 32(R8),[V20.D2,V21.D2]
	MRMUL(8,9)
	FADD(10,2,8)
	FADD(11,3,9)
	FSUB(2,2,8)
	FSUB(3,3,9)
	FADD(12,4,6)
	FADD(13,5,7)
	FSUB(4,4,6)
	FSUB(5,5,7)
	FADD(14,0,10)
	FADD(15,1,11)
	FADD(14,14,12)
	FADD(15,15,13)
	VST2.P [V14.D2,V15.D2],32(R0)
	FMUL(14,10,25)
	FMUL(15,11,25)
	FMUL(18,12,26)
	FMUL(19,13,26)
	FADD(14,0,14)
	FADD(15,1,15)
	FADD(14,14,18)
	FADD(15,15,19)
	FMUL(16,10,26)
	FMUL(17,11,26)
	FMUL(18,12,25)
	FMUL(19,13,25)
	FADD(16,0,16)
	FADD(17,1,17)
	FADD(16,16,18)
	FADD(17,17,19)
	FMUL(6,3,27)
	FMUL(18,5,28)
	FADD(6,6,18)
	FNEG(6,6)
	FMUL(7,2,27)
	FMUL(19,4,28)
	FADD(7,7,19)
	FMUL(8,3,28)
	FMUL(18,5,27)
	FSUB(8,8,18)
	FNEG(8,8)
	FMUL(9,2,28)
	FMUL(19,4,27)
	FSUB(9,9,19)
	FADD(10,14,6)
	FADD(11,15,7)
	FSUB(14,14,6)
	FSUB(15,15,7)
	FADD(12,16,8)
	FADD(13,17,9)
	FSUB(16,16,8)
	FSUB(17,17,9)
	VST2.P [V10.D2,V11.D2],32(R1)
	VST2.P [V12.D2,V13.D2],32(R2)
	VST2.P [V16.D2,V17.D2],32(R3)
	VST2.P [V14.D2,V15.D2],32(R4)
	SUBS $1,R9,R9
	BNE m5loop
	RET

// Packed floating opcodes below operate on V0 = V0 op V1
// (or V1 op V0 for reflected noncommutative scalar operations).
#define BINARY(OP) \
	VLD1.P 16(R1), [V0.B16]; \
	VLD1.P 16(R2), [V1.B16]; \
	WORD $OP; \
	VST1.P [V0.B16], 16(R0); \
	SUBS $1,R3,R3

TEXT ·binary64NEON(SB), NOSPLIT, $0-73
	MOVD dst_base+0(FP),R0
	MOVD x_base+24(FP),R1
	MOVD y_base+48(FP),R2
	MOVD dst_len+8(FP),R3
	LSR $1,R3,R3
	CBZ R3,b64done
	MOVBU op+72(FP),R4
	CMP $1,R4
	BEQ b64sub
	CMP $2,R4
	BEQ b64mul
	CMP $3,R4
	BEQ b64div
b64add:
	BINARY(0x4e61d400) // fadd v0.2d,v0.2d,v1.2d
	BNE b64add
	RET
b64sub:
	BINARY(0x4ee1d400) // fsub
	BNE b64sub
	RET
b64mul:
	BINARY(0x6e61dc00) // fmul
	BNE b64mul
	RET
b64div:
	BINARY(0x6e61fc00) // fdiv
	BNE b64div
b64done:
	RET

TEXT ·binary32NEON(SB), NOSPLIT, $0-73
	MOVD dst_base+0(FP),R0
	MOVD x_base+24(FP),R1
	MOVD y_base+48(FP),R2
	MOVD dst_len+8(FP),R3
	LSR $2,R3,R3
	CBZ R3,b32done
	MOVBU op+72(FP),R4
	CMP $1,R4
	BEQ b32sub
	CMP $2,R4
	BEQ b32mul
	CMP $3,R4
	BEQ b32div
b32add:
	BINARY(0x4e21d400) // fadd v0.4s,v0.4s,v1.4s
	BNE b32add
	RET
b32sub:
	BINARY(0x4ea1d400) // fsub
	BNE b32sub
	RET
b32mul:
	BINARY(0x6e21dc00) // fmul
	BNE b32mul
	RET
b32div:
	BINARY(0x6e21fc00) // fdiv
	BNE b32div
b32done:
	RET

#define BROADCAST(OP) \
	VLD1.P 16(R1), [V0.B16]; \
	WORD $OP; \
	VST1.P [V0.B16], 16(R0); \
	SUBS $1,R3,R3

TEXT ·broadcast64NEON(SB), NOSPLIT, $0-57
	MOVD dst_base+0(FP),R0
	MOVD x_base+24(FP),R1
	MOVD dst_len+8(FP),R3
	LSR $1,R3,R3
	CBZ R3,c64done
	FMOVD scalar+48(FP),F1
	VDUP V1.D[0],V1.D2
	MOVBU op+56(FP),R4
	CMP $1,R4
	BEQ c64sub
	CMP $5,R4
	BEQ c64rsub
	CMP $2,R4
	BEQ c64mul
	CMP $6,R4
	BEQ c64mul
	CMP $3,R4
	BEQ c64div
	CMP $7,R4
	BEQ c64rdiv
c64add:
	BROADCAST(0x4e61d400)
	BNE c64add
	RET
c64sub:
	BROADCAST(0x4ee1d400)
	BNE c64sub
	RET
c64rsub:
	BROADCAST(0x4ee0d420) // fsub v0.2d,v1.2d,v0.2d
	BNE c64rsub
	RET
c64mul:
	BROADCAST(0x6e61dc00)
	BNE c64mul
	RET
c64div:
	BROADCAST(0x6e61fc00)
	BNE c64div
	RET
c64rdiv:
	BROADCAST(0x6e60fc20) // fdiv v0.2d,v1.2d,v0.2d
	BNE c64rdiv
c64done:
	RET

TEXT ·broadcast32NEON(SB), NOSPLIT, $0-53
	MOVD dst_base+0(FP),R0
	MOVD x_base+24(FP),R1
	MOVD dst_len+8(FP),R3
	LSR $2,R3,R3
	CBZ R3,c32done
	FMOVS scalar+48(FP),F1
	VDUP V1.S[0],V1.S4
	MOVBU op+52(FP),R4
	CMP $1,R4
	BEQ c32sub
	CMP $5,R4
	BEQ c32rsub
	CMP $2,R4
	BEQ c32mul
	CMP $6,R4
	BEQ c32mul
	CMP $3,R4
	BEQ c32div
	CMP $7,R4
	BEQ c32rdiv
c32add:
	BROADCAST(0x4e21d400)
	BNE c32add
	RET
c32sub:
	BROADCAST(0x4ea1d400)
	BNE c32sub
	RET
c32rsub:
	BROADCAST(0x4ea0d420) // fsub v0.4s,v1.4s,v0.4s
	BNE c32rsub
	RET
c32mul:
	BROADCAST(0x6e21dc00)
	BNE c32mul
	RET
c32div:
	BROADCAST(0x6e21fc00)
	BNE c32div
	RET
c32rdiv:
	BROADCAST(0x6e20fc20) // fdiv v0.4s,v1.4s,v0.4s
	BNE c32rdiv
c32done:
	RET
